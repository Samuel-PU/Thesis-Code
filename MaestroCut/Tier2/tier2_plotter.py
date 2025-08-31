#!/usr/bin/env python3
r"""
Tier-2 Summary Plotter (camera-ready, Seaborn style)
----------------------------------------------------
Creates a single, unified figure:

  fig-t2-summary.pdf — "Tier 2 Performance Under Stress"
    A) Latency & SLOs (Jitter, TTFR; medians ± 95% CI + target lines)
    B) Success Throughput (kQPS ± 95% CI) + Δ vs Baseline labels
    C) Reliability composition (Success / Timeout / Error, %, with Success CI)

Also writes:
  overhead-summary.csv — per-scenario overhead medians ± 95% CI (in %)

Design notes:
- Works with either raw runs or summarised CSVs; bootstraps medians if needed.
- Robust column aliasing; tolerant to *_ci triplets encoded as JSON strings.
- Color-blind friendly palette; readable text labels; n/a handling.

Usage
-----
python tier2_summary_plotter.py \
  --csv path/to/runs_flat.csv \
  --out figs/tier2 \
  --jitter-target 150 \
  --ttfr-target 220
"""
from __future__ import annotations

import argparse
from pathlib import Path
import json, re
from typing import Tuple, Optional, List, Dict

import warnings
warnings.filterwarnings("ignore", message="Mean of empty slice", category=RuntimeWarning)

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# ------------------------------ Style ----------------------------------------

sns.set_theme(style="whitegrid", context="paper")
sns.set_palette("colorblind")
plt.rcParams.update({
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'figure.autolayout': True,
    'axes.titlepad': 10,
    'mathtext.fontset': 'cm',
    'font.family': 'serif',
})

SCENARIO_ORDER = ["baseline", "noisy", "bursty", "adversarial"]
SCENARIO_LABEL = {
    "baseline": "Baseline",
    "noisy": "Noisy",
    "bursty": "Bursty",
    "adversarial": "Adversarial",
}

# ------------------------------ Robust column aliases ------------------------

CANDIDATES: Dict[str, List[str]] = {
    "jitter": [
        "jitter_ms_ci", "lat_p50_ms_ci", "lat_ms_p50_ci", "lat_p50_ci_ms",
        "jitter_p50_ms_ci", "jitter_ci_ms", "lat_ms_ci", "jitter_ms"
    ],
    "ttfr": [
        "ttfr_ms_ci", "ttfr_p50_ms_ci", "time_to_first_result_ms_ci",
        "time_to_first_token_ms_ci", "ttfr_ci_ms", "ttfr_ms"
    ],
    "success": ["success_rate_ci", "success_pct_ci", "success_ci", "succ_rate_ci", "success_rate", "success_pct"],
    "timeout": ["timeout_rate_ci", "timeout_pct_ci", "timeout_ci", "timeout_rate", "timeout_pct"],
    "error":   ["error_rate_ci", "error_pct_ci", "error_ci", "error_rate", "error_pct"],
    "overhead": [
        "obfuscation_overhead_pct_ci", "phasepad_overhead_pct_ci",
        "overhead_pct_ci", "overhead_ci",
        "obfuscation_overhead_pct", "phasepad_overhead_pct",
        "phasepad_overhead_percent", "overhead_percent",
        "phasepad_overhead", "obfuscation_overhead",
        "overhead_pct", "overhead"
    ],
    "thr": [
        "throughput_qps_ci", "raw_throughput_qps_ci",
        "qps_ci", "throughput_ci",
        "throughput_qps", "raw_qps", "qps",
        "median_qps", "qps_p50", "throughput_p50_qps", "throughput_median_qps"
    ],
    "succ_thr": [
        "success_throughput_qps_ci", "success_qps_ci", "succ_throughput_qps_ci",
        "success_throughput_qps", "success_qps", "succ_qps",
        "success_qps_p50", "success_throughput_p50_qps"
    ],
}

# ------------------------------ CSV + CI helpers -----------------------------

def _maybe_json_list(s: str) -> Optional[Tuple[float, float, float]]:
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not (s.startswith('[') and s.endswith(']')):
        return None
    try:
        vals = json.loads(s)
        if isinstance(vals, list) and len(vals) == 3:
            lo, mid, hi = (float(vals[0]), float(vals[1]), float(vals[2]))
            return lo, mid, hi
    except Exception:
        pass
    try:
        parts = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", s)
        if len(parts) >= 3:
            lo, mid, hi = (float(parts[0]), float(parts[1]), float(parts[2]))
            return lo, mid, hi
    except Exception:
        return None
    return None

def get_ci_triplet(df: pd.DataFrame, rowidx: int, base: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    row = df.iloc[rowidx]
    if base in df.columns:
        v = row[base]
        parsed = _maybe_json_list(v) if isinstance(v, str) else None
        if parsed:
            return parsed
        if isinstance(v, str) and v.startswith("(") and v.endswith(")"):
            parsed = _maybe_json_list(v.replace("(", "[").replace(")", "]"))
            if parsed:
                return parsed
        try:
            val = float(v)
            return (np.nan, val, np.nan)
        except Exception:
            pass
        if isinstance(v, str) and ("lo" in v or "hi" in v or "mid" in v):
            try:
                d = json.loads(v)
                return (float(d.get("lo", np.nan)),
                        float(d.get("mid", np.nan)),
                        float(d.get("hi", np.nan)))
            except Exception:
                pass
    if all((base + suf) in df.columns for suf in ["_lo", "_mid", "_hi"]):
        return (float(row[base + "_lo"]), float(row[base + "_mid"]), float(row[base + "_hi"]))
    if all((base + suf) in df.columns for suf in ["_0", "_1", "_2"]):
        return (float(row[base + "_0"]), float(row[base + "_1"]), float(row[base + "_2"]))
    return (None, None, None)

def _first_scalar(df: pd.DataFrame, rowidx: int, bases: List[str]) -> Optional[float]:
    row = df.iloc[rowidx]
    for b in bases:
        for suf in ("", "_mid", "_p50", "_median"):
            col = f"{b}{suf}"
            if col in df.columns:
                try:
                    return float(row[col])
                except Exception:
                    pass
    return None

def get_ci_triplet_any(df: pd.DataFrame, rowidx: int, bases: List[str]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    for b in bases:
        lo, mid, hi = get_ci_triplet(df, rowidx, b)
        if any(v is not None and np.isfinite(v) for v in (lo, mid, hi)):
            return lo, mid, hi
    mid = _first_scalar(df, rowidx, bases)
    if mid is not None and np.isfinite(mid):
        return (np.nan, float(mid), np.nan)
    return (None, None, None)

def _to_pct(v: Optional[float]) -> Optional[float]:
    if v is None or not np.isfinite(v):
        return None
    v = float(v)
    return 100.0 * v if v <= 1.0000001 else v

def _to_pct_triplet(lo: Optional[float], mid: Optional[float], hi: Optional[float]) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    return _to_pct(lo), _to_pct(mid), _to_pct(hi)

def load_flat_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "scenario" not in df.columns:
        raise SystemExit("CSV must contain a 'scenario' column.")
    df["scenario"] = df["scenario"].astype(str).str.lower().str.strip()
    df = df[df["scenario"].isin(SCENARIO_ORDER)].copy()
    df["scenario"] = pd.Categorical(df["scenario"], categories=SCENARIO_ORDER, ordered=True)
    return df.sort_values("scenario").reset_index(drop=True)

# ------------------------------ Bootstrap aggregator -------------------------

def _bootstrap_ci(series, n_boot=2000, alpha=0.05, seed=123):
    rng = np.random.default_rng(seed)
    arr = np.asarray(series.dropna().values, dtype=float)
    if arr.size == 0:
        return (np.nan, np.nan, np.nan)
    meds = []
    for _ in range(n_boot):
        meds.append(np.median(rng.choice(arr, size=arr.size, replace=True)))
    lo = float(np.percentile(meds, 100*alpha/2))
    mid = float(np.median(arr))
    hi = float(np.percentile(meds, 100*(1 - alpha/2)))
    return lo, mid, hi

def aggregate_runs_bootstrap(df: pd.DataFrame, n_boot=2000, seed=123) -> pd.DataFrame:
    # If already summarised (has *_ci columns), leave it alone.
    if any(c.endswith("_ci") for c in df.columns):
        return df.copy()

    metrics = {
        "jitter_ms": ("jitter_ms_ci", 1.0),
        "ttfr_ms": ("ttfr_ms_ci", 1.0),
        "success_rate": ("success_rate_ci", 1.0),   # keep as fraction
        "timeout_rate": ("timeout_rate_ci", 1.0),
        "error_rate": ("error_rate_ci", 1.0),
        "obfuscation_overhead_pct_mean": ("overhead_pct_ci", 1.0),  # fraction in many CSVs
        "throughput_qps": ("throughput_qps_ci", 1.0),
        "success_throughput_qps": ("success_throughput_qps_ci", 1.0),
    }

    rows = []
    for scen, g in df.groupby("scenario", observed=False):
        row = {"scenario": scen}
        for src, (dst, scale) in metrics.items():
            if src not in g.columns:
                continue
            lo, mid, hi = _bootstrap_ci(g[src], n_boot=n_boot, seed=seed)
            if scale != 1.0:
                lo, mid, hi = [None if not np.isfinite(x) else x*scale for x in (lo, mid, hi)]
            row[dst] = json.dumps([lo, mid, hi])
        rows.append(row)

    out = pd.DataFrame(rows)
    out["scenario"] = out["scenario"].astype(str).str.lower().str.strip()
    out["scenario"] = pd.Categorical(out["scenario"], categories=SCENARIO_ORDER, ordered=True)
    return out.sort_values("scenario").reset_index(drop=True)

# ------------------------------ Summary figure -------------------------------

def _ci_err_from_triplet(lo: Optional[float], mid: Optional[float], hi: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
    if any(v is None or not np.isfinite(v) for v in [lo, mid, hi]):
        return (None, None)
    return (max(0.0, mid - lo), max(0.0, hi - mid))

def fig_tier2_summary(df: pd.DataFrame, outdir: Path,
                   jitter_target_ms: float,
                   ttfr_target_ms: float):
    """
    Unified Tier-2 summary:
      A) Latency & SLOs (Jitter, TTFR; p50 ± 95% CI)
      B) Success Throughput (kQPS) with Δ% vs Baseline
      C) Reliability (Success/Timeout/Error), stacked (%)
    Fixes:
      - Legend label uses 'TTFR (p50, ms)' (typo fixed)
      - Panel B y-axis uses 'Success Throughput (kQPS)' (typo fixed)
      - Δ% computed as ((value - baseline) / baseline) * 100 (sign correct)
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    plt.rcParams.update({'pdf.fonttype': 42, 'ps.fonttype': 42})

    # Ensure canonical scenario order
    order_keys = SCENARIO_ORDER
    order_labels = [SCENARIO_LABEL[s] for s in order_keys]
    n = len(order_keys)
    x = np.arange(n)

    # ---------- Panel A data (Latency & SLOs) ----------
    jitter_mid, jitter_err = [], [[], []]  # lower, upper
    ttfr_mid, ttfr_err = [], [[], []]

    for scen in order_keys:
        i = df.index[df["scenario"].astype(str) == scen][0]

        j_lo, j_md, j_hi = get_ci_triplet_any(df, i, CANDIDATES["jitter"])
        t_lo, t_md, t_hi = get_ci_triplet_any(df, i, CANDIDATES["ttfr"])

        # medians
        jitter_mid.append(float(j_md) if j_md is not None and np.isfinite(j_md) else np.nan)
        ttfr_mid.append(float(t_md) if t_md is not None and np.isfinite(t_md) else np.nan)
        # CIs → asymmetric error
        def _err(lo, md, hi):
            if any(v is None or not np.isfinite(v) for v in (lo, md, hi)):
                return (0.0, 0.0)
            return (max(0.0, md - lo), max(0.0, hi - md))
        el, eh = _err(j_lo, j_md, j_hi); jitter_err[0].append(el); jitter_err[1].append(eh)
        el, eh = _err(t_lo, t_md, t_hi); ttfr_err[0].append(el); ttfr_err[1].append(eh)

    # ---------- Panel B data (Success Throughput, kQPS) ----------
    thr_mid_k, thr_err_k = [], [[], []]
    for scen in order_keys:
        i = df.index[df["scenario"].astype(str) == scen][0]
        # Prefer success throughput; fall back to raw throughput if needed
        s_lo, s_md, s_hi = get_ci_triplet_any(df, i, CANDIDATES["succ_thr"])
        if not any(v is not None and np.isfinite(v) for v in (s_lo, s_md, s_hi)):
            s_lo, s_md, s_hi = get_ci_triplet_any(df, i, CANDIDATES["thr"])
        # convert to kQPS
        if s_md is None or not np.isfinite(s_md):
            thr_mid_k.append(np.nan); thr_err_k[0].append(0.0); thr_err_k[1].append(0.0)
        else:
            thr_mid_k.append(float(s_md) / 1000.0)
            el = max(0.0, float(s_md) - float(s_lo)) / 1000.0 if (s_lo is not None and np.isfinite(s_lo)) else 0.0
            eh = max(0.0, float(s_hi) - float(s_md)) / 1000.0 if (s_hi is not None and np.isfinite(s_hi)) else 0.0
            thr_err_k[0].append(el); thr_err_k[1].append(eh)

    # ---------- Panel C data (Reliability, %) ----------
    succ_pct, to_pct, err_pct = [], [], []
    succ_err_pct = [[], []]  # (lower, upper) for success only (to keep clutter down)
    for scen in order_keys:
        i = df.index[df["scenario"].astype(str) == scen][0]
        sr_lo, sr_md, sr_hi = get_ci_triplet_any(df, i, CANDIDATES["success"])
        to_lo, to_md, to_hi = get_ci_triplet_any(df, i, CANDIDATES["timeout"])
        er_lo, er_md, er_hi = get_ci_triplet_any(df, i, CANDIDATES["error"])

        # Convert to percent if fractions
        sr_lo, sr_md, sr_hi = _to_pct_triplet(sr_lo, sr_md, sr_hi)
        to_lo, to_md, to_hi = _to_pct_triplet(to_lo, to_md, to_hi)
        er_lo, er_md, er_hi = _to_pct_triplet(er_lo, er_md, er_hi)

        def _f(x): return float(x) if x is not None and np.isfinite(x) else 0.0
        succ_pct.append(_f(sr_md)); to_pct.append(_f(to_md)); err_pct.append(_f(er_md))
        # Only success CI drawn (optional — extend to others if you have CIs)
        if all(v is not None and np.isfinite(v) for v in (sr_lo, sr_md, sr_hi)):
            succ_err_pct[0].append(max(0.0, float(sr_md) - float(sr_lo)))
            succ_err_pct[1].append(max(0.0, float(sr_hi) - float(sr_md)))
        else:
            succ_err_pct[0].append(0.0); succ_err_pct[1].append(0.0)

    # ---------- Figure & axes ----------
    fig = plt.figure(figsize=(14, 5.5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 1.0, 1.2], wspace=0.32)
    ax1 = fig.add_subplot(gs[0])  # Latency
    ax2 = fig.add_subplot(gs[1])  # Throughput
    ax3 = fig.add_subplot(gs[2])  # Reliability
    fig.suptitle("Tier 2 Performance Under Stress Scenarios", fontsize=16, fontweight="bold", y=0.98)

    # ---------- Panel A: Latency & SLOs ----------
    width = 0.38
    ax1.bar(x - width/2, jitter_mid, width, yerr=jitter_err, capsize=5,
            label="Jitter (p50, ms)", color="#ff7f0e", alpha=0.85, error_kw={'elinewidth': 1.2})
    ax1.bar(x + width/2, ttfr_mid, width, yerr=ttfr_err, capsize=5,
            label="TTFR (p50, ms)", color="#1f77b4", alpha=0.85, error_kw={'elinewidth': 1.2})
    # SLO lines
    ax1.axhline(jitter_target_ms, color="#ff7f0e", linestyle="--", linewidth=1.2, alpha=0.8, label=f"Jitter Target ({int(jitter_target_ms)} ms)")
    ax1.axhline(ttfr_target_ms,  color="#1f77b4", linestyle="--", linewidth=1.2, alpha=0.8, label=f"TTFR Target ({int(ttfr_target_ms)} ms)")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("A. Latency & SLOs")
    ax1.set_xticks(x); ax1.set_xticklabels(order_labels)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax1.legend(loc="upper left", frameon=True)

    # ---------- Panel B: Success Throughput (kQPS) + Δ ----------
    bars = ax2.bar(x, thr_mid_k, width=0.7, yerr=thr_err_k, capsize=6, color="#2ca02c", alpha=0.8)
    ax2.set_ylabel("Success Throughput (kQPS)")
    ax2.set_title("B. Throughput Under Stress (Δ vs Baseline)")
    ax2.set_xticks(x); ax2.set_xticklabels(order_labels)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.35)

    # annotate absolute values & Δ% vs Baseline (skip Δ on Baseline)
    baseline_k = thr_mid_k[0] if len(thr_mid_k) > 0 and np.isfinite(thr_mid_k[0]) else np.nan
    for i, b in enumerate(bars):
        h = b.get_height()
        if not np.isfinite(h):
            continue
        cx = b.get_x() + b.get_width()/2.0
        ax2.text(cx, h * 1.01, f"{h:.1f}k", ha="center", va="bottom", fontsize=9, fontweight="bold")
        if i != 0 and np.isfinite(baseline_k) and baseline_k > 0:
            dpct = ((h - baseline_k) / baseline_k) * 100.0  # correct sign
            ax2.text(cx, h * 1.09, f"Δ {dpct:+.1f}%", ha="center", va="bottom", fontsize=8, color="dimgray")

    # dashed reference line at Baseline
    if np.isfinite(baseline_k):
        ax2.axhline(baseline_k, linestyle="--", linewidth=1.0, color="0.4", alpha=0.6)

    # ---------- Panel C: Reliability (stacked %) ----------
    p1 = ax3.bar(x, succ_pct,  width=0.7, label="Success", color="#2ca02c", alpha=0.85,
                 yerr=succ_err_pct, capsize=5)
    p2 = ax3.bar(x, to_pct,    width=0.7, bottom=succ_pct, label="Timeout", color="#ff7f0e", alpha=0.85)
    p3 = ax3.bar(x, np.array(err_pct), width=0.7,
                 bottom=np.array(succ_pct) + np.array(to_pct),
                 label="Error", color="#d62728", alpha=0.85)

    # inside label for Success, outside labels for small Timeout/Error if > 0.25%
    for i, r in enumerate(p1):
        s = succ_pct[i]; t = to_pct[i]; e = err_pct[i]
        ax3.text(r.get_x() + r.get_width()/2.0, s/2.0, f"{s:.1f}%",
                 ha="center", va="center", color="white", fontsize=10, fontweight="bold")
        if t > 0.25:
            ax3.text(r.get_x() + r.get_width()/2.0, s + t/2.0, f"{t:.2f}%",
                     ha="center", va="center", color="black", fontsize=8)
        if e > 0.25:
            ax3.text(r.get_x() + r.get_width()/2.0, s + t + e/2.0, f"{e:.2f}%",
                     ha="center", va="center", color="white", fontsize=8, fontweight="bold")

    ax3.set_ylabel("Rate (%)")
    ax3.set_title("C. Reliability Erosion")
    ax3.set_xticks(x); ax3.set_xticklabels(order_labels)
    ax3.set_ylim(0, 105)
    ax3.grid(True, axis="y", linestyle=":", alpha=0.35)
    ax3.legend(loc="lower left", frameon=True)

    # ---------- Save ----------
    outpath = outdir / "fig-t2-summary.pdf"
    outpath.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print("✓ Wrote fig-t2-summary.pdf")


# ------------------------------ Overhead table --------------------------------

def write_overhead_summary(df: pd.DataFrame, outdir: Path):
    """
    Writes overhead-summary.csv with median ± 95% CI (in percent units).
    Uses any overhead alias present; converts from fraction if needed.
    """
    recs = []
    for i in range(len(df)):
        scen = str(df.iloc[i]["scenario"])
        lo, mid, hi = get_ci_triplet_any(df, i, CANDIDATES["overhead"])
        lo, mid, hi = _to_pct(lo), _to_pct(mid), _to_pct(hi)
        recs.append({
            "scenario": SCENARIO_LABEL.get(scen, scen.title()),
            "overhead_mid_pct": mid,
            "overhead_lo_pct": lo,
            "overhead_hi_pct": hi,
            "text": None if mid is None else (
                f"{mid:.2f}% (95% CI [{lo:.2f}, {hi:.2f}])" if (lo is not None and hi is not None)
                else f"{mid:.2f}%"
            )
        })
    out = outdir / "overhead-summary.csv"
    pd.DataFrame(recs).to_csv(out, index=False)
    print(f"✓ Wrote {out}")

# ------------------------------ Main -----------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Tier-2 unified summary figure (camera-ready).")
    ap.add_argument("--csv", type=Path, required=True, help="Path to flattened CSV of runs (raw or summarised).")
    ap.add_argument("--out", type=Path, default=Path("figs/tier2"), help="Output directory.")
    ap.add_argument("--jitter-target", type=float, default=150.0, help="Jitter p50 target in ms.")
    ap.add_argument("--ttfr-target", type=float, default=220.0, help="TTFR p50 target in ms.")
    args = ap.parse_args()

    outdir = args.out.resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    # Load + aggregate to *_ci if needed
    df = load_flat_csv(args.csv)
    df_ci = aggregate_runs_bootstrap(df)

    # Unified figure + overhead table (for text)
    fig_tier2_summary(df_ci, outdir, jitter_target_ms=args.jitter_target, ttfr_target_ms=args.ttfr_target)
    write_overhead_summary(df_ci, outdir)

    print("Done.")

if __name__ == "__main__":
    main()
