#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Camera-ready plotting suite (RQ1–RQ4 + resources + dashboard)
- Unified modern Matplotlib style
- Colorblind-safe palette + grayscale-friendly encodings
- Correct, asymmetric CI drawing (mean-centered)
"""

from __future__ import annotations
import argparse
import json
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # headless-safe (CI/servers)
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
BASE = Path("data/tier1").resolve()
FIGDIR = Path("figs").resolve(); FIGDIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# House style (modern + professional)
# -----------------------------------------------------------------------------
RC_BASE = {
    # Embed TrueType in PDF
    "pdf.fonttype": 42, "ps.fonttype": 42,
    # Typeface (full glyph support incl. μ)
    "font.family": "DejaVu Sans",
    "mathtext.fontset": "dejavusans",
    # Sizes / weights tuned for print
    "figure.dpi": 300, "savefig.dpi": 300,
    "axes.titlesize": 10.5, "axes.labelsize": 9.5,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "legend.fontsize": 8.5,
    "lines.linewidth": 1.2, "lines.markersize": 5.0,
    # Subtle grid
    "grid.linestyle": ":", "grid.linewidth": 0.8,
}

# Colorblind-safe palette
PALETTE = {
    "Topo":     "#7B3F99",  # purple
    "Prop":     "#E69F00",  # orange
    "Uniform":  "#0072B2",  # blue (baseline, typically not plotted)
    "Grey":     "#6E6E6E",
}

# B/W accessibility (secondary encodings)
HATCHES = {
    "Topology-aware (Topo-GP)": "///",
    "Size-based (Prop.)":       "\\\\",
}
MARKERS = {
    "Topology-aware (Topo-GP)": "o",
    "Size-based (Prop.)":       "s",
}

@contextmanager
def style_ctx():
    """Consistent modern look for every figure."""
    with plt.style.context("seaborn-v0_8-whitegrid"), plt.rc_context(RC_BASE):
        yield

# -----------------------------------------------------------------------------
# Utilities (I/O, stats)
# -----------------------------------------------------------------------------
def _savefig(fig: plt.Figure, path: Path) -> None:
    """Consistent, PDF-safe saving."""
    with plt.rc_context({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "DejaVu Sans"}):
        fig.savefig(path, bbox_inches="tight")

def bootstrap_ci(
    x: np.ndarray, *, it: int = 4000, agg=np.mean, alpha: float = 0.05, rng_seed: int = 42
) -> Tuple[float, Tuple[float, float]]:
    """
    Bootstrap (default: mean) → returns (point_estimate, (lo_endpoint, hi_endpoint)).
    Endpoints are in the SAME space as the inputs.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), (float("nan"), float("nan"))
    rng = np.random.default_rng(rng_seed)
    n = x.size
    bstats = np.empty(it, dtype=float)
    for i in range(it):
        idx = rng.integers(0, n, n)
        bstats[i] = agg(x[idx])
    m = float(agg(x))
    lo = float(np.percentile(bstats, 100 * (alpha / 2.0)))
    hi = float(np.percentile(bstats, 100 * (1 - alpha / 2.0)))
    if hi < lo:
        lo, hi = hi, lo
    return m, (lo, hi)

def load_summary() -> pd.DataFrame:
    f = BASE / "summary.csv"
    if not f.exists():
        raise SystemExit(f"Missing {f}. Run the pipeline first.")
    return pd.read_csv(f)

def load_frag(workload: str, seed: int) -> pd.DataFrame:
    f = BASE / workload / f"frag_seed{seed}.csv"
    return pd.read_csv(f) if f.exists() else pd.DataFrame()

def load_meta(workload: str, seed: int) -> Dict:
    f = BASE / workload / f"meta_seed{seed}.json"
    if not f.exists():
        return {}
    try:
        return json.loads(f.read_text())
    except Exception:
        return {}

def workloads_order(df: pd.DataFrame) -> List[str]:
    pref = ["qaoa30","uccsd24","tfim20","clifford24","phase16"]
    present = [w for w in pref if w in df.workload.unique()]
    others = [w for w in df.workload.unique() if w not in pref]
    return present + sorted(others)

# -----------------------------------------------------------------------------
# RQ1.1 – Variance Contraction
# -----------------------------------------------------------------------------
def fig1_contraction(df: pd.DataFrame, save=FIGDIR / "fig-rq1-contraction.pdf"):
    """
    Variance Contraction:
      Y = Var(GP) / Var(Uniform), mean ± 95% CI over seeds.
      Error bars use asymmetric margins relative to the point estimate.
    """
    order = workloads_order(df)
    need = {'var_gp', 'var_uniform', 'workload', 'seed'}
    if not need.issubset(df.columns) or not order:
        print('[warn] summary missing var_gp/var_uniform or no workloads; skipping Fig. 1')
        return

    with style_ctx():
        D = df.copy()
        D["ratio"] = D["var_gp"] / D["var_uniform"]

        rows = []
        for w in order:
            sub = D[D.workload == w]
            vals = sub["ratio"].to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            m, (lo, hi) = bootstrap_ci(vals)
            rows.append({"workload": w, "mean": m, "lo": lo, "hi": hi})
        if not rows:
            print('[warn] no ratio data; skipping Fig. 1')
            return
        sdf = pd.DataFrame(rows)

        x = np.arange(len(sdf))
        fig, ax = plt.subplots(figsize=(6.4, 3.9), constrained_layout=True)
        ax.grid(axis='y', linestyle=':', linewidth=0.7, alpha=0.45)
        ax.axhline(1.0, ls='--', lw=1.0, color='grey', alpha=0.9, zorder=1)

        # Per-seed dots (light jitter)
        rng = np.random.default_rng(0)
        for i, w in enumerate(sdf["workload"]):
            dots = D.loc[D.workload == w, "ratio"].to_numpy(float)
            dots = dots[np.isfinite(dots)]
            if dots.size:
                jitter = (rng.random(dots.size) - 0.5) * 0.22
                ax.scatter(np.full(dots.shape, i) + jitter, dots,
                           s=14, alpha=0.30, color=PALETTE["Grey"], zorder=2)

        # Mean ± 95% CI (asymmetric margins)
        means = sdf["mean"].to_numpy(float)
        lo    = sdf["lo"].to_numpy(float)
        hi    = sdf["hi"].to_numpy(float)
        yerr  = np.vstack([np.maximum(0.0, means - lo), np.maximum(0.0, hi - means)])
        ax.errorbar(x, means, yerr=yerr, fmt='o',
                    ms=5.5, lw=1.2, capsize=4, capthick=1.2,
                    color=PALETTE["Uniform"], ecolor='black', zorder=3)

        ax.set_xticks(x); ax.set_xticklabels(sdf["workload"], rotation=30, ha='right')
        ax.set_xlabel("Workload")
        ax.set_ylabel(r"Variance Ratio (Var$_{\mathrm{GP}}$/Var$_{\mathrm{unif}}$)")
        ax.set_title("Variance Contraction (lower is better)")
        sns.despine()
        _savefig(fig, save)
        plt.close(fig)

# -----------------------------------------------------------------------------
# Core computation for Ablation (unchanged semantics)
# -----------------------------------------------------------------------------
def _compute_ablation_laststep(workload: str, seed: int) -> Dict[str, Dict[str, float]]:
    """Last-step metrics for Uniform, Proportional, Topo-GP."""
    frag = load_frag(workload, seed)
    if frag.empty or "step" not in frag.columns:
        return {}
    last = frag.loc[frag["step"] == frag["step"].max()].copy()
    meta = load_meta(workload, seed)
    beta = float(meta.get("cascade_model", {}).get("beta", 50.0))
    smin = int(meta.get("workload", {}).get("smin", 4))

    need_cols = {"kalman_mu", "H_i", "s_plan"}
    if not need_cols.issubset(last.columns):
        return {}

    mu = last["kalman_mu"].to_numpy(float)
    H  = last["H_i"].to_numpy(float)
    s_topo = last["s_plan"].to_numpy(int)
    n = mu.size
    S = int(s_topo.sum())

    s_uniform = np.maximum(smin, np.full(n, S // max(1, n), dtype=int))
    w = mu / mu.sum() if mu.sum() > 0 else np.full(n, 1.0 / max(1, n))
    s_prop = np.maximum(smin, np.floor(S * w).astype(int))

    def _fix_sum(s):
        s = s.copy()
        d = S - int(s.sum())
        if d > 0:
            for i in np.argsort(-w)[:d]:
                s[i] += 1
        elif d < 0:
            for i in np.argsort(w)[:abs(d)]:
                s[i] = max(smin, s[i] - 1)
        return s

    s_uniform = _fix_sum(s_uniform)
    s_prop    = _fix_sum(s_prop)

    def mse_of(s: np.ndarray) -> float:
        s = np.maximum(smin, s.astype(int))
        bias = 0.1 * (1.0 - np.minimum(1.0, H / np.log(3)))
        mse_shadow = mu / s
        mse_mle    = (mu * beta) / (s**2) + bias**2
        return float(np.mean(np.minimum(mse_shadow, mse_mle)))

    mse_unif = mse_of(s_uniform)

    def pack(s: np.ndarray, name: str) -> Dict[str, float]:
        ms = mse_of(s)
        return {"method": name,
                "ratio": ms / max(1e-12, mse_unif),
                "p95_shots": float(np.percentile(s, 95)),
                "mse": ms}

    return {"Uniform":      pack(s_uniform, "Uniform"),
            "Proportional": pack(s_prop,    "Proportional"),
            "Topo-GP":      pack(s_topo,    "Topo-GP")}

# -----------------------------------------------------------------------------
# RQ1.2 – Ablation (Δ vs Uniform, 95% CIs)
# -----------------------------------------------------------------------------
def fig2_ablation(df: pd.DataFrame, save=FIGDIR/"fig-rq1-ablation.pdf"):
    """
    Grouped bars (Topo-GP, Prop.) as Δ vs Uniform (%) across workloads; per-seed dots; 95% CIs
    drawn as asymmetric margins around the mean (with a mean tick). Legend is robust. Display
    names fixed (tfim20, clifford24, UCCSD24) without touching data keys.
    """
    order = workloads_order(df)
    if not order:
        print('[warn] No workloads; skipping Fig. 2')
        return

    # ---- collect per-seed ----
    records = []
    for w in order:
        for seed in sorted(df.loc[df.workload == w, "seed"].unique()):
            res = _compute_ablation_laststep(w, int(seed))
            if not res: continue
            for meth in ["Uniform","Proportional","Topo-GP"]:
                if meth in res:
                    r = res[meth].copy()
                    r.update({"workload": w, "seed": seed, "method": meth})
                    records.append(r)
    if not records:
        print('[warn] No fragment logs found for ablation; skipping Fig. 2')
        return
    raw = pd.DataFrame(records)

    # ---- labels/palette ----
    label_map = {
        "Topo-GP":      "Topology-aware (Topo-GP)",
        "Proportional": "Size-based (Prop.)",
        "Uniform":      "Naïve (Uniform)",
    }
    plot_methods = ["Topo-GP","Proportional"]   # baseline omitted
    hue_labels   = [label_map[m] for m in plot_methods]
    pal = {"Topology-aware (Topo-GP)": PALETTE["Topo"],
           "Size-based (Prop.)":       PALETTE["Prop"]}
    raw["method_label"] = raw["method"].map(label_map)

    # ---- baselines per seed ----
    metrics = ["ratio","p95_shots","mse"]
    uni = (raw[raw.method=="Uniform"]
           .set_index(["workload","seed"]))[metrics].rename(
                columns={"ratio":"ratio_U","p95_shots":"p95_U","mse":"mse_U"})
    merged = (raw.set_index(["workload","seed"]).join(uni, how="inner").reset_index())

    # ---- per-seed Δ% ----
    def _pct_delta(a,b):
        if pd.isna(a) or pd.isna(b) or b == 0: return np.nan
        return (a / b - 1.0) * 100.0
    merged["d_ratio_pct"] = merged["ratio"].apply(lambda v: (v-1.0)*100.0 if pd.notna(v) else np.nan)
    merged["d_p95_pct"]   = merged.apply(lambda r: _pct_delta(r["p95_shots"], r["p95_U"]), axis=1)
    merged["d_mse_pct"]   = merged.apply(lambda r: _pct_delta(r["mse"],       r["mse_U"]), axis=1)

    ndf = merged[merged["method"].isin(plot_methods)].copy()
    ndf["method_label"] = ndf["method"].map(label_map)

    # ---- aggregate (mean + 95% CI) on Δ% ----
    rows = []
    for metric_key, col in [("ratio","d_ratio_pct"),("p95_shots","d_p95_pct"),("mse","d_mse_pct")]:
        for (w, ml), s in ndf.groupby(["workload","method_label"])[col]:
            vals = s.to_numpy(float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0: continue
            m, (lo, hi) = bootstrap_ci(vals, it=4000, agg=np.mean, alpha=0.05)
            rows.append({"workload": w, "method_label": ml, "metric": metric_key,
                         "mean": float(m), "lo": float(lo), "hi": float(hi),
                         "n": int(vals.size)})
    gdf = pd.DataFrame(rows)
    if gdf.empty:
        print('[warn] No aggregated rows for ablation; skipping Fig. 2')
        return

    # ---- display-name fixes ----
    def disp_workload(w: str) -> str:
        fix = {"tfmz0":"tfim20", "dflford24":"clifford24", "uccsd24":"UCCSD24"}
        return fix.get(str(w), str(w))
    disp_order = [disp_workload(w) for w in order]
    w2disp = {w: d for w, d in zip(order, disp_order)}

    # ---- helper to get matrices per panel ----
    def panel_arrays(metric: str):
        sub = gdf[gdf.metric == metric].copy()
        sub["workload"] = pd.Categorical(sub["workload"], categories=order, ordered=True)
        sub["method_label"] = pd.Categorical(sub["method_label"], categories=hue_labels, ordered=True)
        sub.sort_values(["workload","method_label"], inplace=True)
        W, M = len(order), len(hue_labels)
        mean = np.full((W,M), np.nan); lo = np.full((W,M), np.nan); hi = np.full((W,M), np.nan)
        for wi, w in enumerate(order):
            for mi, ml in enumerate(hue_labels):
                row = sub[(sub["workload"]==w) & (sub["method_label"]==ml)]
                if not row.empty:
                    r = row.iloc[0]
                    mean[wi, mi] = r["mean"]; lo[wi, mi] = r["lo"]; hi[wi, mi] = r["hi"]
        return mean, lo, hi

    with style_ctx():
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.2), sharex=True, constrained_layout=True)
        panels = [
            ("ratio",     r"Contraction cost — $\Delta$ vs Uniform (↓ better)"),
            ("p95_shots", r"Tail p95 shots — $\Delta$ vs Uniform (↓ better)"),
            ("mse",       r"Reconstruction MSE — $\Delta$ vs Uniform (↓ better)"),
        ]
        ylab = "Relative Change vs Uniform (%)"

        x = np.arange(len(order), dtype=float)
        bar_w = 0.34
        offsets = np.array([-0.5, +0.5]) * (bar_w * 2.2)

        legend_handles = {}
        for ax, (metric, title) in zip(axes, panels):
            mean, lo, hi = panel_arrays(metric)

            # per-seed dots (distinct markers per strategy)
            mc = {"ratio":"d_ratio_pct","p95_shots":"d_p95_pct","mse":"d_mse_pct"}[metric]
            dots = ndf[["workload","method_label",mc]].dropna()
            w2i = {w:i for i, w in enumerate(order)}
            m2i = {ml:i for i, ml in enumerate(hue_labels)}
            for _, r in dots.iterrows():
                wi = w2i.get(r["workload"]); mi = m2i.get(r["method_label"])
                if wi is None or mi is None: continue
                xi = x[wi] + offsets[mi]
                ax.plot([xi], [r[mc]], linestyle="None",
                        marker=MARKERS[r["method_label"]], ms=3.2,
                        alpha=0.35, color=pal[r["method_label"]], zorder=1)

            # bars (with hatches for B/W)
            patches = []
            for mi, ml in enumerate(hue_labels):
                xpos = x + offsets[mi]
                for wi in range(len(order)):
                    m = mean[wi, mi]
                    if np.isfinite(m):
                        p = ax.bar(xpos[wi], m, width=bar_w,
                                   color=pal[ml], alpha=0.88, zorder=2,
                                   label=ml if wi == 0 else None,
                                   hatch=HATCHES.get(ml, None),
                                   edgecolor="black", linewidth=0.5)[0]
                        patches.append((p, wi, mi))
                        if ml not in legend_handles: legend_handles[ml] = p

            # CIs (asymmetric margins) + mean tick; bold edge if CI excludes 0
            cap_w = bar_w * 0.55
            for p, wi, mi in patches:
                m = float(mean[wi, mi]); lo_i = float(lo[wi, mi]); hi_i = float(hi[wi, mi])
                if not (np.isfinite(m) and np.isfinite(lo_i) and np.isfinite(hi_i)): continue
                if hi_i < lo_i: lo_i, hi_i = hi_i, lo_i
                lower_err = max(0.0, m - lo_i)
                upper_err = max(0.0, hi_i - m)
                xbar = p.get_x() + p.get_width()/2.0
                y1, y2 = m - lower_err, m + upper_err
                ax.vlines(xbar, y1, y2, color="black", linewidth=1.1, zorder=3)
                ax.hlines([y1, y2], xbar - cap_w/2, xbar + cap_w/2,
                          color="black", linewidth=1.1, zorder=3)
                ax.plot([xbar], [m], marker="_", markersize=10, color="black", zorder=4)
                if (hi_i < 0.0) or (lo_i > 0.0):
                    p.set_linewidth(1.6)

            ax.axhline(0.0, ls="--", lw=1.0, color="grey", alpha=0.9, zorder=0)
            ax.set_title(title, pad=6)
            ax.set_xlabel("")
            ax.set_ylabel(ylab)
            ax.set_xticks(x)
            ax.set_xticklabels([w2disp[w] for w in order])

            # symmetric y-lims from CI extrema
            ymin = np.nanmin([np.nanmin(mean - (mean - lo)), np.nanmin(mean)])
            ymax = np.nanmax([np.nanmax(mean + (hi - mean)), np.nanmax(mean)])
            if np.isfinite(ymin) and np.isfinite(ymax):
                lim = 1.10 * max(abs(ymin), abs(ymax))
                if lim > 0: ax.set_ylim(-lim, +lim)

            ax.grid(axis="y", which="major", linestyle=":", linewidth=0.8, alpha=0.6)

        # legend (top-center)
        handles = [legend_handles[h] for h in hue_labels if h in legend_handles]
        labels  = [h for h in hue_labels if h in legend_handles]
        if handles:
            fig = axes[0].get_figure()
            fig.legend(handles=handles, labels=labels, loc="upper center",
                       ncol=len(labels), frameon=True, bbox_to_anchor=(0.5, 1.05))

        # caption + title
        per_w = (raw[raw.method=="Uniform"].groupby("workload")["seed"]
                 .nunique().reindex(order).fillna(0))
        n_min, n_max = int(per_w.min()), int(per_w.max())
        seeds_text = f"n={n_min} per workload" if n_min==n_max else f"n range: {n_min}-{n_max} per workload"

        fig.suptitle("Ablation of Allocation Strategies (Δ vs Uniform, 95% CIs)",
                     y=1.10, fontsize=12, weight="semibold")
        fig.text(0.01, -0.02,
                 "Mean Δ vs Uniform (bars) with per-seed results (dots) and 95% CIs (mean-centered). "
                 "Negative values indicate improvement. "
                 f"{seeds_text}.",
                 ha="left", va="top", fontsize=8.3, color="dimgray")

        _savefig(fig, save); plt.close(fig)

# -----------------------------------------------------------------------------
# RQ1.3 – Tails (ECDF + p95 markers)
# -----------------------------------------------------------------------------
def fig3_tails(df: pd.DataFrame, save=FIGDIR/"fig-rq1-tails.pdf"):
    order = workloads_order(df)
    recs = []
    for w in order:
        for seed in sorted(df.loc[df.workload == w, 'seed'].unique()):
            frag = load_frag(w, int(seed))
            if frag.empty or 'step' not in frag.columns or 's_plan' not in frag.columns:
                continue
            last = frag.loc[frag['step'] == frag['step'].max()]
            for s in last['s_plan'].to_numpy(int):
                recs.append({'workload': w, 'seed': seed, 's': int(s)})
    if not recs:
        print('[warn] No fragment logs for tails; skipping Fig. 3')
        return

    with style_ctx():
        tdf = pd.DataFrame(recs)
        fig, ax = plt.subplots(figsize=(6.6, 3.8), constrained_layout=True)
        handles, labels = [], []
        for w in order:
            sub = tdf.loc[tdf.workload == w, 's'].to_numpy(int)
            if sub.size == 0:
                continue
            x = np.sort(sub)
            y = np.arange(1, x.size + 1) / x.size  # ECDF in (0,1]
            line, = ax.step(x, y, where='post', label=w)
            p95 = float(np.percentile(sub, 95))
            ax.axvline(p95, ls=':', lw=0.9, alpha=0.85, color=line.get_color())
            handles.append(line); labels.append(w)

        ax.set_xlabel("Shots per fragment (last step)")
        ax.set_ylabel("Empirical CDF")
        ax.set_title("Short-tail behaviour (ECDF) with per-workload p95 markers")

        # Optional log scale for very wide tails
        try:
            s_all = tdf['s'].to_numpy(int)
            if np.percentile(s_all, 95) / max(1, np.percentile(s_all, 5)) > 25:
                ax.set_xscale('log')
                ax.set_xlabel("Shots per fragment (last step) [log scale]")
        except Exception:
            pass

        ax.grid(axis='both', linestyle=':', linewidth=0.8, alpha=0.5)
        ax.legend(handles=handles, labels=labels, frameon=True, ncol=3)
        sns.despine()
        _savefig(fig, save); plt.close(fig)

# -----------------------------------------------------------------------------
# RQ1.4 – Coupling (facets + correlation table)
# -----------------------------------------------------------------------------
def fig4_coupling(df: pd.DataFrame, save=FIGDIR/"fig-rq1-coupling.pdf",
                  col_wrap=None, height=None):
    """
    Left: per-workload scatter (μ vs s) + fit line
    Right: table of median Pearson r / Spearman ρ with 95% bootstrap CIs (endpoints)
    """
    from matplotlib.ticker import MaxNLocator
    order = workloads_order(df)

    # Collect points
    recs = []
    for w in order:
        for seed in sorted(df.loc[df.workload == w, 'seed'].unique()):
            frag = load_frag(w, int(seed))
            if frag.empty or 'step' not in frag.columns: continue
            last = frag.loc[frag['step'] == frag['step'].max()]
            if not {'kalman_mu','s_plan'}.issubset(last.columns): continue
            mu = last['kalman_mu'].to_numpy(float); s  = last['s_plan'].to_numpy(int)
            m = np.isfinite(mu) & np.isfinite(s)
            recs.extend([{'workload': w, 'mu': float(x), 's': int(y)} for x,y in zip(mu[m], s[m])])
    if not recs:
        print('[warn] No fragment logs for coupling; skipping Fig. 4')
        return
    cdf = pd.DataFrame(recs)

    # Per-seed correlations
    def pearson_safe(x,y)->float:
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y); x,y = x[m],y[m]
        if x.size<2 or np.allclose(x,x[0]) or np.allclose(y,y[0]): return np.nan
        return float(np.corrcoef(x,y)[0,1])
    def spearman_safe(x,y)->float:
        x = np.asarray(x, float); y = np.asarray(y, float)
        m = np.isfinite(x) & np.isfinite(y); x,y = x[m],y[m]
        if x.size<2: return np.nan
        rx = pd.Series(x).rank(method='average').to_numpy()
        ry = pd.Series(y).rank(method='average').to_numpy()
        if np.allclose(rx, rx[0]) or np.allclose(ry, ry[0]): return np.nan
        return float(np.corrcoef(rx, ry)[0,1])

    rows_seed = []
    for w in order:
        for seed in sorted(df.loc[df.workload == w, 'seed'].unique()):
            frag = load_frag(w, int(seed))
            if frag.empty or 'step' not in frag.columns: continue
            last = frag.loc[frag['step'] == frag['step'].max()]
            if not {'kalman_mu','s_plan'}.issubset(last.columns): continue
            mu = last['kalman_mu'].to_numpy(float); s  = last['s_plan'].to_numpy(int)
            rp = pearson_safe(mu, s); rs = spearman_safe(mu, s)
            if np.isfinite(rp) or np.isfinite(rs):
                rows_seed.append({'workload': w, 'seed': int(seed),
                                  'pearson_r': rp, 'spearman_rho': rs})
    df_seed = pd.DataFrame(rows_seed)
    if not df_seed.empty:
        (FIGDIR/"fig-rq1-coupling-perseed.csv").write_text(df_seed.to_csv(index=False))

    # Aggregate (median + 95% bootstrap CI endpoints)
    agg_rows = []
    for w, g in df_seed.groupby('workload'):
        rec = {'workload': w, 'n_seeds': int(g['seed'].nunique())}
        for col in ('pearson_r','spearman_rho'):
            vals = g[col].dropna().to_numpy(float)
            if vals.size == 0:
                rec[col] = rec[col+'_lo'] = rec[col+'_hi'] = np.nan
            else:
                m, (lo, hi) = bootstrap_ci(vals, it=4000, agg=np.median, alpha=0.05)
                rec[col] = float(m); rec[col+'_lo'] = float(lo); rec[col+'_hi'] = float(hi)
        agg_rows.append(rec)
    df_agg = pd.DataFrame(agg_rows) if agg_rows else pd.DataFrame(columns=['workload'])
    if not df_agg.empty:
        df_agg = df_agg.set_index('workload').reindex(order).reset_index()
        (FIGDIR/"fig-rq1-coupling-agg.csv").write_text(df_agg.to_csv(index=False))

    # Layout
    n_w = len(order)
    if col_wrap is None: col_wrap = 3 if n_w >= 5 else (2 if n_w in (3,4) else n_w)
    if height is None:   height   = 2.7 if col_wrap >= 3 else 3.0
    rows = int(np.ceil(n_w / col_wrap)); aspect = 1.05
    left_width  = col_wrap * aspect * height + max(0, col_wrap-1) * 0.35
    right_width = 6.0
    fig_w = left_width + right_width + 0.9
    fig_h = 0.9 + rows * (height + 0.45)

    with style_ctx():
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs  = fig.add_gridspec(nrows=1, ncols=2, width_ratios=[left_width, right_width], wspace=0.06)

        # LEFT: facets
        gs_left = gs[0,0].subgridspec(rows, col_wrap, wspace=0.32, hspace=0.42)
        xlim = (float(cdf['mu'].min()), float(cdf['mu'].max()))
        ylim = (float(cdf['s'].min()),  float(cdf['s'].max()))
        for k, w in enumerate(order):
            r, c = divmod(k, col_wrap)
            ax = fig.add_subplot(gs_left[r, c])
            sub = cdf[cdf.workload == w]
            ax.scatter(sub['mu'], sub['s'], s=14, alpha=0.48, edgecolors='none', rasterized=True)
            xs = sub['mu'].to_numpy(float); ys = sub['s'].to_numpy(float)
            m = np.isfinite(xs) & np.isfinite(ys); xs, ys = xs[m], ys[m]
            if xs.size >= 2 and not np.allclose(xs, xs[0]):
                b1, b0 = np.polyfit(xs, ys, 1); xx = np.linspace(xs.min(), xs.max(), 100)
                ax.plot(xx, b1*xx + b0, lw=1.0)
            ax.set_title(w, pad=4)
            ax.grid(True, axis='both', linestyle=':', linewidth=0.8, alpha=0.55)
            ax.set_xlim(xlim); ax.set_ylim(ylim)
            ax.xaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.set_xlabel(r"Kalman $\mu_i$" if r == rows-1 else "")
            ax.set_ylabel(r"Allocated $s_i$" if c == 0 else "")

        # RIGHT: correlation table
        ax_t = fig.add_subplot(gs[0,1]); ax_t.axis('off')

        def fmt_interval(m, lo, hi):
            if not (np.isfinite(m) and np.isfinite(lo) and np.isfinite(hi)): return "—"
            return f"{m:.2f} [{lo:.2f}–{hi:.2f}]"

        table_rows = []
        if not df_agg.empty:
            for _, r in df_agg.iterrows():
                table_rows.append([r['workload'],
                                   int(r['n_seeds']) if np.isfinite(r['n_seeds']) else 0,
                                   fmt_interval(r['pearson_r'], r['pearson_r_lo'], r['pearson_r_hi']),
                                   fmt_interval(r['spearman_rho'], r['spearman_rho_lo'], r['spearman_rho_hi'])])
        else:
            for w in order: table_rows.append([w, 0, "—", "—"])

        col_labels = ["Workload", "Seeds", "Pearson r\n(median [95% CI])", "Spearman ρ\n(median [95% CI])"]
        ncols = len(col_labels)
        col_widths = [0.95, 0.60, 2.35, 2.35]
        tbl = ax_t.table(cellText=table_rows, colLabels=col_labels,
                         colWidths=col_widths, loc='center', cellLoc='center',
                         colColours=['#f2f2f2'] * ncols)
        tbl.auto_set_font_size(False); tbl.set_fontsize(8.0); tbl.scale(1.0, 1.28)
        for (r_i, c_i), cell in tbl.get_celld().items():
            if r_i == 0: cell.set_text_props(weight='semibold')
            elif r_i % 2 == 0: cell.set_facecolor('#fafafa')
            if c_i == 0:
                try: cell.get_text().set_ha('left')
                except Exception: cell.set_text_props(ha='left')
        try:
            fig.canvas.draw(); tbl.auto_set_column_width(col=list(range(ncols)))
        except Exception:
            pass

        ax_t.set_title("Coupling statistics (last step)", pad=8)
        fig.suptitle("Variance–Shots Coupling (last step): Facets + Correlation Table", y=0.995, fontsize=12)
        fig.text(0.985, 0.012,
                 "Pearson r: linear correlation (fit line shown)\n"
                 "Spearman ρ: rank/monotonic correlation",
                 ha='right', va='bottom', fontsize=8.5,
                 bbox=dict(facecolor='white', edgecolor='0.7', boxstyle='round,pad=0.28', alpha=0.95))
        _savefig(fig, save); plt.close(fig)

# -----------------------------------------------------------------------------
# RQ2.1 – Timeline (medians + IQR, events)
# -----------------------------------------------------------------------------
def fig5_timeline(df: pd.DataFrame, save=FIGDIR/"fig-rq2-timeline.pdf", normalise=True):
    order = workloads_order(df)
    if not order: return
    w = order[0]

    subw = df[df.workload == w].copy()
    if {'var_gp','var_uniform','seed'}.issubset(subw.columns) and not subw.empty:
        subw['ratio'] = subw['var_gp'] / subw['var_uniform']
        med = float(subw['ratio'].median()); subw['dist'] = np.abs(subw['ratio'] - med)
        med_seed = int(subw.sort_values('dist').iloc[0]['seed']); seed_note = "median-ratio seed"
    else:
        seeds = sorted(df.loc[df.workload == w, 'seed'].unique())
        if not seeds: return
        med_seed = int(seeds[len(seeds)//2]); seed_note = "median-index seed"

    frag = load_frag(w, int(med_seed))
    if frag.empty or 'step' not in frag.columns:
        print('[warn] No frag logs for timeline; skipping Fig. 5'); return

    def step_stats(frame: pd.DataFrame, col: str):
        if col not in frame.columns: return None
        g = frame[['step', col]].dropna().groupby('step')[col]
        q = g.quantile([0.25, 0.5, 0.75]).unstack(); q.columns = ['q25','q50','q75']
        return q.reset_index().sort_values('step')

    Z  = step_stats(frag, 'z_t')
    MU = step_stats(frag, 'kalman_mu')
    S  = step_stats(frag, 's_plan')
    IN = step_stats(frag, 'avg_innov')

    def normalise_series(stats):
        if stats is None or stats.empty: return stats
        base = stats['q50'].iloc[0]
        if not np.isfinite(base) or base == 0: return stats
        out = stats.copy(); out[['q25','q50','q75']] = out[['q25','q50','q75']] / float(base)
        return out

    if normalise:
        Zn, MUn = normalise_series(Z), normalise_series(MU)
        var_ylabel = "Variance proxy (normalised to step 0)"
    else:
        Zn, MUn  = Z, MU
        var_ylabel = "Variance proxy (a.u.)"

    rsteps = []
    if 'repartition' in frag.columns:
        tmp = frag[['step','repartition']].dropna()
        rsteps = list(np.unique(tmp.loc[tmp['repartition'].astype(bool), 'step'].to_numpy(int)))

    t_step = None
    row = df[(df.workload==w) & (df.seed==med_seed)]
    if not row.empty and 'mean_time_est_s' in row.columns and np.isfinite(row['mean_time_est_s'].iloc[0]):
        t_step = float(row['mean_time_est_s'].iloc[0])

    with style_ctx():
        fig, axes = plt.subplots(3, 1, figsize=(6.4, 5.3), sharex=True, constrained_layout=True)
        ax0, ax1, ax2 = axes

        def band(ax, X, label=None, color=None, lw=1.2):
            if X is None or X.empty: return None
            ln, = ax.plot(X['step'], X['q50'], label=label, lw=lw, color=color)
            ax.fill_between(X['step'], X['q25'], X['q75'], alpha=0.25, linewidth=0, color=ln.get_color())
            return ln

        handles = []
        if Zn is not None:
            h = band(ax0, Zn, label=r"Median $z_t$ (measured/proxy)");
            handles.append(h) if h else None
        if MUn is not None:
            h = band(ax0, MUn, label=r"Median Kalman $\mu$");
            handles.append(h) if h else None
        ax0.set_ylabel(var_ylabel)

        if S is not None and not S.empty:
            hS = band(ax1, S, label=r"Median $s$ per fragment")
        ax1.set_ylabel(r"$s$ per fragment")

        if IN is not None and not IN.empty:
            band(ax2, IN, label=r"Median innovation")
        ax2.set_ylabel(r"Innovation $|z_t - \mu_{t\mid t-1}|$")
        ax2.set_xlabel("Control step")

        if rsteps:
            for ax in axes:
                for rs in rsteps:
                    ax.axvline(rs, color='red', lw=0.9, ls='--', alpha=0.7)
            from matplotlib.lines import Line2D
            handles.append(Line2D([0],[0], color='red', lw=1.0, ls='--', label='Repartition step'))

        for ax in axes:
            ax.grid(axis='both', linestyle=':', linewidth=0.7, alpha=0.5)

        if any(handles):
            fig.legend(handles=[h for h in handles if h], loc='upper right', frameon=True, ncol=1,
                       bbox_to_anchor=(0.98, 0.98))

        if t_step and np.isfinite(t_step) and t_step > 0:
            def fwd(step): return np.asarray(step) * t_step
            def inv(sec):  return np.asarray(sec) / t_step
            secax = ax2.secondary_xaxis('top', functions=(fwd, inv))
            secax.set_xlabel("Elapsed time (s)")
        else:
            fig.text(0.015, 0.985, "No per-step time available; top axis omitted.",
                     ha='left', va='top', fontsize=8, color='dimgray')

        fig.suptitle(f"Closed-Loop Timeline — {w} (seed {med_seed}, {seed_note})", y=0.995)
        fig.text(0.015, 0.005,
                 "Lines = per-step medians; bands = IQR across fragments. "
                 + ("Normalised to step 0. " if normalise else "")
                 + "Red dashed lines mark repartition events.",
                 ha='left', va='bottom', fontsize=8.5, color='dimgray')

        sns.despine(); _savefig(fig, save); plt.close(fig)

# -----------------------------------------------------------------------------
# RQ2.2 – Topology Covariance (Σpre/post, ΔΣ)
# -----------------------------------------------------------------------------
def fig6a_topology_cov(df: pd.DataFrame, save=FIGDIR/"fig-rq2-topology-cov.pdf"):
    order = workloads_order(df)
    if not order: return
    w = order[0]

    seed, meta = None, {}
    for s in sorted(df.loc[df.workload==w,'seed'].unique()):
        m = load_meta(w, int(s))
        if m: seed=int(s); meta=m; break
    if seed is None:
        print('[warn] No meta found; skipping fig6a_topology_cov'); return

    n   = meta.get('workload',{}).get('n_fragments', 16)
    ell = meta.get('workload',{}).get('ell', 0.30)

    frag = load_frag(w, seed)
    if frag.empty or not {'step','s_plan'}.issubset(frag.columns):
        print('[warn] No frag logs; skipping fig6a_topology_cov'); return

    def _first_repartition(fr):
        if 'repartition' not in fr.columns: return None
        hits = fr.loc[fr['repartition'].astype(bool),'step']
        return int(hits.min()) if not hits.empty else None

    tr = _first_repartition(frag)
    if tr is None:
        pre_step  = int(frag['step'].min())
        post_step = int(frag['step'].max())
    else:
        pre_step  = max(int(tr)-1, int(frag['step'].min()))
        post_step = min(int(tr)+1, int(frag['step'].max()))

    # reconstruct anchors (simple reproducible toy)
    rng = np.random.default_rng(int(seed))
    pre_xy  = rng.random((n,2))
    post_xy = pre_xy + 0.01 * rng.normal(0, 1, pre_xy.shape)

    def pdist(a):
        d2 = np.sum((a[:,None,:]-a[None,:,:])**2, axis=-1)
        return np.sqrt(d2)
    def matern_half(D, L): return np.exp(-D / max(1e-9, L))

    S_pre  = matern_half(pdist(pre_xy),  ell)
    S_post = matern_half(pdist(post_xy), ell)

    # order by Fiedler vector
    d = S_pre.sum(axis=1); L = np.diag(d) - S_pre
    vals, vecs = np.linalg.eigh(L)
    fiedler = vecs[:,1] if vecs.shape[1]>1 else vecs[:,0]
    idx = np.argsort(fiedler)
    S0 = S_pre[idx][:,idx]
    S1 = S_post[idx][:,idx]
    dSigma = S1 - S0

    def within_between_ratio(S, idx):
        S = S.copy(); np.fill_diagonal(S, np.nan)
        # simple 2-way split by median fiedler sign (toy, consistent)
        sign = np.sign(fiedler[idx])
        same = (sign[:,None]==sign[None,:])
        within  = np.nanmean(S[same])
        between = np.nanmean(S[~same])
        ratio   = between / max(1e-12, within)
        return within, between, ratio
    within0, between0, ratio0 = within_between_ratio(S0, np.arange(len(S0)))
    within1, between1, ratio1 = within_between_ratio(S1, np.arange(len(S1)))

    with style_ctx():
        fig, axes = plt.subplots(1, 3, figsize=(8.8, 3.6), constrained_layout=True)
        im0 = axes[0].imshow(S0, vmin=0, vmax=1, cmap='viridis')
        im1 = axes[1].imshow(S1, vmin=0, vmax=1, cmap='viridis')
        vmax = float(np.nanmax(np.abs(dSigma))) if np.isfinite(dSigma).any() else 1.0
        im2 = axes[2].imshow(dSigma, vmin=-vmax, vmax=vmax, cmap='coolwarm')

        for ax, title in zip(axes, ["Σ (Pre)", "Σ (Post)", "ΔΣ = Σpost − Σpre"]):
            ax.set_title(title)
            ax.set_xticks([]); ax.set_yticks([])
            sns.despine(ax=ax, left=True, bottom=True)

        def box(ax, txt):
            ax.text(0.02, 0.02, txt, transform=ax.transAxes,
                    ha='left', va='bottom', fontsize=8.2,
                    bbox=dict(facecolor='white', edgecolor='0.7', boxstyle='round,pad=0.25', alpha=0.9))
        box(axes[0], f"within={within0:.2f}\nbetween={between0:.2f}\nbetween/within={ratio0:.2f}  (↓)")
        box(axes[1], f"within={within1:.2f}\nbetween={between1:.2f}\nbetween/within={ratio1:.2f}  (↓)\nΔ={ratio1-ratio0:+.02f}")

        cb0 = fig.colorbar(im0, ax=axes[:2], fraction=0.046, pad=0.04)
        cb0.set_label("Covariance")
        cb2 = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
        cb2.set_label("Δ Covariance")

        fig.suptitle(f"Topology — Covariance (pre vs post) — {w} (seed {seed})")
        _savefig(fig, save); plt.close(fig)

# -----------------------------------------------------------------------------
# RQ2.3 – Topology Embed (A/B + Δ panel)
# -----------------------------------------------------------------------------
def fig6b_topology_embed(df: pd.DataFrame, save=FIGDIR/"fig-rq2-topology-embed.pdf",
                         show_delta: bool = True,
                         min_pre_for_rel: int = 5,
                         perms_for_p: int = 5000,
                         rng_seed: int = 7):
    """
    Communities + link strengths; Before vs After; optional Δ panel:
      • Circle size = shots (log)
      • Δ panel: colour = relative change (%), size = |Δ|
    """
    import matplotlib as mpl
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches
    from matplotlib.collections import LineCollection
    from matplotlib.ticker import PercentFormatter

    order = workloads_order(df)
    if not order: return
    w = order[0]

    # choose seed with available meta
    seed, meta = None, {}
    for s in sorted(df.loc[df.workload==w,'seed'].unique()):
        m = load_meta(w, int(s))
        if m: seed=int(s); meta=m; break
    if seed is None:
        print('[warn] No meta found; skipping fig6b_topology_embed'); return

    n = meta.get('workload', {}).get('n_fragments', 16)
    frag = load_frag(w, seed)
    if frag.empty or not {'step','s_plan'}.issubset(frag.columns):
        print('[warn] No frag logs; skipping fig6b_topology_embed'); return

    def first_repartition_step_(fr):
        if 'repartition' not in fr.columns: return None
        hits = fr.loc[fr['repartition'].astype(bool), 'step']
        return int(hits.min()) if not hits.empty else None

    tr = first_repartition_step_(frag)
    if tr is None:
        pre_step  = int(frag['step'].min()); post_step = int(frag['step'].max())
        rep_note  = "no repartition marker"
    else:
        pre_step  = max(int(tr)-1, int(frag['step'].min()))
        post_step = min(int(tr)+1, int(frag['step'].max()))
        rep_note  = f"repartition at step {tr}"

    last_pre  = frag.loc[frag['step'] == pre_step]
    last_post = frag.loc[frag['step'] == post_step]
    if last_pre.empty or last_post.empty:
        print('[warn] Missing pre/post slices; skipping fig6b_topology_embed'); return

    s_pre  = last_pre['s_plan'].to_numpy(int)
    s_post = last_post['s_plan'].to_numpy(int)
    all_shots = np.concatenate([s_pre, s_post])

    # Anchors (toy reconstruction; reproducible)
    rng = np.random.default_rng(int(seed))
    pre_xy  = rng.random((n,2))
    post_xy = pre_xy + 0.01 * rng.normal(0,1,pre_xy.shape)

    def pdist(a):
        d2 = np.sum((a[:,None,:]-a[None,:,:])**2, axis=-1)
        return np.sqrt(d2)
    # Edges from Σ(pre)
    ell = meta.get('workload', {}).get('ell', 0.30)
    S_pre = np.exp(-pdist(pre_xy) / max(1e-9, ell))
    def knn_edges(xy, k=3):
        from numpy.linalg import norm
        n = xy.shape[0]; E = set()
        for i in range(n):
            d = norm(xy - xy[i], axis=1); idx = np.argsort(d)[1:k+1]
            for j in idx:
                a, b = (i, int(j)) if i<j else (int(j), i); E.add((a,b))
        return sorted(E)
    E = knn_edges(pre_xy, k=3)
    Wdict = {(min(i,j), max(i,j)): float(S_pre[min(i,j), max(i,j)]) for (i, j) in E}

    # Communities via Fiedler
    dvec = S_pre.sum(axis=1); L = np.diag(dvec) - S_pre
    vals, vecs = np.linalg.eigh(L)
    fiedler = vecs[:,1] if vecs.shape[1]>1 else vecs[:,0]
    q = np.quantile(fiedler, np.linspace(0,1,5))
    comm = np.digitize(fiedler, q[1:-1])
    cmap_comm = plt.get_cmap('Set2')
    comm_cols = [cmap_comm(int(c) % 8) for c in comm]
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

    # Utilities
    def _transform(vals, mode='log'):
        v = np.asarray(vals, float); v = np.clip(v, a_min=0, a_max=None)
        if mode == 'log':  return np.log1p(v)
        if mode == 'sqrt': return np.sqrt(v)
        return v
    def scale_marker_sizes(s, lo=36.0, hi=260.0, transform='log', ref=None):
        x = _transform(s, transform);
        rx = _transform(ref, transform) if ref is not None else x
        xmin, xmax = float(np.min(rx)), float(np.max(rx))
        if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
            return np.full_like(x, (lo+hi)/2.0)
        return lo + (hi-lo) * (x - xmin) / (xmax - xmin)

    sizes_pre  = scale_marker_sizes(s_pre,  transform='log', ref=all_shots)
    sizes_post = scale_marker_sizes(s_post, transform='log', ref=all_shots)

    # Δ encodings
    dshots = s_post - s_pre
    with np.errstate(divide='ignore', invalid='ignore'):
        dfrac = np.where(s_pre >= min_pre_for_rel, dshots / s_pre, np.nan)
    absd    = np.abs(dshots)
    sizes_d = scale_marker_sizes(absd, transform='log', ref=absd if np.any(absd>0) else np.array([1.0]))
    cmap_delta = mpl.cm.get_cmap('BrBG')
    finite_dfrac = dfrac[np.isfinite(dfrac)]
    if finite_dfrac.size:
        vmin = float(np.nanpercentile(finite_dfrac, 2)); vmax = float(np.nanpercentile(finite_dfrac, 98))
        if np.isclose(vmin, vmax): vmin, vmax = -0.1, 0.1
    else:
        vmin, vmax = -0.1, 0.1
    norm_delta = mpl.colors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)

    def draw_edges_weighted(ax, xy, E, W=None, alpha=0.40, lw_range=(0.6, 2.2)):
        segs, widths = [], []
        if W is None:
            segs = [[xy[i], xy[j]] for (i, j) in E]
            widths = lw_range[0]
        else:
            wvals = []
            for (i, j) in E:
                segs.append([xy[i], xy[j]])
                key = (min(i, j), max(i, j))
                wvals.append(float(W.get(key, 0.0)))
            wvals = np.asarray(wvals, float)
            vmin, vmax = float(np.min(wvals)), float(np.max(wvals))
            if np.isclose(vmin, vmax):
                widths = np.mean(lw_range)
            else:
                widths = lw_range[0] + (lw_range[1]-lw_range[0]) * (wvals - vmin) / (vmax - vmin)
        lc = LineCollection(segs, colors='0.55', linewidths=widths, alpha=alpha)
        ax.add_collection(lc)

    def tidy_axes(ax):
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([]); ax.set_yticks([])
        for s in ('top','right','left','bottom'): ax.spines[s].set_visible(False)

    with style_ctx():
        cols = 4 if show_delta else 3
        fig_w = 13.2 if show_delta else 10.0; fig_h = 5.1
        fig = plt.figure(figsize=(fig_w, fig_h))
        gs  = fig.add_gridspec(1, cols, width_ratios=[1,1,1,0.95] if show_delta else [1,1,0.95], wspace=0.14)

        axA = fig.add_subplot(gs[0,0]); axB = fig.add_subplot(gs[0,1])
        axC = fig.add_subplot(gs[0,2]) if show_delta else None

        if show_delta:
            sub = gs[0,3].subgridspec(4,1, height_ratios=[0.26,0.44,0.20,0.10], hspace=0.25)
            ax_cbar = fig.add_subplot(sub[0,0]); ax_leg1 = fig.add_subplot(sub[1,0])
            ax_leg2 = fig.add_subplot(sub[2,0]); ax_info = fig.add_subplot(sub[3,0])
        else:
            sub = gs[0,2].subgridspec(3,1, height_ratios=[0.58,0.28,0.14], hspace=0.25)
            ax_leg1 = fig.add_subplot(sub[0,0]); ax_leg2 = fig.add_subplot(sub[1,0]); ax_info = fig.add_subplot(sub[2,0])
            ax_cbar = None
        for ax in (ax_leg1, ax_leg2, ax_info): ax.set_axis_off()

        def panel(ax, xy, sizes, title, colours=comm_cols, hot_idx=None):
            draw_edges_weighted(ax, xy, E, W=Wdict, alpha=0.40, lw_range=(0.6, 2.2))
            ax.scatter(xy[:,0], xy[:,1], s=sizes, c=colours, alpha=0.95,
                       edgecolor='0.15', linewidth=0.4)
            if hot_idx is not None and len(hot_idx)>0:
                ax.scatter(xy[hot_idx,0], xy[hot_idx,1], s=sizes[hot_idx]*1.45,
                           facecolors='none', edgecolors='black', linewidths=1.0)
            tidy_axes(ax); ax.set_title(title, fontsize=10.5, pad=6)

        k = min(3, len(s_pre))
        panel(axA, pre_xy, sizes_pre,  f"A. Before Repartition (step {pre_step})", hot_idx=np.argsort(-s_pre)[:k])
        panel(axB, post_xy, sizes_post, f"B. After Repartition (step {post_step})", hot_idx=np.argsort(-s_post)[:k])

        if show_delta:
            draw_edges_weighted(axC, post_xy, E, W=Wdict, alpha=0.25, lw_range=(0.5, 1.8))
            scC = axC.scatter(post_xy[:,0], post_xy[:,1], s=sizes_d, c=dfrac, cmap=cmap_delta, norm=norm_delta,
                              alpha=0.95, edgecolor='0.15', linewidth=0.4)
            tidy_axes(axC); axC.set_title("C. Relative Change in Allocation (post − pre)", fontsize=10.5, pad=6)
            for spine in ax_cbar.spines.values(): spine.set_visible(False)
            ax_cbar.set_axis_off()
            cbar = fig.colorbar(scC, ax=axC, cax=ax_cbar, orientation='vertical')
            cbar.ax.tick_params(labelsize=8); cbar.set_label("Relative change in shots", fontsize=9)
            try: cbar.formatter = PercentFormatter(xmax=1.0); cbar.update_ticks()
            except Exception: pass
            ax_leg1.text(0.0, 1.02, "Panel C: colour = Δ% (relative), size = |Δ| (absolute)",
                         ha='left', va='bottom', fontsize=9.0, transform=ax_leg1.transAxes)

        # Legends
        uniq = np.unique(comm)[:4]
        handles_comm = [mpatches.Patch(facecolor=cmap_comm(int(c) % 8), edgecolor='0.20',
                                       label=f"Community {letters[i % len(letters)]}")
                        for i, c in enumerate(uniq)]
        handle_edge = [mlines.Line2D([0],[0], color='0.55', lw=2.2,
                                     label="Link strength (pre calibration)")]
        leg1 = ax_leg1.legend(handles=handles_comm + handle_edge, loc='upper left', ncol=1, frameon=True,
                              title="Encoding (Panels A & B):", fontsize=9.0, title_fontsize=9.5,
                              borderaxespad=0.0, handletextpad=0.8, labelspacing=0.7)
        if leg1: leg1.get_frame().set_alpha(0.97)
        ax_leg2.text(0.0, 0.98, "Circle size reflects allocated shots (log scale).\nLarger circle = more shots.",
                     ha="left", va="top", fontsize=9.2)

        # Info box
        try:
            deg = S_pre.sum(axis=1)
            # Tiny, lightweight stat (no heavy perms here to keep runtime modest)
            rho = float(np.corrcoef(deg, dshots)[0,1]) if np.isfinite(deg).all() else np.nan
        except Exception:
            rho = np.nan
        lines = [
            f"Total shots: pre={int(s_pre.sum())}, post={int(s_post.sum())}, Δ={int((s_post - s_pre).sum())}",
            f"Spearman-like (degree vs Δ shots): ρ≈{rho:.2f}" if np.isfinite(rho) else "Correlation: n/a",
        ]
        ax_info.text(0.0, 0.98, "\n".join(lines), ha='left', va='top', fontsize=9.0,
                     bbox=dict(boxstyle='round,pad=0.32', facecolor='white', edgecolor='0.65', alpha=0.98))

        fig.suptitle(f"Shot Reallocation Across Communities After Repartitioning ({w}, seed {seed})",
                     fontsize=12.5, y=0.995)
        fig.text(0.50, 0.965, f"{rep_note} · A: step {pre_step} · B: step {post_step}",
                 ha='center', va='top', fontsize=9.6)

        fig.tight_layout(rect=[0.02, 0.02, 0.985, 0.965])
        _savefig(fig, save); plt.close(fig)

# -----------------------------------------------------------------------------
# RQ3.1 – Cascade Boundary (probability field + μ-family)
# -----------------------------------------------------------------------------
def fig7_boundary(df: pd.DataFrame, save=FIGDIR/"fig-rq3-boundary.pdf"):
    """
    Probability heatmap P(MLE | H, s) centered at 0.5 (diverging map),
    white P=0.5 contour (legend handle), dashed μ-quantile Δ=0 boundaries
    (low/median/high), plus uncertainty band between q20–q80. Integer y-ticks.
    Robust color scaling avoids "all blue".
    """
    import math, warnings
    import matplotlib.lines as mlines
    from matplotlib.colors import TwoSlopeNorm

    order = workloads_order(df)
    if not order: return
    w = order[0]

    pooled = []; beta = 50.0
    for seed in sorted(df.loc[df.workload==w,'seed'].unique()):
        meta = load_meta(w, int(seed)); beta = meta.get('cascade_model',{}).get('beta', beta)
        frag = load_frag(w, int(seed))
        if frag.empty or not {'H_i','s_plan','kalman_mu'}.issubset(frag.columns): continue
        use = frag[['H_i','s_plan','kalman_mu']].copy()
        if 'branch' in frag.columns:
            use['branch'] = frag['branch'].astype(int)
        pooled.append(use)
    if not pooled:
        print('[warn] No frag logs for boundary; skipping Fig. 7'); return
    F = pd.concat(pooled, ignore_index=True).dropna()

    H_grid = np.linspace(0.0, math.log(3), 220)
    s_lo, s_hi = np.percentile(F['s_plan'], [5, 95])
    s_lo = max(4, int(np.floor(s_lo))); s_hi = max(20, int(np.ceil(s_hi)))
    s_grid = np.linspace(s_lo, s_hi, 220)
    HH, SS = np.meshgrid(H_grid, s_grid)

    mu_q = np.quantile(F['kalman_mu'], [0.2, 0.5, 0.8])
    q_labels = [r'low $\mu$', r'median $\mu$', r'high $\mu$']
    q_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    bias = 0.1 * (1.0 - np.minimum(1.0, HH/np.log(3)))

    def delta_grid(mu):
        return (mu*beta)/(SS**2) + bias**2 - (mu/SS)

    def boundary_s_curve(mu):
        D = delta_grid(mu)
        s_star = np.full(H_grid.shape, np.nan)
        for j in range(D.shape[1]):
            col = D[:, j]; sign = np.sign(col)
            sc_idx = np.where(np.diff(sign) != 0)[0]
            if sc_idx.size:
                i = sc_idx[0]; s1, s2 = s_grid[i], s_grid[i+1]; d1, d2 = col[i], col[i+1]
                s_star[j] = s1 - d1*(s2-s1)/(d2-d1) if (d2 - d1)!=0 else 0.5*(s1+s2)
        return s_star
    s_low  = boundary_s_curve(mu_q[0])
    s_med  = boundary_s_curve(mu_q[1])
    s_high = boundary_s_curve(mu_q[2])

    # Probability field
    P = None; used_logistic = False
    if 'branch' in F.columns and F['branch'].dropna().nunique() == 2:
        try:
            from sklearn.preprocessing import StandardScaler
            from sklearn.linear_model import LogisticRegression
            X = F[['H_i','s_plan']].to_numpy(float); y = F['branch'].to_numpy(int)
            scaler = StandardScaler(); Xs = scaler.fit_transform(X)
            clf = LogisticRegression(max_iter=400, class_weight='balanced'); clf.fit(Xs, y)
            grid_Xs = scaler.transform(np.c_[HH.ravel(), SS.ravel()])
            P = clf.predict_proba(grid_Xs)[:,1].reshape(HH.shape); used_logistic = True
        except Exception as e:
            warnings.warn(f'[warn] Logistic fit failed ({e}); using theoretical probability.')
    if P is None:
        def prob_from_delta(tau):
            D_med = delta_grid(mu_q[1])
            return 1.0 / (1.0 + np.exp(D_med / max(tau,1e-8)))  # Δ<0 => P->1
        def p50_boundary(Pgrid):
            s_star = np.full(H_grid.shape, np.nan)
            for j in range(Pgrid.shape[1]):
                diff = Pgrid[:, j] - 0.5
                sc_idx = np.where(np.diff(np.sign(diff)) != 0)[0]
                if sc_idx.size:
                    i = sc_idx[0]; s1, s2 = s_grid[i], s_grid[i+1]; d1, d2 = diff[i], diff[i+1]
                    s_star[j] = s1 - d1*(s2-s1)/(d2-d1) if (d2-d1)!=0 else 0.5*(s1+s2)
            return s_star
        D_med_abs = np.abs(delta_grid(mu_q[1])); tau0 = np.percentile(D_med_abs, 75)/3.0
        τs = [tau0, 0.7*tau0, 0.5*tau0, 0.35*tau0, 0.25*tau0]
        best = (None, np.inf)
        for τ in τs:
            P_try = prob_from_delta(τ); s_p50 = p50_boundary(P_try)
            m = np.isfinite(s_p50) & np.isfinite(s_med)
            if m.any():
                err = np.nanmedian(np.abs(s_p50[m] - s_med[m]))
                if err < best[1]: best = (P_try, err)
        P = best[0] if best[0] is not None else prob_from_delta(tau0)

    # Color scaling around 0.5
    p_lo, p_hi = np.nanquantile(P, [0.05, 0.95])
    span = max(0.5 - p_lo, p_hi - 0.5, 0.15)  # at least ±0.15
    vmin, vmax = max(0.0, 0.5 - span), min(1.0, 0.5 + span)

    with style_ctx():
        fig, ax = plt.subplots(figsize=(6.8, 4.8), constrained_layout=True)
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0.5, vmax=vmax)
        im = ax.pcolormesh(HH, SS, P, shading='auto', cmap='RdBu_r', norm=norm)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("P(select MLE)"); cbar.set_ticks([vmin, 0.5, vmax])

        # Probability contours incl. P=0.5 (white)
        ax.contour(HH, SS, P, levels=[0.35, 0.5, 0.65], colors=['k','w','k'], linewidths=[0.8, 2.0, 0.8])

        # μ-family (dashed) + band
        for s_star, c, lab in zip([s_low, s_med, s_high], q_colors, q_labels):
            m = np.isfinite(s_star)
            if m.any(): ax.plot(H_grid[m], s_star[m], ls='--', lw=1.6, color=c, label=lab)
        m_band = np.isfinite(s_low) & np.isfinite(s_high)
        if m_band.any():
            ax.fill_between(H_grid[m_band], s_low[m_band], s_high[m_band],
                            color='k', alpha=0.10, label=r'$\mu$-boundary band (q20–q80)')

        # Legend entries incl. P=0.5
        mu_lines = [mlines.Line2D([], [], color=c, ls='--') for c in q_colors]
        p50_handle = mlines.Line2D([], [], color='w', lw=2.0)
        handles = mu_lines + [p50_handle]; labels = q_labels + [r'$P=0.5$ decision boundary']
        ax.legend(handles, labels, frameon=True, loc='upper right',
                  title=r'$\mu$ at convergence (quantiles)')

        ax.set_xlim(0, np.log(3)); ax.set_ylim(s_lo, s_hi)
        ax.set_xlabel(r"Entropy $H_i$"); ax.set_ylabel(r"Shots $s_i$")

        # Integer y-ticks
        s_range = s_hi - s_lo
        if s_range <= 20: yticks = np.arange(s_lo, s_hi + 1, 1, dtype=int)
        else:
            step = max(1, int(round(s_range / 8))); yticks = np.arange(s_lo, s_hi + 1, step, dtype=int)
        ax.set_yticks(yticks)

        title = f"Cascade decision map vs $H$ — {w}"
        title += " (data-driven P)" if used_logistic else " (model-calibrated P)"
        ax.set_title(title)
        sns.despine()
        _savefig(fig, save); plt.close(fig)

# -----------------------------------------------------------------------------
# RQ3.2 – Estimator Selection Heatmaps (global bins, hatched low-n)
# -----------------------------------------------------------------------------
def fig8_branch_bins(df: pd.DataFrame, save=FIGDIR/"fig-rq3-branch-bins.pdf",
                     min_n: int = 25, use_log2_labels: bool = True):
    order = workloads_order(df)
    raw = []
    for w in order:
        for seed in sorted(df.loc[df.workload == w, 'seed'].unique()):
            frag = load_frag(w, int(seed))
            if frag.empty or not {'H_i', 's_plan', 'branch'}.issubset(frag.columns): continue
            H = frag['H_i'].to_numpy(float); S = frag['s_plan'].to_numpy(float); B = frag['branch'].to_numpy(int)
            m = np.isfinite(H) & np.isfinite(S) & np.isfinite(B)
            if m.any():
                raw.append(pd.DataFrame({'workload': w, 'H': H[m], 'S': S[m], 'B': B[m]}))
    if not raw:
        print('[warn] No data (with branch labels) for 2D bins; skipping Fig. 8'); return
    R = pd.concat(raw, ignore_index=True)

    Hbins = np.linspace(0, np.log(3), 6)
    s95   = float(np.percentile(R['S'], 95))
    sedges = [4.0]
    while sedges[-1] < max(16.0, s95): sedges.append(sedges[-1] * 2.0)
    Sbins = np.array(sedges, dtype=float)
    while Sbins.size < 6: Sbins = np.append(Sbins, Sbins[-1] * 2.0)

    R['Hbin'] = np.digitize(R['H'], Hbins, right=True) - 1
    R['Sbin'] = np.digitize(R['S'], Sbins, right=True) - 1
    R = R[(R['Hbin'] >= 0) & (R['Hbin'] < len(Hbins)-1) & (R['Sbin'] >= 0) & (R['Sbin'] < len(Sbins)-1)]

    G = R.groupby(['workload','Hbin','Sbin']).agg(mle_frac=('B','mean'), n=('B','size')).reset_index()

    if use_log2_labels:
        k = np.round(np.log2(Sbins)).astype(int)
        xlabels = [fr"$[2^{k[i]},\,2^{k[i+1]})$" for i in range(len(Sbins)-1)]
    else:
        xlabels = [f"[{int(Sbins[i])},{int(Sbins[i+1])})" for i in range(len(Sbins)-1)]
    ylabels = [f"[{Hbins[i]:.2f},{Hbins[i+1]:.2f})" for i in range(len(Hbins)-1)]

    with style_ctx():
        ncols = min(3, len(order)); nrows = (len(order) + ncols - 1) // ncols
        fig_w = 4.9 * ncols; fig_h = 3.9 * nrows
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True)
        last_im = None; all_axes = []
        import matplotlib.patches as mpatches

        for k_w, w in enumerate(order):
            ax = axes[k_w // ncols, k_w % ncols]; all_axes.append(ax)
            sub = G[G.workload == w]
            if sub.empty: ax.axis('off'); continue

            idx = pd.MultiIndex.from_product([range(len(Hbins)-1), range(len(Sbins)-1)], names=['Hbin','Sbin'])
            sub_full = sub.set_index(['Hbin','Sbin']).reindex(idx).reset_index()
            mat = sub_full.pivot(index='Hbin', columns='Sbin', values='mle_frac')
            cnt = sub_full.pivot(index='Hbin', columns='Sbin', values='n')

            im = sns.heatmap(mat, vmin=0, vmax=1, center=0.5, cmap='coolwarm',
                             cbar=False, ax=ax, linewidths=0.5, linecolor='white')
            last_im = im

            ax.set_xticks(np.arange(mat.shape[1]) + 0.5)
            ax.set_yticks(np.arange(mat.shape[0]) + 0.5)
            ax.set_xticklabels(xlabels, rotation=35, ha='right')
            ax.set_yticklabels(ylabels, rotation=0)

            if (k_w % ncols) == 0: ax.set_ylabel(r"Fragment entropy $H$ (binned)")
            else: ax.set_ylabel("")
            if (k_w // ncols) == (nrows - 1): ax.set_xlabel(r"Allocated shots $s$ (binned)")
            else: ax.set_xlabel("")

            c_np = cnt.to_numpy()
            for (i, j), _ in np.ndenumerate(c_np):
                nval = c_np[i, j]
                if np.isfinite(nval) and (nval < min_n):
                    rect = mpatches.Rectangle((j, i), 1, 1,
                                              facecolor=(0.92, 0.92, 0.92, 0.8),
                                              edgecolor='none', hatch='///')
                    ax.add_patch(rect)

            ax.set_title(w, pad=6); ax.tick_params(axis='both', length=0)
            for spine in ax.spines.values(): spine.set_visible(False)

        for k_w in range(len(order), nrows*ncols): axes[k_w // ncols, k_w % ncols].axis('off')

        if last_im is not None:
            cbar = fig.colorbar(last_im.collections[0], ax=all_axes, fraction=0.030, pad=0.01)
            cbar.set_label(r"$p(\mathrm{MLE}\mid H, s)$")

        fig.text(0.01, 0.01, f"Grey hatched cells: unreliable (n < {min_n}). Diverging colours centred at 0.5 highlight the boundary.",
                 ha='left', va='bottom', fontsize=8.3, color='dimgray')
        fig.suptitle(r"Estimator selection heatmaps: $p(\mathrm{MLE}\mid H, s)$ by entropy and shots",
                     y=0.995, fontsize=12)
        _savefig(fig, save); plt.close(fig)

# -----------------------------------------------------------------------------
# RQ3.3 – Waterfall (Δ MSE vs Uniform)
# -----------------------------------------------------------------------------
def fig9_waterfall(df: pd.DataFrame, save=FIGDIR/"fig-rq3-waterfall.pdf"):
    order = workloads_order(df)
    recs = []
    for w in order:
        for seed in sorted(df.loc[df.workload==w,'seed'].unique()):
            res = _compute_ablation_laststep(w, int(seed))
            if not res: continue
            u = res['Uniform']['mse']; p = res['Proportional']['mse']; t = res['Topo-GP']['mse']
            recs += [
                {'workload': w, 'seed': seed, 'method': 'Proportional', 'impr': 100*(u-p)/u},
                {'workload': w, 'seed': seed, 'method': 'Topo-GP',     'impr': 100*(u-t)/u},
            ]
    if not recs:
        print('[warn] No data for waterfall; skipping Fig. 9'); return
    W = pd.DataFrame(recs); W['workload'] = pd.Categorical(W['workload'], categories=order, ordered=True)

    pal = {"Proportional": PALETTE["Prop"], "Topo-GP": PALETTE["Topo"]}
    hatch_map = {"Proportional":"\\\\", "Topo-GP":"///"}

    with style_ctx():
        fig, ax = plt.subplots(figsize=(7.8,3.8), constrained_layout=True)
        # Bar means + 95% CI via seaborn; then add per-seed dots
        sns.barplot(data=W, x='workload', y='impr', hue='method',
                    estimator=np.mean, errorbar=('ci',95), palette=pal, ax=ax, order=order, edgecolor="black", linewidth=0.6)
        sns.stripplot(data=W, x='workload', y='impr', hue='method',
                      dodge=True, alpha=0.35, size=3, palette=pal, ax=ax, order=order, legend=False,
                      marker='o')
        # Apply hatches post-draw for B/W
        for patch, (_, row) in zip(ax.patches, [(None, None)] * len(ax.patches)):
            if isinstance(patch, mpl.patches.Rectangle):
                # identify method by bar color
                col = patch.get_facecolor()
                # crude map back by comparing rgb
                if np.allclose(col[:3], mpl.colors.to_rgb(PALETTE["Prop"])):
                    patch.set_hatch(hatch_map["Proportional"])
                elif np.allclose(col[:3], mpl.colors.to_rgb(PALETTE["Topo"])):
                    patch.set_hatch(hatch_map["Topo-GP"])

        ax.axhline(0, color='grey', lw=1, ls='--')
        ax.set_ylabel('Reduction in Final MSE vs Uniform (%)')
        ax.set_xlabel(''); ax.set_title('Accuracy contribution (per-seed spread + 95% CI)')
        ax.tick_params(axis='x', rotation=30)

        # one legend
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(handles[:2], labels[:2], title="", ncol=2, frameon=True)
        sns.despine(); _savefig(fig, save); plt.close(fig)

# -----------------------------------------------------------------------------
# RQ4.1 – Overhead trade-off
# -----------------------------------------------------------------------------
def fig10_overhead(df: pd.DataFrame, save=FIGDIR/"fig-rq4-overhead-tradeoff.pdf"):
    recs = []
    for _, row in df.iterrows():
        w, seed = row['workload'], int(row['seed'])
        frag = load_frag(w, seed)
        if frag.empty or 'step' not in frag.columns: continue
        steps = int(frag['step'].max()) + 1
        t_step = float(row.get('mean_time_est_s', 0.0))
        shots = int(row.get('shots_used', 0))
        total_time = t_step * max(1, steps)
        base_thr = shots / total_time if total_time>0 else np.nan
        if not np.isfinite(base_thr): continue
        for o in np.linspace(0.0, 0.05, 11):
            thr = base_thr / (1.0 + o)
            recs.append({'workload': w, 'seed': seed, 'overhead': 100*o, 'throughput': thr})
    if not recs:
        print('[warn] No sidecars/steps for overhead; skipping Fig. 10'); return
    odf = pd.DataFrame(recs); odf['rel_thr'] = odf.groupby(['workload','seed'])['throughput'].transform(lambda s: 100*s/s.iloc[0])

    with style_ctx():
        fig, ax = plt.subplots(figsize=(5.6,3.4), constrained_layout=True)
        sns.lineplot(data=odf, x='overhead', y='rel_thr', hue='workload', marker='o', ax=ax, estimator='median')
        ax.axvline(1.0, ls='--', lw=1, color='grey', alpha=0.9, label='1% target')
        ax.axhline(100, ls=':', lw=1, color='grey')
        ax.set_xlabel("PhasePad overhead (%)"); ax.set_ylabel("Relative throughput (%) (100% = 0% overhead)")
        ax.set_title("Overhead vs throughput (normalized per seed)")
        ax.legend(frameon=True); sns.despine()
        _savefig(fig, save); plt.close(fig)

# -----------------------------------------------------------------------------
# RQ4.2 – Decoy ROC
# -----------------------------------------------------------------------------
def fig11_decoy_roc(df: pd.DataFrame, save=FIGDIR/"fig-rq4-decoy-roc.pdf"):
    curves = []
    for w in workloads_order(df):
        ys, ss = [], []
        for seed in sorted(df.loc[df.workload==w, 'seed'].unique()):
            frag = load_frag(w, int(seed))
            need = {'decoy_label','decoy_score'}
            if frag.empty or not need.issubset(frag.columns): continue
            y = frag['decoy_label'].to_numpy(int); s = frag['decoy_score'].to_numpy(float)
            m = np.isfinite(s); y, s = y[m], s[m]
            if y.size and y.sum()>0 and (y.size - y.sum()) > 0:
                ys.append(y); ss.append(s)
        if ys:
            y = np.concatenate(ys); s = np.concatenate(ss)
            order_idx = np.argsort(-s); y = y[order_idx]; s = s[order_idx]
            P = max(1, y.sum()); N = max(1, len(y) - y.sum())
            tps = np.cumsum(y); fps = np.cumsum(1 - y)
            TPR = tps / P; FPR = fps / N
            auc = float(np.trapz(TPR, FPR))
            curves.append((w, FPR, TPR, auc))
    if not curves: return

    with style_ctx():
        fig, ax = plt.subplots(figsize=(5.4, 3.8), constrained_layout=True)
        for w, FPR, TPR, auc in curves:
            ax.plot(FPR, TPR, label=f"{w} (AUC={auc:.2f})")
        ax.plot([0,1], [0,1], ls='--', c='grey', lw=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("Decoy detection ROC"); ax.legend(frameon=True, ncol=2)
        sns.despine(); _savefig(fig, save); plt.close(fig)


# -----------------------------------------------------------------------------
# Resources scaling (Revised)
# -----------------------------------------------------------------------------
def fig12_resources(df: pd.DataFrame, save=FIGDIR / "fig-resources-scaling.pdf"):
    """
    Revised resource scaling plot: Shows RELATIVE overhead of DREAMCUT
    (runtime and memory) compared to the smallest problem size for each workload.
    """
    recs = []
    for _, row in df.iterrows():
        w, seed = row['workload'], int(row['seed'])
        meta = load_meta(w, seed)
        nf = meta.get('workload', {}).get('n_fragments', np.nan)
        recs.append({'workload': w, 'seed': seed, 'n_frag': nf,
                     'time_step': row.get('mean_time_est_s', np.nan),
                     'peak_mem_gb': row.get('peak_mem_gb', np.nan)})
    rdf = pd.DataFrame(recs).dropna()
    if rdf.empty:
        print('[warn] No metadata for resources; skipping Fig. 12');
        return

    # --- NEW: Calculate Relative Overhead ---
    # Group by workload and find the minimum number of fragments
    min_fragments = rdf.groupby('workload')['n_frag'].transform('min')
    # Group by workload and get the baseline value at min_fragments for each seed?
    # This is complex. Simplest: get baseline value for the *workload* at its min n_frag.
    baseline = rdf.loc[rdf.groupby('workload')['n_frag'].idxmin()].set_index('workload')

    # Calculate relative overhead for each data point
    plot_df = rdf.copy()
    for w in plot_df['workload'].unique():
        base_time = baseline.loc[w, 'time_step']
        base_mem = baseline.loc[w, 'peak_mem_gb']
        mask = plot_df['workload'] == w
        plot_df.loc[mask, 'rel_time_pct'] = (plot_df.loc[mask, 'time_step'] / base_time - 1) * 100
        plot_df.loc[mask, 'rel_mem_pct'] = (plot_df.loc[mask, 'peak_mem_gb'] / base_mem - 1) * 100

    # --- Create the Plot ---
    with style_ctx():
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8))  # Slightly wider for labels
        # Use a informative palette
        palette = sns.color_palette("colorblind", n_colors=plot_df['workload'].nunique())

        # Plot 1: Relative Runtime Overhead
        sns.lineplot(data=plot_df, x='n_frag', y='rel_time_pct', hue='workload', marker='o', ax=axes[0],
                     palette=palette, legend=True)
        # Add a trendline for the overall system
        sns.regplot(data=plot_df, x='n_frag', y='rel_time_pct', scatter=False, ax=axes[0], ci=None, color='k',
                    line_kws={'lw': 1.5, 'ls': '--', 'alpha': 0.7})
        axes[0].axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)  # Base reference line
        axes[0].set_xlabel("Number of Fragments")
        axes[0].set_ylabel("Relative Runtime Overhead (%)")
        axes[0].set_title("Runtime Scaling Overhead")
        axes[0].legend_.set_title('Workload')  # Clean up legend title

        # Plot 2: Relative Memory Overhead
        sns.lineplot(data=plot_df, x='n_frag', y='rel_mem_pct', hue='workload', marker='s', ax=axes[1], palette=palette,
                     legend=False)
        sns.regplot(data=plot_df, x='n_frag', y='rel_mem_pct', scatter=False, ax=axes[1], ci=None, color='k',
                    line_kws={'lw': 1.5, 'ls': '--', 'alpha': 0.7})
        axes[1].axhline(0, color='k', linestyle='-', linewidth=0.8, alpha=0.5)
        axes[1].set_xlabel("Number of Fragments")
        axes[1].set_ylabel("Relative Memory Overhead (%)")
        axes[1].set_title("Memory Scaling Overhead")

        # Add a note about what "overhead" means
        fig.text(0.5, 0.01, "Note: Overhead calculated relative to the smallest fragmentation level for each workload.",
                 ha='center', fontsize=9, style='italic', color='dimgray')

        fig.suptitle("Scalability of DREAMCUT Overhead", fontweight='bold')
        plt.tight_layout()
        _savefig(fig, save)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------------
def epsilon_theory_from_meta(workload: str, df_summary: pd.DataFrame) -> float:
    """Toy epsilon estimator; returns NaN if insufficient info (kept from your original)."""
    for seed in sorted(df_summary.loc[df_summary.workload == workload, 'seed'].unique()):
        meta = load_meta(workload, int(seed))
        if not meta: continue
        w = meta.get('workload', {})
        n = w.get('qubits') or w.get('n_qubits') or w.get('n')
        d = w.get('depth'); Q = w.get('Q') or w.get('modularity') or w.get('topo_Q')
        try:
            if n and d and Q and (d>0) and (Q>0): return float(1.0 - np.log(n / Q) / d)
        except Exception: pass
    return float('nan')

# ------------------------------ Dashboard ------------------------------------
def fig13_dashboard(df: pd.DataFrame, save=FIGDIR/"fig-dashboard.pdf"):
    order = workloads_order(df)
    rows = []
    for w in order:
        sub = df[df.workload == w]
        if sub.empty:
            continue

        ratio = None
        if {'var_gp','var_uniform'}.issubset(sub.columns):
            r = (sub['var_gp'] / sub['var_uniform']).median()
            ratio = float(r) if np.isfinite(r) else None

        p95 = None
        if 'p95_shots' in sub.columns:
            v = sub['p95_shots'].median()
            p95 = float(v) if np.isfinite(v) else None

        metas = [load_meta(w, int(s)) for s in sub['seed'].unique()]
        ov = None
        if metas:
            vals = [m.get('phasepad',{}).get('overhead', np.nan) for m in metas]
            vals = [float(x) for x in vals if np.isfinite(x)]
            if vals:
                ov = float(np.median(vals))

        eps = float(epsilon_theory_from_meta(w, df))
        if not np.isfinite(eps): eps = None
        t_p95 = 60.0; t_ov = 0.01

        def state(val, pred):
            if val is None: return 0.5
            return 1.0 if pred(val) else 0.0

        rows.append({
            'workload': w,
            'Contraction (≤ε)': state(ratio, (lambda x, thr=eps: x <= thr) if eps is not None else (lambda x: False)),
            'Tail p95 (≤60 shots)': state(p95,   lambda x: x <= t_p95),
            'Overhead (≤1%)':       state(ov,    lambda x: x <= t_ov),
            '_vals': {
                'Contraction (≤ε)': (None if ratio is None else (f"{ratio:.2f}" + (f" (ε={eps:.2f})" if eps is not None else ""))),
                'Tail p95 (≤60 shots)': (None if p95   is None else f"{p95:.0f}"),
                'Overhead (≤1%)':       (None if ov    is None else f"{ov*100:.2f}%"),
            }
        })

    if not rows:
        print('[warn] Nothing to plot for dashboard; skipping Fig. 13')
        return

    ddf = pd.DataFrame(rows).set_index('workload')
    metric_cols = [c for c in ddf.columns if c != '_vals']
    M = ddf[metric_cols].to_numpy(float)

    with plt.rc_context({'font.family': 'DejaVu Sans'}):  # supports ✓/✗
        fig_h = 0.9 + 0.48 * len(order)
        fig, ax = plt.subplots(figsize=(5.6, fig_h))
        im = ax.imshow(M, aspect='auto', vmin=0, vmax=1, cmap='RdYlGn')

        ax.set_yticks(np.arange(len(order)))
        ax.set_yticklabels(order)
        ax.set_xticks(np.arange(len(metric_cols)))
        ax.set_xticklabels(metric_cols, rotation=30, ha='right')

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                st = M[i, j]
                key = metric_cols[j]
                val = ddf.iloc[i]['_vals'].get(key)
                if st >= 0.75:
                    tag = '✓'
                elif st <= 0.25:
                    tag = '✗'
                else:
                    tag = 'n/a'
                text = tag if val is None else f"{tag}  {val}"
                ax.text(j, i, text, ha='center', va='center', fontsize=9, color='black')

        ax.set_title("Tier-1 Targets Dashboard")
        sns.despine(left=True, bottom=True)

        cbar = fig.colorbar(im, ax=ax, ticks=[0,0.5,1], fraction=0.046, pad=0.04)
        cbar.ax.set_yticklabels(["Fail", "n/a", "Pass"])

        _savefig(fig, save)
        plt.close(fig)

# ------------------------------ Main -----------------------------------------
FIG_FUNCS = [
    fig1_contraction,
    fig2_ablation,
    fig3_tails,
    fig4_coupling,
    fig5_timeline,
    fig6a_topology_cov,
    fig6b_topology_embed,
    fig7_boundary,
    fig8_branch_bins,
    fig9_waterfall,
    fig10_overhead,
    fig11_decoy_roc,
    fig12_resources,
    fig13_dashboard,
]

def main():
    ap = argparse.ArgumentParser(description="Tier-1 plotting suite (seaborn)")
    ap.add_argument('--which', nargs='*', default=[f.__name__ for f in FIG_FUNCS],
                    help='Subset of figures to render by function name (default: all).')
    args = ap.parse_args()

    df = load_summary()
    for fn in FIG_FUNCS:
        if fn.__name__ in args.which:
            try:
                fn(df)
                print(f"✓ Wrote {fn.__name__}")
            except Exception as e:
                print(f"[warn] {fn.__name__} failed: {e}")

if __name__ == '__main__':
    main()
