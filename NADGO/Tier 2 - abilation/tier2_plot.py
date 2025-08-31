#!/usr/bin/env python3
# plots_tier2.py — Tier-2 ablation figures only

from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Headless rendering
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

# ---------------- Styling ----------------
def set_style():
    sns.set_theme(context="talk", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "figure.autolayout": True,
        "axes.titleweight": "bold",
        "axes.labelweight": "regular",
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

def save_fig(fig: plt.Figure, outdir: Path, name: str, tight: bool = True):
    outdir.mkdir(parents=True, exist_ok=True)
    if tight:
        try: fig.tight_layout()
        except Exception: pass
    fig.savefig(outdir / f"{name}.png")
    fig.savefig(outdir / f"{name}.pdf")
    plt.close(fig)

def annotate_bars(ax: plt.Axes, suffix: str = "%", fmt: str = "{:.1f}"):
    for p in ax.patches:
        h = p.get_height()
        if np.isfinite(h):
            ax.annotate(fmt.format(h) + suffix,
                        (p.get_x() + p.get_width() / 2.0, h),
                        ha='center', va='bottom', fontsize=9,
                        xytext=(0, 2), textcoords="offset points")

def add_headroom(ax: plt.Axes, frac: float = 0.08, min_top: float = 1.0, cap: float | None = None):
    """Add headroom above tallest bar so labels don’t collide."""
    heights = [p.get_height() for p in ax.patches if np.isfinite(p.get_height())]
    top = max(heights) if heights else 0.0
    if top <= 0:
        return
    new_top = max(min_top, top * (1.0 + frac))
    if cap is not None:
        new_top = min(new_top, cap)
    ax.set_ylim(0, new_top)

# ---------------- Data & IO utils ----------------
def _load_csv(path: Path, label: str, required: bool = True) -> pd.DataFrame:
    print(f"[load] {label}: {path.resolve()}")
    if not path.exists():
        msg = f"[error] Missing {label} at: {path.resolve()}"
        if required:
            print(msg)
            root = path.parent
            if root.exists():
                print(f"[hint] Contents of {root.resolve()}:")
                for p in list(root.iterdir())[:30]:
                    print("   -", p.name)
            raise SystemExit(2)
        else:
            print(f"[warn] {msg} (optional)")
            return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        print(f"[ok] {label}: {len(df):,} rows")
        return df
    except Exception as e:
        print(f"[error] Failed reading {label}: {e}")
        raise SystemExit(2)

def ensure_pipeline_column(df: pd.DataFrame, fallback_name: str = "unknown") -> pd.DataFrame:
    if df.empty: return df
    if "pipeline" not in df.columns:
        df = df.copy()
        df["pipeline"] = fallback_name
    df["pipeline"] = df["pipeline"].astype(str)
    return df

def coerce_common_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for col in ("attack", "decision"):
        if col in df.columns:
            df[col] = df[col].astype(str)
    if "n" in df.columns:
        df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    if "latency" in df.columns:
        df["latency"] = pd.to_numeric(df["latency"], errors="coerce")
    if "power" in df.columns:
        df["power"] = pd.to_numeric(df["power"], errors="coerce")
    return df

def canonical_pipeline_order(pipes: list[str]) -> list[str]:
    desired = ["vendor","pad-only","jitter-only","full"]
    have = list(dict.fromkeys(pipes))
    # keep desired order for ones that exist, then append any others
    ordered = [p for p in desired if p in have] + [p for p in have if p not in desired]
    return ordered

# ---------------- Ablation plots ----------------
def plot_abort_by_pipeline_intervals(df_int: pd.DataFrame, outdir: Path):
    """Interval-level ABORT% grouped by pipeline."""
    need = {"pipeline","decision"}
    if df_int.empty or not need.issubset(df_int.columns):
        print("[skip] plot_abort_by_pipeline_intervals: required columns missing.")
        return
    set_style()
    d = (df_int.assign(is_abort=lambda x: x["decision"].eq("ABORT"))
               .groupby("pipeline", as_index=False)["is_abort"].mean())
    d["ABORT%"] = d["is_abort"] * 100.0
    order = canonical_pipeline_order(d["pipeline"].tolist())
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    sns.barplot(data=d, x="pipeline", y="ABORT%", order=order, ax=ax, edgecolor="black")
    ax.set_title("Tier-2 (emulation, ablation): ABORT rate by pipeline (interval level)")
    ax.set_xlabel("Pipeline"); ax.set_ylabel("ABORT intervals (%)")
    annotate_bars(ax, fmt="{:.1f}")
    add_headroom(ax, frac=0.10, min_top=1.0)
    sns.despine()
    save_fig(fig, outdir, "ablation_abort_by_pipeline")

def plot_abort_by_pipeline_attack_intervals(df_int: pd.DataFrame, outdir: Path):
    """Interval-level ABORT% grouped by pipeline × attack (if attack exists)."""
    need = {"pipeline","decision","attack"}
    if df_int.empty or not need.issubset(df_int.columns):
        print("[skip] plot_abort_by_pipeline_attack_intervals: required columns missing.")
        return
    set_style()
    d = (df_int.assign(is_abort=lambda x: x["decision"].eq("ABORT"))
               .groupby(["pipeline","attack"], as_index=False)["is_abort"].mean())
    d["ABORT%"] = d["is_abort"] * 100.0
    order = canonical_pipeline_order(d["pipeline"].tolist())
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    sns.barplot(data=d, x="pipeline", y="ABORT%", hue="attack",
                order=order, ax=ax, edgecolor="black")
    ax.set_title("Tier-2 (emulation, ablation): ABORT rate by pipeline × attack (interval level)")
    ax.set_xlabel("Pipeline"); ax.set_ylabel("ABORT intervals (%)")
    for c in ax.containers:
        try: ax.bar_label(c, fmt="%.1f%%", fontsize=9, padding=2)
        except Exception: pass
    add_headroom(ax, frac=0.10, min_top=1.0)
    ax.legend(title="Attack", frameon=False)
    sns.despine()
    save_fig(fig, outdir, "ablation_abort_by_pipeline_attack")

def plot_latency_power_ci_by_pipeline(df_seg: pd.DataFrame, outdir: Path):
    """95% CI for latency & power vs n, separated by pipeline (uses segments.csv)."""
    need = {"latency","power","n","pipeline"}
    if df_seg.empty or not need.issubset(df_seg.columns):
        print("[skip] plot_latency_power_ci_by_pipeline: required columns missing.")
        return
    set_style()
    g = (df_seg.groupby(["pipeline","n"])
             .agg(lat_mean=("latency","mean"), lat_std=("latency","std"), lat_n=("latency","size"),
                  pow_mean=("power","mean"),   pow_std=("power","std"),   pow_n=("power","size"))
             .reset_index())
    # 95% CI for the mean
    g["lat_sem"] = g["lat_std"] / np.sqrt(g["lat_n"].clip(lower=1))
    g["pow_sem"] = g["pow_std"] / np.sqrt(g["pow_n"].clip(lower=1))
    g["lat_lo"]  = g["lat_mean"] - 1.96 * g["lat_sem"]
    g["lat_hi"]  = g["lat_mean"] + 1.96 * g["lat_sem"]
    g["pow_lo"]  = g["pow_mean"] - 1.96 * g["pow_sem"]
    g["pow_hi"]  = g["pow_mean"] + 1.96 * g["pow_sem"]

    order = canonical_pipeline_order(sorted(g["pipeline"].unique().tolist()))

    # Latency
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for pl in order:
        sub = g[g["pipeline"].eq(pl)].sort_values("n")
        if sub.empty: continue
        ax.errorbar(sub["n"], sub["lat_mean"],
                    yerr=[sub["lat_mean"]-sub["lat_lo"], sub["lat_hi"]-sub["lat_mean"]],
                    fmt="o-", capsize=3, label=str(pl))
    ax.set_title("Tier-2 (emulation, ablation): Latency by n with 95% CI (segments)")
    ax.set_xlabel("Number of qubits (n)"); ax.set_ylabel("Latency (sojourn)")
    ax.legend(title="Pipeline", frameon=False)
    sns.despine()
    save_fig(fig, outdir, "ablation_latency_by_n_ci_segments")

    # Power
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    for pl in order:
        sub = g[g["pipeline"].eq(pl)].sort_values("n")
        if sub.empty: continue
        ax.errorbar(sub["n"], sub["pow_mean"],
                    yerr=[sub["pow_mean"]-sub["pow_lo"], sub["pow_hi"]-sub["pow_mean"]],
                    fmt="o-", capsize=3, label=str(pl))
    ax.set_title("Tier-2 (emulation, ablation): Power by n with 95% CI (segments)")
    ax.set_xlabel("Number of qubits (n)"); ax.set_ylabel("Power (arbitrary units)")
    ax.legend(title="Pipeline", frameon=False)
    sns.despine()
    save_fig(fig, outdir, "ablation_power_by_n_ci_segments")

# ---------------- Metrics summary (ablation) ----------------
def write_ablation_summary(df_int: pd.DataFrame, df_seg: pd.DataFrame, outdir: Path):
    rows = []
    if not df_int.empty and {"pipeline","decision"}.issubset(df_int.columns):
        g = (df_int.assign(is_abort=df_int["decision"].eq("ABORT"))
                  .groupby("pipeline")["is_abort"].mean().mul(100.0))
        for pl, v in g.items():
            rows.append({"metric": "ABORT% (interval)", "pipeline": pl, "value": float(v)})

    if not df_seg.empty and {"pipeline","n","latency","power"}.issubset(df_seg.columns):
        g2 = df_seg.groupby(["pipeline","n"]).agg(lat_mean=("latency","mean"),
                                                  pow_mean=("power","mean")).reset_index()
        for _, r in g2.iterrows():
            rows.append({"metric": "Latency mean (segments)", "pipeline": r["pipeline"], "n": int(r["n"]), "value": float(r["lat_mean"])})
            rows.append({"metric": "Power mean (segments)",   "pipeline": r["pipeline"], "n": int(r["n"]), "value": float(r["pow_mean"])})

    if rows:
        out = Path(outdir) / "ablation_summary.csv"
        pd.DataFrame(rows).to_csv(out, index=False)
        print(f"[ok] Wrote ablation summary → {out.resolve()}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Ablation-only plots for NADGO Tier-2 emulation")
    ap.add_argument("--root", type=Path, required=True,
                    help="Folder with combined episodes.csv, segments.csv, intervals.csv (must include 'pipeline' column)")
    ap.add_argument("--out", type=Path, default=Path("plots_tier2_ablation"))
    args = ap.parse_args()

    root = args.root
    if not root.exists():
        print(f"[error] --root does not exist: {root}")
        raise SystemExit(2)

    print(f"[info] Using root: {root.resolve()}")
    print(f"[info] Output dir : {args.out.resolve()}")

    ep_path  = (root / "episodes.csv").resolve()   # not strictly needed, but ok if present
    sg_path  = (root / "segments.csv").resolve()
    iv_path  = (root / "intervals.csv").resolve()

    # Load
    df_ep  = _load_csv(ep_path, "episodes.csv", required=False)
    df_seg = _load_csv(sg_path, "segments.csv", required=True)
    df_int = _load_csv(iv_path, "intervals.csv", required=True)

    # Ensure pipeline & basic types
    df_ep  = ensure_pipeline_column(df_ep,  fallback_name="unknown")
    df_seg = ensure_pipeline_column(df_seg, fallback_name="unknown")
    df_int = ensure_pipeline_column(df_int, fallback_name="unknown")

    df_ep  = coerce_common_types(df_ep)
    df_seg = coerce_common_types(df_seg)
    df_int = coerce_common_types(df_int)

    # Ablation figures
    plot_abort_by_pipeline_intervals(df_int, args.out)
    plot_abort_by_pipeline_attack_intervals(df_int, args.out)   # optional, if 'attack' exists
    plot_latency_power_ci_by_pipeline(df_seg, args.out)

    # Write a small CSV summary
    write_ablation_summary(df_int, df_seg, args.out)

    print(f"[done] Saved ablation plots + summary to: {args.out.resolve()}")

if __name__ == "__main__":
    main()
