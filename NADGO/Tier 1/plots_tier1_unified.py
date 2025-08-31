#!/usr/bin/env python3
# plots_tier1_unified.py — Consolidated Tier-I figures (n=4,8,16)

from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Headless backend
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

# ---- constants ----
ATTACK_ORDER = ["none", "rl", "timing"]
N_ORDER = [4, 8, 16]
PALETTE = {"none": "#7a7a7a", "rl": "#3366cc", "timing": "#ff7f0e"}

# ---- helpers ----
def set_style():
    sns.set_theme(context="talk", style="whitegrid", font_scale=1.0)
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 300,
        "figure.autolayout": True,
        "axes.titleweight": "bold",
        "axes.grid": True,
        "grid.alpha": 0.25,
    })

def read_json(path: Path) -> dict:
    if path and path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}

def load_one_root(root: Path) -> dict:
    root = Path(root)
    out = {"root": root}

    def _rd(name):
        p = root / name
        if not p.exists():
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except Exception:
            return pd.DataFrame()

    out["intervals"]  = _rd("intervals.csv")
    out["episodes"]   = _rd("episodes.csv")
    out["segments"]   = _rd("segments.csv")
    out["thresholds"] = read_json(root / "thresholds_used.json")
    out["settings"]   = read_json(root / "settings.json")

    # tidy types (leave 'n' coercion to concat_by_key for a single normalisation path)
    for key in ["intervals", "episodes", "segments"]:
        df = out[key]
        if df.empty:
            continue
        if "attack" in df.columns:
            df["attack"] = df["attack"].astype(str)
        if "decision" in df.columns:
            df["decision"] = df["decision"].astype(str)
        if "aborted" in df.columns and df["aborted"].dtype != bool:
            try:
                df["aborted"] = df["aborted"].astype(int).astype(bool)
            except Exception:
                df["aborted"] = df["aborted"].astype(str).str.lower().isin(["true","1","yes"])
        if "delta_est" in df.columns:
            df["delta_est"] = pd.to_numeric(df["delta_est"], errors="coerce")

    return out

def concat_by_key(all_runs: list[dict], key: str) -> pd.DataFrame:
    frames = []
    for run in all_runs:
        df = run.get(key, pd.DataFrame()).copy()
        if df.empty:
            continue
        df["_root"] = str(run["root"])
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)

    # Normalise attack
    if "attack" in df.columns:
        df["attack"] = pd.Categorical(df["attack"].astype(str), ATTACK_ORDER, ordered=True)

    # Robust normalisation for n: handles "16", 16.0, Int64, etc.
    if "n" in df.columns:
        df["n"] = pd.to_numeric(df["n"], errors="coerce")   # -> float/NaN
        df["n"] = df["n"].round().astype("Int64")          # 16.0 → 16 (Int64), preserves NA

    return df

def get_knobs(all_runs: list[dict]) -> dict:
    thr, st = {}, {}
    for run in all_runs:
        if run.get("thresholds"):
            thr = run["thresholds"]
        if run.get("settings"):
            st = run["settings"]
    return {
        "beta":     st.get("monitor_beta", None),
        "budget":   thr.get("budget", None),
        "kill":     thr.get("kill", None),
        "alpha":    st.get("monitor_alpha", None),
        "qmin":     st.get("monitor_qmin", None),
        "eps_sync": st.get("monitor_eps_sync", None),
    }

def savefig(fig: plt.Figure, outdir: Path, name: str):
    outdir.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(outdir / f"{name}.pdf")
    fig.savefig(outdir / f"{name}.png", transparent=True)
    plt.close(fig)

def sanity_report(df_int: pd.DataFrame, df_ep: pd.DataFrame, df_seg: pd.DataFrame):
    def _present_ns(df):
        if df.empty or "n" not in df.columns:
            return []
        return sorted([int(x) for x in pd.unique(df["n"].dropna())])
    print("[sanity] intervals n present:", _present_ns(df_int))
    print("[sanity] episodes  n present:", _present_ns(df_ep))
    print("[sanity] segments  n present:", _present_ns(df_seg))

# ---- metrics (AUC) ----
def auc_binary(pos: np.ndarray, neg: np.ndarray) -> float:
    x = np.asarray(pos, float)
    y = np.asarray(neg, float)
    if x.size == 0 or y.size == 0:
        return np.nan
    ranks = pd.Series(np.concatenate([x, y])).rank(method="average").to_numpy()
    rx = ranks[: len(x)].sum()
    n1, n0 = float(len(x)), float(len(y))
    return float((rx - n1 * (n1 + 1) / 2.0) / (n1 * n0))

# ---- FIG 1: Policy overview (heatmap + ep abort + policy mix) ----
def fig1_policy_overview(df_int: pd.DataFrame, df_ep: pd.DataFrame, outdir: Path):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3))

    # (a) Interval ABORT heatmap (n × attack)
    ax = axes[0]
    if not df_int.empty and {"n", "attack", "decision"}.issubset(df_int.columns):
        g = (
            df_int.assign(is_abort=lambda d: d["decision"].eq("ABORT"))
                 .groupby(["n", "attack"])["is_abort"].mean()
                 .mul(100)
                 .reset_index()
        )
        g = g[g["n"].isin(N_ORDER)]
        piv = (g.pivot(index="n", columns="attack", values="is_abort")
                 .reindex(index=N_ORDER)
                 .reindex(columns=ATTACK_ORDER))

        # Determine color scale
        with np.errstate(all='ignore'):
            vmax = np.nanmax(piv.to_numpy())
        vmax = float(vmax) if np.isfinite(vmax) else 20.0
        vmax = max(10.0, vmax)

        # Track rows that are entirely missing
        row_is_all_nan = piv.isna().all(axis=1)
        piv_filled = piv.fillna(0.0)

        sns.heatmap(piv_filled, annot=True, fmt=".2f", vmin=0, vmax=vmax,
                    cbar_kws={"label": "ABORT (%)"}, ax=ax)

        # Overlay "no data" for fully-missing rows
        for ridx, nval in enumerate(piv.index):
            if row_is_all_nan.loc[nval]:
                ax.text(1.5, ridx + 0.5, "no data", ha="center", va="center",
                        fontsize=9, color="red")

        ax.set_title("Interval ABORT heatmap (n × attack)")
        ax.set_xlabel("Attack")
        ax.set_ylabel("n (qubits)")
    else:
        ax.text(0.5, 0.5, "No intervals.csv", ha="center", va="center")
        ax.axis("off")

    # (b) Episode abort rate by attack
    ax = axes[1]
    if not df_ep.empty and {"attack", "aborted"}.issubset(df_ep.columns):
        e = df_ep.groupby("attack", as_index=False)["aborted"].mean()
        e["pct"] = e["aborted"] * 100
        sns.barplot(
            data=e, x="attack", y="pct",
            order=ATTACK_ORDER, palette=PALETTE, ax=ax, edgecolor="black"
        )
        for p in ax.patches:
            h = p.get_height()
            if np.isfinite(h):
                ax.annotate(f"{h:.1f}%",
                            (p.get_x() + p.get_width() / 2, h),
                            ha="center", va="bottom", fontsize=9,
                            xytext=(0, 2), textcoords="offset points")
        ax.set_title("Episode abort rate")
        ax.set_ylabel("Aborted episodes (%)")
        ax.set_xlabel("Attack")
    else:
        ax.text(0.5, 0.5, "No episodes.csv", ha="center", va="center")
        ax.axis("off")

    # (c) Policy action mix (stacked %)
    ax = axes[2]
    if not df_int.empty and {"attack", "decision"}.issubset(df_int.columns):
        g = df_int.groupby(["attack", "decision"]).size().reset_index(name="count")
        g["pct"] = g.groupby("attack")["count"].transform(lambda s: s / s.sum() * 100.0)
        orderD = ["CONTINUE", "WARNING", "ABORT"]
        g["decision"] = pd.Categorical(g["decision"], orderD, True)
        base = np.zeros(len(ATTACK_ORDER))
        for dec in orderD:
            vals = (g[g["decision"].eq(dec)]
                      .set_index("attack")
                      .reindex(ATTACK_ORDER)["pct"]
                      .fillna(0).to_numpy())
            ax.bar(ATTACK_ORDER, vals, bottom=base, label=dec, edgecolor="black")
            base += vals
        ax.set_title("Policy mix (interval)")
        ax.set_ylabel("Share of intervals (%)")
        ax.set_xlabel("Attack")
        ax.legend(title="Decision", frameon=False)
    else:
        ax.text(0.5, 0.5, "No intervals.csv", ha="center", va="center")
        ax.axis("off")

    savefig(fig, outdir, "fig1_policy_overview")

# ---- FIG 2: Δ̂ histograms in one row (none, rl, timing) ----
def fig2_delta_histograms(df_int: pd.DataFrame, knobs: dict, outdir: Path, inset_per_n: bool = False):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), sharey=True)
    beta, budget, kill = knobs.get("beta"), knobs.get("budget"), knobs.get("kill")

    # lock x-limits for comparability
    x_lo, x_hi = None, None
    if not df_int.empty and "delta_est" in df_int.columns:
        vals = df_int["delta_est"].dropna().to_numpy()
        if vals.size:
            lo = np.nanmin(vals)
            hi = np.nanmax(vals)
            cand = [v for v in [lo, hi, beta, budget, kill] if isinstance(v, (int, float))]
            x_lo = min(cand) - 0.01 if cand else lo - 0.01
            x_hi = max(cand) + 0.02 if cand else hi + 0.02

    for i, atk in enumerate(["none", "rl", "timing"]):
        ax = axes[i]

        if df_int.empty or "attack" not in df_int.columns:
            ax.text(0.5, 0.5, f"no data: {atk}", ha="center", va="center")
            ax.axis("off")
            continue

        d = (df_int[df_int["attack"].astype(str).eq(atk)]
                .get("delta_est", pd.Series(dtype=float))
                .dropna())
        if d.empty:
            ax.text(0.5, 0.5, f"no data: {atk}", ha="center", va="center")
            ax.axis("off")
            continue

        sns.histplot(d, bins=40, stat="density", kde=True,
                     color=PALETTE.get(atk, "tab:blue"), ax=ax)

        if isinstance(beta, (int, float)):
            ax.axvline(beta, ls="-.", lw=1.5, color="#666", label=f"β={beta:.3f}")
        if isinstance(budget, (int, float)):
            ax.axvline(budget, ls="--", lw=1.8, color="#333", label=f"budget={budget:.3f}")
        if isinstance(kill, (int, float)):
            ax.axvline(kill, ls=":", lw=2.0, color="#000", label=f"kill={kill:.3f}")

        if x_lo is not None and x_hi is not None:
            ax.set_xlim(x_lo, x_hi)

        ax.set_title(f"Δ̂ distribution — {atk}")
        ax.set_xlabel("Δ̂ (bits)")
        if i == 0:
            ax.set_ylabel("density")
        if i == 2:
            ax.legend(frameon=False, loc="upper right", title="Thresholds",
                      fontsize=9, title_fontsize=9)

        # Optional inset: per-n KDE overlays
        if inset_per_n and "n" in df_int.columns:
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            axins = inset_axes(ax, width="45%", height="40%", loc="upper left", borderpad=1)
            for n in N_ORDER:
                y = df_int[(df_int["attack"].astype(str).eq(atk)) & (df_int["n"] == n)]["delta_est"].dropna()
                if y.empty:
                    continue
                sns.kdeplot(y, ax=axins, lw=1.3, label=f"n={n}")
            axins.set_xticks([]); axins.set_yticks([])
            axins.set_title("per-n", fontsize=9)
            axins.legend(fontsize=7, frameon=False, ncol=2)

    savefig(fig, outdir, "fig2_delta_histograms")

# ---- FIG 3: Latency & Power with %Δ annotations ----
def fig3_latency_power(df_seg: pd.DataFrame, outdir: Path):
    set_style()

    if df_seg.empty or not {"n", "attack"}.issubset(df_seg.columns):
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No segments.csv → skipping cost fig", ha="center", va="center")
        ax.axis("off")
        savefig(fig, outdir, "fig3_latency_power")
        return

    def _agg(col):
        g = df_seg.groupby(["n", "attack"])[col].agg(["mean", "std", "count"]).reset_index()
        g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
        g["lo"] = g["mean"] - 1.96 * g["sem"]
        g["hi"] = g["mean"] + 1.96 * g["sem"]
        return g

    gl = _agg("latency") if "latency" in df_seg.columns else pd.DataFrame()
    gp = _agg("power")   if "power"   in df_seg.columns else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3), sharex=True)
    panels = [
        ("Latency by n (95% CI)", gl, "Latency (sojourn)", axes[0]),
        ("Power by n (95% CI)", gp, "Power (arb.)", axes[1]),
    ]

    for title, g, ylabel, ax in panels:
        if g.empty:
            ax.text(0.5, 0.5, "missing", ha="center", va="center")
            ax.axis("off")
            continue

        for atk in ATTACK_ORDER:
            sub = g[g["attack"].astype(str).eq(atk)].sort_values("n")
            if sub.empty:
                continue
            ax.errorbar(
                sub["n"], sub["mean"],
                yerr=[sub["mean"] - sub["lo"], sub["hi"] - sub["mean"]],
                fmt="o-", capsize=3, label=atk, color=PALETTE.get(atk)
            )

        ax.set_title(title)
        ax.set_xlabel("n (qubits)")
        ax.set_ylabel(ylabel)
        present_ns = set(int(x) for x in pd.unique(g["n"].dropna()))
        xt = [n for n in N_ORDER if n in present_ns]
        ax.set_xticks(xt)
        if xt:
            ax.set_xlim(min(xt) - 0.5, max(xt) + 0.5)
        ax.grid(alpha=0.25)

        try:
            base = g[g["attack"].astype(str).eq("none")][["n", "mean"]].set_index("n")["mean"]
            for n0, m0 in base.items():
                ax.axhline(m0, ls=":", lw=1, alpha=0.25, color="#777")
            for atk in ["rl", "timing"]:
                sub = g[g["attack"].astype(str).eq(atk)]
                for _, r in sub.iterrows():
                    n_val = int(r["n"])
                    if n_val not in base.index:
                        continue
                    pct = 100.0 * (r["mean"] / base.loc[n_val] - 1.0)
                    ax.annotate(
                        f"{pct:+.1f}%",
                        (n_val, r["mean"]),
                        textcoords="offset points",
                        xytext=(0, 8),
                        ha="center",
                        fontsize=9,
                        color=PALETTE.get(atk),
                    )
        except Exception:
            pass

        ax.legend(title="Attack", frameon=False)

    savefig(fig, outdir, "fig3_latency_power")

# ---- FIG 4: Sensitivity sweeps (budget percentile vs ABORT, beta vs AUC) ----
def fig4_sensitivity(df_int: pd.DataFrame, knobs: dict, outdir: Path):
    set_style()

    if df_int.empty or "delta_est" not in df_int.columns or "attack" not in df_int.columns:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No intervals.csv → skipping sensitivity", ha="center", va="center")
        ax.axis("off")
        savefig(fig, outdir, "fig4_sensitivity")
        return

    none_vals = df_int[df_int["attack"].astype(str).eq("none")]["delta_est"].dropna().to_numpy()
    A = pd.DataFrame()
    if none_vals.size:
        pct_grid = list(range(70, 100))
        rowsA = []
        for P in pct_grid:
            thr = np.percentile(none_vals, P)
            for atk in ATTACK_ORDER:
                d = df_int[df_int["attack"].astype(str).eq(atk)]["delta_est"].dropna().to_numpy()
                if d.size == 0:
                    continue
                abort_pct = float(np.mean(d >= thr)) * 100.0
                rowsA.append({"P": P, "attack": atk, "ABORTpct": abort_pct})
        A = pd.DataFrame(rowsA)

    beta_grid = np.round(np.linspace(0.05, 0.12, 8), 3)
    rowsB = []
    for b in beta_grid:
        for atk in ["rl", "timing"]:
            pos = df_int[df_int["attack"].astype(str).eq(atk)]["delta_est"].dropna().to_numpy() - b
            neg = df_int[df_int["attack"].astype(str).eq("none")]["delta_est"].dropna().to_numpy() - b
            if pos.size and neg.size:
                rowsB.append({"beta": b, "attack": atk, "AUC": auc_binary(pos, neg)})
    B = pd.DataFrame(rowsB)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.3))

    ax = axes[0]
    if not A.empty:
        for atk in ATTACK_ORDER:
            sub = A[A["attack"].astype(str).eq(atk)]
            if sub.empty:
                continue
            ax.plot(sub["P"], sub["ABORTpct"], marker="o", label=atk, color=PALETTE.get(atk))
        ax.set_title("Sensitivity A: ABORT% vs budget percentile (from 'none')")
        ax.set_xlabel("Δ_budget percentile (baseline 'none')")
        ax.set_ylabel("ABORT intervals (%)")
        ax.legend(title="Attack", frameon=False)
    else:
        ax.text(0.5, 0.5, "insufficient baseline for sweep", ha="center", va="center")
        ax.axis("off")

    ax = axes[1]
    if not B.empty:
        for atk in ["rl", "timing"]:
            sub = B[B["attack"].astype(str).eq(atk)]
            ax.plot(sub["beta"], sub["AUC"], marker="s", label=f"{atk} vs none", color=PALETTE.get(atk))
        ax.set_title("Sensitivity B: AUC vs β (Δ̂ as detector)")
        ax.set_xlabel("β")
        ax.set_ylabel("AUC")
        ax.text(0.02, 0.06, "AUC ≈ constant → shift-invariant", transform=ax.transAxes, fontsize=10, alpha=.8)
        ax.set_ylim(0.5, 1.0)
        ax.legend(frameon=False)
    else:
        ax.text(0.5, 0.5, "insufficient data for AUC sweep", ha="center", va="center")
        ax.axis("off")

    savefig(fig, outdir, "fig4_sensitivity")

# ---- CLI ----
def main():
    ap = argparse.ArgumentParser(description="Unified Tier-I figures (n=4,8,16)")
    ap.add_argument(
        "--roots", nargs="+", required=True,
        help="One or more results roots (each containing intervals.csv [+ episodes.csv, segments.csv optional])"
    )
    ap.add_argument(
        "--out", type=Path, default=Path("tier1_plots_unified"),
        help="Output folder for consolidated figures"
    )
    ap.add_argument(
        "--inset_per_n", action="store_true",
        help="Add per-n KDE insets on Δ̂ histograms"
    )
    args = ap.parse_args()

    runs = [load_one_root(Path(r)) for r in args.roots]
    df_int = concat_by_key(runs, "intervals")
    df_ep  = concat_by_key(runs, "episodes")
    df_seg = concat_by_key(runs, "segments")
    knobs  = get_knobs(runs)

    # enforce consistent ordering / filtering
    if "attack" in df_int.columns:
        df_int = df_int[df_int["attack"].isin(ATTACK_ORDER)]
    if "attack" in df_ep.columns:
        df_ep = df_ep[df_ep["attack"].isin(ATTACK_ORDER)]
    if "n" in df_int.columns:
        df_int = df_int[df_int["n"].isin(N_ORDER)]
    if "n" in df_ep.columns:
        df_ep = df_ep[df_ep["n"].isin(N_ORDER)]
    if "n" in df_seg.columns:
        df_seg = df_seg[df_seg["n"].isin(N_ORDER)]

    # quick visibility into what actually made it through
    sanity_report(df_int, df_ep, df_seg)

    outdir = Path(args.out)
    fig1_policy_overview(df_int, df_ep, outdir)
    fig2_delta_histograms(df_int, knobs, outdir, inset_per_n=args.inset_per_n)
    fig3_latency_power(df_seg, outdir)
    fig4_sensitivity(df_int, knobs, outdir)

    print(f"[done] wrote: {outdir.resolve()}/fig1_policy_overview.(png,pdf)")
    print(f"[done] wrote: {outdir.resolve()}/fig2_delta_histograms.(png,pdf)")
    print(f"[done] wrote: {outdir.resolve()}/fig3_latency_power.(png,pdf)")
    print(f"[done] wrote: {outdir.resolve()}/fig4_sensitivity.(png,pdf)")

if __name__ == "__main__":
    main()
