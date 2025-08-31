#!/usr/bin/env python3
# plots_tier1.py — Thesis-grade plots for NADGO Tier-1 results (with debug I/O)

from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
import numpy as np, pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Headless
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

# ---------------- Data & IO utils ----------------
def _latest_results_dir(default_hint: Path = Path(".")) -> Path | None:
    """Find the newest directory matching 'nadgo_results*'."""
    candidates = [p for p in default_hint.glob("nadgo_results*") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)

def coerce_episode_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    if 'aborted' in df.columns and df['aborted'].dtype != bool:
        try: df['aborted'] = df['aborted'].astype(int).astype(bool)
        except Exception:
            df['aborted'] = df['aborted'].astype(str).str.lower().isin(["true","1","yes"])
    for col in ["attack"]:
        if col in df.columns: df[col] = df[col].astype(str)
    if "n" in df.columns: df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    return df

def coerce_interval_types(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty: return df
    for col in ["attack", "decision"]:
        if col in df.columns: df[col] = df[col].astype(str)
    if "n" in df.columns: df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)
    if "delta_est" in df.columns: df["delta_est"] = pd.to_numeric(df["delta_est"], errors="coerce")
    return df

def read_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            print(f"[warn] Could not read JSON {path}: {e}")
    return {}

def _load_csv(path: Path, label: str, required: bool = True) -> pd.DataFrame:
    """Load a CSV with helpful debug prints."""
    print(f"[load] {label}: {path.resolve()}")
    if not path.exists():
        msg = f"[error] Missing {label} at: {path.resolve()}"
        if required:
            print(msg)
            # Show dir listing to help spot typos
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
        if df.empty:
            print(f"[warn] {label} loaded but is EMPTY. Check the source run/folder.")
            root = path.parent
            print(f"[hint] Contents of {root.resolve()}:")
            for p in list(root.iterdir())[:30]:
                print("   -", p.name)
        return df
    except Exception as e:
        print(f"[error] Failed reading {label}: {e}")
        raise SystemExit(2)

# ---------------- Plot helpers ----------------
def plot_abort_by_attack_intervals(df_int: pd.DataFrame, outdir: Path):
    if df_int.empty: return
    set_style()
    df = df_int.assign(is_abort=lambda d: d["decision"].eq("ABORT")) \
               .groupby("attack", as_index=False)["is_abort"].mean()
    df["abort_pct"] = df["is_abort"] * 100
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(data=df, x="attack", y="abort_pct", ax=ax, edgecolor="black")
    ax.set_title("Tier-1: ABORT rate by attack (interval level)")
    ax.set_ylabel("ABORT intervals (%)")
    ax.set_xlabel("Attack")
    annotate_bars(ax)
    sns.despine()
    save_fig(fig, outdir, "abort_rate_by_attack_intervals")

def plot_abort_by_attack_and_n_intervals(df_int: pd.DataFrame, outdir: Path):
    if df_int.empty or "n" not in df_int.columns: return
    set_style()
    df = df_int.assign(is_abort=lambda d: d["decision"].eq("ABORT")) \
               .groupby(["n","attack"], as_index=False)["is_abort"].mean()
    df["abort_pct"] = df["is_abort"] * 100
    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.barplot(data=df, x="n", y="abort_pct", hue="attack", ax=ax, edgecolor="black")
    ax.set_title("Tier-1: ABORT rate by attack and n (interval level)")
    ax.set_ylabel("ABORT intervals (%)")
    ax.set_xlabel("Number of qubits (n)")
    for c in ax.containers:
        try: ax.bar_label(c, fmt="%.1f%%", fontsize=9, padding=2)
        except Exception: pass
    ax.legend(title="Attack", frameon=False)
    sns.despine()
    save_fig(fig, outdir, "abort_rate_by_attack_n_intervals")

def plot_delta_histograms(df_int: pd.DataFrame, thresholds: dict, beta: float | None, outdir: Path):
    if df_int.empty or "delta_est" not in df_int.columns: return
    set_style()
    attacks = list(df_int["attack"].astype(str).unique())
    for atk in attacks:
        d = df_int[df_int["attack"].eq(atk)]["delta_est"].dropna()
        if d.empty: continue
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.histplot(d, bins=40, stat="density", kde=True, ax=ax)
        ax.set_title(f"Δ̂ distribution — {atk}")
        ax.set_xlabel("Δ̂ (bits)")
        ax.set_ylabel("density")

        # Lines: beta, budget, kill
        b = thresholds.get("budget")
        k = thresholds.get("kill")
        if isinstance(beta, (int, float)):
            ax.axvline(float(beta), ls="-.", lw=1.5, color="gray", alpha=0.9, label=f"β={beta:.3f}")
        if isinstance(b, (int, float)):
            ax.axvline(float(b), ls="--", lw=1.8, label=f"budget={b:.3f}")
        if isinstance(k, (int, float)):
            ax.axvline(float(k), ls=":",  lw=2.0, label=f"kill={k:.3f}")

        ax.legend(frameon=False, title="Thresholds", loc="upper right")
        ax.text(0.02, 0.95, f"n={len(d):,} intervals", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, alpha=0.9)
        sns.despine()
        save_fig(fig, outdir, f"delta_hist_{atk}")

def plot_policy_mix_intervals(df_int: pd.DataFrame, outdir: Path):
    if df_int.empty: return
    set_style()
    g = df_int.groupby(["attack","decision"]).size().reset_index(name="count")
    if g.empty: return
    g["pct"] = g.groupby("attack")["count"].transform(lambda s: s / s.sum() * 100.0)
    order = ["CONTINUE", "WARNING", "ABORT"]
    g["decision"] = pd.Categorical(g["decision"], categories=order, ordered=True)
    g = g.sort_values(["attack","decision"])
    piv = g.pivot(index="attack", columns="decision", values="pct").fillna(0)
    piv = piv.reindex(columns=order, fill_value=0)
    fig, ax = plt.subplots(figsize=(9.5, 5.5))
    bottom = np.zeros(len(piv))
    for col in order:
        vals = piv[col].values
        ax.bar(piv.index, vals, bottom=bottom, label=col, edgecolor="black")
        bottom += vals
    ax.set_title("Policy action mix (interval level, stacked %)")
    ax.set_ylabel("Share of intervals (%)")
    ax.set_xlabel("Attack")
    ax.legend(title="Decision", frameon=False)
    sns.despine()
    save_fig(fig, outdir, "policy_action_mix_intervals")

def plot_abort_heatmap_intervals(df_int: pd.DataFrame, outdir: Path):
    if df_int.empty or "n" not in df_int.columns: return
    set_style()
    g = (df_int.assign(is_abort=lambda d: d["decision"].eq("ABORT"))
               .groupby(["n","attack"], as_index=False)["is_abort"].mean())
    if g.empty: return
    piv = g.pivot(index="n", columns="attack", values="is_abort") * 100.0
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    sns.heatmap(piv.round(1), annot=True, fmt=".1f", linewidths=.5,
                cbar_kws={"label": "ABORT (%)"}, ax=ax)
    ax.set_title("ABORT rate heatmap by n × attack (interval level)")
    ax.set_xlabel("Attack"); ax.set_ylabel("Number of qubits (n)")
    sns.despine(left=True, bottom=True)
    save_fig(fig, outdir, "abort_rate_heatmap_intervals")

def plot_episode_level(df_ep: pd.DataFrame, outdir: Path):
    if df_ep.empty: return
    set_style()
    # Aborted episodes by attack
    df = df_ep.groupby("attack", as_index=False)["aborted"].mean()
    df["aborted_pct"] = df["aborted"] * 100
    fig, ax = plt.subplots(figsize=(8.5, 5))
    sns.barplot(data=df, x="attack", y="aborted_pct", ax=ax, edgecolor="black")
    ax.set_title("Aborted episodes by attack")
    ax.set_ylabel("Aborted episodes (%)"); ax.set_xlabel("Attack")
    annotate_bars(ax); sns.despine()
    save_fig(fig, outdir, "ep_abort_rate_by_attack")

    # By attack and n
    if "n" in df_ep.columns:
        d2 = df_ep.groupby(["n","attack"], as_index=False)["aborted"].mean()
        d2["aborted_pct"] = d2["aborted"] * 100
        fig, ax = plt.subplots(figsize=(10, 5.5))
        sns.barplot(data=d2, x="n", y="aborted_pct", hue="attack", ax=ax, edgecolor="black")
        ax.set_title("Aborted episodes by attack and n")
        ax.set_ylabel("Aborted episodes (%)"); ax.set_xlabel("Number of qubits (n)")
        for c in ax.containers:
            try: ax.bar_label(c, fmt="%.1f%%", fontsize=9, padding=2)
            except Exception: pass
        ax.legend(title="Attack", frameon=False); sns.despine()
        save_fig(fig, outdir, "ep_abort_rate_by_attack_n")

    # Mean fidelity vs n (with CI)
    if "mean_fidelity" in df_ep.columns:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.lineplot(data=df_ep, x="n", y="mean_fidelity", hue="attack",
                     errorbar=("ci", 95), marker="o", ax=ax)
        ax.set_title("Mean fidelity vs n (95% CI)")
        ax.set_xlabel("Number of qubits (n)"); ax.set_ylabel("Mean fidelity")
        ax.legend(title="Attack", frameon=False); sns.despine()
        save_fig(fig, outdir, "fidelity_vs_n_ci")

    # Total leak violin
    if "total_leak" in df_ep.columns:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        sns.violinplot(data=df_ep, x="attack", y="total_leak", inner=None, cut=0, ax=ax)
        sns.boxplot(data=df_ep, x="attack", y="total_leak", width=0.2, showcaps=True,
                    boxprops={"zorder": 2}, ax=ax)
        ax.set_title("Total leakage distribution by attack")
        ax.set_xlabel("Attack"); ax.set_ylabel("Total leak (episode)")
        sns.despine(); save_fig(fig, outdir, "total_leak_violin")

    # Latency vs power scatter (episode means)
    if {"mean_latency", "mean_power"}.issubset(df_ep.columns):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        sns.scatterplot(data=df_ep, x="mean_latency", y="mean_power", hue="attack", style="attack", ax=ax)
        ax.set_title("Latency vs Power (episode means)")
        ax.set_xlabel("Mean latency"); ax.set_ylabel("Mean power")
        ax.legend(title="Attack", frameon=False); sns.despine()
        save_fig(fig, outdir, "latency_vs_power")

def plot_latency_power_ci_segments(df_seg: pd.DataFrame, outdir: Path):
    """95% CI whiskers for latency & power per (n, attack) using segments.csv."""
    if df_seg.empty or not {"latency", "power", "n", "attack"}.issubset(df_seg.columns):
        return
    set_style()
    g = df_seg.groupby(["n","attack"]).agg(lat_mean=("latency","mean"),
                                           lat_std=("latency","std"),
                                           cnt=("latency","size"),
                                           pow_mean=("power","mean"),
                                           pow_std=("power","std")).reset_index()
    # 95% CI for the mean: mean ± 1.96 * (std / sqrt(n))
    g["lat_sem"] = g["lat_std"] / np.sqrt(g["cnt"].clip(lower=1))
    g["pow_sem"] = g["pow_std"] / np.sqrt(g["cnt"].clip(lower=1))
    g["lat_lo"]  = g["lat_mean"] - 1.96 * g["lat_sem"]
    g["lat_hi"]  = g["lat_mean"] + 1.96 * g["lat_sem"]
    g["pow_lo"]  = g["pow_mean"] - 1.96 * g["pow_sem"]
    g["pow_hi"]  = g["pow_mean"] + 1.96 * g["pow_sem"]

    # Latency
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for atk, sub in g.groupby("attack"):
        ax.errorbar(sub["n"], sub["lat_mean"], yerr=[sub["lat_mean"]-sub["lat_lo"], sub["lat_hi"]-sub["lat_mean"]],
                    fmt="o-", capsize=3, label=str(atk))
    ax.set_title("Latency by n with 95% CI (segments)")
    ax.set_xlabel("Number of qubits (n)"); ax.set_ylabel("Latency (sojourn)")
    ax.legend(title="Attack", frameon=False); sns.despine()
    save_fig(fig, outdir, "latency_by_n_ci_segments")

    # Power
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for atk, sub in g.groupby("attack"):
        ax.errorbar(sub["n"], sub["pow_mean"], yerr=[sub["pow_mean"]-sub["pow_lo"], sub["pow_hi"]-sub["pow_mean"]],
                    fmt="o-", capsize=3, label=str(atk))
    ax.set_title("Power by n with 95% CI (segments)")
    ax.set_xlabel("Number of qubits (n)"); ax.set_ylabel("Power (arbitrary units)")
    ax.legend(title="Attack", frameon=False); sns.despine()
    save_fig(fig, outdir, "power_by_n_ci_segments")

# ---------------- Metrics summary ----------------
def write_metrics_summary(df_int: pd.DataFrame, df_ep: pd.DataFrame, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    if not df_int.empty:
        for atk, grp in df_int.groupby("attack"):
            rows.append({
                "level": "interval", "attack": atk,
                "ABORT%": 100.0 * grp["decision"].eq("ABORT").mean(),
                "WARNING%": 100.0 * grp["decision"].eq("WARNING").mean(),
            })
    if not df_ep.empty:
        for atk, grp in df_ep.groupby("attack"):
            rows.append({
                "level": "episode", "attack": atk,
                "EPabort%": 100.0 * grp["aborted"].mean(),
            })
    if rows:
        pd.DataFrame(rows).to_csv(outdir / "metrics_summary.csv", index=False)
        print(f"[ok] Wrote metrics → {(outdir / 'metrics_summary.csv').resolve()}")

# ---------------- CLI ----------------
def main():
    ap = argparse.ArgumentParser(description="Make plots for NADGO Tier-1 results (with debug)")
    ap.add_argument("--root", type=Path, default=None,
                    help="Folder containing episodes.csv, segments.csv, intervals.csv, thresholds_used.json")
    ap.add_argument("--episodes", type=Path, default=None, help="Override episodes.csv")
    ap.add_argument("--segments", type=Path, default=None, help="Override segments.csv")
    ap.add_argument("--intervals", type=Path, default=None, help="Override intervals.csv")
    ap.add_argument("--out", type=Path, default=Path("plots_tier1"))
    args = ap.parse_args()

    # Resolve root (auto-pick newest nadgo_results* if --root is not given or missing)
    root = args.root
    if root is None or not root.exists():
        auto = _latest_results_dir(Path("."))
        if auto:
            print(f"[auto] Using newest results dir: {auto.resolve()}")
            root = auto
        else:
            # Fall back to legacy default if present
            legacy = Path("nadgo_results")
            if legacy.exists():
                print(f"[auto] Using legacy dir: {legacy.resolve()}")
                root = legacy
            else:
                print("[error] Could not find results dir. Pass --root <path>.")
                raise SystemExit(2)

    # Resolve file paths
    ep_path  = (args.episodes or (root / "episodes.csv")).resolve()
    sg_path  = (args.segments or (root / "segments.csv")).resolve()
    iv_path  = (args.intervals or (root / "intervals.csv")).resolve()
    thr_path = (root / "thresholds_used.json").resolve()
    set_path = (root / "settings.json").resolve()

    print(f"[info] Using root: {root.resolve()}")
    print(f"[info] Output dir : {args.out.resolve()}")

    # Load with debug
    df_ep  = _load_csv(ep_path, "episodes.csv")
    df_seg = _load_csv(sg_path, "segments.csv", required=False)   # optional for some plots
    df_int = _load_csv(iv_path, "intervals.csv")

    df_ep  = coerce_episode_types(df_ep)
    df_int = coerce_interval_types(df_int)

    thresholds = read_json(thr_path)
    settings   = read_json(set_path)

    if thresholds:
        print(f"[ok] thresholds_used.json: {thr_path} (keys: {sorted(thresholds.keys())})")
    else:
        print(f"[warn] thresholds_used.json missing or unreadable at {thr_path}")

    if settings:
        print(f"[ok] settings.json: {set_path} (keys: {sorted(settings.keys())})")
    else:
        print(f"[warn] settings.json missing or unreadable at {set_path}")

    # β + checksum consistency checks
    beta = settings.get("monitor_beta", None)
    thr_checksum = thresholds.get("qref_checksum", None)
    set_checksum = settings.get("qref_checksum", None)
    if thr_checksum and set_checksum and thr_checksum != set_checksum:
        print(f"[WARN] q_ref checksum mismatch: thresholds_used={thr_checksum} vs settings={set_checksum}")
    elif thr_checksum and set_checksum:
        print(f"[ok] q_ref checksum match: {thr_checksum}")

    # Interval-level plots
    plot_abort_by_attack_intervals(df_int, args.out)
    plot_abort_by_attack_and_n_intervals(df_int, args.out)
    plot_delta_histograms(df_int, thresholds, beta, args.out)
    plot_policy_mix_intervals(df_int, args.out)
    plot_abort_heatmap_intervals(df_int, args.out)

    # Episode-level & trade-off plots
    plot_episode_level(df_ep, args.out)
    plot_latency_power_ci_segments(df_seg, args.out)

    # Metrics CSV
    write_metrics_summary(df_int, df_ep, args.out)

    print(f"[done] Saved plots + metrics to: {args.out.resolve()}")

if __name__ == "__main__":
    main()
