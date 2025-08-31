#!/usr/bin/env python3
# tier1_sensitivity.py — Tier-1 Sensitivity Study:
# Curve A: ABORT% vs Δ_budget percentile (P70–P99 of baseline "none")
# Curve B: AUC vs β (sweep 0.05–0.12), using Δ̂_t as a detector

from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Dict, Tuple, Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# put these lines at the very top of tier1_sensitivity.py, before importing pyplot
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

# ---------------- I/O helpers ----------------
def read_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            pass
    return {}

def load_inputs(root: Path) -> tuple[pd.DataFrame, dict, dict]:
    root = Path(root)  # ensure Path, even if a string was passed
    iv_path = root / "intervals.csv"
    st_path = root / "settings.json"
    th_path = root / "thresholds_used.json"
    if not th_path.exists():                  # fallback for Tier-1 runs that wrote thresholds.json
        th_path = root / "thresholds.json"

    if not iv_path.exists():
        raise SystemExit(f"[error] missing intervals.csv at {iv_path}")
    df = pd.read_csv(iv_path)
    if df.empty:
        raise SystemExit("[error] intervals.csv is empty")
    settings = read_json(st_path)
    thr_used = read_json(th_path)
    return df, settings, thr_used

# ---------------- Policy (hysteresis) ----------------
def enforce_gap(budget: float, kill: float, gap_frac: float = 0.05, gap_abs: float = 0.005) -> Tuple[float, float]:
    gap = max(gap_abs, gap_frac * max(budget, 1e-9))
    if not np.isfinite(kill) or (kill < budget + gap):
        kill = budget + gap
    return float(budget), float(kill)

def apply_hysteresis_decisions(deltas: Iterable[float], budget: float, kill: float,
                               kill_strikes: int = 2, warn_cooldown: int = 1) -> Tuple[int, int, int]:
    """
    Recompute decisions over a sequence of delta_est values (time-ordered).
    Returns: (#CONTINUE, #WARNING, #ABORT)
    """
    strikes = 0
    cooldown = 0
    c = w = a = 0
    for d in deltas:
        kill_now = (d >= kill)
        warn_now = (not kill_now) and (d > budget)
        if kill_now:
            strikes += 1
            cooldown = warn_cooldown
            if strikes >= kill_strikes:
                a += 1
            else:
                w += 1
        elif warn_now:
            w += 1
            cooldown = max(cooldown - 1, 0)
            strikes = max(0, strikes - 1)
        else:
            c += 1
            if cooldown > 0:
                cooldown -= 1
                strikes = max(0, strikes - 1)
            else:
                strikes = 0
    return c, w, a

# ---------------- AUC (Mann–Whitney) ----------------
def auc_pos_vs_neg(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random positive score > negative score (ties → 0.5)."""
    x = np.concatenate([pos, neg])
    y = np.concatenate([np.ones_like(pos, dtype=int), np.zeros_like(neg, dtype=int)])
    order = np.argsort(x, kind="mergesort")  # stable for ties
    ranks = np.empty_like(order, dtype=float)
    # average ranks for ties
    i = 0
    r = 1
    while i < len(x):
        j = i + 1
        while j < len(x) and x[order[j]] == x[order[i]]:
            j += 1
        avg_rank = 0.5 * (r + (r + (j - i) - 1))
        ranks[order[i:j]] = avg_rank
        r += (j - i)
        i = j
    R_pos = ranks[:len(pos)].sum()  # ranks of positives (since first len(pos) were pos in x/y concat)
    # careful: our concat put pos first → ranks slices must match positions
    # Recompute positions explicitly to be safe:
    pos_idx = np.where(y == 1)[0]
    R_pos = ranks[pos_idx].sum()
    m = len(pos); n = len(neg)
    if m == 0 or n == 0:
        return float("nan")
    U = R_pos - m * (m + 1) / 2.0
    return float(U / (m * n))

# ---------------- Curve A: ABORT% vs budget percentile ----------------
def curve_abort_vs_budget(df: pd.DataFrame, beta: float, gap_frac: float, gap_abs: float,
                          kill_strikes: int, warn_cooldown: int,
                          percentiles = tuple(range(70, 100))) -> pd.DataFrame:
    df = df.copy()
    # Ensure types & expected columns
    for col in ("attack","interval"):
        df[col] = df[col].astype(str if col=="attack" else int, copy=False)
    if "n" in df.columns:
        df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)

    # Baseline (none) distribution for Δ_budget candidates
    base = df[df["attack"].eq("none")]["delta_est"].dropna().values
    if base.size == 0:
        raise SystemExit("[error] No baseline ('none') intervals found for budget percentiles.")
    pvals = np.percentile(base, percentiles).astype(float)

    rows = []
    # Recompute policy decisions per (attack, n, seed) sequence using new thresholds
    group_keys = ["attack", "n", "seed"] if "seed" in df.columns else ["attack", "n"]
    for p, b in zip(percentiles, pvals):
        budget = float(b)
        # choose kill via policy gap
        budget, kill = enforce_gap(budget, float("nan"), gap_frac, gap_abs)

        for atk, sub in df.groupby("attack"):
            # Apply hysteresis on each timeline, count totals
            A = C = W = 0
            for _, sub2 in sub.sort_values(["n","seed","interval"]).groupby(group_keys):
                deltas = sub2["delta_est"].values
                c, w, a = apply_hysteresis_decisions(deltas, budget, kill, kill_strikes, warn_cooldown)
                C += c; W += w; A += a
            total = A + W + C
            abort_pct = 100.0 * (A / total) if total > 0 else np.nan
            rows.append({"percentile": p, "budget": budget, "kill": kill,
                         "attack": atk, "ABORT%": abort_pct, "intervals": total})
    return pd.DataFrame(rows)

# ---------------- Curve B: AUC vs β ----------------
def curve_auc_vs_beta(df: pd.DataFrame, beta_old: float,
                      beta_grid = np.round(np.arange(0.05, 0.1201, 0.005), 3)) -> pd.DataFrame:
    df = df.copy()
    df["attack"] = df["attack"].astype(str)
    if "n" in df.columns:
        df["n"] = pd.to_numeric(df["n"], errors="coerce").fillna(0).astype(int)

    # We only need Δ̂; shifting β is linear: Δ̂_new = Δ̂_old - β_old + β_new
    base = df[df["attack"].eq("none")]["delta_est"].dropna().values
    rl   = df[df["attack"].eq("rl")]["delta_est"].dropna().values
    tim  = df[df["attack"].eq("timing")]["delta_est"].dropna().values

    rows = []
    for bnew in beta_grid:
        shift = float(bnew) - float(beta_old)
        base_s = base + shift
        rl_s   = rl   + shift
        tim_s  = tim  + shift
        # AUC(attack vs none)
        auc_rl  = auc_pos_vs_neg(rl_s,  base_s) if rl_s.size  and base_s.size else np.nan
        auc_tim = auc_pos_vs_neg(tim_s, base_s) if tim_s.size and base_s.size else np.nan
        auc_mean = np.nanmean([auc_rl, auc_tim])
        rows.append({"beta": float(bnew), "AUC_rl": auc_rl, "AUC_timing": auc_tim, "AUC_mean": auc_mean})
    return pd.DataFrame(rows)

# ---------------- Plotting ----------------
def set_style():
    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 300, "figure.autolayout": True,
        "axes.grid": True, "grid.alpha": 0.25
    })

def plot_curve_a(dfA: pd.DataFrame, outdir: Path):
    if dfA.empty: return
    set_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    attacks = ["none","rl","timing"]
    for atk in attacks:
        sub = dfA[dfA["attack"].eq(atk)]
        if sub.empty: continue
        ax.plot(sub["percentile"], sub["ABORT%"], marker="o", label=atk)
    ax.set_title("Sensitivity A — ABORT% vs budget percentile (baseline 'none')")
    ax.set_xlabel("Δ_budget percentile (from 'none')")
    ax.set_ylabel("ABORT intervals (%)")
    ax.legend(title="Attack", frameon=False)
    fig.savefig(outdir / "sensitivity_abort_vs_budget.pdf")
    fig.savefig(outdir / "sensitivity_abort_vs_budget.png")
    plt.close(fig)

def plot_curve_b(dfB: pd.DataFrame, outdir: Path):
    if dfB.empty: return
    set_style()
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(dfB["beta"], dfB["AUC_rl"], marker="o", label="rl vs none")
    ax.plot(dfB["beta"], dfB["AUC_timing"], marker="o", label="timing vs none")
    ax.plot(dfB["beta"], dfB["AUC_mean"], linestyle="--", label="mean")
    ax.set_title("Sensitivity B — AUC vs β (Δ̂_t as detector)")
    ax.set_xlabel("β")
    ax.set_ylabel("AUC")
    ax.set_ylim(0.5, 1.0)
    ax.legend(title="", frameon=False)
    fig.savefig(outdir / "sensitivity_auc_vs_beta.pdf")
    fig.savefig(outdir / "sensitivity_auc_vs_beta.png")
    plt.close(fig)

# ---------------- CLI ----------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Tier-1 Sensitivity (ABORT% vs budget percentile; AUC vs β)")
    ap.add_argument("--root", type=Path, required=True,
                    help="Tier-1 results folder containing intervals.csv + settings.json (+ thresholds_used.json)")
    ap.add_argument("--out", type=Path, default=Path("plots_tier1_sensitivity"))
    ap.add_argument("--pmin", type=int, default=70)
    ap.add_argument("--pmax", type=int, default=99)
    ap.add_argument("--pstep", type=int, default=1)
    ap.add_argument("--bmin", type=float, default=0.05)
    ap.add_argument("--bmax", type=float, default=0.12)
    ap.add_argument("--bstep", type=float, default=0.005)
    return ap.parse_args()

def main():
    args = parse_args()
    outdir = args.out
    outdir.mkdir(parents=True, exist_ok=True)

    df, settings, thr_used = load_inputs(args.root)

    beta_old = float(settings.get("monitor_beta", 0.10))
    ks = int(settings.get("kill_strikes", 2))
    wc = int(settings.get("warn_cooldown", 1))
    gap_frac = float(thr_used.get("thr_gap_frac", 0.05))
    gap_abs  = float(thr_used.get("thr_gap_abs", 0.005))

    # Curve A
    percentiles = tuple(range(int(args.pmin), int(args.pmax) + 1, int(args.pstep)))
    dfA = curve_abort_vs_budget(df, beta_old, gap_frac, gap_abs, ks, wc, percentiles=percentiles)
    dfA.to_csv(outdir / "sensitivity_abort_vs_budget.csv", index=False)
    plot_curve_a(dfA, outdir)

    # Curve B
    beta_grid = np.round(np.arange(args.bmin, args.bmax + 1e-9, args.bstep), 3)
    dfB = curve_auc_vs_beta(df, beta_old, beta_grid=beta_grid)
    dfB.to_csv(outdir / "sensitivity_auc_vs_beta.csv", index=False)
    plot_curve_b(dfB, outdir)

    print(f"[done] Wrote sensitivity plots + CSVs → {outdir.resolve()}")

if __name__ == "__main__":
    main()
