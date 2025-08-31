#!/usr/bin/env python3
# qa_sweep.py — Policy grid with MANUAL thresholds (and optional locked q_ref)
#
# Usage examples (from "Tier 2" folder):
#   py qa_sweep.py
#   py qa_sweep.py --qref .\nadgo_calib\q_ref.json
#   py qa_sweep.py --ks 2,3 --wc 0,1,2 --steps 300 --seeds 0:10 --outprefix nadgo_grid
#
# Produces:
#   policy_grid.csv  (summary across all ks × wc)
#   policy_grid_deltas.csv (Δ̂ means ± std by attack, per (ks,wc))

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
from typing import Optional, Dict

import pandas as pd

def parse_int_list(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run NADGO Tier-1 policy grid with manual thresholds.")
    ap.add_argument("--budget", default="0.136", help="Manual Δ_budget (bits)")
    ap.add_argument("--kill",   default="0.147", help="Manual Δ_kill (bits)")
    ap.add_argument("--beta",   default="0.1",   help="Monitor β")
    ap.add_argument("--steps",  default="300",   help="Segments per episode")
    ap.add_argument("--depth",  default="64",    help="Logical depth")
    ap.add_argument("--seeds",  default="0:10",  help="Seed spec (start:end) or comma list; passed to tier1_sim.py")
    ap.add_argument("--qubits", default="4,8",   help="n values (comma list), matches tier1_sim default")
    ap.add_argument("--attacks", default="none,rl,timing", help="attacks to run (comma list)")
    ap.add_argument("--ks", default="2,3", help="kill_strikes set (comma list)")
    ap.add_argument("--wc", default="0,1,2", help="warn_cooldown set (comma list)")
    ap.add_argument("--outprefix", default="nadgo_grid", help="Prefix for per-run output folders")
    ap.add_argument("--qref", type=str, default="", help="Path to q_ref.json to lock baseline (recommended)")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging from tier1_sim")
    return ap.parse_args()

def abort_pct_intervals(df: pd.DataFrame, attack: str) -> float:
    s = df[df["attack"].astype(str).eq(attack)]
    return 100.0 * float((s["decision"].astype(str) == "ABORT").mean()) if not s.empty else 0.0

def warn_pct_intervals(df: pd.DataFrame, attack: str) -> float:
    s = df[df["attack"].astype(str).eq(attack)]
    return 100.0 * float((s["decision"].astype(str) == "WARNING").mean()) if not s.empty else 0.0

def ep_abort_pct(df_ep: pd.DataFrame, attack: str) -> float:
    s = df_ep[df_ep["attack"].astype(str).eq(attack)]
    if s.empty: return 0.0
    # coerce aborted -> bool
    aborted = s["aborted"]
    if aborted.dtype != bool:
        try:
            aborted = aborted.astype(int).astype(bool)
        except Exception:
            aborted = aborted.astype(str).str.lower().isin(["true","1","yes"])
    return 100.0 * float(aborted.mean())

def delta_stats_by_attack(df_int: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    if "delta_est" not in df_int.columns:
        return out
    g = df_int.groupby(df_int["attack"].astype(str))["delta_est"]
    for atk, series in g:
        out[str(atk)] = {
            "mean": float(series.mean()),
            "std":  float(series.std(ddof=1)) if series.size > 1 else 0.0,
            "n":    int(series.size),
        }
    return out

def main():
    args = parse_args()
    PY = sys.executable or "python"
    BASE = os.path.abspath(os.path.dirname(__file__))

    ks_list = parse_int_list(args.ks)
    wc_list = parse_int_list(args.wc)

    attacks = [a.strip() for a in args.attacks.split(",") if a.strip()]
    qubits  = [int(n) for n in args.qubits.split(",") if n.strip()]

    rows = []
    rows_delta = []

    # One-time note if q_ref missing
    qref_used: Optional[str] = args.qref if args.qref else None
    if qref_used and (not os.path.exists(qref_used)):
        print(f"[warn] --qref supplied but not found: {qref_used} (continuing without q_ref lock)")
        qref_used = None
    if not qref_used:
        print("[note] No --qref provided; each run will self-calibrate a baseline. "
              "For thesis-grade stability, prefer:  --qref .\\nadgo_calib\\q_ref.json")

    for ks, wc in itertools.product(ks_list, wc_list):
        outdir = f"{args.outprefix}_ks{ks}_wc{wc}"
        if os.path.exists(outdir):
            shutil.rmtree(outdir)

        cmd = [
            PY, "tier1_sim.py",
            "--manual_thresholds", str(args.budget), str(args.kill),
            "--monitor_beta", str(args.beta),
            "--steps", str(args.steps),
            "--depth", str(args.depth),
            "--seeds", str(args.seeds),
            "--kill_strikes", str(ks),
            "--warn_cooldown", str(wc),
            "--out", outdir,
            "--attacks", *attacks,
            "--qubits", *map(str, qubits),
        ]
        if args.verbose:
            cmd.append("--verbose")
        if qref_used:
            cmd.extend(["--qref", qref_used])

        print("RUN:", " ".join(cmd)); sys.stdout.flush()
        subprocess.run(cmd, check=True, cwd=BASE)

        # Validate what was used
        thr_path = os.path.join(outdir, "thresholds_used.json")
        set_path = os.path.join(outdir, "settings.json")
        iv_path  = os.path.join(outdir, "intervals.csv")
        ep_path  = os.path.join(outdir, "episodes.csv")

        thr = json.load(open(thr_path, "r", encoding="utf-8"))
        assert abs(thr["budget"] - float(args.budget)) < 1e-9, f"budget mismatch in {outdir}"
        assert abs(thr["kill"]   - float(args.kill))   < 1e-9, f"kill mismatch in {outdir}"
        assert abs(thr["monitor_beta"] - float(args.beta)) < 1e-9, f"beta mismatch in {outdir}"

        settings = json.load(open(set_path, "r", encoding="utf-8"))
        if qref_used:
            # If we forced a q_ref, ensure it was actually loaded
            if not settings.get("qref_loaded", False):
                print(f"[warn] q_ref was passed but settings say qref_loaded=False in {outdir}")

        # Load data
        iv = pd.read_csv(iv_path)
        ep = pd.read_csv(ep_path)

        # Compute metrics
        row = {
            "kill_strikes": ks, "warn_cooldown": wc,
        }
        for atk in attacks:
            row[f"ABORT%_{atk}"]   = round(abort_pct_intervals(iv, atk), 2)
            row[f"WARNING%_{atk}"] = round(warn_pct_intervals(iv, atk), 2)
            row[f"EPabort%_{atk}"] = round(ep_abort_pct(ep, atk), 2)
        rows.append(row)

        # Δ̂ by attack for this (ks,wc)
        dstats = delta_stats_by_attack(iv)
        for atk in attacks:
            s = dstats.get(atk, {"mean": float("nan"), "std": float("nan"), "n": 0})
            rows_delta.append({
                "kill_strikes": ks, "warn_cooldown": wc,
                "attack": atk, "delta_mean": s["mean"], "delta_std": s["std"], "n": s["n"],
            })

    # Summaries
    res = pd.DataFrame(rows).sort_values(["kill_strikes","warn_cooldown"])
    print("\n=== Policy grid summary ===")
    print(res.to_string(index=False))

    out_csv = "policy_grid.csv"
    res.to_csv(out_csv, index=False)
    print(f"\nWrote {out_csv}")

    if rows_delta:
        df_delta = pd.DataFrame(rows_delta).sort_values(["kill_strikes","warn_cooldown","attack"])
        out_delta = "policy_grid_deltas.csv"
        df_delta.to_csv(out_delta, index=False)
        print(f"Wrote {out_delta}")

if __name__ == "__main__":
    main()
