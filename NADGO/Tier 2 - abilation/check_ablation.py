#!/usr/bin/env python3
# check_ablation.py — sanity checks for Tier-2 ablation runs

import argparse, json
from pathlib import Path
import pandas as pd
import numpy as np
from math import sqrt

def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def wilson_ci(k, n, z=1.96):
    if n <= 0: return (0.0, 0.0)
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = (z/denom) * sqrt((p*(1-p)/n) + (z*z/(4*n*n)))
    return max(0.0, centre-half), min(1.0, centre+half)

def main():
    ap = argparse.ArgumentParser(description="Check ablation runs for counts, thresholds, knobs, parity")
    ap.add_argument("--runs", nargs="+", required=True,
                    help="Run folders (tier2_vendor tier2_padonly tier2_jitter tier2_full)")
    ap.add_argument("--out", default="ablation_summary.csv")
    args = ap.parse_args()

    rows = []
    thr_rows = []
    knob_rows = []
    seed_rows = []

    for r in args.runs:
        rd = Path(r)
        name = rd.name
        intervals = rd/"intervals.csv"
        settings  = rd/"settings.json"
        thr_used  = rd/"thresholds_used.json"

        # --- Counts + ABORT% by attack
        if intervals.exists():
            df = pd.read_csv(intervals)
            if "decision" in df and "attack" in df:
                g = (df.assign(is_abort=lambda d: d["decision"].astype(str).eq("ABORT"))
                       .groupby("attack")["is_abort"].agg(["sum","count"]).reset_index())
                for _, rec in g.iterrows():
                    k, n = int(rec["sum"]), int(rec["count"])
                    lo, hi = wilson_ci(k, n)
                    rows.append({
                        "run": name,
                        "attack": rec["attack"],
                        "intervals": n,
                        "ABORT%": (k/n*100.0) if n else 0.0,
                        "ABORT%_95%_lo": lo*100.0,
                        "ABORT%_95%_hi": hi*100.0,
                        "ABORT_count": k
                    })
            # seeds used
            if "seed" in df.columns:
                seeds = sorted(set(int(s) for s in df["seed"].dropna().tolist()))
                seed_rows.append({"run": name, "seeds": f"{seeds}", "n_seeds": len(seeds)})
        else:
            print(f"[warn] missing {intervals}")

        # --- Calibration & thresholds
        s = load_json(settings) if settings.exists() else {}
        t = load_json(thr_used) if thr_used.exists() else {}
        thr_rows.append({
            "run": name,
            "settings_qref_checksum": s.get("qref_checksum", ""),
            "thr_qref_checksum": t.get("qref_checksum", ""),
            "settings_budget": s.get("budget", None),
            "thr_budget": t.get("budget", t.get("P80_budget", None)),
            "settings_kill": s.get("kill", None),
            "thr_kill": t.get("kill", t.get("P99_kill", t.get("P95_kill", None))),
            "monitor_beta": s.get("monitor_beta", None),
        })
        knob_rows.append({
            "run": name,
            "kill_strikes": s.get("kill_strikes", None),
            "warn_cooldown": s.get("warn_cooldown", None),
            "interval_ms": s.get("interval_ms", None),
            "sigma_t": s.get("sigma_t", None),
            "router_alpha": s.get("router_alpha", None),
            "router_beta": s.get("router_beta", None),
            "queue_coef": s.get("queue_coef", None),
            "policy": s.get("policy",""),
        })

    # Print summaries
    if rows:
        df_rows = pd.DataFrame(rows).sort_values(["run","attack"])
        print("\n=== ABORT% by run × attack (with 95% CI) ===")
        print(df_rows.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
        df_rows.to_csv(args.out, index=False)
        print(f"[ok] wrote {args.out}")

    if thr_rows:
        thr_df = pd.DataFrame(thr_rows)
        print("\n=== Threshold & qref checks ===")
        thr_df["qref_match"] = thr_df["settings_qref_checksum"].astype(str).eq(thr_df["thr_qref_checksum"].astype(str))
        thr_df["budget_match"] = np.isclose(thr_df["settings_budget"], thr_df["thr_budget"], rtol=0, atol=1e-6)
        thr_df["kill_match"]   = np.isclose(thr_df["settings_kill"],   thr_df["thr_kill"],   rtol=0, atol=1e-6)
        print(thr_df[["run","settings_qref_checksum","thr_qref_checksum","qref_match",
                      "settings_budget","thr_budget","budget_match",
                      "settings_kill","thr_kill","kill_match","monitor_beta"]]
              .to_string(index=False))
    if knob_rows:
        knobs = pd.DataFrame(knob_rows)
        print("\n=== Policy/timing knobs (should match across runs) ===")
        print(knobs.to_string(index=False))

    if seed_rows:
        seeds = pd.DataFrame(seed_rows)
        print("\n=== Seeds per run ===")
        print(seeds.to_string(index=False))

if __name__ == "__main__":
    main()
