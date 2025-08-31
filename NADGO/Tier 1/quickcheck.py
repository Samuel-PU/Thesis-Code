#!/usr/bin/env python3
# quickcheck.py — NADGO Tier-1 baseline sanity + config verification

import argparse, json, os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import hashlib

def coerce_episode_types(df_ep: pd.DataFrame) -> pd.DataFrame:
    if "aborted" in df_ep.columns and df_ep["aborted"].dtype != bool:
        try:
            df_ep["aborted"] = df_ep["aborted"].astype(int).astype(bool)
        except Exception:
            df_ep["aborted"] = df_ep["aborted"].astype(str).str.lower().isin(["true","1","yes"])
    if "attack" in df_ep.columns:
        df_ep["attack"] = df_ep["attack"].astype(str)
    return df_ep

def pct(x) -> float:
    try:
        return 100.0 * float(x)
    except Exception:
        return float("nan")

def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

def checksum_qref(path: Path) -> str:
    try:
        blob = path.read_bytes()
        # normalize by parsing json with sorted keys to be robust
        q = json.loads(blob.decode("utf-8"))
        norm = json.dumps(q, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(norm).hexdigest()[:12]
    except Exception:
        return "unknown"

def main():
    ap = argparse.ArgumentParser(description="Check baseline ABORT% and Δ̂ levels for NADGO results.")
    ap.add_argument("--root", type=Path, default=Path("./nadgo_results"),
                    help="Directory containing episodes.csv & intervals.csv & settings.json")
    ap.add_argument("--beta", type=float, default=0.1, help="β floor used by the monitor")
    ap.add_argument("--out_csv", type=Path, default=None, help="Optional CSV path for a small summary table")

    # NEW: optional verification/annotation inputs
    ap.add_argument("--qref_file", "--qref-file", dest="qref_file", type=Path, default=None,
                    help="Optional: path to a q_ref.json to compare checksum with run settings")
    ap.add_argument("--manual_thresholds", "--manual-thresholds", dest="manual_thresholds",
                    nargs=2, type=float, default=None, metavar=("BUDGET","KILL"),
                    help="Optional: manual thresholds to verify against thresholds_used.json")
    args = ap.parse_args()

    intervals_path = args.root / "intervals.csv"
    episodes_path  = args.root / "episodes.csv"
    settings_path  = args.root / "settings.json"
    thr_used_path  = args.root / "thresholds_used.json"

    if not intervals_path.exists() or not episodes_path.exists():
        print(f"[error] Missing inputs. Need both:\n  {intervals_path}\n  {episodes_path}", file=sys.stderr)
        sys.exit(2)

    di = pd.read_csv(intervals_path)
    ep = pd.read_csv(episodes_path)
    ep = coerce_episode_types(ep)

    # ------- Baseline (attack='none') interval/episode abort rates -------
    di["attack"] = di["attack"].astype(str)
    b = di[di["attack"].eq("none")].copy()

    abort_pct_none    = pct(b["decision"].astype(str).eq("ABORT").mean()) if not b.empty else float("nan")
    warning_pct_none  = pct(b["decision"].astype(str).eq("WARNING").mean()) if not b.empty else float("nan")
    continue_pct_none = pct(b["decision"].astype(str).eq("CONTINUE").mean()) if not b.empty else float("nan")
    ep_abort_pct_none = pct(ep[ep["attack"].astype(str).eq("none")]["aborted"].mean()) if not ep.empty else float("nan")

    print("\n=== Baseline (attack='none') ===")
    print(f"ABORT%_none (interval) : {abort_pct_none:6.2f}%")
    print(f"WARNING%_none (interval): {warning_pct_none:6.2f}%")
    print(f"CONTINUE%_none (interval): {continue_pct_none:6.2f}%")
    print(f"EPabort%_none (episode) : {ep_abort_pct_none:6.2f}%")

    # ------- Δ-hat by attack (sanity vs β) -------
    delta_none = float("nan")
    if "delta_est" in di.columns:
        print("\n=== Δ̂ (delta_est) by attack ===")
        g = (di.groupby("attack")["delta_est"]
               .agg(mean="mean", std="std", n="count")
               .sort_index())
        for atk, row in g.iterrows():
            mu = row["mean"]; sd = row["std"]; n = int(row["n"])
            print(f"{atk:>8s} : {mu:.4f} ± {sd:.4f} bits  (n={n})")
        delta_none = float(g.loc["none","mean"]) if "none" in g.index else float("nan")
    else:
        print("\n[warn] 'delta_est' not found in intervals.csv; cannot show Δ̂.")

    # ------- ABORT% by attack (interval) -------
    print("\n=== ABORT% by attack (interval) ===")
    abort_by_attack = (di.assign(is_abort=lambda d: d["decision"].astype(str).eq("ABORT"))
                         .groupby("attack")["is_abort"].mean().mul(100).round(2))
    if abort_by_attack.empty:
        print("(no interval rows)")
    else:
        print(abort_by_attack.to_string())

    # ------- Aborted episodes by attack (episode) -------
    print("\n=== Aborted episodes by attack (episode) ===")
    ep_abort_by_attack = (ep.groupby(ep["attack"].astype(str))["aborted"]
                            .mean().mul(100).round(2))
    if ep_abort_by_attack.empty:
        print("(no episode rows)")
    else:
        print(ep_abort_by_attack.to_string())

    # ------- Context from settings/thresholds (if present) -------
    settings = load_json(settings_path)
    thr_used = load_json(thr_used_path)

    if settings or thr_used:
        print("\n=== Run configuration (detected) ===")
    if thr_used:
        bgt = thr_used.get("budget"); kll = thr_used.get("kill")
        src = thr_used.get("source"); mb  = thr_used.get("monitor_beta")
        print(f"Thresholds : budget={bgt}  kill={kll}  (source={src})  β={mb}")
        # show effective gap if present or compute
        gap = (kll - bgt) if (isinstance(bgt,(int,float)) and isinstance(kll,(int,float))) else None
        if gap is not None:
            frac = (gap / kll * 100.0) if (kll and kll != 0) else float("nan")
            print(f"[thresholds] Effective gap = {gap:.6f} bits (frac={frac:.2f}%)")
    if settings:
        print(f"q_ref      : loaded={settings.get('qref_loaded')}  source={settings.get('qref_source')}  checksum={settings.get('qref_checksum')}")
        print(f"policy     : {settings.get('policy')}  kill_strikes={settings.get('kill_strikes')}  warn_cooldown={settings.get('warn_cooldown')}")

    # ------- Optional verification against user-specified inputs -------
    if args.qref_file and args.qref_file.exists():
        run_sum = (settings or {}).get("qref_checksum") or (thr_used or {}).get("qref_checksum")
        provided_sum = checksum_qref(args.qref_file)
        print("\n=== q_ref verification ===")
        print(f"Provided q_ref checksum : {provided_sum}")
        print(f"Run     q_ref checksum : {run_sum}")
        if run_sum and provided_sum != run_sum:
            print("⚠️  Mismatch: the run did not use the provided q_ref file.")
        else:
            print("✅ q_ref checksum matches the run.")

    if args.manual_thresholds:
        mbgt, mkll = map(float, args.manual_thresholds)
        rbgt = (thr_used or {}).get("budget"); rkll = (thr_used or {}).get("kill")
        print("\n=== manual thresholds verification ===")
        print(f"Provided: budget={mbgt}  kill={mkll}")
        print(f"Run     : budget={rbgt}  kill={rkll}")
        if rbgt is None or rkll is None:
            print("ℹ️  thresholds_used.json missing; cannot verify.")
        elif abs(mbgt - rbgt) > 1e-9 or abs(mkll - rkll) > 1e-9:
            print("⚠️  Mismatch: run thresholds differ from the provided manual thresholds.")
        else:
            print("✅ Manual thresholds match the run.")

    # ------- Light interpretation -------
    print("\n=== Interpretation ===")
    ok_abort = (abort_pct_none <= 1.0) and (ep_abort_pct_none <= 1.0)
    if ok_abort:
        print("✅ Baseline looks good: ABORT%_none ≲ 1% and EPabort%_none ≈ 0%.")
    else:
        print("⚠️  Baseline still high. Expect ABORT%_none ≲ 1% and EPabort%_none ≈ 0% if thresholds & q_ref are correct.")

    if np.isfinite(delta_none):
        if 0.11 <= delta_none <= 0.13:
            print(f"✅ Δ̂_none ≈ {delta_none:.3f} is consistent with β≈{args.beta:.3f} (floor ~0.10–0.13).")
        elif delta_none >= 1.0 and not ok_abort:
            print(f"⚠️  Δ̂_none ≈ {delta_none:.3f} is very high → likely UNIFORM q_ref. Re-run with calibrated q_ref.")
        else:
            print(f"ℹ️  Δ̂_none ≈ {delta_none:.3f}. Check against β≈{args.beta:.3f}; if far above and ABORT is high, verify q_ref & thresholds.")
    else:
        print("ℹ️  Could not assess Δ̂_none (missing delta_est).")

    # ------- Optional CSV summary -------
    out_csv = args.out_csv or (args.root / "metrics_summary.csv")
    try:
        pd.DataFrame([{
            "ABORT%_none_interval": abort_pct_none,
            "WARNING%_none_interval": warning_pct_none,
            "CONTINUE%_none_interval": continue_pct_none,
            "EPabort%_none": ep_abort_pct_none,
            "delta_none_mean": delta_none,
            "beta": args.beta
        }]).to_csv(out_csv, index=False)
        print(f"\nSaved summary → {out_csv}")
    except Exception as e:
        print(f"\n[warn] Could not write summary CSV: {e}")

    # Exit code: 0 on OK baseline; 3 on bad baseline
    sys.exit(0 if ok_abort else 3)

if __name__ == "__main__":
    main()
