#!/usr/bin/env python3
# audit_tier2.py — quick audit for Tier-2 emulation outputs

from __future__ import annotations
import argparse, sys
from pathlib import Path
import pandas as pd
import numpy as np

try:
    from sklearn.metrics import roc_auc_score
    HAVE_SK = True
except Exception:
    HAVE_SK = False


def latest_dir(patterns=("tier2_emul_results*", "nadgo_results*")) -> Path | None:
    here = Path(".")
    cands = []
    for pat in patterns:
        cands += [p for p in here.glob(pat) if p.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda p: p.stat().st_mtime)


def load_csv(path: Path, label: str, required: bool = True) -> pd.DataFrame:
    print(f"[load] {label}: {path.resolve()}")
    if not path.exists():
        if required:
            print(f"[error] Missing {label} at {path.resolve()}")
            parent = path.parent
            if parent.exists():
                print(f"[hint] Contents of {parent.resolve()}:")
                for p in list(parent.iterdir())[:30]:
                    print("  -", p.name)
            sys.exit(2)
        else:
            print(f"[warn] {label} not found (optional).")
            return pd.DataFrame()
    df = pd.read_csv(path)
    print(f"[ok] {label}: {len(df):,} rows")
    return df


def coerce(df_ep: pd.DataFrame, df_seg: pd.DataFrame, df_int: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # episodes
    if not df_ep.empty:
        if "aborted" in df_ep.columns and df_ep["aborted"].dtype != bool:
            try:
                df_ep["aborted"] = df_ep["aborted"].astype(int).astype(bool)
            except Exception:
                df_ep["aborted"] = df_ep["aborted"].astype(str).str.lower().isin(["true","1","yes"])
        if "n" in df_ep.columns:
            df_ep["n"] = pd.to_numeric(df_ep["n"], errors="coerce").fillna(0).astype(int)
        if "attack" in df_ep.columns:
            df_ep["attack"] = df_ep["attack"].astype(str)

    # segments
    if not df_seg.empty:
        for col in ("n","latency","power","leak_bits"):
            if col in df_seg.columns:
                df_seg[col] = pd.to_numeric(df_seg[col], errors="coerce")
        if "n" in df_seg.columns:
            df_seg["n"] = df_seg["n"].fillna(0).astype(int)
        for col in ("attack","policy_action"):
            if col in df_seg.columns:
                df_seg[col] = df_seg[col].astype(str)

    # intervals
    if not df_int.empty:
        for col in ("n","delta_est"):
            if col in df_int.columns:
                df_int[col] = pd.to_numeric(df_int[col], errors="coerce")
        if "n" in df_int.columns:
            df_int["n"] = df_int["n"].fillna(0).astype(int)
        for col in ("attack","decision"):
            if col in df_int.columns:
                df_int[col] = df_int[col].astype(str)

    return df_ep, df_seg, df_int


def pct(x):
    return (100.0 * x) if pd.notnull(x) else np.nan


def main():
    ap = argparse.ArgumentParser(description="Audit summary for Tier-2 emulation results")
    ap.add_argument("--root", type=Path, default=None,
                    help="Folder with episodes.csv, segments.csv, intervals.csv")
    ap.add_argument("--write_csv", action="store_true",
                    help="Write audit_summary.csv next to results")
    args = ap.parse_args()

    root = args.root
    if root is None or not root.exists():
        auto = latest_dir()
        if auto is None:
            print("[error] Could not find a results directory. Pass --root <path>.")
            sys.exit(2)
        print(f"[auto] Using newest results dir: {auto.resolve()}")
        root = auto

    ep   = load_csv(root / "episodes.csv", "episodes.csv")
    seg  = load_csv(root / "segments.csv", "segments.csv")
    ints = load_csv(root / "intervals.csv", "intervals.csv", required=False)

    ep, seg, ints = coerce(ep, seg, ints)

    # ===== Abort rates by attack × n (segments: policy_action if present) =====
    if {"policy_action","n","attack"}.issubset(seg.columns):
        abort_tbl = (
            seg.assign(is_abort=lambda d: d["policy_action"].eq("ABORT"))
               .groupby(["n","attack"])["is_abort"].mean()
               .mul(100).unstack().fillna(0).round(2)
        )
        print("\nABORT % by n × attack (segments):\n", abort_tbl)
    else:
        abort_tbl = pd.DataFrame()
        print("\n[warn] Missing columns for segment-level abort table.")

    # ===== Mean ΔI per attack (segments) =====
    if {"attack","leak_bits"}.issubset(seg.columns):
        leak_tbl = (
            seg.groupby("attack")["leak_bits"]
               .agg(leak_mean="mean", leak_std="std", N="count")
        )
        print("\nΔI (leak_bits) by attack (segments):\n", leak_tbl)
    else:
        leak_tbl = pd.DataFrame()
        print("\n[warn] Missing columns for ΔI table.")

    # ===== AUC: can ΔI classify attack vs none? =====
    auc_val = np.nan
    if {"attack","leak_bits"}.issubset(seg.columns) and HAVE_SK:
        seg_bin = seg[seg["attack"].isin(["none","rl","timing"])].copy()
        if not seg_bin.empty:
            seg_bin["is_attack"] = (seg_bin["attack"] != "none").astype(int)
            try:
                auc_val = roc_auc_score(seg_bin["is_attack"], seg_bin["leak_bits"])
                print(f"\nAUC(ΔI: attack vs none): {auc_val:.3f}")
            except Exception as e:
                print(f"\n[warn] AUC could not be computed: {e}")
        else:
            print("\n[warn] No rows for AUC subset (none/rl/timing).")
    elif not HAVE_SK:
        print("\n[warn] scikit-learn not available; skipping AUC.")

    # ===== Cost deltas (episodes) =====
    ep_tbl = pd.DataFrame()
    need_cols = {"n","attack","mean_latency","mean_power","mean_fidelity"}
    if need_cols.issubset(ep.columns):
        base = ep[ep.attack=="none"].groupby("n")[["mean_latency","mean_power","mean_fidelity"]].mean()
        deltas = (
            ep.groupby(["n","attack"])[["mean_latency","mean_power","mean_fidelity"]].mean()
              .join(base, rsuffix="_base")
        )
        # Avoid divide-by-zero
        for m in ("mean_latency_base","mean_power_base","mean_fidelity_base"):
            if m in deltas.columns:
                deltas[m] = deltas[m].replace(0, np.nan)
        deltas["latency_%"]  = 100*(deltas["mean_latency"] - deltas["mean_latency_base"]) / deltas["mean_latency_base"]
        deltas["fidelity_%"] = 100*(deltas["mean_fidelity"] - deltas["mean_fidelity_base"]) / deltas["mean_fidelity_base"]
        deltas["power_%"]    = 100*(deltas["mean_power"]    - deltas["mean_power_base"])    / deltas["mean_power_base"]
        ep_tbl = deltas[["latency_%","fidelity_%","power_%"]].round(2)
        print("\nCost deltas vs none (%, episode means):\n", ep_tbl)
    else:
        print("\n[warn] Missing columns for episode-level cost deltas.")

    # ===== Interval-level ABORT% (optional) =====
    if not ints.empty and {"attack","decision"}.issubset(ints.columns):
        int_abort = (
            ints.assign(is_abort=lambda d: d["decision"].eq("ABORT"))
                .groupby("attack")["is_abort"].mean().mul(100).round(2)
        )
        print("\nABORT % by attack (intervals):\n", int_abort)
    else:
        print("\n[info] intervals.csv not present or missing columns; skipping interval ABORT%.")

    # ===== Optional CSV =====
    if args.write_csv:
        out = root / "audit_summary.csv"
        rows = []

        if not abort_tbl.empty:
            for n, row in abort_tbl.iterrows():
                for atk, v in row.items():
                    rows.append({"level":"segments", "metric":"ABORT%", "n":n, "attack":atk, "value":float(v)})

        if not leak_tbl.empty:
            for atk, r in leak_tbl.iterrows():
                rows.append({"level":"segments", "metric":"leak_mean", "attack":atk, "value":float(r["leak_mean"])})
                rows.append({"level":"segments", "metric":"leak_std",  "attack":atk, "value":float(r["leak_std"])})
                rows.append({"level":"segments", "metric":"N",         "attack":atk, "value":float(r["N"])})

        if not np.isnan(auc_val):
            rows.append({"level":"segments", "metric":"AUC_attack_vs_none", "value":float(auc_val)})

        if not ep_tbl.empty:
            for (n, atk), r in ep_tbl.reset_index().set_index(["n","attack"]).iterrows():
                rows.append({"level":"episodes","metric":"latency_%","n":n,"attack":atk,"value":float(r["latency_%"])})
                rows.append({"level":"episodes","metric":"fidelity_%","n":n,"attack":atk,"value":float(r["fidelity_%"])})
                rows.append({"level":"episodes","metric":"power_%","n":n,"attack":atk,"value":float(r["power_%"])})

        if rows:
            pd.DataFrame(rows).to_csv(out, index=False)
            print(f"\n[wrote] {out.resolve()}")


if __name__ == "__main__":
    main()
