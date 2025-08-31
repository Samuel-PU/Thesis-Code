#!/usr/bin/env python3
# combine_ablation.py — merge multiple Tier-2 runs into one folder with a `pipeline` column

import argparse, json, os
from pathlib import Path
import pandas as pd

def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()

def pipeline_label(run_dir: Path) -> str:
    # 1) try settings.json
    s = run_dir / "settings.json"
    if s.exists():
        try:
            j = json.loads(s.read_text())
            p = str(j.get("pipeline","")).strip()
            if p: return p
        except Exception:
            pass
    # 2) infer from folder name
    name = run_dir.name.lower()
    for key in ["vendor","pad-only","jitter-only","full"]:
        if key in name: return key
    return "unknown"

def tag(df: pd.DataFrame, pl: str) -> pd.DataFrame:
    if df.empty: return df
    df = df.copy()
    if "pipeline" not in df.columns:
        df["pipeline"] = pl
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="Run folders to combine")
    ap.add_argument("--out", required=True, help="Output folder (will be created)")
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    eps, segs, ivs = [], [], []
    for r in args.runs:
        rd = Path(r)
        pl = pipeline_label(rd)
        eps.append(tag(load_csv(rd/"episodes.csv"), pl))
        segs.append(tag(load_csv(rd/"segments.csv"), pl))
        ivs.append(tag(load_csv(rd/"intervals.csv"), pl))

    df_ep  = pd.concat([d for d in eps  if not d.empty], ignore_index=True) if any(not d.empty for d in eps)  else pd.DataFrame()
    df_seg = pd.concat([d for d in segs if not d.empty], ignore_index=True) if any(not d.empty for d in segs) else pd.DataFrame()
    df_iv  = pd.concat([d for d in ivs  if not d.empty], ignore_index=True) if any(not d.empty for d in ivs)  else pd.DataFrame()

    if df_ep.empty and df_seg.empty and df_iv.empty:
        raise SystemExit("No CSVs found in supplied runs.")

    if not df_ep.empty:  df_ep.to_csv(out/"episodes.csv", index=False)
    if not df_seg.empty: df_seg.to_csv(out/"segments.csv", index=False)
    if not df_iv.empty:  df_iv.to_csv(out/"intervals.csv", index=False)

    print(f"[ok] combined → {out.resolve()}")
    for fname in ["episodes.csv","segments.csv","intervals.csv"]:
        p = out/fname
        if p.exists(): print("   -", fname, ":", sum(1 for _ in open(p)) - 1, "rows")

if __name__ == "__main__":
    main()
