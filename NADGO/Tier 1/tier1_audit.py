# audit_tier1.py
import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

ep = pd.read_csv(r"nadgo_results_20250813-150301/episodes.csv")
seg = pd.read_csv(r"nadgo_results_20250813-150301/segments.csv")

# Abort rates by attack × n
abort_tbl = (seg.assign(is_abort=lambda d: d.policy_action=="ABORT")
               .groupby(["n","attack"])["is_abort"].mean()
               .mul(100).unstack().fillna(0).round(2))
print("\nABORT % by n × attack:\n", abort_tbl)

# Mean ΔI per attack (segments)
leak_tbl = (seg.groupby("attack")["leak_bits"]
              .agg(["mean","std","count"])
              .rename(columns={"mean":"leak_mean","std":"leak_std","count":"N"}))
print("\nΔI (leak_bits) by attack:\n", leak_tbl)

# AUC: can ΔI classify attack vs none?
seg_bin = seg[seg["attack"].isin(["none","rl","timing"])].copy()
seg_bin["is_attack"] = (seg_bin["attack"]!="none").astype(int)
auc = roc_auc_score(seg_bin["is_attack"], seg_bin["leak_bits"])
print(f"\nAUC(ΔI: attack vs none): {auc:.3f}")

# Cost deltas (episodes)
base = ep[ep.attack=="none"].groupby("n")[["mean_latency","mean_power","mean_fidelity"]].mean()
deltas = (ep.groupby(["n","attack"])[["mean_latency","mean_power","mean_fidelity"]].mean()
            .join(base, rsuffix="_base"))
deltas["latency_%"]  = 100*(deltas["mean_latency"]-deltas["mean_latency_base"])/deltas["mean_latency_base"]
deltas["fidelity_%"] = 100*(deltas["mean_fidelity"]-deltas["mean_fidelity_base"])/deltas["mean_fidelity_base"]
deltas["power_%"]    = 100*(deltas["mean_power"]-deltas["mean_power_base"])/deltas["mean_power_base"]
print("\nCost deltas vs none (%, mean over seeds):\n",
      deltas[["latency_%","fidelity_%","power_%"]].round(2))
