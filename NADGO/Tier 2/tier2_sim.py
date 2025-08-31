#!/usr/bin/env python3
# tier2_sim.py — NADGO Tier-2 Emulation Simulator (thesis-aligned, calibrated)

import argparse
import json
import logging
import os
import sys
import hashlib
import inspect
from dataclasses import asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from simulator import (
    NADGOSimulator,
    build_qref_from_counts,
    enforce_threshold_order,
)

# ------------------------ CLI & I/O helpers -------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="NADGO Tier-2 Emulation Simulator (calibrated)")
    # Workload
    p.add_argument("--qubits", nargs="+", type=int, default=[4, 8], help="Qubit counts to simulate")
    p.add_argument("--attacks", nargs="+", default=["none", "rl", "timing"], help="Attack modes")
    p.add_argument("--seeds", default="0:10", help="Seed range (start:end) or comma list")
    p.add_argument("--shots", type=int, default=2048, help="Shots per segment")
    p.add_argument("--depth", type=int, default=64, help="Logical circuit depth")
    p.add_argument("--steps", type=int, default=300, help="Segments per episode")

    # Leakage policy thresholds (bits)
    p.add_argument("--budget", type=float, default=0.1307749780570279, help="Δ_budget (bits)")
    p.add_argument("--kill", type=float, default=0.13731372695987928, help="Δ_kill (bits)")
    p.add_argument("--monitor_beta", type=float, default=0.1, help="β term in Δ̂ (adds to KL, in bits)")

    # Calibration knobs (rarely used in Tier-2; kept for completeness)
    p.add_argument("--thr_p_budget", type=float, default=80.0,
                   help="Percentile for Δ_budget when auto-calibrating (default 80)")
    p.add_argument("--thr_p_kill", type=float, default=99.0,
                   help="Percentile for Δ_kill when auto-calibrating (default 99)")
    p.add_argument("--thr_trim", type=float, default=0.20,
                   help="Warm-up trim fraction for baseline deltas in auto-cal (default 0.20)")

    # Kill–budget gap policy
    p.add_argument("--thr_gap_frac", type=float, default=0.05,
                   help="Min relative gap between kill and budget (default 5% of budget)")
    p.add_argument("--thr_gap_abs", type=float, default=0.005,
                   help="Absolute floor (bits) for kill–budget gap (default 0.005)")

    # Padding / timing / routing (can be overridden by emulation config)
    p.add_argument("--t", type=int, default=4, help="t-design parameter")
    p.add_argument("--epsilon", type=float, default=0.02, help="ε_des for padding")
    p.add_argument("--interval", type=float, default=5.0, help="Interval length (ms)")
    p.add_argument("--sigma", type=float, default=0.8, help="Scheduler jitter σ_t (ms)")
    p.add_argument("--router_alpha", type=float, default=0.5)
    p.add_argument("--router_beta", type=float, default=0.7)
    p.add_argument("--queue_coef", type=float, default=0.25)

    # Hysteresis policy
    p.add_argument("--kill_strikes", type=int, default=2,
                   help="Consecutive kill-level intervals required to ABORT")
    p.add_argument("--warn_cooldown", type=int, default=1,
                   help="Decay horizon (intervals) for strike/cooldown hysteresis")
    p.add_argument("--policy", choices=["default", "strict", "hysteresis"], default="default",
                   help="Preset for kill/warn hysteresis")

    # Emulation
    p.add_argument("--emul_config", type=str, default="", help="Path to emulation JSON (timing/queue model)")
    p.add_argument("--require_qref_checksum", type=str, default="",
                   help="If set, abort unless q_ref checksum matches this 12-hex string")

    # Output
    p.add_argument("--out", type=str, default="tier2_emul_results", help="Output directory")
    p.add_argument("--out_stamp", action="store_true", help="Append a timestamp to --out to avoid overwrite")

    # Baseline / thresholds control
    p.add_argument("--qref", type=str, default="", help="Path to q_ref.json (baseline)")
    p.add_argument("--calibrate", action="store_true",
                   help="Run baseline and write q_ref.json (+ thresholds.json with qref_checksum)")
    p.add_argument("--auto_thresholds", action="store_true",
                   help="Derive Δ thresholds from baseline percentiles (attack=none warm-up)")
    p.add_argument("--manual_thresholds", nargs=2, type=float, metavar=("BUDGET", "KILL"),
                   default=None, help="Hard-set Δ_budget and Δ_kill (bits)")
    p.add_argument("--calib_seed", type=int, default=1234, help="Seed for warm-up calibration")

    p.add_argument("--verbose", action="store_true", help="Enable info logging")
    return p.parse_args()


def seed_iter(spec: str) -> List[int]:
    spec = str(spec).strip()
    if ":" in spec:
        start, end = map(int, spec.split(":"))
        return list(range(start, end))
    return [int(s) for s in spec.split(",") if s.strip() != ""]


def save_json(path: str, obj: dict) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def qref_checksum(qref: Dict[str, float]) -> str:
    """Deterministic short checksum for the baseline distribution."""
    blob = json.dumps(qref, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:12]


def file_checksum(path: str) -> str:
    """12-hex short checksum of a file's bytes (sha256)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()[:12]


def apply_policy_preset(args: argparse.Namespace) -> Tuple[int, int, str]:
    """Return (kill_strikes, warn_cooldown, policy_label) after applying preset."""
    if args.policy == "strict":
        return 1, 0, "strict"
    elif args.policy == "hysteresis":
        return 2, 1, "hysteresis"
    else:
        # honour user-provided numbers for default
        return int(args.kill_strikes), int(args.warn_cooldown), "default"


# ------------------------ Main -------------------------
def main():
    args = parse_args()

    if args.out_stamp:
        args.out = f"{args.out}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(args.out, exist_ok=True)

    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")

    # Apply policy preset
    ks, wc, policy_label = apply_policy_preset(args)
    args.kill_strikes, args.warn_cooldown = ks, wc

    # --- Emulation config (optional) ---
    emul_cfg: Dict = {}
    emul_cfg_path = args.emul_config.strip()
    emul_cfg_checksum = ""
    if emul_cfg_path and os.path.exists(emul_cfg_path):
        with open(emul_cfg_path, "r") as f:
            emul_cfg = json.load(f)
        try:
            emul_cfg_checksum = file_checksum(emul_cfg_path)
        except Exception:
            emul_cfg_checksum = ""
        # Apply timing/router overrides to args so Tier-1 simulator behaves more like emulation
        if "scheduler_sigma_ms" in emul_cfg:
            args.sigma = float(emul_cfg["scheduler_sigma_ms"])
        if "interval_ms" in emul_cfg:
            args.interval = float(emul_cfg["interval_ms"])
        if "router" in emul_cfg and isinstance(emul_cfg["router"], dict):
            r = emul_cfg["router"]
            args.router_alpha = float(r.get("alpha", args.router_alpha))
            args.router_beta  = float(r.get("beta",  args.router_beta))
            args.queue_coef   = float(r.get("queue_coef", args.queue_coef))

    # --- Calibration mode: build q_ref and suggest thresholds (rare in Tier-2) ---
    if args.calibrate:
        sim = NADGOSimulator(
            n_qubits=max(args.qubits), delta_budget=args.budget, delta_kill=args.kill,
            shots=args.shots, t_value=args.t, epsilon_des=args.epsilon,
            sigma_t=args.sigma, router_alpha=args.router_alpha, router_beta=args.router_beta,
            queue_coef=args.queue_coef, interval_ms=args.interval,
            kill_strikes=args.kill_strikes, warn_cooldown=args.warn_cooldown,
            monitor_beta=args.monitor_beta,
        )
        _ep, _ = sim.simulate_episode(seed=args.calib_seed, attack_name="none",
                                      depth=args.depth, steps=args.steps)

        # Build Dirichlet-smoothed q_ref from observed features
        alpha = 1.0
        counts = {f: float(sim.feature_counts.get(f, 0.0)) + alpha for f in sim.monitor.feature_alphabet}
        s = sum(counts.values())
        q_ref = {k: v / s for k, v in counts.items()}
        qref_path = os.path.join(args.out, "q_ref.json")
        save_json(qref_path, q_ref)

        # Threshold suggestions from baseline Δ̂ distribution (trim first thr_trim)
        base_deltas = [rec['delta_est'] for rec in sim.audit_log]
        thr_path = os.path.join(args.out, "thresholds.json")
        if base_deltas:
            cut = int(float(args.thr_trim) * len(base_deltas))
            trimmed = base_deltas[cut:] or base_deltas
            b = float(np.percentile(trimmed, float(args.thr_p_budget)))
            k = float(np.percentile(trimmed, float(args.thr_p_kill)))
            b, k = enforce_threshold_order(b, k, min_gap_frac=args.thr_gap_frac, min_gap_abs=args.thr_gap_abs)
            thr = {"P80_budget": b, "P99_kill": k, "qref_checksum": qref_checksum(q_ref)}
            save_json(thr_path, thr)
            print(f"Calibration complete.\n  q_ref -> {qref_path}\n  thresholds -> {thr_path}")
        else:
            thr = {"qref_checksum": qref_checksum(q_ref)}
            save_json(thr_path, thr)
            print(f"Calibration complete.\n  q_ref -> {qref_path}\n  thresholds -> {thr_path} (no percentiles computed)")
        return

    # --- Inference mode: load q_ref/thresholds or self-calibrate (discouraged for Tier-2) ---
    q_ref: Dict[str, float] = {}
    qref_source = "none"
    thr_source = "cli-defaults"
    thresholds_loaded = False
    loaded_thr_checksum: Optional[str] = None

    # Try to load q_ref from file
    if args.qref and os.path.exists(args.qref):
        with open(args.qref, "r") as f:
            q_ref = json.load(f)
        qref_source = "file"

        # If thresholds.json sits next to q_ref and auto-thresholds not requested, use it if checksum matches
        thr_guess = os.path.join(os.path.dirname(args.qref), "thresholds.json")
        if (not args.auto_thresholds) and os.path.exists(thr_guess):
            with open(thr_guess, "r") as f:
                thr = json.load(f)
            loaded_thr_checksum = thr.get("qref_checksum")
            # Accept either P80/P95 or P80/P99 keys
            key_b = "P80_budget" if "P80_budget" in thr else "budget"
            key_k = "P99_kill" if "P99_kill" in thr else ("P95_kill" if "P95_kill" in thr else "kill")
            if key_b in thr and key_k in thr:
                if loaded_thr_checksum and loaded_thr_checksum != qref_checksum(q_ref):
                    print("[warn] thresholds.json qref_checksum mismatches supplied q_ref; deriving fresh thresholds.")
                    args.auto_thresholds = True
                    thresholds_loaded = False
                else:
                    args.budget, args.kill = enforce_threshold_order(
                        float(thr[key_b]), float(thr[key_k]),
                        min_gap_frac=args.thr_gap_frac, min_gap_abs=args.thr_gap_abs
                    )
                    thresholds_loaded = True
                    thr_source = "file"
                    print(f"[thresholds] Using file thresholds: Δ_budget={args.budget:.4f}, Δ_kill={args.kill:.4f}")

    # Manual thresholds override (highest priority)
    if args.manual_thresholds is not None:
        mb, mk = args.manual_thresholds
        args.budget, args.kill = enforce_threshold_order(
            float(mb), float(mk),
            min_gap_frac=args.thr_gap_frac, min_gap_abs=args.thr_gap_abs
        )
        thresholds_loaded = True
        thr_source = "manual-cli"
        print(f"[manual] Δ_budget={args.budget:.4f}, Δ_kill={args.kill:.4f}")

    # Warm-up length to avoid identical percentiles
    warm_steps = max(1200, args.steps)  # longer warmup for cleaner percentiles

    # If auto-thresholds requested, run a warm-up (attack=none) to derive P80/P99
    if (not thresholds_loaded) and args.auto_thresholds:
        sim_warm = NADGOSimulator(
            n_qubits=max(args.qubits), delta_budget=args.budget, delta_kill=args.kill,
            shots=args.shots, t_value=args.t, epsilon_des=args.epsilon,
            sigma_t=args.sigma, router_alpha=args.router_alpha, router_beta=args.router_beta,
            queue_coef=args.queue_coef, interval_ms=args.interval,
            kill_strikes=args.kill_strikes, warn_cooldown=args.warn_cooldown,
            monitor_beta=args.monitor_beta,
        )
        if q_ref:
            sim_warm.monitor.load_qref(q_ref)
        _ep, _ = sim_warm.simulate_episode(seed=args.calib_seed, attack_name="none",
                                           depth=args.depth, steps=warm_steps)
        base_deltas = [rec['delta_est'] for rec in sim_warm.audit_log]
        if base_deltas:
            cut = int(float(args.thr_trim) * len(base_deltas))
            trimmed = base_deltas[cut:] or base_deltas
            b = float(np.percentile(trimmed, float(args.thr_p_budget)))
            k = float(np.percentile(trimmed, float(args.thr_p_kill)))
            args.budget, args.kill = enforce_threshold_order(
                b, k, min_gap_frac=args.thr_gap_frac, min_gap_abs=args.thr_gap_abs
            )
            thresholds_loaded = True
            thr_source = "auto-percentiles"
            print(f"[auto-thresholds] Δ_budget={args.budget:.4f}, Δ_kill={args.kill:.4f}")

    # If still no q_ref, self-calibrate a baseline and derive thresholds (Tier-2: not typical)
    if not q_ref:
        print("[self-cal] No q_ref supplied → running warm-up baseline (attack=none)")
        sim_warm2 = NADGOSimulator(
            n_qubits=max(args.qubits), delta_budget=args.budget, delta_kill=args.kill,
            shots=args.shots, t_value=args.t, epsilon_des=args.epsilon,
            sigma_t=args.sigma, router_alpha=args.router_alpha, router_beta=args.router_beta,
            queue_coef=args.queue_coef, interval_ms=args.interval,
            kill_strikes=args.kill_strikes, warn_cooldown=args.warn_cooldown,
            monitor_beta=args.monitor_beta,
        )
        _ep, _ = sim_warm2.simulate_episode(seed=args.calib_seed, attack_name="none",
                                            depth=args.depth, steps=warm_steps)
        q_ref = build_qref_from_counts(sim_warm2.feature_counts, sim_warm2.monitor.feature_alphabet, alpha=1.0)
        qref_source = "self-cal"

        base_deltas = [rec['delta_est'] for rec in sim_warm2.audit_log]
        if (not thresholds_loaded) and base_deltas:
            cut = int(float(args.thr_trim) * len(base_deltas))
            trimmed = base_deltas[cut:] or base_deltas
            b = float(np.percentile(trimmed, float(args.thr_p_budget)))
            k = float(np.percentile(trimmed, float(args.thr_p_kill)))
            args.budget, args.kill = enforce_threshold_order(
                b, k, min_gap_frac=args.thr_gap_frac, min_gap_abs=args.thr_gap_abs
            )
            thresholds_loaded = True
            thr_source = "self-cal-percentiles"
            print(f"[self-cal] Δ_budget={args.budget:.4f}, Δ_kill={args.kill:.4f}")

    # If q_ref exists but no thresholds chosen yet (and no auto), derive from warm-up with q_ref
    if (q_ref and not thresholds_loaded and not args.auto_thresholds):
        print("[warm-up] q_ref present but no thresholds; deriving percentiles from warm baseline")
        sim_warm3 = NADGOSimulator(
            n_qubits=max(args.qubits), delta_budget=args.budget, delta_kill=args.kill,
            shots=args.shots, t_value=args.t, epsilon_des=args.epsilon,
            sigma_t=args.sigma, router_alpha=args.router_alpha, router_beta=args.router_beta,
            queue_coef=args.queue_coef, interval_ms=args.interval,
            kill_strikes=args.kill_strikes, warn_cooldown=args.warn_cooldown,
            monitor_beta=args.monitor_beta,
        )
        sim_warm3.monitor.load_qref(q_ref)
        _ep, _ = sim_warm3.simulate_episode(seed=args.calib_seed, attack_name="none",
                                            depth=args.depth, steps=warm_steps)
        base_deltas = [rec['delta_est'] for rec in sim_warm3.audit_log]
        if base_deltas:
            cut = int(float(args.thr_trim) * len(base_deltas))
            trimmed = base_deltas[cut:] or base_deltas
            b = float(np.percentile(trimmed, float(args.thr_p_budget)))
            k = float(np.percentile(trimmed, float(args.thr_p_kill)))
            args.budget, args.kill = enforce_threshold_order(
                b, k, min_gap_frac=args.thr_gap_frac, min_gap_abs=args.thr_gap_abs
            )
            thresholds_loaded = True
            thr_source = "warm-up-percentiles"
            print(f"[warm-up] Δ_budget={args.budget:.4f}, Δ_kill={args.kill:.4f}")

    # Final fallback: use β-aware SAFE defaults
    if not thresholds_loaded:
        beta = float(args.monitor_beta)
        base = max(beta + 0.02, 0.12)   # ensure > β and a sensible minimum
        kill = base + 0.02
        args.budget, args.kill = enforce_threshold_order(
            base, kill, min_gap_frac=args.thr_gap_frac, min_gap_abs=args.thr_gap_abs
        )
        thr_source = "safe-defaults-beta-aware"
        print(f"[fallback] Using β-aware defaults: β={beta:.3f}, Δ_budget={args.budget:.3f}, Δ_kill={args.kill:.3f}")

    # Sanity guard: thresholds must exceed β with a margin and be ordered (with gap)
    if args.budget <= args.monitor_beta or args.kill <= args.budget:
        print(f"[warn] thresholds (budget={args.budget:.4f}, kill={args.kill:.4f}) conflict with β={args.monitor_beta:.4f}; adjusting.")
        base = max(args.monitor_beta + 0.02, args.budget, 0.12)
        args.budget, args.kill = enforce_threshold_order(
            base, base + 0.02, min_gap_frac=args.thr_gap_frac, min_gap_abs=args.thr_gap_abs
        )
        thr_source = f"{thr_source}+sanity-fix"
        print(f"[fixed] Δ_budget={args.budget:.4f}, Δ_kill={args.kill:.4f}")

    # Enforce q_ref checksum if user requested
    qref_loaded = bool(q_ref)
    qref_hash = qref_checksum(q_ref) if qref_loaded else "none"
    if args.require_qref_checksum:
        exp = args.require_qref_checksum.strip()
        if qref_hash != exp:
            raise SystemExit(f"[FATAL] q_ref checksum mismatch. expected={exp}, got={qref_hash}. "
                             f"Pass --qref pointing to the locked q_ref.json.")

    # Report effective gap for auditability
    eff_gap = args.kill - args.budget
    eff_frac = eff_gap / max(args.budget, 1e-9)
    print(f"[thresholds] Effective gap = {eff_gap:.6f} bits (frac={eff_frac:.3%})")

    # Print effective configuration
    print(
        "[config] "
        f"tier=2, qref_source={qref_source}, thr_source={thr_source}, "
        f"qref_loaded={qref_loaded}, qref_checksum={qref_hash}, "
        f"Δ_budget={args.budget:.4f}, Δ_kill={args.kill:.4f}, "
        f"policy={policy_label}, kill_strikes={args.kill_strikes}, warn_cooldown={args.warn_cooldown}, "
        f"β={args.monitor_beta:.4f}, emul_name={emul_cfg.get('name','')}"
    )

    # ---------------- Run experiments ----------------
    all_episodes: List[dict] = []
    all_segments: List[dict] = []
    all_audits: List[dict] = []  # interval-level audits with context

    seeds = seed_iter(args.seeds)

    # Detect if simulator supports emul_config
    sim_accepts_emul = "emul_config" in inspect.signature(NADGOSimulator.__init__).parameters

    for n in args.qubits:
        for attack in args.attacks:
            for seed in seeds:
                logging.info(f"Starting n={n}, attack={attack}, seed={seed}")
                sim_kwargs = dict(
                    n_qubits=n, delta_budget=args.budget, delta_kill=args.kill,
                    shots=args.shots, t_value=args.t, epsilon_des=args.epsilon,
                    sigma_t=args.sigma, router_alpha=args.router_alpha, router_beta=args.router_beta,
                    queue_coef=args.queue_coef, interval_ms=args.interval,
                    kill_strikes=args.kill_strikes, warn_cooldown=args.warn_cooldown,
                    monitor_beta=args.monitor_beta,
                )
                if emul_cfg and sim_accepts_emul:
                    sim_kwargs["emul_config"] = emul_cfg

                sim = NADGOSimulator(**sim_kwargs)

                if q_ref:
                    sim.monitor.load_qref(qref=q_ref)

                ep_result, seg_results = sim.simulate_episode(
                    seed=seed, attack_name=attack, depth=args.depth, steps=args.steps
                )

                all_episodes.append(asdict(ep_result))
                all_segments.extend(asdict(sr) for sr in seg_results)

                # attach context to audit records for CSV export
                for rec in sim.audit_log:
                    rec2 = rec.copy()
                    rec2["seed"] = int(seed)
                    rec2["n"] = int(n)
                    rec2["attack"] = str(attack)
                    all_audits.append(rec2)

    # Save results
    df_ep = pd.DataFrame(all_episodes)
    df_seg = pd.DataFrame(all_segments)
    df_int = pd.DataFrame(all_audits)

    settings = {
        "tier": 2,
        "budget": args.budget, "kill": args.kill, "interval_ms": args.interval,
        "sigma_t": args.sigma, "router_alpha": args.router_alpha,
        "router_beta": args.router_beta, "queue_coef": args.queue_coef,
        "kill_strikes": args.kill_strikes, "warn_cooldown": args.warn_cooldown,
        "qref_loaded": bool(q_ref), "qref_source": qref_source, "threshold_source": thr_source,
        "qref_checksum": qref_hash, "policy": policy_label, "monitor_beta": args.monitor_beta,
        "thr_p_budget": args.thr_p_budget, "thr_p_kill": args.thr_p_kill, "thr_trim": args.thr_trim,
        "thr_gap_frac": args.thr_gap_frac, "thr_gap_abs": args.thr_gap_abs,
        "emul_config_used": bool(emul_cfg),
        "emul_name": emul_cfg.get("name","") if emul_cfg else "",
        "emul_checksum": emul_cfg_checksum,
    }

    ep_path = os.path.join(args.out, "episodes.csv")
    seg_path = os.path.join(args.out, "segments.csv")
    aud_json_path = os.path.join(args.out, "audit_log.json")
    set_path = os.path.join(args.out, "settings.json")
    int_path = os.path.join(args.out, "intervals.csv")

    os.makedirs(args.out, exist_ok=True)
    df_ep.to_csv(ep_path, index=False)
    df_seg.to_csv(seg_path, index=False)
    if not df_int.empty:
        df_int.to_csv(int_path, index=False)
    save_json(aud_json_path, {"records": all_audits})
    save_json(set_path, settings)

    # Also store the exact q_ref used and the emulation config (for provenance)
    try:
        if q_ref:
            save_json(os.path.join(args.out, "q_ref_used.json"), q_ref)
        if emul_cfg:
            save_json(os.path.join(args.out, "emul_config_used.json"), emul_cfg)
    except Exception as e:
        print(f"[warn] Could not save provenance extras: {e}")

    print("Simulation complete.")
    print(f"  Episodes -> {ep_path}")
    print(f"  Segments -> {seg_path}")
    if not df_int.empty:
        print(f"  Intervals -> {int_path}")
    print(f"  Audit    -> {aud_json_path}")
    print(f"  Settings -> {set_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
