"""
emulator_cli.py
---------------
CLI to run a Tier-2 emulation and write JSONL metrics.

Usage examples:
  python emulator_cli.py --scenario noisy --seed 123 --version 1.0 ^
      --hardware A100x1 --workload A=0.7,B=0.3 --n-requests 1000 ^
      --lat-mean 180 --lat-std 40 --timeout-ms 500 --out runs.jsonl ^
      --enable-mem --mem-sample-every 10 --instance-hourly-usd 2.5 ^
      --energy-per-request-j 0.05 --fidelity-mean 0.98 --fidelity-std 0.01

Replace the synthetic latency generator with your real request loop:
  - call `exp.log_request(latency_ms=..., queue_ms=..., retries=..., outcome=...)`
  - update Tier-1 via `exp.record_effectiveness(k=..., NDCG=...)` when appropriate.
"""

import argparse, json, time, random
import numpy as np
from emulation_metrics import ExperimentMetrics, EmulationConfig

try:
    import psutil  # Optional, for memory sampling
except Exception:
    psutil = None

def parse_workload(s: str):
    out = {}
    if not s:
        return out
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = float(v)
    return out

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["baseline","noisy","bursty","adversarial"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--version", type=str, default="1.0")
    ap.add_argument("--hardware", type=str, default="gpu_high_mem")
    ap.add_argument("--workload", type=str, default="A=0.7,B=0.3", help="CSV like A=0.7,B=0.3")
    ap.add_argument("--notes", type=str, default="")
    ap.add_argument("--n-requests", type=int, default=200)
    ap.add_argument("--lat-mean", type=float, default=180)
    ap.add_argument("--lat-std", type=float, default=40)
    ap.add_argument("--timeout-ms", type=float, default=600)
    ap.add_argument("--out", type=str, required=True)

    # Overhead & adversary knobs
    ap.add_argument("--error-prob", type=float, default=0.0, help="Only used for adversarial scenario")
    ap.add_argument("--extra-latency-ms", type=float, default=0.0, help="Added per-request latency in adversarial")
    ap.add_argument("--overhead-base-pct", type=float, default=None, help="Override default obfuscation overhead %% for scenario")

    # Memory/energy logging
    ap.add_argument("--enable-mem", action="store_true", help="Sample memory via psutil if available")
    ap.add_argument("--mem-sample-every", type=int, default=10)
    ap.add_argument("--instance-hourly-usd", type=float, default=0.0, help="Cloud instance hourly cost; used to compute cloud_cost_usd")
    ap.add_argument("--energy-per-request-j", type=float, default=0.0, help="If >0, append this much J per request")

    # Domain metric synthesis (optional — provide means/stds to fill nulls)
    ap.add_argument("--fidelity-mean", type=float, default=None)
    ap.add_argument("--fidelity-std", type=float, default=0.0)
    ap.add_argument("--recon-mse-mean", type=float, default=None)
    ap.add_argument("--recon-mse-std", type=float, default=0.0)
    ap.add_argument("--mi-bits-mean", type=float, default=None)
    ap.add_argument("--mi-bits-std", type=float, default=0.0)
    ap.add_argument("--shots-mean", type=float, default=None)
    ap.add_argument("--shots-std", type=float, default=0.0)
    ap.add_argument("--fragments-mean", type=float, default=None)
    ap.add_argument("--fragments-std", type=float, default=0.0)
    ap.add_argument("--cuts-mean", type=float, default=None)
    ap.add_argument("--cuts-std", type=float, default=0.0)

    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    workload = parse_workload(args.workload)

    exp = ExperimentMetrics()
    cfg = EmulationConfig(
        scenario=args.scenario,
        seed=args.seed,
        version=args.version,
        hardware_profile=args.hardware,
        workload_mix=workload,
        notes=args.notes,
    )

    # Scenario defaults (obfuscation overhead baseline)
    if args.overhead_base_pct is not None:
        overhead_base = args.overhead_base_pct / 100.0
    else:
        overhead_base = {"baseline": 0.0, "noisy": 0.005, "bursty": 0.006, "adversarial": 0.008}[args.scenario]
    error_prob = args.error_prob if args.scenario == "adversarial" else 0.0
    extra_latency = args.extra_latency_ms if args.scenario == "adversarial" else 0.0

    with exp.run(cfg) as run:
        # Synthetic request loop
        for i in range(args.n_requests):
            # Base latency draw
            lat = float(max(1.0, rng.normal(args.lat_mean, args.lat_std)))
            # Tails by scenario
            if args.scenario == "bursty" and rng.random() < 0.05:
                lat *= float(rng.lognormal(mean=0.8, sigma=0.8))
            if args.scenario == "noisy":
                lat *= float(rng.normal(1.05, 0.02))
            if args.scenario == "adversarial":
                lat += extra_latency
            # Queue/service split
            queue_ms = float(rng.uniform(0, min(lat * 0.2, 50.0)))
            service_ms = max(0.0, lat - queue_ms)

            # Outcome
            outcome = "success"
            if lat > args.timeout_ms:
                outcome = "timeout"
            elif error_prob > 0 and rng.random() < error_prob:
                outcome = "error"

            # Record the request (directly to Tier2; equivalent to run.tier2.log_request but faster in a tight loop)
            run.tier2.obfuscation_overhead_pct.append(float(overhead_base * rng.normal(1.0, 0.01)))
            run.tier2.latency_ms.append(lat)
            run.tier2.queue_wait_ms.append(queue_ms)
            run.tier2.service_time_ms.append(service_ms)
            run.tier2.retry_count.append(0)
            run.tier2.n_requests += 1
            if outcome == "success":
                run.tier2.n_success += 1
            elif outcome == "timeout":
                run.tier2.n_timeout += 1
            else:
                run.tier2.n_error += 1

            # Optional memory sampling
            if args.enable_mem and psutil is not None and (i % max(1, args.mem_sample_every) == 0):
                rss_mb = psutil.Process().memory_info().rss / (1024 * 1024)
                run.tier2.mem_samples_mb.append(float(rss_mb))

            # Optional energy sample
            if args.energy_per_request_j > 0:
                run.tier2.energy_j_samples.append(float(args.energy_per_request_j))

            # Optional domain metrics per request
            if args.fidelity_mean is not None:
                run.tier2.fidelity.append(float(clamp(rng.normal(args.fidelity_mean, args.fidelity_std), 0.0, 1.0)))
            if args.recon_mse_mean is not None:
                run.tier2.recon_mse.append(float(max(0.0, rng.normal(args.recon_mse_mean, args.recon_mse_std))))
            if args.mi_bits_mean is not None:
                run.tier2.mi_leakage_bits.append(float(max(0.0, rng.normal(args.mi_bits_mean, args.mi_bits_std))))
            if args.shots_mean is not None:
                run.tier2.shots_used.append(int(max(0, round(rng.normal(args.shots_mean, args.shots_std)))))
            if args.fragments_mean is not None:
                run.tier2.fragment_count.append(int(max(0, round(rng.normal(args.fragments_mean, args.fragments_std)))))
            if args.cuts_mean is not None:
                run.tier2.cut_count.append(int(max(0, round(rng.normal(args.cuts_mean, args.cuts_std)))))

        # Complete run and compute cloud cost
        run.complete_run()
        if args.instance_hourly_usd > 0 and run.end_time is not None:
            elapsed_h = (run.end_time - run.start_time) / 3600.0
            run.tier2.cloud_cost_usd = float(args.instance_hourly_usd * elapsed_h)

        # Example Tier-1 effectiveness metric
        exp.record_effectiveness(k=10, NDCG=float(np.random.default_rng(args.seed).uniform(0.6, 0.9)))

    with open(args.out, "a", encoding="utf-8") as f:
        for r in exp.runs:
            f.write(json.dumps(r.to_dict(), ensure_ascii=False) + "\n")

    print(json.dumps(exp.summarize_by_scenario(), indent=2))

if __name__ == "__main__":
    main()
