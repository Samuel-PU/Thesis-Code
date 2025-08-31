"""
summarise_metrics.py
--------------------
Load a JSONL of runs and emit:
  1) scenario-level summary (printed)
  2) per-run flat CSV (optional with --csv-out)

Usage:
  python summarise_metrics.py --jsonl runs.jsonl --csv-out runs_flat.csv
"""

import argparse, csv, json
import numpy as np
from emulation_metrics import ExperimentMetrics, bootstrap_ci

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--csv-out", default=None)
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    em = ExperimentMetrics.from_jsonl(args.jsonl)

    # Ensure derived metrics exist even if they weren't stored
    for r in em.runs:
        if r.end_time is not None:
            r.tier2.compute_derived_metrics(duration_s=(r.end_time - r.start_time))

    def ci_of(vals):
        return bootstrap_ci(vals, n_bootstraps=args.n_bootstrap, alpha=args.alpha)

    out = []
    scenarios = sorted({r.config.scenario for r in em.runs})
    for s in scenarios:
        runs = [r for r in em.runs if r.config.scenario == s]
        def fld(name):
            xs = []
            for r in runs:
                v = getattr(r.tier2, name, None)
                if v is not None:
                    xs.append(v)
            return xs
        row = {
            "scenario": s,
            "runs": len(runs),
            "lat_p50_ms_ci": ci_of(fld("p50_latency_ms")),
            "lat_p95_ms_ci": ci_of(fld("p95_latency_ms")),
            "lat_p99_ms_ci": ci_of(fld("p99_latency_ms")),
            "jitter_ms_ci": ci_of(fld("jitter_ms")),
            "ttfr_ms_ci": ci_of(fld("ttfr_ms")),
            "throughput_qps_ci": ci_of(fld("throughput_qps")),
            "success_throughput_qps_ci": ci_of(fld("success_throughput_qps")),
            "emulated_throughput_qps_ci": ci_of(fld("emulated_throughput_qps")),
            "success_rate_ci": ci_of(fld("success_rate")),
            "timeout_rate_ci": ci_of(fld("timeout_rate")),
            "error_rate_ci": ci_of(fld("error_rate")),
            "mem_peak_mb_max_ci": ci_of(fld("mem_peak_mb_max")),
            "energy_j_total_ci": ci_of(fld("energy_j_total")),
            "cloud_cost_usd_ci": ci_of(fld("cloud_cost_usd")),
            "obfuscation_overhead_pct_ci": ci_of([np.mean(r.tier2.obfuscation_overhead_pct) for r in runs if r.tier2.obfuscation_overhead_pct]),
            "fidelity_ci": ci_of([np.mean(r.tier2.fidelity) for r in runs if r.tier2.fidelity]),
            "recon_mse_ci": ci_of([np.mean(r.tier2.recon_mse) for r in runs if r.tier2.recon_mse]),
            "mi_leakage_bits_ci": ci_of([np.mean(r.tier2.mi_leakage_bits) for r in runs if r.tier2.mi_leakage_bits]),
            "cut_count_ci": ci_of([np.mean(r.tier2.cut_count) for r in runs if r.tier2.cut_count]),
            "fragment_count_ci": ci_of([np.mean(r.tier2.fragment_count) for r in runs if r.tier2.fragment_count]),
            "shots_used_ci": ci_of([np.mean(r.tier2.shots_used) for r in runs if r.tier2.shots_used]),
        }
        out.append(row)

    print(json.dumps(out, indent=2))

    # Optional per-run CSV with flat means for list metrics
    if args.csv_out:
        def flatten_run(run):
            t2 = run.tier2
            return {
                "scenario": run.config.scenario,
                "seed": run.config.seed,
                "version": run.config.version,
                "hardware_profile": run.config.hardware_profile,
                "started_at": run.start_time,
                "ended_at": run.end_time,
                "duration_s": (run.end_time - run.start_time) if run.end_time else None,
                "n_requests": t2.n_requests,
                "n_success": t2.n_success,
                "n_timeout": t2.n_timeout,
                "n_error": t2.n_error,
                "p50_latency_ms": t2.p50_latency_ms,
                "p95_latency_ms": t2.p95_latency_ms,
                "p99_latency_ms": t2.p99_latency_ms,
                "jitter_ms": t2.jitter_ms,
                "ttfr_ms": t2.ttfr_ms,
                "throughput_qps": t2.throughput_qps,
                "success_throughput_qps": t2.success_throughput_qps,
                "emulated_throughput_qps": t2.emulated_throughput_qps,
                "success_rate": t2.success_rate,
                "timeout_rate": t2.timeout_rate,
                "error_rate": t2.error_rate,
                "mem_peak_mb_max": t2.mem_peak_mb_max,
                "energy_j_total": t2.energy_j_total,
                "cloud_cost_usd": t2.cloud_cost_usd,
                "obfuscation_overhead_pct_mean": float(np.mean(t2.obfuscation_overhead_pct)) if t2.obfuscation_overhead_pct else None,
                "fidelity_mean": float(np.mean(t2.fidelity)) if t2.fidelity else None,
                "recon_mse_mean": float(np.mean(t2.recon_mse)) if t2.recon_mse else None,
                "mi_leakage_bits_mean": float(np.mean(t2.mi_leakage_bits)) if t2.mi_leakage_bits else None,
                "cut_count_mean": float(np.mean(t2.cut_count)) if t2.cut_count else None,
                "fragment_count_mean": float(np.mean(t2.fragment_count)) if t2.fragment_count else None,
                "shots_used_mean": float(np.mean(t2.shots_used)) if t2.shots_used else None,
            }
        rows = [flatten_run(r) for r in em.runs]
        header = list(rows[0].keys()) if rows else []
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=header)
            w.writeheader()
            for row in rows:
                w.writerow(row)

if __name__ == "__main__":
    main()
