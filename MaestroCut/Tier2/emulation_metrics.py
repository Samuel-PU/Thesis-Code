"""
emulation_metrics.py
--------------------
Tier-2 emulation metrics: structures, helpers, and JSONL I/O.

- EmulationConfig / Tier1Metrics / Tier2Metrics / RunMetrics dataclasses
- ExperimentMetrics orchestrator with context-manager run()
- Bootstrap CIs and scenario-level summaries
- Legacy Tier-1 dict-style initializer for back-compat (init_metrics)

Requires: numpy (standard scientific dep; install if missing)
"""

import time, json, random
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Any, Optional, Iterable, Tuple, Callable
import numpy as np
from contextlib import contextmanager

# --------- Config ----------
K_LIST = [1, 3, 5, 10]

# --------- Utilities ----------
def _to_native(x):
    """Make JSON-safe: convert numpy scalars/arrays recursively."""
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, dict):
        return {str(k): _to_native(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_native(v) for v in x]
    return x

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # optional; ignore if unavailable
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except Exception:
        pass

def bootstrap_ci(values: Iterable[float], n_bootstraps: int = 1000, alpha: float = 0.05,
                 reducer: Callable[[np.ndarray], float] = np.mean) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Return (lo, mid, hi) bootstrap CI for a list-like of numbers. Empty -> (None,None,None)."""
    vals = [v for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) == 0:
        return (None, None, None)
    if len(vals) == 1:
        v = float(vals[0])
        return (v, v, v)
    arr = np.asarray(vals, dtype=float)
    boots = np.empty(n_bootstraps, dtype=float)
    n = arr.size
    for i in range(n_bootstraps):
        boots[i] = reducer(arr[np.random.randint(0, n, size=n)])
    lo = float(np.percentile(boots, 100 * (alpha / 2)))
    mid = float(reducer(arr))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (lo, mid, hi)

# --------- Data classes ----------
@dataclass
class EmulationConfig:
    scenario: str              # e.g., "baseline", "noisy", "bursty", "adversarial"
    seed: int
    version: str               # emulator/version tag
    hardware_profile: str
    workload_mix: Dict[str, float]  # proportions per query type
    notes: str = ""            # freeform

@dataclass
class Tier1Metrics:
    P: Dict[int, List[float]] = field(default_factory=dict)
    R: Dict[int, List[float]] = field(default_factory=dict)
    MAP: Dict[int, List[float]] = field(default_factory=dict)
    NDCG: Dict[int, List[float]] = field(default_factory=dict)

    def __post_init__(self):
        for k in K_LIST:
            self.P.setdefault(k, [])
            self.R.setdefault(k, [])
            self.MAP.setdefault(k, [])
            self.NDCG.setdefault(k, [])

    def to_jsonable(self):
        # Convert int keys -> str to preserve on load
        return {
            "P": {str(k): v for k, v in self.P.items()},
            "R": {str(k): v for k, v in self.R.items()},
            "MAP": {str(k): v for k, v in self.MAP.items()},
            "NDCG": {str(k): v for k, v in self.NDCG.items()},
        }

@dataclass
class Tier2Metrics:
    # Per-request logs
    latency_ms: List[float] = field(default_factory=list)
    queue_wait_ms: List[float] = field(default_factory=list)
    service_time_ms: List[float] = field(default_factory=list)
    retry_count: List[int] = field(default_factory=list)

    # Counters
    n_requests: int = 0
    n_success: int = 0
    n_timeout: int = 0
    n_error: int = 0

    # Resources (samples during run; optional)
    cpu_s: List[float] = field(default_factory=list)
    gpu_s: List[float] = field(default_factory=list)
    mem_samples_mb: List[float] = field(default_factory=list)
    energy_j_samples: List[float] = field(default_factory=list)

    # Domain-specific (extend as needed)
    shots_used: List[int] = field(default_factory=list)
    fragment_count: List[int] = field(default_factory=list)
    cut_count: List[int] = field(default_factory=list)
    fidelity: List[float] = field(default_factory=list)
    recon_mse: List[float] = field(default_factory=list)
    mi_leakage_bits: List[float] = field(default_factory=list)
    obfuscation_overhead_pct: List[float] = field(default_factory=list)

    # Derived (filled at end)
    p50_latency_ms: Optional[float] = None
    p95_latency_ms: Optional[float] = None
    p99_latency_ms: Optional[float] = None
    jitter_ms: Optional[float] = None
    ttfr_ms: Optional[float] = None
    throughput_qps: Optional[float] = None
    success_throughput_qps: Optional[float] = None
    emulated_throughput_qps: Optional[float] = None
    success_rate: Optional[float] = None
    timeout_rate: Optional[float] = None
    error_rate: Optional[float] = None
    mem_peak_mb_max: Optional[float] = None
    energy_j_total: Optional[float] = None
    cloud_cost_usd: Optional[float] = None  # fill from your billing model

    def log_request(self, *, latency_ms: float, queue_ms: float = 0.0,
                    service_ms: float = None, retries: int = 0,
                    outcome: str = "success"):
        # Basic validation
        if latency_ms < 0:
            raise ValueError("latency_ms cannot be negative")
        if queue_ms < 0:
            raise ValueError("queue_ms cannot be negative")
        if service_ms is not None and service_ms < 0:
            raise ValueError("service_ms cannot be negative")
        if outcome not in {"success", "timeout", "error"}:
            raise ValueError("outcome must be one of {'success','timeout','error'}")

        self.latency_ms.append(float(latency_ms))
        self.queue_wait_ms.append(float(queue_ms))
        self.service_time_ms.append(float(service_ms if service_ms is not None else max(0.0, latency_ms - queue_ms)))
        self.retry_count.append(int(retries))
        self.n_requests += 1
        if outcome == "success":
            self.n_success += 1
        elif outcome == "timeout":
            self.n_timeout += 1
        else:
            self.n_error += 1

    def compute_derived_metrics(self, duration_s: float):
        if self.latency_ms:
            v = np.asarray(self.latency_ms, dtype=float)
            self.p50_latency_ms = float(np.percentile(v, 50))
            self.p95_latency_ms = float(np.percentile(v, 95))
            self.p99_latency_ms = float(np.percentile(v, 99))
            self.jitter_ms = float(np.std(v))
            self.ttfr_ms = float(self.latency_ms[0])
            total_latency_s = float(np.sum(v) / 1000.0)
            self.emulated_throughput_qps = float(self.n_requests / max(total_latency_s, 1e-9))
        else:
            self.emulated_throughput_qps = None
        # Wall-clock throughput + success throughput
        self.throughput_qps = float(len(self.latency_ms) / max(duration_s, 1e-9))
        self.success_throughput_qps = float(self.n_success / max(duration_s, 1e-9))
        denom = max(self.n_requests, 1)
        self.success_rate = self.n_success / denom
        self.timeout_rate = self.n_timeout / denom
        self.error_rate = self.n_error / denom
        self.mem_peak_mb_max = float(np.max(self.mem_samples_mb)) if self.mem_samples_mb else None
        self.energy_j_total = float(np.sum(self.energy_j_samples)) if self.energy_j_samples else None

@dataclass
class RunMetrics:
    config: EmulationConfig
    tier1: Tier1Metrics
    tier2: Tier2Metrics
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def complete_run(self):
        self.end_time = time.time()
        self.tier2.compute_derived_metrics(duration_s=self.end_time - self.start_time)

    def to_dict(self):
        payload = {
            "config": asdict(self.config),
            "tier1": self.tier1.to_jsonable(),
            "tier2": asdict(self.tier2),
            "duration_s": (self.end_time - self.start_time) if self.end_time else None,
            "started_at": self.start_time,
            "ended_at": self.end_time,
        }
        return _to_native(payload)

class ExperimentMetrics:
    def __init__(self):
        self.runs: List[RunMetrics] = []
        self.current_run: Optional[RunMetrics] = None

    # ---- Lifecycle ----
    def start_run(self, config: EmulationConfig) -> RunMetrics:
        seed_all(config.seed)
        self.current_run = RunMetrics(config=config, tier1=Tier1Metrics(), tier2=Tier2Metrics())
        return self.current_run

    def end_run(self):
        if self.current_run is not None:
            self.current_run.complete_run()
            self.runs.append(self.current_run)
            self.current_run = None

    @contextmanager
    def run(self, config: EmulationConfig):
        rm = self.start_run(config)
        try:
            yield rm
        finally:
            self.end_run()

    # ---- Recording helpers ----
    def record_effectiveness(self, *, k: int, P=None, R=None, MAP=None, NDCG=None):
        if self.current_run is None: raise RuntimeError("No active run")
        if k not in self.current_run.tier1.P:  # allow ad-hoc k
            self.current_run.tier1.P[k] = []
            self.current_run.tier1.R[k] = []
            self.current_run.tier1.MAP[k] = []
            self.current_run.tier1.NDCG[k] = []
        if P is not None: self.current_run.tier1.P[k].append(float(P))
        if R is not None: self.current_run.tier1.R[k].append(float(R))
        if MAP is not None: self.current_run.tier1.MAP[k].append(float(MAP))
        if NDCG is not None: self.current_run.tier1.NDCG[k].append(float(NDCG))

    def log_request(self, **kwargs):
        if self.current_run is None: raise RuntimeError("No active run")
        self.current_run.tier2.log_request(**kwargs)

    # ---- Export/Import ----
    def to_jsonl(self, filepath: str):
        with open(filepath, "w", encoding="utf-8") as f:
            for run in self.runs:
                f.write(json.dumps(run.to_dict(), ensure_ascii=False) + "\n")

    @staticmethod
    def from_jsonl(filepath: str) -> "ExperimentMetrics":
        em = ExperimentMetrics()
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                cfg = EmulationConfig(**obj["config"])
                # Recreate tier1 with str->int keys
                t1 = Tier1Metrics()
                for k_str, vals in obj["tier1"]["P"].items(): t1.P[int(k_str)] = vals
                for k_str, vals in obj["tier1"]["R"].items(): t1.R[int(k_str)] = vals
                for k_str, vals in obj["tier1"]["MAP"].items(): t1.MAP[int(k_str)] = vals
                for k_str, vals in obj["tier1"]["NDCG"].items(): t1.NDCG[int(k_str)] = vals
                # Only keep recognised fields of Tier2Metrics
                t2_fields = set(Tier2Metrics.__dataclass_fields__.keys())
                t2_payload = {k: v for k, v in obj["tier2"].items() if k in t2_fields}
                t2 = Tier2Metrics(**t2_payload)
                rm = RunMetrics(config=cfg, tier1=t1, tier2=t2, start_time=obj.get("started_at", time.time()))
                rm.end_time = obj.get("ended_at", None)
                # ensure derived fields exist if missing
                if rm.end_time is not None:
                    rm.tier2.compute_derived_metrics(duration_s=(rm.end_time - rm.start_time))
                em.runs.append(rm)
        return em

    # ---- Analysis ----
    def efficiency_score(self, k: int, run_index: int = -1, use: str = "p50") -> Optional[float]:
        """NDCG@k per second using p50/p95/p99/mean latency."""
        if not self.runs: return None
        run = self.runs[run_index]
        ndcg_vals = run.tier1.NDCG.get(k, [])
        if not ndcg_vals: return None
        ndcg = float(np.mean(ndcg_vals))
        # choose latency statistic
        if use == "p95": lat_ms = run.tier2.p95_latency_ms
        elif use == "p99": lat_ms = run.tier2.p99_latency_ms
        elif use == "mean": lat_ms = float(np.mean(run.tier2.latency_ms)) if run.tier2.latency_ms else None
        else: lat_ms = run.tier2.p50_latency_ms
        if not lat_ms or lat_ms <= 0: return None
        return ndcg / (lat_ms / 1000.0)

    def summarize_by_scenario(self) -> List[Dict[str, Any]]:
        """Return a compact table of scenario-level aggregates with CIs."""
        out = []
        scenarios = sorted({r.config.scenario for r in self.runs})
        for s in scenarios:
            runs = [r for r in self.runs if r.config.scenario == s]
            # Helper to pull a field across runs safely
            def fld(name):
                vals = []
                for r in runs:
                    v = getattr(r.tier2, name, None)
                    if v is not None:
                        vals.append(v)
                return vals
            row = {
                "scenario": s,
                "runs": len(runs),
                "lat_p50_ms_ci": bootstrap_ci(fld("p50_latency_ms")),
                "lat_p95_ms_ci": bootstrap_ci(fld("p95_latency_ms")),
                "lat_p99_ms_ci": bootstrap_ci(fld("p99_latency_ms")),
                "jitter_ms_ci": bootstrap_ci(fld("jitter_ms")),
                "ttfr_ms_ci": bootstrap_ci(fld("ttfr_ms")),
                "throughput_qps_ci": bootstrap_ci(fld("throughput_qps")),
                "success_throughput_qps_ci": bootstrap_ci(fld("success_throughput_qps")),
                "emulated_throughput_qps_ci": bootstrap_ci(fld("emulated_throughput_qps")),
                "success_rate_ci": bootstrap_ci(fld("success_rate")),
                "timeout_rate_ci": bootstrap_ci(fld("timeout_rate")),
                "error_rate_ci": bootstrap_ci(fld("error_rate")),
                "mem_peak_mb_max_ci": bootstrap_ci(fld("mem_peak_mb_max")),
                "energy_j_total_ci": bootstrap_ci(fld("energy_j_total")),
                "cloud_cost_usd_ci": bootstrap_ci(fld("cloud_cost_usd")),
                "obfuscation_overhead_pct_ci": bootstrap_ci([np.mean(r.tier2.obfuscation_overhead_pct) for r in runs if r.tier2.obfuscation_overhead_pct]),
                "fidelity_ci": bootstrap_ci([np.mean(r.tier2.fidelity) for r in runs if r.tier2.fidelity]),
                "recon_mse_ci": bootstrap_ci([np.mean(r.tier2.recon_mse) for r in runs if r.tier2.recon_mse]),
                "mi_leakage_bits_ci": bootstrap_ci([np.mean(r.tier2.mi_leakage_bits) for r in runs if r.tier2.mi_leakage_bits]),
                "cut_count_ci": bootstrap_ci([np.mean(r.tier2.cut_count) for r in runs if r.tier2.cut_count]),
                "fragment_count_ci": bootstrap_ci([np.mean(r.tier2.fragment_count) for r in runs if r.tier2.fragment_count]),
                "shots_used_ci": bootstrap_ci([np.mean(r.tier2.shots_used) for r in runs if r.tier2.shots_used]),
            }
            out.append(_to_native(row))
        return out

# --------- Legacy Tier-1 dict initializer (back-compat) ----------
def init_metrics():
    m = {}
    for k in K_LIST:
        m[f"P@{k}"] = []
        m[f"R@{k}"] = []
        m[f"MAP@{k}"] = []
        m[f"NDCG@{k}"] = []
    return m

# --------- Optional quick demo ----------
if __name__ == "__main__":
    exp = ExperimentMetrics()
    cfg = EmulationConfig(
        scenario="noisy",
        seed=42,
        version="1.0",
        hardware_profile="gpu_high_mem",
        workload_mix={"A": 0.7, "B": 0.3},
    )
    with exp.run(cfg):
        for lat in [150.5, 210.3, 175.0]:
            exp.log_request(latency_ms=lat, queue_ms=25.0, retries=0, outcome="success")
        exp.record_effectiveness(k=3, NDCG=0.92)
        exp.record_effectiveness(k=3, NDCG=0.88)
    exp.to_jsonl("metrics.jsonl")
    print(json.dumps(exp.summarize_by_scenario(), indent=2))
