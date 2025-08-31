# components.py — NADGO Tier-2: core components & dataclasses

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

LOG2E = 1.0 / math.log(2.0)  # nats → bits conversion factor


# ------------------------ NADGO Components ------------------------
class TDesignPadder:
    """Hardware-aware t-design padding with Clifford scaffolds (sim-level).

    Adds O(n log n) dummy depth with mild randomness; independent of emulation.
    """

    def __init__(self, t: int = 4, epsilon_des: float = 0.02, overhead_coeff: float = 0.5):
        self.t = int(t)
        self.epsilon_des = float(epsilon_des)
        self.k = float(overhead_coeff)  # scales O(n log n) dummy depth

    def pad_segment(self, seg: dict, n_qubits: int, rng: np.random.Generator) -> dict:
        """Simulate padding by increasing logical depth: O(n log n) layers."""
        seg2 = seg.copy()
        base = max(1, int(self.k * n_qubits * max(1, math.ceil(math.log2(max(2, n_qubits))))))
        jitter = int(rng.integers(0, max(1, base // 5)))
        extra_depth = (base + jitter) * max(1, self.t // 2)
        seg2["depth"] = seg["depth"] + extra_depth
        seg2["t"] = self.t
        seg2["epsilon_des"] = self.epsilon_des
        seg2["pad_extra_depth"] = extra_depth
        return seg2


class ParticleFilterScheduler:
    """Drift-adaptive timing randomisation with a simple particle filter.

    Emulation notes:
      - queue_state ∈ [0,1]; higher backlog narrows jitter (weights down large |particles|).
      - Output delay is clamped to [0, l_max] to avoid pathological negative/huge delays.
    """

    def __init__(self, sigma_t: float = 0.8, l_max: float = 10.0):
        self.sigma_t = float(sigma_t)  # jitter scale (ms)
        self.l_max = float(l_max)      # hard cap on absolute delay (ms)
        self.particles: Optional[np.ndarray] = None
        self.last_theta: float = 0.0

    def propose_timing(self, rng: np.random.Generator, queue_state: float) -> float:
        """Propose next dispatch delay (>=0). Higher backlog → narrower jitter."""
        if self.particles is None:
            self.particles = rng.normal(0.0, self.sigma_t / 2.0, 64)

        # Importance weights favour small |particle| when queue is busy
        weights = np.exp(-float(queue_state) * np.abs(self.particles))

        # Robustify against NaN/inf or degenerate sums
        if (not np.isfinite(weights).all()) or (not np.isfinite(weights.sum())) or (weights.sum() <= 0):
            weights = np.ones_like(self.particles) / len(self.particles)
        else:
            weights = weights / weights.sum()

        new_particles = rng.choice(self.particles, size=64, p=weights)
        # Small diffusion noise; keep within ±sigma_t
        new_particles = new_particles + rng.normal(0.0, self.sigma_t / 10.0, 64)
        new_particles = np.clip(new_particles, -self.sigma_t, self.sigma_t)
        self.particles = new_particles

        theta = float(np.median(new_particles))
        # Enforce non-negative delay and l_max cap
        theta = min(max(0.0, theta), self.l_max)
        self.last_theta = theta
        return theta


class CASQUERouter:
    """Topology-aware routing with fidelity–leakage–queue tradeoff (Tier-2 friendly).

    Cost(b) = α * noise(b) + β * risk_ema(b) + queue_coef * queue_len(b) + topo_coef * topo_penalty(b)

    Where:
      - risk_ema is an EMA of a bounded recency metric (higher when used very recently),
        gently scaled by popularity and hard-clamped to avoid runaway costs.
      - topo_penalty(b) is optional (float in backend dict), allowing emulation to bias
        against certain backends due to topology/placement constraints.
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.7,
        queue_coef: float = 0.25,
        ema_decay: float = 0.85,
        popularity_coef: float = 0.10,
        risk_cap: float = 5.0,
        topo_coef: float = 1.0,
    ):
        self.alpha = float(alpha)              # fidelity weight
        self.beta = float(beta)                # leakage risk weight
        self.queue_coef = float(queue_coef)
        self.ema_decay = float(ema_decay)
        self.popularity_coef = float(popularity_coef)
        self.risk_cap = float(risk_cap)
        self.topo_coef = float(topo_coef)

        self.history: Dict[int, List[float]] = {}   # backend_id -> [use_times]
        self.ema_load: Dict[int, float] = {}        # backend_id -> EMA of recency

    def route_segment(self, segment_id: int, backend_options: List[dict], current_time: float) -> dict:
        """Select backend using smoothed, clamped recency-risk + optional topology penalty."""
        # Init maps
        if not hasattr(self, "history") or self.history is None:
            self.history = {}
        if not hasattr(self, "ema_load") or self.ema_load is None:
            self.ema_load = {}
        for b in backend_options:
            self.history.setdefault(b["id"], [])
            self.ema_load.setdefault(b["id"], 0.0)

        costs: List[Tuple[float, dict]] = []
        for backend in backend_options:
            bid = backend["id"]
            noise = float(backend.get("noise_level", 0.0))

            # Recency: bounded ~O(10) when dt→0; 0 when never used
            if self.history[bid]:
                last_use = float(self.history[bid][-1])
                dt = max(1e-3, current_time - last_use)
                inst = 1.0 / (0.1 + dt)
            else:
                inst = 0.0

            # Smooth and clamp risk
            ema_prev = float(self.ema_load.get(bid, 0.0))
            ema = self.ema_decay * ema_prev + (1.0 - self.ema_decay) * inst
            self.ema_load[bid] = ema

            # Scale gently with total uses to reflect popularity; clamp overall
            popularity = 1.0 + self.popularity_coef * len(self.history[bid])
            risk = min(ema * popularity, self.risk_cap)

            queue_penalty = float(len(backend.get("queue", []))) * self.queue_coef

            # Optional emulation hook: topology/placement penalty
            topo_penalty = float(backend.get("topology_penalty", 0.0)) * self.topo_coef

            cost = (self.alpha * noise) + (self.beta * risk) + queue_penalty + topo_penalty
            costs.append((cost, backend))

        _, best_backend = min(costs, key=lambda x: x[0])
        self.history[best_backend["id"]].append(current_time)
        return best_backend


class LeakageMonitor:
    """Per-interval leakage monitor with hysteretic kill-switch.

    Δ̂_t = KL_bits(p̂ || q_ref) + β
      - Dirichlet(α=1) smoothing keeps p̂ strictly > 0.
      - Idle intervals (no features) return β.
      - Hysteresis: ABORT requires a kill-level interval and sufficient strikes.
    """

    def __init__(self, delta_budget: float, delta_kill: float, beta: float = 0.1,
                 kill_strikes: int = 2, warn_cooldown: int = 1):
        self.delta_budget = float(delta_budget)
        self.delta_kill = float(delta_kill)
        self.beta = float(beta)
        self.kill_strikes = int(kill_strikes)
        self.warn_cooldown = int(warn_cooldown)

        self.feature_alphabet = self._create_feature_alphabet()
        self.reference_dist = self._create_reference_distribution()
        self.audit_log: List[dict] = []

        # hysteresis state
        self._strikes = 0
        self._cooldown = 0

    # Coarser features + longer horizon bins (Tier-2 compatible)
    def _create_feature_alphabet(self) -> List[str]:
        intervals = [f"int_{b:.1f}" for b in np.arange(0.0, 5.0, 0.5)]  # 0.5ms bins up to 4.5
        backlogs = ["low", "high"]
        batches = ["small", "medium", "large"]
        events = ["calib", "none"]
        alphabet: List[str] = []
        for i in intervals:
            for b in backlogs:
                for bat in batches:
                    for e in events:
                        alphabet.append(f"{i}|{b}|{bat}|{e}")
        return alphabet

    def _create_reference_distribution(self) -> Dict[str, float]:
        size = len(self.feature_alphabet)
        return {f: 1.0 / size for f in self.feature_alphabet}

    # Robust q_ref loading (floor + renormalise once)
    def load_qref(self, qref: Dict[str, float]) -> None:
        """Load an empirical baseline distribution (smoothed & normalised, alphabet-aligned)."""
        if not qref:
            return
        eps = 1e-9
        tmp = {k: max(eps, float(qref.get(k, 0.0))) for k in self.feature_alphabet}
        s = sum(tmp.values())
        if s > 0:
            self.reference_dist = {k: tmp[k] / s for k in self.feature_alphabet}

    def extract_features(self, event: dict) -> str:
        di = float(event["dispatch_interval"])
        binned = math.floor(min(4.9, max(0.0, di)) / 0.5) * 0.5
        interval_bin = f"int_{binned:.1f}"

        qb = int(event["queue_backlog"])
        backlog = "low" if qb < 3 else "high"

        bs = int(event["batch_size"])
        shots = int(event.get("shots", max(1, bs)))  # baseline shots per segment
        ratio = bs / max(1, shots)
        batch_size = "small" if ratio < 0.7 else ("large" if ratio > 1.4 else "medium")

        event_type = "calib" if bool(event["calibration_event"]) else "none"
        return f"{interval_bin}|{backlog}|{batch_size}|{event_type}"

    def delta_estimate(self, features: List[str]) -> float:
        """Compute Δ̂_t in bits: KL(p̂ || q_ref)/ln2 + β."""
        total = len(features)

        # Treat idle intervals (no features) as baseline β only
        if total == 0:
            return float(self.beta)

        counts = {f: 0 for f in self.feature_alphabet}
        for f in features:
            if f in counts:
                counts[f] += 1

        # Dirichlet smoothing α=1 keeps p̂ strictly > 0
        alpha = 1.0
        denom = total + alpha * len(counts)
        p_hat = {f: (counts[f] + alpha) / denom for f in counts}

        kl_nats = 0.0
        for f in self.feature_alphabet:
            p = p_hat[f]
            q = self.reference_dist[f]  # already floored & normalised
            kl_nats += p * math.log(p / q)

        kl_bits = kl_nats * LOG2E
        return float(kl_bits + self.beta)

    def check_policy(self, delta_est: float, interval: int) -> dict:
        """Hysteretic policy: ABORT can only occur on a kill-level interval."""
        kill_now = bool(delta_est >= self.delta_kill)
        warn_now = (not kill_now) and bool(delta_est > self.delta_budget)

        if kill_now:
            self._strikes += 1
            self._cooldown = self.warn_cooldown
            decision = "ABORT" if self._strikes >= self.kill_strikes else "WARNING"
        elif warn_now:
            decision = "WARNING"
            self._cooldown = max(self._cooldown - 1, 0)
            self._strikes = max(0, self._strikes - 1)
        else:
            decision = "CONTINUE"
            if self._cooldown > 0:
                self._cooldown -= 1
                self._strikes = max(0, self._strikes - 1)
            else:
                self._strikes = 0

        rec = {
            "interval": int(interval),
            "delta_est": float(delta_est),
            "decision": decision,
            "kill_now": kill_now,  # helps audit invariants
            "warn_now": warn_now,
            "strikes": int(self._strikes),
            "cooldown": int(self._cooldown),
            "timestamp": time.time(),
        }
        self.audit_log.append(rec)
        return rec


# ------------------------ Dataclasses ------------------------
@dataclass
class TimingEvent:
    job_id: int
    segment_id: int
    dispatch_time: float
    dispatch_interval: float
    queue_backlog: int
    execution_time: float
    batch_size: int
    calibration_event: bool
    backend_id: int
    power_factor: float
    shots: int                # baseline shots for ratio features
    completion_time: float    # used to compute sojourn latency
    system_backlog: float     # avg backlog across all backends at dispatch


@dataclass
class SegmentResult:
    seed: int
    attack: str
    n: int
    interval: int
    leak_bits: float
    policy_action: str
    latency: float
    power: float


@dataclass
class EpisodeResult:
    seed: int
    attack: str
    n: int
    total_leak: float
    max_leak: float
    aborted: bool
    mean_fidelity: float
    mean_latency: float
    mean_power: float
    max_queue_any: int  # maximum in-flight queue length across any backend


__all__ = [
    "LOG2E",
    "TDesignPadder",
    "ParticleFilterScheduler",
    "CASQUERouter",
    "LeakageMonitor",
    "TimingEvent",
    "SegmentResult",
    "EpisodeResult",
]
