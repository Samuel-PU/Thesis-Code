# simulator.py — NADGO Tier-1: simulator & helper utilities

import math
from collections import deque
from dataclasses import asdict
from typing import Counter as CounterType, Dict, List, Tuple

import numpy as np

from components import (
    TDesignPadder,
    ParticleFilterScheduler,
    CASQUERouter,
    LeakageMonitor,
    TimingEvent,
    SegmentResult,
    EpisodeResult,
)

# --- Helper: build q_ref from observed feature counts -------------------------
def build_qref_from_counts(counts: CounterType, alphabet: List[str], alpha: float = 1.0) -> Dict[str, float]:
    smoothed = {f: float(counts.get(f, 0.0)) + alpha for f in alphabet}
    s = sum(smoothed.values())
    return {k: v / s for k, v in smoothed.items()}

# --- Helper: ensure Δ_kill > Δ_budget with a sensible margin ------------------
def enforce_threshold_order(budget: float, kill: float,
                            min_gap_frac: float = 0.05,
                            min_gap_abs: float = 5e-3) -> Tuple[float, float]:
    """
    Ensure Δ_kill is at least Δ_budget + gap, where gap scales with the magnitude of Δ_budget.

    - min_gap_frac is relative to Δ_budget (5% of budget by default).
    - min_gap_abs is an absolute floor in bits (default 0.005 bits).
    """
    import math
    # Replace NaNs/infs with safe defaults
    if not (math.isfinite(budget)):
        budget = 1.5
    if not (math.isfinite(kill)):
        kill = budget

    # Gap scales with budget (not with max(1.0, budget)), so small budgets don't get huge absolute gaps
    gap = max(min_gap_abs, min_gap_frac * max(budget, 1e-9))

    # Enforce gap
    if kill < budget + gap:
        kill = budget + gap

    return float(budget), float(kill)


class NADGOSimulator:
    def __init__(self, n_qubits: int, delta_budget: float = 0.04, delta_kill: float = 0.07,
                 shots: int = 2048, t_value: int = 4, epsilon_des: float = 0.02,
                 sigma_t: float = 0.8, router_alpha: float = 0.5, router_beta: float = 0.7,
                 queue_coef: float = 0.25, interval_ms: float = 5.0,
                 kill_strikes: int = 2, warn_cooldown: int = 1,
                 monitor_beta: float = 0.1):
        assert delta_kill > delta_budget, "Kill threshold must exceed budget"
        self.n = int(n_qubits)
        self.delta_budget = float(delta_budget)
        self.delta_kill = float(delta_kill)
        self.shots = int(shots)
        self.backends = self.create_backends()
        self.padder = TDesignPadder(t=t_value, epsilon_des=epsilon_des)
        self.scheduler = ParticleFilterScheduler(sigma_t=sigma_t, l_max=5.0)
        self.router = CASQUERouter(alpha=router_alpha, beta=router_beta, queue_coef=queue_coef)
        self.monitor = LeakageMonitor(self.delta_budget, self.delta_kill, beta=float(monitor_beta),
                                      kill_strikes=kill_strikes, warn_cooldown=warn_cooldown)

        self.time = 0.0
        self.interval_length = float(interval_ms)  # ms intervals
        self.audit_log: List[dict] = []
        from collections import Counter
        self.feature_counts: Counter = Counter()   # for calibration
        self.max_queue_any: int = 0                # episode-level queue bound

    def create_backends(self) -> List[dict]:
        """Heterogeneous backend pool."""
        return [
            {'id': 0, 'noise_level': 0.03, 'queue': deque(), 'power_factor': 1.2},
            {'id': 1, 'noise_level': 0.08, 'queue': deque(), 'power_factor': 1.0},
            {'id': 2, 'noise_level': 0.15, 'queue': deque(), 'power_factor': 0.8}
        ]

    def prune_queues(self, current_time: float) -> None:
        for b in self.backends:
            q = b['queue']
            while q and q[0] <= current_time:
                q.popleft()

    def execute_segment(self, backend: dict, segment: dict) -> float:
        base_time = float(segment['depth']) * 0.05  # "ms" model units
        # batch-size impact (gentle): +0.0002 per shot → ~+0.4 at 2048
        batch_factor = 1.0 + 0.0002 * max(1, segment.get('batch_size', self.shots))
        # NOTE: computed before enqueue; reflects pre-enqueue backlog
        noise_factor = 1.0 + (len(backend['queue']) * 0.1)
        return base_time * batch_factor * noise_factor

    def simulate_episode(self, seed: int, attack_name: str,
                         depth: int = 32, steps: int = 10) -> Tuple[EpisodeResult, List[SegmentResult]]:
        rng = np.random.default_rng(int(seed))
        segment_results: List[SegmentResult] = []
        current_interval = 0
        interval_events: List[TimingEvent] = []
        last_dispatch = self.time
        aborted = False

        # reset per-episode state
        self.max_queue_any = 0
        for backend in self.backends:
            backend['queue'].clear()

        for step in range(int(steps)):
            if aborted:
                break

            # Reap completions up to current time
            self.prune_queues(self.time)

            # Attack perturbs batch size
            batch_size = self.apply_attack(attack_name, rng, step)

            # Padding
            segment = {'depth': int(depth), 'batch_size': int(batch_size), 'calibration_event': (step % 10 == 0)}
            padded_segment = self.padder.pad_segment(segment, self.n, rng)

            # Queue-aware scheduler state
            avg_backlog = float(np.mean([len(b['queue']) for b in self.backends])) if self.backends else 0.0
            queue_state = min(1.0, avg_backlog / 5.0)

            # --- Correct event-time progression (per-backend gating)
            dispatch_delay = self.scheduler.propose_timing(rng, queue_state)
            arrival_time = self.time + dispatch_delay  # time when job is ready to dispatch

            # Route first, then respect the selected backend’s tail, not global earliest completion
            backend = self.router.route_segment(step, self.backends, self.time)
            last_finish = backend['queue'][-1] if backend['queue'] else self.time
            dispatch_time = max(arrival_time, last_finish)
            if dispatch_time < self.time:  # numerical/corner-case guard
                dispatch_time = self.time

            # Execute with pre-enqueue backlog
            pre_backlog = len(backend['queue'])
            exec_time = self.execute_segment(backend, padded_segment)
            completion_time = dispatch_time + exec_time
            backend['queue'].append(completion_time)

            # Track episode-level queue bound across all backends
            self.max_queue_any = max(self.max_queue_any, max(len(b['queue']) for b in self.backends))

            # Compute correct sojourn latency (queue wait + service)
            sojourn_latency = completion_time - arrival_time

            # Record timing event (includes completion for sojourn, and system backlog)
            event = TimingEvent(
                job_id=0, segment_id=step,
                dispatch_time=float(dispatch_time),
                dispatch_interval=float(dispatch_time - last_dispatch),
                queue_backlog=pre_backlog,
                execution_time=float(exec_time),
                batch_size=int(batch_size),
                calibration_event=bool(segment['calibration_event']),
                backend_id=int(backend['id']),
                power_factor=float(backend['power_factor']),
                shots=self.shots,
                completion_time=float(completion_time),   # used to compute sojourn latency
                system_backlog=float(avg_backlog),        # interval/system telemetry
            )
            # Store sojourn latency dynamically (keeps dataclass unchanged)
            setattr(event, "sojourn_latency", float(sojourn_latency))

            last_dispatch = dispatch_time
            self.time = dispatch_time
            interval_events.append(event)

            # --- Multi-interval rollover: flush until caught up
            target_interval = int(self.time // self.interval_length)
            while current_interval <= target_interval - 1:
                # Features & ΔĨ_t for this interval
                feats = [self.monitor.extract_features(asdict(e)) for e in interval_events]
                for f in feats:
                    self.feature_counts[f] += 1
                delta_est = self.monitor.delta_estimate(feats) if feats else self.monitor.delta_estimate([])

                audit_rec = self.monitor.check_policy(delta_est, current_interval)

                # --- Interval telemetry (quantitative bounded-queue evidence)
                mean_q_selected = float(np.mean([e.queue_backlog for e in interval_events])) if interval_events else 0.0
                mean_q_system   = float(np.mean([e.system_backlog for e in interval_events])) if interval_events else 0.0
                max_q_selected  = int(max([e.queue_backlog for e in interval_events])) if interval_events else 0
                audit_rec.update({
                    "mean_queue_selected": mean_q_selected,
                    "mean_system_backlog": mean_q_system,
                    "max_queue_selected":  max_q_selected,
                })

                # Segment-level records (SOJOURN latency)
                for ev in interval_events:
                    seg_result = SegmentResult(
                        seed=int(seed), attack=str(attack_name), n=self.n, interval=current_interval,
                        leak_bits=float(delta_est), policy_action=audit_rec['decision'],
                        latency=float(getattr(ev, "sojourn_latency", ev.completion_time - ev.dispatch_time)),
                        power=float(ev.execution_time * ev.power_factor)
                    )
                    segment_results.append(seg_result)

                if audit_rec['decision'] == 'ABORT':
                    aborted = True
                    self.audit_log.append(audit_rec)
                    break

                interval_events = []
                current_interval += 1
                self.audit_log.append(audit_rec)

            if aborted:
                break

        # Flush final partial interval
        if interval_events:
            feats = [self.monitor.extract_features(asdict(e)) for e in interval_events]
            for f in feats:
                self.feature_counts[f] += 1
            delta_est = self.monitor.delta_estimate(feats)
            audit_rec = self.monitor.check_policy(delta_est, current_interval)

            # Telemetry for the trailing (possibly partial) interval
            mean_q_selected = float(np.mean([e.queue_backlog for e in interval_events])) if interval_events else 0.0
            mean_q_system   = float(np.mean([e.system_backlog for e in interval_events])) if interval_events else 0.0
            max_q_selected  = int(max([e.queue_backlog for e in interval_events])) if interval_events else 0
            audit_rec.update({
                "mean_queue_selected": mean_q_selected,
                "mean_system_backlog": mean_q_system,
                "max_queue_selected":  max_q_selected,
            })

            for ev in interval_events:
                segment_results.append(
                    SegmentResult(seed=int(seed), attack=str(attack_name), n=self.n, interval=current_interval,
                                  leak_bits=float(delta_est), policy_action=audit_rec['decision'],
                                  latency=float(getattr(ev, "sojourn_latency", ev.completion_time - ev.dispatch_time)),
                                  power=float(ev.execution_time * ev.power_factor))
                )
            if audit_rec['decision'] == 'ABORT':
                aborted = True
            self.audit_log.append(audit_rec)

        # Episode aggregates
        leaks = [sr.leak_bits for sr in segment_results]
        latencies = [sr.latency for sr in segment_results]
        powers = [sr.power for sr in segment_results]
        mean_leak = float(np.mean(leaks)) if leaks else 0.0

        ep_result = EpisodeResult(
            seed=int(seed), attack=str(attack_name), n=self.n,
            total_leak=float(np.sum(leaks)) if leaks else 0.0,
            max_leak=float(np.max(leaks)) if leaks else 0.0,
            aborted=bool(aborted),
            mean_fidelity=float(max(0.0, 0.95 - (mean_leak * 0.1))),
            mean_latency=float(np.mean(latencies)) if latencies else 0.0,
            mean_power=float(np.mean(powers)) if powers else 0.0,
            max_queue_any=int(self.max_queue_any),     # per-episode queue bound
        )
        return ep_result, segment_results

    def apply_attack(self, attack_name: str, rng: np.random.Generator, step: int) -> int:
        """Attacks that perturb backlog/timing structure via batch size."""
        if attack_name == "none":
            return self.shots
        elif attack_name == "rl":
            base = self.shots
            swing = int(0.8 * base)
            return max(1, base + int(swing * math.sin(step / 3.0)))
        elif attack_name == "timing":
            return (self.shots * 4) if (step % 4 == 0) else max(1, self.shots // 8)
        else:
            return self.shots


__all__ = [
    "NADGOSimulator",
    "build_qref_from_counts",
    "enforce_threshold_order",
]
