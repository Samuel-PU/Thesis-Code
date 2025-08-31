#!/usr/bin/env python3
"""
Tier-1 Dynamic Experiment Driver (MaestroCut-aligned)
----------------------------------------------------
• Discrete-time simulator with drift, Kalman tracking, Topo-GP (topology-smoothed)
  water-filling, variance-aware estimator cascade, optional PhasePad overhead,
  and CUSUM trigger.
• Streams per-step fragment logs to data/tier1/<workload>/frag_seed<seed>.csv
  and aggregates to data/tier1/summary.csv + a sidecar metadata JSON.

Security & robustness notes
• No dynamic imports or eval; strict argparse; bounded file I/O within ./data/tier1
• Deterministic RNG per seed; explicit integer projection for shot budgets
• Defensive floors and input validation; exceptions fail fast with clear messages

Dependencies: numpy, pandas, scipy, tqdm
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.stats import entropy
from tqdm import tqdm

# ------------------------------- Configuration -------------------------------

WORKLOADS: Dict[str, Dict[str, float]] = {
    "qaoa30":     {"qubits": 30, "depth": 60, "Q": 0.18, "n_fragments": 16},
    "uccsd24":    {"qubits": 24, "depth": 80, "Q": 0.22, "n_fragments": 16},
    "tfim20":     {"qubits": 20, "depth": 45, "Q": 0.25, "n_fragments": 16},
    "clifford24": {"qubits": 24, "depth": 50, "Q": 0.12, "n_fragments": 16},
    "phase16":    {"qubits": 16, "depth": 40, "Q": 0.15, "n_fragments": 16},
}

DEFAULTS = dict(
    batch_size=500,            # shots per control update
    total_shots=300_000,       # total shot budget per (workload, seed)
    drift_sigma=0.03,          # Wiener drift scale for latent variances
    kalman_q=0.003,            # process noise
    kalman_r=0.01,             # measurement noise (ShotQC)
    rho=0.05,                  # (kept for future use)
    smin=4,                    # min shots per fragment per batch (on s_plan)
    phasepad_overhead=0.01,    # +1% time overhead
    decoy_rate=0.02,           # 2% of shots are decoys (not used for estimates)
    ell=0.30,                  # Matern-1/2 length-scale (heavy-hex distance proxy)
    alpha=1.0,                 # (unused—compat)
    beta=50.0,                 # cascade: MLE MSE ≈ (mu*beta)/s^2 + bias^2
)

SAFE_BASE = Path.cwd()  # confine outputs under CWD


# ------------------------------- Utilities -----------------------------------

def _safe_join(base: Path, *parts: str) -> Path:
    p = base
    for part in parts:
        p = p / part
    p = p.resolve()
    if not str(p).startswith(str(base.resolve())):
        raise ValueError("Unsafe path traversal attempt")
    return p


def matern_half(dist: np.ndarray, ell: float) -> np.ndarray:
    return np.exp(-np.asarray(dist, dtype=float) / float(ell))


def pairwise_euclid(xy: np.ndarray) -> np.ndarray:
    x = np.asarray(xy, float)
    d2 = np.sum((x[:, None, :] - x[None, :, :]) ** 2, axis=-1)
    return np.sqrt(d2)


def water_filling(u: np.ndarray, S: int, smin: int) -> np.ndarray:
    """
    Integer water-filling for objective ~ sum_i u_i / s_i with floor and exact sum.
    Handles S < n*smin by relaxing the floor for THIS batch (possibly down to 0).
    """
    u = np.maximum(1e-12, np.asarray(u, float))
    n = u.size
    if n <= 0 or S <= 0:
        return np.zeros(max(n, 0), dtype=int)

    # If requested floor is infeasible, relax it; 0 is allowed here.
    smin_eff = smin
    if S < n * smin_eff:
        smin_eff = max(0, S // n)

    w = np.sqrt(u)
    s_float = S * w / np.sum(w)
    s_float = np.maximum(smin_eff, s_float)

    s_int = np.floor(s_float).astype(int)
    delta = int(S - int(np.sum(s_int)))
    if delta != 0:
        residual = s_float - s_int
        order = np.argsort(-residual)
        step = 1 if delta > 0 else -1
        for idx in order[:abs(delta)]:
            s_int[idx] = max(smin_eff, s_int[idx] + step)

    # Final exactness repair
    diff = int(S - int(np.sum(s_int)))
    k = 0
    while diff != 0 and k < n:
        step = 1 if diff > 0 else -1
        if step > 0 or s_int[k] > smin_eff:
            s_int[k] = max(smin_eff, s_int[k] + step)
            diff = int(S - int(np.sum(s_int)))
        k += 1

    assert int(np.sum(s_int)) == int(S), "Water-filling projection failed to meet budget."
    return s_int


# ----------------------------- Simulation State ------------------------------

@dataclass
class SimCfg:
    name: str
    qubits: int
    depth: int
    Q: float
    n_fragments: int
    seed: int
    batch_size: int = DEFAULTS["batch_size"]
    total_shots: int = DEFAULTS["total_shots"]
    drift_sigma: float = DEFAULTS["drift_sigma"]
    kalman_q: float = DEFAULTS["kalman_q"]
    kalman_r: float = DEFAULTS["kalman_r"]
    rho: float = DEFAULTS["rho"]
    smin: int = DEFAULTS["smin"]
    phasepad_overhead: float = DEFAULTS["phasepad_overhead"]
    decoy_rate: float = DEFAULTS["decoy_rate"]
    ell: float = DEFAULTS["ell"]
    alpha: float = DEFAULTS["alpha"]
    beta: float = DEFAULTS["beta"]


class SimulationState:
    __slots__ = (
        "cfg", "rng", "step", "shots_used", "anchors", "dist", "Sigma",
        "true_var", "kalman_mu", "kalman_P", "entropy_est", "cusum", "cut_cost",
        # rolling stats
        "alloc_history", "mle_count", "shadow_count", "time_accum_model", "peak_mem_gb"
    )

    def __init__(self, cfg: SimCfg):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.step = 0
        self.shots_used = 0

        # random anchors (proxy for heavy-hex coordinates)
        self.anchors = self.rng.random((cfg.n_fragments, 2))
        self.dist = pairwise_euclid(self.anchors)
        self.Sigma = matern_half(self.dist, cfg.ell)

        # latent variances and filters
        self.true_var = self.rng.random(cfg.n_fragments) * 0.4 + 0.1
        self.kalman_mu = self.true_var.copy()
        self.kalman_P = np.ones(cfg.n_fragments) * 0.1
        self.entropy_est = np.zeros(cfg.n_fragments)

        # security/partition signals
        self.cusum = 0.0
        self.cut_cost = 10.0

        # rolling stats
        self.alloc_history: List[np.ndarray] = []   # <- ensure this exists
        self.mle_count = 0
        self.shadow_count = 0
        self.time_accum_model = 0.0  # modelled time (not wallclock)
        self.peak_mem_gb = 0.0

    # ---------------------- Single control-update step ----------------------
    def step_once(self) -> pd.DataFrame:
        cfg = self.cfg
        n = cfg.n_fragments
        B = min(cfg.batch_size, cfg.total_shots - self.shots_used)
        if B <= 0:
            return pd.DataFrame()

        # --- ensure locals exist on all paths ---
        s_plan = np.zeros(n, dtype=int)

        # (1) inject drift (Wiener)
        drift = self.rng.normal(0.0, cfg.drift_sigma, n)
        self.true_var = np.abs(self.true_var + drift)  # keep positive

        # (2) Kalman predict
        prior_mu = self.kalman_mu
        prior_P = self.kalman_P + cfg.kalman_q

        # (3) obtain measurement (ShotQC proxy)
        meas_noise = self.rng.normal(0.0, cfg.kalman_r, n)
        z_t = self.true_var + meas_noise

        # (4) Kalman update
        K = prior_P / (prior_P + cfg.kalman_r)
        self.kalman_mu = prior_mu + K * (z_t - prior_mu)
        self.kalman_P = (1.0 - K) * prior_P

        # (5) entropy estimate via tiny pilot (secure floor at 3 counts)
        pilot = 90
        counts = self.rng.integers(0, 3, size=(n, pilot))
        self.entropy_est = np.array([
            entropy(np.bincount(row, minlength=3) / pilot) for row in counts
        ])

        # (6) Topology-aware allocation: smooth μ via row-normalized Σ, then water-fill
        row_sum = np.maximum(1e-12, np.sum(self.Sigma, axis=1))
        mu_smooth = (self.Sigma @ self.kalman_mu) / row_sum
        s_plan = water_filling(mu_smooth, S=B, smin=cfg.smin)

        # Apply decoys (effective shots used by estimators), never exceed s_plan
        s_eff = np.minimum(s_plan, np.floor(s_plan * (1.0 - cfg.decoy_rate)).astype(int))
        self.alloc_history.append(s_plan.copy())

        # (7) Estimator cascade (heteroskedastic in kalman_mu)
        branches = np.zeros(n, dtype=int)  # 0=shadow, 1=MLE
        mse_used = np.zeros(n)
        var_est = np.zeros(n)

        t_batch = 0.0
        for i in range(n):
            # If decoys zero out an entry, use 1 to keep formulas finite.
            s_i = int(max(1, s_eff[i]))
            H_i = float(self.entropy_est[i])
            V_i = float(self.kalman_mu[i])

            bias_mle = 0.1 * (1.0 - min(1.0, H_i / math.log(3)))
            mse_shadow = V_i / float(s_i)
            mse_mle    = (V_i * cfg.beta) / float(s_i ** 2) + bias_mle ** 2

            if mse_mle <= mse_shadow:
                branches[i] = 1
                mse_used[i] = mse_mle
                var_est[i]  = max(1e-12, V_i) / float(s_i ** 2)
                t_batch += 2.5e-5 * s_i
            else:
                branches[i] = 0
                mse_used[i] = mse_shadow
                var_est[i]  = max(1e-12, V_i) / float(s_i)
                t_batch += 1.0e-5 * s_i

        # (8) PhasePad overhead to time
        t_batch *= (1.0 + cfg.phasepad_overhead)
        self.time_accum_model += t_batch
        self.mle_count += int(np.sum(branches == 1))
        self.shadow_count += int(np.sum(branches == 0))

        # (9) CUSUM drift trigger (on innovation magnitude)
        avg_innov = float(np.mean(np.abs(z_t - prior_mu)))
        kappa, h = 0.02, 0.12
        self.cusum = max(0.0, self.cusum + avg_innov - kappa)
        repartition = False
        if self.cusum > h:
            # simulate a repartition effect by slightly tightening distances
            self.anchors += self.rng.normal(0.0, 0.01, size=self.anchors.shape)
            self.dist = pairwise_euclid(self.anchors)
            self.Sigma = matern_half(self.dist, cfg.ell)
            self.cut_cost *= 0.9
            self.cusum = 0.0
            repartition = True

        # accounting
        self.shots_used += int(np.sum(s_plan))

        # memory tracking (peak since start of run)
        _, peak = tracemalloc.get_traced_memory()
        self.peak_mem_gb = max(self.peak_mem_gb, peak / 1e9)

        # detailed record (one row per fragment)
        df = pd.DataFrame({
            "step": self.step,
            "frag_id": np.arange(n, dtype=int),
            "s_plan": s_plan,
            "s_eff": s_eff,
            "branch": branches,
            "H_i": self.entropy_est,
            "kalman_mu": self.kalman_mu,
            "kalman_P": self.kalman_P,
            "z_t": z_t,
            "mse_used": mse_used,
            "var_est": var_est,
            "avg_innov": avg_innov,
            "repartition": repartition,
        })
        self.step += 1
        return df


# ------------------------------- Main routine --------------------------------

def run_one(cfg: SimCfg, out_dir: Path, force: bool = False) -> Tuple[pd.DataFrame, Dict[str, float]]:
    seed_dir = _safe_join(out_dir, cfg.name)
    seed_dir.mkdir(parents=True, exist_ok=True)
    frag_csv = seed_dir / f"frag_seed{cfg.seed}.csv"

    # stream to CSV: overwrite if --force, else skip if exists
    if frag_csv.exists() and not force:
        return pd.DataFrame(), {}

    tracemalloc.start()
    state = SimulationState(cfg)

    # small histogram for p95 of s_plan (bins up to batch_size)
    max_bin = int(cfg.batch_size)
    shot_hist = np.zeros(max_bin + 1, dtype=np.int64)

    last_df: pd.DataFrame | None = None
    repartitions = 0
    header_written = False

    with tqdm(total=cfg.total_shots, desc=f"{cfg.name}-{cfg.seed}", unit="shot") as pbar:
        # open once, append each step
        with frag_csv.open("w", newline="") as f:
            while state.shots_used < cfg.total_shots:
                df_step = state.step_once()
                if df_step.empty:
                    break

                # update histogram for p95
                s_vals = df_step["s_plan"].to_numpy(int)
                s_vals = np.clip(s_vals, 0, max_bin)
                np.add.at(shot_hist, s_vals, 1)

                # repartition counter
                if bool(df_step["repartition"].any()):
                    repartitions += 1

                # stream write
                df_step.to_csv(f, index=False, header=not header_written)
                header_written = True

                # keep only the last step for end metrics
                last_df = df_step

                pbar.update(int(np.sum(df_step["s_plan"].to_numpy())))

    tracemalloc.stop()

    # if nothing was produced, return empty
    if last_df is None:
        return pd.DataFrame(), {}

    frag_df = pd.DataFrame()  # nothing kept in memory; placeholder for signature

    # -------- summary metrics (from last step + histogram) --------
    # p95 from histogram
    cdf = np.cumsum(shot_hist)
    total_ct = int(cdf[-1]) if cdf.size else 1
    kth = int(math.ceil(0.95 * total_ct))
    p95_idx = int(np.searchsorted(cdf, kth, side="left"))
    p95_shots = float(p95_idx)

    # correlation & summaries from last step
    last = last_df
    corr = float(np.corrcoef(last["kalman_mu"].to_numpy(), last["s_plan"].to_numpy())[0, 1])
    mean_entropy = float(np.mean(last["H_i"].to_numpy()))
    mean_var_est = float(np.mean(last["var_est"].to_numpy()))

    # -------- contraction metric: apples-to-apples (Topo-GP vs Uniform) --------
    def cascade_mse_vec(s_vec: np.ndarray,
                        H_vec: np.ndarray,
                        mu_vec: np.ndarray,
                        beta: float) -> np.ndarray:
        s = np.maximum(1, s_vec.astype(int))
        H = np.asarray(H_vec, float)
        mu = np.maximum(1e-12, np.asarray(mu_vec, float))
        bias_mle = 0.1 * (1.0 - np.minimum(1.0, H / math.log(3)))
        mse_shadow = mu / s
        mse_mle    = (mu * beta) / (s ** 2) + bias_mle ** 2
        return np.minimum(mse_shadow, mse_mle)

    def equal_split_with_floor(S: int, n: int, smin: int) -> np.ndarray:
        base = np.full(n, max(smin, S // n), dtype=int)
        diff = int(S - int(base.sum()))
        if diff > 0:
            base[:diff] += 1
        elif diff < 0:
            k = 0
            need = -diff
            while need > 0 and k < n:
                take = min(need, max(0, base[k] - smin))
                base[k] -= take
                need -= take
                k += 1
        assert int(base.sum()) == int(S), "Uniform split failed to match total shots."
        return base

    # Topo-GP (already decoy-aware and heteroskedastic in last step)
    var_gp = float(np.mean(last["mse_used"].to_numpy(float)))

    # Uniform baseline: same TOTAL pre-decoy shots as last step
    n = last.shape[0]
    S_pre = int(np.sum(last["s_plan"].to_numpy(int)))
    s_unif_plan = equal_split_with_floor(S_pre, n, cfg.smin)
    s_unif_eff  = np.minimum(s_unif_plan,
                             np.floor(s_unif_plan * (1.0 - cfg.decoy_rate)).astype(int))
    var_uniform = float(np.mean(cascade_mse_vec(
        s_unif_eff,
        last["H_i"].to_numpy(float),
        last["kalman_mu"].to_numpy(float),
        cfg.beta
    )))

    metrics = dict(
        var_uniform=var_uniform,
        var_gp=var_gp,
        mse=float(var_gp),
        p95_shots=p95_shots,
        variance_shot_corr=corr,
        peak_mem_gb=state.peak_mem_gb,
        mean_entropy=mean_entropy,
        shadow_count=int(state.shadow_count),
        mle_count=int(state.mle_count),
        mean_var_est=mean_var_est,
        mean_time_est_s=float(state.time_accum_model / max(1, state.step)),
        repartitions=int(repartitions),
        shots_used=int(state.shots_used),
    )

    # metadata sidecar
    sidecar = {
        "workload": dataclasses.asdict(cfg),
        "cut_cost_final": state.cut_cost,
        "allocator": "Topo-GP (row-normalized Σ) + √-rule water-filling (integer)",
        "cascade_model": {"beta": cfg.beta},
        "phasepad": {"overhead": cfg.phasepad_overhead, "decoy_rate": cfg.decoy_rate},
        "kalman": {"q": cfg.kalman_q, "r": cfg.kalman_r},
        "drift_sigma": cfg.drift_sigma,
    }
    (seed_dir / f"meta_seed{cfg.seed}.json").write_text(json.dumps(sidecar, indent=2))

    return frag_df, metrics


# --------------------------------- CLI ---------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Tier-1 dynamic simulator (MaestroCut-aligned)")
    p.add_argument("--workloads", nargs="*", default=list(WORKLOADS.keys()),
                   help=f"subset of workloads to run (default: all) from {list(WORKLOADS.keys())}")
    p.add_argument("--seeds", type=str, default="0..99",
                   help="seed range as A..B (inclusive), e.g., 0..9")
    p.add_argument("--out", type=str, default="data/tier1",
                   help="output base directory (confined under ./data/tier1)")
    p.add_argument("--force", action="store_true", help="overwrite existing per-seed CSVs")
    p.add_argument("--batch", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--shots", type=int, default=DEFAULTS["total_shots"])
    return p.parse_args()


def parse_seed_range(rng: str) -> List[int]:
    if ".." not in rng:
        val = int(rng)
        return [val]
    a, b = rng.split("..", 1)
    a_i, b_i = int(a), int(b)
    if a_i > b_i:
        raise ValueError("seeds: range must be non-decreasing")
    width = b_i - a_i + 1
    if width > 10_000:
        raise ValueError("seeds: range too large")
    return list(range(a_i, b_i + 1))


# --------------------------------- Entry -------------------------------------

def main() -> None:
    args = parse_args()

    base_out = _safe_join(SAFE_BASE, args.out)
    must_prefix = _safe_join(SAFE_BASE, "data", "tier1")
    if not str(base_out).startswith(str(must_prefix)):
        raise SystemExit("Refusing to write outside ./data/tier1 for safety.")
    base_out.mkdir(parents=True, exist_ok=True)

    workloads = []
    for wname in args.workloads:
        if wname not in WORKLOADS:
            raise SystemExit(f"Unknown workload '{wname}'. Allowed: {list(WORKLOADS.keys())}")
        workloads.append(wname)

    seeds = parse_seed_range(args.seeds)

    summary_rows: List[Dict[str, float]] = []
    summary_csv = base_out / "summary.csv"

    for wname in workloads:
        wcfg = WORKLOADS[wname]
        for seed in seeds:
            cfg = SimCfg(
                name=wname,
                qubits=int(wcfg["qubits"]),
                depth=int(wcfg["depth"]),
                Q=float(wcfg["Q"]),
                n_fragments=int(wcfg["n_fragments"]),
                seed=int(seed),
                batch_size=max(1, int(args.batch)),
                total_shots=max(10 * int(wcfg["n_fragments"]), int(args.shots)),
            )
            try:
                _, metrics = run_one(cfg, out_dir=base_out, force=args.force)
                if metrics:
                    summary_rows.append({"workload": wname, "seed": seed, **metrics})
            except AssertionError as e:
                raise SystemExit(f"Allocator error: {e}")
            except Exception as e:
                raise SystemExit(f"Failed for {wname}-{seed}: {e}")

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        if summary_csv.exists():
            df_all = pd.concat([pd.read_csv(summary_csv), df_summary], ignore_index=True)
            df_all.drop_duplicates(subset=["workload", "seed"], inplace=True)
        else:
            df_all = df_summary
        df_all.to_csv(summary_csv, index=False)
        print(f"✓ Tier-1 summary written → {summary_csv}")
    else:
        print("No new rows to write (use --force to overwrite existing seeds)")


if __name__ == "__main__":
    main()
