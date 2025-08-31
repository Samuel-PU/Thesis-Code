Tier-1 (Simulation)



Tier-1 provides a \*\*closed-loop simulator\*\* to evaluate allocation strategies under drift.



Run `Tier1\_main.py` to produce logs in `data/tier1/`:



\- `data/tier1/<workload>/frag\_seed<seed>.csv` – per-step fragment allocations and estimator choices.

\- `data/tier1/<workload>/meta\_seed<seed>.json` – configuration metadata.

\- `data/tier1/summary.csv` – aggregated metrics across runs.



Key features:

\- Kalman tracking of latent variances.

\- Topology-aware smoothing (Matern-½ kernel).

\- Water-filling shot allocation with minimum floors.

\- Entropy-pilot estimates for cascade bias.

\- Estimator cascade (MLE vs shadow).

\- PhasePad overhead and decoy shots.

\- CUSUM-based repartition trigger with cut-cost update.



Example:

```bash

python Tier1\_main.py --workload qaoa30 --seed 1



