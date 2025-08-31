MaestroCut — Tiered Simulation \& Emulation Framework



MaestroCut is a research framework for \*\*dynamic circuit cutting\*\* and \*\*secure orchestration\*\* on near-term quantum hardware.  

It provides a two-tier evaluation pipeline:



\- \*\*Tier-1 (Simulation):\*\*  

&nbsp; Closed-loop simulator for drift-aware circuit cutting, Kalman-tracked shot allocation, Topo-GP smoothing, and estimator cascades.  

&nbsp; Produces fine-grained fragment logs and variance contraction studies.



\- \*\*Tier-2 (Emulation):\*\*  

&nbsp; System-level stress testing of workloads under synthetic or adversarial conditions.  

&nbsp; Captures latency, throughput, reliability, confidentiality overhead, and resource/cost trade-offs.



\## Repository Structure



tier1/ # Simulation pipeline

Tier1\_main.py – run simulations (per-workload, per-seed)

Tier1\_Plot.py – generate Tier-1 figures



tier2/ # Emulation pipeline

emulation\_metrics.py – dataclasses, CI helpers, JSONL I/O

emulator\_cli.py – CLI entrypoint to run emulations

metrics\_tier2.py – legacy back-compat helpers

summarise\_metrics.py – aggregate JSONL logs → scenario summaries + flat CSV

tier2\_plotter.py – produce unified Tier-2 summary figure + overhead table



\## Quickstart

\### Tier-1 Simulation

```bash

\# Run a QAOA-30 workload with seed 1

python tier1/Tier1\_main.py --workload qaoa30 --seed 1



\# Generate plots (variance contraction, ablation, tails, coupling, timelines)

python tier1/Tier1\_Plot.py



Outputs:

Logs in data/tier1/<workload>/frag\_seed<seed>.csv

Aggregates in data/tier1/summary.csv

Figures in figs/ (e.g., fig-rq1-contraction.pdf, fig-rq2-timeline.pdf)



Tier-2 Emulation

\# Run 1000 noisy requests with memory + energy sampling

python tier2/emulator\_cli.py \\

&nbsp; --scenario noisy --seed 123 --hardware A100x1 \\

&nbsp; --workload A=0.7,B=0.3 --n-requests 1000 \\

&nbsp; --lat-mean 180 --lat-std 40 --timeout-ms 500 \\

&nbsp; --out runs.jsonl --enable-mem --energy-per-request-j 0.05



\# Summarise into scenario-level stats + per-run flat CSV

python tier2/summarise\_metrics.py --jsonl runs.jsonl --csv-out runs\_flat.csv



\# Create unified Tier-2 summary PDF + overhead table

python tier2/tier2\_plotter.py --csv runs\_flat.csv --out figs/tier2 \\

&nbsp; --jitter-target 150 --ttfr-target 220



Outputs:

runs.jsonl — raw per-run metrics

runs\_flat.csv — flat CSV for analysis

figs/tier2/fig-t2-summary.pdf — latency, throughput, reliability figure

figs/tier2/overhead-summary.csv — per-scenario obfuscation overheads



Dependencies

Python ≥ 3.8

Required: numpy, pandas, matplotlib, seaborn

Optional: scipy, tqdm (Tier-1); psutil (memory sampling), torch (deterministic seeding)



Citation

If you use MaestroCut in your research, please cite:

Punch, S. (2025). Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing. MSc Thesis, University College Cork.





