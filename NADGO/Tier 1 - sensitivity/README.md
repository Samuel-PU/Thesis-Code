\# NADGO — Noise-Adaptive Dummy-Gate Obfuscation (Tier-1 Framework)



\*\*NADGO\*\* is a scheduling \& obfuscation stack for \*\*operational privacy\*\* in quantum clouds.  

It enforces per-interval leakage limits by combining:



\- \*\*t-design padding\*\* — inserts dummy Clifford layers to shape workload entropy.  

\- \*\*Particle-filter timing\*\* — queue-aware jitter randomisation.  

\- \*\*CASQUE routing\*\* — backend selection trading off fidelity, leakage risk, and backlog.  

\- \*\*Leakage monitor\*\* — tracks Δ̂ estimates (bits) with hysteretic WARN/ABORT policy.  



This package implements the \*\*Tier-1 simulator\*\* and analysis tools used in the thesis:



> Punch, S. (2025). \*Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing.\* MSc Thesis, University College Cork.



---



\## Repository Layout





components.py # Core modules: padding, scheduler, router, leakage monitor

simulator.py # NADGO simulator harness + helpers

tier1\_sim.py # CLI: calibration \& attack runs

tier1\_plot.py # Thesis-grade plots (abort rates, Δ̂ histograms, fidelity, etc.)

tier1\_audit.py # Post-hoc metrics: abort %, ΔI, AUC vs attack

tier1\_sensitivity.py # Sensitivity sweeps (ABORT% vs Δ\_budget, AUC vs β)

quickcheck.py # Baseline sanity checker \& config verifier

qu\_sweep.py # Policy grid sweeps (kill\_strikes × warn\_cooldown)





---

\## Quickstart

\### 1. Calibration

Create a baseline reference distribution and thresholds:

```bash

python tier1\_sim.py --qubits 8 --attacks none --steps 300 \\

&nbsp; --calibrate --out nadgo\_calib



Outputs:

q\_ref.json — smoothed baseline feature distribution

thresholds.json — suggested Δ\_budget (P80) and Δ\_kill (P95)



2\. Run Simulations

Simulate multiple attacks with automatic thresholds:

python tier1\_sim.py --qubits 8 --attacks none rl timing \\

&nbsp; --seeds 0:10 --steps 300 --depth 32 --shots 2048 \\

&nbsp; --qref nadgo\_calib/q\_ref.json --auto\_thresholds \\

&nbsp; --out nadgo\_results



Each run writes:

intervals.csv — interval-level Δ̂ values + policy decisions

episodes.csv — episode metrics (latency, fidelity, power, aborts)

segments.csv — raw segment traces

settings.json, thresholds\_used.json



3\. Analysis \& Plots

Generate publication-ready figures:

python tier1\_plot.py --root nadgo\_results



Outputs into figs/:

Abort rates by attack (abort\_rate\_by\_attack\_intervals.pdf, heatmaps, stacked bars)

Δ̂ histograms with β/budget/kill thresholds

Policy mixes, episode abort rates, fidelity vs qubits, leakage violins

Audit and sensitivity scripts:



python tier1\_audit.py

python tier1\_sensitivity.py --root nadgo\_results



4\. Grid Sweep

Explore hysteresis policies:

python qu\_sweep.py --ks 2,3 --wc 0,1,2 \\

&nbsp; --qref nadgo\_calib/q\_ref.json --outprefix nadgo\_grid



Produces:

policy\_grid.csv — abort/warn/continue % across ks×wc

policy\_grid\_deltas.csv — Δ̂ statistics per attack



5\. Quick Checks

Verify a run’s configuration and baseline sanity:

python quickcheck.py --root nadgo\_results --qref\_file nadgo\_calib/q\_ref.json



Dependencies

Python ≥ 3.8

Required: numpy, pandas, matplotlib, seaborn

Optional: scikit-learn (AUC), tqdm (progress), psutil (extra telemetry)



Citation

If you use NADGO in your work, please cite:

Punch, S. (2025). Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing. MSc Thesis, University College Cork.





