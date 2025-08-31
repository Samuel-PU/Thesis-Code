\# NADGO — Noise-Adaptive Dummy-Gate Obfuscation (Tier-2 Framework)



\*\*NADGO Tier-2\*\* extends the Tier-1 simulator into a \*\*full emulation framework\*\* for evaluating  

operational-privacy enforcement under realistic cloud conditions. It integrates workload padding,  

timing randomisation, topology-aware routing, and leakage monitoring with emulation-ready service  

models and system-level stress tests.



This package implements the \*\*Tier-2 simulation and analysis pipeline\*\* used in the thesis:



> Punch, S. (2025). \*Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing.\* MSc Thesis, University College Cork.



---



\## Key Features

\- \*\*Emulation-ready simulator (`NADGOSimulator`)\*\*  

&nbsp; - Service-time models (multiplicative vs affine) with noise, batch, and backlog effects.  

&nbsp; - Configurable heterogeneous backends (noise level, power, topology penalties).  

\- \*\*Operational-privacy enforcement\*\*  

&nbsp; - t-design padding, particle-filter scheduling, CASQUE routing.  

&nbsp; - Leakage monitor with β-floor, dual thresholds (Δ\_budget / Δ\_kill), hysteresis.  

\- \*\*Tier-2 outputs\*\*  

&nbsp; - Interval-level decisions (CONTINUE/WARNING/ABORT).  

&nbsp; - Episode-level metrics (latency, fidelity, power, abort rates).  

&nbsp; - Segment-level traces with leak\_bits and policy actions.  

\- \*\*Analysis \& plotting\*\*  

&nbsp; - Audits of abort %, ΔI distributions, AUC vs attack.  

&nbsp; - Camera-ready plots of policies, leakage histograms, heatmaps, episode stats.



---



\## Repository Layout

components.py # Core modules: padding, scheduler, router, leakage monitor

simulator.py # NADGO Tier-2 simulator + helpers

tier2\_sim.py # CLI: calibrated Tier-2 runs (baseline, attacks, thresholds)

tier2\_audit.py # Audit script: abort %, ΔI stats, AUC, cost deltas

tier2\_plot.py # Thesis-grade plots (abort rates, Δ̂ histograms, policy mix, heatmaps)

quickcheck.py # Baseline sanity \& config verification

qu\_sweep.py # Policy grid sweeps (kill\_strikes × warn\_cooldown)





---



\## Quickstart

\### 1. Calibration (optional)

Run a baseline and derive thresholds:

```bash

python tier2\_sim.py --qubits 8 --attacks none --steps 300 \\

&nbsp; --calibrate --out tier2\_calib

Outputs:

q\_ref.json — baseline distribution

thresholds.json — suggested P80/P99 thresholds



2\. Run Tier-2 Simulations

Simulate attacks (none, rl, timing) with thresholds:

python tier2\_sim.py --qubits 4 8 --attacks none rl timing \\

&nbsp; --seeds 0:10 --steps 300 --depth 64 --shots 2048 \\

&nbsp; --qref tier2\_calib/q\_ref.json --auto\_thresholds \\

&nbsp; --out tier2\_emul\_results --out\_stamp



Writes:

intervals.csv — Δ̂\_t values + decisions

episodes.csv — per-episode averages (latency, fidelity, power)

segments.csv — fine-grained traces

settings.json, thresholds\_used.json



3\. Audit \& Verification

Audit results:

python tier2\_audit.py --root tier2\_emul\_results

Quick sanity check (baseline abort % and Δ̂ levels):

python quickcheck.py --root tier2\_emul\_results --qref\_file tier2\_calib/q\_ref.json



4\. Plotting

Generate camera-ready plots:

python tier2\_plot.py --root tier2\_emul\_results --out figs/tier2



Figures:

abort\_rate\_by\_attack\_intervals.pdf

delta\_hist\_<attack>.pdf (Δ̂ distributions with β, budget, kill overlays)

policy\_action\_mix\_intervals.pdf (stacked shares)

abort\_rate\_heatmap\_intervals.pdf (n × attack)

episode\_stats.pdf (aborted episodes, cost deltas)



5\. Policy Grid Sweep

Explore hysteresis variants:

python qu\_sweep.py --ks 2,3 --wc 0,1,2 \\

&nbsp; --qref tier2\_calib/q\_ref.json --outprefix grid\_test



Outputs:

policy\_grid.csv — decision distributions

policy\_grid\_deltas.csv — Δ̂ stats per attack



Dependencies

Python ≥ 3.8

Required: numpy, pandas, matplotlib, seaborn

Optional: scikit-learn (AUC), tqdm, psutil



Citation

If you use NADGO Tier-2 in your work, please cite:

Punch, S. (2025). Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing. MSc Thesis, University College Cork.

