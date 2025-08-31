\# NADGO — Noise-Adaptive Dummy-Gate Obfuscation



\*\*NADGO\*\* is a scheduling \& obfuscation stack for \*\*operational privacy\*\* in quantum cloud workloads.  

It enforces per-interval limits on information leakage by combining padding, timing randomisation, topology-aware routing, and online leakage monitoring.



This implementation provides the \*\*Tier-1 simulation harness\*\* used in the thesis:



> Punch, S. (2025). \*Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing.\* MSc Thesis, University College Cork.



---



\## Core Ideas



\- \*\*t-design padding:\*\* inserts structured dummy gates with bounded overhead.  

\- \*\*Particle-filter scheduling:\*\* jitter injection adaptive to queue state.  

\- \*\*CASQUE routing:\*\* topology-aware subcircuit routing with fidelity–leakage–queue trade-offs.  

\- \*\*Leakage monitor:\*\* tracks estimated Δ̂ (bits) using finite-sample bounds; triggers WARN/ABORT with hysteresis.  

\- \*\*Kill-switch:\*\* dual thresholds (Δ\_budget, Δ\_kill) with strikes/cooldowns to prevent runaway leakage.



---



\## Repository Structure

components.py # Core modules: padding, scheduler, router, monitor

simulator.py # NADGO simulator harness

tier1\_sim.py # CLI entrypoint: run baseline / attacks, calibrate thresholds

plots\_tier1\_unified.py # Camera-ready figures (policy overview, Δ̂ histograms, ROC, AUC)

tier1\_audit.py # Post-hoc audit of episodes \& intervals

tier1\_sensitivity.py # Sensitivity study (ABORT% vs Δ\_budget; AUC vs β)

quickcheck.py # Sanity checker for baseline Δ̂ and abort rates

qu\_sweep.py # Policy grid sweeps over kill\_strikes × warn\_cooldown

test.py # Minimal epsilon\_est + leakage monitor test





---



\## Quickstart



\### 1. Calibration (baseline only)

Build a reference distribution `q\_ref.json` and suggested thresholds:

```bash

python tier1\_sim.py --qubits 8 --attacks none --steps 300 \\

&nbsp; --calibrate --out nadgo\_calib

Outputs:

nadgo\_calib/q\_ref.json — baseline feature distribution

nadgo\_calib/thresholds.json — suggested Δ\_budget (P80) and Δ\_kill (P95)



2\. Simulation under Attacks

Run NADGO against different attack modes (none, rl, timing):

python tier1\_sim.py --qubits 8 --attacks none rl timing \\

&nbsp; --seeds 0:10 --steps 300 --depth 32 --shots 2048 \\

&nbsp; --qref nadgo\_calib/q\_ref.json --auto\_thresholds \\

&nbsp; --out nadgo\_results



3\. Analysis \& Figures

Generate consolidated figures:

python plots\_tier1\_unified.py --roots nadgo\_results

Figures written to figs/:

fig1\_policy\_overview.pdf — ABORT heatmap, episode abort rates, policy mix

fig2\_delta\_histograms.pdf — Δ̂ distributions under none / rl / timing

fig3\_roc\_auc.pdf — ROC curve + AUC for attack detection

fig4\_sensitivity.pdf — ABORT% vs Δ\_budget, AUC vs β

Audit and sensitivity studies:

python tier1\_audit.py

python tier1\_sensitivity.py --root nadgo\_results



4\. Policy Grid Sweep

Explore hysteresis variants:

python qu\_sweep.py --ks 2,3 --wc 0,1,2 --steps 300 --seeds 0:5 \\

&nbsp; --qref nadgo\_calib/q\_ref.json --outprefix nadgo\_grid

Outputs CSVs summarising abort/warn rates and Δ̂ statistics.



Dependencies

Python ≥ 3.8

Required: numpy, pandas, matplotlib, seaborn

Optional: scikit-learn (for AUC in audits)



Citation

If you use NADGO in your research, please cite:

Punch, S. (2025). Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing. MSc Thesis, University College Cork.



