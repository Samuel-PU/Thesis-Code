NADGO — Noise-Adaptive Dummy-Gate Obfuscation Simulator



NADGO is a simulation and emulation framework for Noise-Adaptive Dummy-Gate Obfuscation — a set of techniques for enforcing operational privacy in quantum cloud execution.

It provides two tiers of evaluation:



Tier-1 (Simulation): lightweight calibrated simulator for leakage monitoring, thresholds, and policy testing.

Tier-2 (Emulation): higher-fidelity simulator with queueing, routing, and workload stress testing.



Repository Structure

tier1\_sim.py — CLI for Tier-1 simulation (baseline calibration, thresholds, attacks).

tier2\_sim.py — CLI for Tier-2 emulation with timing/queue models and richer attack scenarios.

components.py — Core building blocks: t-design padder, particle-filter scheduler, CASQUE router, leakage monitor.

simulator.py — Simulator harness, feature extraction, threshold enforcement, and emulation logic.



Tier-1 Simulation

Tier-1 is used for controlled experiments:

Run tier1\_sim.py with qubit sizes, attack types, seeds, shots, and depth.

Supports calibration mode (--calibrate) which builds a baseline distribution (q\_ref.json) and thresholds (thresholds.json).

Leakage monitoring uses Δ\_budget / Δ\_kill in bits, with hysteresis (strike and cooldown) policies.

Features include: finite-sample estimator slack, q\_min floor, epsilon\_sync timing slack.



Outputs:



JSON files (q\_ref.json, thresholds.json)

Logs of Δ̂\_t distributions and abort events



Tier-2 Emulation

Tier-2 extends Tier-1 by adding timing, queueing, and routing realism:

Run tier2\_sim.py with workload size, attack modes, and emulation config (JSON).

Supports baseline calibration, manual or auto thresholds, and per-interval leakage monitoring.

Emulation config can specify backend pool, noise levels, queue/service model, and topology penalties.

Policies include strict, default, and hysteresis modes.



Outputs:

JSON baseline (q\_ref.json), thresholds (thresholds.json), and per-run logs

Attack vs baseline comparison of abort rates, queue sizes, latency



Core Components

TDesignPadder: adds dummy Clifford layers for t-design obfuscation.

ParticleFilterScheduler: introduces adaptive dispatch jitter.

CASQUERouter: topology-aware routing balancing fidelity, leakage, and queue state.

LeakageMonitor: KL-based leakage estimator with hysteretic kill-switch.

NADGOSimulator: orchestrates episodes, tracks audit logs, and enforces Δ thresholds.



Dependencies

Python 3.8+

NumPy, Pandas





Citation

If you use NADGO in your research, please cite:



Punch, S. (2025). Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing. MSc Thesis, University College Cork.

