Confidential \& Scalable Quantum Cloud Computation



This repository contains the implementation of two integrated frameworks developed as part of the MSc thesis:



Punch, S. (2025). \*Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing.\*  

MSc Thesis, University College Cork.



Together, these systems enable \*\*scalable\*\* and \*\*privacy-preserving\*\* execution of quantum workloads on near-term cloud backends.





Systems Overview

MaestroCut

A \*\*closed-loop framework for dynamic circuit cutting\*\*:

\- Monitors device drift and time-varying noise.  

\- Allocates measurement shots adaptively using Kalman tracking and GP priors.  

\- Selects estimators in an online, workload-specific fashion.  

\- Evaluated through Tier-1 \*\*simulation\*\* and Tier-2 \*\*emulation\*\*.  

➡ See \[`maestrocut/README.md`](maestrocut/README.md) for details.



&nbsp;NADGO (Noise-Adaptive Dummy-Gate Obfuscation)

An \*\*obfuscation and operational-privacy enforcement layer\*\*:

\- Adds \*\*t-design dummy gates\*\* to mask execution.  

\- Applies \*\*drift-adaptive dispatch jitter\*\*.  

\- Routes circuits securely across backends with CASQUE.  

\- Detects and aborts on leakage via a \*\*KL-based monitor\*\* with hysteresis.  

\- Includes ablation pipelines to isolate the effect of padding, jitter, and routing.  



➡ See \[`nadgo/README.md`](nadgo/README.md) for details.



Repository Structure

maestrocut/ # MaestroCut simulation + emulation pipeline

nadgo/ # NADGO simulation + emulation pipeline



Each subfolder has its own README with usage details, CLI examples, and plotting instructions.



Requirements

\- Python 3.8+  

\- Required: `numpy`, `pandas`, `matplotlib`, `seaborn`  

\- Optional: `scikit-learn`, `tqdm`, `psutil`



Citation

If you use this work, please cite:

> Punch, S. (2025). \*Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing.\*  

MSc Thesis, University College Cork.



