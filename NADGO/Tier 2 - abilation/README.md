\# NADGO — Noise-Adaptive Dummy-Gate Obfuscation (Tier-2 Ablation Framework)



\*\*NADGO Tier-2 Ablation\*\* extends the standard Tier-2 emulation to compare \*\*different pipeline variants\*\*.  

It allows isolating the contribution of padding, jitter, and CASQUE routing by selectively disabling  

components and benchmarking \*\*vendor\*\*, \*\*pad-only\*\*, \*\*jitter-only\*\*, and \*\*full\*\* pipelines.



This package implements the ablation study described in the thesis:



> Punch, S. (2025). \*Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing.\* MSc Thesis, University College Cork.



---



\## Pipelines



\- `vendor` — no padding, no jitter, vendor-style routing.  

\- `pad-only` — t-design padding, no jitter, vendor routing.  

\- `jitter-only` — jitter + vendor routing, no padding.  

\- `full` — full NADGO: padding + jitter + CASQUE routing.  



---



\## Repository Layout

components.py # Core components: padder, scheduler, router, monitor

simulator.py # NADGO Tier-2 simulator with pipeline ablations

tier2\_sim.py # CLI for calibrated Tier-2 runs (supports --pipeline vendor|pad-only|jitter-only|full)

tier2\_audit.py # Post-hoc analysis: abort %, ΔI stats, AUC, cost deltas

tier2\_plot.py # Ablation-specific plots (abort % by pipeline, latency/power CIs, etc.)

quickcheck.py # Baseline sanity check \& config verification

qu\_sweep.py # Policy grid sweeps with manual thresholds

check\_ablation.py # Summarises ablation runs: ABORT%, thresholds, knobs, seed coverage

combine\_ablation.py # Merges multiple runs into one folder with a pipeline column



---



\## Quickstart



\### 1. Run Ablation Simulations

Run each pipeline separately, e.g.:

```bash

python tier2\_sim.py --pipeline vendor     --out tier2\_vendor

python tier2\_sim.py --pipeline pad-only   --out tier2\_padonly

python tier2\_sim.py --pipeline jitter-only --out tier2\_jitter

python tier2\_sim.py --pipeline full       --out tier2\_full



Each run produces episodes.csv, segments.csv, intervals.csv, plus settings.json and thresholds\_used.json.



2\. Combine Runs

Merge results for comparison:

python combine\_ablation.py --runs tier2\_vendor tier2\_padonly tier2\_jitter tier2\_full \\

&nbsp; --out ablation\_combined



Outputs a unified folder with pipeline-labelled CSVs.



3\. Check Ablation Consistency

Verify thresholds, seeds, knobs, and ABORT%:

python check\_ablation.py --runs tier2\_vendor tier2\_padonly tier2\_jitter tier2\_full \\

&nbsp; --out ablation\_summary.csv



4\. Plot Ablation Figures

Produce comparative plots:

python tier2\_plot.py --root ablation\_combined --out figs/ablation



Figures:

ablation\_abort\_by\_pipeline.pdf

ablation\_abort\_by\_pipeline\_attack.pdf

ablation\_latency\_power\_ci.pdf (95% CI bars for latency \& power vs pipeline)



5\. Audit Results

Standard audit across pipelines:

python tier2\_audit.py --root ablation\_combined



Dependencies

Python ≥ 3.8

Required: numpy, pandas, matplotlib, seaborn

Optional: scikit-learn (AUC), tqdm, psutil



Note

This framework is similar to the standard Tier-2 package, but adds ablation utilities

(check\_ablation.py, combine\_ablation.py, and ablation-specific plots in tier2\_plot.py) to

benchmark and compare pipeline variants.



Citation

If you use NADGO Ablation results in your work, please cite:

Punch, S. (2025). Dynamic Circuit Cutting and Enforceable Operational Privacy for Confidential Quantum Cloud Computing. MSc Thesis, University College Cork.







