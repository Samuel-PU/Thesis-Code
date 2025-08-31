\### Tier-2 (Emulation)



Tier-2 provides \*\*system-level stress testing\*\* of MaestroCut under request/response workloads. It evaluates latency, throughput, reliability, and confidentiality-related overheads across multiple scenarios.



\*\*Scenarios:\*\*

\- `baseline` — nominal, low-variance behaviour.

\- `noisy` — mild multiplicative noise in service times.

\- `bursty` — log-normal latency bursts (heavy tails).

\- `adversarial` — injected extra latency and errors.



\*\*Running an Emulation\*\*



Use `emulator\_cli.py` with arguments for scenario, workload mix, latency distribution, request count, and timeout. Optional flags allow memory sampling, energy usage, fidelity/error synthesis, and cloud-cost estimation.



Example:

```bash

\# Run 1000 requests in the "noisy" scenario on an A100 GPU profile

python tier2/emulator\_cli.py \\

&nbsp; --scenario noisy --seed 123 --version 1.0 \\

&nbsp; --hardware A100x1 --workload A=0.7,B=0.3 \\

&nbsp; --n-requests 1000 --lat-mean 180 --lat-std 40 \\

&nbsp; --timeout-ms 500 --out runs.jsonl \\

&nbsp; --enable-mem --mem-sample-every 10 \\

&nbsp; --instance-hourly-usd 2.5 \\

&nbsp; --energy-per-request-j 0.05 \\

&nbsp; --fidelity-mean 0.98 --fidelity-std 0.01



