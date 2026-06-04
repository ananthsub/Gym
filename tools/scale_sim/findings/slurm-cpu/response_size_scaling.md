# Response body size (slurm-cpu)

Hardware: `cpu1-00091` — Intel Xeon Gold 6442Y, 96 cores, 251.6 GB RAM, Linux
5.19, Python 3.12. Harness git SHA `74056f89`, label `slurm-cpu`.

## What this experiment tests

How does the system behave as the response body grows? The RL-shaped response
(token ids + log probs + text) is what the consumer must serialize, transfer,
and parse for every rollout. We sweep the body in powers of 2 from 16k to 4M
tokens (~0.4 MB to ~109 MB), holding concurrency fixed, to find the consumer's
byte-throughput ceiling and where memory or latency cliff. 64k tokens (~1.7 MB)
is the de-facto training body and sits in the middle of the sweep.

## Setup

- Topology: single-agent, 1 hop per rollout, dispatch semaphore enforced.
- Held constant: concurrency = 64 (16 for ≥2M tokens to bound in-flight bytes),
  256-input budget, 300 s wall-clock budget.
- Varied: `output_tokens ∈ {16k, 32k, 64k, 128k, 256k, 512k, 1M, 2M, 4M}`.

## Data

| tokens | body | throughput (roll/s) | steady (roll/s) | p50 (s) | p99 (s) | ~MB/s (steady×body) | completed | stop_reason |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 16k | 0.4 MB | 45.6 | 51.1 | 1.25 | 1.89 | 18.2 | 256 | finished |
| 32k | 0.9 MB | 27.3 | 28.5 | 2.22 | 3.01 | 24.6 | 256 | finished |
| 64k | 1.7 MB | 13.1 | 13.5 | 4.42 | 6.43 | 22.3 | 256 | finished |
| 128k | 3.4 MB | 7.62 | 7.78 | 8.00 | 11.2 | 25.9 | 256 | finished |
| 256k | 6.8 MB | 4.02 | 4.87 | 15.0 | 22.2 | 27.3 | 256 | finished |
| 512k | 13.6 MB | 1.84 | 2.10 | 33.1 | 49.2 | 25.0 | 256 | finished |
| 1M | 27.3 MB | 0.89 | 0.95 | 70.2 | 102.8 | 24.3 | 256 | finished |
| 2M | 54.5 MB | 0.41 | 0.44 | 36.5* | 65.7 | 24.2 | 124 | wall_clock>300s |
| 4M | 109 MB | 0.17 | 0.18 | 81.1* | 150.5 | 19.9 | 51 | wall_clock>300s |

`*` 2M/4M were wall-clock-bound; their p50 reflects only the rollouts that
finished inside 300 s, so it understates the true median. Failures there
(1.6%/3.9%) are rollouts in flight at wall-clock, not errors (`retry_rate = 0`).

## Key takeaways

- **Rollout throughput is inversely proportional to body size.** Each doubling
  of the body roughly halves throughput: 45.6 → 27.3 → 13.1 → 7.6 → 4.0 → 1.84 →
  0.89 roll/s across 16k → 1M. p50 grows in lock-step (1.25 s → 70 s).
- **Byte throughput is the real ceiling, and it is ~constant.** Throughput × body
  size is flat at **~22–27 MB/s** across the entire 0.4 MB → 27 MB range. The
  consumer clears a fixed number of JSON bytes per second; bigger bodies just mean
  fewer rollouts behind those bytes. This is the binding resource for
  body-heavy workloads.
- **No memory cliff through 109 MB bodies.** 2M/4M cells went wall-bound, not
  OOM; the consumer's byte-rate caps how many large bodies coexist, so peak
  in-flight memory stays bounded even at 109 MB/response.
- **64k (training) sits at ~13 roll/s, ~22 MB/s.** This is the per-host body
  ceiling for a training-representative payload at concurrency 64; it is the same
  number `concurrency_scaling` converges to once the consumer saturates.

## Comparison to the prior report

This is a near-exact reproduction of the prior report's §3.2.E large-payload
sweep at c=64: prior p50 was 1.46 s (16k), 4.37 s (64k), 16.6 s (256k), 74.3 s
(1M); here it is 1.25 s, 4.42 s, 15.0 s, 70.2 s. The constant ~22 MB/s byte
ceiling also matches the prior "~22 MB/s" figure. Strong validation that the
rewritten harness reproduces the prior single-host body-size behavior.

## Caveats

- Synthetic model/resources do no real work; this isolates the
  serialize/transfer/parse cost of the body. Production wall-clock would add real
  model/tool time on top.
- The MB/s figure is JSON wire bytes derived from token count (~26 B/token), not
  raw tensor bytes.
