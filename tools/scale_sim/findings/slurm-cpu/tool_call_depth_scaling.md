# Tool-call hop depth (slurm-cpu)

Hardware: `cpu1-00091` — Intel Xeon Gold 6442Y, 96 cores, 251.6 GB RAM, Linux
5.19, Python 3.12. Harness git SHA `74056f89`, label `slurm-cpu`.

## What this experiment tests

A "hop" is one model call followed by one tool call inside a single rollout.
Long-horizon agents chain many hops. We want to know whether per-rollout cost
grows linearly with hop depth (each hop a fixed cost), super-linearly (each hop
costs more than the last), or cliffs. Run at concurrency = 1 so there is no
queueing — this isolates the actual per-hop cost from queue-wait.

The body is held small (1k tokens), not the 64k training body, on purpose:
`simple_agent` accumulates prior outputs into each hop's request, so the request
body grows linearly with hop count. At 512 hops even a 1k body produces a ~13 MB
final request; a 64k body would be ~870 MB and OOM.

## Setup

- Topology: single-agent, concurrency = 1 (no queueing).
- Held constant: 1k-token response body (`output_tokens = 1024`), 8 rollouts per
  cell, 900 s wall-clock budget.
- Varied: hops per rollout `∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}` (powers
  of 2). Each hop = one model call + one tool call.

## Data

| hops | throughput (roll/s) | p50 (s) | p99 (s) | per-hop p50 (s) | completed | wall (s) |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 68.9 | 0.0104 | 0.041 | 0.0104 | 8 | 0.12 |
| 2 | 48.9 | 0.0162 | 0.048 | 0.0081 | 8 | 0.16 |
| 4 | 29.5 | 0.0301 | 0.060 | 0.0075 | 8 | 0.27 |
| 8 | 14.4 | 0.0654 | 0.098 | 0.0082 | 8 | 0.56 |
| 16 | 5.84 | 0.167 | 0.203 | 0.0104 | 8 | 1.37 |
| 32 | 1.99 | 0.499 | 0.536 | 0.0156 | 8 | 4.01 |
| 64 | 0.604 | 1.646 | 1.725 | 0.0257 | 8 | 13.2 |
| 128 | 0.162 | 6.183 | 6.284 | 0.0483 | 8 | 49.4 |
| 256 | 0.043 | 23.01 | 24.61 | 0.0899 | 8 | 185.8 |
| 512 | 0.0107 | 92.65 | 97.18 | 0.181 | 8 | 745.4 |

All cells completed all 8 rollouts (no wall-clock cutoff, `completion_rate = 1`),
zero failures, zero retries.

## Key takeaways

- **No cliff through 512 hops; the system stays clean.** Every cell drained all
  8 rollouts with zero errors and zero retries. Deep multi-hop rollouts are slow
  but never break.
- **Per-hop cost is super-linear — the conversation-accumulation tax.** The
  per-hop p50 is flat at ~8 ms through 8 hops, then climbs steadily: 10 ms at 16
  hops → 26 ms at 64 → 90 ms at 256 → 181 ms at 512. That is a ~17× growth in the
  cost of a single hop from depth 16 to depth 512. The cause is structural:
  `simple_agent` folds all prior outputs into each subsequent request, so the
  request body grows linearly with hop number and Pydantic + orjson pay more to
  (de)serialize it on every later hop. Total p50 grows roughly quadratically
  (256 → 512 hops doubles depth but quadruples p50, 23 s → 93 s).
- **Hop count is multiplicative on absolute work.** Even with no queueing, a
  512-hop rollout takes ~93 s. A workload that needs hundreds of hops per rollout
  is fundamentally slow regardless of topology — sharding agents removes queueing,
  not the per-rollout accumulation cost.

## Comparison to the prior report

The qualitative finding matches §3.3.A of the prior report: per-hop cost grows
toward the deep end because of conversation-history accumulation, with no cliff.
The absolute per-hop floor here (~8–10 ms) is much lower than the prior report's
(~0.32 s/hop at 16 hops). The most likely reasons are hardware (Xeon Gold 6442Y
vs the prior host) and that this run adds no model `async_latency_ms`; the prior
floor was closer to a full unloaded round-trip. The accumulation trend is
actually more pronounced here (17× vs ~2×) precisely because the low fixed floor
lets the growing-body cost show through earlier.

## Caveats

- Concurrency = 1, so this is the queueing-free per-hop cost. Under load the
  per-hop number is dominated by queue-wait, not the accumulation tax (the prior
  report saw ~180× inflation at c=8192).
- Synthetic model/resources do almost no work; the per-hop cost here is framework
  + body-accumulation only. Real model/tool calls would add their own latency on
  top, multiplied by hop count.
