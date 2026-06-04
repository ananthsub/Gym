# Work-per-step sensitivity (slurm-cpu)

Hardware: `cpu1-00091` — Intel Xeon Gold 6442Y, 96 cores, 251.6 GB RAM, FD limit
1,048,576→131,072, Linux 5.19, Python 3.12. Harness git SHA `74056f89`,
label `slurm-cpu`.

## What this experiment tests

Every rollout pays a fixed framework cost — the agent loop, the HTTP round
trips, and the Pydantic/JSON serialization of the response body — independent of
how much "real" work the model and tool servers do. This experiment asks how
much that framework overhead matters as each call does progressively more real
work. We add a controlled per-call delay to the synthetic model and watch
whether throughput holds (framework overhead dominates and the added work hides
underneath it) or drops proportionally (the added work dominates).

Because the held-constant body is the 64k training-representative response
(~1.7 MB), this measures the regime that matters for training: a realistic
payload, varying only the per-call work on top of it.

## Setup

- Topology: single-agent (1 head + 1 agent + 1 model + 1 resources, one host).
- Held constant: 64k-token response body (`output_tokens = 65536`), concurrency
  = 64, 1 hop per rollout, dispatch semaphore enforced, 20,000-input budget,
  300 s wall-clock budget. Collapse early-stop disabled so saturated cells run
  the full window.
- Varied: per-call added latency on the synthetic model
  `async_latency_ms ∈ {0, 64, 256, 1024}` (powers of 2). The delay is an
  `asyncio.sleep` — it adds wall-clock latency, not CPU.

## Data

| added work/call (ms) | throughput (roll/s) | steady throughput (roll/s) | p50 (s) | p99 (s) | completed | failure_rate | stop_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 13.59 | 13.64 | 4.58 | 6.40 | 4 081 | 0.0005 | wall_clock>300s |
| 64 | 13.70 | 13.81 | 4.60 | 6.46 | 4 121 | 0.0005 | wall_clock>300s |
| 256 | 13.34 | 13.51 | 4.69 | 6.90 | 4 004 | 0.0005 | wall_clock>300s |
| 1024 | 10.82 | 11.09 | 5.87 | 9.40 | 3 253 | 0.0006 | wall_clock>300s |

`steady throughput` is the warm-up-excluded rate; it tracks the whole-run rate
within ~1% here because the ramp is negligible at concurrency 64. All four cells
were wall-clock-bound at 300 s and drained 16–20% of the 20,000-input budget;
`failure_rate` ≈ 0 (a handful of rollouts in flight when wall-clock fired, not
application errors — `retry_rate = 0`, `error_class_counts = {}` everywhere).

## Key takeaways

- **Up to ~256 ms of per-call work is effectively free at the training body
  size.** Throughput is flat at ~13.5 roll/s from 0 → 256 ms (within noise), and
  p50 barely moves (4.58 → 4.69 s). The framework cost of moving a 64k-token body
  through the agent loop dominates the per-rollout time, so a few hundred ms of
  added server-side latency hides underneath it at no throughput cost.
- **The added work only bites once it approaches the per-rollout floor.** At
  1024 ms, throughput drops ~20% (13.6 → 10.8 roll/s) and p50 rises ~28%
  (4.58 → 5.87 s). The +1.29 s p50 increase is close to the +1.024 s of added
  delay — past ~256 ms the latency stops being absorbed and passes through to the
  critical path.
- **The system is saturated and slot-bound, not broken.** Little's law on every
  cell gives throughput × p50 ≈ 62–64 ≈ the offered concurrency: all 64 slots
  stay full, so throughput = concurrency / latency. Adding work lengthens each
  rollout's hold on its slot, which is the only reason throughput falls. No
  errors, no retries at any work level.
- **This bounds the framework tax at a training payload.** At 64k tokens and
  concurrency 64, the framework/body-processing cost is ~4.6 s of effective
  per-rollout time — large enough to swallow hundreds of ms of real per-call
  work. The heavier the real work per call, the smaller a fraction this framework
  tax becomes.

## Caveats

- The added "work" is `asyncio.sleep` on the synthetic model: it models added
  **latency**, not added **CPU**. Real per-call CPU work on sub-servers would
  contend with the consumer/agent loop for cores and could erode throughput
  sooner than a pure sleep does.
- The body is fixed at 64k tokens. The framework cost that hides the added work
  scales with body size (see `response_size_scaling`); at smaller bodies the same
  added delays would bite at lower thresholds.
- Synthetic model/resources do almost no real work, so these numbers bound
  framework overhead in isolation; they do not predict end-to-end throughput
  with real model inference.
