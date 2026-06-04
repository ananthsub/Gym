# Concurrency scaling (slurm-cpu)

Hardware: `cpu1-00091` — Intel Xeon Gold 6442Y, 96 cores, 251.6 GB RAM, FD limit
131,072, Linux 5.19, Python 3.12. Harness git SHA `74056f89`, label `slurm-cpu`.

## What this experiment tests

How many concurrent rollouts can one head node serve before throughput stops
rising and latency takes over? We sweep offered concurrency in powers of 2 from
the unloaded floor (1) to deep saturation (131072), holding the body at the 64k
training payload. This locates the saturation point, the per-host throughput
ceiling at a training-representative body, and — importantly — whether high
offered concurrency at a 1.7 MB body causes a memory problem.

## Setup

- Topology: single-agent (1 head + 1 agent + 1 model + 1 resources, one host).
- Held constant: 64k-token body (`output_tokens = 65536`, ~1.7 MB), 1 hop,
  dispatch semaphore enforced, 300 s wall-clock budget. Collapse early-stop
  disabled. `total_requests` sized small at the floor and capped at 20,000.
- Varied: concurrency `∈ {1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 131072}`.
  No memory cap on the sweep (see takeaways).

## Data

| concurrency | throughput (roll/s) | steady (roll/s) | p50 (s) | p99 (s) | completion | saturated | stop_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 7.52 | 7.60 | 0.132 | 0.150 | 1.00 | no | finished |
| 4 | 12.39 | 12.79 | 0.312 | 0.505 | 1.00 | no | finished |
| 16 | 14.24 | 14.64 | 1.058 | 1.612 | 1.00 | no | finished |
| 64 | 13.77 | 13.94 | 4.564 | 7.144 | 1.00 | no | finished |
| 256 | 15.19 | 15.74 | 15.79 | 34.23 | 1.00 | no | finished |
| 1024 | 14.80 | 15.11 | 21.45 | 282.0 | 0.54 | yes | wall_clock>300s |
| 4096 | 13.08 | 13.60 | 145.3 | 287.9 | 0.20 | yes | wall_clock>300s |
| 16384 | 10.18 | 12.02 | 191.7 | 298.7 | 0.15 | yes | wall_clock>300s |
| 65536 | 11.17 | 13.42 | 174.1 | 295.8 | 0.17 | yes | wall_clock>300s |
| 131072 | 11.11 | 14.12 | 164.1 | 296.9 | 0.17 | yes | wall_clock>300s |

Failures are ≤0.07% everywhere (rollouts in flight at wall-clock; `retry_rate =
0`, `error_class_counts = {}`).

## Key takeaways

- **L₁ ≈ 0.13 s** for a single 64k-token rollout at concurrency 1 — the unloaded
  round-trip floor at the training body size.
- **Throughput peaks ~15 roll/s around c=16–256, then is concurrency-independent.**
  The consumer clears ~13–15 64k-token rollouts/sec regardless of how much more
  concurrency is offered. Past saturation, additional concurrency only deepens
  the queue: p50 climbs 4.6 s (c=64) → 16 s (c=256) → 145 s (c=4096) → ~165–190 s
  (c≥16384), and p99 pins to the 300 s wall. The system slows; it never breaks.
- **Memory is not the constraint — the consumer throttles it away.** This is the
  key result and it confirms the working hypothesis: at c=131072 with 1.7 MB
  bodies the naive in-flight estimate is ~218 GB, but the cell ran fine (no OOM,
  no errors). Because the consumer caps at ~13 roll/s, actual peak in-flight is
  throughput × latency ≈ 11 × 164 ≈ ~1,800 rollouts ≈ **~3 GB**, not 218 GB.
  Offered concurrency does not equal resident concurrency once the consumer
  saturates. (Matches the prior report's §3.2.C: predicted 55 GB, actual 4.4 GB.)
- **Warm-up exclusion matters at high concurrency.** At c≥16384 the steady rate
  (12–14 roll/s) is meaningfully higher than the whole-run rate (10–11), because
  opening tens of thousands of sockets dilutes the whole-run average. The steady
  number is the one to trust for the saturated ceiling.

## Comparison to the prior report

Same shape as the prior concurrency baseline (§3.1.B): throughput peaks at low
concurrency then goes flat/declines while latency grows linearly, with zero
errors and wall-bound saturation. The absolute ceiling here is ~15 roll/s vs the
prior ~45 roll/s because the prior baseline used a 16k-token body (~0.4 MB) and
this uses the 64k training body (~1.7 MB) — consistent with the inverse
body/throughput relationship in `response_size_scaling`.

## Caveats

- Synthetic model/resources do no real work; this is the orchestration/transport
  ceiling at a 64k body, not an end-to-end-with-inference number.
- The first wave admits up to `concurrency` requests at once; the model
  serializes them at its own rate, which is why peak memory stays bounded. A
  backend that buffered all offered requests eagerly would not enjoy this
  protection.
