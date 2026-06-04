# Agent fan-out (slurm-cpu)

Hardware: `cpu1-00091` — Intel Xeon Gold 6442Y, 96 cores, 251.6 GB RAM, Linux
5.19, Python 3.12. Harness git SHA `74056f89`, label `slurm-cpu`.

## What this experiment tests

A single agent's event loop saturates at a fixed per-rollout rate. Running N
agent instances in parallel (each its own loop, its own resources server, one
shared model) should add more total CPU for per-rollout work. This sweep holds
per-agent concurrency at 256 and grows N, so total offered concurrency grows with
N (256 → 8192). The question: does fan-out keep increasing throughput, and where
does it stop — and what becomes the bottleneck at a training body size?

## Setup

- Topology: multi-agent — N agents + N resources + 1 shared model + 1 head, all
  on one host; the driver round-robins rollouts across the N agents.
- Held constant: per-agent concurrency = 256, 64k-token shared-model body
  (`output_tokens = 65536`), 1 hop, 240 s wall-clock budget. Collapse early-stop
  disabled.
- Varied: `n_agents ∈ {1, 2, 4, 8, 16, 32}` (powers of 2). Total concurrency =
  256 × N.

## Data

| N agents | total conc | throughput (roll/s) | steady (roll/s) | p50 (s) | p99 (s) | completion | stop_reason |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 256 | 15.6 | 16.0 | 15.3 | 26.6 | 0.19 | wall_clock>240s |
| 2 | 512 | 29.3 | 30.1 | 16.1 | 27.0 | 0.35 | wall_clock>240s |
| 4 | 1024 | 38.7 | 39.6 | 17.2 | 73.9 | 0.46 | wall_clock>240s |
| 8 | 2048 | 36.7 | 36.9 | 25.5 | 192.8 | 0.44 | wall_clock>240s |
| 16 | 4096 | 36.6 | 37.3 | 25.3 | 222.6 | 0.46 | wall_clock>240s |
| 32 | 8192 | 38.4 | 39.1 | 12.4 | 201.1 | 0.46 | wall_clock>240s |

All cells wall-bound at 240 s; failures ≤0.05% (in flight at wall-clock, no
errors/retries).

## Key takeaways

- **Fan-out helps only to ~N=4, then the throughput plateaus.** Throughput scales
  cleanly N=1→4 (15.6 → 29.3 → 38.7 roll/s, ~2.5× for 4× agents), then is flat at
  ~37–39 roll/s through N=32. Adding agents past 4 buys no additional throughput.
- **The shared model is the bottleneck at the training body size.** The ~38
  roll/s ceiling (~65 MB/s of 64k bodies) is set by the single shared
  synthetic_model serializing every response. N agents can consume in parallel,
  but they all pull from one model that can only emit bodies so fast. Past N=4
  the agents are starved by the model, not by their own loops.
- **More agents past the knee just deepen queues.** Total offered concurrency
  grows with N, so p99 climbs from ~27 s (N≤2) to ~200+ s (N≥8) even though
  throughput is flat — extra agents add in-flight depth and tail latency without
  adding completion rate. No failures at any N.

## Comparison to the prior report

The prior report's multi-agent sweep (§3.6) used a **1k-token** body and found
fan-out helped all the way to N≈16–32 before plateauing — there the per-agent
**event loop** (Pydantic/JSON on small bodies) was the constraint, so more loops
helped longer. Here, at the 64k training body, the constraint moves upstream to
the **shared model's body serialization**, so the knee arrives much earlier
(N≈4). Same mechanism (a shared component saturates), different component
depending on body size. The implication: at training payloads, scaling agent
count alone is not enough — the model/response-producer side must scale too.

## Caveats

- Synthetic model does no real inference; the "model serialization ceiling" here
  is the synthetic body generator, not a real vLLM endpoint. In production the
  model is remote/GPU and this particular bottleneck would differ — but the
  general lesson (a shared producer caps agent fan-out) holds.
- This sweep stops at N=32 (total 8192); the prior report reached N=256 (65k
  total) at a 1k body. At 64k bodies the shared-model ceiling makes higher N
  uninformative for throughput.
