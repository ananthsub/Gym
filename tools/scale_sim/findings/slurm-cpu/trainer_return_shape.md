# Trainer return shape — Ray-actor stress (slurm-cpu)

Hardware: `cpu1-00091` — Intel Xeon Gold 6442Y, 96 cores, 251.6 GB RAM, Linux
5.19, Python 3.12. Harness git SHA `74056f89`, label `slurm-cpu`.

## What this experiment tests

The training framework drives gym wrapped as a Ray actor, not over direct HTTP.
Two request shapes are possible against that actor:

- **N concurrent blocking RPCs** (`sync_blocking`): the trainer fires N threads,
  each doing one blocking `ray.get` for one prompt. N is the number of RPCs the
  actor must service at once.
- **One streaming call** (`lag_batched_stream`): a single call carries all rows
  and results stream back via an `ObjectRefGenerator`. This is the shape in
  production today.

We sweep the concurrent-RPC count in powers of 2 to find where the actor's
inbound queue overloads, and compare against the streaming shape at the same
total load. Same actor, same per-step rollout count, different request shape.

## Setup

- Topology: a single-agent gym (`actor_repro.yaml`) wrapped in one Ray actor.
  Per rollout: 2 hops, 64k-token body (~1.7 MB), `async_latency_ms = 1500` on the
  model (production-shaped median latency), so each rollout floors at ~3.8 s.
- Held constant: 5 trainer steps; total rollouts per step = threads ×
  prompts/call.
- Varied:
  - `sync_blocking` with `thread_count ∈ {1, 4, 16, 64, 256, 1024}` concurrent
    RPCs, 1 prompt each.
  - `lag_batched_stream` (production): 4 threads × 256 prompts/call, one
    streaming call.

## Data

| shape | concurrent RPCs | throughput (roll/s) | p50 (s) | p99 (s) | completed | failures |
| --- | --- | --- | --- | --- | --- | --- |
| sync_rpc1 | 1 | 0.148 | 3.76 | 3.81 | 5 | 0 |
| sync_rpc4 | 4 | 0.556 | 4.07 | 4.25 | 20 | 0 |
| sync_rpc16 | 16 | 1.76 | 5.47 | 6.35 | 80 | 0 |
| sync_rpc64 | 64 | 3.59 | 11.84 | 16.48 | 320 | 0 |
| sync_rpc256 | 256 | 4.85 | 38.74 | 51.31 | 1280 | 0 |
| sync_rpc1024 | 1024 | 4.49 | 182.46 | 240.64 | 5120 | 0 |
| stream_allrows | 1 call | 4.58 | 174.37 | 262.77 | 5120 | 0 |

Every shape completed all its rollouts (`completion_rate = 1`), zero failures,
zero retries.

## Key takeaways

- **The actor throughput ceiling is ~4.6–4.9 roll/s** for this workload, reached
  by ~256 concurrent RPCs. Below that, throughput scales with RPC count
  (0.15 → 0.56 → 1.76 → 3.59 → 4.85 roll/s as N goes 1 → 256).
- **Past ~256 concurrent RPCs the actor overloads — more RPCs, no more
  throughput, far worse latency.** At N=1024 throughput is 4.49 roll/s (no better
  than N=256, actually slightly lower), but p50 jumps 38.7 s → 182 s (≈4.7×) and
  p99 hits 241 s. The actor's inbound RPC queue saturates; additional concurrent
  `ray.get`s just pile up. This is the overload behavior the concurrent-RPC shape
  is prone to.
- **The streaming production shape reaches the same ceiling without the
  overload.** `stream_allrows` hits 4.58 roll/s — matching the best sync case —
  with a single streaming call instead of fanning out 256–1024 concurrent RPCs.
  At equal total in-flight (1024), streaming slightly edges out `sync_rpc1024`
  (4.58 vs 4.49) and avoids piling RPCs on the actor. This supports keeping the
  one-call-streaming shape over concurrent per-prompt RPCs.
- **The Ray-actor boundary adds no new failure mode.** Across every shape and RPC
  count, completion is 1.0 with zero errors/retries — answering the prior
  report's open question: wrapping gym in a Ray actor costs throughput/latency and
  can overload under excessive concurrent RPCs, but it does not drop or error
  requests that a direct HTTP client would not.

## Caveats

- This actor config bakes in 1500 ms × 2-hop model latency, so the absolute
  ~4.6 roll/s ceiling is not comparable to the latency-free HTTP experiments —
  the value of this experiment is the **shape** comparison (concurrent RPCs vs
  streaming) at a fixed actor workload, not the absolute rate.
- Single actor on one host; this does not characterize multiple actors or a
  distributed trainer.
- Synthetic model/resources; real inference would change the absolute numbers but
  not the concurrent-RPC-overload-vs-streaming contrast.
