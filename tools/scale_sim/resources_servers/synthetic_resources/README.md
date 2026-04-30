# synthetic_resources

A controllable resources server for scale testing. Exposes:

- `/synthetic_tool` — used by the agent loop as the tool the model calls each hop.
- `/verify` — final scoring endpoint (always returns `reward=1.0` unless failure-injected).
- `/seed_session` — inherited from `SimpleResourcesServer`.
- `/aggregate_metrics` — inherited.

Both `/synthetic_tool` and `/verify` accept the same per-endpoint knobs:

| Knob | What it controls |
| --- | --- |
| `async_latency_ms` | `await asyncio.sleep(...)` — yields the event loop. Models I/O-bound waits. |
| `cpu_burn_ms` | `time.perf_counter()` busy-loop — holds the event loop. Models CPU-bound work. |
| `body_size_bytes` | Total byte length of the response body's variable portion. |
| `body_shape` | `flat_padding` (one big string, cheap parse) or `realistic_messages` (many small dict chunks, real GC pressure). |
| `body_dist` | `fixed` or `lognormal` for tail-distribution sweeps. |

Plus failure injection (apply to both endpoints):

| Knob | What it does |
| --- | --- |
| `inject_500_rate` | P(handler returns HTTP 500). Tests retry layering. |
| `inject_hang_rate` | P(handler awaits `inject_hang_seconds`). Simulates a wedged sub-server. |

This server is part of the `tools/scale_sim/` harness. See `tools/scale_sim/README.md` for how to actually run a sweep.
