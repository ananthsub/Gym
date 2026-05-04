# nemo-gym scale-test simulation — design

**Status:** draft (no code yet — this document defines what we will build and what we expect to learn)
**Author:** ansubramania
**Date:** 2026-04-27
**Audience:** post-training infra working on the next-generation natively-multimodal, multi-trillion-parameter model with substantially larger batch size and sequence length than today.

---

## 0. TL;DR

Today's nemo-gym architecture has several un-documented but load-bearing assumptions that will not survive the next scale-up. We have already observed five of them break at 8 K rollout batch / 512 nodes (see `2381291-nemo-gym-timeouts-connection-resets.md`). The next model adds three more pressure axes simultaneously — image/video tokens (much larger payloads), longer sequences, larger batch size — so we must characterize the breaking points **before** we get bug reports from a 2 K-GPU run.

This document defines a self-contained, network-only simulation harness that:

1. Replaces every "real" gym component (model, agents, tools, resources servers) with a parameterized synthetic version that emits controllable bytes / latency / tool-call structure.
2. Drives load through the actual nemo-gym `ServerClient` / aiohttp / `RolloutCollectionHelper` code paths so the failures we surface are real, not artifacts of a different stack.
3. Sweeps three axes that match the user's questions:
   - **Axis A — concurrent in-flight requests** at the head actor (the "fan-in" axis).
   - **Axis B — number of co-located sub-servers per head node** (the "all-servers-on-head-node" assumption).
   - **Axis C — HTTP transport stress** along two sub-axes:
     - C1: tool-call / model-hop count per `/run` (the "depth" axis).
     - C2: per-message and per-`/run` payload size (the "width" axis — long sequences, multimodal).

The deliverable is `tools/scale_sim/` — a small package with a synthetic resources/agent/model triplet, a load driver, and a sweep runner — plus a results report.

---

## 1. Why this needs to exist

### 1.1 What's already known, and what survives the recent fixes

`2381291-nemo-gym-timeouts-connection-resets.md` documents five code-level defects and three structural mechanisms behind the 8 K rollout × 512 node failure population. Of the five defects, four have since been fixed:

| # | Defect | Status |
| --- | --- | --- |
| 1 | Keep-alive idle race (5 s uvicorn vs 15 s aiohttp default) | **fixed** — timeouts realigned |
| 2 | No total client timeout (`ClientTimeout()` constructed with no args) | **fixed** — explicit timeouts now set |
| 3 | `ClientPayloadError` not retried at gym layer | **fixed** — retried in `RolloutCollectionHelper._post_subroutine` (`rollout_collection.py:450-452`) |
| 4 | `_internal=True` retries forever with a flat 0.5 s sleep | **fixed** — retry cap added |
| 5 | No client-side concurrency cap on RL dispatch (`run_examples` called without a `Semaphore`) | **open** — RL still fires the full step batch |

The three structural mechanisms behind the failures map to the defects as follows, with their current status:

- **M1 — Keep-alive idle race** (systemic, all sub-servers). **Closed** by the defect #1 fix. If we still see M1-shaped errors in the simulation, that is a real surprise and worth flagging.
- **M2 — Per-sub-server semaphore meltdown** (e.g. `code_gen.num_processes=8`). **Made transparent** by the defect #3 fix (retried instead of bubbling up), but the underlying cause is still config-only. The simulation does not need to reproduce this — its remedy is known and benchmark-specific.
- **M3 — Consumer-side receive backpressure on the single `NemoGym` Ray actor's event loop** (agent-independent, body-size-correlated; mid-body RST after kernel TCP RTO exhaustion or sub-server OOM). **Survives every defect fix.** A single actor with one asyncio event loop drains all bodies for the entire batch on one CPU; the ceiling is set by `min(per-socket rmem, event-loop tick budget) × N_concurrent_sockets`. **This is the primary thing the simulation must model carefully**, because it is the ceiling the next model will trip without intervention. Defect #5 (no RL-side semaphore) amplifies M3 by synchronizing all in-flight body reads on step boundaries; closing #5 mitigates but does not eliminate M3.

So the simulation's primary target is M3, with secondary targets being:
- the **interaction between defect #5 and M3** (does an RL-side semaphore by itself raise the ceiling enough?),
- the **head-node co-location ceiling** (none of the 2381291 fixes changed the "all sub-servers on one node" assumption — Axis B in §3),
- the **payload-width and hop-depth ceilings** for which we have no measurements at all today (Axis C).

### 1.2 Why we cannot wait for the next 2 K-GPU run to find the limits

Three things change at once for the next model:

| Axis | Today | Next model | Multiplicative pressure on M3 |
| --- | --- | --- | --- |
| Per-rollout response body | ~10 KB – 5 MB (text-only `<think>` + tool I/O) | tens of MB likely (multimodal latents/images in tool I/O, longer sequences, longer think traces) | direct — body bytes are the failing dimension |
| Concurrent in-flight rollouts | 8 K | likely 16 K – 64 K | direct — total inbound byte rate scales linearly |
| Tool-call hops per rollout | 1 (most agents) – ~10 (multi-turn) | 10s plausible (browser, code-exec, multimodal verifiers) | quadratic when combined with width — each hop is an HTTP round-trip with O(body size) cost |
| Sub-servers per head node | ~10 – 20 typical | will keep growing (multimodal verifiers, judges, sandboxes) | direct — every sub-server adds FDs, ports, kernel TCP state, and one more event-loop participant |

Each axis multiplies the others. We have no current data on where the cliff is for the cross product, only point measurements at one (8 K, 512 nodes, ~20 sub-servers, ~5 MB max body) configuration. We need a controllable harness to map the cliff **before** we hit it on a real run.

### 1.3 Why a simulation, not a real-traffic eval

Real-traffic evals have three problems we don't want:

1. **Latency dominated by upstream model serving** — vLLM / Triton tokens-per-second variance is enormous, and the failures we care about have nothing to do with the model. We want a mock that emits *bytes* at controllable rate, not tokens.
2. **No way to dial body size independently of latency** — a real model that produces 5 MB output also takes minutes to generate. We need to decouple them so we can hold latency constant while sweeping body size, and vice versa.
3. **Benchmark-specific behavior contaminates the signal** — `code_gen` has a unique `Semaphore(num_processes)` failure mode that is not architectural. The harness should isolate the architectural ceilings from per-benchmark ones, and let us re-introduce per-benchmark behavior on top once we want to.

---

## 2. Architectural assumptions worth making explicit

A scale test is most useful when it is targeted at *specific* assumptions that are likely to be wrong at the next scale point. The following are the un-documented assumptions baked into today's nemo-gym, drawn from reading `nemo_gym/server_utils.py`, `cli.py`, and `rollout_collection.py`:

| # | Assumption | Where in code | Why it likely breaks at next scale |
| --- | --- | --- | --- |
| **A1** | All sub-servers (resources servers, agents, model adapters) live on the head node as child processes of the head `Popen` loop. | `RunHelper.start` in `cli.py:144-178` | Head node has finite FDs (`ulimit`), one kernel TCP stack, one ephemeral-port range, one CPU. With 50+ sub-servers each receiving 16 K concurrent connections this saturates linearly. |
| **A2** | The consumer (`NemoGym` Ray actor in nemo-rl, or `RolloutCollectionHelper` for eval) is a single asyncio event loop draining all responses. | `rollout_collection.py:441` `_post_subroutine`, with one `aiohttp.ClientSession` | Event loop has a fixed tick budget; aggregate inbound byte rate × handler-per-byte cost saturates one core. This is M3. |
| **A3** | HTTP/1.1 + JSON is the right transport for both control and data. | `nemo_gym/server_utils.py:158` `request()` | At MB-scale bodies + 16 K-way fan-in this saturates kernel buffers, GC, and JSON parse cost. gRPC / protobuf / streaming would not help latency directly but would let backpressure propagate properly. |
| **A4** | Inter-server traffic is loopback, so we don't need to worry about MTU / congestion / cross-node bandwidth. | implicit in `127.0.0.1` defaults in configs | True today (A1) and false the moment we shard. |
| **A5** | Default ports + ephemeral-source-port range are sufficient. | `BaseServerConfig` defaults | At 16 K concurrent loopback connections per sub-server we approach `net.ipv4.ip_local_port_range` × `tcp_tw_reuse` interactions. |
| **A6** | Connection re-use is always cheaper than re-establishment. | aiohttp `TCPConnector(limit=…)` defaults | M1 keep-alive race shows this is false when the server side closes idle sockets at 5 s and the client pools them for 15 s. |
| **A7** | `request.body` and `response.body` are small enough that buffering them in process memory is fine. | aiohttp `await response.read()` everywhere | At tens-of-MB × 16 K-concurrent = hundreds-of-GB resident in the head actor. We need to know the actual ceiling. |
| **A8** | Cookies / session middleware are essentially free. | `SimpleServer.setup_session_middleware` in `server_utils.py:474` | At 16 K concurrent sessions per sub-server the middleware allocates per-request session dicts and writes Set-Cookie headers; needs measurement. |
| **A9** | `num_workers=1` for sub-servers is a reasonable default. | `cli.py` server spawn | Single uvicorn worker per sub-server is a per-process GIL/event-loop bottleneck symmetric to A2 on the *server* side. |

The simulation should probe each of A1–A9 at least once.

---

## 3. Three failure axes — what we want to characterize

The user framed this around three axes. Each has a more precise statement, an *expected* breaking point (informed by 2381291 + the assumptions table), and a measurement that will confirm or refute the prediction.

### Axis A — concurrent in-flight requests (the "fan-in" axis)

**Question.** Holding everything else constant, at how many concurrent `/run` invocations does the head actor (consumer) fall over, and which sub-mechanism dominates the failure (FD exhaustion, M3 receive backpressure, M1 keep-alive race, kernel TCP state)?

**Why it matters next.** Doubling rollout batch from 8 K to 16 K is the cheapest scale-up for the next model. We already saw at 8 K that 4 of the 5 defects above transition from "occasional retried error" to "fatal storm" — see §2.4 of 2381291. We need to map the curve, not just point at it.

**Expected dominant mechanism.** M3 (consumer event-loop receive backpressure) above ~16 K concurrent on a single Ray actor, given a moderate per-response body of 100 KB+. Below that, M1 (keep-alive race) and FD exhaustion dominate. **Predict cliff at ~32 K concurrent for ~100 KB bodies on a single 64-core host.**

**Knob to sweep.** `concurrency = {1, 64, 256, 1024, 4 K, 8 K, 16 K, 32 K, 64 K}` parallel `/run` requests held open against a single fixed-config gym deployment.

**Measure.**
- p50 / p99 / p99.9 `/run` latency.
- Successful completion rate.
- Per-error-class count (`ServerDisconnectedError`, `ClientOSError(104)`, `ClientPayloadError`, `TimeoutError`, 4xx/5xx).
- Head-actor process metrics: RSS, FD count, CPU%, event-loop lag (asyncio `loop.time` deltas).
- Per-host kernel: `/proc/net/sockstat` (TCP `inuse`, `tw`, `orphan`), `/proc/net/tcp` count, `/proc/<pid>/net/sockstat` for the head actor.
- ulimit headroom snapshots.

### Axis B — number of co-located sub-servers (the "all-servers-on-head-node" axis)

**Question.** At what count of co-located sub-servers does the head node stop being a viable single-node host? Which resource ceiling do we hit first — FDs, ports, CPUs, RAM, kernel TCP state?

**Why it matters next.** Multimodal post-training will likely require separate sub-servers per modality verifier (image, video, audio), per judge (LLM-as-judge for each rubric), per sandbox kind (code, browser, CAS, multi-tool environments). We have ~20 today; we will plausibly want 50–100. The current `RunHelper` spawns all of them on the head node and assumes the kernel does not care.

**Expected dominant mechanism.**
- At fixed total concurrency, increasing the number of sub-servers **reduces** per-sub-server load but **increases** the number of long-lived listening sockets, uvicorn worker processes, asyncio event loops, GC graphs, and Ray namespace entries. We expect the head node to first exhaust `ulimit -n` (default 1 M soft after `set_ulimit`, but see `getrlimit` headroom), then RAM (each sub-server ~200–500 MB resident at idle), then CPU under load.
- We also expect M3-equivalent contention *inside the head node*: the kernel network stack is shared, and 50 sub-servers each accepting 1 K connections is one shared `softirq` context.
- At fixed *per-sub-server* concurrency, increasing the number of sub-servers scales the load linearly and we hit FDs / ephemeral ports first.

**Knob to sweep.** `n_servers = {1, 4, 16, 32, 64, 128}` synthetic resources servers + matching agents on one host, both at fixed total concurrency and at fixed per-server concurrency.

**Measure.**
- All Axis-A measures, broken down per sub-server.
- Head-node kernel: `lsof | wc -l`, `ss -s`, `cat /proc/sys/fs/file-nr`, `sysctl net.ipv4.ip_local_port_range`, `cat /proc/net/sockstat`.
- Per-sub-server uvicorn process: RSS, FD count, GC pause distribution.
- **Spinup time** (how long `ng_run` takes from start to "all servers ready") as a function of `n_servers`. This is also a real operational concern — see `1968888-768n-vllm-init-optimization.md` for the symmetric problem on the model side.

### Axis C — HTTP transport stress

#### C1 — tool-call / model-hop depth

**Question.** Holding total bytes constant, does increasing the number of tool-call hops per `/run` (i.e. the depth of agent ↔ model ↔ tool ↔ model ↔ tool ↔ … chains) hit a wall before the bytes do?

**Why it matters next.** Long-horizon reasoning + browser tool use + multimodal verification will plausibly produce 20+ hops per rollout where today's distribution is ~1–10. Each hop is a round-trip through `ServerClient.request` → aiohttp → uvicorn → handler → reply. Every hop multiplies the M1 keep-alive race exposure and M3 event-loop tick cost.

**Expected dominant mechanism.** At fixed concurrency (say 8 K), p99 `/run` latency should grow super-linearly in hop count once we cross the keep-alive idle window per hop, because a fraction of hops will hit the 5 s vs 15 s race per round-trip. Also, the head actor's event-loop tick cost per `/run` is `O(hops × parse_cost(body))`; at 20+ hops × 100 KB bodies × 8 K concurrent the loop saturates. **Predict cliff in p99 at ~10–20 hops at 8 K concurrent / 100 KB per hop.**

**Knob to sweep.** `n_hops = {0, 1, 2, 4, 8, 16, 32, 64}`, where the synthetic agent calls the synthetic model `n_hops` times before returning.

**Measure.**
- All Axis-A measures.
- Per-hop latency distribution.
- Inter-hop event-loop lag (instrument the synthetic agent to record `loop.time()` delta per hop).
- Cookies / Set-Cookie header growth: `len(cookies)` per hop. Session middleware writes a new cookie per hop today; this is a known per-hop overhead.

#### C2 — payload size width

**Question.** Holding hop count and concurrency constant, at what per-message and per-`/run` body size does the consumer event loop saturate? Where do JSON parse + orjson dump times become significant? Where does aiohttp's `await response.read()` start timing out or losing connections to receive backpressure?

**Why it matters next.** Multimodal tool I/O (image/video features, base64-encoded media) will produce per-message bodies in the 100 KB – 50 MB range routinely, vs today's text-mostly distribution of ~10 KB – 5 MB. Tens of MB at 16 K concurrent = hundreds of GB nominal residency at the consumer. Even at only a few hundred concurrent, MB-scale bodies hit single-core JSON-parse and copy-cost ceilings.

**Expected dominant mechanism.** M3 (consumer event-loop receive backpressure) at ~10 MB body × 8 K concurrent. JSON parse + copy time at ≥ 50 MB single-message at any concurrency. **Predict cliff in p99 latency at ~5 MB per response at 8 K concurrent.**

**Knob to sweep.** Two independent knobs:
- `body_size_per_hop = {1 KB, 16 KB, 256 KB, 1 MB, 8 MB, 64 MB}`
- `body_distribution = {fixed, lognormal(mean=X, sigma=2)}` to model the long tail that drives 2390264 (where the failure population is dominated by the tail).

**Measure.**
- All Axis-A measures.
- Time spent in `await response.read()` vs handler time vs JSON decode (instrument with `loop.time()` deltas).
- Memory residency of the head actor in MB as a function of `body_size × concurrency`.
- Whether the failure mode at the cliff is timeout, RST mid-body (M3), or OOM.

---

## 4. Simulation harness — design

The harness is three synthetic services + one load driver + one sweep runner.

### 4.1 Components

```
scale_sim/
├── synthetic_resources_server/   # SimpleResourcesServer subclass; configurable verify() latency + body
├── synthetic_agent_server/       # SimpleResponsesAPIAgent subclass; configurable hop count
├── synthetic_model_server/       # SimpleResponsesAPIModel subclass; configurable token-emit rate + body size
├── load_driver.py                # uses RolloutCollectionHelper.run_examples() against synthetic agent
├── instrumentation.py            # ring-buffer timeseries collector (per-server, per-process)
├── sweep_runner.py               # orchestrates ng_run + load_driver across the test matrix
└── configs/                      # one yaml per axis sweep
```

All four services are written as standard nemo-gym sub-servers so that the test exercises the **real `ServerClient`, real aiohttp, real uvicorn, real session middleware, real cookie propagation** — not a different stack. Failures in this harness are failures in nemo-gym.

### 4.2 Synthetic resources server — knobs

```yaml
synthetic_resources_server:
  resources_servers:
    synthetic_resources:
      entrypoint: app.py
      # latency knobs (split I/O-bound from CPU-bound — see §4.8)
      verify_async_latency_ms: 0    # await asyncio.sleep(...) — yields the event loop
      verify_cpu_burn_ms: 0         # busy-loop on time.perf_counter() — holds the event loop
      verify_latency_dist: fixed    # {fixed, lognormal, bimodal}
      verify_latency_lognormal_mu: 0
      verify_latency_lognormal_sigma: 0
      # response body knobs
      verify_body_size_bytes: 1024
      verify_body_dist: fixed       # {fixed, lognormal}
      verify_body_lognormal_mu: 10
      verify_body_lognormal_sigma: 2
      verify_body_shape: flat_padding   # {flat_padding, realistic_messages} — see §4.8.2
      verify_allocation_pattern: single_blob  # {single_blob, many_small}
      verify_pydantic_validation: off   # {off, light, full}
      # failure-injection knobs (for separate experiments)
      inject_500_rate: 0.0          # P(verify returns 500) ∈ [0, 1]
      inject_close_rate: 0.0        # P(server-side `transport.close()` mid-body) — simulates M3 RST
      inject_hang_rate: 0.0         # P(handler awaits forever) — simulates wedged sub-server
```

The body shape and allocation knobs control whether the handler builds a single large bytes blob (cheap; representative of the kernel-transport axis only) or a realistic nested-dict structure of many small allocations (representative of the full cost paid by real handlers). The `cpu_burn_ms` knob is critical: real handlers spend non-trivial time holding the event loop between awaits, and `asyncio.sleep` does not exercise that. See §4.8 for the full rationale.

### 4.3 Synthetic agent server — knobs

```yaml
synthetic_agent:
  responses_api_agents:
    synthetic_agent:
      entrypoint: app.py
      n_hops: 1                     # number of model<->tool round-trips inside /run
      hop_body_size_bytes: 1024     # body of each /v1/responses request to the model
      hop_resource_call_size_bytes: 1024  # body of each /verify hop call to resources server
      hop_body_shape: flat_padding  # {flat_padding, realistic_messages} — see §4.8.2
      include_session_cookies: true # exercise SessionMiddleware vs not
      cpu_burn_ms_per_hop: 0        # CPU work in the agent between hops
      hold_response_in_memory_ms: 0 # keep parsed response alive after read, mimics handlers that don't release until reply
      pydantic_validation: off      # {off, light, full}
```

The agent posts `n_hops` model calls; on each it issues one resources-server `/verify` call. Final return is a `BaseVerifyResponse` with `reward=1.0`. This mirrors the `SimpleAgent` flow in `responses_api_agents/simple_agent/app.py` exactly except that responses are deterministic. The realism knobs (`cpu_burn_ms_per_hop`, `hold_response_in_memory_ms`, `pydantic_validation`) are calibrated against a real benchmark per §4.8.3 before the main sweep runs.

### 4.4 Synthetic model server — knobs

```yaml
synthetic_model:
  responses_api_models:
    synthetic_model:
      entrypoint: app.py
      response_body_size_bytes: 4096
      generation_async_latency_ms: 50    # await asyncio.sleep — simulated tokens-per-second × output length
      generation_cpu_burn_ms: 0           # CPU-side serialization / log-prob extraction stand-in
      generation_latency_dist: fixed
      response_body_shape: flat_padding   # {flat_padding, realistic_messages}
      response_allocation_pattern: single_blob  # {single_blob, many_small}
```

This is the closest to a real vLLM endpoint and is the body-size dominant component for the C2 width axis. As with the resources server, `cpu_burn_ms` and `response_body_shape` are what make the model side actually exercise the JSON encode + memory-residency path the way a real model adapter does.

### 4.5 Load driver

The load driver does **not** invent a new I/O path — it imports `nemo_gym.rollout_collection.RolloutCollectionHelper` and calls `run_examples(examples, semaphore=Semaphore(N))` exactly the way `ng_collect_rollouts` does. Inputs are synthetic JSONL rows shaped like real ones:

```json
{
  "task_index": 0,
  "rollout_index": 0,
  "agent_ref": {"name": "synthetic_agent"},
  "responses_create_params": {
    "input": [
      {"role": "system", "content": "..."},
      {"role": "user",   "content": "<padding to user_input_size_bytes>"}
    ]
  },
  "verifier_metadata": {}
}
```

We ship one driver mode for now (`sync`, the eval / `ng_collect_rollouts` shape), and document that a second mode (`async-rl`, mimicking `NemoGym.run_rollouts` with `semaphore=None`) will be added when we want to reproduce the un-throttled-dispatch failure. The two modes are interesting separately: eval has inherent step-boundary throttling; RL does not. Defect #5 from 2381291 makes this the one place where eval and training meaningfully diverge.

### 4.6 Instrumentation

Three layers, written once and reused across all sweeps:

1. **Per-process metrics ring buffer.** A small thread inside each sub-server samples (every 100 ms): RSS, FD count, CPU%, asyncio event-loop lag. Dumps to a file at shutdown.
2. **Per-request metrics.** The synthetic agent stamps `loop.time()` at start, before each model hop, after each model hop, before each resources call, after, and at end. Embedded in the response JSON so the load driver sees the breakdown per rollout.
3. **Per-host kernel metrics.** A standalone watcher process samples (every 1 s): `/proc/net/sockstat`, `/proc/sys/fs/file-nr`, `/proc/loadavg`. Writes to one CSV per run.

Outputs are merged into one parquet per sweep run for offline analysis.

### 4.7 Sweep runner

A small Hydra-driven script that:

1. Generates the YAML for one configuration (`n_servers`, `n_hops`, body sizes).
2. Launches `ng_run` to bring everything up.
3. Invokes the load driver to run the configured load.
4. Tears down.
5. Records all artifacts under `tools/scale_sim/results/<run_id>/`.

The key constraint: each row of the test matrix is one `ng_run` invocation, because we need clean process state between runs (FD pool, kernel TCP state, etc).

### 4.8 Are stub services actually using the resources we want to measure?

The natural worry with a synthetic harness is that `await asyncio.sleep(latency)` and `bytes(N)` of zeros bypass everything that real benchmarks pay for, and we end up measuring an idealized lower bound on contention rather than a representative cost. This is partly true and partly false; being explicit about which is which is the difference between a useful simulation and a misleading one.

#### 4.8.1 What stubs DO exercise faithfully (free, no extra knobs needed)

These all live below the application layer. The kernel, aiohttp, and uvicorn cannot tell the difference between a synthetic and a real workload, so they generate real load:

| Subsystem | Why it's exercised faithfully |
| --- | --- |
| Kernel TCP stack | every `ServerClient.post` is a real TCP connect (or pool reuse), real `SYN`/`ACK`/`FIN`/`RST`, real ephemeral-port consumption, real entries in `/proc/net/tcp`, real TIME_WAIT table growth |
| File descriptors | every accepted connection consumes one server-side FD and one client-side FD; the host hits its `ulimit` regardless of payload content |
| Per-socket kernel buffers (`rmem`, `wmem`) | bodies are real bytes traversing real send/receive buffers; if the consumer is slow, the sender's `transport.write` blocks on `flow.drain()` regardless of whether bytes are random or zeros |
| aiohttp connection pool, keep-alive, idle race | exact same `TCPConnector`, same pool reuse logic, same idle window. Mechanism M1 (pre-fix) and the post-fix version are reproducible here |
| uvicorn accept loop, worker count, graceful-shutdown timer | the synthetic services run real uvicorn — same accept queue, same worker-process model, same lifecycle timers |
| `RolloutCollectionHelper.run_examples` dispatch shape | the load driver imports and calls the real function — same `_post_subroutine`, same retry layering, same semaphore semantics |
| Session middleware + cookie propagation | enabled by default in `SimpleServer`; each `/run` walks real Starlette middleware, allocates a real session dict, signs and writes a real cookie |
| HTTP/1.1 parsing on the server side | uvicorn uses `httptools` regardless of body content; parse cost is real |
| JSON request body bytes traveling over the wire | bytes are bytes |

These are exactly the subsystems that drive M3, the keep-alive race, FD exhaustion, and TW-table fill — i.e. the failure modes the simulation primarily targets. So stubs are sufficient for these.

#### 4.8.2 What stubs DO NOT exercise (need explicit knobs to recover)

These all live in application code or above the JSON parse layer. Stubs systematically under-stress them, and we need explicit knobs to bring them back to representative levels:

| Real cost | Why stubs miss it | How we recover it |
| --- | --- | --- |
| **Handler CPU time between awaits** | `await asyncio.sleep(latency_ms)` yields immediately; the event loop runs another coroutine. Real handlers do CPU work (parse, validate, business logic) between awaits, *holding the event loop*. | New knob `cpu_burn_ms` that does `time.perf_counter()` busy-loop, not `sleep`. Mix sleep + burn (`async_latency_ms` for I/O-bound waits, `cpu_burn_ms` for CPU-bound work) so we can sweep the ratio. |
| **JSON encode/decode of nested structures** | `bytes(N)` packed into a single string field of the response is one cheap orjson copy. Real bodies are nested dicts (lists of messages, each with role/content, with tool_calls each with args, with usage, etc). orjson cost on real shapes is ~5–20× what it is on a single big string at the same byte count. | New knob `body_shape ∈ {flat_padding, realistic_messages}`. `realistic_messages` builds a `messages: list[dict]` of the right total size, mimicking the production `responses_create_params.input` shape exactly. |
| **Pydantic validation cost** | `BaseVerifyResponse.model_validate(...)` walks real schemas with discriminated unions, type coercion, and validator functions. Stubs that just return a dict without validation skip this. | New knob `pydantic_validation ∈ {off, light, full}` where `full` runs the request body through the same `NeMoGymResponseCreateParamsNonStreaming.model_validate(...)` real handlers do. |
| **GC pressure from many small allocations** | a single `bytes(N)` is one allocation; a real response is hundreds of small dicts/strings. CPython refcount + cyclic GC behavior is very different. | New knob `allocation_pattern ∈ {single_blob, many_small}`. `many_small` builds the body as `{"chunks": [<small_dict> for _ in range(N // chunk_size)]}` to put representative pressure on the allocator. |
| **Per-request memory residency** | `bytes(N)` of zeros may end up backed by the kernel zero page on the way out (no actual RSS) and is collected immediately on the way in. Real bodies stick around in the parsed-dict form for the duration of the handler. | The `realistic_messages` body shape forces a real parsed structure to be held in memory during the handler. We also add a `hold_response_in_memory_ms` knob on the synthetic agent that keeps a reference alive for a configurable duration after parse, mimicking handlers that don't release until they reply. |
| **Tokenization-equivalent CPU on the model side** | real model adapters do tokenization, log-prob extraction, and serialization of token IDs. Stubs skip this entirely. | The synthetic model gets the same `cpu_burn_ms` + `body_shape` knobs. We do NOT try to match real-model latency; we just give ourselves a knob to dial in CPU work that sits in the same place in the call graph. |

#### 4.8.3 Calibration: how we know the harness is "close enough"

Knobs are only useful if we have a way to set them. Plan:

1. **Pick one real reference benchmark.** Easiest target: `example_single_tool_call` paired with `vllm_model` against a small real model (e.g. a 1B served on the same host). Cheap to run; covers the same call graph as production.
2. **Run the reference under a fixed concurrency** (say 256) and record the baseline timeseries: p50/p99 latency, CPU%, RSS, event-loop lag of the head actor, JSON encode/decode times.
3. **Run the synthetic harness configured to hit the same `/run` body bytes and same hop count**, then sweep `cpu_burn_ms`, `body_shape`, and `pydantic_validation` until the synthetic timeseries matches the reference within ~20 % on each metric.
4. **Lock those knob values as the "realistic baseline"** and reuse them across the §5 sweep matrix as one of the configurations. The other configurations explicitly stress one knob (zero CPU work / single-blob body / no validation) so we can attribute outcomes to specific costs.

Without calibration the sweep numbers describe a system that doesn't exist; with calibration they describe a system that is provably within 20 % of the real one along the dimensions we measured. That's the target.

#### 4.8.4 Cross-checks — running real benchmarks under the load driver

A complementary sanity check, cheap to bolt on after the harness exists: point the load driver at one of the existing real benchmarks (e.g. `simple_agent` + `example_single_tool_call` + `vllm_model`) instead of the synthetic triplet, and run a low-concurrency sweep. We expect:

- The shape of the cliffs (where p99 starts climbing, where errors appear) to match the synthetic harness when knobs are calibrated as in §4.8.3.
- The absolute numbers may differ by a constant factor due to vLLM scheduler latency variance; that's fine.

If the shapes diverge (say the synthetic harness hits a cliff at concurrency 32 K but the real benchmark hits one at 8 K), that is itself a finding and tells us the synthetic harness is missing a cost we haven't added a knob for. Treat any such divergence as a follow-up on this document.

#### 4.8.5 What we deliberately do NOT try to make realistic

To keep the harness focused, we leave these for follow-ups:

- **Real model serving variance.** No effort to mimic vLLM's scheduler tail latency. We want a controllable system, not a high-fidelity replay.
- **Real disk / lustre I/O on the dataset path.** The harness reads its synthetic JSONL once at startup and holds it in memory.
- **Real GPU-attached behavior.** Out of scope; this is a CPU/network simulation.
- **Real flaky-network failure injection.** The `inject_500_rate` / `inject_close_rate` / `inject_hang_rate` knobs are coarse stand-ins. If we want richer failure injection (jitter, partial bodies, slow loris) it is a separate harness layer.

---

## 5. Test matrix

Three sweeps. Defects #1–#4 from 2381291 are closed in the codebase, so we sweep against the current main branch only — no with/without-fixes split. We do split on **defect #5** (RL-side dispatch semaphore on/off) where it matters, because that is the one open defect and we want to know whether closing it alone is sufficient to push the M3 cliff out far enough for the next model.

### 5.1 Axis A — fan-in

| Knob | Values |
| --- | --- |
| `concurrency` | 1, 64, 256, 1 K, 4 K, 8 K, 16 K, 32 K, 64 K |
| `n_servers` | 4 (1 model, 1 resources, 1 agent, 1 head) |
| `n_hops` | 1 |
| `body_size_per_hop` | 16 KB |
| `verify_latency_ms` | 100 |

9 points × {semaphore-on, semaphore-off} = 18 runs. Each run terminates after 5 minutes of steady-state load or 10 K completed `/run`s, whichever comes first. The semaphore split here is specifically to measure whether closing defect #5 is sufficient at high concurrency.

### 5.2 Axis B — sub-server count

| Knob | Values |
| --- | --- |
| `concurrency` | 4 K (fixed) |
| `n_servers` | 4, 8, 16, 32, 64, 128 (each "+1 server" adds one resources + one agent) |
| `n_hops` | 1 |
| `body_size_per_hop` | 16 KB |
| `verify_latency_ms` | 100 |

The 4 K concurrency is split evenly across the agent-resources pairs, so per-sub-server load decreases as `n_servers` grows. We run a second variant where per-sub-server concurrency is held at 256 (so total concurrency grows with `n_servers`) to disentangle "sub-servers cost something just by existing" from "sub-servers cost something proportional to their load."

12 points, single variant (semaphore on, since this axis is about head-node ceilings and not about dispatch shape).

### 5.3 Axis C — depth × width

The cross is interesting; we don't want to sweep them independently because we expect compound effects.

| Knob | Values |
| --- | --- |
| `concurrency` | 4 K, 8 K |
| `n_servers` | 4 |
| `n_hops` | 1, 4, 16, 64 |
| `body_size_per_hop` | 16 KB, 256 KB, 4 MB, 32 MB |

2 × 4 × 4 = 32 cells, single variant (semaphore on). We will probably prune to ~16 once we see the shape of the surface — many cells of the cross are uninteresting (e.g. 64 hops × 32 MB × 8 K is obviously off the cliff).

Total: ~62 runs. At ~10 minutes wall-clock each (5 min spinup + 5 min load + teardown), the full sweep is ~10 hours on one host. Comfortable for an overnight run.

---

## 6. Expected breakage map (predictions to falsify)

We write predictions explicitly so the experiment can refute them.

M1 is closed by the keep-alive realignment (defect #1 fix), so we no longer expect to see it in the failure population. If we do, that is itself a finding. The remaining live mechanisms are M2 (config-only, transparently retried), M3 (consumer event-loop receive backpressure), kernel-level resource exhaustion (FDs, ports, TW table, send buffers), and host RAM.

| Axis | Concurrency | Bodies | Hops | Predicted breakage point | Predicted dominant mechanism |
| --- | --- | --- | --- | --- | --- |
| A | 16 K | 16 KB | 1 | semaphore-off run shows step-boundary burst saturating event-loop tick budget; semaphore-on run is clean | M3 amplified by defect #5 |
| A | 32 K | 16 KB | 1 | Event-loop lag crosses 100 ms; FD count crosses 30 K (semaphore-on) | M3 + FDs |
| A | 64 K | 16 KB | 1 | OS rejects new connections (`EMFILE`) or kernel TW table fills | kernel |
| B | 4 K total / 64 servers | 16 KB | 1 | Spinup time > 90 s (uvicorn cold-start serialized) | spinup, not load |
| B | 256 / server × 64 servers = 16 K total | 16 KB | 1 | Same M3 cliff as A at 16 K — confirms M3 is consumer-side, agent-independent | M3 |
| B | 4 K total / 128 servers | 16 KB | 1 | Head-node RAM > 64 GB at idle (128 × 500 MB) | RAM |
| C1 | 8 K | 16 KB | 16 | p99 doubles vs hops=1 because of compounded per-hop session-middleware + cookie cost | Session middleware + per-hop transport |
| C1 | 8 K | 16 KB | 64 | p99 grows super-linearly; per-hop event-loop lag > 50 ms | M3 amplified by hops |
| C2 | 8 K | 4 MB | 1 | M3 cliff — mid-body RST volume rises sharply | M3 |
| C2 | 8 K | 32 MB | 1 | OOM in head actor (~256 GB nominal residency) | RAM |
| C2 | 4 K | 32 MB | 4 | TCP_NOMEM / send buffer saturation on at least one sub-server | kernel + M3 |

Failure-mode taxonomy used above is the same as 2381291 §2.1.

---

## 7. What this simulation deliberately does **not** model

For honesty, since each of these is a real pressure that the real system has and the simulation will not surface:

- **Cross-node networking.** Today everything is loopback. Once we shard the head actor across multiple nodes (the structural fix for M3), MTU / NIC / TC queue / RDMA-or-not become the bottleneck and this harness will need a multi-node mode.
- **GIL contention** inside any one process. We hold uvicorn `num_workers=1` because that is the production default; we don't try to model what happens with `num_workers=N`. Side investigation, not this one.
- **Real model serving variance.** No vLLM, no scheduler. The C1 hop axis is the closest we get to multi-turn behavior.
- **Real sub-server `Semaphore` meltdowns** (M2). The `inject_hang_rate` knob is a synthetic stand-in but doesn't reproduce the multiprocessing-Manager-spawn cost in `code_gen`. M2 is benchmark-specific and we already know its remedy (`num_processes: 2048`).
- **Ray actor cross-talk** with the rest of nemo-rl (refit, weight transfer, replay buffer pressure). Out of scope here; covered by `async-trajectory-parallelism-model.md`.
- **Trainer-side dynamics** like step-boundary bursts. We can re-introduce these via the `async-rl` driver mode in a follow-up.

---

## 8. Deliverables

1. **`tools/scale_sim/`** — the synthetic harness, in nemo-gym (PR-able as a `synthetic_*` triplet alongside `example_single_tool_call`). Keep it under `tools/` rather than `resources_servers/` so the pre-commit `update-readme-table` hook doesn't add it to the public benchmarks table.
2. **`tools/scale_sim/sweep_runner.py`** — Hydra entry point that runs §5's matrix end-to-end.
3. **`tools/scale_sim/analyze.ipynb`** — minimal analysis notebook that takes the sweep parquet and produces:
   - per-axis cliff plots (concurrency-vs-p99, n_servers-vs-spinup, body-size-vs-event-loop-lag, hop-count-vs-p99).
   - failure-mode attribution table.
4. **`investigations/nemo-gym-scale-simulation-results.md`** — companion writeup with the cliff plots and the actual measured breakage points compared to §6 predictions, plus a prioritized list of structural fixes for the next-model bringup.

Stretch deliverable, after the first sweep ships:

5. **A reduced-form analytic model** (one page) that gives a back-of-envelope ceiling for `(concurrency, body_size, hops, n_servers, n_head_actors)` so we can plan future runs without re-running the sweep. The form will look something like `usable_concurrency ≈ min(FD_budget, RAM_budget / body_size, event_loop_budget / (hops × parse_cost(body_size)))` with constants fitted from the sweep data.

---

## 9. Decisions

Resolved 2026-04-27.

| # | Question | Decision |
| --- | --- | --- |
| 1 | Where does the harness live? | **nemo-gym**, under `tools/scale_sim/` (not `resources_servers/`, to keep it out of the public benchmarks table and the per-server CI / pre-commit hooks). |
| 2 | Single host or multi-host for first run? | **Single host.** Multi-host is the structural-fix experiment, not this one — and we want a clean baseline against the production "all sub-servers on the head node" assumption first. |
| 3 | Defect #5 (RL-side dispatch semaphore) — sweep on/off? | **Yes, on Axis A only.** Other axes run semaphore-on. Defects #1–#4 are closed in main and not a sweep variable. |
| 4 | HTTP/1.1 vs HTTP/2 swap? | **Out of scope.** Logged as a follow-up. First sweep characterizes the shipping system, not a refactor. |
| 5 | Replicate the "single Ray actor head consumer" assumption? | **Yes.** This is the production shape and the source of M3. The sharded-consumer experiment is the structural fix and a separate follow-up. |
| 6 | Failure / retry signal — what do we measure and when do we early-stop? | **Add explicit retry instrumentation** (see §9.1 below). Early-stop on `retry_rate > 30 %` OR `failure_rate > 10 %` OR `head_actor_rss > 90 %` of host RAM. Both rates are computed over a sliding 30 s window so a brief retry storm at startup doesn't kill the run. |
| 7 | Where do artifacts go? | **Local filesystem** at `tools/scale_sim/results/<git_sha>/<run_id>/`. Each run is one directory containing the merged parquet, per-process metrics, kernel-watcher CSV, the resolved YAML config, and stdout/stderr per sub-server. S3 / WandB upload is a follow-up. |

### 9.1 Retry instrumentation — what to log and where

Today nemo-gym has retry counters in two places, both incomplete:

- `nemo_gym/server_utils.py:146-147` has process-global `_NUM_SERVER_DISCONNECTED_ERROR` and `_NUM_CLIENT_OS_ERROR` counters that print to stdout every 100 hits. No structured output, no per-URL breakdown, no notion of "what fraction of `/run` calls needed at least one retry."
- `nemo_gym/rollout_collection.py:450-452` retries `ClientPayloadError` with a `print(f"Retrying /run for agent={…}")` and no counter.

Neither tells us "what % of rollouts needed a retry," which is the headline number the user asked for. We add the following, all behind a single `retry_metrics_enabled: true` config flag so it ships off by default:

1. **Per-`/run` retry attempt counter.** Wrap `RolloutCollectionHelper._post_subroutine` so each completed rollout records `n_attempts`, `total_retry_wait_s`, and a list of error classes encountered. Emit one line per rollout to a JSONL file at `tools/scale_sim/results/<run_id>/per_rollout_retries.jsonl`.
2. **Aggregate retry summary printed on shutdown.** At the end of every sweep run, dump:
   - `% of rollouts that needed ≥ 1 retry`
   - `% that needed ≥ 2 retries`, `≥ 5`, `≥ 10`
   - mean / p50 / p99 / max retry count per rollout
   - retry counts grouped by error class (`ServerDisconnectedError`, `ClientOSError`, `ClientPayloadError`, `TimeoutError`, 5xx, 4xx)
   - retry counts grouped by target sub-server URL (since 2381291 §2.3 showed errors are agent-uniform once normalized; we want to confirm this still holds after the fixes)
3. **Aggregate retry summary printed periodically during the run** (every 30 s) so the early-stop trigger has a metric to fire on. Same shape as the shutdown dump but over the last 30 s window.
4. **Promote the existing `_NUM_SERVER_DISCONNECTED_ERROR` / `_NUM_CLIENT_OS_ERROR` counters to per-process structured metrics** that get written to the per-process metrics ring buffer (§4.6 layer 1), so we can correlate retry storms with FD count / event-loop lag / kernel state on the same timeline.

This instrumentation is also useful outside the simulation — the same counters answer "what fraction of rollouts in real RL training are silently retrying?" which today we cannot answer from production logs. Worth landing in nemo-gym proper as part of step 2 in §10 below, not just in `tools/scale_sim/`.

---

## 10. Next steps

1. Land the retry-instrumentation knobs from §9.1 in nemo-gym proper (per-rollout retry counters, aggregate summaries, periodic 30 s window dump). Cheap, useful outside the simulation, and a precondition for the early-stop logic. ~0.5 day.
2. Stand up `synthetic_resources_server` + `synthetic_agent_server` + `synthetic_model_server` under `tools/scale_sim/` (~1 day; mostly copying `example_single_tool_call`).
3. Stand up the load driver (~0.5 day; mostly importing `RolloutCollectionHelper`).
4. Stand up the sweep runner + instrumentation (~1 day).
5. Run the §4.8.3 calibration step against `example_single_tool_call` + `vllm_model` + a small real model on the same host. Lock the realistic-baseline knob values. ~0.5 day.
6. Run the §5.1 Axis-A sweep once on a single host (~3 hours wall-clock); validate the harness against §6 predictions for the first three rows of the table (these are the cheapest to confirm).
7. Iterate on knob defaults and ranges based on what we see.
8. Run the full §5 matrix overnight.
9. Write up `nemo-gym-scale-simulation-results.md` and circulate.

Total estimated time to first signal: ~1 week of focused work. Total to publishable matrix: ~2 weeks.

---

## Appendix A — Concrete config sketch for one Axis-A run

For reviewability, here is the fully-resolved config for one cell of §5.1 (concurrency=8 K, n_servers=4, n_hops=1, body_size=16 KB):

```yaml
# tools/scale_sim/configs/sweep_axis_a_8k.yaml

global_aiohttp_connector_limit: 102400
global_aiohttp_connector_limit_per_host: 16384

synthetic_resources_server:
  resources_servers:
    synthetic_resources:
      entrypoint: app.py
      verify_latency_ms: 100
      verify_body_size_bytes: 16384
      host: 127.0.0.1
      port: 18001

synthetic_model:
  responses_api_models:
    synthetic_model:
      entrypoint: app.py
      response_body_size_bytes: 16384
      generation_latency_ms: 50
      host: 127.0.0.1
      port: 18002

synthetic_agent:
  responses_api_agents:
    synthetic_agent:
      entrypoint: app.py
      n_hops: 1
      hop_body_size_bytes: 16384
      resources_server: {type: resources_servers, name: synthetic_resources_server}
      model_server:     {type: responses_api_models, name: synthetic_model}
      datasets:
        - name: synthetic_train
          type: train
          jsonl_fpath: tools/scale_sim/data/synthetic_8k.jsonl
          num_repeats: 1
      host: 127.0.0.1
      port: 18003

# Load driver knobs
scale_sim:
  concurrency: 8192
  total_requests: 100000
  early_stop_failure_rate: 0.10
  early_stop_wall_clock_s: 600
  output_dir: tools/scale_sim/results/${git_sha}/axis_a_c8k_b16k_h1
```

Each row of §5 maps to one such config + one invocation of the sweep runner.

## Appendix B — Pointers into the codebase

For the next person reading this, the two files to internalize before writing the harness:

- `nemo_gym/server_utils.py` — `request()`, `ServerClient`, `SimpleServer`, `HeadServer`, `set_global_aiohttp_client`, `set_ulimit`. Every concurrency knob the harness will care about lives here.
- `nemo_gym/rollout_collection.py` — `RolloutCollectionHelper.run_examples`, `_post_subroutine`. The synthetic load driver should reuse `run_examples` rather than re-implement the dispatch loop.

And the prior investigation that everything in §1 and §6 is grounded in:

- `investigations/2381291-nemo-gym-timeouts-connection-resets.md` — five defects, three mechanisms, evidence from two real runs.
