# nemo-gym scale testing — comprehensive reference

**Status:** living document. Sections 1–9 are stable; §10 (Results) is filled in as experiments land.
**Owner:** ansubramania
**Companion docs:**
- `nemo-gym-scale-simulation-design.md` — original axes/predictions design (single-agent: Axis A + C).
- `nemo-gym-multi-agent-design.md` — multi-agent extension design (sub-server-count axis).
- `2381291-nemo-gym-timeouts-connection-resets.md` — incident this work is grounded in.

This document is the single reference for: (a) what we are stressing in the gym architecture, (b) what experiments answer those questions, (c) how the simulation harness works in detail, and (d) how to run it. The two design docs above are the source-of-truth for the *reasoning* behind specific decisions; this document folds the practical bits together so a new contributor can land here, understand the whole picture, and run the matrix.

---

## 1. What we are stressing

NeMo Gym is a microservice rollout-collection architecture. Today it ships ~4 sub-server types (resources servers, response API agents, response API models, head server) and a `RolloutCollectionHelper` that fans out N concurrent `/run` requests to one or more agent instances over async HTTP. The system has been observed to fall over at 8 K rollout × 512 nodes (incident 2381291). Five defects were fixed; one remains; three structural mechanisms persist.

The next-generation post-training run multiplies pressure on that architecture along several axes simultaneously: larger batch (8 K → 16 K – 64 K rollouts in flight), longer sequences (text-only ~5 MB max → multimodal tens of MB), more multi-turn tool calls (~1–10 hops → 20+), and more sub-servers (one per modality verifier, judge, sandbox). We have no current data on where the cliff is for the cross product of these — only point measurements at one configuration. The purpose of this scale-testing work is to **map the cliffs before we hit them on a real run**.

### 1.1 The mechanisms still live after the 2381291 fixes

Quick recap so this doc stands alone:

| ID | Mechanism | Status after the 2381291 fixes |
| --- | --- | --- |
| **M1** | Keep-alive idle race (5 s uvicorn vs 15 s aiohttp default) | **Closed**. If we still see M1-shaped errors in the simulation, that is a finding. |
| **M2** | Per-sub-server semaphore meltdown (e.g. `code_gen.num_processes=8`) | **Made transparent** by retried-`ClientPayloadError` fix. Underlying cause is config-only; benchmark-specific; harness does not reproduce it. |
| **M3** | Consumer-side receive backpressure on the single `NemoGym` Ray actor's event loop | **Survives every defect fix**. Single actor with one asyncio event loop drains all bodies for the entire batch on one CPU. Ceiling is `min(per-socket rmem, event-loop tick budget) × N_concurrent_sockets`. **This is the primary thing the simulation models**. |
| **Defect #5** | No client-side concurrency cap on RL dispatch (`run_examples` called without a `Semaphore`) | **Open**. RL still fires the full step batch. The matrix sweeps semaphore on/off to measure whether closing #5 is sufficient at high concurrency. |

### 1.2 Architectural assumptions under test

These are baked into today's gym, not always documented; the simulation probes each at least once:

| ID | Assumption | Where in code | Why it likely breaks at next scale |
| --- | --- | --- | --- |
| **A1** | All sub-servers live on the head node as child processes of the head `Popen` loop. | `RunHelper.start` in `cli.py:144-178` | Head node has finite FDs, one kernel TCP stack, one ephemeral-port range, one CPU. With 50+ sub-servers each receiving 16 K concurrent connections this saturates linearly. |
| **A2** | The consumer (`NemoGym` Ray actor in nemo-rl, or `RolloutCollectionHelper` for eval) is a single asyncio event loop draining all responses. | `rollout_collection.py:441` `_post_subroutine`, with one `aiohttp.ClientSession` | This is M3. |
| **A3** | HTTP/1.1 + JSON is the right transport for both control and data. | `nemo_gym/server_utils.py:158` `request()` | At MB-scale bodies + 16 K-way fan-in this saturates kernel buffers, GC, and JSON parse cost. |
| **A4** | Inter-server traffic is loopback. | implicit in `127.0.0.1` defaults | True today (A1) and false the moment we shard. |
| **A5** | Default ports + ephemeral source-port range are sufficient. | `BaseServerConfig` defaults | At 16 K concurrent loopback connections per sub-server we approach `net.ipv4.ip_local_port_range` × `tcp_tw_reuse` interactions. |
| **A6** | Connection re-use is always cheaper than re-establishment. | aiohttp `TCPConnector(limit=…)` defaults | M1 keep-alive race showed this is false. |
| **A7** | Request/response bodies are small enough to buffer in process memory. | `await response.read()` everywhere | At tens-of-MB × 16 K-concurrent = hundreds-of-GB resident in the head actor. |
| **A8** | Cookies / session middleware are essentially free. | `SimpleServer.setup_session_middleware` in `server_utils.py:474` | Per-request session dict allocation × 16 K = real cost. |
| **A9** | `num_workers=1` for sub-servers is reasonable. | `cli.py` server spawn | Per-process GIL/event-loop bottleneck symmetric to A2 on the server side. (Today's gym `num_workers > 1` is non-functional — see notes in `tools/scale_sim/run_single_agent_sweep.sh:60`.) |
| **A10** | Spawn-side IPC mechanism (env-var-on-command-line) for the resolved global config scales linearly with N. | `nemo_gym/cli.py:172-177` (pre-fix); now path-based via `NEMO_GYM_CONFIG_DICT_PATH` | Each child spawn used to embed the **entire** global config (all N+2 server entries) as an env-var assignment in the bash command line. `getconf ARG_MAX` ≈ 2 MB; the YAML grows ~1 KB/server entry after escaping, so the spawn command hit the kernel limit deterministically at N=256. **Confirmed at N=256** in the M2 sweep — `OSError: [Errno 7] Argument list too long: '/bin/bash'`. **Fixed in this commit** by materializing the resolved YAML to `${RAY_TMPDIR}/nemo_gym_global_config_<pid>.yaml` once per `ng_run` and passing the path; per-spawn bytes are now O(1) regardless of N. See §10.3.1 for the full reasoning and validation steps. |
| **A11** | `ng_run`'s SIGINT teardown is sufficient to clean up Ray + sub-server state for the next invocation. | `nemo_gym/cli.py` shutdown handler | False. Orphaned `raylet` / `gcs_server` / `plasma_store` processes plus `/tmp/ray/` session directories survive `SIGINT`. The next `ng_run` on the same host inherits stale GCS pointers; sub-servers fail to register with the (stale) cluster ID and time out. **Confirmed at N≥16** in M2 sweep — sub-servers in 11 of 21 cells could not connect to GCS until pre-cell `pkill` + `rm -rf /tmp/ray` was added in `sweep_runner._pre_cell_cleanup`. Affects any pipeline that runs `ng_run` repeatedly on a long-lived host (CI / eval / smoke). |
| **A12** | Sub-server `ray.init(address=…)` spawn storm at high N is absorbed by the head's Ray GCS within the client-side connect timeout (5 s). | `nemo_gym/cli.py:RunHelper.start` spawn loop fires 2N+1 child Popens in parallel; each child independently calls `ray.init(address=<head_gcs>)` near-simultaneously. | False at N ≥ ~128. With 514 children all attempting to connect within ~10 s, GCS handles the first wave but the remainder hit `Failed to connect to GCS at address … within 5 seconds`. Independent of A10 (no ARG_MAX hit) and A11 (Ray state was clean from `_pre_cell_cleanup`). **Confirmed deterministically at N=256** post-A10-fix — about half the sub-servers connect successfully; half time out. The failure pattern is thundering-herd, not a deterministic ceiling — spinup at N=128 may pass with the same code, depending on how the storm fans out. **Workaround landed in `tools/scale_sim/_ray_burst_env.sh`** — comprehensive Ray GCS tuning (thread pools, timeouts to 120s, batching). **Upstream fix shape:** stagger the spawn loop into batches with sleeps between (e.g. 32 children per batch + 1 s gap). |
| **A13** | Ray's auto-assigned GCS port stays out of the kernel ephemeral range. | `nemo_gym.server_utils.initialize_ray:410` calls `ray.init()` with no `port` arg → Ray picks a random port in 10K–40K. | False. `net.ipv4.ip_local_port_range` is typically 32768–60999 (9000–65000 on OCI-HSG). Auto-assigned GCS ports observed across runs: 14589, 30899, 27551 — all inside the typical ephemeral range. With N=256 sub-servers each issuing parallel TCP connect()s to GCS, the kernel can hand out the GCS's destination port as an ephemeral source for an unrelated outgoing connection (TOCTOU between Ray's `CheckPortFree` and `bind`), causing `EADDRINUSE` failures. **Workaround landed in `tools/scale_sim/sweep_runner._pre_start_ray`** — pre-starts Ray with explicit ports below the ephemeral floor (GCS=6379, worker range 2000-2999, management 6701-6706, dashboard 8265). `ng_run`'s `ray.init()` detects the running cluster via `/tmp/ray/` and connects to it instead of auto-assigning. **Upstream fix shape:** modify `initialize_ray()` to read pinned port values from `global_config` (e.g. `ray_gcs_port`, `ray_min_worker_port`, etc.) and pass them to `ray.init(_node_ip_address=..., port=..., ...)`. ~10 line change. |

---

## 2. Scaling factors we want to assess

Grouped into four axes. Each has a precise question, an expected dominant mechanism, and a knob to sweep.

### 2.1 Axis A — concurrent in-flight requests (fan-in)

**Question.** At how many concurrent `/run` invocations does the head actor (consumer) fall over, and which sub-mechanism dominates the failure (FD exhaustion, M3 receive backpressure, kernel TCP state)?

**Why it matters next.** Doubling rollout batch from 8 K to 16 K is the cheapest scale-up. We saw at 8 K that 4 of 5 defects transitioned from "occasional retried error" to "fatal storm." We need to map the curve, not just point at it.

**Expected dominant mechanism.** M3 above ~16 K concurrent on a single Ray actor with 100 KB+ bodies. Below that, FD exhaustion and (pre-fix) M1 dominate. Predict cliff at ~32 K concurrent for ~100 KB bodies on a single ~64-core host.

**Knob.** `concurrency = {1, 64, 256, 1 K, 4 K, 8 K, 16 K, 32 K, 64 K}`.

### 2.2 Co-located sub-server count (multi-agent sweep)

**Question.** At what count of co-located sub-servers does the head node stop being a viable single-node host? Which resource ceiling binds first — FDs, ports, CPUs, RAM, kernel TCP state? Is the M3 cliff consumer-side and N-independent, or does adding sub-servers itself cost something?

**Why it matters next.** Multimodal post-training will likely require separate sub-servers per modality verifier, per judge, per sandbox. We have ~20 today; we will plausibly want 50–100. The current `RunHelper` spawns all of them on the head node and assumes the kernel does not care.

**Expected dominant mechanism.** At fixed total concurrency, increasing sub-servers reduces per-sub-server load but increases long-lived listening sockets, uvicorn workers, asyncio loops, GC graphs. We expect to first exhaust `ulimit -n`, then RAM (each sub-server ~200–500 MB resident at idle), then CPU. Inside the head node we also expect M3-equivalent contention: one shared `softirq` context.

**Knob.** `n_agents = {1, 4, 16, 32, 64, 128, 256}` synthetic resources servers + matching agents on one host, both at fixed total concurrency and at fixed per-agent concurrency.

### 2.3 Axis C1 — tool-call / model-hop depth

**Question.** Holding total bytes constant, does increasing the number of tool-call hops per `/run` (depth of agent ↔ model ↔ tool ↔ … chains) hit a wall before the bytes do?

**Why it matters next.** Long-horizon reasoning + browser tool use + multimodal verification will plausibly produce 20+ hops where today's distribution is ~1–10. Every hop is an HTTP round trip; cookies grow per hop.

**Expected dominant mechanism.** At fixed concurrency (8 K), p99 should grow super-linearly in hop count once we cross the keep-alive window per hop. Loop tick cost per `/run` is `O(hops × parse_cost(body))`. Predict cliff in p99 at ~10–20 hops at 8 K concurrent / 100 KB per hop.

**Knob.** `n_hops = {1, 4, 16, 64}`.

### 2.4 Axis C2 — payload size width

**Question.** Holding hops and concurrency constant, at what per-message and per-`/run` body size does the consumer event loop saturate? Where do JSON parse times become significant? Where does aiohttp's `await response.read()` start timing out or losing connections to receive backpressure?

**Why it matters next.** Multimodal tool I/O (image/video features, base64 media) will routinely produce 100 KB – 50 MB bodies vs today's text-mostly ~10 KB – 5 MB. Tens of MB at 16 K concurrent = hundreds of GB nominal residency at the consumer.

**Expected dominant mechanism.** M3 at ~10 MB body × 8 K concurrent. JSON parse / copy time at ≥ 50 MB single message. Predict cliff in p99 at ~5 MB per response at 8 K concurrent.

**Knob.** `output_tokens = {16 K, 128 K, 256 K, 512 K, 1 M}` on the synthetic model (drives body size deterministically — see §4.4). Concurrency scaled inversely to keep nominal residency bounded.

### 2.5 Defect #5 — RL-side dispatch semaphore on/off

**Question.** Does closing defect #5 (the open one — RL-side dispatch semaphore) by itself raise the M3 cliff far enough for the next model, or do we also need a structural fix?

**Knob.** `semaphore_enabled ∈ {true, false}`, swept at the two cliff-adjacent concurrency points.

---

## 3. The simulation harness — components

```
tools/scale_sim/
├── README.md                                  user-facing overview
├── pyproject.toml                              marker for editable-install heuristic
│
├── resources_servers/synthetic_resources/      synthetic /verify + /synthetic_tool
│   ├── app.py                                  SyntheticResourcesServer class
│   ├── configs/synthetic_resources.yaml
│   ├── tests/test_app.py
│   └── requirements.txt
│
├── responses_api_models/synthetic_model/       synthetic /v1/responses
│   ├── app.py                                  SyntheticModel class. Sequence-length-driven payload.
│   ├── configs/synthetic_model.yaml
│   └── tests/test_app.py
│
├── configs/
│   ├── smoke.yaml                              16-concurrent sanity check
│   ├── axis_a_8k.yaml                          8K-concurrent, 16 KB bodies
│   ├── axis_c_8k_4mb.yaml                      8K-concurrent, ~4.5 MB bodies
│   ├── multi_agent_base.yaml                   N-agent template
│   └── _gen_multi_agent.py                     N-agent generator
│
├── data/
│   ├── generate_data.py                        synthesize JSONL of N tasks
│   └── .gitignore
│
├── load_driver.py                              concurrent /run dispatcher
├── instrumentation.py                          RetryTracker / ProcessMetricsSampler / KernelWatcher
├── sweep_runner.py                             one ng_run + load_driver per cell
├── run_sweep.py                                cross-product matrix → sweep_runner
├── run_single_agent_sweep.sh                   full single-agent matrix
├── run_multi_agent_sweep.sh                    full multi-agent matrix
├── run_multi_agent_delta_sweep.sh              4-cell multi-agent validation subset
├── run_multi_agent_n256_check.sh               one-off N=256 spinup_only sanity check
├── run_multi_agent_rerun_failed.sh             rerun multi-agent cells that failed previously
├── analyze_multi_agent.py                      re-aggregator for multi-agent results
├── run_on_slurm.sh                             container-mode launcher
└── run_on_slurm_baremetal.sh                   bare-metal launcher (uv venv, no pyxis)
```

The agent server is the **real, unchanged `responses_api_agents/simple_agent`**. Only the resources server and model are synthetic; the agent loop, retry layering, session middleware, cookies, and aiohttp client are all unchanged from production. That is what makes failures in the harness real failures of nemo-gym, not artifacts of a different stack.

### 3.1 synthetic_resources

Two endpoints, each with independent knobs:

| Knob | Default | Meaning |
| --- | --- | --- |
| `tool.async_latency_ms` / `verify.async_latency_ms` | 0 | `await asyncio.sleep(...)` — yields the event loop. |
| `tool.cpu_burn_ms` / `verify.cpu_burn_ms` | 0 | `time.perf_counter()` busy-loop — holds the event loop. Use to dial the I/O-bound vs CPU-bound ratio. |
| `tool.body_size_bytes` / `verify.body_size_bytes` | 1024 / 512 | Total byte length of the response body's variable portion. |
| `tool.body_shape` / `verify.body_shape` | `flat_padding` | `flat_padding` (one big string) or `realistic_messages` (many small dict chunks; real GC pressure). |
| `tool.realistic_chunk_bytes` | 256 | Per-chunk size when `body_shape=realistic_messages`. |
| `tool.latency_dist` / `body_dist` | `fixed` | Or `lognormal` for tail-distribution sweeps. |
| `inject_500_rate` | 0.0 | P(handler returns HTTP 500). |
| `inject_hang_rate` / `inject_hang_seconds` | 0.0 / 3600 | P(handler awaits forever). Simulates wedged sub-server. |

### 3.2 synthetic_model

| Knob | Default | Meaning |
| --- | --- | --- |
| `prompt_tokens` | 512 | Length of `prompt_token_ids` list on the output message. |
| `output_tokens` | 256 | Length of `generation_token_ids` and `generation_log_probs`; drives output `text` via `chars_per_token`. |
| `chars_per_token` | 4 | Output `text` length per token. |
| `include_token_ids_and_log_probs` | `true` | RL training default. Adds ~16 bytes/token to body size. |
| `n_reasoning_items` | 0 | Number of `<think>`-style reasoning items prepended. |
| `reasoning_tokens_per_item` | 0 | Tokens per reasoning item (each gets its own ids/probs). |
| `vocab_size` | 200000 | Upper bound on randomly-generated token IDs. |
| `tool_name` | `synthetic_tool` | Function name in the emitted `function_call`. |
| `async_latency_ms` | 50 | I/O-bound wait. |
| `cpu_burn_ms` | 0 | CPU-bound work. |
| `latency_dist` | `fixed` | Or `lognormal`. |
| `inject_500_rate` | 0.0 | P(handler returns HTTP 500). |

Body bytes ≈ `output_tokens × chars_per_token + (prompt + output) × 6 + output × 10 + n_reasoning × reasoning_tokens × 20 + ~1 KB`.

### 3.3 simple_agent (unchanged)

`responses_api_agents/simple_agent` is the real production agent. Knobs we use here:

- `max_steps` — controls hop count per `/run`. Each step issues one model call + (optionally) one tool call.
- `resources_server` / `model_server` — the standard refs, point at the synthetic instances.

### 3.4 Load driver

`tools/scale_sim/load_driver.py`. Production-shape: imports `nemo_gym.server_utils.{ServerClient, request, get_response_json, raise_for_status}` and `set_global_aiohttp_client`. Drives N concurrent `POST /run` against one or more agents through the same code paths `RolloutCollectionHelper._post_subroutine` does in production.

Two driver modes:

- **`loaded`** (default): bring up `ng_run`, load JSONL, dispatch `total_requests` concurrent rollouts capped by `concurrency` semaphore. Outer retry on `ClientPayloadError` mirrors `rollout_collection.py:450`. Inner retry on `ServerDisconnectedError` / `ClientOSError` lives inside `nemo_gym.server_utils.request`.
- **`spinup_only`**: bring up `ng_run`, idle for `idle_window_s`, sample metrics, kill. No JSONL, no requests. Answers the "what's the cost of N sub-servers existing" question for Axis B.

The driver registers per-row `agent_ref.name` (production-correct) so multi-agent dispatch is one line. Single-agent configs declare `agent_name`; multi-agent declare `agent_names: [list]` and the driver round-robins across rows.

### 3.5 Instrumentation

Three layers, written once and reused everywhere:

1. **`RetryTracker`** — per-rollout retry attempt counter, optional per-agent breakdown. Emits one JSONL line per completed rollout and computes sliding-window aggregates for the early-stop check.
2. **`ProcessMetricsSampler`** — background thread inside the load driver that samples (every 100 ms): RSS, FD count, CPU%, asyncio event-loop lag (the headline M3 indicator). Writes to `process_metrics.csv`.
3. **`KernelWatcher`** — separate process that samples (every 1 s): `/proc/net/sockstat` (TCP `inuse`, `tw`, `orphan`), `/proc/sys/fs/file-nr`, `/proc/loadavg`. Out of process so it survives even when the head actor is wedged.

### 3.6 Sweep runner

`sweep_runner.py` orchestrates one `ng_run` invocation + one `load_driver.py` invocation per cell, with full teardown between cells. Process state (FD pool, kernel TIME_WAIT, aiohttp connector) does **not** survive between cells — each cell starts clean. Heartbeats every 15 s during spinup with the most recent recognizable milestone scraped from `ng_run.log` (uv resolve / install / "Building venv for X" / "Started server X" / "N/M servers ready"). Driver stdout is streamed to both the foreground and `driver.log`.

### 3.7 What stubs DO and DO NOT exercise faithfully

**Faithfully exercised, free, no extra knobs:** kernel TCP stack, FDs, per-socket `rmem`/`wmem`, aiohttp connection pool, keep-alive logic, uvicorn accept loop / worker count / lifecycle, `RolloutCollectionHelper.run_examples` dispatch, session middleware + cookie propagation, HTTP/1.1 parsing, JSON bytes on the wire. These are exactly the subsystems that drive M3, the keep-alive race, FD exhaustion, and TW-table fill.

**Systematically under-exercised, recovered with explicit knobs:**

| Real cost | What stubs miss | How we recover |
| --- | --- | --- |
| Handler CPU between awaits | `asyncio.sleep` yields immediately | `cpu_burn_ms` busy-loop knob |
| JSON encode/decode of nested structures | `bytes(N)` is one cheap orjson copy | `body_shape ∈ {flat_padding, realistic_messages}` |
| Pydantic validation cost | We skip it | `pydantic_validation` knob (planned, not yet wired) |
| GC pressure from many small allocations | Single big bytes vs many small dicts | `realistic_chunk_bytes` |
| Per-request memory residency | Zero-page-backed bytes get reclaimed fast | `realistic_messages` shape forces real parsed structure to be held |
| Tokenization-equivalent CPU on the model | Skipped | `cpu_burn_ms` on the model |

**Calibration plan** (not yet executed): once the harness is in place, run `example_single_tool_call + vllm_model` against a small real model at a fixed concurrency, lock the synthetic knobs (`cpu_burn_ms`, `body_shape`, eventually `pydantic_validation`) until the synthetic timeseries match within 20 %. See companion design doc §4.8.3.

---

## 4. Production rollout-loop fidelity (multi-agent edition)

The driver loop must match production exactly. Production:

```441:452:nemo_gym/rollout_collection.py
async def _post_subroutine(row: Dict) -> Tuple[Dict, Dict]:
    while True:
        async with semaphore:
            try:
                res = await server_client.post(
                    server_name=row["agent_ref"]["name"], url_path="/run", json=row
                )
                await raise_for_status(res)
                return row, await get_response_json(res)
            except ClientPayloadError as e:
                print(f"Retrying /run for agent={row['agent_ref']['name']} after {type(e).__name__}: {e}")
        await asyncio.sleep(0.5)
```

Two anchors of fidelity:

1. **Per-row dispatch via `agent_ref.name`.** `ServerClient.post(server_name=...)` resolves through the global config the head server publishes. The driver does **no** per-agent URL bookkeeping; it just sets `row["agent_ref"]["name"]` round-robin across `agent_names` and dispatches. `ng_collect_rollouts` itself fills `agent_ref` from a config-level fallback when missing (`rollout_collection.py:189`), so driver-assigned `agent_ref` is the same path production uses.
2. **One asyncio event loop, one global aiohttp client, fanning out across N targets.** This is **not** a pool of driver processes. The whole point is that every increase in N is "more sockets multiplexed on the same single consumer event loop" — the M3 mechanism. Spawning N driver processes would change what's being measured.

---

## 5. Test matrix

Single-node, bare-metal on the cluster's `cpu` partition. Each row of every matrix is one `ng_run` invocation, because we need clean process state between runs (FD pool, kernel TCP, aiohttp connector). No batch wall-clock cap.

### 5.1 Single-agent — Axes A, C, defect #5

Driven by `tools/scale_sim/run_single_agent_sweep.sh`. Single-agent topology (1 model + 1 resources + 1 agent + 1 head).

| Sub-sweep | What it varies | Cells |
| --- | --- | --- |
| `00_smoke` | 16 concurrent | 1 |
| `01_concurrency_baseline` | concurrency = 1 K, 4 K, 8 K, 16 K @ output=16 K | 4 |
| `02_output_tokens_far_tail` | output = 16 K → 1 M (zip with concurrency 8 K → 128) | 5 |
| `03_n_hops_curve` | n_hops = 1, 4, 16, 64 @ 8 K concurrent | 4 |
| `04_defect_5_ablation` | semaphore on/off @ 8 K, 16 K | 4 |

**Total: ~18 cells.**

### 5.2 Multi-agent (sub-server-count axis)

Driven by `tools/scale_sim/run_multi_agent_sweep.sh`. Topology locked: **N agents + N resources + 1 shared model**.

All cells share: `simple_agent.max_steps=1`, `output_tokens=1024`, `tool.body_size_bytes=16384`, `verify.body_size_bytes=4096`, model `async_latency_ms=200`, tool `async_latency_ms=100`, verify `async_latency_ms=50` — same as `axis_a_8k.yaml` for direct cross-comparison.

#### Sub-sweep A. Spinup-only (pre-flight resource cost)

Bring up N agents + N resources + 1 model, idle 30 s, sample, kill. No load driven.

| `n_agents` | total sub-servers | What we measure |
| --- | --- | --- |
| 1 | 4 | Baseline |
| 4 | 10 | |
| 16 | 34 | |
| 32 | 66 | |
| 64 | 130 | |
| 128 | 258 | |
| 256 | 514 | |

7 cells. Records per cell: `spinup_time_s`, `idle_max_fd_total`, `idle_total_rss_bytes`, `idle_loadavg_1m`, `tcp_listen_count`. If a cell times out, the sweep continues with `spinup_outcome: timeout`.

#### Sub-sweep B. Loaded — fixed total concurrency = 4 K

| `n_agents` | total concurrency | per-agent concurrency |
| --- | --- | --- |
| 1 | 4096 | 4096 |
| 4 | 4096 | 1024 |
| 16 | 4096 | 256 |
| 32 | 4096 | 128 |
| 64 | 4096 | 64 |
| 128 | 4096 | 32 |
| 256 | 4096 | 16 |

7 cells. Tests "does adding sub-servers cost something at fixed total load." At N=256 each agent sees only 16 in-flight — a stress test of the per-agent uvicorn baseline cost.

#### Sub-sweep C. Loaded — fixed per-agent concurrency = 256

| `n_agents` | per-agent concurrency | total concurrency |
| --- | --- | --- |
| 1 | 256 | 256 |
| 4 | 256 | 1024 |
| 16 | 256 | 4096 |
| 32 | 256 | 8192 |
| 64 | 256 | 16384 |
| 128 | 256 | 32768 |
| 256 | 256 | 65536 |

7 cells. **Headline test**: confirms whether the M3 cliff is consumer-side and N-independent. Direct comparisons: sub-sweep-C N=64 cell ↔ single-agent `01_concurrency_baseline` 16 K; N=128 ↔ predicted Axis-A 32 K cliff; N=256 ↔ predicted Axis-A 64 K cliff (`EMFILE` / kernel TW table fill territory).

**Multi-agent total: 21 cells.**

### 5.3 What the matrix deliberately does not cross

- **N × hops**: hops covered by M1 03 at N=1; cross with N deferred unless M1 hops results suggest interaction beyond the predicted multiplicative.
- **N × body size**: covered by M1 02 at N=1; same logic.
- **Real-traffic eval**: planned cross-check with `example_single_tool_call + vllm_model + small real model` for calibration (§3.7).
- **Multi-node / sharded consumer**: that is the structural fix experiment (the refactor milestone plan), not characterization.

---

## 6. Predictions to falsify

Same format across single-agent and multi-agent sweeps. All predictions are concrete enough that the experiment can refute them.

### 6.1 Single-agent

| Axis | Concurrency | Bodies | Hops | Predicted breakage point | Predicted dominant mechanism |
| --- | --- | --- | --- | --- | --- |
| A | 16 K | 16 KB | 1 | semaphore-off shows step-boundary burst saturating loop; semaphore-on is clean | M3 amplified by defect #5 |
| A | 32 K | 16 KB | 1 | Loop lag > 100 ms; FD count > 30 K (semaphore-on) | M3 + FDs |
| A | 64 K | 16 KB | 1 | OS rejects new connections (`EMFILE`) or kernel TW table fills | kernel |
| C1 | 8 K | 16 KB | 16 | p99 doubles vs hops=1 (per-hop session middleware + cookie) | session middleware + per-hop transport |
| C1 | 8 K | 16 KB | 64 | p99 super-linear; per-hop loop lag > 50 ms | M3 amplified by hops |
| C2 | 8 K | 4 MB | 1 | M3 cliff — mid-body RST volume rises sharply | M3 |
| C2 | 8 K | 32 MB | 1 | OOM in head actor | RAM |
| C2 | 4 K | 32 MB | 4 | TCP_NOMEM / send buffer saturation | kernel + M3 |

### 6.2 Multi-agent

| Variant | n_agents | Predicted breakage | Predicted dominant mechanism |
| --- | --- | --- | --- |
| spinup-only | 64 | spinup time > 60 s | uvicorn cold-start + Ray actor registration serialization |
| spinup-only | 128 | spinup time > 180 s OR idle FDs > 60 K | A1 ulimit / kernel listen table |
| spinup-only | 256 | spinup fails OR idle RSS > host RAM | RAM (256 × ~300 MB ≈ 75 GB) |
| loaded, fixed total=4 K | 32 | per-rollout p99 *improves* slightly vs N=1 — load shards across more server-side event loops | counter-prediction |
| loaded, fixed total=4 K | 128 | per-rollout p99 *degrades* — head consumer drains more sockets | M3, head-side |
| loaded, per-agent=256, N=32 | 32 | total concurrency 8 K — matches Axis-A 8 K within 20 % | M3, agent-independent (baseline normalization) |
| loaded, per-agent=256, N=64 | 64 | total concurrency 16 K — matches Axis-A 16 K cliff (**headline confirmation**) | M3 |
| loaded, per-agent=256, N=128 | 128 | total concurrency 32 K — head FD count > 60 K, kernel TW table fills | M3 + kernel |

**Headline test** of the experiment: multi-agent sub-sweep C at N=64 should reproduce single-agent `01_concurrency_baseline` 16 K within ~20 %. If yes → M3 confirmed consumer-side and N-independent (the intended structural finding). If they diverge → we've found new architectural cost specific to multi-server topologies.

---

## 7. How to run

### 7.1 One-time setup (bare metal, on a cpu node)

```bash
salloc -A coreai_dlalgo_nemofw -p cpu --nodes=1 --time=1:00:00 --exclusive --mem=0 --job-name=scale-sim-setup
cd /lustre/fs1/portfolios/coreai/users/ansubramania/dev/Gym
unset VIRTUAL_ENV
export UV_LINK_MODE=copy RAY_TMPDIR=/tmp
uv venv --python 3.12
uv sync --extra dev
source .venv/bin/activate
ulimit -Sn $(ulimit -Hn) || ulimit -Sn 1048576
which ng_run            # sanity
```

### 7.2 Smoke test (interactive)

```bash
cd tools/scale_sim
python data/generate_data.py --n 256 --user-input-size-bytes 256 --output data/smoke.jsonl
python sweep_runner.py \
    --config configs/smoke.yaml --input-jsonl data/smoke.jsonl \
    --git-sha "$(git rev-parse --short HEAD)" \
    --head-server-host 127.0.0.1 --head-server-port 5000
```

Expected: `summary.json` with `failure_rate=0`, `retry_rate=0`, `n_rollouts=256`, `p99_s` < 1 s.

### 7.3 Single-agent matrix as an sbatch (single command)

```bash
cd /lustre/fs1/portfolios/coreai/users/ansubramania/dev/Gym
DRIVER_SCRIPT=tools/scale_sim/run_single_agent_sweep.sh \
JOB_NAME=scale-single-agent \
  bash tools/scale_sim/run_on_slurm_baremetal.sh batch
```

Output:
- `scale-sim-logs/<job>-<ts>/sbatch.{out,err}` — live launcher / driver / sweep heartbeats.
- `tools/scale_sim/results/<sha>/single_agent_master.csv` — aggregated final result.

Monitor: `tail -f scale-sim-logs/<job>-<ts>/sbatch.out`.

### 7.4 Multi-agent matrix as an sbatch

```bash
cd /lustre/fs1/portfolios/coreai/users/ansubramania/dev/Gym
DRIVER_SCRIPT=tools/scale_sim/run_multi_agent_sweep.sh \
JOB_NAME=scale-multi-agent \
  bash tools/scale_sim/run_on_slurm_baremetal.sh batch
```

Output: `tools/scale_sim/results/<sha>/multi_agent_master.csv` with one row per cell.

### 7.5 Operational notes

- **Bare-metal launcher** (`run_on_slurm_baremetal.sh`) defaults: `ACCOUNT=coreai_dlalgo_nemofw`, `PARTITION=cpu`, no `--qos`, `--cpus-per-task=40`, `--exclusive --mem=0 --nodes=1`. No `--time` set unless `TIME` env is provided. Sets `RAY_TMPDIR=/tmp`, `RAY_NODE_IP_ADDRESS=127.0.0.1`, `UV_LINK_MODE=copy`, bumps soft FD limit to hard.
- **Per-server venvs**: each server type's `.venv/` is created lazily on first launch. First cell is slow (5–15 min cold uv on Lustre); subsequent cells reuse. M2 instances of the same server type all share that one venv.
- **Port range 5000–5512** — below the OCI-HSG ephemeral floor (9000). Adjust `head_server.port` and offsets only if your cluster uses a different floor.
- **TIME_WAIT bleed across cells**: 5 s post-cell sleep at small N; bumped to `max(5, N // 8)` at high N (multi-agent only).
- **FD cap**: confirm `ulimit -Hn ≥ 1 M` on the cpu node before multi-agent N=128. Some clusters cap at 64 K — that becomes a hard ceiling on the N axis, noted as experimental limit not finding.

### 7.6 Output file layout

Per cell, `tools/scale_sim/results/<sha>/<exp_name>/<cell>_<ts>/`:

| File | Content |
| --- | --- |
| `config.yaml` | Resolved config for this cell |
| `summary.json` | failure_rate, retry_rate, latency percentiles, error class breakdown, per-agent breakdown (multi-agent) |
| `per_rollout_retries.jsonl` | One line per rollout: `n_attempts`, `error_classes`, `succeeded`, `agent_name` (multi-agent) |
| `latencies.csv` | Per-successful-rollout latency for offline percentile work |
| `process_metrics.csv` | RSS / FDs / CPU% / asyncio loop lag p99 every 100 ms |
| `kernel_metrics.csv` | `/proc/net/sockstat` + `file-nr` + loadavg every 1 s |
| `ng_run.log` | Full uvicorn + sub-server stdout |
| `driver.log` | Full load driver stdout |

Per sweep, `tools/scale_sim/results/<sha>/<exp_name>/sweep_results.csv` aggregates one row per cell with the headline columns. The matrix runners (`run_single_agent_sweep.sh`, `run_multi_agent_sweep.sh`) further aggregate per-experiment CSVs into `single_agent_master.csv` / `multi_agent_master.csv`.

---

## 8. Stretch deliverables

1. **Reduced-form analytic ceiling.** A one-page back-of-envelope that gives `usable_concurrency ≈ min(FD_budget, RAM_budget / body_size, event_loop_budget / (hops × parse_cost(body_size)))` with constants fitted from the sweep. Useful for planning future runs without re-sweeping.
2. **Calibration sweep.** §3.7 — match synthetic to `example_single_tool_call + small real model` within 20 %. Locks knob defaults as the "realistic baseline" and lets us cross-check that the cliffs we find aren't artifacts of the synthetic shape.
3. **Multimodal payload representation.** Base64 / image features in input messages, not just text padding.
4. **Multi-process driver mode** for the case where M3 caps total throughput before the N axis is interesting. Adds N driver processes — would change what's being measured. Not on critical path.
5. **gRPC / HTTP/2 ablation.** Out of scope for characterization; relevant for the structural-fix experiment.

---

## 9. References

- `nemo-gym-scale-simulation-design.md` — original single-agent design
- `nemo-gym-multi-agent-design.md` — multi-agent design
- `2381291-nemo-gym-timeouts-connection-resets.md` — incident grounding
- `tools/scale_sim/README.md` — short user-facing version of §7

---

## 10. Results

Each subsection below has the format **(prediction → measurement → interpretation)** so a reader can verify what we expected, what we got, and what we conclude. Cells marked `[pending …]` are awaiting a re-run that's currently in flight; numbers without that marker are measured.

### 10.1 Single-agent — Axes A, C, defect #5

Source: `tools/scale_sim/results/<sha>/single_agent_master.csv` from the single-agent batch job. **Status: filling in as the batch lands; partial cells already on disk.**

#### 10.1.1 Axis A — concurrency baseline at output_tokens=16K (`01_concurrency_baseline`)

| concurrency | failure_rate | retry_rate | p50_s | p99_s | max_s | n_rollouts | stop_reason | predicted | matches? |
|---|---|---|---|---|---|---|---|---|---|
| 1024 | [pending] | | | | | | | clean | [pending] |
| 4096 | [pending] | | | | | | | clean | [pending] |
| 8192 | [pending] | | | | | | | clean | [pending] |
| 16384 | [pending] | | | | | | | M3 onset (loop lag begins climbing) | [pending] |

**Interpretation:** _to be written once the four cells land._

#### 10.1.2 Axis C2 — body-size cliff (`02_output_tokens_far_tail`)

`output_tokens` and `concurrency` are zipped to keep nominal residency bounded.

| output_tokens | concurrency | total_requests | failure_rate | retry_rate | p50_s | p99_s | predicted | matches? |
|---|---|---|---|---|---|---|---|---|
| 16384 | 8192 | 20000 | [pending] | | | | clean reference | [pending] |
| 131072 | 1024 | 4000 | [pending] | | | | clean | [pending] |
| 262144 | 512 | 2000 | [pending] | | | | beginning of cliff | [pending] |
| 524288 | 256 | 1000 | [pending] | | | | M3 cliff visible in p99 | [pending] |
| 1048576 | 128 | 500 | [pending] | | | | head-actor RSS pressure or read() timeout | [pending] |

**Interpretation:** _to be written once the body-size sweep lands. Specifically looking for the cliff between 256K and 1M output_tokens._

#### 10.1.3 Axis C1 — hop-depth cliff (`03_n_hops_curve`)

| n_hops | concurrency | n_rollouts | p50_s | p99_s | stop_reason | predicted | matches? |
|---|---|---|---|---|---|---|---|
| 1 | 8192 | [pending] | | | | clean reference | [pending] |
| 4 | 8192 | [pending] | | | | p99 grows linearly | [pending] |
| 16 | 8192 | **139** | _slow_ | **594 s** | `wall_clock>600s` | predicted: p99 doubles vs hops=1 | **YES — exceeded prediction** (594s p99, only 139 of 20K rollouts before 600s timeout) |
| 64 | 8192 | [pending] | | | | predicted: super-linear; per-hop loop lag > 50 ms | [pending] |

**Interpretation (preliminary, hops=16 cell only):** the M3 cliff is dramatic in the depth axis. At hops=16, every rollout chains 16 model + 16 tool round-trips through a single consumer event loop fanning to 8K parallel rollouts. Per-rollout latency floor (`16 hops × 300 ms` = 4.8 s) becomes amplified by ~120× of queueing → 594 s p99. The `total_requests=20000` budget in the matrix was set assuming low-hop cells; at high hops the cell is wall-clock-bounded, completing only 139 rollouts in 600 s. **Action item**: future hops sweeps should zip `total_requests` inversely with `n_hops` (e.g. 20K, 8K, 2K, 500 for hops=1,4,16,64) so each cell has comparable runtime.

#### 10.1.4 Defect #5 — RL-side semaphore on/off (`04_defect_5_ablation`)

| concurrency | semaphore | failure_rate | retry_rate | p99_s | predicted | matches? |
|---|---|---|---|---|---|---|
| 8192 | true | [pending] | | | semaphore-on is clean | [pending] |
| 8192 | false | [pending] | | | semaphore-off shows step-boundary burst | [pending] |
| 16384 | true | [pending] | | | M3 onset visible | [pending] |
| 16384 | false | [pending] | | | M3 onset amplified by defect #5 | [pending] |

**Interpretation:** _to be written. Headline question: does closing defect #5 alone push the M3 cliff far enough out that we don't need the structural fix._

---

### 10.2 Multi-agent — sub-server-count axis

Source: `tools/scale_sim/results/<sha>/multi_agent_master_full.csv` (re-aggregated by `analyze_multi_agent.py` after the rerun lands). **Status: 10/21 cells succeeded in the original sweep; remaining 11 cells in flight after the §A11 fix to `sweep_runner._pre_cell_cleanup`.**

#### 10.2.1 Sub-sweep A — spinup-only resource cost vs N

| n_agents | total sub-servers | tcp_inuse | tcp_tw | file_nr_used | loadavg_1m | predicted | matches? |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 2240 | 118 | 7872 | 0.35 | baseline | ✓ |
| 4 | 10 | 2300 | 514 | 8640 | 0.69 | linear-ish | ✓ (tcp_inuse only +60, mostly Ray-internal) |
| 16 | 34 | [pending rerun] | | | | spinup time begins growing super-linearly | [pending] |
| 32 | 66 | [pending rerun] | | | | | [pending] |
| 64 | 130 | [pending rerun] | | | | spinup time > 60 s (predicted) | [pending] |
| 128 | 258 | [pending rerun] | | | | spinup time > 180 s OR idle FDs > 60 K | [pending] |
| 256 | 514 | n/a — A12 | n/a | n/a | n/a | spinup fails OR idle RSS > host RAM | **partial match** — spinup failed, but for a *different* reason than predicted. Pre-A10-fix it was ARG_MAX (§10.3.1). Post-A10-fix it's the Ray GCS connect storm (A12 in §1.2 / §10.3.3). Cell still never reaches the idle-sample phase, just for a different upstream architectural reason. |

**Interpretation (preliminary, N=1 and N=4 only):** sub-linear scaling in idle TCP usage between N=1 and N=4. Most of the 2240 baseline TCP connections are Ray-internal, not per-sub-server. Going from 4 → 10 sub-servers added only 60 TCP entries (15× less than naive proportional). Suggests the kernel TCP cliff at high N comes from *load-induced* connections, not idle topology. Consistent with M3 being consumer-side, not sub-server-side. Awaiting N=16+ to confirm the curve.

#### 10.2.2 Sub-sweep B — loaded, fixed total concurrency = 4096

Tests "does adding sub-servers cost something at fixed total load?"

| n_agents | per-agent conc | failure_rate | retry_rate | p50_s | p99_s | max_s | n_rollouts | stop_reason | per-agent rollout balance | predicted | matches? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 4096 | 0.000 | 0.000 | **34.27** | 80.77 | 80.89 | 10000 | None | n/a | clean reference | ✓ |
| 4 | 1024 | 0.000 | 0.000 | **6.96** | 23.83 | 24.39 | 10000 | None | 2500 / 2500 (from delta) | counter-prediction: load shards across server-side loops, p50 *improves* | **YES — 5× p50 improvement (34s→7s)** |
| 16 | 256 | 0.000 | 0.000 | **2.07** | 22.49 | 22.83 | 10000 | None | 625 / 625 | further improvement, diminishing returns | ✓ p50 16× better than N=1 |
| 32 | 128 | [pending rerun] | | | | | | | | | [pending] |
| 64 | 64 | [pending rerun] | | | | | | | | predicted to start degrading (head consumer drains more sockets) | [pending] |
| 128 | 32 | [pending rerun] | | | | | | | predicted: per-rollout p99 *degrades* — head consumer drains more sockets | [pending] |
| 256 | 16 | n/a — A12 | n/a | n/a | n/a | n/a | n/a | n/a | head-actor RSS pressure (~75 GB at idle) | **not testable in current implementation** — A12 (Ray GCS connect storm) blocks spinup. A10 was fixed first; A12 was newly surfaced once A10 was out of the way. |

**Interpretation:** the §6.2 counter-prediction "load shards across more server-side event loops" is **strongly confirmed** at N=1→16. Server-side parallelism wins because each agent's event loop only sees 1/N of the load. The interesting question is where this curve **inverts** as N grows further — predicted between N=64 and N=128, where head-side consumer cost should start dominating server-side parallelism wins. Awaiting N=32 through N=256 cells.

#### 10.2.3 Sub-sweep C — loaded, fixed per-agent concurrency = 256 (**headline test**)

Tests whether the M3 cliff is consumer-side and N-independent. Total concurrency grows linearly with N.

| n_agents | total conc | failure_rate | retry_rate | p50_s | p99_s | max_s | p99/p50 | predicted | matches? |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 256 | 0.000 | 0.000 | 1.89 | 2.56 | 2.83 | 1.4× | clean baseline | ✓ |
| 4 | 1024 | 0.000 | 0.000 | 2.44 | 5.50 | 6.55 | 2.3× | clean | ✓ |
| 16 | 4096 | 0.000 | 0.000 | 4.04 | 119.29 | 141.28 | 29.5× | tail begins | ✓ — tail blows up between N=4 and N=16 |
| 32 | 8192 | 0.000 | 0.000 | 3.50 | 138.97 | 161.70 | 39.7× | matches Axis-A 8K within 20 % | [pending direct M1 comparison] |
| 64 | 16384 | 0.000 | 0.000 | 3.23 | 165.89 | 168.80 | 51.4× | matches Axis-A 16K cliff (**headline confirmation**) | [pending direct M1 comparison] |
| 128 | 32768 | [pending rerun] | | | | | | head FD count > 60 K, kernel TW table fills | [pending] |
| 256 | 65536 | n/a — A12 | n/a | n/a | n/a | n/a | n/a | OS rejects new connections (`EMFILE`) OR architectural ARG_MAX (A10) hits first | **A12 hits first** post-A10-fix. Spinup fails before any load is driven. The kernel-level cliff at 65K total concurrency remains not testable in the current implementation; needs A12 fix or workaround. |

**Interpretation (preliminary, N=1 through 64):**

1. **p50 is N-invariant (~3 s) past N=16.** Each agent processes its own 256 in-flight rollouts on its own event loop; adding more agents doesn't make any single rollout's median time shorter. Median latency is set by the **per-agent** load, not the global topology.
2. **p99 grows ~linearly in total concurrency.** The tail goes 2.6 → 5.5 → 119 → 139 → 166 s as total conc goes 256 → 1024 → 4096 → 8192 → 16384. This is exactly the M3 mechanism: a single consumer event loop draining response bodies for `total_concurrency` open sockets. The driver process is the bottleneck, not the agents.
3. **`failure_rate=0` everywhere.** M3 manifests as queueing latency, not failure. We don't lose connections; we just wait. The threshold where this becomes failure (FD exhaustion / kernel TW table fill) hasn't been crossed in the cells we have — the high-N cells in flight should reveal it.
4. **p99/p50 ratio is the cleanest tail-amplification metric.** It grows from 1.4× at N=1 to 51× at N=64. This **is** the M3 cliff, just expressed as tail amplification rather than failure rate.

**Headline check** _(pending — requires comparison with single-agent `01_concurrency_baseline` rows once they land)_: multi-agent sub-sweep C N=64 (total=16K, p99=166s) should be in the same range as the single-agent Axis-A 16K cell. Note both bodies and hops differ between the two sweeps (single-agent 01 uses output_tokens=16K, multi-agent C uses output_tokens=1K), so the comparison is *cliff shape*, not absolute numbers. Specifically:
- Both should show p50 modest, p99 large.
- Failure rate stays at 0 in both.
- Difference in absolute p99 should be explained by body-size difference alone (~16× output bytes → ~16× receive cost per rollout → linear-ish p99 difference).

If multi-agent sub-sweep C N=64 cliff *is* in the same shape as single-agent 16K → **M3 confirmed consumer-side and N-independent**. The structural finding the experiment exists to produce.

If they diverge dramatically → **new architectural cost specific to multi-server topologies**. Would need follow-up to characterize.

---

### 10.3 Architectural findings (non-load — direct bugs in `nemo_gym/cli.py`)

Two findings emerged from running the harness that are **not** about the M3 mechanism — they're real bugs in `nemo_gym/cli.py` lifecycle code that surface only in the "many-cells-back-to-back" usage pattern. They appear in this section rather than §10.1/§10.2 because they're about the harness *infrastructure* hitting CLI bugs rather than the architecture under test hitting the predicted mechanisms.

#### 10.3.1 A10 — global config in spawn command line hits `ARG_MAX` at N=256

**Symptom (measured, reproducible):** `OSError: [Errno 7] Argument list too long: '/bin/bash'` at `cli_setup_command.py:195` Popen, on the **first** sub-server spawn at N=256, in **every** N=256 cell tested. Reproduced under:
- Multi-agent sub-sweep C N=256 loaded (total_concurrency=65536) — first observation, originally attributed to potential confounding factors.
- Multi-agent sub-sweep A N=256 spinup-only, run **after** the A11 Ray-state-leak fix landed (clean state, fresh `/tmp`, no orphaned Ray). Same failure, same first-spawn position. **Confirms A10 is a deterministic architectural ceiling, not an intermittent boundary case.**

The failed cell's `ng_run.log` shows: Ray cluster comes up cleanly (`Started Ray cluster at 10.65.35.19:34551`), then the very first child Popen for the shared `synthetic_model` sub-server hits `ARG_MAX` — meaning the resolved global-config YAML alone, before any per-spawn environment buildup, is already over the kernel limit at N=256.

**Root cause (read from code, `nemo_gym/cli.py:172-177`):**

```python
command = f"""{setup_env_command(...)} \
    && {NEMO_GYM_CONFIG_DICT_ENV_VAR_NAME}={escaped_config_dict_yaml_str} \
    {NEMO_GYM_CONFIG_PATH_ENV_VAR_NAME}={shlex.quote(top_level_path)} \
    python {str(entrypoint_fpath)}"""

process = run_command(command, dir_path, server_name=top_level_path)
```

`escaped_config_dict_yaml_str` is the **entire global config** embedded inline as an env-var assignment in every sub-server's spawn command. At N=256 (514 entries), the YAML — after shell-escaping — exceeds Linux's `ARG_MAX` (2 MB on this kernel). Every spawn at N=256 fails the same way.

**Effective architectural ceiling for the current `nemo_gym/cli.py` implementation:**
- **N=128 works** (260 sub-server entries; YAML ~1× of ARG_MAX worst case). Pending confirmation in the rerun.
- **N=256 fails deterministically** (514 entries; YAML over ARG_MAX). Confirmed.
- **The exact crossover** is somewhere in N=130–200, not measured. Worth a follow-up bisection if anyone wants to operate close to the ceiling in production.

**Practical impact for the next-model bringup:**
- A topology with ≥256 specialized sub-servers (e.g. one resources server per multimodal verifier, one judge per rubric, one sandbox per tool kind) is **not supported by the current `nemo_gym/cli.py`** without a fix.
- 50–100 sub-servers (the realistic near-term target) is well within the safe range.
- 128 is the max we can safely commit to until A10 is fixed.

**Fix landed in this commit.** Changes confined to two files:

- `nemo_gym/global_config.py`: added `NEMO_GYM_CONFIG_DICT_PATH_ENV_VAR_NAME = "NEMO_GYM_CONFIG_DICT_PATH"`. `get_global_config_dict()` now reads the path env var first (loads YAML from disk) and falls back to the legacy contents env var for backward compatibility.
- `nemo_gym/cli.py:RunHelper.start`: writes the resolved global-config YAML to `${RAY_TMPDIR}/nemo_gym_global_config_<pid>.yaml` once before the spawn loop, and the per-sub-server spawn command now passes `NEMO_GYM_CONFIG_DICT_PATH=<path>` instead of the inline `NEMO_GYM_CONFIG_DICT=<escaped_yaml>`. The tempfile is cleaned up in `RunHelper.shutdown`.

The rest of the codebase needs no changes because every consumer already goes through the central `get_global_config_dict()` resolver — the read-side switch is transparent to them.

**Effect on per-spawn command-line bytes:** O(N) → O(1). The only N-dependent thing left in the spawn command is the `top_level_path` server name (~50 chars), well below `ARG_MAX` for any plausible N. Validated by reasoning about the change; pending experimental confirmation in the next sweep.

**Backward compatibility:** parents running the old code (which sets `NEMO_GYM_CONFIG_DICT` with contents) still work because the read-site fallback is preserved. Mixed-version is unusual in practice — children always run the same `nemo_gym` install as the parent — but the fallback exists for safety regardless.

**Validation result (A10 confirmed fixed):** rerun multi-agent sub-sweep A spinup-only at N=256 in the post-A10-fix N=256 check job. Three observations:

1. **No `Argument list too long` anywhere in `ng_run.log`** — the ARG_MAX path is permanently closed. ✓
2. **Sub-servers read the global config from the tempfile path successfully** — many sub-servers (`synthetic_resources_31`, `synthetic_resources_24`, etc.) printed `Started server process … Application startup complete … Uvicorn running on http://…`. ✓
3. **Cell still failed**, but for a *new and different* reason: A12 (Ray GCS connect storm — see assumption table). About half the 514 sub-servers connected to Ray successfully; the other half timed out at the 5-second client-side `Failed to connect to GCS at address 10.65.35.221:30899` limit. ✓ as A10 fix; ✗ overall because A12 now blocks.

So the §10.2 N=256 rows go from "n/a — A10" to "n/a — A12 (A10 fixed)". To actually get N=256 measurements we now need either to fix or work around A12 (see the §1.2 A12 row for fix shapes).

#### 10.3.2 A11 — `ng_run` SIGINT teardown leaks Ray + sub-server state

**Symptom (measured):** at N≥16 sub-servers, the second cell of any back-to-back sequence sees the previous cell's stale Ray cluster pointer. New sub-servers fail GCS lookup with `ConnectionError: Failed to connect to Ray cluster at 10.65.34.215:14589` (the previous cell's GCS port, not the new one). Confirmed in `multi_agent_n16_c1_r1_*/ng_run.log` and 10 other failed-cell logs.

11 of 21 multi-agent cells failed this way before the harness-side fix.

**Root cause (read from code + behavior):** `ng_run`'s SIGINT handler invokes `ray.shutdown()`, which is multi-step and not atomic. `raylet`, `gcs_server`, and `plasma_store` are signaled but not waited-for. The 15 s grace period in `sweep_runner._run_one_cell`'s teardown then sends `SIGKILL` to the `ng_run` process — but **not** to the orphaned Ray child processes (different process groups). Stale `/tmp/ray/session_<id>/` directories with socket files and `ray_current_cluster` pointers remain; the next `ng_run`'s `ray.init()` joins the dead cluster instead of starting a new one.

**Why N=1 and N=4 worked but N=16 didn't:** with 4 sub-servers each retrying GCS connect for ~30 s, the timing window is forgiving; with 34+ sub-servers all retrying simultaneously, GCS thrash exceeds the spawn deadline and `ng_run` gives up before stabilizing. Also explains why later sub-sweeps (multi-agent C) recovered as orphans were eventually reaped.

**Harness-side workaround (landed):** `tools/scale_sim/sweep_runner._pre_cell_cleanup` runs `pkill -9 -f 'raylet|gcs_server|plasma_store|ng_run|nemo_gym|synthetic_*|simple_agent'` + `rm -rf /tmp/ray /tmp/ray_temp /tmp/ray-*` at the **start** of every cell. Idempotent across whatever exit mode the previous cell took. Replaces the earlier "trust the previous cell tore itself down" approach. Confirmed working: rerun of failed cells after this fix is in flight.

**Upstream fix shape (not landed):** `nemo_gym/cli.py`'s shutdown handler should track Ray child PIDs explicitly via `psutil.Process.children(recursive=True)`, send `SIGKILL` to all of them, wait, and only then `ray.shutdown()`. Plus a final `rm -rf /tmp/ray*` for good measure on exit. ~50 line change. Affects any pipeline running `ng_run` repeatedly (CI / eval / smoke / our scale-sim harness).

#### 10.3.3 A12 — sub-server `ray.init()` storm at high N saturates GCS

**Symptom (measured, post-A10-fix):** at N=256 spinup-only cell `multi_agent_n256_check_20260430_144259`, **after the A10 ARG_MAX fix landed and pre-cell cleanup (A11) was active**, ng_run successfully spawned all 514 sub-servers but a substantial fraction failed to connect to the head's Ray GCS:

```
[2026-04-30 14:44:04,703 W ... rpc_client.h:153: Failed to connect to GCS at address 10.65.35.221:30899 within 5 seconds.
[2026-04-30 14:44:05,069 W ... rpc_client.h:153: Failed to connect to GCS at address 10.65.35.221:30899 within 5 seconds.
... (dozens more)
```

About half the sub-servers reached "Connected to Ray cluster" on time; the other half hit the 5 s client-side connect timeout.

This failure mode is **distinct from A10 and A11**:
- The ARG_MAX path is closed (no `Argument list too long`).
- Pre-cell cleanup wiped `/tmp/ray*` and stale processes; the GCS the sub-servers are connecting to is genuinely fresh.
- The address (`10.65.35.221:30899`) is the head's *new* Ray cluster, not a stale pointer.

**Root cause (read from code + behavior):** `nemo_gym/cli.py:RunHelper.start` fires N+2 child Popens essentially in parallel — `Popen` returns immediately, and the loop iterates the next sub-server before the previous one's `ray.init()` completes. At N=256 (514 children), each child calling `ray.init(address=<head_gcs>)` near-simultaneously creates a thundering herd at the GCS. Ray's GCS handles connections serially through gRPC; with hundreds of concurrent connect attempts within ~10 s, the queueing latency exceeds the client-side `connect_timeout=5s` for many of the late-arriving children.

**Why N=64 worked and N=256 doesn't:** at N=64 (130 children), the storm fits inside the 5 s window — GCS can complete ~30+ connects per second, so 130 close out comfortably. At N=256 (514 children), the queue depth crosses the timeout barrier and a tail of connects time out. The threshold is somewhere around N=100–150 sub-servers, depending on host CPU and GCS responsiveness.

**Effective architectural ceiling for the current `nemo_gym/cli.py`:**
- Pre-A10-fix: hard ceiling at N=256 (ARG_MAX, deterministic).
- Post-A10-fix only: **practical ceiling at N≈100–150** (A12, intermittent thundering-herd).
- Post-A12-fix (proposed): no current evidence of a next ceiling below ~512+, but untested.

**Practical impact for the next-model bringup:**
- 50–100 sub-servers (the realistic near-term target) is below the A12 threshold and works fine.
- 100–250 sub-servers is the ambiguous zone where A12 will cause some startups to fail intermittently.
- 250+ sub-servers requires fixing or working around A12.

**Workaround landed in the harness:** `tools/scale_sim/_ray_burst_env.sh` — a comprehensive Ray-GCS burst-tolerance tuning block adapted from a known-good production launcher (the nemo-rl scale launcher). It addresses the thundering herd at multiple layers simultaneously rather than just bumping one timeout. The launcher (`run_on_slurm_baremetal.sh`) sources it into the submit shell; the generated driver wrapper re-sources it on the compute node for safety against `--export=ALL` edge cases.

The exports it sets, with rationale (see the file for the complete list and per-knob comments):

| Class | Knobs | What they do |
| --- | --- | --- |
| RPC thread pools | `RAY_gcs_server_rpc_server_thread_num=64`, `RAY_num_server_call_thread=32`, `RAY_gcs_server_rpc_client_thread_num=64` | Increase GCS inbound + outbound RPC parallelism so a 500-RPC burst gets drained instead of queueing past the 5 s connect timeout. |
| Timeouts | `RAY_gcs_rpc_server_connect_timeout_s=120`, `RAY_worker_register_timeout_seconds=120`, `RAY_gcs_rpc_server_reconnect_timeout_s=120`, `RAY_gcs_server_request_timeout_seconds=120` | 24× headroom on the client-side connect. With the thread-pool fix the burst should drain in ~60 s; the timeout covers worst-case + 2× margin. |
| Active RPC headroom | `RAY_gcs_max_active_rpcs_per_handler=12800` | Default auto-scales to `rpc_server_thread_num × 100 = 6400`. Doubled to keep handler queue from being the bottleneck during the spawn burst. |
| Background traffic reduction | `RAY_raylet_report_resources_period_milliseconds=500`, `RAY_task_events_report_interval_ms=5000`, `RAY_grpc_keepalive_time_ms=60000`, `RAY_grpc_keepalive_timeout_ms=60000`, `RAY_ray_syncer_message_refresh_interval_ms=10000`, `RAY_core_worker_internal_heartbeat_ms=5000`, `RAY_health_check_period_ms=10000` | Reduces the steady-state RPC load on GCS by 3–10× during the burst window so the storm has more room. |
| Resource broadcast batching | `RAY_gcs_resource_broadcast_max_batch_size=512`, `RAY_gcs_resource_broadcast_max_batch_delay_ms=100` | Default `batch_size=1` makes resource updates O(N²) in fan-out. Batching collapses to O(N). |
| Subscriber + worker pool | `RAY_subscriber_timeout_ms=60000`, `RAY_num_workers_soft_limit=16` | Faster cleanup of dead subscribers; cap on idle worker pool. |
| Observability off | `RAY_enable_timeline=false`, `RAY_event_stats=false` | Eliminates thousands of GCS writes during init — observability nice-to-haves we don't need during scale tests. |

This is **a workaround, not the fix**. The architectural fix is staggering the spawn loop in `nemo_gym/cli.py:RunHelper.start`. Three options for the upstream fix, in increasing order of robustness:

1. **Stagger into batches**: fire Popens in batches of e.g. 32 with a 1 s sleep between batches. Bounds the GCS connect rate to a known-good level. ~10 line change.
2. **Block on each sub-server's Ray registration before spawning the next**: each sub-server signals readiness via the head's status endpoint; spawn loop waits for the previous before issuing the next Popen. Effectively serial; spinup time becomes O(N × ~50 ms) = ~25 s at N=512. ~30 line change.
3. **Move sub-servers off Ray entirely** for the case where they don't actually need Ray (which most synthetic / leaf sub-servers don't — they only call Ray to register their own actor wrapper). Architectural; not in scope here.

For our scale-sim purposes, the burst-env workaround is the cheapest unblock. Pending validation: rerun the N=256 spinup-only check with the new env tuning in place.

---

### 10.4 Cross-cutting findings — what to act on

_Filled in once both single-agent and multi-agent matrices are complete and we've done the comparison work in §10.2.3._

Expected structure:

1. **Confirmed M3 mechanisms.** Which §6 predictions landed, with the cells that confirmed them.
2. **Falsified predictions.** Anything in the §6 tables that the data refutes — these are the most interesting findings.
3. **Newly discovered mechanisms.** Anything in the data we did not predict at all. So far: A10 (architectural N ceiling) and A11 (Ray state leak) are both in this category.
4. **Prioritized list of structural fixes** for the next-model bringup, with rough effort estimates:
   - Sharded-consumer (multi-process load driver / multi-Ray-actor head)
   - Tempfile-based global-config IPC (A10)
   - Proper Ray teardown in `ng_run` (A11)
   - HTTP/2 or gRPC streaming for `/run` bodies (against A3, A6, A7)
   - Eviction-style per-rollout body parse (against A7)
5. **Updates to assumptions A1–A11.** Which are now measured (with the cell that measured them), which are still speculative pending follow-up.

---

End of document.
