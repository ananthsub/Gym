# nemo-gym multi-agent (sub-server count) harness design

**Status:** draft (no code yet — implementation gated on this doc being approved)
**Author:** ansubramania
**Date:** 2026-04-30
**Sibling-of:** `nemo-gym-scale-simulation-design.md` (the harness this extends)
**Scope:** make the scale-sim harness drive load against N agent instances on one head node, so we can characterize the §3 sub-server-count "all-servers-on-the-head-node" cliff that the single-agent matrix does not currently cover.

---

## 0. TL;DR

Today's `tools/scale_sim/load_driver.py` rewrites `agent_ref` to a single `agent_name` on every row, so 100 % of dispatched `/run` requests go to one agent regardless of input. That is the single line that prevents the multi-agent sweep. Production's `RolloutCollectionHelper._post_subroutine` (`nemo_gym/rollout_collection.py:446`) honors per-row `agent_ref.name` and dispatches per-request — and that is what the multi-agent harness must mimic exactly.

The plan:

1. **Deterministic config generator** (`tools/scale_sim/configs/_gen_multi_agent.py`) emits a YAML containing N agent instances + N resources instances + 1 shared model on non-overlapping ports below the ephemeral floor. **Topology is locked** (§3) — no flag, generator emits one shape only.
2. **Driver extension** (one new flag + ~30 lines in `load_driver.py`) reads a list of agent names from config, round-robins `agent_ref.name` across input rows, and otherwise leaves the dispatch loop and instrumentation unchanged.
3. **Two driver modes**:
    - `--mode=spinup_only`: bring up N, idle for 30 s, sample metrics, kill. Answers "what's the FD/RAM/spinup-time cost of N sub-servers existing." Cheap.
    - `--mode=loaded` (default): bring up N, drive load round-robin across agents, measure. Answers everything Axis A measures, broken down per-agent and in aggregate.
4. **Sweep**: 19 cells total (7 spinup-only at `n_agents = {1,4,16,32,64,128,256}` + 6 loaded at fixed total concurrency + 6 loaded at fixed per-agent concurrency, both at `n_agents = {1,4,16,32,64,128}`). One `ng_run` invocation per cell, full teardown between cells. No batch wall-clock cap (CPU partition allows long jobs on this cluster).
5. **Predictions** below in §8 — what we expect to see, so the experiment can falsify them.

Everything else (instrumentation, retry tracking, kernel watcher, sweep-runner-per-cell isolation, the unchanged `simple_agent` and `synthetic_*` server implementations) is reused as-is. **No changes to existing checked-in cell configs (`smoke.yaml`, `single_agent_base.yaml`, `axis_c_8k_4mb.yaml`)** — they keep working unchanged because `agent_names` falls back to the existing `agent_name` field when not provided.

---

## 1. What's already there, what's missing

### 1.1 Already there
- `synthetic_resources_server` and `synthetic_model_server` with all the knobs (latency, body size, cpu burn, failure injection).
- `simple_agent` (the real one, unchanged) that talks to one model + one resources server per instance.
- `sweep_runner.py` that brings up `ng_run`, waits for all sub-servers ready, drives load, tears down, records metrics, repeats per-cell.
- `instrumentation.py` (`RetryTracker`, `ProcessMetricsSampler`, `KernelWatcher`) — reusable wholesale.
- `bare-metal Slurm launcher` (`run_on_slurm_baremetal.sh`) — reusable wholesale, just needs a new `DRIVER_SCRIPT`.

### 1.2 Missing
1. **Per-row agent dispatch in `load_driver.py`.** Single offending line:
   ```125:125:tools/scale_sim/load_driver.py
   row["agent_ref"] = {"name": self.agent_name}
   ```
   needs to become "round-robin across `self.agent_names: list[str]`", and the inner `server_client.post(server_name=self.agent_name, ...)` needs to read from the row.
2. **Config generator for N agents.** No way today to produce a YAML with 64 agent instances + matching model + resources + ports, by hand or otherwise.
3. **Spinup-only driver mode.** Today's driver always loads input data and drives requests. For the multi-agent sweep we want a "just bring up N, sample idle resource cost, kill" mode — no JSONL, no requests.
4. **Per-agent breakdown in `summary.json`.** Today's summary aggregates over the whole run. For the multi-agent sweep we want `per_agent: {agent_0: {n_rollouts, p99, retry_rate}, ...}` so we can see whether the cliff is uniform (every agent slows together — head-node-shared cause) or skewed (some agents fail harder than others — uvicorn-process-local cause).

That's the whole gap.

---

## 2. Production rollout loop — what we're mimicking

The shape we must match, copied verbatim from `nemo_gym/rollout_collection.py`:

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

Two points worth nailing:

- **Dispatch is via `server_name` lookup**, not direct URL. `ServerClient` resolves names through the global config that the head server publishes at startup (`load_from_global_config` in `load_driver.py:131`). This means: as long as `ng_run` brought up agents named `agent_0` through `agent_{N-1}`, the driver doesn't need any per-agent URL bookkeeping — `server_client.post(server_name="agent_42", ...)` Just Works.
- **One `aiohttp.ClientSession`, one event loop, fanning out across N targets.** The whole point of the multi-agent sweep is that every increase in N is "more sockets multiplexed on the same single consumer asyncio event loop" — the M3 mechanism the design doc calls out as the load-bearing one. We **must not** spawn N driver processes or N event loops; that would change what's being measured.

The driver's outer retry on `ClientPayloadError` and inner retry on `ServerDisconnectedError`/`ClientOSError` (in `nemo_gym.server_utils.request`) stay as-is.

---

## 3. Topology — locked

**Topology: N agents + N resources + 1 shared model.** Total sub-servers per cell = `2N + 1` (+ 1 head).

Rationale: this is the production shape (Ultra has ~1 vLLM cluster + ~10–20 specialized verifiers), and answers the operationally relevant question. Variant topologies (per-agent model, fully shared) are deferred to follow-ups; the generator hardcodes the shared-model layout for now and exposes no `--topology` flag. If a follow-up wants the alternatives, the generator can grow a flag at that time without changing anything else in the harness.

This means each cell of the multi-agent sweep brings up exactly:
- 1 head server
- 1 `synthetic_model` instance (shared)
- N `synthetic_resources` instances (one per agent)
- N `simple_agent` instances (each pointing at its own resources + the shared model)

Total uvicorn processes per cell: `2N + 2`.

---

## 4. Driver modes

### 4.1 `--mode=loaded` (default — multi-agent with traffic)

Same shape as today's driver:

1. Read `scale_sim.agent_names: [str]` from the cell config (instead of `agent_name`).
2. Cycle / pad input rows to `total_requests` (existing logic).
3. Round-robin `row["agent_ref"]["name"]` = `agent_names[i % N]` for the i-th rollout (replaces `load_driver.py:125`).
4. Dispatch loop unchanged: one event loop, one global aiohttp client, semaphore-capped, `server_client.post(server_name=row["agent_ref"]["name"], ...)`.
5. Instrumentation unchanged. Add a per-agent counter to `RetryTracker` so the final summary breaks down by agent.

This is the shape that mimics `RolloutCollectionHelper.run_examples` end-to-end. ~30 LOC change.

### 4.2 `--mode=spinup_only` (multi-agent without traffic)

For each cell:

1. Generate the YAML for `N` agents + the topology's model/resources count.
2. Launch `ng_run` exactly as `sweep_runner.py` does today.
3. Wait for "All M / M servers ready!" (M = topology-dependent total). Record `spinup_time_s`.
4. Sleep for `idle_window_s` (default 30 s) while `ProcessMetricsSampler` and `KernelWatcher` sample. Per-process metrics are a list of (head_actor + every sub-server uvicorn process) — the existing sampler already enumerates child PIDs of `ng_run`, no change needed.
5. Tear down `ng_run`, snapshot final metrics.
6. Record per-cell summary: `n_agents`, `spinup_time_s`, `idle_max_fd_total`, `idle_total_rss_bytes`, `idle_loadavg_p99`, `idle_softirq_cpu_pct` (from `/proc/stat`), `tcp_listen_count` (from `/proc/net/tcp`).

Spinup-only is the cheap pre-flight: one cell takes (spinup + 30 s + teardown) ≈ 1–2 minutes. The whole sweep at `n_agents=[1,4,16,32,64,128]` × 2 topologies = 12 cells ≈ 20 minutes wall-clock. Worth running first because if `n_agents=128` can't even spin up, the loaded-mode sweep at that N is moot.

### 4.3 Both modes share the same per-cell life cycle

Same `sweep_runner._run_one_cell` orchestration: launch ng_run, wait ready, run driver in chosen mode, kill, sleep 5s for TIME_WAIT to drain. The mode is a flag on `load_driver.py`, not a separate runner.

---

## 5. Config generation

A new file `tools/scale_sim/configs/_gen_multi_agent.py` (underscore prefix → not a checked-in cell config, generated on the fly per sweep).

### 5.1 CLI

```
python tools/scale_sim/configs/_gen_multi_agent.py \
    --n-agents 64 \
    --base configs/multi_agent_base.yaml \
    --out configs/_generated/multi_agent_n64.yaml
```

The base file (`multi_agent_base.yaml`, checked in) carries the per-server knobs (latencies, body sizes, etc.) and the `scale_sim:` block (concurrency, total_requests, early-stop). The generator stamps out the shared model + N resources + N agents on top.

### 5.2 Port allocation

Single port range below the ephemeral floor (9000 on OCI-HSG, 32768 on stock Linux). At `n=N`:

| Server | Port |
| --- | --- |
| head | 5000 |
| model (shared) | 5001 |
| resources_0 | 5002 |
| ... | ... |
| resources_{N-1} | 5001 + N |
| agent_0 | 5002 + N |
| ... | ... |
| agent_{N-1} | 5001 + 2N |

Port budget: 8999 - 5000 = 3999 usable. At `N=256` we use 514 ports — comfortable. The ceiling is realistically `N ≈ 1995` before we run out of port space, which is well past where every other resource will have failed first.

### 5.3 Generated YAML shape (T-shared-model, N=2 example)

```yaml
head_server: { host: 0.0.0.0, port: 5000 }

global_aiohttp_connector_limit: 102400
global_aiohttp_connector_limit_per_host: 16384

# One model, shared by all agents
shared_synthetic_model:
  responses_api_models:
    synthetic_model:
      entrypoint: app.py
      port: 5001
      async_latency_ms: 200
      prompt_tokens: 512
      output_tokens: 1024
      include_token_ids_and_log_probs: true

# Per-agent resources
synthetic_resources_0:
  resources_servers:
    synthetic_resources:
      entrypoint: app.py
      port: 5002
      tool: { async_latency_ms: 100, body_size_bytes: 16384, body_shape: flat_padding }
      verify: { async_latency_ms: 50, body_size_bytes: 4096, body_shape: flat_padding }

synthetic_resources_1:
  resources_servers:
    synthetic_resources:
      entrypoint: app.py
      port: 5003
      # ... same knobs ...

# Per-agent agents, each pointing at its own resources but the shared model
synthetic_simple_agent_0:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      port: 5004
      max_steps: 1
      resources_server: { type: resources_servers, name: synthetic_resources_0 }
      model_server:     { type: responses_api_models, name: shared_synthetic_model }
      datasets:
        - { name: multi_agent, type: example, jsonl_fpath: data/multi_agent_10k.jsonl, num_repeats: 1 }

synthetic_simple_agent_1:
  responses_api_agents:
    simple_agent:
      entrypoint: app.py
      port: 5005
      max_steps: 1
      resources_server: { type: resources_servers, name: synthetic_resources_1 }
      model_server:     { type: responses_api_models, name: shared_synthetic_model }
      datasets:
        - { name: multi_agent, type: example, jsonl_fpath: data/multi_agent_10k.jsonl, num_repeats: 1 }

scale_sim:
  agent_names: [synthetic_simple_agent_0, synthetic_simple_agent_1]
  concurrency: 4096
  total_requests: 10000
  early_stop_failure_rate: 0.10
  early_stop_retry_rate: 0.30
  early_stop_wall_clock_s: 600
  early_stop_window_s: 30
```

Key observation: the **server-type directory** (`responses_api_models/synthetic_model/`, `resources_servers/synthetic_resources/`, `responses_api_agents/simple_agent/`) is shared across all instances; only the top-level **instance name** differs. This means there is **one `.venv/` per server type, not one per instance** — that is what makes N=128 feasible. The first cell of the sweep pays the cold uv cost three times (one per server type); every replica after that reuses the same on-disk venv. Verified by reading `nemo_gym/cli.py`'s server-spawn code path.

If this assumption turns out wrong (i.e. ng_run does something instance-specific in the venv-build path that we're missing), the fallback is: pre-create the three venvs once in the base setup, then symlink them in for each instance. Detected on first spinup-only cell at N=4 — if disk usage explodes, switch to the symlink fallback.

### 5.4 Data file

A single shared `data/multi_agent_10k.jsonl` (generated once via `data/generate_data.py`) without `agent_ref` — the driver assigns it round-robin. The dataset declared on each agent is just for ng_run's "register a dataset" lifecycle plumbing; the driver reads its own JSONL directly and does not consult agent.datasets.

---

## 6. Driver changes — minimal

`tools/scale_sim/load_driver.py`:

1. Replace `self.agent_name = scale_sim_cfg.agent_name` with:
   ```python
   self.agent_names = list(scale_sim_cfg.get("agent_names") or [scale_sim_cfg.agent_name])
   ```
   so single-agent configs (smoke, single_agent_base, …) keep working unchanged.
2. Replace `row["agent_ref"] = {"name": self.agent_name}` in `_load_input_rows` with:
   ```python
   row["agent_ref"] = {"name": self.agent_names[i % len(self.agent_names)]}
   ```
3. Replace `server_name=self.agent_name` in `_post_subroutine` with `server_name=row["agent_ref"]["name"]` (production-shape).
4. Add `--mode {loaded,spinup_only}` CLI flag. When `spinup_only`, skip `_load_input_rows`, skip the dispatch loop, just sample for `idle_window_s` and dump the single-row spinup summary.
5. Extend `RetryTracker` with `record_completion(rollout_idx, succeeded, agent_name)` and emit per-agent breakdown in `summary_all()`.

That is the entire driver-side change. ~50 LOC, well-tested by the smoke + single_agent_base cells continuing to pass after the refactor (single-element `agent_names` list is the regression check).

`sweep_runner.py`:

- Add `--driver-mode` flag that pipes through to `load_driver.py`. Default `loaded`.
- No other change. Spinup-only mode reuses every life-cycle hook.

---

## 7. Sweep matrix

Time budget on this cluster's `cpu` partition is unbounded at submission, so the matrix is sized to give a clean characterization rather than to fit a wall-clock. Variants below run as one batch via `run_on_slurm_baremetal.sh batch` with `DRIVER_SCRIPT=tools/scale_sim/run_multi_agent_sweep.sh`.

All cells share the body-size and per-hop knobs from `single_agent_base.yaml` so results are directly comparable to the single-agent baseline:
- `output_tokens` = 1024 (model body ~30 KB)
- tool body 16 KB / verify body 4 KB
- model `async_latency_ms=200`, tool `async_latency_ms=100`, verify `async_latency_ms=50`
- `simple_agent.max_steps = 1` (single hop). Multi-hop is covered by M1's `03_n_hops_curve` at fixed N=1 and is not crossed with N here — see §7.4.

### 7.1 Spinup-only — pre-flight resource cost of N existing

| Knob | Values |
| --- | --- |
| `n_agents` | 1, 4, 16, 32, 64, 128, 256 |
| `idle_window_s` | 30 |

7 cells. Cheap (each cell is bring-up + 30 s idle + teardown). Run **first** — if N=256 can't even spin up, the loaded sweep at that N is moot and we cap there. If it does, we have a clean RSS / FD / spinup-time / tcp-listen-count curve in N as the first deliverable.

### 7.2 Loaded sweep — fixed total concurrency

| Knob | Values |
| --- | --- |
| `n_agents` | 1, 4, 16, 32, 64, 128 |
| `concurrency` (total) | 4096 |
| `total_requests` | 10000 |

6 cells. Per-agent concurrency is `4096 / n_agents`, so N=64 means each agent sees only 64 in-flight at peak. Tests "does adding more sub-servers cost something at fixed total load" — i.e. is the cost N-multiplicative just from sub-server existence, or only proportional to per-server load?

### 7.3 Loaded sweep — fixed per-agent concurrency

| Knob | Values |
| --- | --- |
| `n_agents` | 1, 4, 16, 32, 64, 128 |
| `concurrency` (per-agent) | 256 |
| total concurrency | 256 × n_agents (1024 at N=4 → 32 K at N=128) |
| `total_requests` | 4000 × n_agents |

6 cells. The "concurrency grows with N" variant. Total fan-in at N=64 matches Axis-A's 16 K cell — this is the head-side comparison that confirms whether M3 is consumer-side and N-independent (§8 prediction 1).

### 7.4 What's deliberately *not* crossed

- **N × hops**: hops covered by the single-agent `03_n_hops_curve` at N=1. Crossing them would add 4–6 cells per N value with predictable interaction (per-hop session middleware × per-agent fan-in is straightforwardly multiplicative). Defer until single-agent results say something surprising about hops at N=1 first.
- **N × body size**: body size covered by M1's `02_output_tokens_far_tail` at N=1. Same logic.
- **N × topology**: locked to shared-model per §3.
- **N × multi-turn realism beyond `max_steps=1`**: today's `simple_agent` is what production uses for single-shot benchmarks. Realistic multi-turn (proof_refinement_agent, swe_agent) involves agent-side state we don't model here. Out of scope for the multi-agent sweep.

### 7.5 Total cell count

19 cells (7 spinup-only + 12 loaded). Wall-clock estimate, given each cell pays no cold uv after the first (we share venvs per server type per §5.3): roughly `spinup_time(N) + load_time` per cell, expected to range from <1 min at N=1 to several minutes at N=128 once spinup grows. Whatever the total, the batch job has no `--time` cap.

---

## 8. Predictions to falsify

Same format as the design doc's §6. Predictions are concrete enough that the experiment can refute them.

| Variant | n_agents | Predicted breakage | Predicted dominant mechanism |
| --- | --- | --- | --- |
| spinup-only | 64 | spinup time crosses 60 s | uvicorn cold-start + Ray actor registration serialization |
| spinup-only | 128 | spinup time crosses 180 s OR idle FD count crosses 60 K | A1 ulimit / kernel TCP listen-table |
| spinup-only | 256 | spinup either fails (timeout / ENFILE) or idle RSS exceeds host RAM | RAM (256 × ~300 MB ≈ 75 GB just at idle) |
| loaded, fixed total=4 K | 32 | per-rollout p99 *improves* slightly vs N=1 — load actually shards across more event loops on the *server* side | counter-prediction worth checking |
| loaded, fixed total=4 K | 128 | per-rollout p99 *degrades* — more sockets the single consumer event loop must drain dominates the per-server win on the agent side | M3, head-side |
| loaded, per-agent=256, N=32 | 32 | total concurrency 8 K — should match Axis-A 8 K results within 20 % | M3, agent-independent, baseline normalization |
| loaded, per-agent=256, N=64 | 64 | total concurrency 16 K — same M3 cliff predicted for Axis A 16 K. Confirms that the cliff is consumer-side and N-independent | M3 |
| loaded, per-agent=256, N=128 | 128 | total concurrency 32 K — head-actor FD count crosses 60 K, kernel TW table fills | M3 + kernel |

Key falsifiable claims:

1. **The cliff is consumer-side, not sub-server-side.** If `loaded fixed-per-agent N=64` and `Axis-A concurrency=16K with N=1` show the same p99 / failure rate, the M3 mechanism is confirmed agent-count-independent. If they diverge, sub-server count is contributing something we hadn't modeled.
2. **Spinup time is super-linear in N**. The §3 design-doc prediction is "spinup > 90s at n_servers=64". Easy to confirm.
3. **Per-agent retry distribution is uniform**. If N=64 causes some agents to retry 10× more than others, we have skewed contention (some uvicorn process is wedged) rather than uniform head-side back-pressure. The per-agent breakdown in `summary.json` is what tests this.

---

## 9. Output schema

`results/<sha>/<exp_name>/<cell>/summary.json` for loaded mode gets a `per_agent` block:

```json
{
  "config_path": "...",
  "agent_names": ["synthetic_simple_agent_0", ..., "synthetic_simple_agent_63"],
  "concurrency": 4096,
  "topology": "shared_model",
  "n_agents": 64,
  "retry_summary": { "...": "as today, aggregate" },
  "latency_summary": { "...": "as today, aggregate" },
  "per_agent": {
    "synthetic_simple_agent_0": {
      "n_rollouts": 156,
      "failure_rate": 0.0,
      "retry_rate": 0.02,
      "p50_s": 0.18, "p99_s": 0.42, "max_s": 1.1
    },
    "...": "..."
  },
  "stop_reason": null
}
```

For spinup-only mode, a different but flatter shape:

```json
{
  "n_agents": 64,
  "topology": "triplet",
  "n_total_subservers": 192,
  "spinup_time_s": 81.4,
  "spinup_outcome": "ready",      // or "timeout" or "process_died"
  "idle_window_s": 30,
  "idle_max_fd_total": 18342,
  "idle_total_rss_bytes": 38123456789,
  "idle_loadavg_1m": 1.2,
  "tcp_listen_count": 195,
  "softirq_cpu_pct_p99": 4.1,
  "per_subserver": {
    "synthetic_simple_agent_0": { "rss_bytes": 198000000, "fd_count": 23, "cpu_pct_idle": 0.1 },
    "...": "..."
  }
}
```

`run_multi_agent_sweep.sh` aggregates both sets into `multi_agent_master.csv` with one row per cell — same shape as the existing single-agent master CSV.

---

## 10. Failure-mode handling

A few specific cases the harness must handle gracefully. None require new infra; all are caught by extending the existing sweep_runner timeouts and logging.

1. **Spinup timeout at high N**. `sweep_runner._wait_for_servers_ready` already takes `timeout_s`. For multi-agent cells we'll bump per-cell to `max(300, 60 * sqrt(N))` to give cold uv + N-process spawn enough time. If timeout fires, the cell records `spinup_outcome: timeout` and the sweep continues to the next N.
2. **Port collision with stragglers**. If a previous cell didn't tear down cleanly and ports 5000-5512 are still bound, ng_run will fail on `bind()`. Existing 5 s post-cell sleep is not enough at N=64 (3 × 64 sockets in TIME_WAIT). Bump to `max(5, N // 8)` seconds, and have `sweep_runner` log `ss -tnlp | grep 50` before each cell as a tripwire.
3. **FD exhaustion at high N**. The bare-metal launcher already sets `ulimit -Sn` to the hard limit; verify it's ≥ 1 M before running the N=128 cell. If the cluster's hard cap is lower (some sites cap at 64K), reduce the max N in the sweep and note the cap as an experimental limit, not a finding.
4. **Head actor RSS oversubscription**. `ProcessMetricsSampler` should refuse to start the next cell if `psutil.virtual_memory().available < 4 GB` from the previous cell — likely indicates a Ray actor leak. Existing teardown is `SIGINT` → `SIGKILL` after 15 s; that should be enough but worth logging RSS after teardown to confirm.

---

## 11. Implementation plan (post-approval)

Linear, single contributor, ~half a day:

1. **Read-only validation pass (10 min)**: confirm `nemo_gym/cli.py`'s server-spawn code reuses `<server_type>/<name>/.venv` regardless of how many top-level config keys reference it. Fall back to symlinks only if this assumption fails.
2. **`load_driver.py` changes (~1 hour)**: agent_names list, round-robin in `_load_input_rows`, per-agent breakdown in `RetryTracker`, `--mode` flag with `spinup_only` branch. Smoke + single_agent_base must keep passing as regression.
3. **`configs/multi_agent_base.yaml` + `configs/_gen_multi_agent.py` (~1 hour)**: deterministic generator. Unit-test by generating N={1,4,8} configs and running them through `ng_dump_config` to confirm the merged shape is identical to a hand-written N=4 reference.
4. **`run_multi_agent_sweep.sh` (~30 min)**: a new driver script for `DRIVER_SCRIPT=`, modeled on `run_single_agent_sweep.sh`. Iterates the §7 matrix, calls `_gen_multi_agent.py` then `sweep_runner.py` per cell.
5. **First spinup-only run (~30 min wall + analysis)**: validates everything below N=128 actually starts; surfaces any cluster-specific limits (FDs, ports, RAM) before committing to the loaded run.
6. **Loaded run (~2 h wall)**: the rest of §7.
7. **Results writeup**: extend `nemo-gym-scale-simulation-results.md` with a "multi-agent" section comparing measurements to the §8 predictions.

Total: ~3.5–4 hours of focused implementation + ~3 hours of compute. Comfortable in a single working day once the single-agent sweep finishes.

---

## 12. Decisions to lock before implementing

These are the questions where the doc has a position but reasonable people might pick differently. Worth flagging.

| # | Question | Default | Alternative |
| --- | --- | --- | --- |
| 1 | Topology | `shared_model` (locked, §3) | per-agent model or fully-shared — defer to follow-up |
| 2 | Round-robin assignment vs. weighted | round-robin | weighted (e.g. 80/20) — useful for "what if one benchmark dominates" but out of scope here |
| 3 | Multi-process driver fallback if M3 caps total throughput before we hit the N axis | not implemented | adds N driver processes — would change what's being measured. Keep it single-process and let M3 be the cap. |
| 4 | Per-server-type venv sharing | trust ng_run does it (default) | symlink fallback if it doesn't |
| 5 | Should `agent_ref` be on the input JSONL (production-realistic) or assigned by the driver (cleaner) | assigned by driver | on JSONL — but then the JSONL itself is N-specific, breaks reuse across cells. Production's `RolloutCollectionHelper` already fills `agent_ref` from a config-level fallback when missing, so driver-assigned matches that path. |
| 6 | Time bound on the batch job | none — let it run as long as needed | wall-clock cap (was 4 h before this revision; removed per cluster policy) |

Default for all six is what §1–§11 above describe. If any of these decisions change, several sections need updates.

---

## 13. Out of scope for this design (deliberately)

For honesty:

- **Multi-node multi-agent** (the structural fix where head + sub-servers shard across nodes). This is exactly what `nemo-gym-refactor-milestone-plan.md` covers; not duplicating it here.
- **Mixing real and synthetic benchmarks in one multi-agent sweep**. The §4.8.4 cross-check in the parent design doc covers single-benchmark calibration; mixing is a follow-up.
- **Realistic per-agent load skew** (some agents 10× heavier than others). Production today is mostly uniform-per-agent; if asymmetric loads become a thing, add a `agent_weights: [w0, w1, ...]` knob in a follow-up.
- **TLS / cookie sharding across N agents**. SimpleServer's session middleware writes one cookie per request regardless of agent; the per-agent cookie growth from the parent doc's §3 Axis C1 still applies, just multiplied by N. We measure aggregate, not per-agent cookie cost.

---

## Appendix — file plan

```
investigations/nemo-gym-multi-agent-design.md   (this doc)

tools/scale_sim/
├── configs/
│   ├── multi_agent_base.yaml             (NEW)  — knob template, no agents/models
│   ├── _gen_multi_agent.py               (NEW)  — config generator
│   └── _generated/                       (gitignored) — written by the generator at sweep time
├── load_driver.py                        (CHANGED) — agent_names list, --mode flag
├── instrumentation.py                    (CHANGED) — RetryTracker per-agent breakdown
├── sweep_runner.py                       (CHANGED) — --driver-mode pipe-through
├── run_multi_agent_sweep.sh              (NEW)  — driver matrix runner
└── run_on_slurm_baremetal.sh             (UNCHANGED) — already extended for DRIVER_SCRIPT env
```

End of design.
