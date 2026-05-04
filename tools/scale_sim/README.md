# scale-sim

Scale-test harness for nemo-gym. Drives synthetic load against the real
`ServerClient` / aiohttp / `RolloutCollectionHelper` stack to find where the
existing gym architecture breaks under load.

## What it tests

Four axes against the unchanged gym architecture:

1. **Concurrent requests** — how many in-flight `/run` requests one head node can sustain
2. **HTTP payload size** — sequence-length-driven body sizes, `output_tokens` from 16K to 1M
3. **HTTP hops** — number of model↔tool round-trips per `/run` (1, 4, 16, 64)
4. **Number of co-located sub-servers** on one head node (the multi-agent sweep — `run_multi_agent_sweep.sh`)

For each cell, the harness records: failure rate, retry rate, per-rollout retry
counts, latency p50/p99/p999/max, and per-process + kernel-level resource usage
sampled over time.

## How it works

Three synthetic services stand in for real ones; gym's existing `simple_agent`
is reused unchanged.

- **synthetic_resources** — controllable `/verify` and `/synthetic_tool` endpoints. Knobs for latency, body size, body shape, and failure injection.
- **synthetic_model** — controllable `/v1/responses`. Returns a `NeMoGymResponse` shaped like real RL training output (token IDs, log probs, optional reasoning items). Body size is *derived* from `prompt_tokens` / `output_tokens` / `n_reasoning_items`, not set as raw bytes.
- **simple_agent** — the existing gym agent, unchanged. `max_steps = N` controls hop count.

The load driver (`load_driver.py`) fires N concurrent POST `/run` requests
through gym's real `ServerClient` / aiohttp / `RolloutCollectionHelper` stack
against the agent. Per-rollout retry tracking, sliding-window early-stop,
per-process and kernel-level instrumentation are all in-process.

A sweep runner (`run_sweep.py`) generates a matrix of cell configs from a base
yaml + override knobs, runs each cell as a separate `ng_run` invocation with
clean teardown between cells, and aggregates per-cell summaries into one CSV.

## Knobs

### Load driver (`scale_sim:` block in any cell yaml)

| Knob | Default | Meaning |
| --- | --- | --- |
| `agent_name` | required | Top-level config key of the agent server. |
| `concurrency` | required | Max in-flight `/run` requests at once. |
| `total_requests` | required | Total `/run` invocations to drive. Cycles input rows if shorter. |
| `semaphore_enabled` | `true` | If `false`, fire all requests with no client-side throttle (defect #5 ablation). |
| `early_stop_failure_rate` | `0.10` | Stop cell early if rolling-window failure rate exceeds this. |
| `early_stop_retry_rate` | `0.30` | Stop cell early if rolling-window retry rate exceeds this. |
| `early_stop_wall_clock_s` | `600` | Hard wall-clock deadline. |
| `early_stop_window_s` | `30` | Sliding-window size for the rate computations. |

### synthetic_model (`/v1/responses`)

| Knob | Default | Meaning |
| --- | --- | --- |
| `prompt_tokens` | 512 | Length of `prompt_token_ids` list on the output message. |
| `output_tokens` | 256 | Length of `generation_token_ids` and `generation_log_probs`; drives output `text` via `chars_per_token`. |
| `chars_per_token` | 4 | Output `text` length per token. |
| `include_token_ids_and_log_probs` | `true` | RL training default. Adds ~16 bytes/token to body size. |
| `n_reasoning_items` | 0 | Number of `<think>`-style reasoning items prepended to output. |
| `reasoning_tokens_per_item` | 0 | Tokens per reasoning item (each gets its own ids/probs). |
| `vocab_size` | 200000 | Upper bound on randomly-generated token IDs. |
| `tool_name` | `synthetic_tool` | Function name in the emitted `function_call`. |
| `async_latency_ms` | 50 | `await asyncio.sleep(...)` — yields the event loop. |
| `cpu_burn_ms` | 0 | `time.perf_counter()` busy-loop — holds the event loop. |
| `latency_dist` | `fixed` | `fixed` or `lognormal`. |
| `inject_500_rate` | 0.0 | P(handler returns HTTP 500). |

Body bytes ≈ `output_tokens * chars_per_token + (prompt + output) * 6 + output * 10 + n_reasoning * reasoning_tokens * 20 + ~1 KB`.

### synthetic_resources (`/synthetic_tool` and `/verify`)

Each endpoint has its own block with the same shape:

| Knob | Default | Meaning |
| --- | --- | --- |
| `async_latency_ms` | 0 | `await asyncio.sleep(...)`. |
| `cpu_burn_ms` | 0 | Busy-loop. |
| `body_size_bytes` | 1024 | Total byte length of the response body's variable portion. |
| `body_shape` | `flat_padding` | `flat_padding` (one big string) or `realistic_messages` (many small dict chunks — real GC pressure). |
| `realistic_chunk_bytes` | 256 | Per-chunk size when `body_shape=realistic_messages`. |
| `latency_dist` / `body_dist` | `fixed` | Or `lognormal` for tail-distribution sweeps. |

Plus failure injection at the server level:

| Knob | Default | Meaning |
| --- | --- | --- |
| `inject_500_rate` | 0.0 | P(handler returns HTTP 500). |
| `inject_hang_rate` | 0.0 | P(handler awaits `inject_hang_seconds`). Simulates a wedged sub-server. |
| `inject_hang_seconds` | 3600 | Hang duration when `inject_hang` fires. |

### Per-sub-server uvicorn

| Knob | Default | Meaning |
| --- | --- | --- |
| `num_workers` | 1 | Uvicorn worker count for that sub-server. >1 spawns N processes via uvicorn's multi-worker mode. |
| `host` | `0.0.0.0` | |
| `port` | (required) | Sub-server ports. Pinned 5001-5003 in the shipped configs. |

### Sweep matrix knobs (`run_sweep.py` CLI flags)

Each flag takes a comma-separated list. Cross-product or `--mode zip` for parallel lists.

| Flag | Maps to |
| --- | --- |
| `--concurrency` | `scale_sim.concurrency` |
| `--total-requests` | `scale_sim.total_requests` |
| `--semaphore-enabled` | `scale_sim.semaphore_enabled` |
| `--num-workers` | All 3 sub-servers' `num_workers` |
| `--n-hops` | `simple_agent.max_steps` |
| `--prompt-tokens` | `synthetic_model.prompt_tokens` |
| `--output-tokens` | `synthetic_model.output_tokens` |
| `--n-reasoning-items` | `synthetic_model.n_reasoning_items` |
| `--reasoning-tokens-per-item` | `synthetic_model.reasoning_tokens_per_item` |
| `--include-token-ids-and-log-probs` | `synthetic_model.include_token_ids_and_log_probs` |
| `--tool-output-chars` | `synthetic_resources.tool.body_size_bytes` |
| `--tool-body-shape` | `synthetic_resources.tool.body_shape` |
| `--tool-async-latency-ms` | `synthetic_resources.tool.async_latency_ms` |
| `--model-async-latency-ms` | `synthetic_model.async_latency_ms` |
| `--model-cpu-burn-ms` | `synthetic_model.cpu_burn_ms` |

## Code organization

```
tools/scale_sim/
├── README.md                                  this file
├── pyproject.toml                             marker so gym's editable-install heuristic
                                               recognizes the deeper directory nesting
├── .gitignore                                 results/, .venv/, __pycache__/
│
├── resources_servers/synthetic_resources/     synthetic /verify + /synthetic_tool server
│   ├── app.py                                 SyntheticResourcesServer class
│   ├── configs/synthetic_resources.yaml       standalone defaults
│   ├── data/.gitignore                        gym dataset gitignore boilerplate
│   ├── tests/test_app.py
│   ├── requirements.txt                       -e nemo-gym[dev] @ ../../../../
│   └── README.md
│
├── responses_api_models/synthetic_model/      synthetic /v1/responses server
│   ├── app.py                                 SyntheticModel class. Sequence-length-driven payload.
│   ├── configs/synthetic_model.yaml
│   ├── tests/test_app.py
│   ├── requirements.txt
│   └── README.md
│
├── configs/
│   ├── smoke.yaml                             16 concurrent, ~5 KB body, ~30s sanity check
│   ├── axis_a_8k.yaml                         8K concurrent, output=16K (~430 KB body)
│   ├── axis_c_8k_4mb.yaml                     8K concurrent, output=16K + 1 reasoning × 128K tokens (~4.5 MB body)
│   ├── multi_agent_base.yaml                  multi-agent sweep base template (N agents + N resources + 1 model)
│   └── _gen_multi_agent.py                    multi-agent cell-config generator
│
├── data/
│   ├── generate_data.py                       generates synthetic JSONL of N tasks
│   └── .gitignore                             *.jsonl
│
├── load_driver.py                             drives load + records per-rollout retries
├── instrumentation.py                         RetryTracker + ProcessMetricsSampler + KernelWatcher
├── sweep_runner.py                            orchestrates one ng_run + load_driver per cell with teardown
├── run_sweep.py                               cross-product matrix generator → calls sweep_runner per cell
├── run_single_agent_sweep.sh                  runs the full single-agent matrix end-to-end (5 sub-sweeps)
├── run_multi_agent_sweep.sh                   runs the full multi-agent matrix end-to-end (3 sub-sweeps × 7 N values)
├── run_multi_agent_delta_sweep.sh             4-cell multi-agent validation subset (~45-60 min)
├── run_multi_agent_n256_check.sh              one-off N=256 spinup_only sanity check
├── run_multi_agent_rerun_failed.sh            reruns the cells that failed in a prior multi-agent sweep
├── analyze_multi_agent.py                     re-aggregates multi-agent cells into multi_agent_master_full.csv
├── run_on_slurm.sh                            Slurm launcher (interactive / batch modes; pyxis container)
└── run_on_slurm_baremetal.sh                  Slurm launcher (interactive / batch modes; bare-metal venv)
```

## Run

The full single-agent matrix end-to-end as a Slurm batch job:

```bash
DRIVER_SCRIPT=tools/scale_sim/run_single_agent_sweep.sh \
JOB_NAME=scale-single-agent \
  bash tools/scale_sim/run_on_slurm.sh batch
```

The full multi-agent (sub-server-count) matrix:

```bash
DRIVER_SCRIPT=tools/scale_sim/run_multi_agent_sweep.sh \
JOB_NAME=scale-multi-agent \
  bash tools/scale_sim/run_on_slurm.sh batch
```

On a cluster without pyxis containers, swap the launcher for the bare-metal one
(`run_on_slurm_baremetal.sh`); the `DRIVER_SCRIPT=` and `JOB_NAME=` knobs are
identical.

Defaults: cpu partition, cpu-normal qos, 4h walltime. Override `PARTITION`,
`SLURM_QOS`, `GPUS_PER_NODE`, `TIME` to switch to GPU node + batch partition.
The launcher generates an sbatch script under `scale-sim-logs/<job>-<ts>/`,
submits it, and returns immediately.

For one-off cells inside an interactive allocation:

```bash
# Inside the container, after `bash tools/scale_sim/run_on_slurm.sh interactive`:
source /opt/ray_venvs/nemo_rl.environments.nemo_gym.NemoGym/bin/activate
cd /opt/nemo-rl/3rdparty/Gym-workspace/Gym/tools/scale_sim

python data/generate_data.py --n 256 --user-input-size-bytes 256 --output data/smoke.jsonl

python run_sweep.py \
    --base configs/smoke.yaml \
    --input-jsonl data/smoke.jsonl \
    --concurrency 16 \
    --output-tokens 32 \
    --exp-name smoke
```

## Output

Each sweep produces `results/<git_sha>/<exp_name>_<ts>/`:

```
sweep_results.csv               one row per cell, all metrics — primary plotting input
spec.json                       the sweep parameters, for reproducibility
configs/<cell>.yaml             generated per-cell config
<cell>/
├── config.yaml                 the cell's resolved config
├── summary.json                failure_rate, retry_rate, latency percentiles, error class breakdown
├── per_rollout_retries.jsonl   one line per rollout — answers "what % retried, on what"
├── latencies.csv               per-successful-rollout latency for offline percentile work
├── process_metrics.csv         RSS / FDs / CPU% / asyncio loop lag p99, sampled every 100ms
├── kernel_metrics.csv          /proc/net/sockstat, /proc/sys/fs/file-nr, loadavg, sampled every 1s
├── ng_run.log                  uvicorn + sub-server stdout
└── driver.log                  load driver stdout
```

`run_single_agent_sweep.sh` aggregates every per-experiment `sweep_results.csv`
into one `results/<git_sha>/single_agent_master.csv` at the end.
`run_multi_agent_sweep.sh` writes `results/<git_sha>/multi_agent_master.csv` with
one row per cell.

## Operational notes

- **Bind-mount your local checkout**: `run_on_slurm.sh` overlays `${NRL_GYM_DIR}` (the gym repo on the host) over `/opt/nemo-rl/3rdparty/Gym-workspace/Gym` inside the container. Edits show up without rebuilding.
- **Per-server venvs are bind-mounted and persistent.** First cell of a fresh allocation pays cold uv-sync cost (~5-10 min). Subsequent cells reuse.
- **Each cell is a fresh `ng_run`.** sweep_runner.py tears down between cells and pauses 5s. No FD/TIME_WAIT bleed across cells.
- **Run from `tools/scale_sim/`.** `ng_run` resolves sub-server paths from cwd; the launcher does the cd for you.
- **Ports 5000-5003** by default (head + 3 sub-servers). Below the 9000 ephemeral floor on OCI-HSG.
- **`RAY_TMPDIR=/tmp` and `RAY_NODE_IP_ADDRESS=127.0.0.1`** baked into the container env so Ray works on Lustre and binds GCS where in-container peers can reach it.

## Known gaps

- Calibration step (matching synthetic harness against a real benchmark like `example_single_tool_call` to lock realistic knob values) not done.
- Multimodal payload representation (base64 images / video features in input messages) not implemented.

## Pointers

- Scalability design: `<nemo-rl-internal>/investigations/nemo-gym-scale-simulation-design.md`
- Refactor milestone plan: `<nemo-rl-internal>/investigations/nemo-gym-refactor-milestone-plan.md`
- Prior incident the work is grounded in: `<nemo-rl-internal>/investigations/2381291-nemo-gym-timeouts-connection-resets.md`
- Per-server READMEs: `resources_servers/synthetic_resources/README.md`, `responses_api_models/synthetic_model/README.md`
