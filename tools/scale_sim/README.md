# scale_sim

A load-testing harness for the NeMo Gym rollout path. It replaces the model and
resources servers with controllable stand-ins, drives load through the real gym
HTTP stack, and records where latency and failures grow as load increases.

The harness answers two questions:

1. How many concurrent rollouts can one head node serve before latency or
   failures become unacceptable?
2. Does wrapping gym in a Ray actor (the way the training framework does)
   introduce failures that a direct HTTP client does not see?

It uses synthetic servers, so it measures the request and orchestration paths,
not model quality. Everything runs on a single host.

## Layout

```
tools/scale_sim/
  resources_servers/synthetic_resources/   controllable /verify + /synthetic_tool
  responses_api_models/synthetic_model/     controllable /v1/responses
  configs/
    smoke.yaml          16-concurrent end-to-end sanity check
    single_agent.yaml   1 head + 1 model + 1 resources + 1 agent
    multi_agent.yaml     N agents + N resources + 1 shared model (base template)
    gen_multi_agent.py   stamp out a runnable N-agent config from multi_agent.yaml
    actor_repro.yaml     single-agent topology used by the Ray-actor reproducer
  data/generate_data.py  synthesize a JSONL of rollout tasks
  instrumentation.py     RetryTracker, ProcessMetricsSampler, KernelWatcher
  load_driver.py         drive concurrent /run requests over HTTP (single or multi-agent)
  gym_actor.py           wrap a gym topology as a Ray actor
  mock_trainer.py        drive trainer-side load on the Ray actor
  sweep_runner.py        run ng_run + load_driver per config, with clean teardown
  pyproject.toml         marker so ng_run treats the synthetic servers as editable
  results/               run artifacts (gitignored)
```

The agent is the real `responses_api_agents/simple_agent`. Only the model and
resources servers are synthetic, so the agent loop, retry logic, session
middleware, and aiohttp client are exactly what production uses.

## Setup

Run on a host with many cores and a high file-descriptor limit (high concurrency
opens tens of thousands of sockets). From the gym repo root:

```bash
uv venv --python 3.12
uv sync --extra dev
source .venv/bin/activate
export RAY_TMPDIR=/tmp UV_LINK_MODE=copy
ulimit -Sn $(ulimit -Hn)        # raise the soft FD limit to the hard limit
```

The synthetic servers get their own per-server venvs the first time `ng_run`
launches them, so the first run is slow.

## Experiment suite

`run_experiments.py` runs the defined experiments end to end, captures the host
specs, and writes comparable summaries. Each experiment varies one thing and
holds the rest fixed:

| Name | Varies | Tells us |
| --- | --- | --- |
| `concurrency_scaling` | simultaneous rollouts (64 → 131072) | most rollouts/sec the node sustains, and where latency climbs |
| `response_size_scaling` | response body (token ids + log probs for 16K → 10M context) | cost of serializing/transferring/parsing big bodies; where memory bites |
| `tool_call_depth_scaling` | tool/model calls per rollout (1 → 8) | the fixed overhead repeated on each HTTP round trip |
| `work_per_step_sensitivity` | simulated work per call (0 → 1000 ms) | how much framework overhead matters once each step does real work |
| `agent_fan_out` | number of agent servers (1 → 32) at fixed per-agent load | how spreading load across agents changes throughput and latency |
| `trainer_return_shape` | whole-batch vs streaming returns through the Ray actor | whether the Ray-actor boundary adds failures |

Run it interactively (a menu) or with flags:

```bash
cd tools/scale_sim
python run_experiments.py --list
python run_experiments.py --experiment concurrency_scaling --label workstation
python run_experiments.py --all --label workstation
```

Outputs:

- Raw per-cell artifacts under `results/<label>/...` (gitignored).
- One compact summary CSV per experiment under `findings/<label>/`, plus
  `findings/<label>/host.json` with CPU/RAM/FD-limit/OS/git-SHA. These are
  committed, so results from different machines (e.g. `workstation`, `slurm-cpu`)
  sit side by side for comparison.

### Running on a Slurm cluster

Build the venv once in an interactive allocation (the compute partition may lack
internet, and a cold `uv sync` over Lustre would eat job wall-clock):

```bash
salloc -A coreai_dlalgo_nemofw -p cpu --nodes=1 --time=1:00:00 --exclusive --mem=0
cd <repo-root>
export UV_LINK_MODE=copy RAY_TMPDIR=/tmp
uv venv --python 3.12 && uv sync --extra dev
exit
```

**Recommended: submit one job per experiment in parallel.** The `cpu_short`
partition caps wall-clock at **4 hours** (via its own `p_cpu_short` QoS, applied
automatically — don't pass `--qos`, our account only has the `normal` QoS). The
full suite is split one job per experiment. `submit_slurm_suite.sh` submits one
`--exclusive` job per experiment (each under the 4h cap), all sharing
`LABEL=slurm-cpu` so their per-experiment CSVs land together. End-to-end
wall-clock becomes the slowest single experiment, not the sum. Run from the repo
root:

```bash
bash tools/scale_sim/submit_slurm_suite.sh                      # all experiments
bash tools/scale_sim/submit_slurm_suite.sh concurrency_scaling agent_fan_out  # subset
# overrides: LABEL=slurm-cpu TIME=03:59:00 PARTITION=cpu_short \
#     bash tools/scale_sim/submit_slurm_suite.sh
```

The submitter pre-generates the shared driver data (`data/bench.jsonl` and the
declared dataset copies) **before** submitting, so the parallel jobs don't race
to create the same files on the shared filesystem.

**Single job (one experiment, or the workstation baseline).** `run_on_slurm.sh`
runs one job. Its `#SBATCH` defaults target this cluster (`coreai_dlalgo_nemofw`
/ `cpu_short` / `--exclusive` / `--mem=0` / `--cpus-per-task=40` /
`-t 03:59:00`); CLI flags override them:

```bash
LABEL=slurm-cpu EXPERIMENTS="--experiment concurrency_scaling" \
    sbatch tools/scale_sim/run_on_slurm.sh
# fast end-to-end sanity check before the long runs:
SMOKE=1 LABEL=smoke sbatch --export=ALL,SMOKE=1 -t 00:30:00 tools/scale_sim/run_on_slurm.sh
# workstation baseline (no Slurm, builds .venv if missing):
LABEL=workstation bash tools/scale_sim/run_on_slurm.sh
```

The wrapper sets `RAY_TMPDIR=/tmp`, raises the FD soft limit to the hard limit
(handling an `unlimited` hard limit), and sources `_ray_burst_env.sh` so ng_run
reaches "All N/N servers ready" at high sub-server fan-out. Use a distinct
`LABEL` per hardware so cluster results land in their own `findings/` directory
and sit beside the workstation results rather than colliding with them.

### Reproducing / collecting results

- Per-experiment summaries: `tools/scale_sim/findings/<LABEL>/<experiment>.csv`
  (committed). Host specs: `findings/<LABEL>/host.json`. Raw per-cell artifacts:
  `results/<LABEL>/...` (gitignored).
- Monitor: `squeue -u $USER`; live stdout per job: `tail -f scale-sim-<jobid>.out`
  (written to the submit dir / repo root).
- Parallel jobs do **not** clobber each other: each writes a distinct
  `<experiment>.csv` and `results/<LABEL>/<experiment>/` subtree. `host.json` is
  shared (last writer wins); since all CPU nodes are identical this is fine.
- Don't edit `run_experiments.py` / `sweep_runner.py` / `load_driver.py` while
  jobs are queued or running — they're read from the working tree at runtime, so
  in-flight edits would change behavior mid-run.

## Smoke test

```bash
cd tools/scale_sim
python data/generate_data.py --n 256 --user-input-size-bytes 256 --output data/smoke.jsonl
python sweep_runner.py --config configs/smoke.yaml --input-jsonl data/smoke.jsonl
```

Expect `summary.json` with `failure_rate=0`, `retry_rate=0`, 256 rollouts, and
sub-second p99.

## Single-agent concurrency sweep

The direct driver opens N concurrent `/run` requests against one agent through
one asyncio event loop and one aiohttp client — the same shape the rollout
collector uses. Raise `scale_sim.concurrency` to find where the consumer event
loop saturates.

```bash
cd tools/scale_sim
python data/generate_data.py --n 10000 --user-input-size-bytes 1024 \
    --output data/single_agent_10k.jsonl

# One cell at the config's default concurrency:
python sweep_runner.py --config configs/single_agent.yaml \
    --input-jsonl data/single_agent_10k.jsonl

# Or start ng_run yourself and drive it directly:
ng_run "+config_paths=[configs/single_agent.yaml]"      # in one shell
python load_driver.py --config configs/single_agent.yaml \
    --input-jsonl data/single_agent_10k.jsonl           # in another
```

To sweep concurrency, make one config per value (copy `single_agent.yaml` and
edit `scale_sim.concurrency`) and pass them all to `sweep_runner.py`. Each cell
runs in a fresh `ng_run` so process and kernel state do not carry over.

## Multi-agent sweep

The multi-agent topology runs N agents and N resources servers behind one shared
model. The driver round-robins rollouts across the agents, so total concurrency
grows with N while the consumer side stays a single event loop. This separates
server-side parallelism (more agents) from the consumer-side ceiling.

```bash
cd tools/scale_sim
python data/generate_data.py --n 10000 --output data/multi_agent_10k.jsonl

# Generate a runnable config per agent count:
for N in 1 4 16 32 64; do
  python configs/gen_multi_agent.py --base configs/multi_agent.yaml \
      --n-agents "$N" --out "configs/_generated/multi_agent_n${N}.yaml" \
      --total-requests 10000
done

# Run them back-to-back with clean teardown between cells:
python sweep_runner.py \
    --config configs/_generated/multi_agent_n*.yaml \
    --input-jsonl data/multi_agent_10k.jsonl
```

`gen_multi_agent.py` is deterministic and accepts `--concurrency`,
`--total-requests`, and `--semaphore-enabled` overrides per cell.

## Ray-actor reproducer

`mock_trainer.py` drives load the way the training framework does: it wraps a
single-agent gym topology in a Ray actor and calls it with blocking `ray.get`.
The actor spawns its own `ng_run`, so you do not start one separately.

```bash
cd tools/scale_sim
python data/generate_data.py --n 10000 --output data/actor_repro.jsonl

# Whole-batch returns through the Ray object store (one ray.get per prompt group):
python mock_trainer.py --config configs/actor_repro.yaml \
    --input-jsonl data/actor_repro.jsonl --output-dir results/actor_repro_sync \
    --mode sync_blocking --thread-count 256 --prompts-per-call 1 --num-steps 5

# Streaming returns (one ray.get per rollout, lag-bounded thread pool):
python mock_trainer.py --config configs/actor_repro.yaml \
    --input-jsonl data/actor_repro.jsonl --output-dir results/actor_repro_stream \
    --mode lag_batched_stream --thread-count 4 --prompts-per-call 256 --num-steps 5
```

## Output

Each run writes to `results/<run>/`:

| File | Contents |
| --- | --- |
| `summary.json` | failure rate, retry rate, latency percentiles, error-class counts |
| `per_rollout_retries.jsonl` / `per_call.jsonl` | one record per rollout / actor call |
| `latencies.csv` | per-rollout latency for offline percentiles |
| `process_metrics.csv` | driver RSS, file descriptors, CPU, event-loop lag |
| `kernel_metrics.csv` | host TCP socket counts, open file count, load average |
| `ng_run.log` / `driver.log` / `trainer.log` | captured stdout |

## Notes

- Servers use ports 5000-5999, below the typical kernel ephemeral-port floor.
  Adjust `head_server.port` and the per-server ports if your host differs.
- `sweep_runner.py` pre-starts Ray with pinned ports and clears leftover Ray and
  sub-server state before each cell. This keeps back-to-back cells reliable and
  matters most at high agent counts.
- Confirm `ulimit -Hn` is large (1M is comfortable) before high-concurrency or
  high-agent-count runs; a low hard limit becomes an artificial ceiling.
- Results under `results/` are run artifacts and are gitignored.
```
