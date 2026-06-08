# NeMo Gym scale test plan

Systematic matrix to run on **both** hosts (`workstation` 24-core / 251 GB, and
`slurm-cpu` 96-core / 251 GB) at the current branch HEAD, so results are
comparable. Results land in `findings/<label>/` and `results/<label>/`.

## Questions this plan answers

- **Q1. Can NeMo Gym, on its own (no RL), sustain 128K+ concurrent `/run`?** At a
  realistic (64k-token) body, where is the ceiling and what binds — the single
  consumer event loop, JSON byte rate, FDs/ephemeral ports, or the model?
- **Q2. Does MR 292 change Gym's own capacity?** The fix is trainer-side (cut
  concurrent `ray.get` from ~1000 to 2, actor streams results). Hypothesis: it
  bounds how hard RL *drives* Gym; it does **not** raise Gym's internal HTTP
  throughput. Verify by comparing Gym-standalone vs the fixed calling pattern at
  matched in-flight.
- **Q3. How far does the MR 292 fix let the integrated system scale** before even
  the streaming shape hits a wall in the current regime?
- **Q4. Does the production connection-reset (`ClientPayloadError`) reproduce, and
  which mitigation resolves it** — lower concurrency, streaming return,
  spawn-jitter, or keep-alive force-close?
- **Q5. How does the picture change with real model latencies** instead of the
  near-zero synthetic proxy?

## Dimensions (swept across the matrix)

| Dimension | Values |
| --- | --- |
| Host | workstation (24c), slurm-cpu (96c) — run everything on both |
| Driver / load shape | direct concurrency (`load_driver`); actor threaded-RPC; actor streaming; burst+refit; streaming+refit |
| In-flight concurrency | 1 → 131072 (toward the 128K target) |
| Response body | 16k → 8M tokens (64k = training default) |
| Model latency | near-zero; added-fixed (0–1024 ms); realistic distribution (pareto/lognormal); small real vLLM (calibration) |
| Agents | 1 → 32 |
| Tool-call hops | 1 → 512 |

## Experiment matrix

Status legend: **rerun** = exists, re-run at HEAD on both hosts for a clean
comparable set; **new** = not yet built/run.

| # | Experiment | Varies (fixed) | Answers | Status |
| --- | --- | --- | --- | --- |
| 1 | `concurrency_scaling` | in-flight 1→131072 (64k body, 1 hop, no added work) | Q1, Q2 | rerun (both) |
| 2 | `response_size_scaling` | body 16k→8M tokens (concurrency 64) | Q1 | rerun (both) |
| 3 | `tool_call_depth_scaling` | hops 1→512 (concurrency 1, small body) | Q1 | rerun (both) |
| 4 | `work_per_step_sensitivity` | added per-call latency 0→1024 ms (64k body, conc 64) | Q5 (latency lever) | rerun (both) |
| 5 | `agent_fan_out` | agents 1→32 (64k body, 256/agent) | Q1 | rerun (both; workstation was partial) |
| 6 | `trainer_shape` | threaded whole-batch vs streaming, in-flight 256→65536 — **extend to 131072** | Q2, Q3 | rerun + extend (both) |
| 7 | `burst_repro` | baseline / spawn-jitter / keep-alive force-close — 16 agents, 8192 in-flight — **add 16k/32k/64k in-flight cells** | Q4 | new — run (both) |
| 8 | `streaming_under_refit` (new) | streaming return shape **with** a periodic refit drain — does streaming avoid the idle-reset? | Q4 | new — build + run |
| 9 | `realistic_latency` (new) | re-run #1 and #7 with `latency_dist=pareto` matched to real vLLM p50/p99 | Q5 | new — build + run |
| 10 | `real_model_calibration` (new) | `example_single_tool_call` + a small real vLLM at fixed concurrency; lock synthetic knobs to match within ~20% | Q5 | new — cluster (GPU) only |

## How each question gets answered

**Q1 (Gym standalone to 128K):** #1 is the headline — direct concurrency to
131072 at the 64k body. Read against #2 (byte ceiling), #5 (fan-out), and the
instrumentation (loop-lag, RSS, FDs, TCP). The 24-core workstation already
collapses ~15x at ≥16K in-flight where the 96-core sustains — so the standalone
answer is hardware-dependent and almost certainly "no, not at a realistic body
without sharding the consumer."

**Q2 (does MR 292 raise Gym's capacity?):** compare #1 (Gym standalone at
in-flight X) against #6 streaming (same in-flight X). If the sustained throughput
ceiling is the same, the fix does **not** make Gym faster — it only stops RL from
creating the pathological 1000-way burst. That's the expected result and it's the
crux: MR 292 fixes the *driver*, not the *backend*.

**Q3 (how far the fix scales):** #6 streaming swept to 131072 shows where the
streaming shape itself plateaus or breaks; #7 shows whether the connection-reset
regime is actually escaped or just pushed out.

**Q4 (reset reproduction + mitigation):** #7 (baseline vs jitter vs force-close)
± #8 (streaming under refit). Baseline at ≥8192 in-flight should reproduce; the
A/B isolates the fix.

**Q5 (real model latencies):** see next section.

## Real-model-latency: why and how the experiments change

The synthetic model uses `asyncio.sleep` (near-zero / fixed). Real vLLM differs in
ways that change the conclusions:

- **Per-rollout latency dominates.** Real forwards are 100s ms–seconds; #4 already
  shows ≤256 ms of per-call work is absorbed for free at the 64k body, so the
  framework-overhead share shrinks and Gym's loop is a smaller fraction of
  wall-clock under real latency.
- **The model becomes a bottleneck of its own** — `max_num_seqs`, prefill/decode
  contention, engine queueing (the exact issues MR 292 items 4–6 fixed: 1→16
  proxy workers, uncapped `max_num_seqs`, partitioned engines). Our synthetic runs
  cannot see these.
- **Connection hold times get longer and variable**, which changes the
  connection-lifecycle: more connections stay in-flight into the refit drain,
  which can make the reset regime worse, not better. The long tail (sequential
  judge calls, MR 292 item 7) also lives here.

Plan to cover it without a full GPU matrix:
- **#9 realistic_latency**: drive the synthetic model with `latency_dist=pareto`
  (the knob already exists) tuned to a measured real p50/p99, and re-run #1 and #7.
  Tells us if the ceilings and the reset regime shift under realistic timing.
- **#10 real_model_calibration**: one GPU run of `example_single_tool_call` + a
  small real vLLM at a fixed concurrency; lock the synthetic knobs (latency dist,
  `cpu_burn_ms`, body shape) until the synthetic timeseries match within ~20%.
  - If they match: the synthetic scale matrix transfers; report with that caveat.
  - If they diverge: the scale matrix (especially #1, #6, #7) must be repeated on
    real models — flag this as a GPU-cost dependency for the architecture decision.

## Execution

- Run on both hosts at branch HEAD: `python run_experiments.py --all --label <workstation|slurm-cpu>`
  (or `--experiment <name>` per cell). Cluster: `tools/scale_sim/run_on_slurm.sh`.
- Heavy cells (131072 in-flight, 8M body, 64k-in-flight burst) take hours; budget
  accordingly and keep `RAY_TMPDIR=/tmp`, FD soft limit raised to hard.
- Commit `findings/<label>/` after each host completes so workstation and
  slurm-cpu sit side by side for comparison; raw artifacts under `results/` are
  gitignored except where explicitly committed for analysis.
- The 24-core vs 96-core comparison is itself a result (per-rollout single-core
  speed vs aggregate-core throughput) — keep both columns in the report.
