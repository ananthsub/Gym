# nemo-gym scale test — summary

Companion to `nemo-gym-scale-testing.md` (which has the full reasoning, predictions, and per-cell numbers).

## What we did

Built a synthetic-but-real stress harness for nemo-gym. Replaces the model and tool-verifier sub-servers with knob-controllable stand-ins, drives load through the production aiohttp / `RolloutCollectionHelper` stack, and measures where things fall over. Found the predicted load-side bottleneck and three real bugs in `nemo_gym/cli.py` that nobody knew about.

## Branch

`ananthsub/scale-benchmark`. Contains:
- `tools/scale_sim/` — the harness.
- Fix for the spawn-time `Argument list too long` bug (see findings below).
- Workarounds for two more cli.py bugs that surfaced during testing.
- This summary + the comprehensive reference + two design docs under `investigations/`.

## Questions we wanted to answer

| Question | Why it matters |
|---|---|
| At how many concurrent rollouts does one head node fall over? | Today's largest run is 8K concurrent. The next model needs 16K–64K. |
| Does adding more sub-servers (resources servers, agents) hurt the head node beyond their direct load? | Multimodal post-training will need 50–100+ specialized sub-servers vs ~20 today. |
| What happens at multi-MB response bodies? | Image/video tool I/O will routinely produce 10–50 MB bodies vs today's text-mostly few-MB. |
| What happens when each rollout chains many tool-call hops? | Long-horizon reasoning will routinely produce 20+ hops vs today's 1–10. |
| Does the open RL-side rate-limit fix actually help? | It's the one outstanding fix from the August incident; we need to know if it's enough. |

## What we varied

Two stress tests, run separately on a CPU-only single node.

### Single-agent stress

One agent server, one resources server, one model. Vary one knob at a time:

| What we varied | Values |
|---|---|
| Concurrent rollouts in flight | 1024, 4096, 8192, 16384 |
| Response body size (output tokens) | 16K, 128K, 256K, 512K, 1M |
| Tool-call hops per rollout | 1, 4, 16, 64 |
| RL-side rate-limit semaphore | on / off |

### Multi-agent stress

N agent + resources servers + 1 shared model on the head node. Three sub-tests:

| Sub-test | Description |
|---|---|
| Just bring it up, no traffic | How much does N sub-servers existing cost? Memory, file descriptors, kernel state. |
| Fixed total load, vary N | At constant 4096 concurrent rollouts total, does sharding across more agents help or hurt? |
| Fixed per-agent load, vary N | Each agent serves 256 concurrent rollouts, total grows with N (256 → 65536). The headline test for "is the bottleneck the consumer or the agents?" |

N values tested: 1, 4, 16, 32, 64, 128, 256.

## What we found

### Load-side findings

**1. Sharding load across more agents at fixed total volume helps a lot.**
At 4096 concurrent rollouts total:
- 1 agent doing all 4096 → median rollout takes 34 seconds.
- 4 agents doing 1024 each → median 7 seconds (5× better).
- 16 agents doing 256 each → median 2 seconds (16× better).

Each agent has its own asyncio loop on the server side. More agents = more parallel server-side processing. Direct production implication: a single agent at high concurrency leaves substantial throughput on the table.

**2. The single asyncio loop on the *client* side is the system's hard ceiling.**
With each agent capped at 256 concurrent and total rollouts growing:
- 256 total concurrent: p99 latency 2.6 s.
- 4096 total: p99 = 119 s.
- 16384 total: p99 = 166 s.
Median rollout latency stays flat (~3 s past N=16) because the server side has plenty of parallelism. The tail explodes because **a single Python event loop is reading every response body**. No connection failures up to 16K concurrent — just queueing.

**3. Many tool-call hops makes the cliff much worse.**
At 8192 concurrent rollouts with 16 hops per rollout: only 139 of 20000 rollouts completed in 10 minutes. Every hop is one more body to drain through the same single event loop, so the queueing is multiplicative.

### Three real bugs in `nemo_gym/cli.py` (not load-related)

These all gate scaling at high N regardless of how much load you drive. None were predicted; all surfaced because the harness ran cells back-to-back at high N. Without the fixes/workarounds, the practical ceiling was ~100 sub-servers.

| Bug | What it does | Status |
|---|---|---|
| Spawn command embeds the entire global config inline | Hits Linux's 2 MB command-line limit deterministically at N=256. | **Fixed** — config now written to a tempfile, path passed instead. |
| Ray cluster teardown leaves orphaned processes and `/tmp/ray/` state | Subsequent runs join the dead cluster instead of starting fresh. Failed 11 of 21 cells in our first multi-agent run. | **Worked around** in the harness (kill-and-clean before each cell). Upstream fix needed. |
| Sub-server `ray.init()` storm at high N + auto-assigned GCS port lands inside the ephemeral port range | Hundreds of sub-servers all calling Ray simultaneously on a port the kernel can also hand out for outgoing connections. Some `EADDRINUSE`, some timeout. | **Worked around** — Ray pre-started with explicit ports below the ephemeral floor, plus burst-tolerance tuning. Upstream fix needed. |

## Status

| | Done | Pending |
|---|---|---|
| Single-agent stress | Concurrency, body-size, hops sub-tests (partially) | RL-side semaphore on/off comparison |
| Multi-agent stress | N=1, 4, 16, 32, 64 with various load shapes | N=128 and N=256 cells (rerun in flight with all three bug workarounds) |

The remaining open question once the rerun lands: does the consumer-side cliff at 16384 total concurrent depend on whether you reach 16384 via one agent or 64 — i.e., is the bottleneck consumer-side and agent-count-independent, as the design predicts? Same total concurrency, different shapes. Strong directional evidence already; pending the apples-to-apples cell.

## Pointers

- Full reference (numbers, predictions, every cell): `investigations/nemo-gym-scale-testing.md`
- Original design doc: `investigations/nemo-gym-scale-simulation-design.md`
- Multi-agent design extension: `investigations/nemo-gym-multi-agent-design.md`
- The harness: `tools/scale_sim/`
