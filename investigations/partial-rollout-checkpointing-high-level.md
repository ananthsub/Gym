# NeMo-RL partial rollout checkpointing

## High-level design: ownership, triggers, and recovery

> Recommended proof of concept: Use TQ as the durable source for completed model-call records and canonical rollouts. Let Single Controller own checkpoint policy and reconciliation, while a background checkpoint task invokes the TQ save/load API. On restart, reuse valid canonical rows, resume only at a completed-call boundary when environment state is recoverable, and redispatch everything else.

## Purpose and scope

The objective is to reduce wasted rollout generation after a generation-worker failure or a full Ray or Slurm restart, without duplicating training consumption or trusting incomplete or corrupt state.

The design builds on the existing TQ × Gym prototype:

- Gym provides call identity and a sealed receipt.
- vLLM writes token-bearing call records into TQ.
- A finalizer publishes canonical trajectories.
- Single Controller coordinates recovery.

### In scope for the first implementation

1. Checkpoint the full live TQ system, including `rollout_staging` and the canonical `rollout_data/train` partition.
2. Handle both a generation-worker failure and a full Ray or Slurm restart.
3. Reuse completed canonical rollout groups.
4. Redispatch missing, corrupt, stale, mid-generation, or otherwise non-resumable work using stable rollout and group identity.

### Initially out of scope

1. Checkpointing every N generated tokens or restoring vLLM KV cache. The first boundary is one completed model call.
2. Transparent restoration of arbitrary in-memory Gym state or black-box sandbox memory.

“Partial rollout” initially means completed calls or completed siblings that have not yet formed, or have not yet been consumed as, a full training group. It does not initially mean an unfinished token stream.

A saved call can be continued only if the next external action is known and the Gym or sandbox state needed to execute it is recoverable. Otherwise the rollout is safely regenerated from its prompt.

## Recovery flow

After restart:

1. Restore Single Controller and TQ.
2. Inventory surviving rollout records.
3. Classify each rollout.
4. Redispatch non-resumable work from a fresh gate.
5. Resume at the next completed-call boundary when the environment state is recoverable.
6. Rehydrate or register the rollout at a new gate.

## Component responsibilities

| Component | Checkpoint and recovery responsibility |
| --- | --- |
| Single Controller | Schedule checkpoints; maintain stable IDs, the checkpoint catalog, dispatch and group ledger, consumption horizon, and recovery decisions. |
| Gym ingress or gate | Mint `call_id`, preserve rollout correlation, seal an immutable call manifest, and expose session or snapshot references when available. |
| vLLM model owner | Write one completed-call record to `rollout_staging` before treating that call as captured. It must not run the full TQ checkpoint. |
| TQ | Persist controller state and storage data for every partition. Restore into a clean initialized system. |
| Forest registry or finalizer | Rebuild lineage, reconcile receipts against staging, verify digests and versions, and publish one canonical rollout row. |
| Policy trainer | Commit training consumption and establish the model-checkpoint horizon after updates become durable. |
| Gym or sandbox | Optionally provide a durable state or session reference. Without it, incomplete multi-turn rollouts are regenerated. |

## Work items

| Work item | Target |
| --- | --- |
| TQ version bump and asynchronous save/load checkpoint integration in Single Controller | July 31 |
| Benchmark TQ checkpoint latency | July 31 |
| TQ × Gym gate integration | Not specified |
| Single Controller checkpoint integration | Not specified |
| Wire the recovery mechanism into Single Controller | August 10 |
| Testing and ablations | August 21 |

## Partial rollout checkpointing variants

### Goal

Minimize rollout work lost after worker, process, node, or job failure while preserving correct training-data consumption and providing explicit recovery guarantees for supported rollout types.

Ideally:

- Completed training-ready data survives.
- Finished siblings within incomplete GRPO groups survive.
- Multi-turn rollouts resume from an unambiguous turn boundary.
- Supported environments restore logical episode state.
- Supported sandboxes restore their durable workspace.
- Unsupported state fails clearly or falls back to redispatch.

### Levels

| Level | Unit of recovery | Recovery behavior | Main state persisted | Value | Checkpointing overhead |
| --- | --- | --- | --- | --- | --- |
| 0. Completed-data durability | Prompt group | Restore completed TQ data; redispatch unfinished work | Fully completed prompt groups | Prevents loss of training-ready rollouts | Minimal; largely implemented |
| 1. Completed-sibling recovery | Completed rollout | Reuse durable completed siblings; restart only unfinished siblings | Completed individual rollouts | Avoids repeating completed GRPO generations | Low |
| 2. Single-turn token-prefix recovery | Limited token chunk | Continue an unfinished generation from its latest token prefix | Completed individual rollouts and token deltas for one turn | Valuable for long single-turn generations | Medium; periodic token captures |
| 3. KV-cache-assisted recovery | Token-level when compatible | Restore KV cache and generation RNG state | KV blocks, RNG state, and engine metadata | Improves recovery latency and reduces prefill work | High; large engine-dependent payloads |
| 4. Replay-based environment recovery | Turn boundary | Restore the partial rollout forest, token and action deltas, and Single Controller metadata; start a fresh environment and replay committed actions | Forest registry, model segments, actions, observations, and controller metadata | Supports multi-turn recovery without requiring environment snapshot APIs | Medium; replay may be costly or nondeterministic |
| 5. Selected Gym environment recovery | Turn boundary for supported environments | Restore logical state for explicitly supported environments | Versioned environment snapshot | Continues expensive multi-turn episodes | High; environment-specific |
| 6. Selected sandbox recovery | Turn boundary | Recreate a sandbox using durable workspace state | Filesystem or workspace plus bootstrap metadata | Particularly valuable for SWE rollouts | High; frequent filesystem snapshots can be expensive |
| 7. Live process-memory recovery | Turn boundary | Restore RAM, processes, file descriptors, and possibly networking | Runtime or OS checkpoint | Strongest continuation fidelity | High |

## Level 0: completed rollouts survive and incomplete rollouts restart

This is the simplest useful guarantee and what the current TQ checkpointing foundation largely provides.

On restart:

- Reload training-ready data from TQ.
- Restore replay-buffer metadata and controller or dataloader position.
- Do not duplicate already committed samples.
- Redispatch incomplete rollouts from their prompt.

Advantages:

- No vLLM modification.
- No Gym state serialization.
- Retains expensive completed rollouts and cached fields such as log probabilities.
- Relatively simple consistency model.

Limitations:

- A generation that ran for 20 minutes and crashed one token before completion must restart.
- Multi-turn episodes restart entirely.
- Redispatched output need not match the lost output.

Determinism:

- Persisted completed data can be restored exactly.
- Training consumption can be idempotent.
- Redispatched work is only statistically reproducible, not necessarily token-identical.

## Level 1: reuse completed GRPO siblings

Suppose a prompt requests four generations and three have completed:

```text
Generation 0: durable — reuse
Generation 1: durable — reuse
Generation 2: durable — reuse
Generation 3: unfinished — redispatch from prompt
```

This requires:

- Stable group and sibling IDs.
- A durable sealed receipt for each completed sibling.
- A recovery ledger identifying missing siblings.
- Idempotent finalization into canonical TQ rows.
- Policy-version compatibility checks.
- Frequent checkpointing of TQ, Single Controller, and replay metadata state.

This is likely the highest-value near-term version because it saves work without attempting to resume inside vLLM or Gym.

## Level 2: single-turn token-prefix recovery

Periodically persist an unfinished assistant prefix:

```text
Prompt + saved assistant tokens 0…2047 + generate remaining suffix
```

Minimum state:

- Prompt and logical rollout identity.
- Generated token IDs.
- Number of committed tokens.
- Model or policy version.
- Tokenizer and chat-template version.
- Maximum response length and remaining budget.

Advantages:

- Token payloads are comparatively small.
- Avoids repeating most decoding for long single-turn responses.
- Is more engine-independent than KV serialization.

Costs and limitations:

- Restoring a long prefix incurs prefill cost.
- Frequent token callbacks or RPCs could affect the generation hot path.
- The saved prefix is exact, but the continuation may differ.
- Log probabilities for the prefix must either be persisted or recomputed under the exact policy.

A reasonable initial checkpoint policy would be every 128–512 tokens, possibly combined with a time threshold.

## Level 3: KV-cache persistence

Persist the KV cache alongside tokens. This can reduce recovery latency from loading tokens and prefilling the entire prefix to loading compatible KV blocks and continuing decoding.

Constraints:

- Payloads are very large.
- The format is generation-backend dependent.
- Moving payloads between GPU and CPU adds overhead.
- Frequent checkpointing is impractical.
- KV cache does not make the suffix deterministic. Exact continuation also requires compatible weights, sampling and RNG state, and sufficiently deterministic kernels and scheduling.

## Level 4: replay-based environment recovery

Reconstruct a multi-turn environment by starting a fresh Gym session and replaying previously completed environment actions using the stored rollout forest and token or action deltas.

Persist:

- The partially completed rollout forest.
- Forest-registry topology and node statuses.
- Assistant token deltas.
- Parsed tool or environment actions.
- Environment results and observations.
- Stable action IDs and execution order.
- Policy version for each model-generated segment.
- Relevant Single Controller metadata.

Recovery:

1. Restore the rollout forest and controller metadata.
2. Start a fresh environment session.
3. Restore the initial task and configuration.
4. Replay committed environment actions in order.
5. Compare replayed observations with stored observations.
6. Stop at the last durable turn boundary.
7. Continue the incomplete rollout from the next required action.

Advantages:

- Avoids regenerating completed model turns.
- Does not require every Gym environment to implement snapshot serialization.
- Can support multi-turn recovery earlier than logical environment snapshots.
- Reuses forest-registry and token-capture history already being persisted.
- Provides a fallback when environment snapshotting is unavailable.

Limitations:

- Replay cost grows with the number and expense of environment actions.
- Replay may produce different environment results.
- External side effects may be unsafe to repeat.
- A long SWE episode may need to rerun package installation, builds, and tests.
- Hidden state cannot be reconstructed unless it is deterministically produced by replayed actions.
- Sandbox filesystem restoration may be required in addition to action replay.

## Level 5: selected Gym environment recovery

Start with white-box environments that can explicitly serialize logical episode state:

```text
snapshot_episode(session_id) -> VersionedEpisodeSnapshot
restore_episode(snapshot) -> new_session_id
```

Persist pure logical data rather than arbitrary live Python objects. Reconstruct clients, locks, and connections.

Advantages:

- Strong semantic recovery for selected high-value workloads.
- Versioned and testable.
- Does not pretend that every environment is serializable.

Costs:

- Each environment needs a state inventory, schema, codec, migration policy, and fault-injection tests.
- External services, clocks, and network effects may still prevent exact replay.

The support contract should be capability-based:

- `NONE`
- `REPLAYABLE`
- `LOGICAL_SNAPSHOT`
- `LOGICAL_SNAPSHOT_WITH_EXTERNAL_STATE`

Multi-turn conversations can resume for supported environments. Other environments must be redispatched.

## Level 6: sandbox filesystem recovery

For SWE environments, a practical target is:

- Persist conversation and agent state.
- Place `/workspace` on durable storage or periodically snapshot it.
- Recreate the sandbox or container.
- Mount or restore the workspace.
- Restart required processes from a bootstrap script.
- Resume at a turn boundary.

This preserves files but not:

- RAM.
- Shell or interpreter state.
- Background processes.
- Open file descriptors.
- Network connections.

That limitation is often acceptable for SWE tasks if tools are designed to restart from filesystem state.

## Level 7: general recovery

“All environments and sandboxes” should mean a common capability framework with provider-specific implementations, not a universal Python-object serializer.

Black-box environments may support only:

- Retain and reconnect.
- Filesystem recovery plus replay.
- Restart from the beginning.

Live process-memory checkpointing requires runtime-specific mechanisms such as CRIU and compatible kernels, container runtimes, mounts, and networking. It is expensive and brittle enough to remain an explicit stretch goal or non-goal.

## Open questions

1. Which environments and sandbox providers are supported for post-training?
2. Is filesystem recovery sufficient for SWE?
3. Which environments should be prioritized for long rollouts?

## Other bottlenecks

- Checkpointing router replay information alongside rollouts can require terabytes for a large model and batch size.
- Can the Mooncake backend be used for TQ checkpointing, and would it help?

The environment-level recovery inventory is maintained separately in [partial-rollout-environment-recovery-tracker.md](partial-rollout-environment-recovery-tracker.md).
