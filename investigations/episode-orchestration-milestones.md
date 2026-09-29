# Episode orchestration milestones

This file tracks delivery status and remaining integration gates for [`episode-orchestration-design.md`](episode-orchestration-design.md). Status reflects the Environment Server PR stack ending at `ananthsub/episode-orchestration-prototype` as inspected on 2026-09-28.

Status meanings:

- **Implemented:** the prototype contains the end-to-end contract and implementation.
- **Partial:** a meaningful slice exists, but the milestone still has a named gate.
- **Deferred:** the design is retained for future work and is not part of the shipped foundation.

## Foundation milestones

### 0. Record current behavior — Partial

Focused tests cover Environment Server routing, native request and result projection, failure sidecars, resources and agent sessions, cleanup, direct access models, aggregation, Simple Agent, Hermes, and SWE-bench Pro.

Remaining gates:

- preserve behavior for additional legacy agents before migrating them;
- keep explicit regression coverage for token capture, model-call capture, masking, resume attempts, and aggregate labels across native and compatibility paths;
- distinguish historical behavior from known ownership and cleanup defects rather than freezing those defects.

### 1. Add the Environment Server foundation — Implemented

Implemented:

- `environment_servers` config and references;
- discovery, startup, addressing, readiness, status, telemetry, and test inclusion;
- mandatory Agent Server frontends;
- `EpisodeId`, `TaskId`, `MaterializedTask`, and generic episode request and response envelopes;
- `BaseEnvironmentServer.run_request()`;
- `CleanupContext` and idempotent `CleanupHandle`;
- optional admission, queue timeout, episode deadline, separate cleanup timeout, cancellation shielding, and response identity checks;
- native Environment Server result and failure projection;
- Environment Server aggregation routing.

Current limit:

- admission and cleanup state are worker-local;
- cleanup is bounded best effort, not durable recovery.

### 2. Add taskset routing and compatibility frontends — Implemented

Implemented:

- native materialized tasks route through `environment_server_routes` by `TaskId.taskset`;
- flat rows can route by their agent's mandatory frontend or an explicit compatibility Environment Server;
- selected deployments are stamped as `_ng_environment_server`;
- stored results are stamped with `_ng_result_type`;
- `LegacyAgentEnvironmentServer` relays an unmigrated `/run`;
- `SingleAgentTurnLegacyEnvironmentServer` converts flat rows to the typed protocol and projects results back;
- report labels use unique Environment Server run keys when agent names are ambiguous.

Remaining gate:

- remove direct-agent dispatch code only after every supported path uses a frontend and downstream consumers no longer depend on it.

### 3. Extract the Simple Agent Server — Implemented for the single-agent protocol

Implemented:

- caller-assigned agent sessions;
- episode-scoped `ToolAccess`;
- configured and request-granted tool overlay by name;
- `/v1/responses` remains the behavior endpoint;
- agent close returns observations and final resources cookies;
- `SingleAgentTurnEnvironmentServer` owns seed → session → activation → close → verify.

Remaining gates:

- complete parity checks for skipped verification, max-step behavior, tool failures, reasoning content, usage, and capture;
- continue to keep the legacy `/run` path only as compatibility while deployments migrate.

### 4. Establish direct sandbox handoff with Hermes and SWE-bench Pro — Partial

Implemented:

- SWE-bench Pro creates and owns the task sandbox;
- resources seed returns direct `SandboxAccess`;
- Hermes reconnects as a borrower or creates its own fallback sandbox;
- Hermes runs its harness in the sandbox and calls the attempt-qualified Gym Model Server endpoint directly;
- agent close stops the runner, returns observations, and disconnects borrowed access;
- verification begins only after agent close;
- SWE-bench Pro extracts the patch, uses fresh verifier sandboxes, and closes resources state idempotently.

Remaining gates:

- run and inspect a real standalone rollout with several model/tool iterations;
- confirm no task, agent-owned, or verification sandbox remains;
- record the exact rollout artifact, failure-sidecar outcome, observations, and benchmark fields;
- test provider and worker failure paths that cannot be proven by unit tests.

### 5. Extract OpenCode and deduplicate local and sandboxed variants — Deferred

No OpenCode agent-session implementation exists in the inspected prototype.

Required gates:

- move OpenCode behavior behind agent seed, `/v1/responses`, and close;
- preserve installation or discovery, CLI configuration, transcript conversion, diagnostics, output limits, and capture;
- support both agent-owned fallback sandboxes and resources-owned borrowed access;
- validate SWE-bench Pro and DeepSWE first;
- prove semantic parity under both placements;
- retain old deployment names as compatibility aliases before deleting duplicate implementations.

### 6. Migrate remaining agent classes — Partial

Hermes and Simple Agent demonstrate the session boundary. Unmigrated agents remain reachable through `LegacyAgentEnvironmentServer`.

Remaining gates:

- inventory each agent as Responses-capable, run-only, step protocol, or self-contained integration;
- migrate only where the Agent Server and Task/Resources Server boundaries fit;
- add concrete Environment Servers for protocols with different ordering;
- keep self-contained integrations in concrete Environment Servers rather than forcing artificial participant APIs;
- move benchmark-specific harvesting and verification into the component that owns that state.

### 7. Add native NeMo RL consumption — Partial

Implemented in Gym:

- attempt-qualified `EpisodeId`;
- Environment Server routing and stamps;
- native result and failure projection;
- sidecar retry and terminal markers;
- token and model-call capture correlation;
- Environment Server-aware aggregation and report labels.

Remaining gates:

- route NeMo RL directly by Environment Server deployment where appropriate;
- preserve trainable participant attribution and chronological projection;
- consume terminal and retryable failures without turning infrastructure loss into reward zero;
- preserve response token IDs, log probabilities, masks, reward components, multimodal fields, batch cardinality, and row order;
- validate sharded replica affinity for process-local agent and resources sessions.

## Lifecycle and persistence milestones

### 8. Make cleanup restart-safe — Deferred

`CleanupContext` handles success, handled failure, timeout, and caller cancellation while its Environment Server worker remains alive. It does not survive process or host loss.

Required gates:

- define external-object TTL requirements;
- add owner shutdown cleanup;
- add a durable lease or reaper for resources that must outlive a worker;
- reconcile cleanup after a lost seed or close response;
- prove stale attempts cannot destroy a newer attempt's resources.

### 9. Add durable attempt ownership and terminal publication — Deferred

Current resume uses successful rollout JSONL, failure sidecars, and attempt counters. This is useful persistence but not a durable episode ledger.

Required gates:

- atomic attempt claims and ownership epochs;
- stale-writer fencing at every mutable boundary;
- explicit per-operation replay policy;
- immutable terminal result written before completion acknowledgement;
- recovery that reconciles terminal results before redispatch;
- fault injection between every lifecycle phase.

### 10. Add checkpoint restoration — Deferred

No coordinated episode checkpoint exists.

Required gates:

- quiesce model, tool, agent, resources, and runtime activity;
- stage component snapshots;
- publish one immutable manifest last;
- resume into a new fenced attempt;
- declare fidelity as replay, reconnect, or exact snapshot;
- reject restoration when the weakest required component cannot satisfy the requested fidelity.

## Runtime and security milestones

### 11. Establish direct access contracts — Implemented with trust limits

Implemented:

- named direct-HTTP and MCP tool access;
- direct sandbox reconnection through a named top-level provider configuration;
- explicit owner/borrower cleanup order;
- no infrastructure credentials in the Responses request body.

Current limit:

- direct provider descriptors and credentials do not enforce borrower restrictions.

### 12. Add enforceable runtime grants — Deferred

Required gates:

- authority-held provider credentials;
- owner and borrower grants scoped to rollout, attempt, subject, operations, workdir, and expiry;
- durable grant and revocation state;
- provider admission and attestation;
- scoped model and tool routes;
- crash-surviving cleanup and reconciliation;
- one live-sharing benchmark migrated through the authority before broader adoption.

### 13. Add a sandbox server where required — Deferred

Add a sandbox server only for a provider that cannot reconnect across processes or a deployment that requires server-enforced authorization.

Required gates:

- define `SandboxServerRef` and connection model;
- define owner and borrower operations;
- enforce lease expiry and revocation;
- preserve provider-specific state without exposing owner credentials;
- prove cleanup under server and client failure.

## Agent architecture milestones

### 14. Add a minimal in-process Agent API — Deferred

The shipped boundary remains Agent Server sessions plus `/v1/responses`.

Required gates:

- define the concrete consumer that benefits from in-process invocation;
- preserve identity, tool and sandbox access, observations, cancellation, and close semantics;
- provide an HTTP and in-process client with equivalent behavior;
- migrate one agent without changing its rollout result.

### 15. Add user simulation and multi-agent scheduling — Deferred

Fan-out remains several independent episodes.

Required gates:

- define one concrete protocol's participant roles;
- define visibility and private state;
- define ordering, concurrency, termination, and join behavior;
- distinguish participant identity from policy identity;
- preserve simulated-user tokens as context rather than trainable actions;
- define idempotent steps and checkpoint barriers before restart support.

## Integration gates

| Gate | Status | Evidence still required |
| --- | --- | --- |
| Gym resolves, starts, and reports Environment Servers | Implemented | Keep configuration and health tests |
| Every dispatchable agent has an Environment Server frontend | Implemented | Remove bypass only after downstream migration |
| Native tasksets route by `TaskId.taskset` | Implemented | Add broader mixed-batch coverage |
| Simple Agent runs through `SingleAgentTurnEnvironmentServer` | Implemented | Maintain result and capture parity |
| Hermes and SWE-bench Pro share one owner-managed sandbox | Partial | Real rollout and leak inspection |
| OpenCode uses the agent-session boundary | Deferred | Implementation and parity evidence |
| OpenCode local and sandboxed code is deduplicated | Deferred | Both placement modes through one implementation |
| Native NeMo RL consumes Environment Server results | Partial | End-to-end training adapter validation |
| Timeout and cancellation leave no owned sandbox running | Partial | Real provider and failure-injection evidence |
| Attempts are restart-safe | Deferred | Durable claims, fences, and result spool |
| Borrower grants are enforceable | Deferred | Authority service or provider-native restricted credentials |
| Checkpoint restore is coordinated | Deferred | Immutable manifest and fenced restore |
| Multi-agent scheduling is concrete and attributable | Deferred | One fully specified protocol |

## Required checks for the shipped foundation

- malformed native and compatibility requests fail before task side effects;
- missing or ambiguous Environment Server routes fail before dispatch;
- queue timeout creates no resources session;
- resources and agent cleanup are registered before seed;
- seed failure never invokes the agent;
- a required unsupported tool or sandbox access fails agent-session seed;
- an agent close failure prevents verification;
- resources verify sees the final direct-tool cookie jar;
- agent-owned sandboxes stop on success, failure, timeout, and cancellation;
- borrowed sandbox clients disconnect without destroying resources-owned state;
- resources close is idempotent and validates episode identity;
- cleanup runs outside the episode deadline and remains bounded by its own timeout;
- native responses preserve Environment Server-specific fields;
- native handled failures go to the sidecar with stage and terminality;
- failure rows do not enter scoring unless explicitly counted as metrics-only zeros;
- result and report labels identify the Environment Server when agent names are ambiguous;
- capture paths include the attempt suffix on retries;
- one real Hermes and SWE-bench Pro rollout leaves no sandbox running.
