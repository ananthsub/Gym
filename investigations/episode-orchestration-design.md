# Episode orchestration: shipped Environment Servers and future runtime authority

Status: implementation-aligned design, 2026-09-28. The shipped behavior described here comes from the Environment Server PR stack ending at `ananthsub/episode-orchestration-prototype`. The longer-term sections preserve the useful architecture from draft 3, but label work that does not exist in that stack.

## Executive summary

Gym now routes rollout collection through an Environment Server. This is not an optional remote processor layered beside agent `/run`. Every configured Agent Server that rollout collection can dispatch to must be fronted by an `environment_servers` deployment. Native materialized tasks can select an Environment Server directly by taskset. Compatibility deployments use an Environment Server relay or adapter so old flat rows and old agent-owned `/run` methods continue to work.

The shipped `BaseEnvironmentServer` owns the framework lifecycle around one `/run`: request validation, optional admission, an episode deadline, correlation, handled-failure conversion, response identity checks, and bounded cleanup. A concrete Environment Server owns protocol order. The shipped `SingleAgentTurnEnvironmentServer` performs one Responses API activation, which may itself contain many model and tool turns.

The resources-backed single-agent protocol is:

1. assign a resources session identifier and register its cleanup;
2. seed the Task/Resources Server;
3. resolve direct-HTTP or MCP tool access and optional sandbox access;
4. assign an agent session identifier and register its cleanup;
5. seed the Agent Server session;
6. invoke its attempt-qualified `/v1/responses` route;
7. close the Agent Server session and collect observations and final resources cookies;
8. verify through the Task/Resources Server;
9. return the verifier fields plus observations;
10. close remaining resources state after `run()` finishes, outside the episode deadline.

This implementation establishes protocol ownership and cleanup order. It does not yet provide a durable episode ledger, restart-safe leases, checkpoint restoration, a minimal in-process `Agent` API, OpenCode deduplication, enforceable sandbox grants, or generic multi-agent scheduling.

## Why orchestration moved out of agent `/run`

Draft 3 started from four recurring problems.

- Agent Servers duplicated seed → act → verify logic.
- CLI integrations duplicated process, installation, and sandbox code.
- Agent and resources code could both believe they owned the same sandbox.
- A serialized provider descriptor made a sandbox reachable but did not define who could operate or destroy it.

Those problems remain the historical motivation, but the first shipped step is narrower than draft 3's proposed `EpisodeRunner` and runtime-authority stack. Gym introduced an Environment Server as the `/run` owner and kept behavior behind Agent Server APIs. It also introduced explicit resources and agent session boundaries so each participating server cleans up what it creates.

The new boundary fixes an architectural ownership problem before it solves every durability or authorization problem. `CleanupContext` is process-local. `SandboxAccess` can reconnect through a named provider configuration, but direct access is not an enforceable security capability. These limits are part of the current contract, not hidden implementation details.

## Shipped component boundaries

| Component | Shipped responsibility | Important limit |
| --- | --- | --- |
| Rollout collector | Resolve or stamp an Environment Server, construct native episode requests, persist results, route failures, and aggregate by the server that ran the rollout | It does not own the concrete episode protocol |
| Environment Server | Expose `/run`, order participating calls, apply episode limits, assemble a typed result, and request participant cleanup | Cleanup registrations disappear if its process dies |
| Task/Resources Server | Own task setup, stateful tools, verification, resources sessions, and task sandboxes it creates | Existing implementations may still rely on worker-local state and cookies |
| Agent Server | Own agent behavior, agent sessions, model/tool loops, subprocesses, borrowed connections, and fallback sandboxes it creates | Session state is process-local and requires one-worker affinity |
| Model Server | Serve inference and capture requests under attempt-qualified rollout paths | It is not an episode owner |
| Sandbox provider | Allocate or reconnect execution state | A direct descriptor is reachability, not authorization |

An Environment Server may also be self-contained. An integration such as Tau2 can run its complete protocol and verification inside a concrete Environment Server without forcing its state into Agent Server or Task/Resources Server contracts.

## Environment Server routing is mandatory

The shipped configuration parser rejects a dispatchable Agent Server with no Environment Server. The error points users to `scripts/add_legacy_agent_environment_servers.py`, which adds compatibility frontends. Rollout collection therefore sends `/run` to an Environment Server even when the underlying agent still owns the old episode implementation.

There are three routing forms.

### Agent-derived compatibility routing

Flat rows can retain `agent_ref`. Rollout collection maps that agent to the one Environment Server whose `agent_server` reference names it. No matching server is a configuration error. More than one matching server is ambiguous unless the row is routed directly to a server.

### Explicit compatibility routing

`environment_routing_mode=legacy` sends flat rows to `environment_server_name`. The collector validates that a row's `agent_ref` matches the Environment Server's configured Agent Server and that `task_source` matches its configured Task/Resources Server.

### Native taskset routing

A native row contains `task_id.taskset` and `task_input`. It always uses `environment_server_routes[taskset]`. `environment_routing_mode=taskset` rejects flat rows. Agent and legacy modes may mix native materialized tasks with compatibility rows.

Before writing materialized inputs, rollout collection stamps the selected deployment as `_ng_environment_server`. Compatibility rows also receive the Agent Server's `agent_ref` when needed. The Environment Server stamp survives retries and is copied to stored results. Each result also receives `_ng_result_type`, the configured Environment Server implementation name.

The native request does not contain routing configuration. The collector converts a materialized row to:

```yaml
episode_id:
  rollout_id: stable-logical-rollout
  attempt: 0
task:
  task_id:
    taskset: swebench_pro
    task_id: instance-123
  task_input:
    responses_create_params: {}
    task_data: {}
```

`EpisodeId.capture_key` is the logical rollout ID for attempt zero and appends `-a<N>` for later attempts. A logical rollout ID may not already end in that reserved suffix.

## The shipped base lifecycle

`BaseEnvironmentServer[EpisodeRequestT, EpisodeResponseT]` registers typed `/run` and `/aggregate_metrics` routes. Concrete servers bind `request_model`, `response_model`, implement `run(request, cleanup)`, and implement aggregation.

`run_request()` executes in this order:

1. If configured, wait for the worker-local admission semaphore.
2. Return a non-terminal failure if queue admission exceeds `queue_timeout_seconds`.
3. Create `CleanupContext` with the request's `EpisodeId`.
4. Enter rollout correlation using the attempt-qualified capture key.
5. Run the concrete protocol under `default_episode_timeout_seconds`.
6. Convert an expired episode deadline to a non-terminal failure.
7. Convert `HandledEpisodeError` to the concrete response's failure field.
8. Convert any other exception to a terminal failure with bounded diagnostic text.
9. Shield cleanup from cancellation and run it outside the episode deadline.
10. Release the admission slot.
11. Re-raise caller cancellation after cleanup.
12. Validate the concrete response and require matching episode and task identity.

The base response contains exactly one of `result` and `failure`. A handled failure records bounded `message` text and whether another attempt is allowed. The base deliberately does not claim that a retry is idempotent.

### `CleanupContext` is the current lifecycle authority

`CleanupContext` stores process-local async callbacks in registration order and unwinds them in reverse order. A `CleanupHandle` lets the protocol close one participant before final unwind. An entry is marked inactive only after its callback succeeds, so a later final unwind can retry a failed early close.

All remaining callbacks share one `cleanup_timeout_seconds` budget. Callback failures are logged and later callbacks continue. A callback that hangs until the shared timeout can prevent earlier registrations from running. Close operations must therefore be idempotent and independently bounded where practical.

Cleanup runs after the episode deadline has ended. This distinction is required for safe teardown: an agent or verifier can consume the full episode budget without eliminating the separate cleanup budget.

The guarantee ends at process lifetime. A worker or host failure loses the registry and its retries. External owners still need provider TTLs, shutdown handling, or a future reaper.

## `SingleAgentTurnEnvironmentServer`

The shipped name is `SingleAgentTurnEnvironmentServer`. “Turn” names one Environment Server protocol activation. It does not mean one model completion: the Agent Server's one `/v1/responses` call may contain many model and tool iterations.

Its configuration binds:

- one `ResourcesServerRef`;
- one `AgentServerRef`;
- zero or more resources tool transports: `direct_http` and `mcp`;
- optional worker-local admission;
- queue, episode, and cleanup timeouts.

### Resources session

The Environment Server assigns `resources-session-<uuid>` before contacting resources. It registers `POST /close_session` before seed so a lost seed response cannot hide the identifier that may have created remote state.

`ResourcesSeedSessionRequest` carries:

- the caller-assigned `resources_session_id`;
- immutable `EpisodeId`;
- immutable `TaskId`;
- protocol-specific `task_data`.

The Task/Resources Server must return the same identifier. The Environment Server also requires seed to establish a resources cookie. That cookie jar remains private lifecycle state held by the Environment Server.

### Direct tool access

If `direct_http` is configured, the Environment Server constructs `DirectHTTPToolAccess` with the resources base URL and current session cookies. This is the shipped trusted-Python path. It gives the Agent Server direct access to typed resources routes but does not give it resources seed, verify, or close authority.

### MCP tool access

If `mcp` is configured, resources seed must return HTTP MCP metadata. The Environment Server resolves it into `MCPToolAccess` with one absolute streamable-HTTP URL and the returned headers. Missing metadata or another transport is a terminal seed-stage failure.

Agent configuration and episode grants use the same named `ToolAccess` union. Episode grants overlay configured declarations by name. Duplicate names within either source are rejected.

### Agent session

The Environment Server assigns `agent-session-<uuid>` and registers its close before seed. `AgentSeedSessionRequest` carries:

- caller-assigned `agent_session_id`;
- `EpisodeId` and `TaskId`;
- resolved `tool_accesses`;
- optional `SandboxAccess` returned by resources.

The Agent Server must return the same identifier. Its session cookie selects the process-local state for the later Responses call and close.

### Agent activation and capture

The Environment Server calls:

```text
/ng-rollout/<capture_key>/v1/responses
```

When run-level token capture is enabled and the bound agent opts in, it calls:

```text
/ng-rollout/<capture_key>/training-token-capture/v1/responses
```

The body remains the task's `responses_create_params`. The attempt-qualified path correlates downstream model calls and captures without adding orchestration fields to the Responses API schema.

### Close before verify

After a valid agent response, the Environment Server closes the agent session before verification. Close must stop agent-controlled activity, disconnect borrowed sandbox clients or stop agent-owned sandboxes, and return any observations. For direct-HTTP tool access, it may also return a final resources cookie jar. The Environment Server adopts that jar before verify and resources close.

If agent close fails, verification does not run. The failure stage is `cleanup`; a valid partial agent response is retained.

### Verify and final result

Verification receives `EpisodeId`, `TaskId`, the original Responses request, and the validated agent response under `SingleAgentTurnResourcesVerifyRequest`. The resources cookie jar authorizes the call.

The successful `SingleAgentTurnResult` is the Task/Resources Server's `BaseVerifyResponse` subclass with extra benchmark fields preserved at the top level. `ng_agent_observations` is added from agent close. This layout intentionally matches historical stored `/run` results rather than adding a nested `verification` object.

Failures may occur at `seed`, `agent`, `cleanup`, or `verification`. They carry terminality and may retain `partial_response`. An infrastructure failure never becomes a synthetic zero reward.

## Compatibility Environment Servers

Two relays keep migration incremental.

### `LegacyAgentEnvironmentServer`

This is an opaque HTTP relay. It forwards the incoming `/run` body, cookies, and end-to-end headers to an unmigrated Agent Server's `/run`, then returns the upstream status, body, repeated headers, and cookies. The agent retains its old orchestration and limits. Environment Server timeout and admission settings are intentionally not applied and produce warnings if configured.

Aggregation is forwarded to the legacy Agent Server.

### `SingleAgentTurnLegacyEnvironmentServer`

This adapter migrates one resources-backed pairing while preserving flat rows. It validates `task_source` and `agent_ref`, derives stable rollout and attempt identity, converts the flat row into `SingleAgentTurnRequest`, and delegates to the base lifecycle.

On success it flattens the native result and restores `agent_ref`. On failure it emits the historical `_ng_failure_*` keys, including stage, terminality, and any partial response. Aggregation remains owned by the Task/Resources Server through `SingleAgentTurnEnvironmentServer`.

These two classes serve different purposes: the opaque relay preserves an unmigrated agent protocol; the typed adapter preserves an old caller while running the migrated protocol.

## Stored results, failures, aggregation, and report labels

### Native result projection

When a native materialized task returns `BaseEpisodeResponse`, rollout collection validates the envelope. A successful `result` is copied without interpreting protocol-specific fields. Keys reserved for collector-owned `_ng_*`, trajectory, capture, and performance fields are rejected. The collector adds:

- `_ng_task_id`;
- `_ng_task_index` and `_ng_rollout_index`;
- `agent_ref`, `task_source`, `skills_ref`, attempt, and explicit rollout ID when present;
- `_ng_environment_server`;
- `_ng_result_type`;
- capture, trajectory, and performance fields assembled by the collector.

### Failure persistence

A handled Environment Server failure becomes a failure-sidecar record with:

- `_ng_failure_class: environment_server_failed`;
- `_ng_failure_terminal`;
- `_ng_failure_message`;
- optional `_ng_failure_stage`;
- optional `_ng_failure_partial_response`;
- task, rollout, attempt, routing, and Environment Server stamps added by the collector.

Non-terminal failures can be retried until the configured attempt limit. Terminal failures are not retried. Successful results go to the main rollout JSONL. Non-kill-shaped failures go to `<output_stem>_failures.jsonl`. Kill-shaped failures are not persisted, so resume discovers them by set difference.

Failures are absent from aggregate scoring unless the run explicitly names their class in `count_failure_classes_as_zero`. That opt-in contributes a metrics-only zero and does not rewrite the failure sidecar into a synthetic successful rollout.

### Aggregation routing and labels

Aggregation groups compatibility results by agent, then calls `/aggregate_metrics` on that agent's Environment Server. `SingleAgentTurnEnvironmentServer` forwards to resources because resources owns verification. `LegacyAgentEnvironmentServer` forwards to the legacy agent.

Native records identify what ran them with `_ng_environment_server`. Reporting uses the Environment Server as the run key and falls back to the historical agent only for older records. If one Environment Server fronts the only run for an agent, labels can retain the agent name. When several servers front the same agent, labels use unique Environment Server run keys. Per-rollout debug and trajectory labels prefer `agent_ref` when present and otherwise use the Environment Server stamp.

## Direct `ToolAccess` and `SandboxAccess`

The shipped design passes agent-visible access directly from Environment Server to Agent Server. The Environment Server transports these values but does not proxy each tool or sandbox operation.

`SandboxAccess` currently contains:

- `DirectSandboxConnection.kind = "direct"`;
- `provider_config_ref`, naming a top-level provider configuration;
- a provider-specific serialized descriptor;
- `workdir`.

The Task/Resources Server creates and owns the task sandbox. The Agent Server resolves the same named provider configuration, reconnects, operates the sandbox during its session, and disconnects on close. It must not call the owner operation that destroys the sandbox.

This is an ownership convention implemented by the participating code. A provider descriptor plus broad provider credentials is not an enforceable grant. Provider-native restricted credentials or a sandbox authority service are future work.

If resources returns no sandbox access, a sandbox-dependent Agent Server may create a fallback sandbox from its own configuration. That agent session owns and stops the fallback. Verification cannot inspect its live contents after close. A benchmark that verifies live task state must therefore have the Task/Resources Server own and lend the task sandbox.

## Actual SWE-bench Pro and Hermes lifecycle

The prototype implements the shared-sandbox path with SWE-bench Pro and Hermes.

### SWE-bench Pro seed

SWE-bench Pro accepts the caller-assigned resources session ID, stores it in the cookie session, and binds it to `EpisodeId` and `TaskId`. Repeating seed with the same identity returns the existing sandbox access. Reusing the ID for another identity fails.

It validates `task_data` as a SWE-bench Pro instance, creates and normalizes `/app`, records pristine untracked files, serializes the sandbox, and stores the owner handle. It returns `SandboxAccess` with the configured provider reference, descriptor, and `/app` workdir.

### Hermes agent session

Hermes rejects required tool grants it does not implement. It requires either borrowed `sandbox_access` or its own configured sandbox provider.

For borrowed access it resolves the named provider and reconnects to the descriptor. For fallback execution it creates a sandbox and marks itself owner. It installs the pinned Hermes runtime into the sandbox, uploads the runner and observer, and creates a session directory keyed by the caller-assigned agent session ID.

One attempt-qualified `/v1/responses` call launches the Hermes runner in a sandbox PTY. Before launch, the Agent Server resolves the attempt-qualified Model Server base URL and gives that URL to the sandbox runner. Hermes then calls the Gym Model Server directly from the sandbox; there is no host-side model relay or model-request polling loop. The Model Server records the correlated calls, while the Agent Server waits for runner completion, converts the Hermes trajectory to a `NeMoGymResponse`, and captures observations.

On agent close, Hermes cancels an active activation, terminates the runner, removes its session directory, and then either disconnects the borrowed sandbox or stops its own sandbox. It returns observations after mutation has stopped.

### SWE-bench Pro verify and close

Only after Hermes close succeeds does SWE-bench Pro verify. It validates episode and task identity, extracts the patch from the original task sandbox, and stops that task sandbox during extraction. It then creates fresh verification sandboxes, retries inconclusive verification within a total budget, and stops every verification sandbox.

Resources close is idempotent. It stops any remaining task sandbox, clears process-local task and identity state, records the closed episode identity, and rejects a later close using a different episode.

This flow proves caller-assigned sessions, owner/borrower cleanup, capture correlation, and benchmark-specific result preservation. It does not prove restart recovery or security isolation between a compromised Agent Server and provider credentials.

## What is implemented, partial, and future

| Capability | Status | Current evidence or remaining limit |
| --- | --- | --- |
| Mandatory Environment Server routing | Implemented | Config validation rejects dispatchable agents without a frontend; collection posts `/run` to Environment Servers |
| Base lifecycle and cleanup | Implemented | Typed validation, admission, deadlines, cancellation shielding, LIFO cleanup, identity checks |
| Resources-backed single-agent protocol | Implemented | `SingleAgentTurnEnvironmentServer` |
| Caller-assigned resources and agent sessions | Implemented | Seed and close contracts validate returned identity |
| Compatibility migration | Implemented | Opaque legacy relay and typed flat-row adapter |
| Native taskset routing and stamps | Implemented | Taskset routes, `_ng_environment_server`, `_ng_result_type` |
| Result/failure persistence | Implemented | Main JSONL, failures sidecar, retry and terminal markers |
| Environment-aware aggregation and report labels | Implemented | Environment Server aggregation path and unique run labels |
| Direct tool and sandbox handoff | Implemented with trust limits | Explicit access contracts; direct provider credentials remain broad |
| SWE-bench Pro plus Hermes | Implemented in prototype | Real borrowed-sandbox session flow and focused tests exist |
| OpenCode migration and deduplication | Deferred | OpenCode has not implemented the new agent-session boundary |
| Runtime authority and enforceable grants | Deferred | No authority service, revocation store, or restricted provider token |
| Durable attempts and result publication | Deferred | Resume uses JSONL and sidecars; no atomic lease or stale-writer fence |
| Checkpoint restoration | Deferred | No coordinated snapshot manifest or restore protocol |
| Minimal in-process `Agent` API | Deferred | Shipped Agent Servers expose Responses and session HTTP APIs |
| Multi-agent scheduling | Deferred | Concrete protocol required before adding scheduler abstractions |

## Future architecture retained from draft 3

The following sections are direction, not descriptions of current behavior.

### Runtime authority

A future runtime authority should decide placement, validate capabilities and isolation, issue scoped routes, mediate provider credentials, and reconcile leases. It may return no-op, local-process, sandbox, or scheduler-backed runtimes. Placement should remain independent from agent behavior.

Cross-process sandbox sharing needs enforceable owner and borrower grants. A grant should bind issuer, sandbox, rollout, attempt, subject, allowed operations, workdir, issue time, expiry, unique ID, and policy digest. Only an owner grant can destroy or delegate. A borrower can operate and release its lease.

Direct provider descriptors remain useful reconnection data. They do not become authorization merely because a Python wrapper omits `stop()`.

### Durable attempts and result publication

`EpisodeId.attempt` currently separates capture keys and retry rows. Restart-safe execution additionally needs atomic claims, leases, ownership epochs, stale-writer fencing, and idempotent finalization in a process-shared store.

A durable terminal-result spool should be written before completion is acknowledged. Recovery should reconcile that spool before redispatch so a crash after verification cannot rerun the agent or emit the result twice.

Every externally mutating operation needs an explicit replay policy. A transport retry does not establish exactly-once behavior. Unknown outcomes should normally fence the attempt and reconcile cleanup rather than replay arbitrary agent, tool, verifier, or sandbox effects.

### Coordinated checkpoint restoration

Checkpointing is an episode operation, not a JSONL append at a convenient tool boundary. A future coordinator must quiesce new work, stage component snapshots, and publish one immutable manifest last.

That manifest should bind:

- logical rollout, source attempt, ownership epoch, and event sequence;
- agent conversation or implementation state;
- model-call and token-custody receipts;
- tool effects and result digests;
- Task/Resources Server restore state;
- runtime reconnect or snapshot reference and its exactness level;
- configuration, policy, prompt, tokenizer, verifier, and codec digests.

Resume creates a new fenced attempt. Its effective fidelity is the weakest required component: replayable conversation, reconnectable workspace, and exact process snapshot are different guarantees.

### Minimal in-process Agent API

Draft 3 proposed one method:

```python
class Agent(Protocol):
    async def run(self, request: AgentRequest, context: AgentContext) -> AgentResult: ...
```

That API remains a possible internal simplification. It is not shipped. The current stable boundary is Agent Server session APIs plus `/v1/responses`. Any future in-process API must preserve the same identity, access, observation, cancellation, and cleanup semantics before replacing HTTP inside a colocated deployment.

### OpenCode deduplication

The target remains one OpenCode behavior implementation that can use either an agent-owned runtime or resources-owned borrowed access. The prototype has not completed this migration. Removing sandboxed duplicates is gated on agent-session extraction, semantic parity, borrowed and owned cleanup tests, capture parity, and real rollout evidence.

### Multi-agent and user-simulation scheduling

Fan-out creates independent episodes and is not multi-agent scheduling. A future concrete Environment Server may coordinate several participants, but it must first define roles, private and shared observations, ordering or concurrency, termination, verification input, token attribution, failure semantics, and checkpoint barriers.

Participant identity and policy identity must remain separate from Agent Server configuration. Simulated-user tokens must not silently become trainable assistant tokens.

## Decisions established by the shipped design

1. `/run` belongs to an Environment Server.
2. Environment Server routing is mandatory, with compatibility frontends for unmigrated agents.
3. The base server owns protocol-neutral limits and cleanup; the concrete server owns call order and result shape.
4. `SingleAgentTurnEnvironmentServer` closes agent-controlled activity before verification.
5. The Task/Resources Server owns task state, verification, and any task sandbox it creates.
6. The Agent Server owns behavior, its process-local session, and any fallback sandbox it creates.
7. Direct tool and sandbox access does not transfer lifecycle ownership.
8. Cleanup has a separate deadline and runs after the episode deadline.
9. Failures are data in the sidecar, not successful zero-reward rollouts.
10. Native results and reporting identify the Environment Server that ran the episode.

## Remaining gates

The implementation roadmap and specific integration gates are maintained in [`episode-orchestration-milestones.md`](episode-orchestration-milestones.md).

The highest-priority remaining gates are:

- run and inspect a real SWE-bench Pro plus Hermes rollout, including zero leaked task and verification sandboxes;
- finish native NeMo RL consumption of Environment Server results and retryable failures;
- migrate OpenCode through agent sessions before attempting code deduplication;
- define restart-safe attempt ownership before claiming durable retries;
- define enforceable sandbox grants before treating direct handoff as a security boundary;
- define checkpoint and multi-agent protocols only from concrete consumer requirements.

## Historical design retained

Draft 3's main enduring insight was the separation of behavior, protocol, and placement. The shipped architecture realizes the protocol split with Environment Servers and explicit participant sessions. It does not yet realize the proposed runtime authority or minimal Agent object.

The prior design's useful sandbox topologies still apply:

| Topology | Current owner rule |
| --- | --- |
| Response-only verification | Agent may own a fallback sandbox and destroy it at close |
| Verification of live task state | Task/Resources Server owns and lends the task sandbox |
| Fresh grading sandbox | Task/Resources Server extracts artifacts, creates a verifier sandbox, and stops it |
| Verifier-only sandbox | Task/Resources Server owns the sandbox; agent never receives it |
| Process-bound shared sandbox | Requires a future sandbox authority or colocated owner/operator |

The historical `EpisodeRunner`, durable event log, checkpoint manifest, scoped runtime routes, OpenCode unification, and participant scheduler remain design inputs. They are no longer presented as prerequisites that already exist or as alternate routing paths beside the shipped Environment Server.
