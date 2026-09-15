# Episode orchestration milestones

Status: proposed delivery plan, 2026-09-11.

This file defines implementation order and integration gates for [`episode-orchestration-design.md`](episode-orchestration-design.md). The architecture document defines the contracts. This file can change as implementation evidence changes without changing those contracts.

## Foundation milestones

### 0. Record current behavior

Add characterization tests before moving ownership. Cover `RolloutCollectionHelper`, current `/run` request and response bodies, the simple-agent HTTP self-call, cookie updates, resources calls, result projection, token capture, NeMo RL consumption, and aggregate metrics.

Record successful and failed paths for:

- `simple_agent` with `example_single_tool_call`;
- OpenCode with SWE-bench;
- OpenCode with DeepSWE;
- OpenCode with SWE-bench Pro;
- OpenCode with Terminal Bench 2.1.

The tests must distinguish existing behavior from known defects. Missing-sandbox fallback, double stop, leaked sandboxes, lost workdirs, incomplete SWE-bench patch collection, and Terminal Bench's incorrect stateless reverification declaration are not compatibility requirements.

### 1. Add server types and the processor foundation

Implement:

- `episode_processors`, `EpisodeProcessorRef`, and their type and instance config models;
- config discovery, reference validation, host and port assignment, startup, readiness, status, telemetry, dataset loading, test discovery, and manifest handling for episode processors;
- migration of dataset ownership from each old agent deployment to its processor deployment;
- legacy `agent_ref` resolution to migrated processor deployment names;
- `EpisodeId`, `TaskId`, `BaseEpisodeRequest`, `BaseEpisodeResponse`, and concrete single-agent contracts;
- agent-session create and close request and response models;
- processor-neutral resources-session seed and close models, direct-HTTP/MCP tool access, and typed verification inputs;
- `BaseEpisodeProcessor` and `EpisodeContext`;
- `SingleAgentEpisodeProcessor`;
- resources-session seed, verification, and close APIs;
- validation, optional admission, episode and cleanup timeouts, cancellation-resistant LIFO cleanup, compatibility translation, and HTTP projection.

This milestone is complete when Gym can spawn and address episode processors, the processor validates before side effects, all registered cleanup runs on every exit path, and the processor reproduces the recorded legacy results.

### 2. Extract the Simple Agent server

Keep the existing model-and-tool loop behind `/v1/responses`. Add request-scoped agent-session create and close endpoints, scoped resources access, model-cookie isolation, trajectory capture, usage accumulation, max-step behavior, and skipped-verification compatibility behavior. Keep the `/v1/responses` request and response models unchanged.

This milestone is complete when `simple_agent` with `example_single_tool_call` runs through `SingleAgentEpisodeProcessor` without changing its caller-visible result.

### 3. Establish direct sandbox handoff with Hermes and SWE-bench Pro

Keep Hermes behavior behind `/v1/responses`. Add agent-session setup and cleanup, install its sandbox-provider dependency in the agent-server environment, and route terminal commands to a borrowed sandbox by immutable agent session ID.

SWE-bench Pro creates and seeds the task sandbox, returns direct `SandboxAccess`, verifies while the sandbox remains alive, and stops it during resources-session cleanup. Hermes reconnects and disconnects without owner authority. The complete flow must retain the agent response, model/tool observations, benchmark-specific verifier fields, and attempt-qualified capture path.

This milestone is complete when a real standalone rollout performs several model/tool iterations, verifies, produces no failure-sidecar row, and leaves no task or verification sandbox running.

### 4. Extract OpenCode and generalize direct sandbox access

Move OpenCode behavior behind its agent server. Preserve installation or discovery, configuration, CLI execution, transcript export, Responses conversion, diagnostics, and output limits.

Implement both sandbox paths:

- OpenCode creates and owns its configured sandbox when resources returns no access.
- OpenCode connects as a borrower when resources returns `sandbox_access`.

Implement direct reconnection through the same named top-level `sandbox_provider` configuration in the resources and agent-server processes. The resources server serializes the sandbox, the agent server reconnects through that provider, and agent close disconnects without destroying the resources-owned sandbox.

Migrate SWE-bench Pro and DeepSWE first. Add Terminal Bench after its Gym resources server is ready. This milestone is complete when each pairing preserves benchmark-specific verification and leaves no owned sandbox running.

### 5. Migrate remaining agent classes

Migrate Hermes, OpenCode, OpenClaw, Pi, and Codex in that order unless benchmark readiness changes the dependency chain. Inventory agents that use direct `responses()`, remote run-only services, step protocols, or local grading. Add behavior endpoints only where an integration must remain a server. Use separate concrete processors for protocols whose ordering differs from the single-agent flow.

For GDPVal-AA-V2, first move ordinary deliverable harvesting into resources. Add its cached-judging processor branch and separate preparation operation before retiring legacy control modes.

### 6. Add native NeMo RL consumption

Define processor routing, `EpisodeId` creation, terminal model-call attribution, capture finalization, retryable failure transport, masking, and chronological projection for every trainable participant. Retain legacy projection until this path is deployed.

## Follow-on milestones

These capabilities are not part of the initial design.

### Sandbox server

Add `sandbox_servers` and `SandboxServerRef` when a required provider cannot serialize and reconnect across processes or when a deployment requires server-enforced borrower authorization. Define operate leases, revocation, owner operations, routing, and cleanup in that design. Directly reconnectable providers do not depend on this server.

### User simulation

Allow repeated `/v1/responses` activations within agent sessions used by a user-simulation processor. The processor owns canonical event ordering, role visibility, termination, and verification input. Add another agent endpoint only if a concrete protocol cannot express an activation through the Responses API.

The first protocol binds `assistant` and `simulated_user` agent servers. It records their outputs separately so simulated-user tokens are context rather than trainable assistant actions.

### Additional multi-agent protocols

Add a concrete processor only when a use case defines its roles, visibility, ordering or concurrency, termination, verification input, and failure semantics. Do not add a generic participant scheduler without those requirements.

### Restart-safe attempts

Back `EpisodeId` with atomic claims, leases, ownership epochs, stale-writer fencing, and idempotent finalization in a process-shared store. Worker-local admission and dictionaries do not provide these guarantees.

### Checkpoint restoration

Coordinate snapshots of processor position, resources state, serializable agent state, runtime references, model continuation state, and ownership epoch. A reconnectable sandbox alone is not a restorable episode.

### Retained artifacts

Add retained artifacts only for a concrete caller requirement. Define storage ownership, opaque references, authorization, retention, garbage collection, size limits, and deletion before exposing them in episode results.

## Integration gates

1. Gym resolves, spawns, and reports health for `episode_processors`.
2. Episode contracts and compatibility characterization are agreed.
3. `simple_agent` runs through `SingleAgentEpisodeProcessor`.
4. Hermes runs with resources-provided OpenSandbox access against SWE-bench Pro.
5. OpenCode runs with an agent-created sandbox against Reasoning Gym.
6. OpenCode runs with resources-provided sandbox access against SWE-bench Pro and DeepSWE.
7. GDPVal moves ordinary deliverable harvesting into resources.
8. Cancellation and injected failures leave no owned sandbox running.
9. Measured latency and throughput regressions remain within an agreed budget.
10. A provider that cannot reconnect directly runs through a configured sandbox server without exposing owner authority to the agent.

## Required foundation tests

- Malformed native and legacy requests fail before admission.
- Missing or mistyped processor, agent, resources, and model references fail during configuration validation.
- Episode-processor deployments receive distinct addresses and readiness checks.
- Queue timeout creates no resources session.
- Scope-entry failure cleans partially acquired state.
- Seed failure never invokes an agent.
- Resources fails seed when verification requires a shared task sandbox but access cannot be returned.
- An agent rejects incompatible sandbox access during `POST /v1/agent_sessions`.
- Direct sandbox access rejects an inline, missing, or unavailable named provider before inference.
- Closing a resources-provided connection cannot stop the resources-owned sandbox.
- Closing an agent session stops its agent-owned sandbox on success, failure, timeout, and cancellation.
- Simple Agent rejects non-null sandbox access during session creation.
- Cancellation during reconnect, guest startup, agent execution, verification, or cleanup runs every registered cleanup.
- Resources-access unions retain their required discriminator after client serialization.
- A thinking model can continue after a tool call without losing or rejecting assistant reasoning content.
- An agent-side model or tool failure fails the episode instead of producing a valid empty response.
- Agent output limits reject oversized or malformed responses.
- SWE-bench extraction failure is not verified as an empty patch.
- SWE-bench submissions include supported new files.
- Terminal Bench verifies the original live task sandbox.
- Resources cleanup is idempotent.
- Verification does not start unless agent-session close reports that no agent activity can continue mutating resources-owned state.
- Every create, attempt-qualified Responses invocation, and close request for one agent session reaches the same one-worker agent-server replica.
- Legacy projection preserves response, reward, metrics, tokens, completion accounting, and mask location.
- One real Hermes and SWE-bench Pro rollout leaves no sandbox running.

Tests for shared claims, owner leases, user-simulation ordering, role visibility, and checkpoint restoration land with their corresponding follow-on milestones.
