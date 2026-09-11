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

### 1. Add the processor foundation

Implement:

- `EpisodeKey`, `EpisodeRequest`, and `EpisodeResponse`;
- `BaseEpisodeProcessor` and `EpisodeContext`;
- `SingleAgentEpisodeProcessor`;
- `LegacyAgentRunProcessor`;
- resources-session registration and cleanup;
- validation, admission, deadlines, cancellation, finalization, compatibility translation, and HTTP projection.

This milestone is complete when the processor validates before side effects, closes all registered resources on every exit path, and reproduces the recorded legacy results.

### 2. Extract the simple-agent harness server

Keep the existing model-and-tool loop behind `/v1/responses`. Add harness-session open and close, scoped resources access, model-cookie isolation, trajectory capture, usage accumulation, max-step behavior, and skipped-verification behavior.

This milestone is complete when `simple_agent` with `example_single_tool_call` runs through `SingleAgentEpisodeProcessor` without changing its caller-visible result.

### 3. Extract OpenCode and sandbox access

Move OpenCode behavior behind its harness server. Preserve installation or discovery, configuration, CLI execution, transcript export, Responses conversion, diagnostics, and output limits.

Implement both sandbox paths:

- OpenCode creates and owns its configured sandbox when resources returns no access.
- OpenCode connects as a borrower when resources returns `harness_sandbox_access`.

Implement direct reconnection first. Add sandbox-server lease redemption only when a provider cannot support direct cross-process connection.

Migrate SWE-bench first, followed by DeepSWE, SWE-bench Pro, and Terminal Bench 2.1. This milestone is complete when all four pairings preserve their benchmark-specific verification and leave no owned sandbox running.

### 4. Migrate remaining agent classes

Inventory agents that use direct `responses()`, remote run-only services, step protocols, or local grading. Add behavior endpoints only where an integration must remain a service. Use separate concrete processors for protocols whose ordering differs from the single-agent flow.

For GDPVal, first move ordinary deliverable harvesting into resources. Add its cached-judging processor branch and separate preparation operation before retiring legacy control modes.

### 5. Add native NeMo RL consumption

Define processor routing, `EpisodeKey` creation, terminal model-call attribution, capture finalization, retryable failure transport, masking, and chronological projection for every trainable participant. Retain legacy projection until this path is deployed.

## Follow-on milestones

These capabilities are not part of the two foundations.

### Turn execution and user simulation

Add a separate `turn()` endpoint for harnesses that return control after one participant activation. Keep role, deadline, resources access, sandbox access, and capture state in the opened harness session. The turn request carries only the events visible to that role. The processor owns canonical event ordering, role visibility, termination, and verification input.

The first protocol binds `assistant` and `simulated_user` harness servers. It records their outputs separately so simulated-user tokens are context rather than trainable assistant actions.

### Additional multi-agent protocols

Add a concrete processor only when a use case defines its roles, visibility, ordering or concurrency, termination, verification input, and failure semantics. Do not add a generic participant scheduler without those requirements.

### Additional sandbox connectivity

Add the sandbox-server path when resources and a harness cannot directly reconnect to the same provider sandbox. Resources declares that service as a dependency before allocation. The service issues separate owner and operate capabilities. It is not a fallback after direct connection fails.

### Restart-safe attempts

Back `EpisodeKey` with atomic claims, leases, ownership epochs, stale-writer fencing, and idempotent finalization in a process-shared store. Worker-local admission and dictionaries do not provide these guarantees.

### Checkpoint restoration

Coordinate snapshots of processor position, resources state, serializable harness state, runtime references, model continuation state, and ownership epoch. A reconnectable sandbox alone is not a restorable episode.

### Retained artifacts

Add retained artifacts only for a concrete caller requirement. Define storage ownership, opaque references, authorization, retention, garbage collection, size limits, and deletion before exposing them in episode results.

## Integration gates

1. Episode contracts and compatibility characterization are agreed.
2. `simple_agent` runs through `SingleAgentEpisodeProcessor`.
3. OpenCode runs with a harness-created sandbox against Reasoning Gym.
4. OpenCode runs with resources-provided sandbox access against SWE-bench and Terminal Bench.
5. All four OpenCode sandbox pairings preserve verification behavior.
6. GDPVal moves ordinary deliverable harvesting into resources.
7. Cancellation and injected failures leave no owned sandbox running.
8. Measured latency and throughput regressions remain within an agreed budget.

## Required foundation tests

- Malformed native and legacy requests fail before admission.
- Queue timeout creates no resources session.
- Scope-entry failure cleans partially acquired state.
- Seed failure never invokes a harness.
- Resources fails seed when verification requires a shared task sandbox but access cannot be returned.
- A harness rejects incompatible sandbox access before inference.
- Closing a resources-provided connection cannot stop the resources-owned sandbox.
- Closing a harness session stops its harness-owned sandbox on success, failure, timeout, and cancellation.
- The simple-agent harness rejects non-null sandbox access before inference.
- Cancellation during reconnect, guest startup, harness execution, verification, or cleanup runs every registered cleanup.
- Harness output limits reject oversized or malformed responses.
- SWE-bench extraction failure is not verified as an empty patch.
- SWE-bench submissions include supported new files.
- Terminal Bench verifies the original live task sandbox.
- Resources cleanup is idempotent.
- Cleanup failures appear in `EpisodeResponse.cleanup_failures`.
- Legacy projection preserves response, reward, metrics, tokens, completion accounting, and mask location.
- One real OpenCode and SWE-bench rollout leaves no sandbox running.

Tests for shared claims, owner leases, turn ordering, role visibility, and checkpoint restoration land with their corresponding follow-on milestones.
