# Episode architecture branch guide

Status: orientation, 2026-09-11.

This branch contains the public Gym architecture RFC and a concrete proposal for episode processing, agent harness execution, and sandbox ownership.

## The two foundations

1. **Episode processor.** Each migrated `responses_api_agents` deployment hosts one concrete processor. `BaseEpisodeProcessor.run()` supplies validation, admission, cancellation, cleanup, finalization, HTTP status, and compatibility projection. A concrete `process()` method owns its interaction protocol.
2. **Agent harness and sandbox execution.** `AgentHarness.responses()` preserves Gym's complete Responses API operation. `NativeHarnessExecutor`, `RemoteHarnessExecutor`, and `SandboxHarnessExecutor` place that behavior without changing its contract. A host-side OpenCode adapter controls the OpenCode CLI through a bounded `HarnessSandbox`; Gym does not install a generic Python guest runner in that sandbox.

Existing JSONL, `task_source`, `agent_ref`, `/run`, cookie affinity, and NeMo RL result behavior remain available during migration. TaskSet and manifest redesign are follow-on routing work, not a third foundation.

## Three representative deployment paths

The proposal includes complete merged configuration for three placements:

1. A Gym-native `simple_agent` runs in a supervised subprocess and uses typed model and resources clients. No task sandbox exists.
2. OpenCode paired with `reasoning_gym` runs in an executor-owned sandbox because the environment returns no task workspace. The executor's trusted deployment config supplies the image and destroys the sandbox before verification.
3. OpenCode paired with SWE-bench or Terminal Bench borrows an environment-owned task workspace. The resources server derives the image and sandbox spec, retains the workspace through verification, and destroys it during resources-session cleanup.

The harness config contains only behavior settings such as OpenCode version and context limits. Sandbox provider, image, resources, connection mode, supervision, and byte limits belong to the environment or executor that owns them. Most users select a shipped environment-and-agent preset; the expanded deployment configuration is for authors, operators, and reviewers.

The proposal also contains complete step-by-step episode flows. Its HTML render provides interactive walkthroughs for the OpenCode and simple-agent paths.

The resources server is the environment and the sole logical owner of any task workspace. An executor-owned sandbox is a different resource: a harness-only runtime for an environment that verifies solely from the returned response. The processor never owns or transfers task state.

## Contracts defined by the proposal

- episode identity, request, context, result, failure, metrics, diagnostics, and cleanup;
- processor setup within the existing agent-server category and the single-agent protocol;
- complete Responses harness behavior and the separate future turn behavior;
- trusted harness-binding and adapter validation;
- native, remote, and sandbox harness execution;
- resources-owned workspace handoff and borrower enforcement;
- typed `SimpleAgentHarnessConfig` and `OpenCodeHarnessConfig` with three complete deployment examples;
- exact current-to-target behavior mappings for OpenCode and `simple_agent`;
- worker-local resources sessions, cookie updates, identity checks, and idempotent cleanup;
- legacy JSONL, `agent_ref`, `/run`, HTTP behavior, and NeMo RL projection;
- migration classes for existing agents and a concrete GDPVal migration.

## Capabilities added after the foundations

1. User simulation adds `turn()`, `assistant` and `simulated_user` roles, role visibility, and chronological trainable-role attribution.
2. Other multi-agent processors add protocol-specific ordering, concurrency, and termination.
3. A sandbox server adds operate leases only for providers that cannot reconnect directly.
4. Restart-safe attempts add shared claims, leases, ownership epochs, and stale-writer fencing.
5. Checkpoint restoration adds coordinated snapshots across every state owner.
6. Caller-retained artifacts add durable storage and lifecycle policy.

`EnvironmentProfile`, a generic participant scheduler, a combined run/turn request, generic CLI installation plans, and generic artifact payloads are not needed for these foundations.

## Review position

- A concrete processor runs behind one existing `responses_api_agents` deployment; the proposal adds no fourth server category or umbrella host.
- The framework supplies a neutral execution envelope, not a universal single-agent loop.
- `SingleAgentEpisodeProcessor` is a peer of user-simulation and future multi-agent processors, not their superclass.
- Harness behavior is independent of deployment. The OpenCode adapter remains host-side while its CLI runs in either an executor-owned harness workspace or an environment-owned task workspace.
- `HarnessSandbox` is the bounded, operate-only subset of `AsyncSandbox` used by CLI harnesses; it is not a generic command-session protocol.
- The resources server owns task workspaces. A sandbox service may hold a physical provider handle, but resources decides when the task workspace is destroyed. Executors own and stop only separate harness runtimes.
- `responses()` and `turn()` are separate behavior contracts.
- Wire compatibility around a migrated processor is distinct from `LegacyAgentRunProcessor`, which temporarily forwards to an unmigrated agent's episode-level `/run`.
- Current task-data and routing conventions remain in place until a separate proposal addresses them.
- Reliability, checkpoint, and retained-artifact contracts arrive with their required backing systems.

## Parallel implementation work

Five workstreams can progress against reviewed contracts:

1. processor lifecycle, resources sessions, compatibility projection, and legacy passthrough;
2. native simple-agent harness extraction and subprocess execution;
3. OpenCode adapter, executor-owned sandbox support, and environment-workspace borrowing;
4. migration of step-based and locally graded agents, including GDPVal;
5. compatibility characterization for Gym and NeMo RL.

The integration gates first prove the processor with the simple-agent path, then prove an executor-owned OpenCode harness sandbox with a response-only verifier, then prove environment-owned task workspaces with OpenCode plus SWE-bench and Terminal Bench, and finally prove GDPVal's resources-owned deliverable harvesting.

## Reading order

1. `episode-orchestration-design.md` for the normative proposal and worked episode flows.
2. `episode-orchestration-design.html` for the standalone interactive walkthrough.
3. `rfcs/gym-architecture.md` for the public RFC being reviewed.

## File map

- `rfcs/gym-architecture.md`: sanitized public RFC snapshot at revision `6c57b803`.
- `investigations/episode-orchestration-design.md`: normative architecture proposal.
- `investigations/episode-orchestration-design.html`: standalone interactive render.
- `investigations/episode-orchestration-overview.md`: this guide.
- `investigations/episode-orchestration-design-review.md`: audit of the proposal against Gym and NeMo RL upstream main, with the change needed for each finding.
