# Episode architecture branch guide

Status: orientation, 2026-09-11.

This branch contains the public Gym architecture RFC and a concrete proposal for episode processing, agent harness execution, sandbox ownership, and task routing.

## The three foundations

1. **Episode processor.** Every concrete processor is directly deployable as a server. `BaseEpisodeProcessor.run()` supplies validation, admission, cancellation, cleanup, finalization, and compatibility projection. A concrete `process()` method owns its interaction protocol.
2. **Agent harness and sandbox execution.** `AgentHarness.responses()` preserves Gym's complete Responses API operation. Executors supply typed service clients, remote transport, supervised subprocesses, or a bounded command session. A host-side OpenCode adapter controls the OpenCode CLI in a sandbox; Gym does not install a generic Python guest runner in that sandbox.
3. **TaskSet and routing.** A strict `TaskSet` yields immutable `TaskData`. Trusted run configuration binds the task set to an environment processor, resources server, participant harnesses, and models. Task rows cannot select executable deployments.

These contracts are reviewed together and implemented in order. Existing JSONL, `agent_ref`, `/run`, cookie affinity, and NeMo RL result behavior remain available while the native path is introduced.

## Three representative deployment paths

The proposal includes complete merged configuration for three placements:

1. A Gym-native `simple_agent` runs in a supervised subprocess and uses typed model and resources clients. No task sandbox exists.
2. OpenCode paired with `reasoning_gym` runs in an executor-owned sandbox because the environment returns no task workspace. The executor's trusted deployment config supplies the image and destroys the sandbox before verification.
3. OpenCode paired with SWE-bench or Terminal Bench borrows an environment-owned task workspace. The resources server derives the image and sandbox spec, retains the workspace through verification, and destroys it during resources-session cleanup.

The harness config contains only behavior settings such as OpenCode version and context limits. Sandbox provider, image, resources, connection mode, supervision, and byte limits belong to the environment or executor that owns them. Most users select a shipped environment-and-agent preset; the expanded deployment configuration is for authors, operators, and reviewers.

The proposal also contains complete step-by-step episode flows. Its HTML render provides interactive walkthroughs for the OpenCode and simple-agent paths.

## Contracts defined by the proposal

- episode identity, request, context, result, failure, metrics, diagnostics, and cleanup;
- concrete processor server setup and single-agent protocol;
- complete Responses harness behavior and the separate future turn behavior;
- trusted harness deployment and adapter validation;
- native, remote, and sandbox-command execution;
- resources-owned workspace handoff and borrower enforcement;
- typed `SimpleAgentHarnessConfig` and `OpenCodeHarnessConfig` with three complete deployment examples;
- exact current-to-target behavior mappings for OpenCode and `simple_agent`;
- task identity, TaskData, TaskSet, environment binding, selection, sharding, repeat expansion, and native dispatch;
- legacy JSONL, `agent_ref`, `/run`, and NeMo RL projection.

## Capabilities added after the foundations

1. User simulation adds `turn()`, two participant roles, role visibility, and chronological policy attribution.
2. Other multi-agent processors add protocol-specific ordering, concurrency, and termination.
3. A sandbox server adds operate leases only for providers that cannot reconnect directly.
4. Restart-safe attempts add shared claims, leases, ownership epochs, and stale-writer fencing.
5. Checkpoint restoration adds coordinated snapshots across every state owner.
6. Caller-retained artifacts add durable storage and lifecycle policy.

`EnvironmentProfile`, a generic participant scheduler, a combined run/turn request, generic CLI installation plans, and generic artifact payloads are not needed for these foundations.

## Review position

- A concrete episode processor is a server; there is no umbrella host that imports a processor implementation selected inside `/run`.
- The framework supplies a neutral execution envelope, not a universal single-agent loop.
- `SingleAgentEpisodeProcessor` is a peer of user-simulation and future multi-agent processors, not their superclass.
- Harness behavior is independent of deployment. The OpenCode adapter remains host-side while its CLI runs in either an executor-owned harness workspace or an environment-owned task workspace.
- `CommandSession` standardizes the narrow execution operations shared by CLI harnesses without standardizing their installers or transcript formats.
- Sandbox ownership follows creation. Borrowers disconnect; owners stop.
- `responses()` and `turn()` are separate behavior contracts.
- `EnvironmentManifest` evolves as the environment metadata authority; TaskSet handles typed task supply and agent-independent routing.
- Reliability, checkpoint, and retained-artifact contracts arrive with their required backing systems.

## Parallel implementation work

Five workstreams can progress against reviewed contracts:

1. processor server and lifecycle;
2. native simple-agent harness extraction and subprocess execution;
3. OpenCode adapter, executor-owned sandbox support, and environment-workspace borrowing;
4. TaskSet, manifest evolution, and routing;
5. compatibility characterization for Gym and NeMo RL.

The integration gates first prove the processor with the simple-agent path, then prove an executor-owned OpenCode sandbox with a response-only verifier, then prove environment-owned workspaces with OpenCode plus SWE-bench and Terminal Bench, and finally drive the representative paths through native TaskSet routing.

## Reading order

1. `episode-orchestration-design.md` for the normative proposal and worked episode flows.
2. `episode-orchestration-design.html` for the standalone interactive walkthrough.
3. `rfcs/gym-architecture.md` for the public RFC being reviewed.

## File map

- `rfcs/gym-architecture.md`: sanitized public RFC snapshot at revision `6c57b803`.
- `investigations/episode-orchestration-design.md`: normative architecture proposal.
- `investigations/episode-orchestration-design.html`: standalone interactive render.
- `investigations/episode-orchestration-overview.md`: this guide.
