# Episode architecture branch guide

Status: orientation, 2026-09-11.

This branch contains the public Gym architecture RFC and a concrete proposal for episode processing, harness-server sessions, and sandbox ownership.

## The two foundations

1. **Episode processor.** Each migrated `responses_api_agents` deployment hosts one concrete processor. `BaseEpisodeProcessor.run()` supplies validation, admission, cancellation, cleanup, finalization, HTTP status, and compatibility projection. A concrete `process()` method owns its interaction protocol.
2. **Agent harness servers.** Each harness remains an independently deployed `responses_api_agents` server with its own package, virtual environment, process, and `POST /v1/responses` behavior endpoint. The processor opens role-scoped sessions and transports resources and optional sandbox access; it does not import harness implementations or their dependencies.

Existing JSONL, `task_source`, `agent_ref`, `/run`, cookie affinity, and NeMo RL result behavior remain available during migration. TaskSet and manifest redesign are follow-on routing work, not a third foundation.

## Four representative deployment paths

The proposal includes complete configuration for three pairings:

1. A Gym-native `simple_agent` harness server uses the existing model server and scoped resources tools. Its implementation supports no sandbox.
2. OpenCode paired with `reasoning_gym` receives no sandbox access, so the OpenCode harness server creates and owns its configured sandbox.
3. OpenCode paired with SWE-bench or Terminal Bench receives `harness_sandbox_access` because verification requires the CLI to modify the same task sandbox. Resources retains ownership and destroys that sandbox during session cleanup.
4. Terminus-2 keeps its Python loop and Harbor dependencies in its harness-server virtualenv. Its terminal commands use resources-provided `SandboxAccess` when present and otherwise use its configured server-local workspace.

The processor config contains service references and protocol limits. Harness-specific configuration contains behavior settings and any sandbox it may create when resources returns no access. Most users select a shipped environment-and-agent preset; the expanded configuration is for authors, operators, and reviewers.

The proposal also contains complete step-by-step episode flows. Its HTML render provides interactive walkthroughs for the OpenCode and simple-agent paths.

The resources server is the environment. Its state may be public internet, remote services, process-local data, or private sandboxes. `harness_sandbox_access` exposes only a sandbox that a harness may or must operate; it does not list every environment resource or say where harness code runs. Resources cannot implicitly inspect a separate sandbox created by the harness.

A multi-agent processor opens one harness-server session per role or agent instance. If seed returns one access, the protocol may share it through independent borrower connections. If seed returns no access, each harness follows its own configuration. Agents can communicate through shared task state, resources tools, remote services, or the public internet. A protocol requiring several distinct resources-owned sandboxes defines an extended seed contract because their names and sharing rules are protocol semantics.

## Contracts defined by the proposal

- episode key, request, context, result, failure, metrics, diagnostics, and cleanup;
- processor setup within the existing agent-server category and the single-agent protocol;
- the existing `/v1/responses` behavior endpoint and a separate future turn endpoint;
- harness-session open, behavior invocation, and idempotent close;
- trusted harness-server bindings and implementation-owned capability validation;
- optional resources-provided sandbox access, harness-created sandboxes, direct and sandbox-server connections, and borrower enforcement;
- typed simple-agent and OpenCode harness-server configuration with three complete deployment examples;
- exact current-to-target behavior mappings for OpenCode and `simple_agent`;
- role-scoped resources-session access, cookie compatibility, episode-key checks, and idempotent cleanup;
- legacy JSONL, `agent_ref`, `/run`, HTTP behavior, and NeMo RL projection;
- migration classes for existing agents, GDPVal's ordinary path, and its separate control-mode requirements.

## Capabilities added after the foundations

1. User simulation adds `turn()`, `assistant` and `simulated_user` roles, role visibility, and chronological trainable-role attribution.
2. Other multi-agent processors add protocol-specific ordering, concurrency, and termination.
3. A sandbox server becomes a declared resources-server dependency only when an exposed sandbox uses a provider that cannot reconnect directly.
4. Restart-safe attempts add shared claims, leases, ownership epochs, and stale-writer fencing.
5. Checkpoint restoration adds coordinated snapshots across every state owner.
6. Caller-retained artifacts add durable storage and lifecycle policy.

A generic participant scheduler, a combined run/turn request, generic CLI installation plans, and generic artifact payloads are not needed for these foundations.

## Review position

- A concrete processor runs behind one existing `responses_api_agents` deployment; the proposal adds no fourth server category or umbrella host.
- The framework supplies a neutral execution envelope, not a universal single-agent loop.
- `SingleAgentEpisodeProcessor` is a peer of user-simulation and future multi-agent processors, not their superclass.
- Harness behavior remains behind a dependency-isolated server. The OpenCode control loop runs in that service while its CLI operates either a harness-owned or resources-owned sandbox.
- `HarnessSandbox` is the bounded, operate-only subset of `AsyncSandbox` used by CLI harnesses; it is not a generic command-session protocol.
- Resources owns every sandbox it creates. A sandbox service may hold a physical provider handle, but resources decides when its task sandbox is destroyed. A harness server owns and stops only a sandbox it creates.
- `responses()` and `turn()` are separate behavior contracts.
- Wire compatibility around a migrated processor is distinct from `LegacyAgentRunProcessor`, which temporarily forwards to an unmigrated agent's episode-level `/run`.
- Current task-data and routing conventions remain in place until a separate proposal addresses them.
- Reliability, checkpoint, and retained-artifact contracts arrive with their required backing systems.

## Delivery plan

Implementation order, workstreams, integration gates, and required tests are maintained in `episode-orchestration-milestones.md`.

## Reading order

1. `episode-orchestration-design.md` for the normative proposal and worked episode flows.
2. `episode-orchestration-milestones.md` for implementation order and gates.
3. `episode-orchestration-design.html` for the interactive diagram companion.
4. `rfcs/gym-architecture.md` for the public RFC being reviewed.

## File map

- `rfcs/gym-architecture.md`: sanitized public RFC snapshot at revision `6c57b803`.
- `investigations/episode-orchestration-design.md`: normative architecture proposal.
- `investigations/episode-orchestration-milestones.md`: implementation sequence and integration gates.
- `investigations/episode-orchestration-design.html`: standalone interactive render.
- `investigations/episode-orchestration-overview.md`: this guide.
- `investigations/episode-orchestration-design-review.md`: audit of the proposal against Gym and NeMo RL upstream main, with the change needed for each finding.
