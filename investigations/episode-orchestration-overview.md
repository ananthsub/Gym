# Episode architecture branch guide

Status: orientation, 2026-09-11.

This branch contains the public Gym architecture RFC and a concrete proposal for episode processing, agent-server sessions, and sandbox ownership.

## Core design

1. **Episode processor.** Each `episode_processors` deployment hosts one concrete processor. `BaseEpisodeProcessor.run()` supplies validation, admission, cancellation, cleanup, finalization, HTTP status, and compatibility projection. A concrete `process()` method owns its interaction protocol.
2. **Composed Gym services.** A processor may use independently deployed agent and resources servers when those boundaries fit its protocol. It may instead run a self-contained external integration such as Tau2. Agent servers retain their existing `POST /v1/responses` request and response models.

Gym adds `episode_processors` as a fourth server type and `sandbox_servers` as an optional fifth type. A sandbox server is configured only when provider state cannot be reconstructed across the resources-server and agent-server processes. Existing JSONL, `task_source`, `agent_ref`, `/run`, cookie affinity, and NeMo RL result behavior remain available during migration. Native callers use `EpisodeProcessorRef`. Broader routing redesign remains follow-on work.

## Representative deployment paths

1. A Gym-native `simple_agent` server uses the existing model server and scoped resources tools. It supports no sandbox.
2. OpenCode paired with `reasoning_gym` receives no sandbox access, so the OpenCode agent server creates and owns its configured sandbox.
3. OpenCode paired with SWE-bench or Terminal Bench receives `sandbox_access` because verification requires the CLI to modify the same task sandbox. Resources retains ownership and destroys that sandbox after verification.
4. Terminus-2 keeps its Python loop and Harbor dependencies in its agent-server virtual environment. Its terminal commands use resources-provided `SandboxAccess` when present and otherwise use its configured local workspace.
5. Tau2 runs as a self-contained episode processor that calls policy and simulated-user model servers without a Gym agent or resources server.

The processor config contains service references and protocol limits. Agent-specific configuration contains behavior settings and any sandbox the agent may create when resources returns no access. Most users select a shipped resources-and-agent preset; the expanded configuration is for authors, operators, and reviewers.

The proposal also contains complete step-by-step episode flows. Its HTML companion provides interactive walkthroughs for the representative deployments and the optional sandbox-server path.

When used, the resources server owns benchmark state, tools, verification, and task sandboxes. `sandbox_access` exposes only a sandbox that an agent must operate; it does not list every object owned by the resources server or say where agent-server code runs. Resources cannot implicitly inspect a separate sandbox created by the agent.

Agent sessions are process-local. The initial deployment uses one Uvicorn worker per agent-server replica, admits several asynchronous sessions in that worker, and scales through replicas or NeMo RL shards. A future routed pool must keep create, `/v1/responses`, and close on the replica that owns the session.

## Contracts defined by the proposal

- `EpisodeId`, `EpisodeRequest`, `EpisodeResponse`, `EpisodeVerification`, failure, and cleanup;
- the `episode_processors` server category, `EpisodeProcessorRef`, and the single-agent protocol;
- the unchanged `/v1/responses` behavior endpoint;
- `POST /v1/agent_sessions`, behavior invocation, and `POST /v1/agent_sessions/close`;
- resources-session seed, verification, and `POST /close_session`;
- agent-server bindings validated by each concrete processor;
- the optional `sandbox_servers` category, `SandboxServerRef`, resources-provided sandbox access, agent-created sandboxes, direct connections, borrower tokens, and borrower enforcement;
- typed Simple Agent, OpenCode, and Terminus-2 agent-server configuration with four deployment examples;
- exact current-to-target behavior mappings for OpenCode and `simple_agent`;
- scoped resources-tool and sandbox access, `EpisodeId` checks, and bounded cleanup;
- legacy JSONL, `agent_ref`, `/run`, HTTP behavior, and NeMo RL projection;
- horizontal agent-server scaling through one-worker replicas and session-affine routed pools.

## Capabilities added after the foundations

1. User simulation adds repeated `/v1/responses` activations, `assistant` and `simulated_user` roles, role visibility, and chronological trainable-role attribution.
2. Other multi-agent processors add protocol-specific ordering, concurrency, and termination.
3. Restart-safe attempts add shared claims, leases, ownership epochs, and stale-writer fencing.
4. Checkpoint restoration adds coordinated snapshots across every state owner.
5. Caller-retained artifacts add durable storage and lifecycle policy.

A generic participant scheduler, a separate turn API, generic CLI installation plans, and generic artifact payloads are not needed for the initial design.

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
