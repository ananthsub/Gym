# Episode orchestration branch guide

Status: orientation, 2026-09-10.

This branch holds the public RFC, the target architecture, and a concrete analysis of the current sandboxed OpenCode benchmark stack. `rfcs/gym-architecture.md` is the public RFC (sanitized snapshot at revision `6c57b803`). `investigations/episode-orchestration-design.md` defines the target architecture and its contracts. `investigations/opencode-sandboxed-pairings.md` traces the current OpenCode integration with four resources servers. This guide states the design in one page, maps the vocabulary, lists the disagreements, and gives a reading order.

## The design in six decisions

1. One component owns the episode. `EpisodeProcessorServer` receives `POST /run`, and `StandardEpisodeProcessor` inside it runs seed, harness execution, verification, and cleanup in a fixed order. The collector still makes one control-plane call per rollout. Existing agent-server deployments become processor deployments on the same host, port, and worker count, so routing does not change on day one.
2. The agent is a Python behavior contract, not a server. `AgentHarness.responses(params, context)` runs a complete loop; `TurnAgent.act(turn, context)` runs one scheduled turn. Neither seeds, verifies, selects a sandbox provider, or cleans up. An agent that must stay an HTTP service is reached through a remote executor that calls only `/v1/responses`.
3. Placement is configuration. `AgentRuntimeConfig` puts a harness in a supervised local worker, inside the task workspace the benchmark created, in a dedicated sandbox the processor creates, or behind a remote endpoint. Command-line harnesses run in a sandbox in production; a missing sandbox is a preflight error, never a host fallback.
4. Sandbox ownership is explicit. Whoever calls `start()` owns the sandbox and is the only one who calls `stop()`. Every other trusted component connects with `owns_lifecycle=False` and disconnects. `/seed_session` states who created the task workspace through `WorkspaceRequest.mode` and returns operate-only `SandboxWorkspace`. The harness process never receives a descriptor. A sandbox server is used only for providers that cannot reconnect natively.
5. Verification stays with the resources server, and it extracts before the owner destroys. SWE-bench extracts a diff, DeepSWE runs its commit-aware collect hook, and SWE-bench Pro filters pristine files before those servers grade patches in fresh verifier sandboxes. Terminal Bench 2.1 grades the live task sandbox. Artifacts leave the sandbox as bounded payloads or durable references, never as paths.
6. Compatibility is preserved by translation, not by a second lifecycle. `agent_ref.name`, `_ng_rollout_id`, `response.output`, `reward`, `reward_components`, and `instance_config.mask_sample` keep their meanings for NeMo RL. A `legacy_routes` table lets an existing deployment translate today's `/run` body into an `EpisodeRequest`.

## The vocabulary map

| Concept | Design | Public RFC | Today's code |
| --- | --- | --- | --- |
| Episode owner | `EpisodeProcessorServer` hosting `StandardEpisodeProcessor` | Episode processor: the agent server renamed, with a concrete `SimpleEpisodeProcessor.run()` | Agent server `/run` |
| Processor contract | `EpisodeProcessor` protocol with one `process` method and no inherited body | `BaseEpisodeProcessor` with concrete `run()` and no hooks | None |
| Agent behavior | `AgentHarness.responses(params, HarnessContext)` and `TurnAgent.act(turn, HarnessContext)` | `AgentHarness.responses(params, EpisodeContext)` | `SimpleResponsesAPIAgent.responses()` |
| Where the harness runs | `HarnessExecutor`: local worker, sandbox guest in the task workspace or a dedicated sandbox, or remote endpoint | Inside the processor process, in a venv keyed by the harness; `RemoteAgentHarness` otherwise | Inside the agent server process; a second `_sandboxed_agent` directory for sandboxes |
| What crosses to the harness | `HarnessContext`: rollout and participant identity, reachable server URLs, credential files, workdir, deadline | `EpisodeContext`: rollout id, resources client, the live `AsyncSandbox` | Env vars and a rollout-prefixed model URL |
| Task workspace handoff | `SandboxWorkspace` with provider name, serialized descriptor, owner, operate access, attested capabilities | Serialized descriptor to the verifier; live object to the harness | `sandbox_handle: str` on the seed response |
| Who owns a sandbox | Its creator: resources server or processor, recorded in `WorkspaceRequest.mode` | The processor, always | Whoever created it, unrecorded |
| Sandbox server (PR #2085) | Conditional adapter for docker, apptainer, enroot | Dependency of Phase 3 | Unmerged |
| Routing key on rows | `EpisodeProcessorRef` in run configuration; `execution_name` projects to a compatibility `agent_ref` | The processor, which names the harness; author note proposes removing `agent_ref.name` | `agent_ref.name` or `task_source` |
| More than one agent | `participants`, `ScheduleSpec`, `TurnAgent`, `/apply_turn` | Not modeled; tau2 handled by a custom processor | tau2 drives both seats internally |
| Task data | `task_data` validated by the resources server's `TaskData` | `EpisodeContext` declared on base request models | Flat row extras with `extra="allow"` |
| Migration | Eighteen dependency-ordered steps; compatibility deployments and concrete OpenCode benchmark migrations precede routing changes | Legacy processor first, then releases 0.7.0, 0.8.0, 0.9.0 | None |

## Where the two documents disagree

- The processor base. The RFC gives the base class a concrete `run()` and forbids hooks, so one class exists per protocol. The design makes `EpisodeProcessor` a protocol and puts the lifecycle in `StandardEpisodeProcessor`.
- Where the harness runs. The RFC runs it inside the processor interpreter and accepts the loss of fault isolation. The design runs trusted Python in a supervised worker and untrusted CLIs in a sandbox guest, and keeps the processor's event loop free of harness code.
- What the harness receives. The RFC hands it the live `AsyncSandbox`. The design hands it a working directory and endpoints, and keeps every descriptor in trusted host code.
- Who owns sandboxes. The RFC assigns all of them to the processor. The design assigns each to its creator, which is what pooled tool sandboxes and benchmark-created workspaces already require.
- The routing key. The RFC moves it to the processor. The design keeps source task rows agent-agnostic, selects the processor and harness from run configuration, and emits compatibility `agent_ref.name` only in materialized requests and results while NeMo RL requires it.
- The sandbox server. The RFC gates its Phase 3 on it. The design uses native reconnect for OpenSandbox and E2B and reserves the server for providers that cannot reconnect.
- Multiple agents. The RFC defers them. The design models participants and turns from the first request.

## Reading order

1. This guide, then `design-questions.md`, which answers each design question in a paragraph with the decision, the reason, and the evidence.
2. Read `opencode-sandboxed-pairings.md` for the current execution path and the differences among SWE-bench, DeepSWE, SWE-bench Pro, and Terminal Bench 2.1.
3. In the design: "One episode in plain terms", "The current OpenCode benchmark stack is the migration baseline", and "Contract summary". These sections explain the proposal and its concrete migration target.
4. In the design: "The agent execution decision changes one boundary" compares the agent-server and pure-harness paths. Its expandable call reference contains the complete rollout sequence. Every named type is defined in the "Complete definitions" sections and can be read on demand.
5. In the RFC: "Problem statement" and "Personas and Use Cases" for the motivation, then "Proposed solution" to see the alternative the design departs from. The RFC's appendix is evidence about today's code and does not need to be read to understand either proposal.

## What each file is

| File | What it is | Author and status |
| --- | --- | --- |
| `rfcs/gym-architecture.md` | Public RFC, sanitized snapshot | Gym architecture working group; in review |
| `investigations/episode-orchestration-design.md` | Target architecture with complete contracts | This branch; proposal |
| `investigations/episode-orchestration-design.html` | Rendered page of the design with SVG diagrams | Generated from the Markdown |
| `investigations/opencode-sandboxed-pairings.md` | Current OpenCode and resources-server execution analysis | This branch; analysis |
| `investigations/episode-orchestration-overview.md` | This guide | This branch |
| `investigations/design-questions.md` | Direct answers to the design questions | This branch |
