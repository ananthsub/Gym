# Episode orchestration branch guide

Status: orientation, 2026-09-10.

This branch holds the public RFC, the target architecture, and a concrete analysis of the current sandboxed OpenCode benchmark stack. `rfcs/gym-architecture.md` is the public RFC (sanitized snapshot at revision `6c57b803`). `investigations/episode-orchestration-design.md` defines the target architecture and its contracts. `investigations/opencode-sandboxed-pairings.md` traces the current OpenCode integration with four resources servers. This guide states the design in one page, maps the vocabulary, lists the disagreements, and gives a reading order.

## The design in eight decisions

1. A Gym environment is a composition. `TaskSet` supplies immutable tasks, the resources server owns task state and verification, and a concrete processor owns the interaction protocol. Agent harnesses and models are bound separately by a run.
2. Every concrete processor is directly deployable. `SingleAgentEpisodeProcessor`, user simulation, environment loops, best-of-N, solver/judge, and other multi-agent processors are peers. None extends a supposedly standard single-agent lifecycle.
3. The framework still supplies mandatory scaffolding. `BaseEpisodeProcessor.run()` establishes admission, attempt fencing, cancellation, participant capture, cleanup registration, failure classification, and result sealing around protocol-specific `process()`. It does not prescribe seed or verification order.
4. The agent is a behavior contract, not necessarily a server. `AgentHarness.responses(params, context)` runs a complete loop; `TurnAgent.act(turn, context)` runs one scheduled activation. Existing independently deployed agents remain supported through a behavior-only remote executor.
5. Placement is configuration. A supervised local worker, task workspace, dedicated sandbox, or remote service can execute the same harness contract. Command-line harnesses run in a sandbox in production; missing isolation is a preflight error.
6. Sandbox ownership is explicit. The creator alone calls `stop()`. Borrowers connect with `owns_lifecycle=False` and disconnect. The guest receives a workdir and scoped endpoints, never provider credentials or a reconnect descriptor.
7. Verification stays with the resources server. Each sandbox survives through its final required operation: SWE-bench, DeepSWE, and SWE-bench Pro can destroy the task sandbox after durable patch extraction while fresh verifier sandboxes continue grading; Terminal Bench 2.1 keeps the live task sandbox through grading.
8. Compatibility is an exact projection. Existing NeMo RL sees its current route, complete primary `response.output`, token or receipt lineage, scalar and component rewards, mask location, failure sentinels, and aggregation behavior. Multi-participant episodes wait for native chronological projection.

## The vocabulary map

| Concept | Design | Public RFC | Today's code |
| --- | --- | --- | --- |
| Environment | `TaskSet` + resources semantics + required processor protocol + runtime requirements | Primarily resources server plus processor configuration | Resources and agent server pairing |
| Framework episode wrapper | `BaseEpisodeProcessor.run()` opens a fenced `EpisodeContext`, calls protocol-specific `process()`, seals, and cleans | `SimpleEpisodeProcessor.run()` implements one seed → harness → verify flow | Repeated agent-specific `/run` implementations |
| Episode owner | A directly deployed `SingleAgentEpisodeProcessor`, `UserSimulationEpisodeProcessor`, or other concrete processor server | Episode processor: the agent server renamed, with a concrete `SimpleEpisodeProcessor.run()` | Agent server `/run` |
| Processor contract | `BaseEpisodeProcessor(SimpleServer)` with common transport behavior and abstract protocol-specific `process()` | `BaseEpisodeProcessor` with concrete `run()` and no hooks | None |
| Agent behavior | `AgentHarness.responses(params, HarnessContext)` and `TurnAgent.act(turn, HarnessContext)` | `AgentHarness.responses(params, EpisodeContext)` | `SimpleResponsesAPIAgent.responses()` |
| Where the harness runs | `HarnessExecutor`: local worker, sandbox guest in the task workspace or a dedicated sandbox, or remote endpoint | Inside the processor process, in a venv keyed by the harness; `RemoteAgentHarness` otherwise | Inside the agent server process; a second `_sandboxed_agent` directory for sandboxes |
| What crosses to the harness | `HarnessContext`: rollout and participant identity, reachable server URLs, credential files, workdir, deadline | `EpisodeContext`: rollout id, resources client, the live `AsyncSandbox` | Env vars and a rollout-prefixed model URL |
| Task workspace handoff | `SandboxWorkspace` with provider name, serialized descriptor, owner, operate access, attested capabilities | Serialized descriptor to the verifier; live object to the harness | `sandbox_handle: str` on the seed response |
| Who owns a sandbox | Its creator: resources server or processor, recorded in `WorkspaceRequest.mode` | The processor, always | Whoever created it, unrecorded |
| Sandbox server (PR #2085) | Conditional adapter for docker, apptainer, enroot | Dependency of Phase 3 | Unmerged |
| Routing key on rows | `EpisodeProcessorRef` in run configuration; `execution_name` projects to a compatibility `agent_ref` | The processor, which names the harness; author note proposes removing `agent_ref.name` | `agent_ref.name` or `task_source` |
| More than one agent | Peer user-simulation, environment-loop, solver/judge, best-of-N, or other processors over `participants`, `TurnAgent`, and `/apply_turn` | Not modeled; tau2 handled by a custom processor | tau2 drives both seats internally |
| Task discovery | Versioned `TaskSet` yields immutable `TaskRecord` values with `EnvironmentRef` | Unresolved | JSONL paths configured on servers |
| Task data | `task_data` validated by the resources server's `TaskData`; never a deployment route | `EpisodeContext` declared on base request models | Flat row extras with `extra="allow"` |
| Migration | Nineteen dependency-ordered steps; parallel characterization and extraction precede routing changes | Legacy processor first, then releases 0.7.0, 0.8.0, 0.9.0 | None |

## Where the two documents disagree

- The processor hierarchy. The RFC gives the base class a concrete protocol lifecycle. The design gives `BaseEpisodeProcessor` only common server behavior and uses peer concrete servers for distinct interaction protocols. `SingleAgentEpisodeProcessor` is not a superclass for user simulation or multi-agent behavior.
- The framework envelope. The design requires common fencing, cancellation, capture, cleanup, and sealing around every protocol but does not force one seed, one harness call, or one verification.
- Environment and task routing. The design introduces immutable `TaskSet` identity and an environment profile that binds resources semantics to a required processor protocol while leaving agent and deployment selection to the run.
- Where the harness runs. The RFC runs it inside the processor interpreter and accepts the loss of fault isolation. The design runs trusted Python in a supervised worker and untrusted CLIs in a sandbox guest, and keeps the processor's event loop free of harness code.
- What the harness receives. The RFC hands it the live `AsyncSandbox`. The design hands it a working directory and endpoints, and keeps every descriptor in trusted host code.
- Who owns sandboxes. The RFC assigns all of them to the processor. The design assigns each to its creator, which is what pooled tool sandboxes and benchmark-created workspaces already require.
- The routing key. The RFC moves it to the processor. The design keeps canonical task rows agent-agnostic and selects the processor and harness from run configuration. Unmodified NeMo RL receives a legacy materialized view with `agent_ref` before dispatch and the existing result shape until native integration lands.
- The sandbox server. The RFC gates its Phase 3 on it. The design uses native reconnect for OpenSandbox and E2B and reserves the server for providers that cannot reconnect.
- Multiple agents. The RFC defers them. The design models participants, turns, and a chronological event log from the first request.

## Reading order

1. This guide, then `design-questions.md`, which records supported decisions, pushes back on incorrect premises, and identifies the choices that remain open.
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
| `investigations/design-questions.md` | Design decisions, pushback, and open questions | This branch; reference |
