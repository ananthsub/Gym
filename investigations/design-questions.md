# Episode architecture decisions and open questions

Status: review reference, 2026-09-10.

`episode-orchestration-design.md` is the normative proposal. This page records the reasoning behind its most consequential decisions, pushes back where a broader abstraction is not yet justified, and distinguishes current behavior, MVP behavior, and future guarantees.

## Does a Verifiers environment map to a Gym episode processor?

Partly.

- Verifiers `Env.run()` is closest to a concrete Gym processor's `process()`: it owns the interaction protocol.
- Verifiers `Env.run_episode()` is closest to framework-supplied `BaseEpisodeProcessor.run()`: it wraps protocol logic with common execution behavior.
- Verifiers task loading and environment-owned verification span Gym's task source and resources server.

A Gym environment remains a composition. The resources server owns benchmark state, tools, and verification. The episode processor owns participant interaction. The task source supplies immutable work. Agent harness and model deployments are bound to processor roles.

Calling either the resources server or episode processor the entire environment would hide one of those responsibilities.

## Why does the framework supply `run()`?

Every processor needs the same safety and operational envelope:

- pure request validation and compatibility translation;
- bounded, deadline-aware admission;
- cancellation propagation;
- cleanup registration;
- result validation and finalization;
- native or legacy result projection.

The framework should implement those once. A concrete processor implements `process(request, context)`.

The framework should not implement a universal seed → agent → verify body. That sequence is the first concrete `SingleAgentEpisodeProcessor`, not a law for user simulation or future multi-agent protocols.

## What exactly do the base processor helpers do?

`translate_or_validate` parses a native request or translates the legacy `/run` body. It has no side effects.

`admission.slot` acquires one per-worker capacity slot, respects the queued request's deadline, rejects during shutdown, and releases in `finally`. It is local backpressure, not a distributed attempt claim.

`episode_scope` creates a cancellation token, a session-aware resources client, and an `AsyncExitStack`. It runs registered cleanup on success, failure, cancellation, and timeout.

`finalize_result` validates and freezes the protocol result and attaches framework termination metadata. Verification is protocol logic and is not hidden in finalization.

`project_result` emits the native wire result or the exact legacy result shape.

The proposal does not use `context.seal()`. Result ownership belongs to the processor.

## Where do setup and teardown go?

There are four scopes:

1. Server setup/shutdown initializes registries, admission, executors, and reusable clients, then drains them on shutdown.
2. Episode setup/teardown creates cancellation and cleanup state for one attempt.
3. Resources-session seed/cleanup creates and destroys benchmark-owned state, including resources-owned sandboxes.
4. Harness invocation setup/cleanup connects and disconnects execution machinery for one call.

A single generic lifecycle hook would make ownership and failure ordering ambiguous.

## Is `EpisodeContext` an environment object?

No. It is a small, in-memory execution utility owned by one processor invocation. The MVP contains only:

- deadline;
- cancellation;
- session-aware resources client;
- cleanup stack.

It is not serialized and does not contain task routing, participant graphs, deployment configuration, artifact staging, checkpoint state, or generalized services. Protocol-specific state belongs to the concrete processor.

## Is every episode processor directly a server?

Yes. `SingleAgentEpisodeProcessor` directly subclasses `BaseEpisodeProcessor(SimpleServer)` and serves `/run`.

There is no umbrella episode-processor server that imports a selected processor implementation. Such a host would duplicate Gym's deployment selection, health, routing, and configuration while forcing heterogeneous processor dependencies into one runtime.

Pure protocol logic remains unit-testable through `process()` and injected collaborators.

## How does a processor define its agents?

The base processor does not.

Concrete processor configuration defines named participant roles. The MVP single-agent processor has one `policy: ParticipantBinding`. A future user-simulation processor has separate `policy` and `simulated_user` bindings. Each binding selects a harness deployment and model.

The processor class defines ordering and visibility among those roles. A second generic `schedule.kind` field would create a competing protocol selector and is intentionally omitted.

## Why is there no “standard episode processor”?

“Standard” incorrectly implies that other protocols are variants of a single-agent lifecycle.

`SingleAgentEpisodeProcessor` is one peer implementation of the base interface. User simulation and future multi-agent protocols are other peers. They reuse the framework envelope, executors, and lifecycle helpers; they do not subclass the single-agent processor.

Reward judges remain verifier internals in the resources server. There is no foundational `SolverJudgeEpisodeProcessor`. An interactive judge participant would require a distinct demonstrated use case.

## Why are full-loop and turn calls separate?

A full-loop call gives the harness control until it returns a final response. A turn call gives the processor control after one participant activation.

They differ in:

- who owns scheduling;
- what continuation state is required;
- event visibility;
- termination;
- checkpoint boundaries.

One struct with optional `response_params` and `turn` allows invalid combinations and hides those semantics. The MVP defines `FullLoopHarnessCall`. `TurnHarnessCall` arrives with the user-simulation milestone.

## Is `implementation: str` enough for harness deployment?

No.

Arbitrary import strings:

- execute module code during loading;
- may name code available only in a sandbox guest;
- cannot represent non-Python or remote implementations;
- do not pin behavior or package versions;
- pair naturally with untyped configuration.

The MVP uses an allowlisted `(factory, version)` key and validates configuration with the factory's typed Pydantic model. Imports may populate the registry, but request data does not choose arbitrary import paths.

The OpenCode factory can produce a known sandbox program even though execution happens in B. Future deployment forms may explicitly describe reviewed Python plugins, immutable guest bundles, or remote services. The behavior API remains independent of those artifact forms.

## What goes into seed-session request and response?

The request contains:

- rollout and attempt identity;
- benchmark-specific seed payload;
- workspace intent: no workspace or resources-server-owned workspace.

The resources server decides how to provision its internal sandbox.

The response contains:

- benchmark-specific initialized seed data;
- optionally, one serialized `SandboxWorkspace` granting operate-only access to B.

It does not contain a live sandbox, owner handle, provider credentials, harness deployment configuration, agent runtime configuration, executor, or installation instructions.

The processor already knows which harness and executor it selected. Returning those from resources would invert responsibility.

## Who owns sandbox B?

In the MVP, the resources server creates and owns B. It alone calls `stop()`.

The executor receives a serialized operate-only workspace, connects as a borrower, runs the harness, and calls `disconnect()`. It cannot destroy B. The resources server retains B until final extraction or live verification completes, then destroys it during resources-session cleanup.

The current OpenCode pairings do not enforce this consistently: resources and agent code both attempt to stop B on successful paths, while some exceptional paths leak it. The MVP removes both behaviors.

Processor-owned B and `/sandbox_spec` are deferred until an environment requires them.

## Does the seed response contain the harness runtime?

No. `SandboxWorkspace` describes task workspace B, not the program that will execute in it.

Harness deployment is selected by processor configuration. The executor resolves that trusted deployment and materializes it in B. Keeping workspace authority separate from executable selection prevents the resources server from becoming an agent deployment registry.

## Why not pass a live `AsyncSandbox`?

It cannot cross a process boundary as a stable wire contract, and it obscures ownership. A serialized workspace record carries provider identity, reconnect data, workdir, owner, access, and capabilities.

The owner string is descriptive, not the only authorization control. Borrowed host objects must enforce `owns_lifecycle=False`, and the guest must never receive provider control-plane credentials or a reconnect descriptor.

## Does every Python agent need a sandbox?

No. Implementation language is not the trust boundary.

A trusted native harness that only makes approved HTTP calls may run in a supervised local worker. A harness that executes model-directed shell commands, loads untrusted plugins, or requires task filesystem access runs through a sandbox executor. A remote harness can remain a service when dependency or scaling isolation requires it.

CLI harnesses run inside a sandbox in production. There is no host-execution fallback.

## Should the agent remain an HTTP server?

Not universally.

The episode processor remains a server. Harness deployment is selected independently:

- native behavior can execute in process or in a supervised worker;
- a CLI harness runs as a program in B;
- an independently scaled or non-Python harness can remain remote.

Removing an internal HTTP hop may reduce latency, but process startup and sandbox I/O usually dominate CLI execution. Forcing every harness into the processor would couple failures and exclude guest-only programs. The executor/factory boundary preserves both options.

The MVP does not remove existing `/v1/responses` compatibility endpoints.

## How does the verifier get the agent's work?

This remains resources-server-internal:

- SWE-bench, DeepSWE, and SWE-bench Pro extract benchmark-specific patch data from B and grade it in fresh verifier sandbox V.
- Terminal Bench 2.1 uploads tests and grades live B.

The processor asks resources to verify but does not carry generic files between B and V.

## Why are submission transfer, observations, and retained artifacts different?

Submission transfer is private movement of benchmark state among resources-owned components.

Observations and diagnostics are bounded structured evidence returned by one harness invocation.

Retained artifacts are caller-retrievable files that outlive episode cleanup. They require a storage owner, authorization, retention, garbage collection, and opaque identifiers.

The MVP therefore has no `ArtifactPayload`, `ArtifactSource`, or `DurableArtifactRef`. Base64 blobs and sandbox-local paths are not durable public interfaces. A retained-artifact subsystem can be designed later if callers need it.

## How do TaskData, TaskSet, and routing work in the MVP?

They do not change.

The MVP preserves current JSONL data, `agent_ref`, and named deployment routing. Compatibility translation validates task data before side effects and projects the result back to the existing caller shape.

After the processor and sandbox boundary works, Gym can evolve its existing `EnvironmentManifest` with processor protocol, roles, task schema, and capabilities. A versioned `TaskSet` can then yield immutable task records and agent-agnostic routing metadata.

Creating a parallel `EnvironmentProfile` system at the start is unnecessary.

## What does NeMo RL see during migration?

The same route and result it sees today.

`SingleAgentEpisodeProcessor` is deployed behind the existing named `agent_ref`. It translates the legacy request, invokes one internal participant, and projects the exact response, reward, token lineage, completion, and mask fields currently consumed by NeMo RL.

Native NeMo RL support is follow-up work. Multi-participant episodes are not exposed through the single-participant adapter.

## Where is attempt fencing?

Not in the MVP.

Admission means a local processor worker acquired capacity. Fencing means a shared attempt store rejected an obsolete writer. Conflating them would imply restart safety that in-memory cookie-bound resources state does not provide.

A later milestone adds atomic attempt claims, leases, ownership epochs, stale-writer fencing, idempotent finalization, and orphan cleanup.

## Does reconnecting to a sandbox provide checkpointing?

No. Reconnection says a runtime may still exist. It does not capture processor event position, resources state, participant continuation, process state, PTYs, model-call lineage, or ownership epoch.

Checkpointing requires a coordinator and a coherent commit across every required owner. CLI subprocesses may support checkpointing only at clean invocation or turn boundaries. The MVP is best effort across restarts.

## What can implementation teams start in parallel?

After the native request/result and lifecycle contracts are accepted:

1. Processor foundation: base server, admission, episode scope, translation, projection, and cleanup tests.
2. Harness and sandbox bridge: trusted OpenCode factory, executor, serialized workspace, and ownership enforcement.
3. Compatibility characterization: existing `/run`, resources calls, NeMo RL consumption, and failure goldens.

The integration gate is one real OpenCode plus SWE-bench rollout with equivalent result behavior and no leaked or double-stopped task sandbox.

## Which implementation questions remain open?

The MVP still needs concrete choices for:

- the exact legacy request and result schemas captured by golden tests;
- the package and version identity used by the trusted OpenCode factory;
- provider-specific redaction and authorization of workspace descriptors;
- bounded observation and diagnostic limits;
- cancellation behavior when a sandbox command does not terminate promptly;
- acceptable latency and throughput regression budgets;
- shutdown behavior for active episodes after the grace period.

These choices affect implementation but do not require expanding the architectural foundation.
