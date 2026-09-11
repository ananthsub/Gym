# Episode orchestration design decisions and open questions

Status: reference, 2026-09-10. This page answers questions where the code and proposal support a decision, pushes back on incorrect premises, and leaves implementation-critical choices open. `episode-orchestration-design.md` is the normative proposal; `opencode-sandboxed-pairings.md` records current behavior.

## How is a sandbox represented when it crosses from one server to another

Decision: use one small `SandboxWorkspace` JSON record. It names the configured provider, carries the provider's reconnect descriptor and working directory, records lifecycle ownership, and describes attested capabilities. It never carries a live `AsyncSandbox` object and is never just a bare sandbox id.

Pushback: `owner` records who must clean up; it is not an authorization grant. The descriptor is sensitive operate authority, and the host-side borrowed `AsyncSandbox` enforces `owns_lifecycle=False`. Owner reconnection uses a separate owner descriptor held in trusted state. Native provider descriptors do not all enforce `scope`, so the design must not rely on a caller honoring the `owner` string.

Current evidence is narrower than the original answer implied. SWE-bench and Terminal Bench 2.1 return a bare id. DeepSWE returns an id plus a reconnect descriptor that is typically `{sandbox_id, workdir}`, not a complete `SandboxWorkspace`. SWE-bench Pro also returns a PTY session id. OpenCode consumes only the bare id.

## Who creates a sandbox, and who destroys it

Decision: whoever calls `start()` is the sole lifecycle owner and the only component that calls `stop()`. A borrower connects with operate authority and disconnects without ending the physical sandbox.

Image selection does not determine ownership. The resources server can create task sandbox B and return an operate descriptor. Alternatively, it can return a `SandboxSpec` so the processor creates B and seed borrows it. A dedicated harness sandbox A is processor-owned. Sandboxes used only by resources-server tools remain resources-server-owned. When patch verification requires fresh sandbox V, the resources server creates, uses, and destroys V.

For the first OpenCode migration, SWE-bench, DeepSWE, SWE-bench Pro, and Terminal Bench 2.1 continue creating B in their resources servers. On the normal successful path today, both the resources server and OpenCode attempt to stop B. OpenSandbox and E2B tolerate a repeated close. Several exceptional paths instead skip agent-side cleanup entirely. The target removes both the double-stop and the leak.

## What does the agent program receive, and what must it never receive

Decision: a sandbox guest receives its working directory, approved endpoint addresses, and only rollout-scoped credential files needed to call those endpoints. It does not receive a provider credential, a reconnect descriptor, verifier-only data, or another participant's private input.

Pushback: possessing `SandboxWorkspace.owner` would not itself let a guest reconnect as owner. The real boundary is that the guest has no need for provider control-plane authority at all. The trusted executor retains the operate descriptor and invokes sandbox operations on the guest's behalf.

## Do Gym's own Python agents need a sandbox

Pushback: ownership by Gym and implementation language are not security boundaries. A trusted Python harness such as `simple_agent` that only calls approved HTTP tools can run in a supervised local worker. A Python harness that executes model-directed shell commands, loads untrusted plugins, or otherwise requires isolation must use a sandbox. `HarnessRequirements` and deployment trust policy decide placement.

The helper process is target behavior, not current behavior. It isolates crashes and blocking calls from the processor event loop without implying that all Python agents are safe.

## Where does a CLI agent run

Decision: a CLI agent runs inside a sandbox in production. There is no host-execution fallback.

For the referenced SWE-style benchmarks, the preferred placement is compatible task sandbox B because the repository is already there and verification extracts or grades its state. If B cannot satisfy the harness runtime requirements, the processor must either use a compatible dedicated sandbox A with environment-mediated tools or reject the pairing during preflight. If the benchmark has no task sandbox, the processor can create A. Missing or incompatible isolation fails before any model call.

## Agent as an HTTP server versus a Python class: what changes

Pushback: `/v1/responses` is not exclusively private. Normal rollout orchestration calls `/run`, and many agents then self-call `/v1/responses`, but repository client utilities also call that route directly. OpenCode's `/run` currently calls its `responses()` method directly rather than taking an internal HTTP hop.

An agent server can reach a sandbox; `opencode_sandboxed_agent` proves that by reconnecting, executing, exporting, and downloading through `AsyncSandbox`. The problem is that the generic agent contract has no workspace handoff or ownership contract, so specialized agents duplicate benchmark lifecycle state and cleanup.

Decision: the default Gym-native boundary is a Python `AgentHarness` executed by the processor through a supervised local worker or sandbox executor. A CLI guest uses the sandbox as its isolation boundary and does not require an HTTP server inside the sandbox. Existing independently deployed agents remain supported through `RemoteAgentHarnessExecutor`. Direct `/v1/responses` clients remain a compatibility surface rather than evidence that every harness must be an episode-owning server.

## Where does the resources server keep state between seed and verify

Open question: changing the key from a session cookie to `(rollout_id, attempt)` does not make process-local state available to another worker or replica. A reconnect descriptor recovers sandbox access but not benchmark state, PTYs, attempt ownership, cleanup records, or other session data.

The initial migration requires one resources-server worker, or an explicitly guaranteed affinity mechanism, and keeps the authenticated session cookie as the state selector. `rollout_id` and `attempt` fence operations within that selected session. Supporting several workers or replicas requires a process-shared session and attempt store for the complete resources-server state. The proposal does not claim that a new base-class dictionary solves this.

## What happens when the same rollout is seeded twice

Decision: seed is idempotent per `(rollout_id, attempt)`. A duplicate with the same validated input returns the committed seed response and workspace. A conflicting duplicate is rejected. An older attempt is rejected before side effects. A newer attempt atomically fences the older attempt before replacing its state, then retires the older resources without allowing stale verify or cleanup to affect the new attempt.

Open question: the process-shared claim, compare-and-swap, in-progress duplicate behavior, and committed-response storage still need an implementation contract. “Stop or reuse” is not sufficient because concurrent duplicate requests could stop an active sandbox or create two different sessions.

## How does the verifier get the agent's work

Decision: the resources server extracts the work before its owner destroys task sandbox B:

- SWE-bench, DeepSWE, and SWE-bench Pro copy a benchmark-specific patch from B and grade it in fresh sandbox V. SWE-bench must be fixed to include new files and canonical binary changes. DeepSWE keeps its commit-aware extraction. SWE-bench Pro keeps its pristine-untracked filtering.
- Terminal Bench 2.1 uploads tests and grades the live B because a fresh sandbox would discard the machine state under test.

Anything returned beyond the sandbox boundary is a bounded payload or durable artifact reference, not a path that becomes invalid after cleanup.

## Who installs the CLI, and where does the binary come from

Decision: the harness defines the versioned behavior and entrypoint. The executor materializes the approved runtime into the sandbox. Production should use a prepared image or an immutable digest-verified bundle. Downloaded installers remain a development-only transition.

Pushback: current OpenCode defaults download and install OpenCode on every invocation. “Production does not download per task” is a target policy, not current behavior. The bundle manifest, extraction layout, operating-system and architecture constraints, authenticated retrieval, and cache behavior remain to be specified before immutable bundles are a complete executable contract.

## What happens when the harness fails for reasons that are not the model's fault

Pushback: `HarnessResult` does not carry a failure class. It represents a completed harness invocation with a response, observations, diagnostics, and artifacts. If an executor cannot produce that result, the processor classifies the failure as `EpisodeFailure` and chooses terminal result or retryable transport behavior.

Execution failure, invalid harness output, observation loss, and verification failure are distinct. A lost optional transcript is an `ObservationGap`, not automatically an infrastructure failure or reward zero. If a deployment declares a transcript mandatory, failure to preserve it can invalidate the invocation. Installation or endpoint reachability failures can be infrastructure failures, while invalid deployment configuration is terminal.

Current behavior is mixed: OpenCode converts some execution and export failures into degraded responses and still verifies the sandbox, while uncaught `/run` failures can already receive collector failure classes and enter the optional sidecar. The new contract must preserve that evidence while making retryability explicit.

## Is a sandbox server needed

Decision: select this from provider capability and deployment topology, not from benchmark name. OpenSandbox and E2B currently implement Gym's cross-process reconnect contract. Docker, Apptainer, and local do not; the existing `ConnectableProvider` contract says such providers require a sandbox server when another process must operate them.

Pushback: calling host-local reconnect “a small provider change” is unsupported. Reconnection, owner/borrower authority, client cleanup, process death, and host affinity require provider-specific design. PR #2085 remains the conditional adapter for a non-connectable provider or an enforced cross-process lease. It is not inserted in front of providers that already satisfy the required contract.

The referenced OpenCode pairings can begin with directly connectable OpenSandbox or E2B. That does not prove that no repository benchmark needs a sandbox server under another provider or topology.

## How does routing work, and what does NeMo RL see during migration

Decision: new canonical task datasets are agent-agnostic. Run configuration selects the processor, resources server, participants, harnesses, and models.

Pushback: current NeMo RL cannot consume such a row directly. It reads `agent_ref.name` before calling low-level `run_examples`, and that path does not apply rollout-collection routing. NeMo RL also does not currently consume the proposal's `_ng_rollout_id` or `terminal_response_id` contracts.

The backward-compatible path is therefore explicit and single-participant:

1. Materialize a legacy training view that injects `agent_ref` before NeMo RL receives the row.
2. Host `StandardEpisodeProcessor` behind the existing named agent deployment and `/run` configuration shape while old Gym clients do not know the new server type.
3. Translate the legacy request into one internal participant and project the result back to the exact legacy response, reward, token, completion, and `instance_config.mask_sample` shape.
4. Preserve current token-capture behavior rather than requiring new identifiers from old NeMo RL.
5. Do not expose multi-participant episodes through this adapter.

Native NeMo RL support is follow-up work. It must define rollout identity, terminal model-call attribution, capture finalization ownership, failure transport, masking, and primary-trajectory projection before the legacy view can be removed.

## Is the processor base class an interface

Decision: `EpisodeProcessor` is a protocol with one `process()` method and no inherited lifecycle body. `StandardEpisodeProcessor` is the concrete standard lifecycle.

Open question: a custom implementation cannot be accepted merely because it satisfies one method signature. Registration must be deployment-selected and trusted, and construction, injected services, shared fencing state, validation, verification ordering, and cleanup conformance must be specified. Until that plugin contract exists, the first implementation uses `StandardEpisodeProcessor`.

## Does the design support more than one agent in an episode

Decision: the request, schedule, and verification contracts represent multiple participants from the start. Each participant has a role, harness, model bindings, and capture identity.

Open question: representation is not yet a complete NeMo RL training projection. Selecting only the final primary response would drop earlier primary actions. Native integration must order every primary activation, include the intervening visible context, assign response and model-call identities, and construct action masks without admitting non-primary tokens into the policy update.

## What can start now

Work can begin on changes that preserve the legacy wire contract: extract harness behavior behind the Python interface, move shared `/run` lifecycle into `StandardEpisodeProcessor` inside existing named agent deployments, add supervised local workers, extend seed with optional workspace fields, and add borrowed `AsyncSandbox` connections that cannot stop the physical sandbox.

The first real target remains OpenCode with SWE-bench, followed by DeepSWE, SWE-bench Pro, and Terminal Bench 2.1. Agent-agnostic canonical data can be introduced at the same time only if existing NeMo RL jobs receive a legacy materialized view with `agent_ref`. Multi-worker resources state, immutable bundle packaging, custom processors, native NeMo RL identity and capture, and multi-participant training projection remain separate qualification work.
