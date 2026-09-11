# Review of the episode architecture proposal against Gym and NeMo RL at upstream main

Status: design feedback with disposition, 2026-09-11.

This review checks `episode-orchestration-design.md` at commit `53af3c711` against the code it must fit. Commit `ebe52e96c`, which landed while the review was written, adds typed harness configurations and three complete deployment examples; it changes no finding below, and the examples still use the `episode_processors` category discussed first. The baseline is NeMo Gym upstream `main` at `e3dd5f6b1` and NeMo RL upstream `main` at `5d49fbf4e`. NeMo RL pins Gym as a submodule at `fd5e84d6b`, which is Gym pull request #2872 from 2026-09-04. Where the pin and Gym `main` differ, the pin is the compatibility target.

## What the proposal is trying to achieve

The proposal separates three things that one agent server does today: running the agent's model-and-tool loop, orchestrating the episode around it, and owning the sandbox the agent works in. It wants a benchmark author to define setup, tools, verification, and cleanup without choosing an agent. It wants an agent author to implement one behavior contract without knowing which benchmark calls it. It wants a command-line agent to run inside the benchmark's own sandbox without every agent copying sandbox lifecycle code. It wants all of this to keep the current `/run` route, the current `agent_ref` routing key, and the current result shape working for NeMo RL while the change lands.

Those goals are right, and the proposal's core decisions serve them. The framework owns the execution envelope and the concrete processor owns the protocol. Whoever creates a sandbox destroys it. The seed response returns a serialized, operate-only workspace record instead of a live object or a bare id. Verification runs while the task sandbox is alive. Compatibility is an explicit adapter with golden tests. None of those need to change.

What follows are the places where the proposal, as written, does not reach its goals or changes something else in Gym or NeMo RL that it does not account for. Each section states the goal, what happens as written, the evidence in the two codebases, and the change that meets the goal.

| Issue | Where in the design | Effect if unchanged |
| --- | --- | --- |
| A new server category costs more than the design assumes | 6.3, 8 stage 3 | Processor deployments launch but never get a host or port |
| Two binding sites for the harness, one unreachable | 4.5, 6.3 | Multiple harnesses per benchmark cannot be configured; RL cannot see capacity |
| Failure is carried in the body, but RL reads only the HTTP status | 4.2, 7.2 | Native failures look like successes to RL; admission rejections are misclassified |
| Reward components are explanatory, but RL requires them to sum to the reward | 7.2 | RL raises on the first multi-reward result |
| The native response drops fields that RL and aggregation read | 4.2, 5.5, 7.2 | Masking, per-verifier metrics, and legacy extras are lost |
| The resources session has no cleanup endpoint and no rollout identity | 4.3, 5.5 | Stage 1 cannot be built without base resources-server changes the design does not list |
| Entering rollout context does not select the token-capture prefix | 5.1, 5.6 | Training runs through a native harness capture nothing |
| The behavior contract reaches about two thirds of the agents | 5.1, 5.2 | Step-based and locally graded agents have no migration path |
| The remote executor loses the resources session | 5.2, 5.6 | Stateful benchmarks break for every agent kept as a server |
| Provider connectivity is a three-way split, not two | 5.7 | Single-host Docker and Apptainer users are pushed through a proxy they do not need |
| An executor-owned workspace cannot hand work to the verifier | 5.3 | The vibench pattern has no contract |
| TaskSet conflicts with the task-data and dataset conventions Gym already has | 6 | A second schema system next to the one that shipped |
| No deadline reaches the processor from RL | 4.2, 4.4 | Abandoned attempts keep sandboxes alive |
| Lifecycle hooks and worker semantics are assumed, not present | 4.4 | Startup validation and shutdown drain need base-class work |

## The processor must stay an agent server in configuration, not become a fourth server category

The design adds an `episode_processors` category with its own reference type so that native routing can select a processor explicitly. The goal is a clean name for the orchestration boundary.

As written, that category does not launch. Gym's configuration layer enumerates exactly three server categories in the reference types, the type-config classes, the instance-config union, and the helper that decides whether a top-level block is a server. A block under an unknown key is skipped when hosts and ports are assigned. The process launcher walks every top-level key and starts a process for any block with an `entrypoint`, so the processor would start, but its instance config would have no host and no port, and the readiness poll would fail. Beyond the launcher, the agent pairing rules, the dataset-to-agent resolution, the telemetry type map, the discovery helpers, the environment manifest's `agent_server` field, and NeMo RL's per-agent recovery overrides all assume the routing key names an entry under `responses_api_agents`. Adding a category is a change to about a dozen sites in Gym plus one in RL, and the design does not list any of them.

The category also buys nothing the design needs. The processor exposes `POST /run`, is named by `agent_ref`, and is what the collector and RL already address. From the configuration's point of view it is an agent server whose behavior happens to be extracted.

The fix is to keep processors under `responses_api_agents`. `BaseEpisodeProcessor` subclasses `SimpleServer` and is declared with an `entrypoint` like any agent server today. `AgentServerRef` names it. The `allowed_agents` list on a resources server, which today holds agent directory names, holds harness deployment names once processors are in use, and the pairing check compares against the processor's configured harness rather than its directory. The `EpisodeProcessorRef` type and the `episode_processors` category are removed from the proposal. If native routing later needs to distinguish a processor from a legacy agent server, a `kind` field on the agent instance config does that without a new category.

The evidence is in `nemo_gym/config_types.py` lines 114 to 133, 622 to 660, 691 to 707, and 742; `nemo_gym/global_config.py` lines 159 to 167, 489 to 588, 1624, 1652 to 1700, and 1756; `nemo_gym/cli/env.py` lines 376 to 497; `nemo_gym/server_utils.py` lines 734 to 738; `nemo_gym/environment/manifest.py` lines 185 to 265; and in NeMo RL `nemo_rl/environments/config.py` lines 655 to 700.

## One harness binding per deployment is the scaling unit, and the second binding site must go

The design wants a benchmark to run with several harnesses and wants NeMo RL to keep routing by name. Section 4.5 binds one `policy` harness on the processor's configuration. Section 6.3 also binds `participants` on the run configuration. `EpisodeRequest` has no participant field, so the run-configuration binding can never reach a deployed processor. Two harnesses against one benchmark cannot be expressed at all in native mode, and the two sites contradict each other.

The right binding site is the deployment. NeMo RL does not read Gym hosts, ports, or worker counts. It forwards the whole `env.nemo_gym` block to Gym and delegates discovery to Gym's own client. It reads the resolved `agent_ref` only after the call returns, and uses that name for recovery granularity and metric namespaces. Its concurrency bound is a per-prompt semaphore, default 32, not a per-server limit. Nothing on the RL side can size against a server that selects an implementation per request. Admission is per deployment in the design, and mixing a three-hour sandboxed harness with a ten-second native harness in one admission pool starves the short one. Startup validation of the harness, executor, and provider only means something when the binding is fixed before the server accepts work.

The fix is to delete `participants` from `EpisodeRunConfig`. Run configuration names a processor deployment; the deployment already binds harness and model. Several harnesses against one benchmark are several deployments sharing an entrypoint and a resources server, which is what `agent_map` and `fan_out` produce today. Multi-participant protocols keep per-role bindings on their own processor configuration. The design should also state the capacity contract that RL can read later: a deployment's concurrent episode capacity is `max_concurrent_episodes` times `num_workers`, since admission is per uvicorn worker.

The evidence is in NeMo RL `nemo_rl/environments/nemo_gym.py` lines 559 to 561, 616 to 621, and 1297 to 1340; `nemo_rl/experience/rollouts.py` line 1879; `nemo_rl/experience/rollout_manager.py` lines 1004 to 1011 and 1149; `nemo_rl/algorithms/single_controller.py` line 1568; and in Gym `nemo_gym/rollout_collection.py` lines 1311 to 1314.

## Failure must be carried in the HTTP status, because NeMo RL never reads the body for it

The design gives `EpisodeResponse` a terminal `status` and an `EpisodeFailure` with a `kind` and a `retryable` flag, so that callers can classify what went wrong. Section 7.2 says the compatibility projector creates "existing failure sentinels" but does not say what they are.

NeMo RL classifies a Gym failure only by HTTP status. A 5xx, 408, or 429 becomes a transport failure and is retried at the row level, up to three attempts, then at the prompt level with backoff. Any other 4xx becomes a data failure with a lower retry cap. A 200 with no trainable output item raises a value error that kills the stream for the whole batch. RL does not read `failure_reason`, `failure_kind`, or any body-level status. On the Gym side, the collector's failure sidecar is also keyed by HTTP status, and the judge failsafe returns a 200 with reward zero and `_ng_failure_class` set to `judge_failed`. Gym's exception middleware turns every uncaught exception into a 500.

A native `EpisodeResponse` with `status="failed"` inside a 200 is therefore invisible to RL as a failure and fatal to the batch as a success. An admission queue timeout returned as a 200 with a `deadline` failure is retried never. The `retryable` flag in the body is read by nobody.

The fix is to make the HTTP status part of the contract in both wire modes and to define the mapping once:

| `EpisodeFailure.kind` | HTTP status | RL classification |
| --- | --- | --- |
| `invalid_request` | 422 | data failure, low retry cap |
| `infrastructure`, `deadline`, `internal` when retryable | 503 with `Retry-After` | transport failure, retried |
| `harness` | 500 | transport failure, retried |
| `verification` | 200 with reward 0 and `_ng_failure_class` | not a failure to RL; sidecar row in Gym |
| admission queue timeout or shutdown | 503 with `Retry-After` | transport failure, retried |

The body still carries `EpisodeFailure` for callers that read it. The last row preserves today's judge failsafe behavior on purpose. It also documents a real gap: RL trains on a reward-zero row when verification failed, and neither the current code nor the design changes that. That is a separate decision to make with the RL owners.

The evidence is in NeMo RL `nemo_rl/environments/nemo_gym.py` lines 139 to 169 and 1115 to 1125; `nemo_rl/experience/rollout_manager.py` lines 1228 to 1278; `nemo_rl/environments/failures.py` lines 192 to 202; and in Gym `nemo_gym/judge.py` lines 85 to 110, `nemo_gym/rollout_collection.py` lines 840 to 859, and `nemo_gym/server_utils.py` lines 890 to 918.

## Reward components must sum to the reward

Section 7.2 says reward components are explanatory values and that a verifier may compute the scalar reward as a weighted aggregate, a minimum, or a mean. The goal is to let benchmarks define their own aggregation.

NeMo RL rejects that. After every batch it checks each result that carries `reward_components` and raises a value error if the scalar reward does not equal the sum of the components within tolerance. The error message states the rule: a multi-reward verifier must set reward equal to the sum so single-reward and multi-reward consumers read the same aggregate. Gym's own base class docstring says the scalar is "the aggregate expected by single-reward consumers" without naming the sum, which is how the design arrived at the looser reading.

The fix is one invariant in section 4.2: when `reward_components` is non-empty, `reward` equals their sum. `finalize_response` enforces it. A benchmark that wants a non-additive score reports the parts under `metrics.verification` and puts the score in `reward`. The Gym docstring should say the same.

The evidence is in NeMo RL `nemo_rl/environments/nemo_gym.py` lines 1215 to 1232 and in Gym `nemo_gym/base_resources_server.py` lines 112 to 125.

## The native response drops fields that RL and metric aggregation read today

The design narrows the verifier's reply to `reward`, `reward_components`, scalar `metrics`, and `diagnostics`, and rejects unknown fields on every native model. The goal is a typed, bounded contract instead of an open dictionary.

Three things that consumers read today do not survive that narrowing.

- Masking. RL reads `instance_config.mask_sample` from the result to exclude a sample from the loss. One resources server produces that field on its verify response, and the design mentions the field once, in the legacy projection list. The native verify reply has no place for it, so the value is lost before projection. Separately, the token-capture delivery step stamps a top-level `mask_sample` on the result inside the agent process; the processor's finalization must run that step, and the design does not say so.
- Per-verifier metrics. Aggregate metrics are computed from the top-level numeric and boolean fields of each verify response, merged with the response usage. Verifiers put dozens of such fields at the top level. The design's `metrics.verification` map can carry them only if the resources-session adapter copies every top-level scalar into it and the projector flattens them back. That rule is not written.
- Legacy extras. Verify responses carry non-scalar fields that evaluation tooling reads, such as verifier logs and observation bundles. The native model drops them. The projector cannot restore what the native model never carried.

The fix has three parts. Add `mask_sample: bool = False` to `EpisodeVerifyResponse` and `EpisodeResponse`, and have the projector emit `instance_config.mask_sample`. State that the resources-session adapter moves every top-level scalar of the environment's verify response into `metrics.verification`, and that the compatibility projector flattens that map back to the top level. Keep the environment's complete verify response on the resources-session client as an opaque compatibility payload that only the projector reads, so legacy fields reach legacy callers without entering the native contract. Also drop the claim that `failure_kind` exists; it is a comment in the base class and nothing sets or reads it.

The evidence is in NeMo RL `nemo_rl/experience/rollouts.py` lines 297 to 304; and in Gym `resources_servers/conversational_tool_use_simulation/app.py` line 435, `nemo_gym/token_id_capture/delivery.py` lines 49 to 201, `nemo_gym/reward_profile.py` lines 386 to 440, and `nemo_gym/base_resources_server.py` lines 104 to 109.

## The resources session has no cleanup endpoint and no rollout identity, and stage 1 depends on both

Section 4.3 registers `resources.close` as cleanup, section 5.5 defines `CleanupSessionRequest` with a rollout id and attempt, and the validation list says rollout id and attempt must match the active resources session. The goal is that every episode releases benchmark state on every exit path.

None of that exists on the resources-server side. The base resources server exposes seed, verify, aggregate metrics, and reverify mode. There is no cleanup or close route in core. Five servers have a private close method that only their own step loop calls. The session identity is a signed cookie holding a fresh UUID, assigned by middleware on every request. It is never derived from the rollout id. The rollout id reaches a resources-server handler only as a context variable set by the rollout middleware, and no handler stores it. Each of the four sandbox benchmarks keeps its sandbox in a per-process dictionary keyed by the cookie's session id, and none sets `num_workers`, which is the only reason seed and verify land on the same process.

So the design's cleanup call has no receiver, and its identity check has nothing to check against. Stage 1 lists no resources-server work, yet it cannot pass its own gate without it.

The fix is to make the base resources server part of stage 1 with three additions. A session table in the base class, keyed by the cookie session id, that records rollout id, attempt, and the resources the environment registered, with sandboxes registered through one helper. A `POST /cleanup_session` route in the base class that releases everything registered for the cookie's session, is idempotent, and reports `released`. A check at verify and cleanup that the request's rollout id and attempt match what seed recorded. The four sandbox benchmarks then register their task sandbox instead of holding it in their own dictionary, and the repeated-seed leak in two of them goes away with the dictionary. The single-worker constraint stays until stage 7 and must be stated as a deployment rule, not left implicit.

The evidence is in `nemo_gym/base_resources_server.py` lines 152 to 203; `nemo_gym/server_utils.py` lines 868 to 888; `nemo_gym/rollout_correlation.py` lines 109 to 118; `resources_servers/swebench/app.py` line 301; `resources_servers/terminal_bench_2_1/app.py` line 198; and `resources_servers/gymnasium/base.py` lines 96 to 110.

## Entering the rollout context does not select the token-capture prefix

Section 5.1 says the native executor enters Gym's rollout context around the harness call and that the shared client then applies the rollout prefix to model and resources calls "exactly as it does for a server-hosted agent". The goal is that an in-process harness gets the same capture correlation as a server-hosted one without an HTTP hop.

The shared client does less than that. It prefixes resources calls and model calls only when observability is enabled, and it never emits the training-token-capture variant. The training variant is chosen by the agent server, not the client: `url_path_for_run` builds the prefix with a `token_capture` flag that comes from configuration, the global `token_id_capture.enabled` together with either `all_agents` or the agent's own `token_id_capture` flag, and the agent posts to that explicit path on its self-call. NeMo RL enables capture globally with `all_agents` true. A native harness that relies on the context variable alone would post to the plain model route during training and capture nothing. The same gap applies to the OpenCode path, where the executor writes the model URL into the CLI's configuration.

The fix is for the executor, not the context variable, to own the model path. The executor computes `rollout_path_prefix(capture_key, token_capture=enabled)` from processor configuration that mirrors today's agent fields, and hands the native harness the explicit model path, or writes it into the CLI configuration, in the same way the agent's `url_path_for_run` does now. Entering the rollout context remains useful for resources calls and observability. Section 5.1 should say both.

The evidence is in `nemo_gym/server_utils.py` lines 489 to 527 and 1203 to 1208, and `nemo_gym/base_responses_api_agent.py` lines 104 to 178.

## The behavior contract reaches about two thirds of the agents

Section 5.1 makes `responses()` the behavior contract and section 4.5 fixes the single-agent protocol as seed, then behavior, then verify on the resources server. The goal is that every agent becomes a harness and every benchmark keeps its verifier.

The inventory at upstream `main` has 46 agent implementations. Nine of them raise `NotImplementedError` from `responses()` and do all their work in `run()`, including the Harbor agents, the mini SWE agents, OSWorld, tau2, and pinchbench. Fourteen never post to `/verify`: the gymnasium, aviary, and toolsandbox agents drive a `/step` loop where the environment scores as it goes, and the anyswe and swe_agents agents compute reward locally. Only one agent subclasses `simple_agent`; seven others carry a verbatim copy of its loop.

For the `responses()` family the migration is direct. For the nine run-only agents there is no behavior endpoint for a remote executor to call and no loop to extract without rewriting. For the step-based agents the fixed seed, behavior, verify order does not describe the episode, since the environment's step reply is the verification. For the locally graded agents the design's rule that the processor never asks the harness to verify blocks them unless reward computation moves into a resources server.

The fix is to state the migration classes in section 7 and to size them. The `responses()` family goes through the native executor. Agents with a behavior endpoint but no extractable loop go through the remote executor. Run-only agents get a thin behavior endpoint added, with the CLI-driving code kept as is, before they can be reached. Step-based environments are a second concrete processor whose protocol is reset, step until done, read the environment's score; that is the same shape the design already uses for user simulation and it should be named as such. Locally graded agents move their scoring into a resources-server verify, which is a benchmark change to schedule, not a framework change. The design should say plainly that the single-agent processor covers the first two classes and that the others are scheduled work.

The evidence is the `responses()` and `run()` implementations under `responses_api_agents/`, for example `harbor_agent/app.py` lines 236 to 237, `mini_swe_agent_2/app.py` lines 772 to 773, `gymnasium_agent/app.py` line 166, `anyswe_agent/app.py` line 690, and `labbench2_vlm_agent/app.py` line 103.

## The remote executor must round-trip the resources cookies

Section 5.2 says a remote executor forwards the Responses request to a behavior-only endpoint, and section 5.6 says it applies the rollout prefix across that boundary. The goal is that agents kept as servers work behind a processor without change.

The processor now holds the resources cookies. A remote agent that calls resources tools with no cookies lands in a fresh session on any stateful benchmark, because the session id is whatever the cookie says or a new UUID. The design says nothing about cookies on this path.

The mechanism already exists on the agent side. The agent's behavior endpoint takes the caller's cookies from the request, forwards them on every tool call, replaces its copy from each tool reply, and sets every accumulated cookie on its own HTTP reply. The agent's `run()` reads those cookies off the reply and sends them to verify. The cookie names are per server, `ClassName___instance`, so model and resources cookies never collide by name; the risk is replacement dropping unchanged keys, not name collision.

The fix is two methods on the resources-session client, `export_cookies` and `merge_cookies`, and a remote executor that sends the exported cookies with the behavior request and merges the reply's cookies back. The reply carries model and resources cookies mixed, as it does today, so on this path the processor merges the union, which is exactly what `run()` sends to verify now. The clean separation of model and resources cookies is a property of the native executor only, and section 5.6 should say so. A remote agent that crashes mid-episode loses cookies set by its tool calls, but the seed cookie still names the session, so cleanup can still release state.

The evidence is in `responses_api_agents/simple_agent/app.py` lines 252 to 297 and `nemo_gym/server_utils.py` lines 863 to 888.

## Provider connectivity is a three-way split, and the registry cannot express it yet

Section 5.7 classifies providers as reconnectable from anywhere or process-local, sends the second kind through a sandbox server, and offers colocating owner and operator in one process as an alternative. The goal is to use the sandbox server only when a provider cannot hand a sandbox to another process.

The classification is wrong for the providers people run locally, and the registry has no field to hold it. Docker runs a fresh `docker exec` subprocess per command against a container name. Apptainer runs a fresh `apptainer exec instance://name` per command. The local provider runs commands in a host directory. None of these hold anything in memory that matters, and any process on the same host can address them by that name. They lack only `serialize_handle` and `connect`. OpenSandbox and E2B implement both and reconnect from anywhere, though OpenSandbox's connect needs the provider's connection configuration on the connecting side as well. Fargate is the one provider whose handle is truly bound to its process, through an SSH tunnel and an HTTP session it owns. Daytona declares a `connect` that takes a bare string instead of a descriptor and no `serialize_handle`, so the runtime check for a connectable provider rejects it. The provider registry stores only name to class and carries no capability metadata, so `connection_scope` in the design has nowhere to live. There is no sandbox server and no remote provider in the tree; both appear only in docstrings, and pull request #2085 is their only implementation.

Colocation is not available either. Gym runs the resources server and the processor as separate processes by construction.

Two more facts change the facade design. A connected sandbox gets no ownership marker, and `stop()` calls both `provider.close(handle)` and `provider.aclose()`, so a borrower that reaches `stop()` tears down its provider client as well as the container. The facade must hide both, and disconnect must close only client-side state.

The fix is a three-valued scope recorded in the registry: reconnectable anywhere, reconnectable on the same host, process-bound. Docker, Apptainer, and local gain `serialize_handle` and `connect`, with a descriptor that carries the container or instance name, working directory, shell, and a host identifier, and a connect that fails fast on the wrong host. Direct connection is used whenever the scope allows it in the deployment's topology; the decision is made at configuration validation, not per episode. A proxy is required only for process-bound providers and for same-host providers when the two servers are on different nodes, and it can be either the standalone server from pull request #2085 or the same lease-and-operate routes mounted in the resources server. The brokered connection record embeds pull request #2085's `SandboxRef` rather than a lookalike. Daytona's signature is fixed or the provider is documented as not connectable.

The evidence is in `nemo_gym/sandbox/api.py` lines 521 to 565; `nemo_gym/sandbox/providers/base.py` lines 368 to 386; `nemo_gym/sandbox/providers/registry.py` lines 96 to 120; `nemo_gym/sandbox/providers/docker/provider.py` lines 179 to 184 and 449; `nemo_gym/sandbox/providers/apptainer/provider.py` lines 153 to 159 and 609; `nemo_gym/sandbox/providers/daytona/provider.py` line 899; `nemo_gym/sandbox/providers/ecs_fargate/engine.py` lines 583 to 584 and 927 to 949; and in pull request #2085 `nemo_gym/sandbox/ref.py` lines 39 to 53.

## An executor-owned workspace needs the same workspace record on the verify request

Section 5.3 allows an executor-owned harness workspace when the environment can verify from the Responses reply "or from another explicit transfer contract", and section 8 defers a processor-owned task workspace to stage 6. The goal is to support the pattern where the agent builds in its own sandbox and the environment grades the result.

That transfer contract is not defined, and the verify request has no field for it. The pattern is live in vibench: the agent tars the app inside its sandbox, downloads the tarball to a directory shared with the resources server, stops the sandbox, and passes the host path in the verify request through the extra-fields allowance. The resources server checks the path stays inside the shared directory and unpacks it. That works on one host only, puts vibench's tar excludes in the agent, and passes a host path across a service boundary, which is exactly what section 5.10 rules out.

The fix reuses the design's own record with the roles reversed. The executor keeps its workspace alive through verification. The verify request gains `workspace: SandboxWorkspace | None`, and the owner literal gains `harness_executor`. The resources server connects as a borrower, runs its own extraction in the workspace, grades in its own verifier sandbox, and disconnects. The executor stops the workspace after verify returns, on every exit path. In the ordering list of section 4.4, the stop for an executor-owned workspace moves after verify. Startup validation pairs an executor-owned deployment only with a resources server that declares it verifies from the reply alone or from a borrowed workspace. Vibench's tar command moves into the vibench verifier, and the harness does nothing benchmark-specific. The provider constraint of the previous section applies in this direction too.

The evidence is in `responses_api_agents/vibench_agent/app.py` lines 236 to 322 and `resources_servers/vibench/app.py` lines 114 to 137 and 172 to 174.

## TaskSet conflicts with the task-data and dataset conventions Gym already has

Section 6 defines a core `TaskData` model that rejects unknown fields, a `TaskSet` registry, and an `EnvironmentConfig` with accepted task sets, and section 6.5 maps `task_source` onto them. The goal is a typed boundary between agent-visible and verifier-only data and routing that does not read executable names from rows.

Gym already has a task-data convention, and it is the opposite shape. There is no core `TaskData` class. Each server ships a `task_data.py` that exports a `TaskData` symbol, loaded into a type adapter at collate time. The documented policy is that extra fields are allowed by default, `forbid` is opt-in, and `ignore` is banned. Reserved row keys are `responses_create_params`, `agent_ref`, `task_source`, `_ng_task_index`, and `_ng_rollout_index`, and a normalizer splices `verifier_metadata` and `task_data` sub-objects up to the top level. Collate strips a legacy `agent_ref` and stamps `task_source`; dispatch resolves `task_source` back to an agent through the dataset's declaring instance. Datasets are declared on the agent block in 155 configs and on a resources server in none. `EnvironmentManifest` already exists with `agent_server`, `datasets`, `rollout_driver`, `grading_mode`, `session_model`, `sandbox`, and `state` fields, and forbids extras.

So the design's `TaskData` shadows a name every server already exports with a different policy, its `EnvironmentConfig.accepted_task_sets` duplicates the manifest's `datasets`, and its assumption that datasets live on the resources server describes zero in-repo configurations. Section 6.5's `TaskData.input` field does not exist on its own model; the input lives inside `responses_create_params`.

The fix is to take Foundation 3 out of this proposal and file it as its own, referencing this one. What this proposal needs from task data is small and can be built on what exists: the resources-session adapter splits a row into `responses_create_params`, `resources_data`, and `agent_data` using the server's own `task_data.py` schema, with a marker on the fields the harness may see. The manifest gains the fields this proposal actually introduces, the processor protocol and the workspace requirement. Dataset declaration stays where it is. The `task_source` to `agent_ref` resolution at dispatch is already the routing seam the design wants; it needs a processor name, not a new registry.

The evidence is in `nemo_gym/task_data.py` lines 30 to 68 and 131 to 153; `nemo_gym/train_data_utils.py` lines 386 to 400 and 873 to 875; `nemo_gym/global_config.py` lines 1652 to 1700; `nemo_gym/config_types.py` lines 443 to 466, 533 to 548, and 615 to 620; and `nemo_gym/environment/manifest.py` lines 185 to 265.

## No deadline reaches the processor from RL, and cancellation has one carrier

`EpisodeRequest.deadline` is optional and the processor clamps downstream timeouts to it. The goal is that an episode never outlives its caller's interest.

NeMo RL sends no deadline and no timeout to Gym. It keeps its own group deadline, and when a row does not come back it re-dispatches the row under a new attempt id while the old episode keeps running in Gym. Nothing in the design bounds that orphan except harness-level timeouts, and the OpenCode main command timeout is three hours. The one cancellation signal that does reach Gym is client disconnect: the cancellation middleware cancels an in-flight request when the caller's connection drops, and it is registered last so it wraps the whole stack. The design does not mention it.

The fix is two lines in section 4.4 and one in 7.2. Name the client-disconnect middleware as the cancellation carrier for compatibility callers, and require the processor's episode scope to unwind on it. Give the compatibility translator a processor-configured default deadline when the body has none, so an orphaned attempt is bounded even before RL learns to send one. Note for the RL owners that sending a deadline derived from `rollout_timeout_s` is the native-path change that closes the gap.

The evidence is in NeMo RL `nemo_rl/experience/rollout_manager.py` lines 1228 to 1278 and 2131 to 2150, and in Gym `nemo_gym/server_utils.py` lines 750 to 780 and 1034.

## Lifecycle hooks and worker semantics are assumed, not present

Section 4.4 has the processor validate its bindings at startup and drain active episodes at shutdown through startup and shutdown hooks. The goal is fail-fast configuration and clean shutdown.

`SimpleServer` exposes no lifecycle hooks. It does not pass a lifespan to FastAPI and the only subclass hook is `setup_webserver`. Three places wrap the lifespan after the fact, and shutdown work today is a `finally` around the uvicorn call. `num_workers` above one switches uvicorn to import-string mode with that many worker processes on one port; the opencode agent config sets four. Each worker has its own admission counter, its own executor state, and its own copy of whatever the design calls process-scoped.

The fix is to add a lifespan to `SimpleServer` with `startup` and `shutdown` methods that subclasses override, as part of stage 1, and to state that everything the design calls process-scoped is worker-scoped. Capacity visible to RL is per worker times workers. The single-worker rule for resources servers with in-memory sessions, from the session section above, must be written next to this.

The evidence is in `nemo_gym/server_utils.py` lines 818 to 823, 927 to 938, 1000 to 1105, and `responses_api_agents/opencode_sandboxed_agent/configs/opencode_sandboxed_agent.yaml` line 5.

## The document and the branch need the same care as the contracts

The design is 19,400 words across 12 sections and an appendix. Commit `ebe52e96c` removed the questions page and the pairings note and rewrote the overview, so the branch now has one proposal, one guide, and the RFC snapshot. That is the right shape. Two mismatches remain. The overview names a `CommandSession` contract that the design does not define; the design's facade is `HarnessSandbox` and its implementation is `HarnessAsyncSandbox`. Appendix A is now the only record of the current OpenCode pairing behavior, which is fine, but it belongs in a companion rather than the normative document. Commit messages on the branch still have no bodies, and the history has cycled between five thousand and twenty thousand words twice.

The naming fix is mechanical. Moving the evidence is optional editorial work: the proposal author has chosen to keep the worked examples and current pairing appendix with the contracts because those examples are the proof that the boundaries preserve current behavior. Give each commit a body that says what changed since the previous draft.

## Resolution in the revised proposal

The revised proposal adopts the server-category, single binding, HTTP status, additive rewards, response preservation, resources-session, token-capture, migration-class, remote-cookie, provider-scope, deadline, worker-lifecycle, naming, and TaskSet findings.

It changes the executor-owned verification recommendation after ownership review. A task workspace now always belongs logically to the resources-server environment. A sandbox service may retain the physical provider handle and issue capabilities, but the processor never owns or transfers task state. An executor may create a separate harness-only workspace only when verification depends solely on the returned response. Vibench and GDPVal therefore migrate their benchmark-specific harvesting into resources and keep the task workspace alive there through verification instead of reversing the workspace handoff.

Wire compatibility and legacy passthrough are now separate mechanisms. A migrated `SingleAgentEpisodeProcessor` translates and projects old wire shapes. An unmigrated agent remains behind `LegacyAgentRunProcessor`, which forwards the whole episode-level call verbatim until that agent has a behavior-only boundary.

The revised proposal also removes RL-oriented `policy` role names from foundational examples. `SingleAgentEpisodeProcessor` binds `agent`; user simulation binds `assistant` and `simulated_user`; other processors choose protocol-specific role names and define their ordering in `process()`.

## What stands after these changes

With the changes above, the proposal keeps every decision that makes it worth doing and drops the parts that fight the code. The processor runs under the existing agent-server category with extracted behavior and a framework-owned envelope. One deployment binds one harness, and RL routes and sizes by that name. Failures travel in the HTTP status with the body as detail. The verifier's reply keeps everything consumers read today. The resources server gains a session table and a cleanup route so the design's cleanup guarantee has a receiver. Environment-owned task workspaces are handed to harness executors through serialized operate-only records, connected directly whenever provider scope and topology allow, and brokered only when they do not. Task data continues through each server's existing adapter. The task-set redesign proceeds separately on its own evidence.
