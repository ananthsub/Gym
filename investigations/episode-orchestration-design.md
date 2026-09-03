# Episode orchestration: processors, runtimes, and sandbox authority

Status: design draft 2, 2026-09-02. Builds on issue #2159 (Felipe Frujeri's episode-processor proposal, closed without an implementing PR) and PR #2085 (sandbox server, unmerged). Evidence is from this checkout (`session-state-prototype` at `996a10f9c`) cross-checked against `upstream/main` at `9287fb779` and the NeMo RL checkout at `~/dev/RL` (`9f166d8b0`). Draft 2 incorporates an adversarial code review of draft 1; Appendix C lists each finding and what changed.

## 1. The problem

`/run` is hosted on the agent server. That single choice is the root of four separate problems.

**Every agent re-implements episode orchestration.** `SimpleResponsesAPIAgent` declares `run()` abstract (`nemo_gym/base_responses_api_agent.py:195`), so all 42 agent servers write their own seed → agent → verify sequence. Nineteen are copies of `simple_agent.run()` with local edits. Twelve never call `/verify` and compute reward themselves (anyswe, anyterminal, swe_agents, mini_swe_agent, mini_swe_agent_2, harbor, osworld, tau2, verifiers, pinchbench, gymnasium, vcqa). Eight implement `/run` but raise `NotImplementedError` from `responses()`. Three override `setup_webserver` and register `/run` directly, which bypasses the rollout-context wrapper and drops session middleware (`mini_swe_agent/app.py:104`, `mini_swe_agent_2/app.py:626`, `harbor_agent/app.py:227`).

**Every CLI harness is copied, then copied again for a sandbox.** Eight host-CLI agents (codex, claude_code, opencode, cline, kilocode, pi, openclaw, prime) are near-identical 600-line files, plus eight copies of one `setup_<tool>.py`. When a harness needs a sandbox a second agent appears with no shared code: `opencode_agent` (830 lines) versus `opencode_sandboxed_agent` (459 lines), and on upstream main `terminus_2_agent` versus `terminus_2_sandboxed_agent`. The two opencode variants differ in install method, trajectory extraction, and model routing; the host variant bypasses the Gym model server entirely, so it has no model-call capture.

**Sandbox lifecycle has no owner.** Twelve agents implement "start a sandbox, run a harness, collect, score" on five substrates. Nothing in the framework knows a sandbox exists. Three resources servers (swebench, and deepswe and terminal_bench_2_1 on upstream) hand the agent a `sandbox_handle: str` on an ad-hoc field; swebench sources it from a private attribute (`resources_servers/swebench/app.py:260`) and keeps a process-local dict that is never popped (`:190`). Both sides stop the same box, which is safe only because of a provider idempotency fix. `AsyncSandbox.serialize()` has one production caller (deepswe on upstream, at seed).

**Cross-process sandbox sharing is provider-specific and unenforced.** PR #2083 merged the `ConnectableProvider` capability and #2084 implemented it for OpenSandbox; e2b on upstream is also connectable. `scope=` is threaded through the API and discarded by every implementation. `AsyncSandbox.connect` checks nothing about which rollout a descriptor belongs to. Daytona defines `connect(sandbox_id: str)` with the wrong signature and would fail the protocol the day it grew `serialize_handle`.

Two things went right and the design keeps them. The rollout id is already the universal key: minted by the caller, stamped as `/ng-rollout/<id>/` on downstream URLs, deriving the resources-server session id, and keying token capture, model-call capture, and session-state storage. And the resources server already has the right contract for an environment: `seed_session`, tools, `verify`, `export/restore_session_state`, MCP exposure.

## 2. Roles

Five roles. Three exist today; two are new.

| Role | Owns | Component |
| --- | --- | --- |
| Environment | Task authority: dataset, seed, tools, state, verify, close; declares sandbox specs and sharing policy | `resources_servers/` (contract extended, nothing removed) |
| Agent | Policy behavior: one `/v1/responses` call that turns a task into a completed response | `responses_api_agents/` (`/run` becomes a compatibility shim) |
| Model | Stateless inference, capture, admission | `responses_api_models/` (unchanged) |
| Episode processor (new) | Episode orchestration: identity, seed, runtime provisioning, agent dispatch, harvest, verify, close, episode record | `episode_processors/` |
| Runtime (new) | Where an agent's harness executes and how it reaches each model server: host process or sandbox | `nemo_gym/runtime/`, a library |

The processor is the only component that sees the whole episode, so it is the only place lifecycle ownership can live. Agents never create sandboxes. Environments may own sandboxes for their own tools or, for existing environments, for the agent's workspace. Everyone else operates under a lease.

One processor instance serves one environment and any number of agents. This is what "multiple agents per processor for a given resources server" means concretely, and it is the runtime half of the config-level work already on upstream main: `task_source`, `agent_map`, `fan_out`, `--agent-type`, and `allowed_agents` (#2640, #2641, #2661, #2710, #2713, #2724) choose the agent; the processor runs it.

## 3. The episode contract

A processor runs one episode as eight phases. Each has a typed input and output and emits one entry in the episode record.

| Phase | Processor does | Talks to |
| --- | --- | --- |
| identity | Derive rollout id and attempt from the row (`execution_identity_from_run_body`), or mint one when the row carries none; enter `rollout_context`; fence stale artifacts of earlier attempts; detect `_ng_resume` | local |
| seed | `POST /seed_session` under the rollout prefix; receive an `EpisodeDescriptor`. Idempotent per (rollout id, attempt): a second seed for the same key returns the same descriptor | environment |
| provision | Build each runtime the agent declares; create processor-owned sandboxes; attach to environment-owned ones; compute one rollout-scoped URL per model server the agent declares, reachable from that runtime | sandbox provider |
| act | `POST /ng-rollout/<id>[/training-token-capture]/v1/responses` on the agent with the task params plus an `ng_episode` block; loop for step-style environments | agent |
| harvest | Run the environment's declared harvest (files, commands) inside the workspace runtime; collect the response and observations | runtime |
| verify | `POST /verify` with response, harvested artifacts, and a `SandboxRef` when the environment asked for one | environment |
| close | `POST /close_session` (the hook from PR #2612), always, in `finally` | environment |
| release | Destroy processor-owned sandboxes, release leases, write the episode record | sandbox provider |

Three rules make the phases work across processes. They came out of the review and are the load-bearing part of this section.

**Rule 1: identity is always on under a processor.** Today the resources-server prefix is applied only when observability is enabled (`server_utils.py:355-362`), so native agents' tool calls are unprefixed and rely on the session cookie to reach the right session. The processor always seeds inside `rollout_context`, always prefixes calls to the environment, and never forwards cookies. Agent servers gain `RolloutContextMiddleware` so a prefixed `/v1/responses` call sets the context on arrival and every tool call the agent makes carries identity headers; `add_session_id` then derives the session id from the rollout id (`server_utils.py:606-618`). This is a hard prerequisite of Phase 0, not a cleanup. The MCP session token minted at seed rides in `ng_episode.mcp.headers`, since for blackbox harnesses the MCP token, not the URL prefix, is the tool-side identity (`mcp_auto_exposure.py:684-696`).

**Rule 2: the processor addresses the agent by the prefixed path and chooses the capture segment from the agent's config.** A native agent reads its rollout id from the inbound path and copies the `training-token-capture` segment from it when building the model URL (`simple_agent/app.py:378-388`, `base_responses_api_agent.py:171-183`). Whether that segment applies is the agent block's `token_id_capture` flag combined with the run-level block, which is exactly how NeMo RL selects capture per agent today (`nemo_rl/environments/nemo_gym.py:737-740`). The processor reads the same flag from the global config. No capture setting moves.

**Rule 3: seed is idempotent, close is idempotent, and both are the environment's teardown contract.** The environment gets a teardown method. `close_session` (PR #2612) is called from the processor's `finally` on every path including agent failure and processor cancellation, and a `session_ttl_s` idle sweeper on the environment is the backstop for the path no call reaches. Environment-owned sandboxes are released in `close_session`, not in `verify`. This replaces five incompatible conventions in the tree (`aviary /close`, `newton_bench /end_session`, gymnasium `close_session` only on terminal, openenv inside `verify`, and nothing).

### EpisodeDescriptor (seed response)

Merges the `SessionDescriptor` from the swe_bench branch, the `SeedResult` from the external_harness branch, `MCPServerMetadata`, and the `env_session_id` from PR #2613. Every field is optional; today's empty `BaseSeedSessionResponse` is a valid descriptor.

```yaml
env_session_id: "..."                 # PR 2613: the environment's own handle, for correlation
sandboxes:                            # zero or more, each with exactly one owner
  - role: workspace                   # workspace | tools | desktop | service
    owner: processor                  # processor | environment
    spec: {image, workdir, env, resources, ports, ttl_s}   # when owner: processor
    ref: {SandboxRef}                 # when owner: environment (already created)
    sharing: fresh                    # none | live | fresh   (what verify needs)
harvest:
  files: [answer.txt]
  commands: ["git -C /testbed diff"]
mcp: {server_name, url_path, transport, headers}
egress: {env: {}}
turns:                                # multi-party only, see section 4
  protocol: single | step
verifier_metadata: {}
```

`sandboxes` is a list because one episode can legitimately hold a processor-owned workspace and an environment-owned tool box at the same time (litmus and ns_tools already pool tool boxes). `owner: environment` with a `ref` is the swebench and terminal_bench_2_1 pattern, kept because it exists in production and because environments that pool boxes must own them. `sharing: live` means verify receives an operate lease on the same box. `sharing: fresh` means verify grades a harvested artifact in a box it creates itself, and the workspace is released before verify.

### SandboxRef

The one type a sandbox handle uses to cross a process boundary: `{provider, descriptor, rollout_id, scope: owner | operate, workdir, expires_at}`. `descriptor` is whatever the provider's `serialize_handle` returns. `AsyncSandbox.connect(ref)` is the only way a second process attaches, and it must verify `rollout_id` against the sandbox's metadata label before returning a handle; today it checks nothing (`sandbox/api.py:410-422`). For providers with an external control plane (OpenSandbox, e2b, and Daytona once its `connect` signature is fixed), scope is advisory and enforced by convention: `stop()` on an operate ref releases and never destroys. This is a library change, not a new server. The `sandbox_handle: str` fields on swebench, deepswe, and terminal_bench_2_1 become `SandboxRef`.

### Verify request and episode record

Verify gains three optional typed fields: `artifacts` (the harvest output), `sandbox_ref`, and `runtime` (a descriptor of where the agent ran). Environments that grade the response ignore them. `sharing: live` implies `REVERIFY_MODE = UNSUPPORTED`, and reverification strips `sandbox_ref` and `runtime` before re-posting, because a persisted ref is single-use per attempt.

The episode record is the processor's structured output merged into the `/run` result: per-phase timing and outcome, runtime descriptors (provider, sandbox id, image), `env_session_id`, failure class per phase, and the observability bundle. Today `_ng_failure_class` is produced in five unrelated places; the processor becomes the single producer and each phase failure is classified at its boundary, which is what the error-boundary program (#2750) asks for.

## 4. Agents after the split

An agent implements `POST /v1/responses` and declares a runtime and its model servers.

```yaml
responses_api_agents:
  claude_code_agent:
    entrypoint: app.py
    model_servers: {policy: policy_model}          # one or more, each gets a rollout-scoped URL
    runtime:
      kind: harness            # native | harness
      harness: claude_code     # HarnessAdapter name
      sandbox_provider: opensandbox
      allow_unsandboxed: false
```

Three agent shapes cover the repo.

**Native agents** run a Python loop in the agent server process against the model server: simple_agent and its 18 clones, langgraph, remote_agent, the conversational simulator. Their runtime is the host process. Their loop does not change. Their `run()` is deleted once the base-class default lands. Tool-boundary commits stay inside this loop because the loop is where the boundary is.

**Harness agents** launch an external program. There is one `HarnessAdapter` per tool and one `harness_agent` server that dispatches on the adapter named in config. The adapter contract is the one from the external_harness branch: `runtime()` setup commands, `prepare()` config files, `launch()` argv and env, `parse()` stdout to output items. It runs through the Runtime, never through `subprocess` directly. The agent server process is an adapter host; only the harness runs inside the sandbox.

**Self-contained agents** bring their own environment loop and grading: harbor, mini_swe_agent, osworld, tau2, verifiers, swe_agents. They keep their current shape, declared as `integration_profile: external-agent-loop`, and the processor skips seed, verify, and close for them. Today this is accidental; the design makes it explicit.

### User simulation and multi-agent episodes

Two patterns exist and both keep working.

*User simulation as an environment endpoint.* The conversational simulator's agent loop asks the resources server for the next user turn at `/next_user_message` and the server's `should_continue` ends the episode (`conversational_tool_use/simulation/app.py:263-283`). That is a native agent whose environment happens to talk back. Nothing changes.

*Multiple model servers per episode.* tau2 drives the tau2 library with two rollout-prefixed model URLs, one for the agent and one for the simulated user (`tau2/app.py:119-146`). The provision phase therefore produces one rollout-scoped, runtime-reachable URL per model server the agent declares, not one. Capture and admission see two logical participants under one rollout id, which they already do today.

*Multi-agent episodes* (orchestrator plus workers, debate, two policies in one world) are the one shape the repo does not have and the design adds through the step processor: the environment's `/step` response names the next actor, the row or config lists the participants as agent refs, and the processor dispatches each turn to the named agent. Participants share the rollout id and the environment session; each carries its own token-capture selection by agent name, which is how RL keys capture today.

## 5. Runtime

`nemo_gym/runtime/` is a small library over `nemo_gym.sandbox`. A runtime is an `AsyncSandbox` plus three things the sandbox API does not give today.

1. **Reachability per model server.** `runtime.reach(url)` returns a URL as seen from inside the runtime. Five agents hand-roll this today (loopback rewrites, `docker_network: host`, hostname lookups, Fargate's SSH tunnel, OSWorld's forwarder). It becomes a provider capability: local returns the URL unchanged; docker with host networking returns loopback; OpenSandbox, e2b, and Daytona return a routable host; a provider without a route gets a per-rollout reverse proxy started by the processor. The rollout prefix is applied after reachability is resolved.
2. **Policy.** `allow_unsandboxed` defaults to false for harness agents. A harness agent configured with the `local` provider fails at startup unless the flag is set.
3. **Teardown and attribution.** Processor-owned runtimes are stopped in `finally`, every spec gets `ttl_s`, and attribution labels go on every provider, not only OpenSandbox, so run-scoped cleanup (#2559) works everywhere.

The host runtime is the `local` sandbox provider from upstream #2863. A harness agent has one code path: it calls `runtime.exec`, and the provider decides isolation. `opencode_agent` and `opencode_sandboxed_agent` collapse into one adapter with two YAML files.

## 6. Sandbox authority

### Use cases

| Use case | Example in tree | Workspace owner | Tool box owner | Verify needs | Descriptor |
| --- | --- | --- | --- | --- | --- |
| SWE-style, extract a patch, grade fresh | swe_bench branch; mbien's SandboxCliAgent; anyswe | processor | none | harvested artifact; verifier creates its own eval box | `workspace/processor/fresh` + `harvest.commands` |
| Terminal-bench style, inspect the live box | terminal_bench_2_1 + terminus_2_sandboxed (upstream); pinchbench | processor (or environment, today) | none | operate lease on the same box | `workspace/processor/live` |
| Verifier-only sandbox, agent agnostic | ns_tools pools, math_formal_lean, code_gen, litmus | none | environment | nothing from the agent | `tools/environment` |
| Environment-owned workspace | swebench, deepswe | environment | none | in place | `workspace/environment` with `ref` |
| Desktop or service VM with ports | osworld | processor | none | evaluator inspects live state over declared ports | `desktop/processor/live` + `spec.ports` |
| Whole interaction graded inside the image | pinchbench (grading in the benchmark image) | processor | none | harvest a result file | `workspace/processor/none` + `harvest.files` |
| Files downloaded, posted in the verify body | cvdp | processor | none | artifacts | `workspace/processor/none` + `harvest.files` |
| Multi-container task (compose sidecars) | harbor tasks with services | processor | environment (services) | live | two entries in `sandboxes` |
| Long-lived box across turns of one rollout | multi-turn SWE, gymnasium-style repos | processor | none | live at the end | same box across `act` iterations |
| Pooled or prewarmed boxes at RL scale | ns_tools `sandbox_pool` | environment | environment | none | `tools/environment`, pooled |

The first three are the ones the design must get right; the rest fall out of the same three fields (`role`, `owner`, `sharing`).

### Rules

- Exactly one owner per sandbox. The owner holds the `owner` scope and is the only party that destroys. Everyone else holds `operate`; `stop()` on an operate handle releases and never destroys.
- Processor-owned sandboxes are destroyed in the processor's `release` phase. Environment-owned sandboxes are destroyed in `close_session`, never in `verify`. Both have `ttl_s` and attribution labels as the crash backstop.
- A `SandboxRef` is bound to one rollout id and one attempt. `connect` refuses a ref whose rollout does not match the box's label. A ref is single-use per attempt; reverification never carries one.
- Seed is idempotent per (rollout id, attempt). A retried seed returns the existing descriptor instead of creating a second box (swebench today overwrites and leaks; deepswe on upstream stops the previous one; the rule replaces both).
- Sharing `live` implies `REVERIFY_MODE = UNSUPPORTED`.

### Conflicts and how they resolve

| Conflict | Resolution |
| --- | --- |
| Environment declares `owner: environment` for the workspace and the agent config also names a `sandbox_provider` | The descriptor wins. The agent's provider is used only for processor-owned sandboxes. The processor logs the ignored setting once at startup for that agent-environment pair. |
| Environment-owned tool box and processor-owned workspace in one episode | Two entries in `sandboxes`, each with one owner. No shared box. |
| Two seeds arrive for one rollout id (retry after a lost response) | Idempotent seed returns the first descriptor. The environment keys its registry by (rollout id, attempt), not by cookie. |
| Two processors configured for one environment | Config validation error, same shape as #2724's rule for two agents on one resources server with no pin. |
| Reverification against a live ref | Refused by `REVERIFY_MODE`; reverify strips `sandbox_ref` and `runtime`. |
| Processor crashes between seed and release | `close_session` is not reached; the environment's `session_ttl_s` sweeper and the sandbox `ttl_s` reap. The next attempt of the same rollout seeds fresh because the attempt index differs. |
| Agent stops a box it only operates | The operate handle's `stop()` releases the lease. Destroying requires the owner scope; a connectable provider that cannot enforce this documents it, and the framework never hands an owner-scope ref to an agent. |

### Connectable, not a sandbox server

The requirement is a connectable sandbox: a provider whose handle can be serialized in one process and reconnected in another. OpenSandbox and e2b already are; Daytona is one signature fix away. The sandbox server from PR #2085 is one way to make docker, apptainer, enroot, and the local provider connectable across processes, and it is the only way for those. It is not required for any provider with an external control plane, and the design does not build it in this program. Where a non-connectable provider is used, the constraint is simply that the party that created the box is the party that operates it, which is what `owner: environment` plus in-process verify already expresses.

| Provider | Connectable today | Notes |
| --- | --- | --- |
| opensandbox | yes | `scope` ignored; add the rollout-label check in `connect` |
| e2b (upstream) | yes | same |
| daytona | no, wrong `connect` signature | fix to accept a descriptor mapping |
| docker, local, apptainer, enroot, openshell, ecs_fargate | no | owner and operator must share a process; sandbox server is the future path if ever needed |

## 7. Identity, capture, and partial rollout checkpointing

### Identity and capture

Nothing in the rollout-identity machinery changes; it moves up one level and is switched on unconditionally under a processor (Rule 1). The processor's `/run` establishes `rollout_context` exactly as the agent's wrapper does today (`base_responses_api_agent.py:89-102`). Every downstream call carries the prefix and headers through `ServerClient.request` (`server_utils.py:355-384`). Token capture and model-call capture are unchanged: the processor addresses the agent by the prefixed path (Rule 2), harness agents get their rollout-scoped model URL from the runtime, and the model server recovers identity from the path as it does now. `rollout_correlation_enabled` from PR #2613 is the right knob; under a processor it is forced on.

### Partial rollout checkpointing

The session-state prototype on this branch and the checkpoint choreography note define the mechanism: the rollout id keys everything; after each tool step the loop owner commits a `ToolBoundaryRecord` after the environment has exported its state; restore selects the last resumable boundary, restores the environment, rebuilds the conversation, and re-enters the loop; sandbox-backed environments export a reconnect descriptor rather than state; blackbox harnesses have no boundaries and are marked non-resumable, with the model server holding their calls during a cut.

The design changes who does which step, and nothing else.

| Step | Today (prototype) | Under a processor |
| --- | --- | --- |
| Commit a tool boundary | agent loop (`simple_agent/app.py:313-370`) | whichever component runs the tool loop: the native agent for rlvr episodes, the processor for step episodes (gymnasium's commits, including boundary 0 with the reset observation, move with its loop) |
| Export environment state | agent posts `/ng-session/export` | same caller as the commit; sandbox-backed environments return a `SandboxRef` |
| Store | `<session_state_dir>/<rollout_id>/boundaries.jsonl`, agent and environment share the directory | three-party shared-filesystem contract (processor, agent, environment); unchanged layout |
| Resume intent | `_ng_resume` on the run body, relayed to the loop by a query param | `_ng_resume` on the run body to the processor; relayed in `ng_episode.resume` |
| Restore | agent posts `/ng-session/restore` inside `run()` | processor posts it under the prefix, then reconnects the workspace runtime with `AsyncSandbox.connect(ref)`; a 409 or a failed connect abandons and redispatches a fresh attempt |
| Rerun hygiene | agent calls `clear_rollout` | processor calls it, and the store gains an attempt fence: an append whose attempt is older than the fence is rejected, so a stale agent attempt cannot write after the processor cleared |
| Class C exact rewind | deferred | `SupportsSandboxPauseResume` from the pause/resume branch is the primitive; `serialize()` is the descriptor; a provider without pause makes the boundary non-resumable, explicitly |
| Control plane | components are model, agent, resources | `ControlCapabilities.component` gains `episode_processors`; the processor declares `stateless` unless it runs a step loop, in which case it declares the same export-restore capability an agent does today |

The RL side is unchanged: the SingleController still cuts at model-server quiescence, the NemoGym actor still redispatches rows with `_ng_resume`, and the token ledger is still keyed by rollout attempt.

## 8. Dataset ownership and routing

Upstream main has already moved the data model to where the processor design needs it, so the design adopts it rather than inventing a parallel scheme.

- Datasets live on the resources server block (#2724 decoupled layout; 154 resources-server configs declare `datasets:` upstream versus 37 agent configs). Self-contained agents declare their own datasets and their own `task_data.py` schema.
- Rows carry `task_source`, the name of the config instance that declared the dataset, normally the environment. Rows that still carry `agent_ref` behave byte-identically and emit a deprecation warning (#2713).
- The agent for a row is resolved in order: `agent_map` entry, row `agent_ref`, `task_source` naming an agent, `task_source` naming a resources server with exactly one referencing agent, else a pre-dispatch error. `fan_out={rs: [a, b]}` runs one dataset under N agents and stamps each in-memory copy. Benchmark datasets pin an `agent:` when the config is ambiguous.

The processor slots in with one rule: **the row's environment selects the processor.** The collector resolves `task_source` (or the `agent_ref` agent's `resources_server`) to an environment, then to the unique `episode_processors` block whose `resources_server.name` matches, with the same zero-or-two error shape as `resolve_dataset_agent`. An environment with no processor block gets an implicit default `rlvr` processor composed at config-parse time, the same way `--agent-type` composes agents. `_validate_agent_names` learns that a processor is a valid `/run` target. N agents on one environment is expressed with `fan_out` or `agent:` pins, exactly as upstream requires today; the processor does not relax #2724's ambiguity error, it consumes its result.

Materialized inputs keep carrying the resolved `agent_ref`, because RL and the gdpval orchestrator read it from rows. The processor reads the same field to pick the agent.

## 9. NeMo RL and other framework integrations

NeMo RL dispatches to Gym through Gym's own code: the NemoGym actor calls `RolloutCollectionHelper.run_examples` (`nemo_rl/environments/nemo_gym.py:748`), which posts `/run` to `row["agent_ref"]["name"]` (`rollout_collection.py:1042`). The actor also: stamps `_ng_rollout_id` on each row before dispatch (`:722-725`); selects token capture per row by `token_id_capture_enabled_for_agent(config, agent_ref.name)` (`:737-740`); clears and finalizes token captures by rollout id, out of band from `/run` (`:742-745`, `:781-784`); reads `response.output` items with `prompt_token_ids` and `generation_token_ids`, `responses_create_params.input`, `reward`, `reward_components`, `mask_sample`, and `instance_config` from the result; groups prompt-group completion by `agent_ref.name`; and starts Gym servers with `RunHelper.start` from a config it assembles (`:648-656`). verl routes rows the same way, by `agent_ref` (`fern/.../verl.mdx:33`).

The contract the design preserves, item by item:

| RL depends on | Status under the design |
| --- | --- |
| `run_examples(rows)` posts `/run` to `row["agent_ref"]["name"]` | Unchanged call. Inside `run_examples`, the collector resolves the processor for the row's environment and posts there; if none is configured, the agent's `/run` shim runs the same phases in-process. RL never learns the difference. |
| `agent_ref.name` on every row and result | Unchanged. Materialized rows carry the resolved agent; the result echoes it. |
| Per-agent token-capture selection by agent name | Unchanged. The processor applies the same flag when choosing the capture segment (Rule 2). |
| Token captures keyed by `_ng_rollout_id`, cleared and finalized out of band | Unchanged. The processor uses the id RL stamped. |
| `response.output` with token ids; `reward`, `mask_sample`, `reward_components` | Unchanged. The episode record is additive. |
| `/aggregate_metrics` by agent name | Unchanged. The agent shim keeps proxying; the processor also serves it. |
| `RunHelper.start` spawns every top-level config block with an `entrypoint` | Unchanged. `episode_processors` blocks are spawned the same way. Implicit default processors run inside the agent shim, so an RL config with no processor block spawns no extra process. |
| Control-plane capabilities per component | Additive: `episode_processors` joins the component list. |

The one thing RL gains without changing: an RL config that names a sandboxed harness agent gets processor-owned sandboxes with mandatory TTL and run-scoped cleanup, which is the current source of stranded boxes in training runs.

## 10. Processor types and configuration

Two processors cover the repo; both are thin because the phases live in `nemo_gym/episode/`.

- `rlvr_episode_processor`: seed → act → harvest → verify → close. Sparse outcome reward. Every current agent that calls `/verify` maps here.
- `step_episode_processor`: reset → (act → step)* → close. Dense reward from the environment. gymnasium_agent's loop moves here with its `_ng_step_request_id` idempotency and explicit `/close`; micro agents (return tool calls, do not execute them) become possible for any environment that exposes `/step`; multi-party turns dispatch on the actor named by `/step`.

A benchmark that needs something else composes phases into a custom processor, which replaces `rollout_collection_driver`.

```yaml
swebench_env:
  resources_servers:
    swebench:
      entrypoint: app.py
      sandbox_provider: opensandbox
      datasets: [{name: verified, type: benchmark, jsonl_fpath: ...}]

swebench_processor:                     # optional; omitted = implicit rlvr processor inside the agent shim
  episode_processors:
    rlvr_episode_processor:
      entrypoint: app.py
      resources_server: {type: resources_servers, name: swebench_env}

claude_code_swe:
  responses_api_agents:
    harness_agent:
      entrypoint: app.py
      resources_server: {type: resources_servers, name: swebench_env}
      model_servers: {policy: policy_model}
      runtime: {kind: harness, harness: claude_code, sandbox_provider: opensandbox}
```

Rows carry `task_source: swebench_env`. The collector resolves the agent by the upstream rules, the processor by the environment, and posts `/run` to the processor. `fan_out={swebench_env: [claude_code_swe, opencode_swe]}` runs both agents through the same processor.

## 11. Backward compatibility, sized

| Surface | Count | Change required | When |
| --- | --- | --- | --- |
| `/run` on agent servers | 42 agents; RL, verl, and every collector post here | None. `/run` stays as a shim that runs the same phases in-process when no processor is configured. | never removed in this program |
| Dataset rows with `agent_ref` | every committed dataset | None (upstream already treats them as legacy with a warning). | none |
| Environment configs | 26 `environments/`, ~150 resources-server configs upstream | None. A missing `episode_processors` block means an implicit default. | opt-in |
| Resources servers: seed and verify schemas | 102 servers | None. New fields are optional; `BaseSeedSessionResponse` empty is valid. Three servers replace `sandbox_handle: str` with `SandboxRef`. | Phase 3 for the three |
| Resources servers: `close_session` | 11 override `seed_session` and hold state; 5 cleanup conventions | Override the no-op hook. Existing `/close`, `/end_session` keep working until migrated. | Phase 0, incremental |
| Native agents | 19 simple_agent clones | Delete `run()` (optional; the override keeps working). Middleware is added by the base class. | Phase 0 |
| Host-CLI agents and sandboxed twins | 8 + 2 | Replaced by adapters under one `harness_agent`. Old directories kept one release as aliases. | Phase 2 |
| Self-contained agents | 6 | Declare `integration_profile: external-agent-loop`. No code change. | Phase 1 |
| Session-state store layout | `boundaries.jsonl`, snapshots | Unchanged; attempt fence added on append. | Phase 0 |
| Control plane | 3 components | `episode_processors` added to the literal. RL reads capabilities, does not enumerate them. | Phase 1 |
| RolloutCollectionHelper public API | `run_examples`, `run_from_config` | Unchanged signatures; processor resolution is internal. | Phase 1 |
| Manifest and validation | 1:1 agent-environment | Accept a processor block; N agents via existing `fan_out` and pins. | Phase 1 |

Nothing in the table is a breaking change for a user who does not opt in. The cost lands on maintainers of the ten CLI agents and the three `sandbox_handle` servers.

## 12. Integrating existing frameworks

| Framework | Today | Under the design | Easier? |
| --- | --- | --- | --- |
| NeMo RL | `run_examples` → `/run` by `agent_ref`; capture out of band | Identical call path; processor is invisible unless configured | same, plus sandbox lifecycle for free |
| verl, unsloth | rows routed by `agent_ref` to `/run` | identical | same |
| Harbor agents (harbor_agent) | Harbor owns the loop, the environment, and the verifier; Gym receives a trial result; `/run` registered directly with no rollout context | Stays `external-agent-loop`; gains rollout context from the shim; no other change | same |
| Harbor tasks with a Gym harness agent | Not possible: Harbor's environment builds from a task Dockerfile and Harbor's verifier runs `tests/` inside it; a Gym harness would have to be written as a Harbor `BaseAgent` | A `harbor_task` environment: seed turns a task dir into a `workspace/processor/live` sandbox entry, verify uploads `tests/`, runs `test.sh`, reads `reward.txt`, which is exactly what `terminal_bench_2_1` does today on upstream. Any Gym harness adapter then runs the task. The blocker is image build: `EpisodeDescriptor.spec` takes an image, no provider builds a Dockerfile except e2b's `build.py`; tasks must be pre-built, as the Terminus integration already requires. | yes, once tasks are pre-built |
| verifiers (Prime) | `verifiers_agent` loads the vf environment in-process, calls `run_group` with an unprefixed model URL, takes reward from the rubric; no capture, no identity | Stays `external-agent-loop` for multi-turn vf environments. For single-turn ones, a `verifiers_env` resources server can expose the dataset as rows and the rubric as `/verify`, and any Gym agent runs it. The design gives the slot; wrapping the rubric is per-environment work. | partially |
| OpenHands, mini-swe-agent | Library owns the loop inside Docker or Apptainer; grading in the agent | `external-agent-loop` unchanged, or a harness adapter when the tool can be launched non-interactively (mini-swe-agent can) | same or easier |
| tau2 | Library owns the loop with two model servers | `external-agent-loop`; provision yields two rollout-scoped URLs | same |

## 13. Migration

Each phase is independently shippable and leaves the tree working.

**Phase 0: base-class defaults and always-on identity.** Add `RolloutContextMiddleware` to agent apps. Force rollout correlation on for calls the base class makes. Give `SimpleResponsesAPIAgent.run()` a concrete seed → prefixed self `/v1/responses` → verify → close implementation using the optional descriptor and verify fields. Land `close_session` (#2612) and `env_session_id` (#2613). Add the attempt fence to the session-state store. Delete `run()` from the simple_agent clones. No new server type, no config change.

**Phase 1: `episode_processors/` server type.** Register the fourth server type (the `config_types` change from #2085 shows the shape). Processor resolution by environment in the collector; implicit default processor when absent; `_validate_agent_names` accepts processors; `episode_processors` in the control-plane literal; manifest accepts a processor block. The agent shim forwards to a configured processor. Move gymnasium's loop into `step_episode_processor`.

**Phase 2: harness adapters and runtime.** Land `nemo_gym/runtime/` (reachability per model server, policy, teardown) and `nemo_gym/harness/` (adapter protocol, shared install and process-group helpers). Port claude_code and opencode first, then codex, pi, cline, kilocode, openclaw, prime, terminus_2. Delete the sandboxed twins. Source material: the external_harness adapters, mbien's `SandboxCliAgent`, cmunley's `sandbox_agent`.

**Phase 3: sandbox authority.** `SandboxRef` as the wire type with the rollout-label check in `connect`; fix Daytona's `connect` signature; processor-owned lifecycle; `ttl_s` and attribution on every provider; `sharing: live` reverify rule. Convert swebench, deepswe, and terminal_bench_2_1 from `sandbox_handle: str` to `SandboxRef` in a `sandboxes` entry. Implement class C export for sandbox-backed environments on top of pause/resume.

**Phase 4: environments out of agents.** Move grading out of anyswe, anyterminal, swe_agents, pinchbench, cvdp, and vcqa into environments with `sharing: fresh` or `live`, using the swe_bench environment server on Felipe Frujeri's branch and `terminal_bench_2_1` as templates. Add `harbor_task`. Agents that wrap a whole external eval loop stay `external-agent-loop`.

## 14. Decisions to take now

1. **Processor placement: one per environment, implicit by default.** Assumed above. The implicit default is what keeps RL and every existing config unchanged.
2. **Sandbox server: out of scope.** Connectable providers are the requirement. Non-connectable providers keep owner and operator in one process. Revisit only if a docker-only deployment needs cross-process sharing.
3. **Harness agents: one `harness_agent` server with adapters.** Per-tool directories survive as adapter modules and YAML.
4. **Environment teardown: `close_session` from PR #2612, called by the processor.** Merge that PR as part of Phase 0.
5. **Phase 0 first.** It is prerequisite for everything else and deletes the most code.

## Appendix A: coupling points the refactor carries

- Rollout id is derived, not assigned: `_ng_rollout_id`, else `{task_index}-{rollout_index}` plus `-a{n}` (`rollout_correlation.py:77-113`). Six subsystems recompute it and must agree.
- Agent servers have no `RolloutContextMiddleware`; the `/run` wrapper is their only identity source (`base_responses_api_agent.py:89-102`).
- Native agents' tool calls are unprefixed unless observability is on and rely on the cookie (`simple_agent/app.py:134-135`, `:236-241`; `server_utils.py:357-362`).
- The capture segment on the model URL is chosen from the agent block's `token_id_capture` flag (`base_responses_api_agent.py:121-131`).
- `/ng-session/export|restore` require the rollout prefix (`base_resources_server.py:285-295`).
- `skip_verification` produces a synthetic reward (`simple_agent/app.py:503-508`, `global_config.py:476-483`).
- `/verify` response is `/run` response; `judge_failsafe` sets failure classes inside verify (`judge.py:103-108`).
- `/aggregate_metrics` is called by agent name (`rollout_collection.py:975-979`).
- Model-call capture is cleared before dispatch and merged after by rollout id (`rollout_collection.py:797-832`); token capture is read by the trainer out of band.
- A new process type must be a top-level config key with an `entrypoint` (`cli/env.py:408-423`).
- Upstream `_validate_agent_names` rejects a `/run` target that is not an agent (`rollout_collection.py:1822-1833` upstream).
- MCP tool identity for blackbox harnesses is the session token from seed, not the URL prefix (`mcp_auto_exposure.py:684-696`).

## Appendix B: prior art consulted

| Source | What it contributes | What it leaves open |
| --- | --- | --- |
| Issue #2159 (ffrujeri) | `episode_processors/` owning `/run`; agents as pure `/v1/responses`; rlvr and gymnasium processor types | runtimes, sandboxes, harness agents, verifier sandbox access, N agents per processor |
| PR #2085 (`upstream/ansubramania/sandbox-server`) | `SandboxRef`, owner/operate leases bound to rollout id, admission and TTL reaping | positioned as a server; here only the ref type and rules are taken |
| `upstream/ansubramania/blackbox-8-sandbox-server` | harness-agnostic orchestrator, `HarnessAdapter` protocol, `SeedResult` with harvest and sharing, four grading topologies | still an agent-server subclass hosting `/run` |
| `upstream/feat/sandbox-cli-agents` (mbien) | `SandboxCliAgent` lifecycle, capture proxy, trajectory selection, node install helper | grading inside the agent |
| `upstream/cmunley1/sandbox_agent` | run an existing agent's `responses()` inside a sandbox; model URL rewriting | copies `run()` |
| `upstream/ffrujeri/agent-env-sandbox` | `SessionDescriptor` with placement topology; environment-owned grading in a fresh box; `Task` model | every agent must branch on topology |
| `upstream/main` `terminal_bench_2_1` + `terminus_2_sandboxed_agent` | the live-sharing shape in production: environment creates at seed, agent connects, verify runs tests in place | `sandbox_handle: str`; verify destroys; declared stateless yet not reverifiable |
| `upstream/main` `deepswe` | `serialize()` descriptor returned at seed | at seed, not export |
| `session-state-prototype` (this branch), checkpoint choreography note | rollout id as the universal key; tool-boundary records; export/restore hooks; blackbox hold at the model server | class C restore |
| `upstream/hemild/feat-sandbox-pause-resume` | `SupportsSandboxPauseResume` for OpenSandbox | wiring into export/restore |
| PR #2612, #2613 | `close_session` + idle sweeper; `env_session_id`; `rollout_correlation_enabled` | adoption |
| upstream `local` provider (#2863), #2713, #2724 | host runtime behind the sandbox API; `task_source`, `fan_out`, datasets on resources servers | processor resolution |

## Appendix C: review findings and what changed in draft 2

| Finding | Change |
| --- | --- |
| A processor cannot post plain `/v1/responses`; the agent reads rollout id and capture mode from the inbound path, and capture mode is the agent's own flag | Rule 2: prefixed addressing; processor reads the agent's `token_id_capture` flag, as RL does |
| Native tool calls rely on the cookie because the resources-server prefix is gated on observability; agents have no rollout context | Rule 1: identity always on under a processor; middleware on agents is a Phase 0 prerequisite; cookies never forwarded; MCP token in `ng_episode` |
| Boundary commits do not "stay in the agent" for step environments; the store is a three-party shared-FS contract; nothing fences stale attempts | Section 7 table: commits belong to the loop owner; attempt fence on append; pause/resume as the class C primitive |
| Section 2 said `/run` is removed while Phase 1 kept a shim; RL and upstream validation hard-require an agent target | `/run` stays on agents as a shim for the whole program; collector resolves processors internally; `_validate_agent_names` learns processors |
| Upstream already routes by `task_source` with datasets on resources servers; N agents per environment collides with #2724's error | Section 8 adopts upstream routing; environment selects the processor; N agents via `fan_out` and pins |
| tau2 needs two model URLs; user simulation is an environment endpoint | Section 4: one reachable URL per declared model server; simulator unchanged; multi-agent via step processor |
| No rules for ownership conflicts | Section 6: single-owner rule, `sandboxes` list, idempotent seed, reverify rule, crash path, operate-only agents |
| Live sharing already exists in `terminal_bench_2_1`; Harbor needs image build; verifiers has no environment path | Sections 6 and 12 cite tb21 as the template; image build named as the Harbor blocker; verifiers path scoped to single-turn |
| `sandbox_handle` is on three servers; deepswe already serializes at seed; gymnasium needs `/close` and step idempotency | Corrected throughout; step processor carries both |
