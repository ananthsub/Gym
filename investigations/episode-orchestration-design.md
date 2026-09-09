# Episode orchestration: processors, runtimes, and sandbox authority

Status: design draft 5, 2026-09-09. Builds on issue #2159 (Felipe Frujeri's episode-processor proposal, closed without an implementing PR) and PR #2085 (sandbox server, unmerged). Evidence is from this checkout (`session-state-prototype` at `996a10f9c`) cross-checked against `upstream/main` at `9287fb779` and the NeMo RL checkout at `~/dev/RL` (`9f166d8b0`). Draft 2 incorporated an adversarial code review of draft 1 (Appendix C). Draft 3 adds two audits (Appendix D): harness-to-benchmark coupling across the ten CLI harnesses, and how every consumer opts into a sandbox; they produced the new sections 5 and 6 and the reordered migration. Draft 4 (same day) added a comparison with the public RFC on `ananthsub/public-gym-architecture-rfc`, prototype PR #3100, PR #2085, and the `upstream/ffrujeri/sandboxes` placement branch. Draft 5 (same day) replaces that comparison with a critical review (section 7) of `rfcs/gym-architecture.md`, now on this branch, with concrete findings against each artifact and the alternatives they lead to, and rewrites backward compatibility (section 13) around the NeMo RL main and verl recipe contracts as they stand today.

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

**Harness agents** launch an external program. They come in two shapes under one `runtime` block (section 5.6): a manifest harness, where a CLI tool is described in YAML and executed inside the runtime with no Gym install in the box, and a placed harness, where a Python agent that owns its loop is run unchanged inside the box. The ten CLI agents migrate to manifests; hermes, langgraph, and strands are placed; the sandboxed twins are retired by configuration either way.

**Self-contained agents** bring their own environment loop and grading: harbor, mini_swe_agent, osworld, tau2, verifiers, swe_agents. They keep their current shape, declared as `integration_profile: external-agent-loop`, and the processor skips seed, verify, and close for them. Today this is accidental; the design makes it explicit.

### User simulation and multi-agent episodes

Two patterns exist and both keep working.

*User simulation as an environment endpoint.* The conversational simulator's agent loop asks the resources server for the next user turn at `/next_user_message` and the server's `should_continue` ends the episode (`conversational_tool_use/simulation/app.py:263-283`). That is a native agent whose environment happens to talk back. Nothing changes.

*Multiple model servers per episode.* tau2 drives the tau2 library with two rollout-prefixed model URLs, one for the agent and one for the simulated user (`tau2/app.py:119-146`). The provision phase therefore produces one rollout-scoped, runtime-reachable URL per model server the agent declares, not one. Capture and admission see two logical participants under one rollout id, which they already do today.

*Multi-agent episodes* (orchestrator plus workers, debate, two policies in one world) are the one shape the repo does not have and the design adds through the step processor: the environment's `/step` response names the next actor, the row or config lists the participants as agent refs, and the processor dispatches each turn to the named agent. Participants share the rollout id and the environment session; each carries its own token-capture selection by agent name, which is how RL keys capture today.

## 5. Runtime as a first-class citizen

Today the sandbox API is opt-in per consumer. Sixteen servers declare their own `sandbox_provider`, `sandbox_config`, or `sandbox_spec` fields in five incompatible spellings (a bare name, `${sandbox}` interpolation, an inline mapping, `str` only, `dict` only); `nemo_gym/config_types.py` and `global_config.py` contain no mention of sandboxes, and the top-level `sandbox:` block survives config merge only because every server-type config sets `extra="allow"`. The ten host-CLI agents use no sandbox at all and hand-roll subprocess isolation ten different ways. The single place to fix this is not one place but one contract instrumented at three points.

### 5.1 One `runtime` block on every server

Add a typed `runtime` field to `BaseRunServerInstanceConfig` (`config_types.py:604`), the class that annotates `BaseServer.config` and is inherited by both `BaseResponsesAPIAgentConfig` and `BaseResourcesServerConfig`. One field lands on every agent and every resources server, which is what `skip_verification` and `token_id_capture` already do one level down.

```yaml
runtime:
  kind: sandbox                 # none | process | sandbox
  provider: opensandbox         # name of a sandbox: block, or an inline single-key mapping
  spec: {image, workdir, env, resources, ports, ttl_s}   # defaults; the environment's descriptor overrides per episode
  allow_unsandboxed: false      # harness agents only; lets kind fall back to process
  env_passthrough: [HOME, PATH] # allowlist for process and sandbox runtimes
```

The existing spellings are accepted by a before-validator that maps `sandbox_provider`, `sandbox_config`, and `sandbox_spec` into `runtime`, with a deprecation warning, so no config breaks. `resolve_provider_config` keeps its string-or-mapping fork but is called from exactly one place. Only litmus rejects unknown spec keys today; the typed block rejects them everywhere, which will surface latent typos as startup errors rather than silent behavior.

### 5.2 Three instrumentation points

| Scope | Where | What it does | Why here |
| --- | --- | --- | --- |
| Config | `BaseRunServerInstanceConfig.runtime` | One validated shape; alias mapping from the five legacy spellings | The only class every server's config inherits |
| Process | `SimpleServer.run_webserver` (`server_utils.py:912-1016`), next to `setup_telemetry`, `set_ulimit`, and the `finally` that already shuts telemetry down | Build the provider once per process, attach it as `self.runtime_provider`, `aclose()` it in the same `finally`; skip on `DRY_RUN` | The only code path every server goes through. `model_post_init` is not usable: 68 servers override it and 33 never call `super()`, including claude_code, codex, cline, kilocode, opencode, openclaw, pi, prime, terminus_2, mini_swe_agent_2, osworld, harbor, and stirrup |
| Rollout | The `/run` wrapper `run_with_rollout_context` (`base_responses_api_agent.py:91-99`) today; the processor's provision phase once processors exist | Enter a `RuntimeScope`: per-rollout workspace, env, `reach(url)` for each declared model server, the sandbox handle (created, or attached from the descriptor), teardown in `finally` | The only rollout-scoped choke point that already wraps every agent's `run()`. It also covers `/v1/responses` for the harness base class, which is where the CLI agents actually spawn their process |

Resources servers get the same three points. A server that pools tool boxes (ns_tools, litmus) builds them from `self.runtime_provider` and the same spec type; it stops re-implementing provider resolution and shutdown, and litmus's lifespan-scoped teardown becomes the default for every server.

### 5.3 Runtime kinds and defaults

| Kind | Meaning | Default for |
| --- | --- | --- |
| `none` | The agent's loop runs in the agent server process and calls tools over HTTP. No runtime object is created. | native agents (simple_agent and its clones, langgraph, remote_agent, conversational simulator) |
| `process` | A host subprocess with a per-rollout workspace, scrubbed env, process group, timeouts, and group kill. Implemented by the `local` provider. No security boundary. | harness agents only when `allow_unsandboxed: true`; resources servers that shell out (bigcodebench, nvarc) |
| `sandbox` | A provider-backed box with the same `exec`, `pty`, `upload`, `download`, `reach` surface. | harness agents; any resources server that runs untrusted code |

The rule the user asked for is therefore explicit: a native agent never gets a sandbox it does not need, a harness agent never runs on the host unless the config says so, and both are decided by one field the framework reads, not by code each agent writes. Subprocess isolation is `process`; it is a real kind, not the absence of one, so the ten host-CLI agents keep their current behavior under `allow_unsandboxed: true` while gaining the shared workspace, env, and kill semantics.

### 5.4 What the `local` provider still needs

The `local` provider on upstream main gives exec, upload, download, a per-handle workspace, a concurrency semaphore, timeouts, and group kill on timeout. For the `process` kind, and for the adapter-launch optimization in section 5.6, the host-CLI agents need six more things before they can run on it unchanged, and each is small.

| Need | Who needs it | Change |
| --- | --- | --- |
| Streaming stdout with timestamps | pi_agent reads events incrementally; claude_code emits `stream-json` | Implement `SupportsSandboxPty` on `local` (pipe mode); the harness base uses `pty.create` when the adapter asks for a stream |
| Stdin control | codex blocks on inherited stdin; stirrup holds a persistent shell | `stdin: inherit | devnull | pipe` on the pty spec; `devnull` default for exec |
| Env allowlist | stirrup's `env_passthrough`; swe_agents `--cleanenv` | `env_passthrough` on the runtime block, applied by every provider |
| PATH prepend | every CLI installs into a private prefix | `path_prepend` on the launch plan, composed into `spec.env` |
| Signal-death classification | pinchbench distinguishes OOM and preemption from failure | `error_type: killed` with the signal on `SandboxExecResult` |
| Cancellation kill | openclaw's SIGTERM salvage; prime's orphan sweep | `session.send_signal` on the pty session; the rollout scope kills the group on cancel |

With those, claude_code, codex, cline, kilocode, opencode, simple_strands, terminus_2, and vcqa's local backend move unchanged; pi and openclaw move once streaming and signals exist; prime, stirrup, and swe_agents keep provider-specific behavior behind the same interface (stirrup already is a PTY consumer, swe_agents needs bind mounts and a memory watchdog that belong on the apptainer and enroot providers, not in an agent).

### 5.5 Reachability and identity, once

Six places rewrite the model server URL so a box can reach it: `global_config.py:1219`, `vibench_agent/app.py:122`, `anyterminal_agent/app.py:456`, `anyswe_agent/app.py:648`, `deepswe/app.py:205`, `osworld_agent/sandbox_provider.py:230`, plus the `NEMO_GYM_SANDBOX_MODEL_BASE_URL` convention in eight configs. `RuntimeScope.reach(url)` replaces all of them: `none` and `process` return the URL unchanged, docker with host networking returns loopback, OpenSandbox, e2b, and Daytona return a routable host, and a provider with no route gets a per-rollout reverse proxy. The rollout prefix and the capture segment are applied after `reach`, in one place, so the identity rules in section 3 hold for every runtime kind.

### 5.6 Implementation: process placement

Sections 5.1 to 5.3 describe the contract. Two harness shapes implement it, both configured in YAML. A **manifest harness** (section 7.7) is a CLI described by image, install, launch template, model-endpoint env var, and output parser; the runtime executes it through `exec` and `pty` with no Gym install in the box, which is the Harbor-style contract and the default for a new CLI tool. A **placed harness** is a Python agent that owns a loop; the placement branch (`upstream/ffrujeri/sandboxes`, reviewed in section 7.4) supplies its mechanism, and this design adopts that mechanism with the four conditions in section 7.6 (no agent config in the box, a processor-held ref, a built and cached runtime image, an HTTP bridge when the box can reach the host):

- The agent's `runtime.kind: sandbox` makes `create_server` construct a host instead of the harness. The host runs the processor phases (seed, connect or create, execute, verify, close) and, in the execute phase, runs a worker inside the box that imports the harness class, builds its FastAPI app, and invokes `/ng-rollout/<id>/v1/responses` in-process with the seed cookies. The harness's own subprocesses, streaming, and signal handling run where they always did, now inside the box.
- Dependencies are installed from a snapshot of the checkout into a sandbox-local venv, or skipped for a prepared image. A per-task install is the cost of "unchanged harness"; a prepared image with `dependencies.enabled: false` is the production path.
- `runtime.server_urls` is the interim form of `reach(url)`; the `RuntimeScope` from section 5.2 subsumes it.
- Field mapping onto the block in section 5.1: `type: local` is kind `process`, `type: sandbox` is kind `sandbox`, `sandbox_source: environment` is a `sandboxes` entry with `owner: environment`, `sandbox_source: runtime` is `owner: processor`. The branch's validators (no `spec` when the environment owns the box, no placement override from inside the box) carry over.

What placement does not cover, and where the adapter path from draft 3 still applies: images without Python, per-CLI launch without a Gym install, and the `process` kind on the host. Those are the cases for `HarnessAdapter` and the `local` provider additions in section 5.4, and they become an optimization rather than a prerequisite.
## 6. Harness decomposition: what actually unblocks composing harness and environment

The audit asked one question of the ten CLI harnesses (opencode, hermes, openclaw, claude_code, codex, pi, cline, kilocode, prime, opencode_sandboxed) and simple_agent: what does it take to run each against an arbitrary resources server?

### 6.1 Finding: the harnesses are not coupled to benchmarks. They are missing capabilities.

Nine of the ten share one near-identical `run()`: seed with the whole row, self-call `/v1/responses`, verify with the row plus the response. No harness has a benchmark-specific config field, reads a dataset-specific metadata key, or requires a benchmark-specific verify field. No resources server requires an agent-specific field in its verify request. The eight harness-times-math pairings in `environments/` landed with zero Python changes (commit `233d3c01b` touches only YAML and READMEs), and upstream's `--agent-type`, `agent_map`, `fan_out`, and `task_source` already swap harnesses on a resources server without editing anything. For response-graded environments, decomposition is unblocked today at the config level.

What blocks it is four capability gaps, each shared by most of the ten:

| Gap | Who | Effect |
| --- | --- | --- |
| The seed response body is discarded; only cookies are kept | opencode, hermes, openclaw, pi, cline, kilocode, prime | MCP metadata, sandbox handles, and seeded files never reach the harness |
| No path to environment tools | the same seven plus opencode_sandboxed; only claude_code and codex render MCP metadata into the CLI config, and only in `run()`, not in `responses()` | Eight of ten harnesses cannot use a resources server's tools at all; the one MCP-exposing server in the tree is `example_mcp_weather` |
| No way to run inside a sandbox the environment owns | all ten except opencode_sandboxed | Pointing opencode_agent at swebench with the pairing guard bypassed scores a silent zero: the seed's sandbox is ignored, OpenCode edits an empty temp dir on the host, and verify diffs the untouched box |
| `/aggregate_metrics` computed locally instead of proxied | all ten CLI harnesses; only simple_agent proxies | Every CLI pairing with math_with_judge silently loses `pass@k` and `majority@k` |

Two smaller ones: nine collapse `responses_create_params.input` to the last user message (hermes is the only harness that preserves history), and `skip_verification` is declared on the base config but honored only by simple_agent.

The one harness that is coupled is `opencode_sandboxed_agent`: it hardcodes a SWE-bench image, requires five OpenCode-specific fields on its verify response, asserts single-message input, and mutates the verify response. It is the shape to retire, not to copy.

### 6.2 Is lifting `/run` sufficient?

No, and it is not what unblocks the response-graded half. Five mechanical changes do, and they can land this month without a new server type:

1. Read the seed response body in the seven harnesses that discard it.
2. Lift the MCP wiring from claude_code and codex into a shared helper that returns `{name, url, headers}`, and have each CLI render it into its own config format (OpenCode `opencode.json`, OpenClaw `openclaw.json`, Cline `CLINE_MCP_SETTINGS_PATH`, Kilo `kilo.json`, Pi, Hermes toolsets).
3. Pass the MCP config through `responses()` as well as `run()`.
4. Proxy `/aggregate_metrics` to the resources server, as simple_agent does.
5. Replace the twelve duplicated `environments/<harness>_<env>/` directories (byte-identical datasets stamped with a per-harness `agent_ref`, resources-server blocks copied verbatim) with the `config_paths` plus `_inherit_from` shape that `cline_math` and `kilocode_math` already use, one dataset per environment, rows stamped with `task_source`.

What lifting `/run` adds is that items 1, 3, 4, and `skip_verification` happen once in the base class or processor instead of ten times, and the "seed body discarded" class of bug becomes structurally impossible because the processor, not the harness, receives the descriptor. That is a maintenance argument, and a strong one, but it is not the unblocking change.

For the sandbox-graded half (SWE-bench, terminal-bench, deepswe, swebench_pro, cvdp), the blocker is the runtime contract in section 5, and there lifting `/run` is load-bearing: the environment's descriptor must reach something that can create or attach a runtime and hand it to the harness, and today that something does not exist. Concretely the harnesses need a typed handle on the seed response instead of `sandbox_handle: str` (swebench_pro already drifted to an unread `pty_session_id`), a "run my CLI inside this runtime" path (only opencode_sandboxed has one), sandbox authority out of the agent (opencode_sandboxed creates a box from a hardcoded image when the seed gives none), a workdir and env handoff (swebench documents that OpenCode opens a new terminal instead of reusing the seeded one, so the diff measures the wrong tree), and an artifact placement rule (opencode_sandboxed writes its transcript to `/tmp` because a `git add -N .` sweep would otherwise grade it). Sections 3, 5, and 8 are those five items.

### 6.3 What "compose opencode, hermes, openclaw with various resources servers" looks like after both halves

```bash
gym eval run --resources-server reasoning_gym --agent-type hermes_agent     # works today
gym eval run --resources-server math_with_judge --agent-type openclaw_agent # works today, loses pass@k until item 4
gym eval run --resources-server swebench --agent-type opencode_agent        # silent zero today; works after section 5 and a typed seed handle
gym eval run --resources-server example_mcp_weather --agent-type opencode_agent   # no tools today; works after items 1 and 2
```

`allowed_agents` stays the guard, and it should widen as the gaps close rather than be bypassed: bypassing it today yields the silent zero, not an error.
## 7. Critical review of the RFC, PR 3100, PR 2085, and the placement branch

Four artifacts overlap this design. This section reviews each against the code rather than adopting it, and section 7.7 collects the alternatives that came out of the review.

| Artifact | What it is | Status |
| --- | --- | --- |
| `rfcs/gym-architecture.md` (originally on `ananthsub/public-gym-architecture-rfc`, now merged onto this branch) | Public RFC: `episode_processors/` owning `/run` and the sandbox, agents owning only `/v1/responses`, class D integrations as standalone processors, metrics on the verify owner, a legacy processor for migration. Built on #1866, #2950, #2858, #2763 and prototype #3100 | draft |
| PR #3100 (Felipe Frujeri) | `nemo_gym.agents` and `nemo_gym.processors` packages, a `processors/` server type, `SingleAgentTurnProcessor`, a `<agent>__processor` sidecar generated at config load for `simple_agent` instances, `/run` and `/aggregate_metrics` routed to it | open |
| PR #2085 (issue #2082) | Sandbox server: a server that owns boxes and lends them by signed `SandboxRef` leases so non-connectable providers can be shared across processes | unmerged |
| `upstream/ffrujeri/sandboxes` | Runtime placement: `runtime` on `BaseResponsesAPIAgentConfig`; `create_server` swaps in a `SandboxedAgentHost` that runs the unchanged harness's `/v1/responses` inside the task sandbox through a worker after installing Gym and the harness there; `SandboxWorkspace` on the seed response; `/cleanup_session` | proof of concept |

### 7.1 What all four and this design agree on

A processor exists and owns `/run`. Agents own `/v1/responses` and nothing else. Metrics belong to the verify owner, grouped by (verify owner, agent) with `agent_ref.name` as the entry identity. A sandbox handle crosses a process boundary only as a serialized descriptor. The environment gets a teardown endpoint. Orchestration does not move to the resources server. Class D integrations (tau2, harbor, verifiers, pinchbench) should not be agent servers whose `responses()` raises; they become standalone processors, which replaces draft 2's "external-agent-loop agents".

### 7.2 The RFC

**R1. The routing-key change breaks the one contract external frameworks hold.** The RFC makes the processor the row's routing key and has the processor name the agent. NeMo RL main (`edf6b8c42`, 2026-09-09) calls `run_examples`, then reads `row["agent_ref"]["name"]` for prompt-group accounting and returns the resolved `agent_ref` to its caller (`nemo_rl/environments/nemo_gym.py:619-694`); the verl recipe raises if a dataset row lacks `agent_ref` (`verl-recipe/nemo_gym/agent_loop.py:291`) and labels results by it (`:216`). The RFC's own metrics section says `agent_ref.name` must stay the identity because downstream consumers index on it. So the routing key on rows and results has to stay `agent_ref`, and processor selection has to be a function the collector computes from it. The RFC's "the processor names the agent" inverts a dependency that two trainers have already built on.

**R2. The legacy processor in Phase 1 costs a process and a hop per agent instance and buys nothing until Phase 2.** "Everything routes through it immediately" adds one uvicorn process, one port, one health check, and one HTTP hop per agent for a call that returns the agent's `/run` verbatim. The RFC lists the process cost as accepted downside 3. The cheaper shape is the inverse: keep `/run` on the agent as a thin shim that runs the phase library in-process, and spawn a processor only where a config declares one. PR 3100's sidecar generation is the right mechanism for the declared case; it should not be mandatory for the undeclared one.

**R3. A single sandbox owner does not fit the repo.** "Neither the agent nor the resources server provisions or tears one down" is contradicted by litmus and ns_tools, which pool tool boxes across sessions and must own them, and by every verifier that grades in a fresh box it creates itself (swebench, deepswe, swebench_pro, code_gen). The RFC's accepted downside 2 acknowledges the second case and hand-waves the first. Ownership has to be per sandbox, with a role, which is what the `sandboxes` list in section 3 does.

**R4. `EpisodeContext` on the Responses create params pollutes the model-server contract.** The RFC declares the context "on `BaseRunRequest`, `BaseSeedSessionRequest`, `BaseVerifyRequest` and the responses-create params". The create params are the OpenAI Responses type that every model server accepts and that blackbox harnesses send verbatim; a Gym-only field there either has to be stripped before every model call or leaks into providers. The rollout-prefixed path and the identity headers already carry the context to the model server; the body field belongs on Gym-internal requests only, as `ng_episode` in section 3.

**R5. "No hooks, seven processors" overfits to today's inventory.** The RFC counts protocols (A single-request, B staged, C stepwise, D external) and mandates one processor class per protocol with no hooks inside `run()`. Class B is a declared list of stages and class C is a loop over `/step`; both are parameters of one processor, which the RFC itself notes "with more effort". A hook-free base class means every retry policy, timeout, and turn budget that the appendix classifies as "policy or data, not protocol" has nowhere to go except a subclass, which recreates the 38-way fork the RFC is trying to remove. The phase library in section 3 is the alternative: composable phases with typed inputs, and a processor is a composition, not a class hierarchy.

**R6. Phase 3 depends on the sandbox server, and does not need to.** The RFC gates the P0 benchmark migration (swebench, swebench_pro, terminal_bench_2_1, gdpval) on #2082 landing first. All four run on OpenSandbox today, which is already connectable. The dependency exists only for docker and apptainer deployments, and section 7.5 shows most of those do not need a server either.

**R7. The strongest insight in the RFC is under-used.** The competitive analysis identifies Harbor's minimal task contract, "you get a working directory, do anything, exit, then tests run", as the reason 50 harnesses work there without per-agent effort. The RFC then proposes nothing that gives Gym that contract. Section 7.7 does.

**R8. UC11 is named as the one property with no demonstrator and then left open.** `allowed_agents` refuses without explaining. The `capabilities` and `requires` declarations in section 7.7 are the concrete form.

**R9. Missing from the RFC entirely:** the NeMo RL and verl dispatch contract (section 11), partial-rollout checkpointing (section 9), multi-worker uvicorn on the resources server, and how token capture selection by agent name survives the routing change. The RFC also says "49 agent servers"; this checkout has 42 and upstream main 47, which is a reminder that the inventory moves faster than the design.

**What the RFC gets right and this design takes:** the four-object framing (work package, driver, harness, runtime) from Prime; standalone processors for class D; `/sandbox_spec` before seed so the environment can populate a processor-owned box; metrics on the verify owner grouped by (owner, agent); the rejection of orchestration-in-the-resources-server with its four reasons; the observation that `rollout_collection_driver` is a planner above the episode and out of scope.

### 7.3 PR 3100

**P1. The sidecar is generated only for `simple_agent` instances that name a resources server.** `normalize_simple_agent_rollouts` skips every other implementation, so the 40 other agents keep their own `/run`. That is fine as a first step and wrong as the general mechanism, because it keys generation on the implementation directory name rather than on the environment. Generate per environment instead: one sidecar for a resources server, serving whichever agents route to it.

**P2. `_processor_server_name` is called on every `/run` and every `/aggregate_metrics`,** and the aggregate route goes to the processor, which proxies to the resources server. That is two hops for a call the collector could send to the verify owner directly (the RFC's own recommendation). Resolve once at preprocessing, not per row.

**P3. `BaseProcessor` copies `_capture_correlation_enabled` and `_token_id_capture_enabled` from the agent base** and reads `self.config.token_id_capture` on the processor config. Capture selection is per agent (NeMo RL keys it by agent name). A processor serving two agents needs the flag per agent, which is Rule 2 in section 3: the processor reads the target agent's flag, not its own.

**P4. Trajectory plumbing rides on a private key.** The processor pops `INTERNAL_TRAJECTORY_KEY` from the response JSON and infers `task_id` from four candidate row extras. That is the same untyped side channel the design is trying to remove; the episode record in section 3 is the typed home.

**What PR 3100 gets right:** the package split (`nemo_gym.agents`, `nemo_gym.processors`), transparent routing inside `run_examples` so NeMo RL and verl need no change, keeping `SimpleResponsesAPIAgent` untouched for compatibility, and a concrete `run()` that is the seed → agent → verify sequence.

### 7.4 The placement branch

**S1. Secrets are shipped into the sandbox.** `_worker_payload` builds the worker config from `self.config.model_dump(mode="json")`, which includes every field of the agent config. `opencode_agent` carries `openai_api_key` and `opencode_config.provider.*.apiKey: ${nvidia_api_key}`; codex and claude_code carry provider keys in the same way. The payload is written to a `mkdir -m 700` directory inside the box the untrusted harness runs in. The comment says "never the verifier config, task answers, provider credentials" and that is true of the global config subset, not of the agent config. The box must receive only endpoints and per-rollout tokens; the rollout-scoped model URL already needs no key.

**S2. The resources-server workspace registry is process-local and refuses retries.** `_session_sandboxes` is a `PrivateAttr` dict on the server instance; `register_sandbox_workspace` raises "Session already owns a sandbox workspace" on a second seed for the same session and stops the new box. With `num_workers > 1` on the resources server, seed, verify, and cleanup can land on different uvicorn workers and miss the dict (the doc admits "seed, verify, and cleanup must reach the same resources-server process"). A retried seed after a lost response fails outright. The processor-held ref in section 7.7 removes the registry.

**S3. The outer host answers 400 on `/v1/responses`.** Anything that drives the agent by `/v1/responses` directly, which includes `remote_agent` composition, reverification paths, and any parent agent, loses the agent when placement is on. Placement should make `/v1/responses` on the host provision a runtime and forward, or the agent should declare that its `/v1/responses` requires an episode context and fail with that message.

**S4. Per-task dependency install with no cache is not an RL-scale path.** Each rollout uploads a source tarball, bootstraps `uv`, creates a venv with a downloaded Python, and installs Gym plus the harness. That is minutes of wall clock and a network dependency on PyPI, astral.sh, and the harness's distribution per rollout. The branch's answer is a prepared image with `dependencies.enabled: false`, which is right, but nothing builds that image. The source snapshot also includes untracked files (`git ls-files --others`), so two runs from the same commit can ship different code.

**S5. The worker runs the harness's whole FastAPI app.** It calls `agent.setup_webserver()` and enters the app's lifespan inside the box. For the CLI agents that is harmless; for agents whose `setup_webserver` starts telemetry, proxies, or background tasks (osworld's asset prewarm, browsecomp's tokenizer client, tau2's data clone) it runs those inside every sandbox. The bridge also bypasses HTTP, so the session middleware, the rollout-context middleware, and the traced endpoints run with a synthetic ASGI scope rather than a real request, and there is no telemetry span from inside the box.

**S6. Placement is hosted in the agent server, which is the wrong long-term home and the right short-term one.** `SandboxedAgentHost.run()` is the seed → connect or create → execute → verify → cleanup sequence, and `create_server` swaps it in for the harness. It duplicates the processor's job in the agent process, which is acceptable as an interim exactly because it is `run()`-shaped: it moves into the processor without changing shape (section 15, Phase 2).

**S7. Smaller points.** `runtime` lives on `BaseResponsesAPIAgentConfig`, so resources servers that shell out get nothing; the block belongs on `BaseRunServerInstanceConfig`. `/cleanup_session` competes with PR #2612's `close_session`; one name. `server_urls` is a manual reachability table per server; `reach(url)` from section 5 subsumes it. `sandbox_source: environment` rejects any agent `spec`, which is correct, and `sandbox_source: runtime` rejects an environment-supplied workspace, which is also correct; both validators carry over. The `_harness_import` fallback assumes `responses_api_agents.<dir>.<entrypoint>` and fails for out-of-tree agents discovered through `NEMO_GYM_EXTRA_ROOTS`.

**What the placement branch gets right and this design keeps:** the unchanged-harness principle, the worker bridge for Python harnesses, the typed workspace on the seed response with a `serialize()` descriptor rather than a bare id, cleanup in `finally` on every path, the placement validators, and the demonstration that `opencode_sandboxed_agent` is a configuration of `opencode_agent`.

### 7.5 PR 2085

Reviewing my own proposal with the same standard.

**B1. The registry is in-memory and single-process.** `_entries` lives in one server process; a restart orphans every box (TTL is the only recovery), and `num_workers > 1` splits the registry. A sandbox server that holds state has to persist it or derive it from the provider (labels, names), not from a dict.

**B2. Leases have a 24-hour `max_age` and no renewal,** which is shorter than a long RL run and longer than a rollout; neither bound is the right one. Renewal belongs with the `SupportsSandboxRenew` capability on the `terryk/sandbox-renewable` branch.

**B3. It adds a hop to every `exec`.** Every command the harness runs crosses HTTP twice. For OpenSandbox and e2b that hop is pure cost, which is why the server must never sit in front of a connectable provider.

**B4. Most "non-connectable" providers are connectable on the same host without a server.** A docker container is addressable by id from any process that can reach the daemon; an apptainer instance by name; a local workspace by path. Only the `raw` handle is process-bound, and only because the provider stores a live client object in it. Implementing `serialize_handle` and `connect` on docker, apptainer, and local (same-host reconnect by id or path) covers the workstation and single-node cases that make up all in-repo uses of those providers. The server is then needed only for cross-host sharing of host-local providers, which no in-repo benchmark does.

### 7.6 What the review changes in this design

| Draft 4 said | Draft 5 says | Because |
| --- | --- | --- |
| Adopt placement as the `sandbox` kind | Adopt the placement mechanism with four conditions: no agent config in the box, processor-held ref instead of a resources-server registry, a built and cached runtime image, HTTP bridge when the box can reach the host | S1, S2, S4, S5 |
| Sidecar per environment from PR 3100 | Same, but generated for any agent that names a resources server, not only `simple_agent`, and resolved once at preprocessing | P1, P2 |
| Sandbox server conditional | Sandbox server not needed for any in-repo provider once docker, apptainer, and local implement same-host `connect`; server reserved for cross-host local providers | B4 |
| Class D standalone processors | Same | agreed |
| `EpisodeContext` on base requests including create params | `ng_episode` on Gym-internal requests only; path and headers to the model server | R4 |

### 7.7 Better solutions

**The harness contract should be Harbor's, not Gym's.** For CLI harnesses the minimal contract is a manifest, not Python: image, install command, launch template with the prompt and workdir, the env var the tool reads its model endpoint from, and how to parse its output. mbien's `custom_agent/harness.yaml` on `feat/sandbox-cli-agents` is that manifest, and the audit found the eight host-CLI agents differ from each other by exactly its fields. A manifest harness needs no Gym install in the box, no Python in the image, and no `HarnessAdapter` subclass; the runtime executes it through `exec` and `pty`. Placement remains for Python harnesses that own a loop (hermes, langgraph, strands). Two harness shapes, both configured in YAML, both under the same `runtime` block, and the user writes no Python for a new CLI tool. This is the single place the user asked for.

**The processor holds the sandbox ref and passes it in every request.** Seed returns the descriptor; the processor carries it into `ng_episode` for the agent and into the verify request for the environment. No server keeps a per-session dict, so `num_workers > 1` works, a retried seed is idempotent, and reverification decides from the request whether a ref is present. This is the RFC's `EpisodeContext` applied to Gym-internal requests only, and it removes both swebench's `_session_id_to_sandbox` and the placement branch's `_session_sandboxes`.

**Same-host `connect` for docker, apptainer, and local.** `serialize_handle` returns the container id, instance name, or workspace path plus a host id; `connect` rebuilds the handle from the daemon, `apptainer instance list`, or the path, and refuses when the host id differs. Three small provider changes replace the sandbox server for every in-repo use.

**A built runtime image, cached by content hash.** `gym runtime build <agent>` produces an image layer with Python, Gym, and the harness's requirements, keyed by the hash of `pyproject.toml`, the harness's manifest, and `uv.lock`; placement uses it and falls back to the per-task install only on a cache miss. The per-task path stays for development and never runs in a training loop.

**Endpoints and tokens into the box, never config.** The worker payload is the request body, the rollout-scoped routes, the MCP session token, and the resolved harness settings with secret-typed fields removed. Agent configs mark secrets with `SecretStr`, which the framework already uses for provider credentials, and the worker builder drops them.

**Declared capabilities, checked before compute.** An agent declares `capabilities` (`tools: mcp | http | none`, `runtime: [none, process, sandbox]`, `input: single | multi_turn`); an environment declares `requires` (`sandbox`, `tools`, `multi_turn`). The pairing check names the missing capability. `allowed_agents` stays as the benchmark owner's override, and a bypass records the reason in the episode record so a zero is never silent.
## 8. Sandbox authority

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

## 9. Identity, capture, and partial rollout checkpointing

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

## 10. Dataset ownership and routing

Upstream main has already moved the data model to where the processor design needs it, so the design adopts it rather than inventing a parallel scheme.

- Datasets live on the resources server block (#2724 decoupled layout; 154 resources-server configs declare `datasets:` upstream versus 37 agent configs). Self-contained agents declare their own datasets and their own `task_data.py` schema.
- Rows carry `task_source`, the name of the config instance that declared the dataset, normally the environment. Rows that still carry `agent_ref` behave byte-identically and emit a deprecation warning (#2713).
- The agent for a row is resolved in order: `agent_map` entry, row `agent_ref`, `task_source` naming an agent, `task_source` naming a resources server with exactly one referencing agent, else a pre-dispatch error. `fan_out={rs: [a, b]}` runs one dataset under N agents and stamps each in-memory copy. Benchmark datasets pin an `agent:` when the config is ambiguous.

The processor slots in with one rule: **the row's environment selects the processor.** The collector resolves `task_source` (or the `agent_ref` agent's `resources_server`) to an environment, then to the unique `episode_processors` block whose `resources_server.name` matches, with the same zero-or-two error shape as `resolve_dataset_agent`. An environment with no processor block gets an implicit default `rlvr` processor composed at config-parse time, the same way `--agent-type` composes agents. `_validate_agent_names` learns that a processor is a valid `/run` target. N agents on one environment is expressed with `fan_out` or `agent:` pins, exactly as upstream requires today; the processor does not relax #2724's ambiguity error, it consumes its result.

Materialized inputs keep carrying the resolved `agent_ref`, because RL and the gdpval orchestrator read it from rows. The processor reads the same field to pick the agent.

## 11. NeMo RL and other framework integrations

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

## 12. Processor types and configuration

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

## 13. Backward compatibility, sized

Backward compatibility is the constraint the rest of the design bends around. Two external contracts exist, and upstream main has already changed one of them under everyone's feet.

### 13.1 The dispatch contract external frameworks hold

Every trainer integration reaches Gym the same way: it starts servers with `RunHelper.start` from a config it assembles, calls `RolloutCollectionHelper.run_examples(rows)`, and reads results keyed by `agent_ref`. Inside `run_examples`, Gym resolves `task_source` to `agent_ref` synchronously and in place, validates names and pairings, and posts `/run` to `row["agent_ref"]["name"]` (`rollout_collection.py:1910-1920` upstream).

| Consumer | How it dispatches | What it reads | Status against upstream main |
| --- | --- | --- | --- |
| NeMo RL main (`edf6b8c42`, 2026-09-09) | `run_examples` | resolved `row["agent_ref"]` after the call, `response.output` token ids, `reward`, `mask_sample`, `reward_components` | Adapted: builds its per-agent counter after `run_examples` and returns the resolved ref to its caller |
| NeMo RL older (`9f166d8b0`, the tokcap branch) | `run_examples` | `row["agent_ref"]` before the call, for the counter and capture selection | Breaks on `task_source`-only rows with a `KeyError` before dispatch |
| verl recipe (`verl-recipe/nemo_gym`) | `run_examples` | `agent_ref` from the dataset row; `dataset.py` and `agent_loop.py:291` raise when it is missing | Breaks on rows collated by upstream, which strips `agent_ref` |
| `gym eval run`, `gym eval reverify` | `run_from_config` | `agent_ref` or `task_source`; reverify rebuilds the agent-to-environment map | Works |
| Direct HTTP clients (`client.py` scripts, notebooks) | `POST /run` to an agent name | the verify response | Works as long as `/run` exists on the agent |

The rows have changed underneath them. Upstream collate now strips a legacy `agent_ref` and stamps `task_source` (`train_data_utils.py:853-865`), datasets live on the resources server (#2724), and the row schema is the resources server's `TaskData` with `RESERVED_ROW_KEYS = {responses_create_params, agent_ref, task_source, _ng_task_index, _ng_rollout_index}`. The agent is "a run-time choice, never part of the data". That is the right end state and it already broke two of the five consumers above without any processor.

### 13.2 The invariants this design commits to

These hold for the whole program, not for a deprecation window.

1. `run_examples(rows)` keeps its signature and semantics: it accepts rows with `agent_ref`, with `task_source`, or with both; it resolves `task_source` in place; it returns futures that resolve to `(row, result)` with `result["agent_ref"]` equal to the resolved ref.
2. `POST /run` on an agent server name keeps working and returns the verify response shape. Processor selection happens inside `run_examples`; a caller that posts directly to an agent gets the agent's shim, which runs the same phases.
3. `agent_ref.name` is the identity on results, in `_aggregate_metrics.json`, and in MLflow metric names. `/aggregate_metrics` keeps answering on the agent name; the collector may route it to the verify owner internally.
4. Token capture is selected per agent name and read out of band by rollout id.
5. Server discovery through the head server and top-level config blocks with an `entrypoint` is unchanged; `episode_processors` is one more block type.
6. The verify response is a superset of today's: every new field is optional.

### 13.3 Task data ownership

Upstream's direction is that the resources server owns the row schema, the dataset, and the routing stamp, and the agent owns nothing about the row. The processor is the natural consumer of that split, and the design makes the split explicit at dispatch:

- The processor validates the row against the environment's `TaskData` (loaded through `load_task_data_schema`, with `normalize_task_fields` splicing a legacy `verifier_metadata` wrapper) and forwards the whole row to `/seed_session` and `/verify`.
- The agent receives `responses_create_params` plus `ng_episode` (rollout id, runtime handle, MCP headers, egress env, `skills_ref`, resume flag). It never receives task-owned fields, which closes the privileged-data leak that today's `/run` body carries into every harness (a harness that reads `verifier_metadata` can read the answer key).
- Rows in the legacy flat shape and rows in the planned `task_data` shape both validate, because `normalize_task_fields` already handles both; the processor does not add a third shape.

Two agents read row extras today and keep working: claude_code and codex read `skills_ref` from the run body, which moves into `ng_episode`; the swe_bench branch's claude_code reads `verifier_metadata` for the topology, which becomes the descriptor.

### 13.4 Repairs the design owes upstream's migration

Both breaks in section 13.1 are independent of the processor and should be fixed first.

- Collate should keep an opt-in `--stamp-agent-ref` (or resolve at collate time when exactly one agent references the environment) so verl datasets and older NeMo RL checkouts keep a resolvable row. The stamp is redundant with `task_source` and harmless.
- `run_examples` should tolerate a row that carries both keys and prefer the resolution rules already documented (`agent_map` > `agent_ref` > `task_source`), which upstream does; the documentation for trainers should say that `agent_ref` is read after the call, not before.

### 13.5 Sized by surface

| Surface | Count | Change required | When |
| --- | --- | --- | --- |
| `/run` on agent servers | 42 agents; RL, verl, and every collector post here | None. `/run` stays as a shim that runs the phases in-process when no processor is configured. | never removed in this program |
| Dataset rows with `agent_ref` | every committed dataset before #2713 | None. Rows with either key resolve. | none |
| Dataset rows with `task_source` only | every dataset collated on upstream main | None in Gym; verl and older RL need the collate stamp in section 13.4 | now |
| Environment configs | 26 `environments/`, ~150 resources-server configs upstream | None. A missing `episode_processors` block means the in-process shim. | opt-in |
| Resources servers: seed and verify schemas | 102 servers | None. New fields are optional; `BaseSeedSessionResponse` empty is valid. Four servers replace `sandbox_handle: str` with a typed entry. | Phase 1 for the four |
| Resources servers: `close_session` | 11 override `seed_session` and hold state | Override the no-op hook. Existing `/close`, `/end_session` keep working until migrated. | Phase 0 |
| Native agents | 19 simple_agent clones | Delete `run()` (optional). Middleware is added by the base class. | Phase 0 |
| Host-CLI agents and sandboxed twins | 8 + 2 | Unchanged code; `runtime.kind: sandbox` places them. The two twins are retired as directories once their configs point at the base agent. | Phase 1 |
| Self-contained agents | 6 | Become standalone processors or declare `external-agent-loop`. | Phase 2 |
| Session-state store layout | `boundaries.jsonl`, snapshots | Unchanged; attempt fence added on append. | Phase 0 |
| Control plane | 3 components | `episode_processors` added to the literal. RL reads capabilities, does not enumerate them. | Phase 2 |
| Manifest and validation | 1:1 agent-environment | Accept a processor block; N agents via existing `fan_out` and pins. | Phase 2 |

Nothing in the table is a breaking change for a user who does not opt in. The cost lands on the ten CLI agents, the four `sandbox_handle` servers, and the collate repair.
## 14. Integrating existing frameworks

| Framework | Today | Under the design | Easier? |
| --- | --- | --- | --- |
| NeMo RL | `run_examples` → `/run` by `agent_ref`; capture out of band | Identical call path; processor is invisible unless configured | same, plus sandbox lifecycle for free |
| verl, unsloth | rows routed by `agent_ref` to `/run` | identical | same |
| Harbor agents (harbor_agent) | Harbor owns the loop, the environment, and the verifier; Gym receives a trial result; `/run` registered directly with no rollout context | Stays `external-agent-loop`; gains rollout context from the shim; no other change | same |
| Harbor tasks with a Gym harness agent | Not possible: Harbor's environment builds from a task Dockerfile and Harbor's verifier runs `tests/` inside it; a Gym harness would have to be written as a Harbor `BaseAgent` | A `harbor_task` environment: seed turns a task dir into a `workspace/processor/live` sandbox entry, verify uploads `tests/`, runs `test.sh`, reads `reward.txt`, which is exactly what `terminal_bench_2_1` does today on upstream. Any Gym harness adapter then runs the task. The blocker is image build: `EpisodeDescriptor.spec` takes an image, no provider builds a Dockerfile except e2b's `build.py`; tasks must be pre-built, as the Terminus integration already requires. | yes, once tasks are pre-built |
| verifiers (Prime) | `verifiers_agent` loads the vf environment in-process, calls `run_group` with an unprefixed model URL, takes reward from the rubric; no capture, no identity | Stays `external-agent-loop` for multi-turn vf environments. For single-turn ones, a `verifiers_env` resources server can expose the dataset as rows and the rubric as `/verify`, and any Gym agent runs it. The design gives the slot; wrapping the rubric is per-environment work. | partially |
| OpenHands, mini-swe-agent | Library owns the loop inside Docker or Apptainer; grading in the agent | `external-agent-loop` unchanged, or a harness adapter when the tool can be launched non-interactively (mini-swe-agent can) | same or easier |
| tau2 | Library owns the loop with two model servers | `external-agent-loop`; provision yields two rollout-scoped URLs | same |

## 15. Migration

Reordered so the two urgent needs, harness decomposition and a first-class runtime, land before the processor server type, and revised after the comparison in section 7 so each phase reuses the artifact that already exists.

**Phase 0: mechanical harness fixes and always-on identity.** The five changes in section 6.2. Add `RolloutContextMiddleware` to agent apps and force rollout correlation on for base-class calls. Give `SimpleResponsesAPIAgent.run()` a concrete seed → prefixed self `/v1/responses` → verify → close implementation that reads the descriptor, renders MCP metadata, proxies `/aggregate_metrics`, and honors `skip_verification`; delete `run()` from the simple_agent clones and the CLI harnesses that match it. Land `close_session` (#2612, reconciled with the placement branch's `/cleanup_session` under one name) and `env_session_id` (#2613). Add the attempt fence to the session-state store. No new server type, no config change.

**Phase 1: runtime first-class.** Land the `runtime` block on `BaseRunServerInstanceConfig` with alias mapping from the five legacy spellings, the process-scoped provider in `run_webserver`, and the rollout scope in the `/run` wrapper. Land the manifest harness shape (mbien's `harness.yaml` as the schema) and migrate opencode and claude_code to it first; land the placement mechanism from `upstream/ffrujeri/sandboxes` for Python harnesses with the four conditions from section 7.6, hermes first. The seed response carries a typed `sandboxes` entry with a `serialize()` descriptor; the processor holds the ref and passes it to the agent and to verify, and swebench, deepswe, terminal_bench_2_1, and swebench_pro drop their per-session dicts. Implement same-host `connect` on docker, apptainer, and local. Add `capabilities` and `requires` and the explaining pairing check. Ship the collate repair from section 13.4. This is what makes `--agent-type opencode_agent` work on swebench, and it needs no sandbox server.

**Phase 2: `episode_processors/` server type.** Take PR 3100's packages and sidecar generation, with the sidecar keyed per environment and rows keeping `agent_ref`. Move `SandboxedAgentHost.run()` into the processor unchanged; the `/run` on agents becomes the shim that forwards to it. `_validate_agent_names` accepts processors; `episode_processors` in the control-plane literal; manifest accepts a processor block. Adopt the `/sandbox_spec` before seed ordering. Move gymnasium's loop into `step_episode_processor`. Class D integrations become standalone processors.

**Phase 3: sandbox authority.** Rollout-label check in `connect`; fix Daytona's `connect` signature; `ttl_s` and attribution on every provider; `sharing: live` reverify rule; class C export on top of pause/resume. Revive PR 2085 as the `remote` provider only if a docker-only deployment needs cross-process sharing.

**Phase 4: environments out of agents.** Move grading out of anyswe, anyterminal, swe_agents, cvdp, and vcqa into environments with `sharing: fresh` or `live`, using the swe_bench environment server and `terminal_bench_2_1` as templates. Add `harbor_task`. Retire the plugin hosts (anyswe, anyterminal, cvdp) once placement covers their harness matrix.

## 16. Decisions to take now

1. **Phase 0 and Phase 1 first, processor second.** The audits show the urgent needs are met by the base-class `run()` and the runtime block, neither of which requires a new server type.
2. **Two harness shapes, both YAML: manifest harnesses for CLI tools (no Gym in the box) and placed harnesses for Python agents (the placement mechanism with the four conditions in section 7.6).** A new CLI tool is onboarded without Python.
3. **Runtime block on `BaseRunServerInstanceConfig`, provider built in `run_webserver`, scope in the `/run` wrapper.** Never `model_post_init`. Kinds `none`, `process`, `sandbox` with the defaults in section 5.3.
4. **`agent_ref` stays the external routing key on rows and results; processor selection is internal to `run_examples`.** Sidecar generation from PR 3100, keyed per environment and for any agent that names a resources server, is the mechanism when a processor is declared; the in-process shim is the default. This is the disagreement with the RFC's routing-key change, and the contracts in section 13.1 are the reason.
5. **Per-entry sandbox ownership, a processor-held ref passed in every request, and `/sandbox_spec` before seed for processor-owned workspaces.** Disagrees with the RFC's single owner and with both branches' per-session registries; agrees with the RFC's ordering.
6. **No sandbox server for any in-repo provider.** Same-host `connect` on docker, apptainer, and local replaces it; PR 2085 stays parked for cross-host sharing of host-local providers, which nothing in the repo does.
7. **Class D integrations become standalone processors,** per the RFC.
8. **Environment teardown: one endpoint,** `close_session` from #2612 or the placement branch's `/cleanup_session`, not both.
9. **Fix the collate stamp for verl and older NeMo RL before anything else** (section 13.4); upstream's `task_source` migration already broke them.
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
| `rfcs/gym-architecture.md` (public RFC, merged onto this branch from `ananthsub/public-gym-architecture-rfc`) | processor as fourth server type owning `/run` and the sandbox; legacy processor; class D as standalone processors; metrics on the verify owner; competitive analysis against Harbor and Prime | routing key moves to the processor; single sandbox owner; sandbox server as a Phase 3 dependency |
| PR #3100 (ffrujeri) | `nemo_gym.agents` / `nemo_gym.processors`, `processors/` server type, `<agent>__processor` sidecar generated at load, transparent routing | sidecar per agent instance; no runtime |
| `upstream/ffrujeri/sandboxes` | `runtime` block on the agent config; `create_server` placement host; in-box worker invoking the harness's `/v1/responses`; automatic dependency install; `SandboxWorkspace` on seed; `/cleanup_session` | agent-side only; per-task install; no pooling; no rollout binding on the handle |
| issue #2082 | the four verifier relationships (none, fresh from artifact, same box, fresh from response) | the server as the only mechanism |
| upstream `local` provider (#2863), #2713, #2724 | host runtime behind the sandbox API; `task_source`, `fan_out`, datasets on resources servers | processor resolution |

## Appendix C: review findings and what changed in draft 2

| Finding | Change |
| --- | --- |
| A processor cannot post plain `/v1/responses`; the agent reads rollout id and capture mode from the inbound path, and capture mode is the agent's own flag | Rule 2: prefixed addressing; processor reads the agent's `token_id_capture` flag, as RL does |
| Native tool calls rely on the cookie because the resources-server prefix is gated on observability; agents have no rollout context | Rule 1: identity always on under a processor; middleware on agents is a Phase 0 prerequisite; cookies never forwarded; MCP token in `ng_episode` |
| Boundary commits do not "stay in the agent" for step environments; the store is a three-party shared-FS contract; nothing fences stale attempts | Section 9 table: commits belong to the loop owner; attempt fence on append; pause/resume as the class C primitive |
| Section 2 said `/run` is removed while Phase 1 kept a shim; RL and upstream validation hard-require an agent target | `/run` stays on agents as a shim for the whole program; collector resolves processors internally; `_validate_agent_names` learns processors |
| Upstream already routes by `task_source` with datasets on resources servers; N agents per environment collides with #2724's error | Section 9 adopts upstream routing; environment selects the processor; N agents via `fan_out` and pins |
| tau2 needs two model URLs; user simulation is an environment endpoint | Section 4: one reachable URL per declared model server; simulator unchanged; multi-agent via step processor |
| No rules for ownership conflicts | Section 8: single-owner rule, `sandboxes` list, idempotent seed, reverify rule, crash path, operate-only agents |
| Live sharing already exists in `terminal_bench_2_1`; Harbor needs image build; verifiers has no environment path | Sections 8 and 14 cite tb21 as the template; image build named as the Harbor blocker; verifiers path scoped to single-turn |
| `sandbox_handle` is on three servers; deepswe already serializes at seed; gymnasium needs `/close` and step idempotency | Corrected throughout; step processor carries both |
## Appendix D: audit summaries (2026-09-09)

Two code audits on this checkout, cross-checked against upstream main at `4fb721d0b`.

**Harness coupling matrix.** Severity: none (nothing benchmark-specific), low (generic one-line fix), med (capability gap or silent wrong answer), high (hard-wired to one benchmark).

| Agent | config | run() | responses() | env tools | verify shape |
| --- | --- | --- | --- | --- | --- |
| simple_agent | none | none | low (single-shot, full item fidelity) | none (HTTP tools) | none |
| opencode_agent | none | low (discards seed body) | med (last user message only) | med (none) | low |
| hermes_agent | none | low | none (preserves history) | med (none) | low |
| openclaw_agent | none | low | med | med (none) | low |
| claude_code_agent | none | none | med (MCP only in run()) | low (MCP) | low |
| codex_agent | none | none | med | low (MCP) | low |
| pi_agent, cline_agent, kilocode_agent, prime_agent | none | low | med | med (none) | low |
| opencode_sandboxed_agent | high | high | high | med (none) | med |

**Sandbox opt-in per consumer.** Sixteen servers use `nemo_gym.sandbox` (ten agents, six resources servers); the image comes from config in five, from the dataset row in seven, from the seed response in two, and is hardcoded in one. Creation happens per `/run` in most, per seed in the sandbox-owning resources servers, in a Ray worker for mini_swe_agent_2 and osworld, and lazily pooled in litmus. Only litmus ties teardown to the server lifespan. Of the twelve host-subprocess agents, eight could run on the `local` provider unchanged (claude_code, codex with stdin closed, cline, kilocode, opencode, simple_strands, terminus_2, vcqa local), two need streaming or signals (pi, openclaw), and three need provider-level features (prime's orphan sweep, stirrup's persistent PTY, swe_agents's bind mounts and memory watchdog). No host-subprocess agent scrubs its environment; stirrup's `env_passthrough` is the only allowlist in the repo.
