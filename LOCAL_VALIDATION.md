# Local validation harness: the whole capture stack without a GPU

Branch `capture-validation-harness`, stacked on `terminal-attribution` (see
`TERMINAL_ATTRIBUTION.md` for the attribution design this validates). This document
carries the design of the GPU-free validation approach, its open questions, the matrix
of validations still to build, and the harnesses to sweep it across.

## 1. The idea

Everything in the capture/attribution/lineage stack except actual generation is
CPU-testable, provided we replace **only the generation backend** and keep every Gym
layer real: the `vllm_model` server's preprocessing and extraction, the dialect
converters, capture middleware, stores, the builders, delivery, agents, resources
servers, and — for tier 3 — real external harness CLIs.

The core piece (committed here as `tests/unit_tests/test_capture_fake_backend_e2e.py`)
is a deterministic OpenAI-compatible fake backend with one load-bearing property: **the
tokenizer is consistent** — an assistant message re-entering a later prompt tokenizes to
exactly the generation ids it was served with, so multi-turn conversations produce
*genuine* token-prefix chains, not simulated ones. Per-message tokens are stable hashes
(`crc32(role) + crc32(word)…`), replies are scripted per call, and the message carries
the NeMo-RL-style token bundle (`prompt_token_ids`/`generation_token_ids`/
`generation_log_probs` on the assistant message — no vLLM transport flags needed).

### Three tiers

| Tier | What runs | Status |
|---|---|---|
| **T1** | TestClient over the real `vllm_model` app + fake backend on real HTTP | **Committed, green.** Proves: outbound hop, bundle extraction, converter round-trips, prefix chaining, per-dialect `response_id` (incl. the Messages id-reuse), the §2.2.0 wire-vs-captured fingerprint invariant on all three routes, attribution + delivery |
| **T2** | Real uvicorn servers, real agents and resources servers, dispatch via `run_examples` with caller-supplied `_ng_rollout_id`s | Not built. Validates the identity plan (dispatcher-minted ids end to end), agent loops over the fake model, the echo-contract sweep against real verifiers, masked-fraction plumbing |
| **T3** | Real external harness CLIs against the served endpoint | `tests/manual/claude_code_smoke.py` committed (blocked only on a machine with the `claude` binary); the other CLIs follow the same pattern |

### Two integration facts T1 already surfaced

- The global config lazily Hydra-parses `sys.argv` on the first outbound call
  (`get_global_aiohttp_client` → `get_global_config_dict`) — tests must pre-seed
  `_GLOBAL_CONFIG_DICT`, and any embedding of Gym servers outside the CLI hits this.
- A response still carrying inline token arrays bypasses attribution by design (the
  native arrays are the policy's data and take precedence); attribution engages only on
  token-free results. Worth knowing when reading finalize metrics.

## 2. Open questions

1. **Should the fake backend become a first-class model server?** A
   `responses_api_models/fake_model` usable via `gym env start --model-type fake_model`
   would make T2/T3 one-command for *any* agent/resources pair and let `gym env test`
   suites run capture assertions GPU-free. Strong candidate; the test-local FastAPI app
   is the prototype. (The existing test-only fakes fake at the wrong layer — they
   replace `responses()` and bypass the entire vllm_model path.)
2. **Scenario scripting API.** Retries, divergent resamples, mid-rollout failures, and
   latency need per-request scripting (nth request → 500 / different sample / delay).
   What's the right knob surface — request-count rules in the backend app state, or a
   control route the test drives?
3. **Streaming fidelity.** Which SSE behaviors must the fake reproduce for blackbox
   streaming validation to be meaningful (chunk envelope ids, usage frames, event
   ordering per dialect)? The Gym side synthesizes SSE from complete responses, so T1
   covers Gym's serving; the open half is harness-side reconstruction (T3).
4. **Reasoning parametrization.** Which combinations of `uses_reasoning_parser` ×
   `preserve_reasoning_in_assistant_content` are contract-bearing for the §2.2.0
   invariant? The fake can emit `<think>` content today; the test matrix below needs the
   sanctioned combo list.
5. **Determinism vs realism.** Hash tokens give exact prefix properties but unrealistic
   vocab and lengths. Is realism ever load-bearing locally (masked-fraction realism,
   length-dependent paths), or does that legitimately belong to the GPU tier?
6. **CI placement.** T1 is plain pytest (already in `tests/unit_tests/`). T2 fits the
   `gym env test` shape. T3 needs CLIs installed — manual/nightly only?

## 3. Tests to validate (the build-out matrix)

From `TERMINAL_ATTRIBUTION.md` §6's low-confidence list plus the identity plan, in
rough priority order:

| Validation | Tier | Notes |
|---|---|---|
| Reasoning-item transcripts per config | T1 | Fake emits `<think>…`; assert the wire-vs-captured fingerprint invariant per reasoning combo — pins the §2.2.0 crack |
| Tool-call finals | T1 | Fake replies with `tool_calls`; terminal ends on a `function_call`; attribution + fingerprint through real converters |
| Streaming id propagation per dialect | T1 | `stream: true` on all three routes; assert the reconstructed payload carries the captured id |
| Divergent-retry with real serving | T1 | Scripted resample on identical prompt; id witness resolves; content abstains |
| Incomplete-capture ordering | T1 | Abort between `begin_call` and response (scripted backend hang + client timeout); attributed-but-incomplete must still mask |
| Caller-supplied `_ng_rollout_id` through `run_examples` | T2 | The identity contract: dispatcher mints, everything downstream agrees; attempt-suffix retries |
| Echo-contract sweep | T2 | Real resources servers verify over fake-model rollouts; assert `response` unchanged fleet-wide (PR 3's test, made cheap) |
| Masked-fraction plumbing E2E | T2 | Aux-call and fork scenarios through a real agent; the kill-switch reason aggregation |
| Session kill-and-resume over fake backend | T2 | Ties the session-state prototype's boundary records to really-served capture |
| Prefix supply verification (`required_prefix_token_ids`) | T1, post-#2181 | Fake honors the field by echoing it as the prompt prefix; `prefix_supplied` proof path runs for real |
| Conformance kit against a TQ-shaped sink | T1, post-#2180 | The kit's multi-process bar, fake-backend-fed |

## 4. Harness matrix: what to check this against

Per harness, the assertions are the same five: capture completeness, `response_id`
presence and join, attribution method distribution, masked fraction, and the
wire-vs-captured fingerprint. What differs is which failure mode each harness is the
canary for:

**Native loops (T2, no CLI needed)**
- `simple_agent` — the merged-transcript-on-final-envelope path; the id witness's
  reference case.
- `gymnasium_agent` + `blackjack` — the `/step` protocol (no verify): result synthesis
  without an echo; also the session-state resume pairing.
- `langgraph_agent` family (`orchestrator`, `parallel_thinking`, `rewoo`) — native
  multi-call forests: multiple chains per rollout, the §2.8 preview.
- `tool_simulation_agent` — tool-heavy single chains.

**Blackbox, observation-instrumented (T3)**
- `claude_code_agent` — Messages dialect, streaming, compaction events, sub-agent
  invocations, the id-reuse smoke (script committed).
- `openclaw_agent` / `opencode_agent` — chat dialect over SSE; the no-in-band-id case
  where content + observation witnesses carry everything.
- `codex_agent` — Responses-over-SSE; item ids riding the transcript.
- `cline` / `kilocode` / `pi` / `prime` — same class, sweep for wrapper-fidelity
  variance (the §2.5.1 contract).

**Synthesizers with harness-native terminals (T3/T2)**
- `swe_agents` — declared-terminal path (completion files) + token re-attachment.
- `mini_swe_agent(_2)`, `harbor`, `anyswe`, `anyterminal` — default-response-object
  synthesis; the never-mint-ids rule's targets.

**Known-deviant cases (regression canaries)**
- `image_tools_agent` — scores/persists mismatch (pre-existing; attribution should
  surface it as disagreement or no-match, not silence).
- `labbench2_vlm_agent` — deliberate redaction; must fail closed with a declared gap.
- `terminus_judge` as verifier — the surrogate scrub; assert zero attribution cost on
  clean content.
- `remote_agent` / `browsecomp_agent` failure paths — replaced responses must land as
  honest abstentions.

## 5. Relationship to the other branches

- `terminal-attribution` — the mechanism this validates; keep that branch scoped to its
  PR split. This branch stacks on it and holds all validation-harness work.
- The observability/identity refactor discussions (rollout-id-first-class, observe-once
  /project-twice) land their own branches; T2 here is the natural regression bed for the
  identity PR.
