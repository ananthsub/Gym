# Token-ID capture: native stack vs the Switchyard integration

Analysis notes, 2026-08-24. Compares Gym's native token-ID capture stack with the Switchyard proxy integration (Gym #2026). For a full description of the native stack, see [token_id_capture_overview.md](token_id_capture_overview.md).

## Summary

The native stack has become the mainline and has pulled well ahead. Its base is merged into Gym main (#2124 capture, #2125 chaining, #2126 delivery, #2190 sampling pinning), an active open stack (#2180 request-time parent resolution, #2181 exact prefix supply, #2278 worker-locus custody) extends it, two NeMo RL consumer PRs (#3407, #3456) exercise it end to end, and an 8-GPU A/B on the #3456 branch measured tighter logprob agreement than the legacy token-echo path.

The Switchyard integration (#2026) is a well-engineered design that reached full unit coverage and then stalled. The PR has been open since 2026-07-14 with its one remaining test — a real GPU end-to-end rollout — never run, and the Switchyard-side capture machinery it depends on is stranded on the `rl-token-capture-beta` branch, whose Python serving substrate Switchyard main has since deleted. Its durable ideas (fail-closed `mask_sample`, wire-contract coupling, never repairing a divergent chain) already live on in the native stack.

## The two designs

**Native stack (in-process).** Capture happens inside Gym's own model server, where the exact prompt and generation token IDs and logprobs still exist: an ASGI middleware records every call before dialect conversion or SSE synthesis can drop the token fields. #2278 pushes custody further down into the inference worker itself, with the model server keeping only a token-free lineage ledger. Correlation is a URL prefix (`/ng-rollout/<id>/training-token-capture/`) with caller-assigned identity, and records are durable before the harness sees the response.

**Switchyard integration (on the wire).** OpenHands policy calls are routed through the Switchyard proxy, which injects `return_token_ids` and `logprobs` into the upstream vLLM request and records each call's engine-authoritative token triple as a versioned JSON file under `sessions/<uuid>/`. Gym retrieves the session after the run (`GET /v1/sessions/<uuid>/completions`) and rebuilds the records into its trainer-facing Responses rollout. The integration is zero-fork: the pinned OpenHands client is steered entirely by per-run configuration — the agent TOML `model` becomes the Switchyard route ID, the session UUID rides `completion_kwargs.proxy_x_session_id` (stripped before forwarding), and the agent container's model-server entry is rewritten to Switchyard's host and port. Gym imports nothing from Switchyard; the coupling is a wire contract guarded by a `schema_version` check.

## Where the two designs agree

The convergence validates the program's core doctrine from two independent directions:

- **Engine-authoritative token IDs.** Both take vLLM's own `prompt_token_ids`, `token_ids`, and sampled logprobs, and never re-derive tokens from text.
- **Capture before lossy layers.** Switchyard records before its translation engine drops token fields; Gym's middleware records before dialect conversion and stream synthesis. Both buffer streaming upstream and replay it, because token arrays exist only on complete responses.
- **The harness never sees token data**, so prompt histories stay strictly extending.
- **Fail closed, never repair.** Both mask a sample on any retrieval, schema, or contiguity failure rather than guessing. #2026's design doc states it directly ("strict-history reconstruction never repairs a divergent chain"), and the native builder's quarantine and mask behavior is the same principle. #2026 is where `BaseVerifyResponse.mask_sample` was first introduced; the native delivery path now carries the same verdict at the top of every rollout record.

## Where they diverge

### Lineage: post-hoc validation vs recorded resolution

This is the deepest technical difference. Switchyard records are ordered by `(captured_at, uuid)` and validated after the fact by two independent contiguity checks in `nemo_gym/switchyard_trace.py`: message-level strict extension and token-level prefix extension. That is equivalent in strength to Gym's merged builder (token-prefix inference over a trie), and both share the same blind spots — a retried call, a compaction, or a parallel sub-agent produces a shape that can only be detected and masked, costing the whole sample.

The open native stack moves past this. #2180 resolves each call's parent at request time in the model server — fingerprint lookup over model-authored turns, context-digest verification, and a persisted `ROOT` / `RESOLVED` / `UNRESOLVED` decision written in the same durable write as the tokens — and the builder re-verifies every recorded link by digest. Most retries become resolvable instead of fatal, and ambiguity becomes a per-boundary fragment rather than a dead session.

### Continuation exactness

#2181 supplies a resolved parent's exact cumulative tokens back to vLLM (`required_prefix_token_ids`) and demands generation-time proof, persisting intent (`prefix_requested`) separately from proof (`prefix_supplied`). Chain contiguity then holds by construction rather than being discovered at rebuild time. Switchyard's beta branch developed the same idea independently — a token-injection backend (~530 lines) that serves matched calls from accumulated raw token chains — but with in-process chain state (one proxy must own a session) and silent fallback to the chat path, where Gym's version is durable, multi-worker, and proof-carrying.

### Trainer handoff

#2026 puts the token triples inside the rollout payload: ordinary Responses output items carrying `TokenIDLogProbMixin` fields, with no out-of-band store and no lifecycle beyond the per-run log directory. The native stack made the opposite choice: an out-of-band `TokenSink` / `TokenSource` protocol surface with freeze, rebuild, and conditional-retire semantics; transport-agnostic custody (file store, or the trainer's own data plane, e.g. TransferQueue in RL #3456); delivery metrics; a mask-fraction kill switch; and, in #2278, no token data on any agent-facing response at all. For a trainer that owns retirement, replay, and off-policy accounting, the protocol surface is the version that scales.

### Coverage and dialects

Switchyard's structural advantage is breadth by proxy: any OpenAI-compatible client can be captured zero-fork through a header, body field, or API-key channel, and Switchyard's translation layer can face Anthropic-native clients while speaking to vLLM. The native stack asks only that the harness's base URL point at Gym's model server — an environment-variable change in practice — and natively captures Chat, Responses, and Anthropic Messages, including buffered streaming. In practice, #2026 wired exactly one harness (OpenHands), so the breadth advantage stayed theoretical, while the native path's reference cases (Claude Code CLI; OpenHands under RL #3456) actually ran.

### Operations

The proxy is an extra service reachable from the wrapper process and every agent container, with full conversation text on its disk. The native stack runs in-process with the model server the run already has. There is also a structural problem on the Switchyard side:

> **The capture substrate is stranded.** Switchyard #63 (the capture implementation) merged on 2026-08-05 into `rl-token-capture-beta`, not main. It requires the legacy Python serve path (`--enable-rl-logging`), which the Rust server rejects and which Switchyard main has since deleted outright. Reviving #2026 today means either deploying a beta branch frozen at a July main, or first porting capture into the Rust `switchyard-server` — a prerequisite project that does not exist yet.

## Head to head

| Axis | Native stack | Switchyard integration (#2026) |
|---|---|---|
| Capture point | Inside Gym's model server; #2278 moves it into the inference worker | Standalone proxy between harness and vLLM, pre-translation |
| Token source | vLLM inline token bundle / logprobs, validated as coherent; server pins sampling params | vLLM `return_token_ids` + `logprobs` injected into `extra_body` |
| Identity | Caller-assigned rollout ID in the URL path; per-call ID minted by middleware | Per-run session UUID in a body field, stripped before forwarding |
| Lineage | Request-time resolution + digest-verified links (#2180); chained hashes and receipts (#2278) | Post-hoc timestamp order + dual prefix-extension validation; divergence masks the session |
| Retries | Mostly disambiguated at request time; final-call retries mask | Any non-extending record refuses the whole session |
| Continuation supply | #2181: exact parent tokens to the engine with persisted generation-time proof | Beta-branch injection backend: in-process chains, silent fallback, no proof record |
| Trainer handoff | Out-of-band sink/source protocols; freeze → rebuild → conditional retire; metrics + `mask_sample`; #2278 keeps tokens off agent-facing responses entirely | Token triples inline in the rollout Responses payload; post-run GET; `mask_sample` on failure |
| Dialects | Chat, Responses, Anthropic Messages, buffered streaming (#2278 topology: non-streaming chat) | OpenAI chat via proxy; any OpenAI-compatible client zero-fork; MVP wires OpenHands only |
| Footprint | In-process; node-local file store; shared lineage store for multi-worker | Extra service + per-run log dir; conversation text on proxy disk |
| Consumers | NeMo RL #3407 and #3456, A/B-validated on GB200s | None wired; trainer reads the rollout payload as-is |
| Status | Base merged; #2180/#2181 well-developed (v2 line adds schema-v5 deltas, conformance kit, DESIGN.md); #2278 active | Open since 2026-07-14; unit-complete; GPU end-to-end never run; capture stranded on a Switchyard beta branch |

## Status and trajectory

**Native:** the merged base is stable with roughly 2,500 lines of tests covering transport contracts, torn-tail repair, tombstones, and the mask-fraction abort. The open stack is deep but converging. Two facts the PR descriptions do not tell you: #2180/#2181 have a v2 line (consolidated under the #2349 docs branch) that supersedes the PR refs — schema-v5 parent-relative deltas, an `IncrementalLineageStore`, golden fingerprint vectors, a conformance kit, and `DESIGN.md` — and #2278's branch has evolved past its own description (the earlier stateful custody component was replaced by a token-free `CaptureLedger` served over one bearer-authenticated manifest route). The two custody topologies — Gym-custody (#2180/#2181) and worker-custody (#2278) — are explicitly not yet reconciled; that reconciliation is the stack's main open design item, tracked in `DESIGN.md`.

**Switchyard:** #2026 remains a clean, honest PR — 226 tests passing, a thorough design doc, disciplined wire-contract coupling — but its Gym-side alternative shipped, its consumer never materialized, and its proxy-side substrate lost its home branch. What survives is conceptual: `mask_sample` is now load-bearing across the native stack and NeMo RL, and `switchyard_trace.py`'s dual contiguity validation reads today as a specification of the invariants the native builder enforces natively.

## Bottom line

- **The native stack won the architectural decision.** Everything #2026 guarantees, the native stack guarantees with stronger machinery — plus request-time resolution, proof-carrying prefix supply, a retirement lifecycle, weight-version accounting, and two working trainer integrations.
- **#2026's remaining unique value is narrow:** capture for clients that cannot be repointed at Gym's model server, or multi-provider translation in front of vLLM. If that need arises, the honest cost estimate includes porting capture to the Rust `switchyard-server` first.
- **Decide the PR's fate explicitly.** Either park #2026 with a note pointing at the native stack as the successor, or scope the Switchyard-side Rust port before investing further Gym-side. Leaving it open implies a parallel path that no longer exists.

## Sources

- Gym main: #2124 (`354babf7e`), #2125 (`945116e5e`), #2126 (`ce5850d3d`), #2190 (`b50cfb170`)
- Gym open PRs: #2180, #2181 (+ v2 line under the #2349 docs branch), #2278 (`tq-tokidcap-capture`)
- Gym #2026 @ `188061ec1`: `gym_switchyard_integration.md`, `nemo_gym/switchyard_trace.py`, `responses_api_agents/swe_agents/app.py`
- Switchyard: main @ `cef42319`; capture on `rl-token-capture-beta` via #63 (`97424fc1`, 2026-08-05)
- NeMo RL consumers: #3407, #3456 (incl. lineage-ledger A/B report, 2026-08-20)
