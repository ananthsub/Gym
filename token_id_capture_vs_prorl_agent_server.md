# Token-ID capture: native stack vs ProRL-Agent-Server (Polar)

Analysis notes, 2026-08-24. Compares Gym's native token-ID capture stack for training with NVIDIA-NeMo/ProRL-Agent-Server, whose installable package is named `polar` ("Polar" below). For a full description of the native stack, see [token_id_capture_overview.md](token_id_capture_overview.md).

## Summary

The two projects independently converged on the same non-negotiables: engine-authoritative token IDs, no retokenization of sampled text, strict prefix relationships as the chain criterion, and truncate-or-mask over fabrication. They part on custody, verification depth, and scope.

Polar is a complete rollout platform — scheduler, per-session containers, a four-dialect intercepting gateway, evaluators, and a Slime bridge — whose capture layer is proxy-side and structurally trusted: no digests, in-memory-authoritative records, post-hoc prefix merging. Gym's stack is a capture substrate embedded in its model server and inference workers, with materially stronger integrity machinery — request-time lineage resolution, versioned domain-separated digests, generation-time prefix-supply proof, receipts re-verified in the consumer, per-call weight-version spans — and a transport-agnostic trainer contract that NeMo RL exercises over both file custody (#3407) and TransferQueue custody (#3456).

Polar ships a working end-to-end GRPO recipe on Slime today, at the cost of version-pinned source patches to SGLang and Slime and weaker durability. Gym's equivalent proof point is the #3456 A/B on GB200s, which showed tighter logprob agreement than the token-echo baseline. The most actionable asymmetries: Polar delivers multiple trajectories per session (demultiplexing interleaved sub-agents) where Gym delivers one chain and masks the rest; Gym proves and verifies what Polar checks structurally.

## Scope: platform vs substrate

Polar is not a token-capture library; it is a "Rollout-as-a-Service" server that overlaps with NeMo Gym itself. Its concepts map almost one-to-one onto Gym's, which makes the capture comparison fair but the project comparison broader:

| Concern | NeMo Gym | Polar |
|---|---|---|
| Task execution env | Resources server state / harness sandboxes | `runtime` — per-session Docker/Apptainer container |
| Agent | Agent server (Responses-native) or external harness via base-URL redirect | `agent` — unmodified CLI/SDK preset, traffic intercepted (11 presets: Claude Code, Codex, Gemini CLI, OpenHands, …) |
| Scoring | Verifier in the resources server | `evaluator` — incl. SWE-bench, Harbor, Harbor + rubric judge |
| Trainable rows | `token_id_capture` package → rebuilt Responses payload / linearized row | `builder` — prefix merging → flattened tokens + loss mask |
| Trainer | NeMo RL (#3407 batch, #3456 async single-controller) | Slime bridge (demo-grade; NemoRL/VERL on the roadmap, unbuilt) |

## How Polar's capture works

A gateway impersonates every provider endpoint (OpenAI Chat, OpenAI Responses, Anthropic, Google) via injected environment variables, with the API key set to the session ID. Every call is detected by API family, transformed to canonical OpenAI Chat Completions, and forwarded with token-echo parameters injected — native `return_token_ids` flags on vLLM, and a source patch on SGLang 0.5.13, since stock SGLang does not expose token metadata on the OpenAI route. Each request/response pair becomes a `CompletionRecord` keyed by session in an in-memory store (a background writer persists best-effort copies to disk for observability). Streaming clients get one buffered backend call replayed as synthetic SSE.

At post-run, a `prefix_merging` builder reconstructs trajectories: a completion joins the open chain whose last prompt is the longest strict token-prefix of its prompt (comparing server-side prompt tokenizations only, never sampled response IDs). The merged stream is the first prompt, then per turn the raw sampled response IDs (loss mask 1, real logprobs) plus the interstitial tokens sliced from the next call's prompt at the first end-of-turn token (loss mask 0). A prefix break — e.g. harness compaction rewrote history — truncates the chain at the break, never repairs it. The Slime bridge converts each trace to a Slime `Sample`, validates masks and logprobs fail-closed, drops stale or under-complete groups, and computes leave-one-out GRPO baselines per prompt group.

## Shared doctrine

- **No local retokenization.** Polar has no tokenizer dependency at all; Gym records tokens where the engine produced them. Both treat sampled response IDs as untouchable — Polar explicitly never prefix-matches on them because re-serialized text can re-tokenize differently, which is the same drift Gym's #2181 exists to defeat.
- **Streaming is buffered.** Both make one complete backend call and replay synthetic SSE.
- **The agent never sees token data**, so histories stay strictly extending, and the only tamper vector is history rewriting — which both detect by prefix checks.
- **Truncate or mask, never guess.** Polar truncates chains at prefix breaks and drops invalid groups with metrics; Gym masks with `mask_sample` and a reason taxonomy.
- **Health as a first-class signal** against silent partial training: Polar's `reconstruction_stats` and fail-closed bridge validation; Gym's `token_capture/*` and `finalize/token_in_rate` metrics.

## Where they diverge

### Chaining: recorded lineage vs post-hoc inference

Polar decides chain membership entirely after the run, by longest strict token-prefix. The approach is simple and has one property Gym currently lacks: it demultiplexes interleaved parallel sub-agents into separate chains and delivers all of them as separate training samples sharing a group ID. Gym's merged builder does the same inference (a prefix trie) but delivers only the main chain and masks or quarantines the rest; multi-trace delivery is explicit future work in `DESIGN.md`.

The open Gym stack (#2180) then goes where post-hoc inference cannot: the parent is resolved at request time from the live request, persisted in the same durable write as the tokens, and re-verified by digest at rebuild. That disambiguates retries — identical prompts with different generations — which pure prefix matching is structurally blind to. Polar's answer to retries is coarser: single-use session IDs, timestamp order, and truncation on any break.

### Loss masks and delivery shape

Polar's builder emits what a trainer consumes directly: one flattened token stream per trace with an explicit 0/1 loss mask and real logprobs on trained positions, with interstitials split at an end-of-turn token (auto-detected or configured — a heuristic Gym avoids by recording prompt/generation boundaries per call). Gym's #3407 path rebuilds a Responses payload and lets NeMo RL derive the mask from the user/assistant role split; #2278/#3456's `verify_and_linearize` converges on Polar's output shape — flattened token IDs, mask, and logprobs — while adding per-call weight-version spans and routed-expert tensors. Same destination; one side carries proofs.

### Integrity: structural vs verified

Polar's trust model is structural: the agent cannot touch tokens, prefix checks catch rewrites, validators catch length mismatches, and an engine-equivalence test pins that SGLang- and vLLM-shaped responses produce byte-identical traces. There are no digests; the proxy's records are believed. Gym assumes less: versioned, domain-separated, length-delimited digests pinned by golden vectors; recorded parent links re-verified before use; #2181's separation of intent (`prefix_requested`) from proof (`prefix_supplied`); #2278's chained ancestry hashes and receipts re-verified field by field in the consumer. For heterogeneous deployments — third-party harnesses, framework-owned transports, multi-worker model servers — Gym's posture is the defensible one.

### Durability and lifecycle

Polar's authoritative store is in-memory per session; the background disk writer is best-effort (queue-full drops, size truncation), and a gateway crash loses the session. There is no retirement protocol — records live until post-run, then results ship. Gym's file store fsyncs before acknowledging, repairs torn tails, freezes atomic snapshots, retires them conditionally only after durable trainer handoff, and keeps tombstones so late writers cannot resurrect consumed attempts. #3456 extends the same discipline across processes: stage-before-acknowledge in the worker, fatal-on-unknown-outcome in the finalizer.

### Partial-data policy

A real philosophical split. Polar salvages: an agent timeout still yields a partial trajectory from the calls captured so far, gated by an accept-fraction knob (0.6 in the SWE-Gym recipe). Gym masks anything unproven and aborts a run whose mask fraction crosses the kill switch. Polar risks training on truncated behavior; Gym risks discarding usable signal — the #3456 A/B saw 0.57% of calls poisoning about 36% of rollouts before a carve-out relaxed the strictest rule, which shows Gym already adjusting toward the policy Polar started with.

### Trainer coupling and off-policy accounting

Polar couples over HTTP plus a Slime-specific bridge that needs a patched Slime router to carry token fields; staleness is handled by dropping whole groups past a policy-version horizon. Gym couples through in-process protocol objects (`TokenSink`/`TokenSource`, `StagingSink`/`StagingSource`) that the framework implements over its own data plane — no engine or trainer source patches on the Gym side of the contract, though #3456 relies on NeMo RL's framework-patched vLLM honoring `required_prefix_token_ids`. Off-policy accounting is finer-grained on the Gym side: per-call weight-version spans with allow/reject policies for mixed-version rollouts, versus Polar's per-task metadata stamp.

Note the patch surface behind Polar's "zero integration" claim: unmodified agents, yes, but the exact-token path requires editing SGLang 0.5.13 source and Slime v0.3.0's router via version-pinned shell patches. Gym concentrates that burden in components the NeMo stack already owns.

## Head to head

| Axis | Gym native stack | Polar (ProRL-Agent-Server) |
|---|---|---|
| Capture point | Inside Gym's model server; #2278: inside the inference worker | Intercepting gateway proxy in front of a stock engine |
| Engines | vLLM (server-managed token echo; framework-patched vLLM for prefix supply) | SGLang (source patch) and vLLM (native flags) |
| Agent surface | Chat, Responses, Anthropic dialects via base-URL redirect (Claude Code, OpenHands reference cases) | 4 API families intercepted; 11 unmodified-CLI presets |
| Identity | Rollout ID in URL path, caller-assigned; per-call ID from middleware | Session ID doubles as the injected API key |
| Chaining | Request-time resolution, persisted + digest-verified (#2180); prefix-trie inference for legacy records | Post-hoc longest-prefix merging at post-run |
| Multi-trajectory | One chain delivered; rest masked/quarantined (future work) | All chains delivered as separate samples per group — handles parallel sub-agents |
| Loss mask | Role split (#3407) or linearized mask row (#2278/#3456) | Explicit 0/1 mask; interstitials split at an end-of-turn-token heuristic |
| Continuation exactness | #2181: exact parent tokens supplied with generation-time proof | None — chat-template re-render each turn; drift breaks and truncates the chain |
| Integrity | Versioned digests, golden vectors, receipts re-verified in the consumer | Structural: prefix property + length validators; no digests |
| Durability | fsync-before-ack; freeze/retire lifecycle; tombstones | In-memory authoritative; best-effort disk; crash loses sessions |
| Partial rollouts | Mask unproven data; mask-fraction kill switch | Salvage partial trajectories; accept-fraction gate |
| Trainer | NeMo RL #3407 + #3456 (protocols over file / TransferQueue); GB200 A/B validated | Slime bridge (patched); working 8xH100 SWE-Gym GRPO recipe; other trainers unbuilt |
| Off-policy accounting | Per-call weight-version spans; mixed-version policies | Per-task policy-version stamp; staleness dropping in the bridge |

## What each should take from the other

**Gym should take from Polar:**

- **Multi-trace delivery.** Polar's longest-tip demultiplexing turns sub-agents and auxiliary calls into extra training samples instead of masked fraction. This is the single highest-leverage idea for tokidcap's yield roadmap.
- **Salvage policy as a knob.** An accept-fraction gate for partial-but-verified chains would recover signal the kill switch currently discards; the #3456 poison carve-out already points this direction.
- **Dialect breadth.** Polar's Anthropic/Gemini/Responses transforms show what a universal harness surface looks like; Gym's Anthropic path covers the reference case but not the field.

**Polar should take from Gym:**

- **Prefix supply.** Polar has no defense against chat-template re-render drift — the exact failure that silently truncates its chains. #2181's proof-carrying supply is the fix.
- **Durable custody.** fsync-before-acknowledge and a freeze/retire lifecycle would remove the crash-loses-session and delivery-race failure classes entirely.
- **Digest verification.** Cheap to add at the record layer, and the only way to keep integrity claims once records cross process or transport boundaries — which Polar's roadmap (more trainer bridges) will force.

## Bottom line

Polar is the pragmatic system: broader harness surface, simpler mental model, a runnable Slime recipe today — with integrity and durability that hold until something crosses a process boundary or crashes. Gym's stack is the infrastructure investment: heavier, still landing its open PRs, but the only one of the two whose guarantees survive multi-worker serving, framework-owned transports, retries, and untrusted harnesses — and it is already wired into NeMo RL on both rollout paths. They are converging on the same output shape (flattened tokens + mask + logprobs + per-call provenance); the durable difference is that Gym verifies what Polar assumes. Since both are NVIDIA-NeMo projects targeting agentic RL, the strategic question is not which wins, but whether Polar's gateway breadth and Gym's verified substrate get composed — Polar-style dialect coverage in front, tokidcap-grade custody underneath — rather than developed as parallel answers.

## Sources

- ProRL-Agent-Server @ `6a1ead6b` (2026-08-13): `src/polar/gateway/` (`engine.py`, `server.py`, `storage.py`), `src/polar/trajectory/` (`builder/prefix_merging.py`, `evaluator/harbor*.py`), `src/slime_bridge/`
- Gym main: #2124, #2125, #2126, #2190; open stack #2180, #2181 (+ v2 line), #2278 (`staging/`, lineage ledger, control routes)
- NeMo RL: #3407 (rollout-actor custody), #3456 (TransferQueue worker custody; lineage-ledger A/B, 2026-08-20)
