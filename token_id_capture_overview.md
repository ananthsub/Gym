# Token-ID capture in NeMo Gym

Analysis notes, 2026-08-24. Describes Gym `main` at `e87d5d9f9` plus the open PR stack (#2180, #2181, #2278 and the v2 line under the #2349 docs branch). Sections that describe unmerged work say so explicitly.

Companion documents: [token_id_capture_vs_switchyard.md](token_id_capture_vs_switchyard.md) and [token_id_capture_vs_prorl_agent_server.md](token_id_capture_vs_prorl_agent_server.md).

## The problem

External agent harnesses — the Claude Code CLI over Anthropic Messages is the reference case — drive their own model-calling loop and return text transcripts with no field for token IDs. Training frameworks need the exact token IDs the policy sampled, the prompt token IDs each call was conditioned on, and the sampling-time log probabilities. Re-tokenizing transcript text does not reliably reproduce them: chat templates render differently than the engine sampled, tool-call parsers truncate assistant turns, and re-serialized text can tokenize into different IDs than were generated.

Token-ID capture records this data inside Gym's model server, where it still exists, keys it to the rollout, and rebuilds each rollout's calls into one contiguous, trainable Responses payload.

## Design goals

Stated in the docs line (#2341/#2349, `fern/versions/latest/pages/training-tutorials/external-agent-harnesses.mdx`) and held throughout the implementation:

- **Exact token identity.** Tokens are recorded where they exist and never re-derived from text. Rollouts that already carry token IDs are left untouched.
- **Structural loss mask.** Prompt positions get mask 0, generation positions mask 1, derived from record structure rather than heuristics.
- **Fail closed.** Every ambiguous or lossy condition converges on `mask_sample: true` rather than a fabricated token sequence. There is no path on which a guessed or corrupted sequence is delivered as trainable.
- **Framework-neutral write/read split.** Gym owns the record shape and the capture point; the training framework may own transport, storage, and readback through protocol objects.

## Architecture overview

Four stages, all merged in `main` (#2124 capture, #2125 chaining, #2126 delivery, #2190 sampling pinning):

1. **Identify.** The caller assigns a rollout ID; the agent tags the model URL with it.
2. **Capture.** Model-server middleware records each call's exact token data durably before the harness sees the response.
3. **Store.** Records land in a `TokenSink` (file-based by default, pluggable for framework transports).
4. **Rebuild and deliver.** After the rollout completes, a builder chains the calls into one contiguous Responses payload, or masks the sample if it cannot do so safely.

## Rollout identity and correlation

- `nemo_gym/config_types.py` defines `ROLLOUT_PATH_PREFIX = "ng-rollout"` and `TOKEN_CAPTURE_PATH_SEGMENT = "training-token-capture"`.
- `nemo_gym/rollout_correlation.py` builds the capture key: an explicit `_ng_rollout_id` in the run body (validated against `ROLLOUT_ID_PATTERN`; malformed IDs are rejected, not sanitized), or `"{task}-{rollout}"` from task/rollout indices, with `-a{n}` appended for re-dispatch attempts. `RolloutContextMiddleware` strips the `/ng-rollout/<id>/…` prefix and exposes the ID through a ContextVar.
- An opted-in agent (`nemo_gym/base_responses_api_agent.py`) rewrites the downstream model URL to `…/ng-rollout/<id>/training-token-capture/v1/…`. Evaluation observability uses the same prefix without the `training-token-capture` segment; the extra segment is the explicit training-intent marker.
- Enablement is run-level `token_id_capture.enabled` AND (per-agent `token_id_capture: true` OR run-level `all_agents: true`).

Because capture identity is a URL property, any model-server worker can serve any call of a rollout; no request affinity is required.

## Capture inside the model server

`nemo_gym/base_responses_api_model.py`:

- `_CaptureMiddleware` (pure ASGI, installed by `install_model_call_capture()`) matches the rollout path, strips the prefix, mints a `model_call_id`, and sets a request-scoped `CaptureContext(rollout_id, model_call_id, token_sink)`. Sink precedence: configured sink (`token_id_capture.sink`, constructed per worker process) → process-installed sink (`install_token_sink`) → file store. A capture-prefixed request to a path the middleware cannot capture marks the rollout incomplete before forwarding, so an uncapturable call cannot masquerade as a complete rollout.
- All dialect entrypoints — `/v1/chat/completions` (including `stream: true`, which is buffered and replayed as synthetic SSE), `/v1/responses`, and Anthropic `/v1/messages` — funnel through `_invoke_responses` / `_invoke_chat_completions`. These call `register_call_intent()` before dispatch (an optional durable `begin_call` intent; failure fails the model call before generation) and `capture_tokens(response)` on the complete response, before dialect conversion drops token fields and before stream synthesis.
- The sink write is awaited before the response returns. This ordering matters: a sequential harness cannot send call N+1 until call N's record is durable, so the next continuation can always resolve against it.
- Capture failure after generation never fails the model call; it durably marks the rollout incomplete instead (`mark_incomplete`), which masks the sample at freeze time.

The vLLM model server (`responses_api_models/vllm_model/app.py`) produces the token data: with `return_token_id_information`, the outbound engine request pins `logprobs=True, top_logprobs=0`; token IDs and logprobs come from an inline message bundle (preferred) or choice logprobs plus `/tokenize` for the prompt, validated as a coherent bundle, and are attached to exactly one output item as `prompt_token_ids` / `generation_token_ids` / `generation_log_probs`. #2190's `sampling_overrides` applies configured sampling parameters last on every generation path, so interactive harnesses that send no sampling fields still sample on-policy.

## What gets recorded

`nemo_gym/token_id_capture/records.py` defines `TokenEntry` (schema version 1 on `main`): `schema_version`, `rollout_id`, `model_call_id`, `model`, `prompt_token_ids`, `generation_token_ids`, `generation_log_probs` (length-validated against generation IDs), optional `routed_experts` (opaque MoE routing data), `output_items` (response content with token arrays stripped, kept for text-based penalties), `token_item_index`, and a non-semantic `created_at`. Records with a newer schema version than the reader supports are rejected rather than partially parsed.

## Protocols

`nemo_gym/token_id_capture/protocols.py` has no FastAPI, Ray, Torch, or aiohttp imports, so inference workers can import it.

- `TokenSink`: `put(entry)` (durable before return; same-ID re-put with the same payload is idempotent, with a different payload must fail; write-after-freeze must fail), `mark_incomplete(rollout_id, model_call_id)`, `close()`. Optional `begin_call` extension for call-intent custody.
- `TokenSource`: `freeze(rollout_id) -> TokenCaptureSnapshot` (atomic, idempotent; carries `entries`, `incomplete`, `snapshot_id`, `version`), `drop(rollout_id, *, snapshot_id, version) -> bool` (conditional retirement; returns `False` if state advanced), `close()`.
- Process-global `install_token_sink` / `install_token_source`, with the documented caveat that uvicorn-spawned workers do not inherit launcher installs — which is why config supports `sink: module.path:ClassName` factory descriptors constructed per worker with startup protocol validation.

## The file store

`nemo_gym/token_id_capture/store.py::TokenCaptureStore` is the default `TokenSink` and `TokenSource`. Per rollout: an entries JSONL, a state index (frozen/retired/incomplete flags, snapshot ID and version, per-entry sha256 digests), an intents file, an incomplete-marks file, and an fcntl lock.

Durability contract: the JSONL entry line is fsynced before `put` returns; the state index is written atomically and reconstructed from the durable JSONL tail on demand, including truncate-repair of a torn, never-acknowledged final line. `freeze_now` assigns the snapshot ID once; dangling intents (a `begin_call` with no matching entry) force `incomplete=True`. `drop` retires conditionally on `(snapshot_id, version)`, deletes payloads, and leaves a retired tombstone so a late writer of a consumed attempt cannot resurrect it. `clear_token_captures_for_rollouts` runs before dispatch for rows about to be re-attempted, since rollout IDs are deterministic and append mode would otherwise merge attempts. The file store is single-node; multi-node deployments need a shared backend that meets the protocol contracts.

## Rebuilding a trainable response

`nemo_gym/token_id_capture/builder.py`:

- `prefix_merging(entries)` is the default. It drops empty-generation calls first (a filtered call must not become its retry's parent), orders entries by prompt length, and infers each call's parent as the earlier call whose cumulative sequence (prompt + generation) is the longest strict prefix of this call's prompt, using a token trie. Multiple identical candidates quarantine the subtree instead of guessing. Retry siblings (same parent and prompt, different generation) resolve in favor of the sibling a later call extends; a retry of the final call is unresolvable and reported in `unresolved_retries`, which masks the sample.
- One chain is delivered per rollout — the chain with the earliest root — with `BuildNotes` reporting `roots`, `chains`, `delivered_fraction`, `quarantined`, `empty_generation_calls`, and `unresolved_retries`. Auxiliary calls (title generation, compaction) and parallel sub-agents are known unsupported shapes, documented in code as future work; they surface as extra chains and reduce `delivered_fraction`.
- Projection re-attaches token arrays to the original carrier items with contiguous `prompt_token_ids`, wraps them as a Gym-native Responses payload, and re-asserts prefix contiguity on the final payload.

`nemo_gym/token_id_capture/consumer.py` freezes a snapshot (from the local store or any `TokenSource`), runs build + validate + project, and catches all shape errors into a masked failed-build result. `mask_sample` is set when there are unresolved retries, more than one root, more than one chain, or the snapshot is incomplete.

`nemo_gym/token_id_capture/delivery.py::finalize_rollout_token_capture(result, source)` never raises: it leaves rollouts that already carry token IDs unchanged, replaces only `response.output`, and attaches metrics under `_ng_token_capture` plus a top-level `mask_sample` flag — placed at the top of the record so a consumer does not need to know the feature exists to find the verdict. `retire_rollout_token_capture` drops the exact frozen snapshot only after durable handoff, and never for masked or failed builds — those retain their capture evidence.

`nemo_gym/rollout_collection.py` wires this into runs: it resolves the token source, pre-clears captures for dispatched rows, finalizes each participating agent's result, aggregates mask reasons, enforces the `max_mask_fraction` / `mask_fraction_min_samples` kill switch (aborting a run that is collecting mostly token-less data), and fsyncs Gym's results JSONL before retiring snapshots.

## Configuration

The `token_id_capture` YAML block (`nemo_gym/token_id_capture/config.py`, validated with `extra="forbid"` so typos fail instead of silently not capturing): `enabled`, `all_agents`, `dir` (node-local, absolute), `sink` / `sink_kwargs` (worker factory descriptor), `rebuild_response`, `max_mask_fraction`, `mask_fraction_min_samples`. The open stack adds `lineage_store` / `lineage_store_kwargs` (#2180) and `supply_prefix_token_ids` on the vLLM model config (#2181).

## The open PR stack

### #2180 — request-time parent resolution (unmerged)

Token-prefix inference has a structural blind spot: a retried call shares its prompt with the original and differs only in generation, so prefix matching cannot tell which one the harness kept. #2180 resolves each call's parent at request time in the model server, from the request as received:

- `assistant_fingerprint(messages)` hashes model-authored turns only (assistant messages and tool calls, normalized across dialects; reasoning is deliberately excluded because harnesses need not echo it). This is the lookup key.
- `conversation_digest(messages)` hashes every turn including tool results. This is the verifier: a candidate parent must match the recorded context digest, which rejects compacted or rewritten histories whose model output is unchanged.
- The outcome is a persisted trichotomy — `ROOT` / `RESOLVED` / `UNRESOLVED(reason)` — written in the same durable sink write as the tokens. The builder gives recorded decisions precedence over inference and re-verifies every `RESOLVED` link by digest before using it. `UNRESOLVED` starts a masked fragment; Gym never crosses that boundary with prefix inference.
- Lineage stores: an in-memory bounded index for single-worker serving, and `FileLineageStore` (tails the token JSONL under the store's lock) for multi-worker. Capture with `num_workers > 1` requires a process-shared lineage store at startup.
- Record schema moves to v3 (parent decision, cumulative length, digest, continuation lookup metadata). Digests are versioned, domain-separated, and length-delimited so independent implementations hash identical bytes.

### #2181 — exact prefix supply (unmerged)

The engine rebuilds each prompt by re-rendering the whole conversation through the chat template, which can produce a different token sequence than was sampled — a tool parser truncates the assistant turn, a template re-tokenizes it differently, a reasoning template drops earlier thinking. The chain then breaks at rebuild time. #2181 removes the re-render from the continuation boundary: when request-time resolution produced a unique verified parent, the outbound vLLM request carries `required_prefix_token_ids` (the parent's exact cumulative tokens). Intent and proof are separate persisted facts — `prefix_requested` records that the extension was asked for; `prefix_supplied` becomes true only after the generation response's `prompt_token_ids` prove the served prompt extended the exact requested tokens. An unproven supply fails the call loudly (one of two deliberate exceptions to "capture never fails the model call"). Stock vLLM does not implement the extension; the backend must honor it.

### The v2 line (#2349 docs branch, unmerged)

Iterates on #2180/#2181 and is the form to assess for merge: schema v5 parent-relative prompt deltas (a `RESOLVED` continuation may store only the prompt suffix beyond its parent, cutting per-rollout storage from O(T²) to O(T); a broken delta chain masks), a schema floor of v3, `IncrementalLineageStore` with backend adapter hooks, an executable conformance kit for sink/source/lineage adapters, golden fingerprint vectors pinning the cross-repo hash contract, worker health counters, and `nemo_gym/token_id_capture/DESIGN.md` — the authoritative architecture document.

### #2278 — worker-locus staging (unmerged, second topology)

#2278 moves token custody out of Gym entirely, for trainers that own their inference workers and data plane (NeMo RL's TransferQueue is the target). The inference worker stages each call's token delta durably in framework storage before acknowledging the response; Gym's model server keeps only a token-free per-rollout capture ledger (typed commit coordinates with digests) and serves it read-only over one bearer-authenticated manifest route. Request-time resolution becomes a strict admission decision (`token_in` continuation with a verified staged-prefix chain, `text` root, or a poison row — an unresolved request is never converted into a new root). Each staged record carries a chained ancestry hash; the framework's finalizer fetches the manifest and staged rows, re-verifies every digest, and linearizes the terminal chain into a flat trainable row with per-call weight-version spans and optional routed-expert tensors. Token IDs, logprobs, and routing data never appear on any agent-facing response. The branch has evolved past its PR description: the earlier stateful custody component was replaced by the ledger (commit `1e1a16cb2`); its serving surface is deliberately narrow (non-streaming chat).

The two custody topologies — Gym-custody (#2180/#2181/v2) and worker-custody (#2278) — share the resolution machinery but differ in protocol APIs and schema numbering. Their reconciliation is the stack's main open design item, tracked in `DESIGN.md`.

## Trainer integration (NeMo RL)

- **RL #3407** (batch GRPO/PPO path): NeMo-RL stamps `_ng_rollout_id` on every row, constructs the `TokenSource` inside its rollout actor (Gym's file store by default, or a framework-provided `token_source_factory`), calls Gym's shared finalizer per rollout, and retires each frozen snapshot only after the trainer-side consumer accepts the yielded item. The rebuilt output items carry the same token fields as the native path, so the existing postprocess walk — which derives the loss mask from the user/assistant role split — consumes them unchanged. It also fixes GRPO baseline grouping to use dataset prompt identity instead of the captured first-call prompt.
- **RL #3456** (async single-controller path): the worker-custody topology over #2278. The vLLM worker hosts Gym's capture core with a TransferQueue-backed sink; rollout results carry token-free receipts; CPU finalizer actors rebuild rows via Gym's `verify_and_linearize`. A five-step A/B on GB200s (2026-08-20) measured tighter logprob agreement than the token-echo baseline (`token_mult_prob_error` ~1.014 vs 1.02–1.066).

What the integrations pin down about the contract: identity is caller-assigned; the finalizer freezes and rebuilds but never retires; retirement belongs to the consumer after durable handoff; failed or masked builds retain evidence; and health metrics (`token_capture/rebuilt_fraction`, `delivered_fraction_mean`, `parent_link_failures`, `finalize/token_in_rate`, …) are part of the contract, because the failure mode they exist for is a run that looks green while training on a fraction of each rollout.

## Guarantees and failure semantics

- Exact token identity end to end; prefix supply extends it across continuation boundaries with generation-time proof.
- Every claimed parent link is re-verified by digest before use; #2278 re-verifies full ancestry in the consumer.
- Fail-closed masking for: missing calls, dangling intents, unresolved parent boundaries, multiple roots or chains, final-call retries, digest mismatches, broken delta chains, empty generations, poisoned ledger rows.
- Two deliberate loud failures: pre-generation intent-write failure, and unproven prefix supply.
- Multi-worker serving requires a process-shared lineage store, enforced at startup; multi-node requires a shared sink backend (the conformance kit is the acceptance bar).

## Status and open questions

Merged and stable: the full Gym-custody pipeline v1 with roughly 2,500 lines of tests (transport contracts, torn-tail repair, tombstones, mask-fraction abort). Pending and well-developed: #2180/#2181 and their v2 successors. Pending and more experimental: #2278, heavily unit-tested but on a narrower serving surface, with active churn.

Open items visible in code and `DESIGN.md`: multi-chain shapes (auxiliary calls, compaction, parallel sub-agents — only one chain is delivered today; the roadmap runs measurement → auxiliary-call routing → terminal-ancestry selection → segment delivery); reconciling the two custody topologies; reasoning excluded from fingerprints (reasoning-only-differing calls mask, unmeasured on reasoning-heavy harnesses); no published masked-fraction numbers from a real harness run yet — the ~2.3 ms/call p50 serving overhead is the only measured performance claim.

## File map

| Area | Path |
|---|---|
| Identity and correlation | `nemo_gym/config_types.py`, `nemo_gym/rollout_correlation.py`, `nemo_gym/base_responses_api_agent.py` |
| Capture middleware | `nemo_gym/base_responses_api_model.py` |
| Token production | `responses_api_models/vllm_model/app.py` |
| Records, protocols, sink | `nemo_gym/token_id_capture/records.py`, `protocols.py`, `sink.py` |
| File store | `nemo_gym/token_id_capture/store.py` |
| Rebuild and delivery | `nemo_gym/token_id_capture/builder.py`, `consumer.py`, `delivery.py`, `nemo_gym/rollout_collection.py` |
| Config | `nemo_gym/token_id_capture/config.py` |
| Lineage (#2180, unmerged) | `nemo_gym/token_id_capture/lineage.py` |
| Prefix supply (#2181, unmerged) | `responses_api_models/vllm_model/app.py` |
| Worker staging (#2278, unmerged) | `nemo_gym/token_id_capture/staging/`, `control_routes.py`, `adapters/vllm.py` |
| Docs line (#2341/#2349, unmerged) | `fern/versions/latest/pages/training-tutorials/external-agent-harnesses.mdx`, `nemo_gym/token_id_capture/DESIGN.md` |
