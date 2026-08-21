# Token-ID Capture — Design

**Maintenance rule: this document is updated in the same commit as any change to the
architecture, interfaces, control flow, data flow, storage model, or error handling of this
package.** A PR that changes behavior here without touching this file is incomplete. Append a
line to the Changelog (§10) for every such change.

## 1. Purpose and the one invariant

RL training needs, for every model call an agent made: the exact prompt token ids the engine
conditioned on, the exact token ids it sampled, and their sampling-time log probabilities.
Blackbox agent harnesses (Claude Code, SWE agents, …) return text, not tokens, and reshape
conversation history between calls — so tokens must be captured at the one chokepoint every
call passes through: Gym's model server.

**The guaranteed invariant is token-chain exactness:** a delivered trajectory contains exactly
the tokens the policy emitted, conditioned on exactly the recorded context, as one contiguous
sequence. Conversation *fidelity* is not promised — fields the lineage hashes deliberately
ignore (reasoning content, an item inserted between the verified context and the echoed output)
may differ from the harness's rendering without breaking the invariant. Every layer prefers
losing a sample (masking) to fabricating one; there is no path on which a guessed or corrupted
token sequence is delivered as trainable.

## 2. Components

```
harness ──► _CaptureMiddleware ──► model route (vllm_model) ──► engine
                │                        │
                │ mints rollout/call ids │ assembles token arrays, prefix supply + proof
                ▼                        ▼
          CaptureContext ◄────── resolve_parent / register_call_intent / capture_tokens
                │                        │
                ▼                        ▼
          LineageStore  ◄───────  TokenSink.put  (awaited before the response returns)
        (read-only view)                 │
                                         ▼
                              durable log (TokenCaptureStore JSONL)
                                         │
        consumer:  TokenSource.freeze ──► builder ──► delivery ──► conditional drop
```

| Module | Role |
|---|---|
| `protocols.py` | The external contract: `TokenSink`, `TokenSource`, `LineageStore` (+ optional sink extension `begin_call`). Leaf-importable: no fastapi/ray/torch. |
| `records.py` | `TokenEntry` (schema §5), digests (`compute_digest`, `encode_token_ids`), `stamp_lineage`. |
| `lineage.py` | Canonicalization + hashing (`assistant_fingerprint`, `conversation_digest`, `FINGERPRINT_VERSION`), the matcher (`RolloutLineage`), `IncrementalLineageStore` (the base every backend subclasses: two hooks — fetch-new-entries-since-cursor and load-entry-by-ref — inherit the matcher, LRU metadata index, locks, and lazy digest-checked materialization incl. delta chains), `FileLineageStore` (the file-backed reference implementation), `InMemoryLineageStore` (tests only). |
| `sink.py` | `CaptureContext` (per-call identity + parent decision + delta mode + prefix intent/proof), `resolve_parent`, `register_call_intent`, `capture_tokens`/`commit_entry`, worker health counters (`capture_health_snapshot`). |
| `store.py` | `TokenCaptureStore`: per-rollout JSONL + flock + state (freeze/version/tombstone), intents, `sweep_retired` GC. Default `TokenSink` and `TokenSource`. |
| `builder.py` | Offline reconstruction: delta materialization, parent-link chaining with digest verification, missing-parent recovery, retry-sibling resolution, projection. |
| `consumer.py` / `delivery.py` | Freeze → build → mask decision → durable handoff → conditional retirement. |
| `config.py` | `token_id_capture` settings block; startup validation (incl. sink-requires-resolver). |
| `conformance.py` | Executable contract checks an external backend (e.g. NeMo-RL/TransferQueue) runs against its sink/source/lineage adapters. |

## 3. Control flow — one captured model call

1. Harness POSTs to `/ng-rollout/<rollout_id>/training-token-capture/v1/...`. The middleware
   mints a fresh `model_call_id`, resolves the sink (configured → installed → file store), and
   sets a `CaptureContext` contextvar. Unobserved paths under the capture prefix mark the
   rollout incomplete (an uncapturable call must not look like a complete rollout).
2. **Pre-dispatch, on the request as received** (before any dialect conversion):
   `resolve_parent()` computes the parent decision (§6) once; prefix supply and capture share
   it. `register_call_intent()` durably records "this call is about to happen" via the sink's
   optional `begin_call` — intent failure fails the call *before* generation, at zero compute
   cost.
3. The model route calls the engine. With `supply_prefix_token_ids` and a RESOLVED parent, the
   exact cumulative parent tokens ride as `required_prefix_token_ids`; the response's
   generation-time prompt ids (message-bundle first, then top-level) must extend that prefix —
   `prefix_requested` records intent, `prefix_supplied` records proof (of chain contiguity,
   not of extension support: a prefix-stable re-render also passes, which is what training
   needs). Unproven supply fails the call loudly.
4. `capture_tokens()` extracts the arrays, stamps continuation metadata
   (`stamp_continuation`: fingerprint + context digest + `FINGERPRINT_VERSION`), applies the
   delta transform when enabled (§5), stamps lineage (`cum_len`/`digest` always describe the
   FULL sequence), and **awaits `TokenSink.put` before the response returns**. That ordering
   is the happens-before edge everything rests on: the harness cannot send a continuation
   before the previous record is durable and resolver-visible.
5. On any capture failure: the model call still succeeds; `mark_incomplete` durably records
   the hole. If both put and mark fail, the dangling intent from step 2 still masks the
   rollout at freeze — the correlated double-failure cannot be silent.

## 4. Data flow — record lifecycle

Write: entry line appended to `<rollout>.tokens.jsonl` + fsync (the durability guarantee),
then the state index updated atomically without fsync (reconstructible from the tail;
fsynced on lifecycle transitions only). Intents live in a `.intents` sidecar.

Read (serving path): `FileLineageStore` tails the JSONL under a shared flock + a per-rollout
in-process lock, indexing **metadata only** (~hundreds of bytes per call: fingerprint, context
digest, `cum_len`, digest, byte offset). A RESOLVED match lazily materializes tokens from the
log at the stored offset, behind a digest interlock. The cache is true-LRU bounded
(default 65,536 rollouts); eviction of a live rollout is a performance event (cold re-tail),
never a correctness event.

Consume: `TokenSource.freeze` returns an atomic snapshot (entries + incomplete flag +
snapshot id + version); dangling intents force `incomplete`. The builder materializes delta
chains, verifies every recorded parent link by digest, reconstructs one contiguous chain, and
delivery masks anything unsafe. Retirement is a conditional `drop(snapshot_id, version)` after
durable downstream handoff; late writes bump the version so a stale drop fails and evidence is
retained. `sweep_retired(age)` is the operator GC hook for tombstones.

## 5. Record schema (`TokenEntry`)

- v3 — parent resolution (ROOT/RESOLVED/UNRESOLVED) + continuation lookup metadata.
- v4 — `prefix_requested`/`prefix_supplied` proof fields.
- v5 — `prompt_is_delta`: a RESOLVED continuation whose prompt provably extends its parent's
  cumulative tokens stores only the suffix (O(T²)→O(T) storage; config
  `token_id_capture.delta_records`). ROOT/UNRESOLVED records always store full prompts, so
  reconstruction is anchored fail-closed. Delta records require a durable-log-backed resolver.
- **Supported floor: v3** (`TOKEN_ENTRY_MIN_SCHEMA_VERSION`). No v1/v2 records were ever
  written outside development; readers refuse them, and the pre-v3 prefix-*inference*
  reconstruction path has been removed. Readers also refuse versions above what they
  understand. Additive optional fields do not bump the version; meaning changes do.

## 6. Lineage semantics

`assistant_fingerprint` (model-authored turns only, dialect-normalized, boundary-free) is the
lookup key; `conversation_digest` (every leading context item, length-delimited,
domain-separated) verifies a candidate before its tokens are reused. Outcomes, persisted on
every record with their diagnostic `parent_resolution_reason`:

- **ROOT** — no model-authored history.
- **RESOLVED** — exactly one context-verified candidate. Candidates with *identical*
  cumulative tokens (identical retries) collapse to one parent — same tokens, same
  continuation. Distinct-token candidates under one fingerprint stay ambiguous.
- **UNRESOLVED** — everything else. Never guessed across; starts a masked fragment.

The hashes are a cross-repo wire contract: `FINGERPRINT_VERSION` is stamped on entries and
gated at indexing; golden vectors (`tests/unit_tests/token_capture_golden_vectors.json`) pin
them, including cross-dialect equality. External resolvers subclass
`IncrementalLineageStore` (two hooks) rather than reimplement hashing or caching. The builder independently re-verifies every claimed link by
digest — the defense-in-depth that makes the §7 contract relaxations safe. Token-prefix
matching survives for exactly one purpose: recovering a RESOLVED link whose parent is absent
from the build (e.g. filtered for an empty generation); a missing decision masks
(`missing_resolution`) rather than infers.

## 7. External contract (Gym ↔ training framework)

`TokenSink.put` must be durable **and resolver-visible cluster-wide (per rollout key)** before
returning; "same payload" for idempotent retries means byte-identical serialization; without
CAS, conflicting same-id payloads may instead be refused by the reader (zero candidates).
`begin_call` is an optional sink extension (intent custody). Freeze is atomic with unique
entries per call id (at-least-once transports dedupe; the builder also dedupes defensively);
the strict "no writes after freeze" fence is *not* required — a racing write must merely bump
the version so a stale conditional drop fails. `LineageStore` is read-only over sink-committed
records. Full details live in `protocols.py` docstrings; `conformance.py` is the executable
acceptance gate; `allow_unresolved_continuations`, `max_mask_fraction`, and the resolver-
required startup check are the config-level guardrails.

## 8. Error handling

Philosophy: **capture failures never fail the model call** (except two deliberate cases:
pre-generation intent failure, which is free, and unproven prefix supply, which would
otherwise silently break the exactness the operator opted into). Every failure lands in one of
three durable states: a recorded hole (`mark_incomplete`), a dangling intent, or a masked
build — all of which converge on `mask_sample: true` at delivery. Torn JSONL tails from
crashes are truncate-repaired (the torn entry was never acked). `finalize_rollout_token_capture`
never raises into the collection loop. Observability: per-worker resolution/failure counters
(`capture_health_snapshot`), persisted resolution reasons, per-rollout build metrics
(`_ng_token_capture`), and the run-level `max_mask_fraction` kill switch — a run producing
mostly-masked data dies loudly instead of burning its budget.

## 9. Known limits and agreed direction

- **Yield policy (open, measured before built):** aux calls, sub-agent forks, and compaction
  mask under the one-clean-chain policy. Roadmap: measure masked-fraction on a real harness
  run → aux suppression/routing → terminal-ancestry selection (converging with #2278's gate) →
  segment delivery for compaction → think-tag normalization (with a `FINGERPRINT_VERSION`
  bump).
- The file store is single-node (flock); multi-node deployments need a shared backend
  (TransferQueue adapters implementing §7). Session affinity by rollout id keeps caches warm.
- The gate/worker-locus topology (#2278) predates several of these mechanisms and carries a
  rebase checklist (tri-state adoption, single publication door, shared custody record).

## 10. Changelog

- **`IncrementalLineageStore` extracted.** External lineage backends now subclass a base with
  two hooks (`_fetch_new_entries`, `_load_entry`) and inherit the matcher, LRU metadata-only
  index, per-rollout locking, and lazy digest-checked materialization including delta chains;
  `FileLineageStore` is the reference subclass; the conformance suite demonstrates the pattern
  with a ~15-line memory adapter.

- **v3 floor; legacy prefix-inference reconstruction removed.** No v1/v2 records exist in the
  wild; missing resolution metadata now masks (`missing_resolution`) instead of triggering
  inference. Prefix matching retained solely for missing-parent recovery.
- **Simplification pass:** removed `per_request` builder (no consumers), the caller-less
  `parent_call_id` params on `capture_tokens`/`commit_entry` (one way to declare a parent:
  the pre-dispatch resolution), and per-rollout resolver-unavailable bookkeeping.
- **Delta records (schema v5)** + true-LRU metadata-only lazy lineage index with digest-checked
  materialization; `InMemoryLineageStore` demoted to reference/tests.
- **Full remediation:** intent/commit custody (F1), lineage race fix (F2), identical-retry
  collapse (F16), startup resolver validation (F3), never-raise delivery (F6), bundle proof +
  startup validation for supply (F7), snapshot dedupe (F10), missing-parent recovery (F13),
  retirement on every path (F14), empty-delivery guard (F15), install-time sink validation,
  unobserved-dialect fail-closed, health counters + kill switch, torn-tail repair, fsync diet,
  `sweep_retired`, protocols contract rewrite, conformance kit, golden vectors.
- **Baseline:** capture core (#2124), builder (#2125), delivery (#2126), request-time lineage
  (#2180), prefix supply (#2181).
