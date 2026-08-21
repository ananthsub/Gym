# Token Capture Stack Review

A three-lane deep review of the token-id-capture stack — base mechanism (#2124–#2126) and the
lineage/rebuild layer (#2180–#2181) — covering soundness, reliability, and performance, with a test
plan and a change → PR mapping. Companion document: [SIMULATION_REPORT.md](SIMULATION_REPORT.md)
(the empirical validation of every claim marked *verified* below).

- PRs: 2124 · 2125 (merged), 2126 · 2180 · 2181 (open); #2278 (gate layer) reviewed in §12.
- Method: three independent review lanes over full diffs + main sources, findings cross-checked,
  highest-severity findings re-verified against source, then validated by simulation.

**Verdicts.** *Sound?* Yes, with a short must-fix list — fail-closed discipline is unusually
consistent and no wrong-parent RESOLVED path was found. *Reliable?* Mostly, but it degrades
silently — no aggregate metrics, no kill switch, three unbounded-growth paths (all now fixed on
the `sim-fixes` branch). *Performant?* Fine locally; ceilings are known and measured.

---

## 0. Top findings scoreboard

SEV-1 = can corrupt training data or permanently lose samples silently. SEV-2 = burns runs,
resources, or debuggability. SEV-3 = contract/design debt. All code-traced; several empirically
reproduced and fixed (see status notes).

| ID | Sev | Layer | Finding |
|---|---|---|---|
| F1 | SEV-1 | 2124/sink | **A rollout can lose its last call and nobody can tell.** Saving a record and recording the failure share a backend, so they fail together — and then no trace remains. Mid-rollout the next call catches it (parent unresolvable → masked); the *final* call has no next call. The evidence describes a rollout with one fewer turn: one clean chain, delivered-tokens metric 100%. The trainer gets an unmasked trajectory missing its last turn, with the full-behavior reward. *Empirically reproduced; fixed via intent/commit custody (durable `begin_call` intent pre-dispatch; dangling intent at freeze masks; pre-generation intent failure fails the call at zero compute cost).* |
| F2 | SEV-1 | 2180/lineage | **A timing race permanently breaks parent lookups for a rollout.** Two same-rollout calls resolving concurrently on one worker both add the same call to the lookup index; it is listed twice, so every later continuation sees "two candidates," refuses to guess, and resolves UNRESOLVED forever. Parallel sub-agents are exactly this traffic. *Reproduced (10% under scheduler amplification, exact corrupted state `['call-14','call-14']`); fixed with a per-rollout in-process lock + candidate dedupe: 0/220 post-fix. A second transient torn-read hazard on the early-return path was found and is covered by the same lock.* |
| F3 | SEV-2 | 2180/config | **Forgetting one config line silently disables the feature.** Custom sink + single worker + no `lineage_store` → every continuation resolves "unresolved" with zero log output; every multi-call rollout masks. *Fixed: startup now refuses the configuration (escape hatch `allow_unresolved_continuations: true`), and the no-resolver branch warns once per rollout.* |
| F4 | SEV-2 | cross | **Nothing stops a run producing garbage.** A dead sink burns the full compute budget on masked rollouts; the resolution *reason* is discarded at commit; the only aggregate mask signal was an accident (`mask_sample` being a bool summed into agent metrics). *Fixed: worker counters + `capture_health_snapshot()`, persisted `parent_resolution_reason`, and a `max_mask_fraction` kill switch in the collector.* |
| F5 | SEV-2 | 2180+2126 | **Three things grow without bound:** (a) the lineage resolver's cache holds full token arrays per rollout, never evicted; (b) masked/failed rollouts keep capture files forever with no GC — failure spirals disk usage into more failure; (c) retired rollouts leave tombstone files in one flat directory forever, worst on NFS/Lustre. *(a) fixed properly: the lineage index is now metadata-only (~hundreds of bytes/call) with true LRU eviction (asserted by test — the first bound shipped FIFO, which would have evicted the longest-lived rollout first) and a 65,536-rollout default; a RESOLVED match materializes tokens lazily from the durable log with a digest interlock, so eviction mid-rollout is a performance event, never a correctness event. (b)/(c) get a `sweep_retired` GC hook; retention policy remains the operator's.* |
| F6 | SEV-2 | 2126 | **A "never raises" function can crash the run.** The delivery step's first line — parsing the rollout id — runs before its `try:` and deliberately raises on a malformed id; one bad `_ng_rollout_id` echoed into a result row kills the collection loop. *Fixed: guarded in delivery and in the collector's retire branch (skip + retain evidence).* |
| F7 | SEV-2 | 2181 | **Prefix supply rejects one valid proof shape and discovers broken backends the expensive way.** The proof check read only top-level `prompt_token_ids` though the extractor treats the message-level bundle as primary; incapable backends were discovered per-call post-generation as 500 loops; supply + responses-native errored at request time, not startup. *Fixed: bundle proof accepted (mirroring extractor source order), responses-native rejected at construction, actionable error text; a live startup probe remains deferred.* |
| F8 | SEV-3 | protocols | **One contract clause cannot be implemented on a distributed backend.** "No write may succeed after freeze returns" needs cluster-linearizable fencing (TransferQueue has no CAS). Training safety only needs: a late write never enters an already-consumed snapshot. Relaxation: late writes may land but must bump the version so the conditional delete backs off and evidence is kept; attempt-scoped rollout ids make retirement plain namespace clearing. *Fixed as documentation: the relaxed fence is now the sanctioned strategy in `protocols.py`.* |
| F9 | SEV-3 | 2180 | **The hash algorithms became a cross-repo contract with no version number.** An external resolver must reproduce Gym's fingerprint/digest byte-for-byte or everything silently resolves unresolved. *Fixed: `FINGERPRINT_VERSION` stamped on entries and gated at indexing; golden vectors exported (cross-dialect fingerprint equality pinned); conformance kit ships the matcher-reuse pattern.* |
| F10 | SEV-3 | 2125 | **A transport that delivers a record twice ruined a good rollout.** A duplicated snapshot entry became a phantom second chain-start → masked; and "same payload" idempotency was byte-level including the timestamp. *Fixed: snapshots dedupe by call id + payload; conflicting same-id payloads mask; payload equality defined in the contract. The duplicate-entry rollout is now trainable, not just unmasked.* |
| F11 | SEV-2 | 2125/policy | **How many good rollouts is the strictness throwing away?** Anything that isn't one clean chain masks: aux calls (title generators), parallel sub-agents, compacted context, retried final calls. If Claude Code's aux calls aren't suppressed, potentially every rollout masks (unconfirmed). **Action item: measure masked-fraction and reasons on a real harness run before #2126 merges.** Knobs on the table: aux-prompt filters, tree/forest delivery with broadcast reward, orchestrator-vs-subagent selection. *Open — deliberately blocked on the measurement.* |
| F12 | CLEAR | 2180 | **The truly dangerous bug is not there.** Matching a request to the *wrong* parent — splicing in tokens from a conversation that never happened — was hunted across every canonicalization path with adversarial inputs and not found. Every mismatch matches correctly or safely refuses. The remaining gray zone (right parent, harness edited details the hashes ignore — §4.2) never breaks the token sequence. The builder independently re-verifies every link by digest — a second lock on the same door. |
| F13 | SEV-2 | 2180/builder | **A missing parent was punished like a lying parent.** A recorded parent absent from the build (an empty-generation call, filtered) quarantined the child's whole chain, though prefix matching would safely reattach it. *Fixed: missing-parent falls back to prefix matching; quarantine is reserved for digest mismatch (actual conflict).* |
| F14 | SEV-3 | 2126 | **Token-native agents under `all_agents` left capture debris.** The finalizer's early return on re-finalization handed the collector no snapshot to retire, so redundant records accumulated. *Fixed: re-finalization still freezes (idempotent) and returns the snapshot, so retirement works on every path.* |
| F15 | SEV-3 | 2125 | **An identical-token retry could deliver an empty row marked trainable** (per a review thread). *Could not be reproduced in two constructions — likely fixed by earlier delivery hardening; a one-line "empty delivery is never trainable" guard added regardless, plus regression tests.* |
| F16 | SEV-2 | 2180/lineage | **An identical retry anywhere masks everything after it** (discovered by the simulation). A timeout + resend that samples the same answer commits two identical calls; the next continuation matches both, the resolver refuses to guess, and the rollout is UNRESOLVED from there — a regression vs the legacy builder's retry-sibling resolution. *Fixed: candidates with identical cumulative tokens are the same tokens — collapsed to one parent instead of declared ambiguous. Divergent retries were already safe (the echo identifies the kept copy).* |

---

## 1. Base mechanism (#2124–#2126) · soundness

**1.1 Contract vs implementation.** All eight documented `TokenSink`/`TokenSource` invariants are
actually enforced by the file store (idempotent put via digest index; conflict marks incomplete and
raises; put-after-freeze fails; freeze idempotent and atomic under the same exclusive flock;
conditional drop with tombstone; durable incomplete markers). The gaps were in the prose, all now
fixed in `protocols.py`: the "durable put" vs "close() flushes pending work" contradiction;
mark-incomplete-after-freeze being load-bearing (it bumps the version that invalidates stale
retirement) but unspecified; snapshot entry uniqueness and byte-level payload equality unstated.

**1.2 Crash & durability.** The write order (JSONL append+fsync → state tempfile+fsync+replace+dir
fsync) with tail re-indexing is coherent: a crash between the writes recovers, and a crash before
the entry fsync means the response was never acked, so the lost generation never entered later
context. Edge cases: a torn tail line used to poison the rollout id (now truncate-repaired — the
torn entry was never acked); a crash inside `_drop` leaves a recoverable-but-fail-closed state; and
flock on Lustre is either loud (no `flock` mount option) or *unsafe* (`localflock` = node-local
locks) — the node-local single-writer constraint must be enforced or documented hard.

**1.3 Builder audit.** Fifteen edge cases traced; the trie-based longest-strict-prefix matcher is
sound (equality-with-empty-interstitial allowed; truncated-generation children fork-and-mask;
quarantined subtrees excluded; newer schema records rejected at read; `created_at` only affects
labeling when already masked, so clock skew cannot corrupt data). The genuine issues were policy
(F10, F11, F13, F15 — see scoreboard).

**1.4 Delivery (#2126).** Ordering is right: results-file flush+fsync strictly before `drop`; a
crash between them leaks evidence but never duplicates training rows. Pre-dispatch "no source, no
dispatch" holds; installed sources are used-not-closed. F6/F14 were the open items (fixed).

## 2. Base mechanism · reliability

One consistent pattern: catch, warn, mark incomplete, never fail the model call — and the chain
from builder exceptions to masked delivery holds. Beyond F1/F4: `install_token_sink` validated
nothing (fixed — refuses a sink lacking `put`/`mark_incomplete`/`close`); calls to unobserved
dialects under the capture prefix were forwarded with nothing recorded — a silent hole (fixed —
best-effort mark-incomplete); token-free backends burn whole runs before being noticed (the F4
counters + kill switch are the mitigation); the capture contextvar does not survive
shared-batching-worker server designs (documented constraint). Same-rollout concurrency serializes
on the exclusive flock; freeze racing an in-flight put is sound.

## 3. Base mechanism · performance (measured)

| Cost | Shape | Status |
|---|---|---|
| Record format stores full prompt ids per call | O(T²) bytes/rollout (35 MB for a 200-call growing rollout) | The long-pole; delta-encoding becomes safe once verified links are the norm. Deferred. |
| `append`: fsyncs under the per-rollout flock | 0.8–2.8 ms/call local; ~3 fsyncs | Append-path state fsync removed (index is reconstructible); entry-line fsync retained — that is the durability guarantee. |
| Freeze + build | 926 ms at 50 calls/64k ctx — 90% was trie insertion | Trie now skipped for all-resolved snapshots. |
| Rebuilt row repeats the running prompt per item | O(N²) per row (≈ raw entries, ratio 0.99) | Representation decision deferred; measured, not fixed. |
| Serial finalization + per-rollout results fsync in the collector | run-tail | Deferred (collector restructure). |

## 4. Lineage layer (#2180–#2181) · soundness

**4.1 The wrong-RESOLVED hunt: clear.** The fingerprint's boundary-free concatenation is what makes
Chat/Anthropic/Responses tool shapes hash identically, and length-delimited text parts prevent
aliasing; distinct-call collisions land in one bucket and fail the single-candidate check. Tool-arg
reserialization canonicalizes; float re-typing, id renumbering, whitespace trims, compaction,
multimodal changes, and tool-result edits all land safe-UNRESOLVED. The builder independently
re-verifies every claimed link by digest — the defense-in-depth that makes every §7 relaxation safe.

**4.2 The drift-RESOLVED class — the real invariant.** A narrow class remains where the *right*
parent resolves but the harness's conversation was mutated in ways both hashes ignore: an item
inserted between the verified context and the echo (the adjacency hole); reasoning-item content;
Anthropic `tool_result.is_error`; empty assistant messages; the deprecated message-level
`function_call` field. None break token-chain exactness — the invariant training needs — and the
docs now state it precisely: *the trained sequence is exactly what the policy emitted over exactly
the recorded context*; conversation fidelity is not promised.

**4.3 Notable findings.** The inline-`<think>` gap undercuts 2181's headline case: with
`uses_reasoning_parser=true` think-text lives in message *content*, so a harness stripping it
defeats both lineage and supply in the most common reasoning-model dialect (documented; simulation
confirms safe-masking). Cross-dialect switching mid-rollout misaligns context length →
safe-UNRESOLVED (docstring overclaim fixed). The envelope hash is internally consistent but a Gym
upgrade adding a defaulted tool-model field breaks lineage across the deploy boundary (pairs with
F9 versioning). `InMemoryLineageStore` is never selected by shipped serving code; its eviction is
insertion-order (the longest-lived rollout is evicted first) and its docstring claimed a fallback
that v3 records refuse (fixed). 2181 mechanics otherwise check out: resolve-before-dispatch
consumed after body reshaping; streaming is buffer-then-replay; the tokenize fallback is
unreachable for supplied calls; `prefix_supplied` proves *chain contiguity*, not that the engine
honored the field — a prefix-stable re-render also passes, which is what training needs (now
documented).

## 5. Lineage layer · reliability

F2 and F3 were the headline items (fixed, verified). No timeout exists on the resolve path (a
wedged NFS flock pins one of ~32 default-executor threads; deferred with the executor guidance
below). Fail-closed 500 on unproven prefix is the right default — the harness never received the
response, so no token hole exists — but discovery is per-call (F7 probe deferred), and the errors
are now actionable. Worker restarts: the file resolver re-tails correctly; retirement tombstones
invalidate warm cursors (tested). Behavior change worth a release note: `freeze` after `drop` now
raises instead of returning an empty snapshot.

## 6. Lineage layer · performance (measured)

~5 canonicalize+SHA-256 passes over the conversation per call → O(L²) per rollout, measured at
39 ms *total* for a 100-call rollout — real but tiny (incremental hashing deferred on that basis).
`_request_messages` was computed twice per request (fixed — once). `encode_token_ids` was a
per-token pack loop — 7.7 ms at 65k tokens (fixed — vectorized, byte-identical, 2.1×). Four full
copies of the cumulative array per supplied call (~3 MB churn at 100k tokens; deferred, minor).
`required_prefix_token_ids` rides alongside `messages` (≈2–2.5× request payload for long
continuations; accepted and documented; by-reference supply is the Topology-B fix).

## 7. Gym ↔ RL contract satisfiability

Nineteen obligations across the three protocols + wiring. Loudly enforced: per-worker construction,
the multi-worker process-shared guard, pre-dispatch source check, freeze/drop semantics. Previously
stated only in PR bodies or nowhere — now in `protocols.py`: the happens-before edge (top billing:
a harness cannot send a continuation before the previous record is durable *and resolver-visible*;
a transport that acks early shows up as a load-dependent trickle of masked samples), payload
equality, mark-incomplete-vs-late-put ordering, entry size expectations, engine-side
`stamp_continuation` duty, and the relaxed freeze fence (F8). Key relaxations, all training-safe
because the builder re-verifies by digest: the entry+index "transaction" is really an ordering rule
(a single KV write whose tags are the index satisfies it); duplicate detection without CAS moves to
the reader (conflicting copies = zero candidates); visibility is required per rollout key only.
Don't ask implementers to reproduce the hashing — hand them the matcher (leaf-importable) plus
golden vectors, and the **conformance kit** (`nemo_gym/token_id_capture/conformance.py`, 10
checks) as the acceptance gate for any adapter.

## 8. Operations: failure modes & observability

| Failure | Run experiences | Now visible? |
|---|---|---|
| Sink down (put fails, mark succeeds) | rollouts masked, compute wasted | failure counters + kill switch |
| Sink down and mark fails, final call | was: silent truncation (F1) | masked via dangling intent |
| Sink/lineage slow | 1:1 latency; timeout→retry→mask cascades | measured; executor guidance |
| Disk full | F1 path + retention spiral | sweep hook; counters |
| flock hang (NFS) | executor starvation stalls all `to_thread` users | known ceiling; deferred timeout |
| Worker restart mid-rollout | file store fine same-node; cross-node needs shared backend | unchanged (by design) |
| Harness retries / duplicate dispatch / forks | correct-by-masking; identical retries now chain (F16) | per-rollout notes + run metrics |
| 2181 vs incapable backend | hard-fail loop after paying generation | actionable errors; probe deferred |
| Schema skew in rolling upgrade | masked until reader upgrades | policy documented (reader-first) |

## 9–10. Test plan & microbenchmarks

The ~35-test plan (conformance, crash injection, concurrency, property-based builder, adversarial
echo matrix, prefix-supply faults, degradation visibility) and the 14-benchmark suite with
red-flag thresholds are implemented in large part by the simulation harness in this directory —
see [SIMULATION_REPORT.md](SIMULATION_REPORT.md) for the results that closed them out.

## 11. PR-by-PR review

Superseded by §14: every item from the original per-PR improvement lists is either implemented on
this branch (and mapped to its destination PR below) or explicitly deferred with rationale.

## 12. The gate layer (#2278) & the framework side

Gym #2278 (+12.6k lines) is the Topology-B implementation: the inference worker saves token data
straight into the framework's storage and Gym supervises — capability-authenticated admission,
worker-side durable staging before the response releases, commit with cross-worker lineage
publication, token-free receipts at seal, terminal-ancestry rebuild. Consumers: RL #3455 (closed,
gate-authoritative TransferQueue integration) and RL #3407 (open Topology-A slice). **#2278/#3455
predate the current 2180/2181 push** — findings touching lineage shape are pending-rebase items,
not design conflicts. What the gate resolves: F1 on its path (transactional poison states), TTL/
tombstone lifecycle, gate metrics, staging-digest golden vectors. Top risks: the shared-state file
rewrites everything under one global lock ≥3× per model call (a scalability wall against the
4k–65k target); a seal retry after the 5-minute tombstone TTL loses the receipt and the staging
keys unrecoverably; lineage JSONLs are never cleaned on the gate path; `claude_code_agent` has no
capability delivery path yet; and an integrator now faces two complete capture systems whose
trainability policies differ (the gate's terminal-ancestry selection is arguably the better answer
to F11 — make it one policy).

**Rebase checklist (12.4):** adopt the tri-state in admission (an unprovable continuation must say
so, not silently start a new root); answer the parent question once, pre-dispatch; close the
second lineage-write door (publish through the sink path, or keep the public interface a pure
reader); carry the v3/v4 fields through gate commits with one meaning for `prefix_supplied`; unify
the two `vllm_model` prefix-supply branches; map receipt ancestry onto the builder's statuses;
extend the golden vectors to the lineage hashes; retarget #3455's proxies and prune its workaround
ledger. New addition: adopt the same intent/commit custody record the base path now has — they are
the same model at two sizes.

## 13. How the design got here

Every open issue from the PR threads is folded into the findings above. The history worth keeping:
the external contract was shaped by its first consumer (the installed-sink hook originally wasn't
read; `mark_incomplete` wasn't on the protocol; `commit_entry`, `schema_version`,
`rebuild_response: false`, explicit `_ng_rollout_id`, and `all_agents` all came from the
integration threads); the lineage safety properties were earned under attack (the context digest
exists because a working parent-splice exploit was demonstrated; output-item canonicalization and
resolve-before-dispatch because reproduced round-trips missed parents); and two roads-not-taken
deserve ADRs — the response-object-as-canonical-carrier alternative (declined for the flat
`TokenEntry`), and first-class rollout/model-call identity (agreed and deferred; several §7 wiring
subtleties are symptoms).

## 14. Change → PR mapping

Every filed finding, performance issue, and documentation gap has a working, simulation-verified
implementation on this branch (`720efbf6e` custody/race/retry + `86742d332` full remediation;
2,931 tests green, 21-scenario matrix clean, races 0/220). This branch is the executable spec;
the table is where each piece lands.

| Destination | Changes (all implemented & verified here) |
|---|---|
| **New follow-up PR on the merged base** (#2124/#2125 are merged) | Custody & store hardening: `begin_call` intents + dangling-intent masking (F1); `register_call_intent` wiring; torn-tail repair; append fsync diet; `sweep_retired` (F5b/c); `install_token_sink` validation (R2); unobserved-dialects mark incomplete (R4); capture/resolution counters + `capture_health_snapshot()` (F4 worker side); single `_request_messages` pass; vectorized `encode_token_ids`. Builder: snapshot dedupe (F10), missing-parent recovery (F13), trie-skip, empty-delivery guard (F15). Contract: `protocols.py` rewrite (F8), conformance kit, golden vectors (F9). |
| **#2126** (open — absorb) | Finalize never raises (F6); retirement on every path (F14); mask-fraction kill switch + `max_mask_fraction`/`mask_fraction_min_samples` config (F4 run side). |
| **#2180** (open — absorb) | Per-rollout lock + candidate dedupe (F2); identical-cumulative collapse (F16); sink-without-resolver refused at startup + `allow_unresolved_continuations` (F3); resolver cache bound (F5a); `FINGERPRINT_VERSION` stamped and gated (F9); `parent_resolution_reason` persisted; loud `resolver_unavailable`; lineage docstring corrections + the stated invariant. |
| **#2181** (open — absorb) | Message-bundle proof (F7a); responses-native rejected at startup (F7b); actionable proof errors (F7c); supplied/eligible/total accounting; dead `LOG` removed; `prefix_supplied`-is-contiguity-proof documented. |
| **#2341 / #2349** (docs PRs) | Fern updates to write there: `begin_call` custody, `allow_unresolved_continuations`, the kill switch, conformance-kit usage, golden vectors as wire contract, the invariant. |
| **#2278 / RL #3455** | The §12.4 rebase checklist, plus: adopt the shared intent/commit custody record. |
| **New follow-up PR: delta records (schema v5)** | `token_id_capture.delta_records`: a RESOLVED continuation whose prompt provably extends its parent's cumulative tokens stores only the suffix — per-rollout storage O(T²)→O(T); digests still cover the full sequence; the builder and the lazy lineage index reconstruct by walking the chain; broken chains mask fail-closed. Plus the lazy metadata-only lineage index + true-LRU eviction (which land with #2180 if it moves faster). |
| **Deliberately deferred** | Incremental digest caching (measured 39 ms/100-call rollout); parent-token copy elimination; `sink.py` rename; bounded-concurrency finalization; live startup capability probe; `mark_incomplete` local-spill channel; kill-switch collection-loop test (the sim now demonstrates the signal directionally: a total sink outage yields mask fraction 1.0, tripping the 0.5 guard; the loop-level test still belongs with #2126's harness); F11 policy knobs — blocked on the real-harness masked-fraction measurement, the one outstanding empirical to-do. Yield-policy roadmap (measure → aux routing/suppression → terminal-ancestry mode → segment delivery for compaction → think-tag normalization with a FINGERPRINT_VERSION bump) is agreed direction, not yet implemented. |

Post-remediation simulation state: 7 OK, 5 SAFE_MASKED, 4 FIXED (F3, F7, F10, F16),
3 NOT_REPRODUCED (F1 now masks; F15 twice), and 2 remaining reproductions that are decisions
rather than defects: F11 (aux-call masking — policy) and the empty-generation resolve-time
ambiguity, where the candidates carry *different* tokens and refusing to guess is correct.
