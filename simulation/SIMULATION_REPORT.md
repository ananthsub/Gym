# Token Capture Simulation

Empirical validation of the token-id-capture stack on branch `pr-2181` (`ba88c3270`), then of the
fixes on this branch: 21 adversarial rollout scenarios through the **real** server, 9
microbenchmarks, and 4 concurrency experiments. Environment: Apple Silicon Mac, local NVMe,
Python 3.13. Companion: [STACK_REVIEW.md](STACK_REVIEW.md) (the predictions these runs tested).

Rerun from the worktree root:

```bash
.venv/bin/python simulation/sim_scenarios.py     # scenario matrix
.venv/bin/python simulation/race_experiments.py  # concurrency + soak
.venv/bin/python simulation/bench_capture.py     # microbenchmarks
```

## 1. Method

Everything between the harness and the trainer is real: the vllm_model server, the capture
middleware (requests go through the actual `/ng-rollout/<id>/training-token-capture/...` path),
lineage resolution, prefix supply and its proof check, the file store, the builder, and the
consumer. Two things are simulated:

- **A deterministic fake engine** with a concatenation-safe tokenizer — a faithful re-render
  reproduces prompt+generation exactly, so the prefix property holds by construction, and every
  mutation breaks it the way the real failure does (different bytes ⇒ different tokens).
  Configurable behaviors: honor/ignore `required_prefix_token_ids`, re-tokenize history on
  re-render, proof placement (top-level / message bundle / none), scripted empty generations.
- **A scripted blackbox harness** echoing history with controlled mutations: tool-arg
  reserialization, think-tag stripping, inserted system reminders, history edits, sub-agent
  forks, auxiliary calls, identical and divergent retries.

Every run ends with a **fabrication detector**: the engine records the ground truth of every
(prompt, generation) it served; a delivered trajectory must be a contiguous chain of exactly
those pairs. Delivering a token sequence the engine never processed is the corruption class the
stack exists to prevent — **never observed, in any scenario, including the bug reproductions.**

## 2. Scenario matrix — final state (post-fix)

Verdicts: OK = healthy path behaved as promised · SAFE_MASKED = adverse input correctly refused ·
FIXED = a previously reproduced defect no longer fires and the rollout is healthy ·
NOT_REPRODUCED = predicted defect didn't fire, behavior safe · BUG_REPRODUCED = still reproduces.

| Scenario | Verdict | Statuses | Delivered | What it shows |
|---|---|---|---|---|
| faithful_linear | OK | root,resolved,resolved | 3/3 | Baseline: faithful harness, supply on. |
| tool_args_reserialized | OK | root,resolved | 2/2 | Reordered tool-call JSON still resolves. |
| drift_no_supply | SAFE_MASKED | root,resolved,resolved | 1/3 | Engine re-tokenizes history: lineage resolves but the chain breaks; digest check masks. Lineage alone cannot save drift. |
| drift_with_supply | OK | root,resolved,resolved | 3/3 | **Flagship value demo:** same drift, exact-prefix supply preserves the chain. |
| think_stripped | SAFE_MASKED | root,unresolved,resolved | 1/3 | Inline `<think>` stripped: fingerprint miss, no supply — the documented 2181 gap. |
| reminder_inserted | OK | root,resolved,resolved | 3/3 | Adjacency hole: item between context and echo still RESOLVES; with supply it has no token effect. |
| history_edited | SAFE_MASKED | root,res,res,unresolved | 3/4 | Compaction-style edit: digest verification refuses the parent. |
| fork_subagents | SAFE_MASKED | root,resolved,resolved | 2/3 | Two children of one parent → two chains → masked (F11 policy). |
| aux_title_call | BUG (F11) | root,res,res,root | 3/4 | One auxiliary call masks an otherwise perfect rollout — the open policy question. |
| retry_nonfinal_identical | **FIXED** (F16) | root,res,res,resolved | 3/4 | Was: identical retry → ambiguity → masked-from-there. Now: identical-cum candidates collapse; fully trainable. |
| retry_nonfinal_divergent | OK | root,res,res,resolved | 3/4 | Divergent retry chains fine; the abandoned attempt correctly excluded. |
| retry_final_identical | NOT_REPRODUCED (F15) | root,resolved,resolved | 2/3 | Predicted unmasked-empty delivery didn't fire; safely masked. |
| retry_only_call_identical | NOT_REPRODUCED (F15) | root,root | 1/2 | Second construction also safe. |
| empty_generation_parent | BUG (by design) | root,resolved,unresolved | 1/3 | Dies at *resolve*-time ambiguity: the candidates carry different tokens, so refusing is correct. (The build-time F13 path is fixed.) |
| engine_ignores_prefix_stable | OK | root,resolved,resolved | 3/3 | Backend ignores the field but is prefix-stable: proof passes. `prefix_supplied` is contiguity proof, not extension proof. |
| proof_bundle_shape | **FIXED** (F7) | root,resolved,resolved | 3/3 | Was: message-bundle proof spuriously hard-failed every call. Now accepted; fully trainable. |
| sink_outage_final_call | NOT_REPRODUCED (F1) | root,resolved | 2/3 masked | Was: unmasked truncated delivery (the SEV-1). Now the dangling intent masks it. |
| sink_outage_pre_dispatch | OK | root,resolved | 2/2 | Backend already down: intent fails pre-generation, the call fails at zero compute cost, delivered 2/2 is the complete story. |
| sink_outage_mid_rollout | SAFE_MASKED | root,unresolved | 1/3 | Mid-rollout loss: next call can't resolve its parent → masked (2180's repair). |
| custom_sink_no_lineage | **FIXED** (F3) | — | — | Was: all-UNRESOLVED with zero log lines. Now refused at startup. |
| duplicate_snapshot_entry | **FIXED** (F10) | root,resolved,resolved | 3/3 | Was: duplicate → phantom root → masked. Now deduped and **trainable**. |

Final tally: **7 OK, 5 SAFE_MASKED, 4 FIXED, 3 NOT_REPRODUCED, 2 remaining reproductions that are
policy decisions, 0 surprises.**

## 3. Microbenchmarks — 9 measured, both flags fixed

| Benchmark | Result | Status |
|---|---|---|
| Store append (1k/8k/64k-token entries) | p50 0.83 / 0.86 / 2.8 ms (p99 ≤ 3.5 ms) | PASS |
| Append growth (200-call rollout) | 0.7 → 2.6 ms/append; 35 MB JSONL; no state-rewrite pathology | PASS |
| Fingerprint / digest hashing | 496–841 / 245–285 MB/s | PASS |
| `stamp_continuation` over 100 calls | 39 ms cumulative (quadratic but tiny) | PASS |
| Token digest at 65k tokens | 7.7 ms — vectorized pack **byte-identical, 2.1×** (3.8 ms) | was FLAG → **fixed** |
| Lineage resolve (50 × 8k entries) | cold 32 ms, warm 0.47 ms | PASS |
| End-of-life freeze+build (50 calls/64k ctx) | 926 ms, 90% trie insertion that RESOLVED entries never consult | was FLAG → **trie now skipped** |
| Rebuilt row vs raw entries | ratio 0.98–0.99; both O(N²) with chain depth | informational |
| Serving-path overhead (real middleware, 10-turn chain) | **+2.29 ms/call p50** vs capture off | PASS |

## 4. Concurrency & soak — 4 experiments

| Experiment | Result | Verdict |
|---|---|---|
| F2 concurrent-resolve poisoning | Pre-fix: reproduced at 10% under scheduler amplification, exact predicted corruption (`by_fingerprint = ['call-14','call-14']`, permanent ambiguity); plus a second transient torn-read hazard on the early-return path. **Post-fix: 0/220** incl. the amplified config. | REPRODUCED → FIXED |
| Freeze racing concurrent puts (50 iter) | 0 violations: pre-freeze puts in snapshot; post-freeze writes refused; late mark bumps version; conditional drop behaves. | PASS |
| Cross-process read-after-write soak | 380/380 RESOLVED with correct parent; visibility p50 0.26 ms, p99 0.53 ms. The happens-before edge holds. | PASS |
| Executor saturation | Throughput flat from K=8 (~1,400–1,600 ops/s, fsync+flock + 12-thread default executor); K=512 only buys p99 (315 ms). A 20 ms-slow sink collapses throughput to ~400 ops/s — engineer for slow storage: dedicated capture executor + the kill switch. | CONFIRMS |

## 5. What the data changed

1. **F1 confirmed at SEV-1, then closed.** The silent case existed only because the harness
   received a response whose record was lost. The intent/commit custody record (a durable
   `begin_call` intent pre-dispatch; a dangling intent at freeze masks; pre-generation intent
   failure fails the call at zero cost) is #2278's admit→commit pattern minimized — the two
   topologies now share one custody model.
2. **F2 went from traced to reproduced to dead** (0/220 post-fix, throughput unchanged).
3. **F16 discovered and fixed:** identical mid-rollout retries (the most common retry shape)
   masked everything after the retry; identical-cumulative candidates now collapse.
4. **F15 was already fixed** — the predicted defect would not reproduce; a guard and regression
   tests added anyway.
5. **Both perf flags were validated fixes, not hypotheses:** vectorized digest (byte-identical,
   2.1×) and trie-skip (~90% of end-of-life build cost).
6. **The empty-generation case dies earlier than the review traced** — at resolution, where
   refusing is genuinely correct; the build-time F13 path is fixed independently.
7. **The core invariant held everywhere:** across all scenarios, pre- and post-fix, no delivered
   token sequence ever diverged from what the engine actually processed. The stack's failures are
   yield failures — never fabrication. That was the design's central claim; it is now demonstrated.

## 6. Fix inventory

Two commits on this branch: `720efbf6e` (F1 custody, F2 lock+dedupe, F16 collapse; ~91 lines) and
`86742d332` (full remediation: F3, F6, F7, F10, F13, F14, F15-guard, R2, R4, F4 counters + kill
switch, F5 cache bound + `sweep_retired`, `protocols.py` contract rewrite, conformance kit, golden
vectors; +1,464/−163 across 26 files). Verification: **2,931 tests pass** (one pre-existing enroot
environment failure, unrelated), scenario matrix clean, races 0/220, soak 380/380. Destination of
each change: [STACK_REVIEW.md §14](STACK_REVIEW.md#14-change--pr-mapping).

A third commit (`d1dc93747`) adds the scale hardening: true-LRU metadata-only lineage index with
lazy, digest-checked token materialization (eviction mid-rollout is now provably a performance
event, not a correctness event), `InMemoryLineageStore` demoted to a reference/test building
block, and **delta records (schema v5)** — RESOLVED continuations store only the suffix beyond
their parent's cumulative tokens (O(T²)→O(T) storage), with fail-closed chain reconstruction in
the builder and the resolver. 2,942 tests pass; the scenario matrix stays clean.

A cleanup pass (`e1915d0be`) removed obsolete code (`tokens_for`, the `_infer_parent`
forwarder), corrected every stale comment the fixes had invalidated, and added directional
simulation coverage for the previously deferred limits: a delta-mode baseline scenario
(identical 3/3 delivery), a storage comparison (12 turns: **2.35× smaller with delta records**,
ratio grows with rollout length — `directional_results.json`), a kill-switch signal check
(total sink outage → mask fraction 1.0, trips the 0.5 guard), and an explicit test that the
in-memory reference resolver refuses delta records. 2,943 tests; 22 scenarios, 0 surprises.

A simplification pass (`716fab0e6`) then removed what the fixes made obsolete: the `per_request`
builder (zero consumers; superseded-in-principle by future segment/terminal-ancestry delivery),
the caller-less `parent_call_id` compat params on `capture_tokens`/`commit_entry` (a second,
verification-bypassing way to declare a parent — one way remains: the pre-dispatch
`LineageResolution`), and the per-rollout resolver-unavailable bookkeeping (the state now requires
an explicit startup opt-in, so one process-level note + counters suffice). Deliberately kept:
`stamp_lineage`'s checks (the only stamp-time enforcement — pydantic doesn't re-validate on
mutation), the legacy prefix-matching path (gating/removal is a destination-PR decision), and
`LineageIndex` (consolidation into the demoted in-memory reference recommended for #2180).
