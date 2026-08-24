# Auxiliary model calls the server does not understand

Branch `model-aux-calls`. Tracking doc for one issue: a blackbox harness's utility calls
to endpoints Gym's model server does not serve currently mask the entire rollout, and
they should not. This document states the problem with evidence, the two-stage fix, and
the piece of #2180 worth splitting out early because stage 2 depends on it. Companion to
the terminal-attribution design doc on `ansubramania/terminal-attribution` (PRs #2675,
#2676).

## 1. The problem

A blackbox harness is configured with exactly one base URL — the capture-prefixed one
(`/ng-rollout/<id>/training-token-capture`) — so **every** HTTP request it makes lands
under the prefix: generation calls, but also token-count probes, model discovery,
anything else its SDK does.

The capture middleware recognizes three observed paths (`/v1/responses`,
`/v1/chat/completions`, `/v1/messages` — `_OBSERVED_PATHS`,
`nemo_gym/base_responses_api_model.py`). Any *other* prefixed request triggers
`token_sink.mark_incomplete(rollout_id, "")` before forwarding. An incomplete capture
masks the rollout at build time, unconditionally.

Consequence: **one stray utility call costs the whole training sample.** Concrete
suspect: the Claude Code CLI calls `/v1/messages/count_tokens` (version-dependent). If it
does so against a Gym server, every Claude Code rollout is marked capture-incomplete
today — before any trajectory-building logic runs at all. Confirming this is a
live-smoke-test item (one rollout + capture-log inspection); it may be silently masking
100% of yield in some configurations.

## 2. Why the blanket rule exists — and its precise scope

The incomplete marker guards a real hazard, stated in `sink.py`'s `_capture_missing`
docstring: an **uncaptured generation** whose output re-enters the conversation would
ride into the next call's prompt as mask-0 tokens — policy tokens trained as if the
environment produced them. Silent corruption, the worst failure class.

But the hazard exists only for calls **whose output can enter the chain**. A token
count, a model listing, an embeddings vector — their outputs structurally cannot appear
in a prompt. Marking the rollout incomplete for them buys zero protection and costs the
sample. The rule is a blunt overapproximation: it conflates "we did not capture a
generation" (dangerous) with "we saw a request that is not a generation" (harmless).

## 3. The fix, in two stages matching the evidence available

### Stage 1 — non-generative endpoint safe-list (independent, small, now)

Prefixed requests to known non-generative paths (`**/count_tokens`, `/v1/models`,
embeddings-style endpoints) forward without capture and **without the incomplete
marker** — at most a per-rollout counter for observability. Classification is by
endpoint semantics, knowable statically. Unknown paths keep today's conservative
behavior.

### Stage 2 — demote "incomplete" from verdict to evidence

Record **typed** incompleteness (`unobserved_path:<path>` vs `missing_token_ids` vs
`capture_write_failed`) instead of one bit, and let the build decide: if the delivered
chain is terminal-attributed (PRs #2675/#2676) **and every link on it is
content-verified** (§4), then any hole is provably off-chain — deliver the sample and
keep the marker in metrics. Mask only when a hole could sit on the delivered path.

Why stage 2 cannot ship on today's `main`: the builder infers links by token-prefix
alone, and an uncaptured generation's output would ride undetected as mask-0
interstitial. Content-level link verification is what detects the dangerous case — an
uncaptured assistant turn inserted into the conversation breaks the model-authored
spine between parent and child. That is the same evidence-over-events principle as the
rest of the capture design: decide from token/content evidence at build time, not from
events at capture time.

## 4. The #2180 pre-split this depends on: continuation stamping

#2180 bundles two separable things:

1. **Record-time stamping** — computing content-evidence fields on each `TokenEntry`
   from the request the capture middleware already sees:
   `continuation_fingerprint` (= `assistant_fingerprint(request items + own output)` —
   what a continuation of this call hashes to), context digest/length, and — new, for
   build-time use — the entry's own **`request_fingerprint`**
   (= `assistant_fingerprint(request items)`).
2. **The resolver machinery** — the `LineageStore` protocol, request-time parent
   resolution, process-shared stores, the prefix-supply handoff, the conformance kit.

Only (1) is needed here, and it is a small, self-contained slice on top of the merged
`fingerprint.py` (#2676): plumb request items into `capture_tokens` (the signature
change #2180 makes anyway) and stamp three fields. **Link verification then needs no
store at all**: a trie-inferred parent→child link is *content-verified* iff
`child.request_fingerprint == parent.continuation_fingerprint` — the model-authored
spines match, so no uncaptured generation slipped into the interstitial (tool and user
content never contribute to either hash, so legitimate environment content passes).

What the slice buys, in order of immediacy:

- **Activates the cumulative content witness in the already-open #2676** — `terminal.py`
  reads `continuation_fingerprint` via `getattr` today; stamping closes the
  repeated-identical-output attribution gap with zero attribution-code changes.
- **Enables stage 2 here** — "every delivered link content-verified" becomes checkable.
- **De-risks #2180** — its schema fields land and bake first; the resolver later
  upgrades build-time verification to request-time resolution over the same fields, and
  its review shrinks again.

Schema coordination (same rule as the attribution stack: no deployed consumers, whoever
lands second renumbers, never two meanings on one integer): #2675 owns v2
(`response_id`); this slice takes **v3** (continuation stamping + `request_fingerprint`);
#2180's remaining delta (parent links, resolution status) renumbers to **v4+**; #2181
follows.

## 5. Sequencing

1. Stage 1 safe-list — independent of everything; can go first or in parallel with
   #2675/#2676. Include the Claude Code endpoint-probe check in the same live smoke test
   that validates #2675's Anthropic id-reuse.
2. Continuation-stamping slice (v3) — after #2676 (uses `fingerprint.py`).
3. Stage 2 typed incompleteness + hole-tolerant delivery — after the stamping slice.
4. #2180 (lineage store, request-time resolution, renumbered v4+) rebases on top.

## 6. Open questions

- Does the Claude Code CLI actually hit `count_tokens`/`models` against a custom base
  URL, and per what config? (Smoke test; determines urgency of stage 1.)
- Safe-list contents and shape: hardcoded path list vs config; whether unknown paths
  should record the path string in the typed marker (yes, proposed above).
- Is persisting `request_fingerprint` worth the bytes vs recomputing at build time?
  (It cannot be recomputed — request items are not stored on the entry; that is exactly
  why it must be stamped. Stated here so the reviewer does not re-derive it.)
- Interaction with §2.9/§6 of the terminal-attribution design doc: that doc now points
  here; keep the two in sync on the stage-2 gating condition.
