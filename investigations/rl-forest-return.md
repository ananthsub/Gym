# Returning the full forest to NeMo RL

Status: investigation, September 2, 2026. Base: `upstream/main` @ 9287fb779 (lineage,
prefix supply, and terminal attribution merged; fingerprint helpers deduplicated in
#2996). Scope: how Gym should return *every* contiguous sequence a rollout produced —
not one flattened chain — and how that works for NeMo RL v1, NeMo RL v2, and eval.

## 1. The problem in one paragraph

An agent harness produces many artifacts per problem: the main conversation, each
sub-agent's conversation, each continuation after a context compaction, side calls such
as title generation, and retries the harness threw away. Today Gym delivers one
sequence per rollout and NeMo RL recipes append the conversation into a single chat
template. Everything that is not the main conversation is dropped from the training
pool — in a long run that is most of the generated tokens. Worse, appending turns from
after a compaction onto the history from before it builds a context the model never saw
as one sequence (it saw a summary), and re-rendering through the chat template can
re-tokenize text differently from what was served. The flatten is incorrect at
compaction boundaries, not merely lossy.

## 2. Three terms, one object

These are three levels of grouping of the same thing — the model calls a rollout made.

| Term | Meaning | Limit |
|---|---|---|
| Per-request | One sample per model call: the prompt that call saw plus what it generated. No grouping. | Never drops anything, but call 10's prompt repeats calls 1–9's output as prompt — quadratic re-processing. |
| Contiguous sequence | Calls that extend one another token for token, merged into one sequence — prompt, generation, tool output, generation… — with a loss mask over the generated spans. | Valid only when every call really extended the previous one. A compaction breaks it (the prompt was rewritten); a sub-agent never joins it (a different conversation). |
| Multi-artifact (the forest) | All the contiguous sequences one rollout produced, each token-exact and labeled. | N calls → K sequences; deliver all K, not one. |

("At request time" in the lineage code means "when the request arrives at the model
server"; it is unrelated to per-request samples.)

## 3. What exists today

**Builder.** `prefix_merging` (`nemo_gym/token_id_capture/builder.py`) already
materializes every chain: nodes link to parents by recorded lineage (or by
token-prefix inference as a fallback), retry siblings are resolved, and root-to-leaf
chains are produced for the whole rollout. Chain *selection* then keeps one — the chain
ending at the attributed terminal call, or the earliest root when no terminal is
attributed — and labels the rest `branch-N`. `BuildOutput.chains` is the forest; only
delivery discards it. `delivered_fraction` in the build notes measures what is dropped.

**Delivery (v1 shape).** `finalize_rollout_token_capture` replaces `response.output`
with the selected chain's projection and sets `mask_sample` plus metrics under
`_ng_token_capture`. One sequence per rollout.

**v2 shape (Gym #2872 + RL #3837, in review).** The model server keeps a token-free
ledger per call (parent, lengths, digests, served response id) and serves it as a
manifest; the vLLM worker stages each call's token delta into TransferQueue; RL's
finalizer fetches the referenced rows and rebuilds one chain per rollout, verifying
digests. Also one sequence per rollout.

**Eval.** `gym eval run` and reward profiling write rollout JSONL rows through the same
delivery path as v1, so offline consumers see one sequence per rollout too.

## 4. Design: the forest plan

Forest assembly is two jobs, and separating them is what makes both RL paths workable.

**Structure — the plan.** Deciding what the segments are: which calls chain, which are
branches, which pair is a retry of the same prompt, which call is the final one, which
segment is structurally uncertain. This needs only per-call metadata — parent pointer,
lengths, digests, served response id, content fingerprints — and never a token. Gym's
ledger holds that metadata in both paths. **Gym computes the plan.**

**Materialization — the tokens.** Turning a segment into an actual token sequence with
its loss mask, and checking that the tokens match the plan's digests. Gym has the tokens
in v1 (its file store); only RL has them in v2 (TransferQueue). **Whoever holds the
tokens materializes, verifying against the plan.**

The plan is the contract. Per segment:

- `segment_id`, ordered `call_ids`, `root_call_id`, `terminal_call_id`
- `label`: `main` | `compaction_ancestor` | `subagent` | `machinery` | `abandoned_retry` | `quarantined`
- links: `continues_segment` (compaction), `spawned_by_call` (sub-agent), hops from the terminal
- mask boundaries as token offsets; `expected_len`; expected digest per link and cumulative
- `generated_tokens`; flags (`retry_group_id`, `retry_kept`, quarantine reason)
- per-segment mask verdict and reason strings

Rules: nothing is dropped — quarantined subtrees become their own flagged root segments
(their internal contiguity still holds; only the parent link is uncertain). Terminal
attribution labels the rewarded segment and resolves retry groups when a witness exists;
when it cannot, it emits flags, not a rollout-level mask. RL never makes structural
decisions.

## 5. Per path

### 5.1 NeMo RL v1 (no SingleController, no TransferQueue)

Gym does both jobs in-process, inside delivery. The result keeps `response.output` as the
main chain (unchanged, for compatibility) and adds `_ng_token_capture.segments`: the plan
plus each segment's tokens in **link form** — the root prompt once, then per link the
interstitial tokens, the generated tokens, and their log probabilities. Link form is
linear in tokens; the current per-item cumulative prompts are quadratic per segment.

The v1 consumer expands link form into one row per segment, keyed `{rollout}_s{k}`, with
the rollout id as the grouping key so a rollout's segments share its advantage and are
normalized together. Open item: the v1 GRPO path's handling of `mask_sample` (and
per-segment masks) is unverified; RL #3766 gives only the SingleController path sample
masking. Verify in the v1 consumer before shipping segments.

### 5.2 NeMo RL v2 (SingleController + TransferQueue)

Gym builds the plan from its ledger and serves it as part of the manifest (extend
`RolloutManifest` with `segments`). RL's finalizer runs its existing rebuild once per
segment: fetch the referenced rows from TransferQueue, concatenate, lay out the mask,
verify against the plan's digests, publish rows `{rollout}_s{k}` with labels as tags.
Sample masking (#3766) applies per row. Grouping ids ride the batch as a column (the
#3407 grouping fix becomes a prerequisite, not a bug fix).

The line to hold in #3837's review: RL fetches and verifies; it does not decide which call
is final or which sibling was kept. Today it orders its own witnesses and falls back to a
parent-link heuristic — those decisions belong in the manifest Gym serves.

### 5.3 Eval (no trainer)

Eval is the v1 path without a consumer, so it gets the forest for free once v1 delivery
carries segments: rollout JSONL rows carry `_ng_token_capture.segments`, and offline
consumers (SFT/DPO data pipelines, analysis) see every artifact instead of one.

Two things eval adds beyond training:

- **Profiling the harness before training.** Segments per rollout, `delivered_fraction`,
  label counts, retry groups, quarantine reasons, per-segment mask reasons — the numbers
  that say whether a harness's artifacts are usable, measured on a reward-profiling run
  rather than discovered in a training run.
- **A declared forest without tokens.** Runs without token capture can still produce the
  plan's *structure* from the observation bundle (`ng_agent_observations`: the invocation
  tree with `parent_invocation_id`/`spawned_by_tool_call_id`, per-call served response
  ids, compaction events). That is the harness-declared view; with capture on, the
  token-derived view corroborates it. The two must agree — a disagreement is a harness
  wrapper defect, and eval is the cheapest place to find it.

Consistency requirement: eval and training must produce the same segment structure for
the same rollout, so what you profile is what you would train on. That is automatic
because eval *is* the v1 delivery path.

## 6. Credit assignment over segments

The baseline is the rule multi-turn GRPO already uses: a segment is a turn that ran in a
different context window, so every segment inherits its rollout's advantage, normalized
at the rollout level. Excluded always: `abandoned_retry` (did not influence the outcome)
and `quarantined` (structure we cannot verify). Compaction ancestors take the rollout's
advantage (same agent, same task). Sub-agents take it too by default; a local subtask
reward or a judged return replaces it when available — both need the spawning link the
plan carries. Machinery calls are a per-label knob (include compaction summaries,
exclude titles by default). The credit policy is RL's; Gym only guarantees the labels
and links are correct. The first experiment once segments ship: main-only vs. main +
compaction ancestors vs. everything, on a long-horizon harness.

## 7. Open questions

1. Result size in v1 with many segments — link form is linear, but a rollout with dozens
   of sub-agents may still need a cap or a spill-to-file option.
2. Capture-incomplete masking is rollout-level today. With recorded lineage, a missing
   call's output entering a later prompt would fail that request's parent resolution, so
   incompleteness can become segment-scoped. Decide when segments ship.
3. Retry groups whose kept sibling is unknown: exclude the group (small loss) or include
   both (small noise)? Default proposed: exclude, configurable.
4. Does the QoRoboros wrapper populate `ng_agent_observations`? If not, its declared
   forest is unavailable and eval corroboration falls back to tokens only.
5. Machinery-call defaults (compaction summaries in, titles out) are a guess until the
   ablation runs.

## 8. Sequencing

1. **Gym: `project_forest` + plan schema + tests.** Pure builder work; no delivery change.
   Parity test: the plan for a rollout must be identical whether built from complete
   records (v1) or ledger rows (v2).
2. **Gym: v1 delivery + eval.** Segments in link form under `_ng_token_capture`; per-segment
   masks; profiling metrics. Eval JSONL gains the forest at the same time.
3. **Gym #2872 follow-up: `segments` in the manifest.** Same plan object, served.
4. **RL #3837 follow-up: per-segment materialization.** Existing rebuild per segment;
   labels as tags; grouping ids; structural decisions removed from the RL side.
5. **RL v1 consumer:** expand link form, honor per-segment masks, grouping ids.
6. **Ablation on a long-horizon harness** to set the credit defaults.

Attribution stays as merged (#2675/#2676): once segments ship, its "unattributed" verdict
means "forest delivered, terminal label absent" rather than "mask the rollout".
