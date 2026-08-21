# Terminal attribution: design and PR split

Branch `terminal-attribution`, based on `upstream/main` @ `fa0d25158`. This document is
self-contained: it states the problem, the design and its evidence, what changed, how it
was validated, and how the branch decomposes into PRs. It is written for a reviewer who
was not part of the design discussions.

---

## 0. The mechanism in plain terms

A blackbox agent — say the Claude Code CLI — never tells us which model call was its
last. It doesn't have to, because it can't hide anything: every model call it makes goes
through Gym's model server. Each time the server answers a call, it writes one record in
the capture log: the exact reply text it sent back, the ID stamped on that reply, and the
tokens behind it.

When the agent finishes, its wrapper reads the agent's own transcript and hands back the
final conversation — the object the grader scores. So at the end we hold two things: the
finished conversation, and our own record of every reply we ever served.

Finding the final call is then a matching problem, solved in one central place — the step
that builds the training row after grading (`delivery.py` → `terminal.py`), with no
per-agent code. Up to three independent checks run: a caller can name the kept call
outright; the conversation's reply ID (when it kept one) must point at exactly one record;
and the conversation's model-written text must be word-for-word identical to exactly one
record's reply (hashed over model-authored content only, so reformatted tool output
doesn't matter). Checks that can run must agree. If none matches, or two records match
equally well, we don't guess: the sample is dropped from training and the reason is
logged. The worst case is losing a sample — never training on the wrong conversation.

Once the final call is identified, the rest is mechanical: each record's prompt contains,
verbatim, the tokens of the calls before it, so we walk backward from the final call to
the first and deliver exactly that chain. Side calls — a title generator, a sub-agent, a
retry the agent threw away — are left out instead of poisoning the sample or masking a
healthy rollout.

The rest of this document is the precise version of the above, with evidence.

---

## 1. Problem

Token-id capture records **every** model call a rollout's serving path handles: the main
agent loop's calls, auxiliary calls (title generators, summarizers), sub-agent branches,
and abandoned retries. Trajectory building must then decide which captured call is the
**terminal** — the call whose root-to-terminal ancestry becomes the training row and
receives the reward.

Today (`main`, #2124–#2126 merged) there is no terminal identification at all. The builder
(`prefix_merging`) reconstructs call chains from token-prefix relationships and applies a
strict single-chain policy: if the rollout produced more than one root or more than one
chain, or a final-call retry cannot be resolved, the sample is **masked**. Chain selection
falls back to "the chain whose root call completed first," which is wrong in known shapes
(the builder's own comments call them out):

- **Auxiliary calls.** A short title-generation call that completes before the first task
  call becomes a second root → `roots != 1` → a perfectly healthy rollout masks. Worse,
  legacy selection picks the *auxiliary* call's chain as "main".
- **Sub-agent forks.** Parallel sub-agent calls split the capture into multiple chains →
  masked, or the wrong chain is picked by completion order.
- **Final-call retries.** A harness retry after a timeout leaves two sibling generations
  for the same prompt. No later call identifies the survivor → masked, even though the
  harness knows exactly which response it kept.

The invariant this work enforces: **the trained trajectory must be the token ancestry of
the response the verifier scored.** Anything else can pair a reward with tokens from a
conversation that did not earn it.

### 1.1 Why heuristics at the model server cannot work

Two heuristics have circulated and both are unsound:

- *"A response without a new tool call is terminal."* Fails in both directions: auxiliary
  calls and sub-agent finals are tool-free but not terminal; truncated rollouts end *with*
  a pending tool call; and if the server has no tool parser for a dialect, every response
  looks tool-free.
- *"The last committed call is terminal."* A background sub-agent committing after the
  main chain's final call steals the terminal. (This is precisely the current behavior of
  the RL integration's seal path, which sends only `{"reward"}` — an accepted wart this
  design replaces.)

The root cause is structural: terminality is a property of **what the harness did after
the response** — kept it and stopped, discarded it and retried, handed it to a sub-agent —
and the model server cannot observe any of that. The server sees candidates; only the
returner knows what it kept. Therefore terminal identity must be **declared or derived
from what the run returned**, never inferred from capture-store ordering or response
shape.

---

## 2. Design

### 2.1 The join surface: the `/run` result's `response`

Every agent's `/run` result carries `response` — `BaseVerifyResponse` inherits from
`BaseVerifyRequest`, so `response: NeMoGymResponse` is a **required** field
(`nemo_gym/base_resources_server.py:92-97`); dropping it is structurally impossible. That
object is:

1. **The termination declaration.** The transcript in `response.output` is, by
   construction, "the conversation the loop ended with."
2. **What the reward binds to.** The verifier scored exactly this object, so attributing
   the terminal from it guarantees reward/trajectory consistency.
3. **Available centrally.** `finalize_rollout_token_capture` already receives the finished
   record; attribution needs zero per-agent code.

### 2.2 The witnesses join (`nemo_gym/token_id_capture/terminal.py`)

Attribution is **independent witnesses with corroboration**, not a trust hierarchy. The
fact "which call was kept" is emitted in different places by different agent classes, so
up to three witnesses testify:

| Witness | Source | What it proves |
|---|---|---|
| `explicit` | `_ng_terminal_model_call_id` on the result (a gate seal or agent declaration) | The caller names the kept call directly |
| `response_id` | `response["id"]` equals the served id recorded on exactly one entry | Possession of the id proves which served response the client received — the one case content cannot resolve (identical-content retries) |
| `content` | `assistant_fingerprint` of the response's model-authored items matches one entry | The transcript's content identifies the producing call |

Rules, in order of importance:

- **Abstention, not guessing.** Ambiguity inside a witness (a duplicated served id, several
  content matches with different token sequences) is an abstention with a recorded reason,
  never a vote.
- **Corroboration, not ranking.** All present witnesses must name the same call — or calls
  whose full token sequences are identical (interchangeable for training; they collapse to
  the smallest call id deterministically). A contradiction attributes **nothing** and
  persists `witness_disagreement[explicit=X;response_id=Y]`, because a contradiction is
  evidence of a real defect (stale seal mapping, backend id reuse, transcript-synthesis
  bug) that outranking would silently bury.
- **The trail is kept on success too.** A witness that abstained while another attributed
  is a diagnosable defect; the reasons string records it.
- **Never raises.** Malformed content is an abstention; attribution failure falls back to
  the strict single-chain policy.

The content witness pools **three readings** of one response before deciding, because
different agent shapes hand back differently-shaped transcripts:

1. **Full transcript** — the fingerprint over *all* the response's model-authored items
   matches an entry's cumulative `continuation_fingerprint` (stamped by lineage-aware
   writers — see §2.6; inert on this base until #2180 lands).
2. **Final response only** — the same fingerprint matches one entry's own `output_items`
   (an agent that returned the last response verbatim, or a single-turn task, where the
   whole transcript *is* the final turn).
3. **Trailing block** — the fingerprint of the transcript's final run of consecutive
   model-authored items matches one entry's own `output_items`. This is what carries
   **multi-turn synthesized transcripts** (the blackbox and `simple_agent` shapes) on
   this base: the merged transcript's full fingerprint matches nothing without lineage,
   but its trailing assistant block is exactly the terminal call's own output. A
   transcript that ends in a non-model item (a pending tool result — the truncated-rollout
   shape) has no trailing block and skips this reading rather than matching a mid-chain
   call.

The readings can name different calls — a first call's cumulative fingerprint *is* its
own-output fingerprint — so candidates from all keys pool into **one** ambiguity
decision: one surviving candidate (or token-identical candidates) attributes; several
different candidates abstain with `content_ambiguous`; none abstains with
`no_content_match`.

#### 2.2.0 Cross-dialect matching: what is actually compared

Capture stores the **normalized view**: the `/v1/messages` route captures at the inner
Responses layer (the client received Anthropic blocks it never sees), the chat route
stores the assistant message lightly wrapped as a `message` item, the responses route
stores items as-is. The wrapper's transcript arrives in whatever dialect the harness log
used. The comparison is well-defined anyway because it happens in **fingerprint space**,
which is dialect-blind by construction: plain-string content, `output_text` parts, and
Anthropic `text` blocks all hash as the same text part; Responses `function_call` items,
Anthropic `tool_use` blocks, and chat `tool_calls` arrays all hash as the same
`(call_id, name, canonical_args)` tuple; and Gym's converters preserve text and tool-call
ids verbatim across shapes. The invariant: *fingerprint(served payload in the client's
dialect) == fingerprint(captured normalized items)* whenever content crossed the
converters unmodified. The served envelope id is the same invariant in scalar form — one
string, identical on both sides of the conversion boundary (that is what the Messages
id-reuse buys) — but it cannot replace content matching, because chat and Anthropic
transcripts carry no in-band envelope id.

Known crack in the invariant: the **reasoning parser**. When `<think>` content is split
into a separate reasoning item, the captured assistant text is the stripped text; a
deployment serving tags inline (`preserve_reasoning_in_assistant_content`) hands the
harness tagged text → hashes diverge → mask. Fail-closed, but a yield hole on reasoning
models. The pin this needs (not yet written): a per-dialect conformance test asserting
fingerprint equality between the served wire payload's assistant items and the captured
entry's `output_items`, run per reasoning configuration.

#### 2.2.1 What happens when checks disagree — the complete outcome table

| Situation | Outcome |
|---|---|
| All present witnesses name the same call | Attribute; the method label is the first witness, the rest recorded as `corroborated_by=` |
| Witnesses name different calls with **identical token sequences** | Attribute (interchangeable for training; smallest call id wins deterministically) |
| Witnesses name different calls with **different sequences** | **Attribute nothing.** `witness_disagreement[explicit=X;response_id=Y]` persists; the sample masks. A disagreement is evidence of a real defect — a stale declared id, backend id reuse, a wrapper synthesis bug — and picking a winner would silently bury it. The reason string aggregates, so a *systematic* defect appears as a spike in one reason, not as quiet training corruption. |
| One witness abstains (ambiguous id, no content match), another attributes | The attributing witness wins — abstention is not a vote against — and the abstention stays in the trail as a diagnosable note |
| All witnesses abstain | Unattributed → the strict single-chain legacy policy (deliverable only if the rollout is a single clean chain) |

The content witness deliberately uses `assistant_fingerprint` **alone**, without a
request-context digest: attribution selects a chain whose tokens the builder verifies
independently, and never reuses tokens across the matched boundary, so context
verification would only spuriously refuse synthesized transcripts that reformat tool
output.

### 2.3 The served envelope id (`response_id` on `TokenEntry`)

**Definition.** The top-level `id` of the outermost payload at the dialect boundary the
client called. One model call = one served envelope = one id:

| Dialect | Envelope id | Minted by |
|---|---|---|
| `/v1/responses` | `resp_<uuid4>` | Gym's `ResponsesConverter` (`responses_converter.py:631`) while converting the backend chat completion |
| `/v1/chat/completions` | `chatcmpl-…` | the backend (vLLM), passed through |
| `/v1/messages` | previously a fresh `msg_<uuid4>` (`anthropic_converter.py`, response mapping) — **now reuses the inner Responses id** (this branch) |

It is explicitly *not* the output-item ids (`msg_…`/`rs_…`), tool-call ids, the
capture-internal `model_call_id`, or a client-minted logical request id — different
namespaces.

**The observed-never-minted rule.** Capture records `payload["id"]` off the response
object it is handed (`sink.py`). Who minted the id is irrelevant; the witness needs only
two properties: unique per serving (uuid4 or backend request id — both hold; identical
retries get distinct ids) and observed identically by recorder and holder (same object).

**The `/v1/messages` fix.** Capture runs inside `_invoke_responses`
(`base_responses_api_model.py`) — deliberately, because it needs the complete typed
response before SSE wrapping, and the Anthropic mapping drops token fields. For the
Messages dialect the client then received an *Anthropic* envelope whose id was minted
*after* capture — so the recorded id and the kept id never matched. Rather than moving
capture, the Anthropic response mapping now **reuses the inner Responses envelope id** on
its outer envelope. Both parties then hold the same string in every dialect. (Anthropic
clients treat the id as opaque; nothing requires the `msg_` prefix.)

**Which ids survive into the transcript** (why `response_id` and `content` are *both*
needed): in the Responses dialect, output items ride into the next request's `input`
verbatim, item ids included, and `simple_agent`'s merged result rides the final call's
envelope (its `id` comes from the last model call) — so the envelope id survives for
native agents. In the chat and Anthropic dialects, an assistant message appended to
history carries **no** message-level id, and the terminal call is frequently tool-free —
so a blackbox chat-dialect harness may hold no in-band id at all, and the content witness
carries it. Neither witness is sufficient alone; that is the corroboration structure's
justification, not an accident.

### 2.4 Terminal-anchored building (`builder.py`, `consumer.py`)

`prefix_merging` gains an optional `terminal_call_id`. With one:

- **Chain selection is anchored.** The delivered chain is the root-to-terminal path —
  never the earliest-completion heuristic.
- **Truncation.** Calls that extend the terminal were served after the kept response; they
  are outside the verified trajectory and truncate away.
- **Retry resolution.** In a retry group, a sibling on the terminal path is the kept one.
  Groups off the terminal path cannot reach the delivered chain, so they no longer mark
  the rollout unresolved.
- **Status is explicit.** `BuildNotes.terminal_chain` reports `"delivered"` (clean
  root-to-terminal chain), `"broken"` (the path crosses a quarantined boundary),
  `"not_captured"` (the named call is not among buildable entries), or `""` (no terminal
  given). The verdict survives failed builds in metrics (`terminal_attribution` with
  `chain: "error"` fallback) so a broken chain stays diagnosable in aggregate.

Mask semantics in the consumer:

| Situation | `mask_sample` |
|---|---|
| `terminal_chain == "delivered"` | **False** — off-path calls (aux, forks, abandoned retries) are *excluded from delivery* instead of masking |
| Attributed, but `broken`/`not_captured`/unbuildable | **True** — the verified trajectory is known and undeliverable |
| Unattributed | **Legacy policy, bit-for-bit**: `unresolved_retries or roots != 1 or chains != 1` (plus the existing no-trainable-tokens projection failure) |

Attribution only ever *adds* yield; no previously-trainable rollout changes behavior
(regression-checked: the 72 pre-existing capture tests pass unmodified).

### 2.5 The echo contract, and its measured cost

The content and id witnesses read the transcript out of the verify **echo** (agents
typically return the verifier's `BaseVerifyResponse` from `/run`). A verifier that
rewrites `response` costs attribution. How big is that assumption? Measured across the
fleet on `main` (2026-08-21):

**Resources servers (114):** 106 echo `response` unchanged (the documented
`XVerifyResponse(**body.model_dump(), reward=…)` splat idiom); 0 replace; 0 drop
(structurally impossible — the field is required); 7 use the gymnasium `/step` protocol
with no verify; exactly **1** mutates: `terminus_judge`, which round-trips the response
through a unicode-surrogate scrub. (Full per-server audit with evidence: Appendix A.) **Deliberately not "fixed" here:** the scrub is
identity on clean strings, and lone surrogates cannot be written by the capture store
anyway (`orjson` rejects them → the call is marked capture-incomplete → the rollout is
already masked). The scrub therefore costs zero real attribution yield; it is the
documented exception. Two existing framework features already silently depend on the echo
(re-verification rebuilds the verify request from the persisted row's `response`;
MCP auto-exposure's name-restore assumes items survive), so a conformance test hardens
things that were already assumed.

**Agents (39):** 25 call `/verify` and return its response untouched (the `response_id`
witness works natively — `simple_agent`'s merged transcript keeps the final call's
envelope id). 2 mutate/replace after verify (`image_tools_agent` replaces the echo with a
different object than the one scored — a pre-existing reward/transcript mismatch worth its
own issue; `labbench2_vlm_agent` redacts image blocks). 2 replace only on failure paths.
**12 never call `/verify`** and synthesize the response from a harness log — including
every blackbox wrapper (`claude_code`, `openclaw`, `opencode`, `codex`, `cline`,
`kilocode`, `pi`, `prime`), which today mint a fresh `resp_<uuid4>` envelope id. For
these, "keep the response as-is" is a **fidelity contract** instead: assistant content
byte-faithful to what was served. That single obligation makes the content witness carry
blackbox agents; per-agent id extraction is deliberately *not* required (§2.5.1).

#### 2.5.1 Blackbox synthesizers: content is the mechanism, ids are opportunistic

Per-agent served-id extraction from harness logs would be tedious and error-prone, and
the witness analysis shows it is not needed as a program:

- **Divergent final retries** — the case the id witness exists for on native agents — are
  self-resolving for synthesizers: the kept response's *content* is what the harness log
  contains, and divergent generations have divergent content, so the content witness names
  the kept sibling uniquely.
- **Identical retries** collapse without any id (interchangeable token sequences).
- The residue that only an id can resolve is thin: identical final text with *different*
  token sequences (vanishingly rare under sampling; collapse-identical covers the common
  identical-tokens case), repeated identical assistant output at different conversation
  depths (resolved by the cumulative fingerprint reading once lineage-aware writers stamp
  `continuation_fingerprint`), and deliberately redacted transcripts (self-inflicted;
  already fail closed).

What *is* centralizable is the inverse rule: **never mint an envelope id in a synthesized
run result.** A fabricated `resp_<uuid4>` is strictly worse than no id — it can only
produce `response_id_no_match` noise (and, in principle, a collision), whereas an empty
`id` is an honest abstention the join handles by design (`response_has_no_id`). One shared
synthesis helper enforces this mechanically: default `id=""`, with an optional
`response_id=` parameter for harnesses whose logs already carry the served id (Claude
Code's JSONL message records, the `swe_agents` completion files) — a one-line opt-in where
it is free, never a parsing project.

**And for the instrumented agents, the parsing already exists** — in the observability
plane, not the run result. `ng_agent_observations` (`nemo_gym/rollout_observability.py`)
is a typed `AgentObservationBundle` that five blackbox wrappers (claude_code, opencode,
openclaw, pi, pinchbench) already populate: `AgentInvocation` records with
`parent_invocation_id`/`spawned_by_tool_call_id` (the harness-declared invocation tree),
per-invocation `ModelCallRef`s joined by `model_call_id` **or `(model_ref, response_id)`**
(the Claude Code producer reads each served message's `id` out of the harness JSONL and
records a `model_response_id_missing` gap when absent), and `ContextCompactionObservation`
events. The eval-side `ModelCallRecord` has carried `response_id` all along —
`TokenEntry.response_id` (PR 1) brings the *training* plane to parity, making bundle refs
directly resolvable against captured token entries. `join_model_call_observations`
already resolves bundle refs against captured calls "without guessing ownership," with
typed gaps on failure — the same fail-closed philosophy as the witnesses join.

This yields a **fourth witness at near-zero cost**: the root invocation's last
`ModelCallRef` is the harness-declared terminal, consumed by `resolve_terminal` as one
more declaration to corroborate — never to trust. Caveats: the bundle exists only for
instrumented agents, is explicitly best-effort (gaps), and requires observability capture
— an additional witness, not a replacement for the content and id witnesses. Listed as a
follow-up in §5.

Failure containment either way: a violated contract produces `no_content_match` /
`witness_disagreement` → the rollout **masks**. Never misattribution; the cost is yield,
visible in the existing mask-reason aggregation and kill switch.

### 2.6 Compatibility with the in-flight stack (#2180/#2181, gate/ledger successor)

- **Fingerprints.** `fingerprint.py` is extracted byte-compatibly from #2180's
  `lineage.py` (same hash domains `nemo-gym-lineage` / `nemo-gym-lineage-context`, same
  tagged length-delimited layout, same dialect normalization). Fingerprints computed here
  equal #2180's, so nothing re-hashes at rebase time. The functions are **pure and
  stateless** — importable anywhere; lineage *state* stays confined to the capture locus
  and the finalizer. When #2180 lands, its `lineage.py` should import from this module
  (natural first slice of the #2180 split).
- **Cumulative content reading.** Reads `continuation_fingerprint` /
  `fingerprint_version` via `getattr` — inert on this base, activates automatically when
  #2180's writer stamps them (covered by a test that simulates the stamp).
- **Sequence identity.** `_collapse_identical` uses `(digest, cum_len)` when a
  lineage-aware writer recorded them, else direct token-array comparison — no change
  needed at rebase.
- **Gate/RL seal.** In the TQ deployment the explicit witness's transport is the seal:
  today RL sends only `{"reward"}` and the receipt's terminal defaults to the last
  committed call (the unsafe heuristic, §1.1). The seal should carry the declared terminal
  (the harness-known logical request id / served id); `resolve_terminal` consumes it as
  `explicit` with the other witnesses corroborating token-free (fingerprint columns exist
  on ledger rows in the gate-successor design). RL-side change, lands with that rebase.

### 2.7 Gaps: is text enough to go on?

Content matching alone is sufficient for the common cases, but it has enumerable holes.
Every one **fails closed** — the design converts wrong-training-data risk into measurable
yield loss — and the other witnesses exist precisely to shrink them. The honest list:

| Gap | What happens | What closes it |
|---|---|---|
| The model produced **identical output at two different depths** ("Done." twice) | Trailing-block reading matches both entries; their token sequences differ → `content_ambiguous` → mask | The served id (native agents, id-carrying harness logs); the cumulative reading once #2180's `continuation_fingerprint` exists (different depths → different transcript prefixes → different fingerprints) |
| **Identical final text via different token sequences** (a retry that resampled the same words differently) | Sequences differ → abstain → mask | Nothing but the id — and it is vanishingly rare; the common identical-tokens retry collapses fine |
| **Wrapper infidelity**: synthesized text is normalized, truncated, or rewritten — including regenerated **tool-call ids** (the fingerprint hashes `call_id`, name, and canonical arguments, so renumbered ids break the match) | `no_content_match` → mask | The fidelity contract (§2.5.1): byte-faithful model-authored content *and* tool-call ids. Violations surface as a per-agent spike in one mask reason, which is the enforcement mechanism |
| **Reasoning-only final output** (no message, no tool call) | Fingerprints deliberately exclude reasoning items → no model-authored content → abstain | Nothing; deliberate. Harness finals essentially always carry a message or tool call |
| **Transcript ends with a pending tool result** (truncated rollout) | No trailing block → that reading skips (by design — it must not match a mid-chain call) | The id witness, the cumulative reading, or the legacy single-chain policy |
| **The terminal call's record is missing** (a crashed capture write) and another call has identical text | The text match would name the wrong call — but an incomplete capture **independently masks the rollout** (`begin_call` intent custody marks the hole) | Invariant worth stating: content attribution is only trusted alongside capture completeness, which is separately enforced. Attribution never overrides the incomplete flag |
| **Sub-agent calls routed to a different model server** | Outside the capture set entirely; the main chain's attribution is unaffected, and the missing calls were never part of the trained chain | Nothing needed for attribution; it is a coverage question for capture, not a join question |
| **No witness at all** (chat-dialect blackbox with rewritten text and no logged ids) | Unattributed → legacy strict policy → masked unless the rollout happens to be a single clean chain | This is the designed floor. Its size is exactly what the masked-fraction baseline measures |

### 2.8 Toward forest delivery — the end state this must not foreclose

The eventual goal is to train on **every** captured chain — the full forest: the main
chain, sub-agent branches, auxiliary calls — not just the single rewarded chain.
Implications, so no upcoming review simplifies it away:

- **The forest is already computed.** `prefix_merging` materializes every chain
  (`main` + `branch-N`); only the *delivery* contract drops all but one
  (single-response delivery replaces one `response.output` — the same constraint that
  rejects `per_request`). `BuildOutput.chains` is the forest; keep it.
- **Attribution's role shifts from selection to labeling, and remains a prerequisite.**
  With a forest, everything is delivered — but the task reward still attaches to exactly
  one chain (the one ending at the terminal; without the join, every tree would claim
  the score), and branches must be classified: the rewarded chain; legitimate sibling
  trees (trainable with their own reward semantics); and **abandoned retries**, which
  the harness rejected and which must never train as positives. The terminal-anchored
  keep-set/quarantine logic is exactly that classifier. Post-terminal calls stop being
  truncated away and become their own branch — truncation generalizes to a split point.
- **For instrumented blackbox agents, the forest is harness-declared, not inferred.**
  `ng_agent_observations` (§2.5.1) carries the invocation tree — `AgentInvocation` with
  `parent_invocation_id` and `spawned_by_tool_call_id`, per-invocation model-call refs
  joinable by `(model_ref, response_id)`, and compaction events that explain new roots.
  Branch labels (main vs sub-agent vs compaction restart) come from the bundle and are
  *verified* against the token-prefix forest, rather than reconstructed from prefixes
  alone.
- **Prefix matching is the within-tree mechanism; the forest also needs a typed
  between-tree graph, and compaction edges are its first edge type.** Compaction
  deliberately breaks token continuity, so no prefix relation can link a pre-compaction
  tree to its post-compaction continuation — yet for reward attribution they are one
  logical attempt, and under single-chain delivery the dropped pre-compaction trees are
  the *longest* segments of exactly the long agentic rollouts (the same bias argument
  that motivates partial-rollout checkpointing). The linkage pass is a finalize-time
  join over already-durable evidence — nothing new is captured — with three independent
  sources corroborating, in the witnesses posture: (1) **declared**:
  `ContextCompactionObservation` carries `before_model_call` / `after_model_call` /
  summary `model_calls` / `first_kept_item_id`, joinable to TokenEntries by
  `response_id`; (2) **the captured summary call**: the compaction summary is generated
  through our own server, so it is already in the capture log, and its output text
  reappears embedded in the new root's prompt — a content bridge needing no harness
  cooperation; (3) **kept item ids**: Responses-dialect items carry their ids verbatim
  into the compacted prompt, so item-id intersection between the new root's request and
  the old tree's served outputs is a token-plane link. Corroborated edges attribute;
  unlinked trees stay unlinked — conservative credit, never wrong credit. Delivery then
  labels trees (terminal; compaction ancestor of the terminal — natural rule: reward
  inheritance; summary/machinery calls; sub-agent; abandoned retry) and the credit
  policy over those labels remains a trainer decision the mechanism exposes, not makes.
- **What forest delivery needs that this branch deliberately does not build:** a
  multi-sequence delivery contract (natural in the TQ deployment, where rows are
  per-call and the finalizer can publish multiple training rows per rollout);
  per-sequence masking (`mask_sample` is per-rollout today; a broken branch should mask
  only itself); trainer-side sample identity and grouping (multiple rows per rollout
  break prompt-based GRPO grouping — the RL #3407 grouping-ids-as-batch-column work is
  exactly the machinery forest rows need); and reward semantics for unrewarded trees
  (zero-advantage, sub-agent-local rewards, or a reward model — a trainer decision the
  mechanism should expose, not make).

### 2.9 Auxiliary calls the model server does not understand

A blackbox harness gets one base URL — the capture-prefixed one — so *every* endpoint it
touches lands under the prefix. The capture middleware observes three paths; any other
prefixed request (a count-tokens probe, `/v1/models`, embeddings) currently triggers
`mark_incomplete(rollout_id, "")` before forwarding: one stray utility call masks the
whole rollout. If the Claude Code CLI calls `/v1/messages/count_tokens`
(version-dependent), this silently costs every rollout — a live-harness smoke-test item.

The blanket rule guards a real hazard — an *uncaptured generation* whose output re-enters
the conversation would train policy tokens as environment tokens — but the hazard only
exists for calls whose output can enter the chain. The refinement, in two stages matching
the evidence available:

1. **Now (independent PR): a non-generative endpoint safe-list** at the middleware.
   Prefixed requests to known non-generative paths forward without capture and without
   the incomplete marker (a counter at most). Classification by endpoint semantics,
   knowable statically.
2. **Post-#2180: demote "incomplete" from verdict to evidence.** Typed incompleteness
   reasons (`unobserved_path:<path>` vs `missing_token_ids` vs `capture_write_failed`),
   and the build delivers despite off-chain holes when the delivered chain is
   terminal-attributed **and every link is verified RESOLVED** — verified lineage
   *detects* the dangerous case automatically (an uncaptured assistant turn inserted into
   the conversation breaks the model-authored spine → UNRESOLVED → masks for the right
   reason). On trie-only main an uncaptured generation would ride undetected as mask-0
   interstitial, which is why the blanket rule cannot be fully relaxed before #2180.

Same principle as §2.8 and the compaction discussion: decide from token evidence at
build time, not from events at capture time. And the same courtesy this branch already
extended to aux calls the server *does* understand (excluded, not masking) applies to
ones it doesn't: exclusion with evidence, never silent acceptance.

---

## 3. What changed (file map)

| File | Change |
|---|---|
| `nemo_gym/token_id_capture/records.py` | `TokenEntry.response_id: str \| None`; schema 1 → 2 with changelog line |
| `nemo_gym/token_id_capture/sink.py` | `capture_tokens` observes `payload["id"]` → `entry.response_id` |
| `nemo_gym/anthropic_converter.py` | Response mapping reuses the Responses envelope id (mints only when the response has none) |
| `nemo_gym/token_id_capture/fingerprint.py` | **New.** Pure fingerprint functions (§2.6) |
| `nemo_gym/token_id_capture/terminal.py` | **New.** `resolve_terminal` + `TerminalAttribution` (§2.2) |
| `nemo_gym/token_id_capture/builder.py` | `terminal_call_id` anchoring: ancestry set, retry resolution, root-to-terminal chain + truncation, `terminal_chain` status (§2.4) |
| `nemo_gym/token_id_capture/consumer.py` | Attribution before build; mask semantics; `terminal_attribution` metric survives failed builds |
| `nemo_gym/token_id_capture/delivery.py` | `TERMINAL_CALL_KEY = "_ng_terminal_model_call_id"`; passes the result's `response` + explicit key into the build |
| `tests/unit_tests/test_terminal_attribution.py` | **New.** 29 tests (§4) |
| `tests/unit_tests/test_trajectory_builder.py` | One monkeypatch wrapper accepts the new `run_builder` kwarg |

---

## 4. Validation

Tests ride with the commit (and therefore the PR) whose behavior they pin — three files,
39 tests:

- **`test_served_response_id.py`** (6, with commit 1): all three dialect routes record
  the id the client received (the Messages test proves the converter reuse end-to-end);
  converter-unit check; v1 records accepted (`response_id=None`); newer records rejected.
- **`test_conversation_fingerprint.py`** (7, with commit 2): the same content in Chat,
  Responses, and Anthropic shapes hashes identically — text and tool calls; argument
  reserialization tolerated, argument changes not; reasoning items excluded; no
  model-authored turn → empty fingerprint; `conversation_digest` sees tool results the
  fingerprint ignores; non-object items raise for the caller to handle.
- **`test_terminal_attribution.py`** (26, with commits 3 and 4):
  - *Witnesses* (14, commit 3): each witness attributes alone; explicit-not-captured
    abstains; merged transcript attributes by id with the trailing-block reading
    corroborating; **a merged transcript with no id attributes by the trailing block
    alone** (the blackbox multi-turn case); a transcript ending in a pending tool result
    skips the trailing reading; **repeated identical output at two depths abstains**
    (the documented gap, §2.7); final-turn response attributes by content; simulated
    `continuation_fingerprint` activates the cumulative reading; identical retries
    collapse; divergent retries resolve with corroboration; disagreement fails closed;
    **a mutated echo matches nothing**; duplicated served id abstains while content
    still attributes; no response object abstains.
  - *Builder anchoring* (5, commit 4): unattributed aux rollout masks and picks the
    wrong root (documents the pre-existing behavior); attributed terminal delivers the
    verified chain and excludes the aux call; post-terminal truncation; final-retry
    resolution via the terminal path (and its unresolved legacy twin); `not_captured`
    reporting.
  - *Mask semantics* (4, commit 4): delivered → unmasked with the verified chain's exact
    token arrays; unattributed → strict policy; named-but-uncaptured falls back
    unattributed; broken terminal chain masks with the `broken` verdict surviving a
    failed projection.
  - *Delivery e2e* (3, commit 4): `finalize_rollout_token_capture` attributes from the
    result's response, honors the explicit key, and keeps the strict policy without
    witnesses.

Suite state: 29/29 new; 72/72 pre-existing capture tests and 34/34 trajectory-builder
tests pass; full `tests/unit_tests/`: 2747 passed, 1 pre-existing unrelated failure
(`test_enroot_provider::test_create_start_timeout_cleans_up`, fails on clean `main`).
Ruff lint + format clean.

**Not yet exercised:** a live vLLM rollout (`gym env start` + a real harness), and the
masked-fraction measurement on a harness that makes auxiliary calls — the standing
baseline, expected to drop to near zero with attribution.

---

## 5. PR split

### Land order: this stack goes in **before** #2180/#2181

This branch has zero dependency on the lineage stack — it is built and validated on
current `main` (the trailing-block content reading, §2.2, is what removed the last
dependency: multi-turn transcripts no longer need the cumulative fingerprint to match).
Landing it first is strictly better:

- **The schema-number decision dissolves.** This ships as v2, exactly as coded. #2180
  renumbers its ladder (parent linkage, resolution) to v3/v4 and #2181 to v5/v6 during
  the split they are already undergoing. Never let two meanings share a version integer
  (the #2278 v3-collision lesson).
- **#2180's diff shrinks.** `fingerprint.py` is merged and reviewed first; #2180's
  `lineage.py` — its largest file — imports the pure functions instead of defining them.
- **Immediate yield.** The aux-call/sub-agent/final-retry masking class is fixed on the
  already-merged #2124–#2126 base, and the masked-fraction baseline becomes measurable
  now.
- **The rebase cost falls on #2180, with a reference in hand.** The real conflict surface
  is `prefix_merging` and `_assemble`, modified substantially by both sides — but the
  `stack-v2-fixes` branch already validated exactly that merged end-state (terminal
  anchoring on top of the 2180/2181 builder), so the rebase is mechanical porting, not a
  design merge. One coordination point for #2180's renumber commit: its hard
  minimum-schema floor (reject-below-v3) must account for v2 records existing.

### PR 1 — record the served envelope id — **open as [#2675](https://github.com/NVIDIA-NeMo/Gym/pull/2675)** (draft)
`records.py`, `sink.py`, `anthropic_converter.py` + the route-level and schema tests.
Pure capture-side; no delivery behavior changes; schema v2, final. **Review focus:** the
observed-never-minted rule; the Messages-dialect id-reuse (confirm no client depends on
the `msg_` prefix); the schema bump.

### PR 2 — the witnesses join + terminal-anchored building — **open as [#2676](https://github.com/NVIDIA-NeMo/Gym/pull/2676)** (draft, stacked on #2675)
`fingerprint.py`, `terminal.py`, `builder.py`, `consumer.py`, `delivery.py` + the rest of
the tests. Depends on PR 1. **Review focus:** corroboration/fail-closed semantics (§2.2,
§2.2.1); the mask table (§2.4) — especially that the unattributed path is bit-identical
to today; the truncation rule; the trailing-block reading's skip condition (a transcript
ending in a tool result must not match a mid-chain call); `fingerprint.py`
byte-compatibility with #2180's functions, since #2180 will import from it.

### PR 3 — echo-contract conformance (independent, any time)
A shared test asserting `verify()` returns `response` unchanged + a doc line in the
environment-tutorial pages. Carries the survey evidence from §2.5 and documents
`terminus_judge` as the known, cost-free exception.

### Follow-ups (separate PRs, not on this branch)
- **Shared transcript-synthesis helper** (one PR, central): the never-mint-envelope-ids
  rule of §2.5.1 — default `id=""`, optional `response_id=` opt-in — and migrate the
  blackbox wrappers onto it. This *replaces* any per-agent id-extraction program; the
  content witness carries synthesizers by design.
- **The observation witness** (one PR, central): consume `ng_agent_observations` in the
  finalize path — the root invocation's last `ModelCallRef`, resolved against
  `TokenEntry.response_id`, joins as a harness-declared terminal witness (§2.5.1).
  Zero per-agent work: the five instrumented wrappers already produce the bundle.
- **Compaction linkage pass** (forest scope, §2.8): the typed between-tree graph —
  declared compaction edges corroborated by the captured summary call and kept item
  ids, tree labels exposed to delivery for reward attribution. Gated on the
  multi-sequence delivery contract existing first.
- **Non-generative endpoint safe-list** (independent, small): prefixed requests to
  known non-generative endpoints stop marking the rollout incomplete (§2.9). The
  typed-incompleteness / deliver-despite-off-chain-holes refinement follows with #2180.
- **RL seal carries the declared terminal** — with the gate-successor (#2278 rebase); the
  seal's terminal feeds the `explicit` witness, replacing last-committed-wins.
- **`image_tools_agent` scores/persists mismatch** — pre-existing, surfaced by the survey;
  file as its own issue.
- **Masked-fraction measurement** on a real harness run once PR 2 lands.

### Commit slicing on this branch

Every commit carries the tests for the behavior it introduces, so each is independently
testable (and bisectable):

1. `feat(token-id-capture): record the served envelope id on every captured call`
   (+ `test_served_response_id.py`) → PR 1
2. `feat(token-id-capture): extract pure conversation fingerprints`
   (+ `test_conversation_fingerprint.py`) → PR 2
3. `feat(token-id-capture): attribute the verified terminal call by corroborating witnesses`
   (+ the witnesses half of `test_terminal_attribution.py`) → PR 2
4. `feat(token-id-capture): anchor chain selection to the attributed terminal`
   (+ the builder/mask/delivery half of `test_terminal_attribution.py`) → PR 2
5. `docs: terminal attribution design + PR split` (this file; drop or move under
   `nemo_gym/token_id_capture/` docs before opening PRs)

---

## 6. Self-review: risks, low-confidence areas, and what this does not cover

An honest pre-review of this branch by its author. Items here are stated so the reviewer
can weigh them deliberately rather than discover them.

### 6.1 Highest-risk spots — challenge these first

- **The Anthropic id-reuse is the riskiest line in the branch.** The `/v1/messages`
  envelope id changed from a minted `msg_<uuid>` to the inner `resp_<hex>`. Nothing in
  the protocol requires the `msg_` prefix, but that claim is unproven against the client
  that matters most — the Claude Code CLI and the Anthropic SDK's response parsing. If
  anything validates the id format, the whole Messages dialect breaks for the largest
  blackbox harness. **Needs one real Claude Code rollout against a Gym server before
  PR 1 opens.** Fallback design if it fails: keep minting `msg_…` and report the outer id
  back into the capture context instead (more plumbing; that is why reuse was tried
  first). The same smoke test must also check whether the CLI hits non-message endpoints
  (`count_tokens`, `models`) under the capture prefix — each such hit marks the rollout
  incomplete today (§2.9) and may be silently masking every Claude Code rollout.
- **Streaming paths are untested for id propagation.** All three dialects have SSE
  variants (Responses streaming, chat chunk replay, Anthropic SSE). The synthesized
  streams are built from the same complete response object, so the ids should ride along
  — but no test asserts that a harness reconstructing from chunks ends up holding the id
  capture recorded.
- **Schema v2 collides with #2278's v2** (its parent-linkage bump) — a second instance of
  the two-meanings-one-integer hazard. The #2278-successor rebase must renumber past
  whatever this stack and #2180/#2181 have claimed. (The separate writer–reader lockstep
  concern — an older reader hard-rejecting v2 records — is **not** a live issue: this
  capture stack has no deployed consumers today, so there is no mixed-version fleet to
  break. It becomes a real upgrade-ordering rule only once trainers pin Gym versions
  against shared capture stores.)
- **The mask-semantics change in `consumer.py`** is the one place existing behavior is
  deliberately altered. Verify independently that the unattributed path is bit-identical
  to today's policy; the delivered path's unmask is the intended new behavior.

### 6.2 Low confidence — mostly missing coverage, not suspected bugs

- **Reasoning items.** The fingerprint deliberately skips reasoning and the trailing-block
  walk stops at a reasoning item; hand-tracing says the common shapes stay consistent
  (both sides skip reasoning, so hashes agree), but there is zero test coverage of
  `[…, reasoning, message]` transcripts — and reasoning-parser deployments are exactly
  the RL training case. See also the §2.2.0 reasoning-parser crack.
- **Tool-call finals** (submit/finish-style harnesses ending with a `function_call`). The
  fingerprint hashes `call_id`/name/args; the stack-v2 suite covered this shape, this
  branch's port does not.
- **Incomplete-capture ordering.** Capture-incompleteness masks *after* `_assemble`, so an
  attributed "delivered" chain still masks on incompleteness — verified by reading the
  code path, not pinned by a test. The §2.7 missing-record gap depends on this invariant.
- **The #2180 rebase must restore the `unresolved_boundary` check.** On this base the
  terminal path's "broken" test covers only `quarantined`; the stack-v2 reference also
  refuses a digest-unproven boundary. If the rebase drops that, a broken link could be
  reported "delivered".
- **Process gates not yet run on the branch:** `pre-commit run --all-files` and the 96%
  coverage requirement on the new modules (only ruff on changed files + the three
  affected suites have been run).

### 6.3 Not covered by this branch — still needed by the three threads

**Blackbox training:**
- Live-harness validation: nothing here has run against a real model server + harness
  (TestClient only). The masked-fraction baseline is the first real measurement.
- The shared transcript-synthesis helper (never-mint-ids, §2.5.1) and wrapper migrations.
- The echo-contract conformance PR (designed as PR 3, not built).
- The cross-dialect fingerprint conformance test (§2.2.0).
- #2180/#2181 themselves: attribution picks the right chain, but the *bit-exact tokens*
  guarantee still depends on verified lineage and prefix supply landing.
- Sub-agent branches are excluded from the trained chain, not trained: turning discarded
  branches into additional training samples is unaddressed everywhere.

**Lineage / TransferQueue integration:** untouched by this branch. Still open: the
first-party TQ adapter, the read-after-ack measurement, the conformance kit in RL CI —
and, made visible by this work, **attribution has no transport in the TQ deployment
yet**: the seal sends only `{"reward"}` (no explicit witness), and the gate-successor's
ledger rows carry `logical_request_id` but no served envelope id. Either a `response_id`
custody column is added, or the observation is exploited that gate-mode's fallback
(`logical_request_id := payload["id"]` when no header is sent) already *is* the served
id. Mapping the witnesses join onto the token-free manifest is designed in spirit,
unmapped in schema.

**Partial-rollout checkpointing:** untouched, deliberately. One interaction to keep in
view: the session-state boundary record's `response_id` anchor and the TokenEntry's
`response_id` are now the same concept by construction, but what "terminal" means across
a resume boundary — attempt 1's kept prefix stitched to attempt 2's chain in the
finalizer — remains undesigned.

---

## Appendix A — verify()/run-result audit (main @ fa0d25158, 2026-08-21)

The evidence behind §2.5. Method: every directory under `resources_servers/` (114) and
`responses_api_agents/` (39) was read for how its `verify()` (or run-result construction)
treats the rollout's `response`. Line numbers refer to `main` @ `fa0d25158`.

### A.1 The contract as defined

`nemo_gym/base_resources_server.py:88-97`: `BaseVerifyRequest(BaseRunRequest)` adds
`response: NeMoGymResponse`; `BaseVerifyResponse(BaseVerifyRequest)` adds `reward`.
Because the verify response *inherits from the request*, `response` is a required,
non-Optional field — a true "drop" is structurally impossible unless a subclass
redeclares it, and none does. `verify` is abstract (`:171-173`); there is no shared echo
helper — the echo happens through the `SomeVerifyResponse(**body.model_dump(),
reward=…)` idiom, which is what every tutorial page teaches
(`single-step-environment.mdx:263`, `mcp-resources-server.mdx:140`,
`stateful-environment.mdx:136`, `new-environment.mdx:222`,
`multi-reward-verification.mdx:155`).

Framework-level touchpoints:
- `nemo_gym/judge.py:85-108` — `judge_failsafe` wraps every `/verify` route and echoes
  on its failure path too (`data = body.model_dump() | {...}`).
- `nemo_gym/mcp_auto_exposure.py:620-672` — `_wrap_verify` hands verify a deep copy with
  normalized tool names and restores the emitted names on the result by `call_id`. Net
  identity for an echoing verify; the restore silently no-ops on a verify that
  reconstructs function-call items.
- `nemo_gym/rollout_reverification.py:413` — reverify rebuilds the verify request from
  the persisted row's `response`; a non-echoing verify makes reverify score a different
  object than the original run. (An existing echo dependency, predating this work.)

### A.2 Resources servers — deviations (the only non-echo cases)

| Server | Evidence | Class |
|---|---|---|
| `terminus_judge` | `app.py:270` verify → `_build_response` (`:282-297`): `_sanitize_pydantic_model(body.response)` round-trips model_dump → recursive `_sanitize_for_json` → model_validate; `_sanitize_unicode_string` (`:154-162`) rewrites every string via `encode("utf-8","surrogatepass").decode("utf-8","replace")`. All 12 return paths go through it. Its own test asserts the rewrite (`tests/test_app.py:810`). Identity on clean strings; only lone-surrogate content changes — which the capture store's orjson cannot persist anyway, so affected rollouts are already capture-incomplete and masked. | MUTATES |
| gymnasium family — `gymnasium`, `blackjack`, `example_multi_turn_gymnasium`, `grl_sokoban`, `grl_tetris`, `openair_congestion`, `tales` | `resources_servers/gymnasium/base.py:116-117`: `verify()` raises `NotImplementedError` — the `/step` protocol; the agent builds the run result | N/A |

### A.3 Resources servers — the 106 echoing servers

Three idioms, no shared base helper. **splat** — `X(**body.model_dump(), reward=…)` (or a
dict copy popping only request-side keys; grep for `pop("response")` /
`del …["response"]` / `exclude={…"response"…}` across `resources_servers/` returns zero
hits, as does grep for in-place mutation of `body.response`). **explicit** —
`response=body.response`: `genrm_compare:214,280`, `image_tools:564`, `swe_pivot:754`,
`polymath:81`. **inherited** — `math_with_autograder:82` and `physics_judge:89` inherit
`math_with_judge`'s splat verify.

Full list (`server  L<verify def>  idiom`):

```
aalcr L61 splat            abstention L319 splat        arc_agi L99 splat
arena_judge L201 splat     asr_with_pc L327 splat       aviary L162 splat
bbq L178 splat             bigcodebench L82 splat       bird_sql L137 splat+helper
browsecomp_advanced_harness L1171 splat                 bunsenbench_chemistry_mcq L157 splat
calendar L49 splat         circle_click L72 splat       circle_count L44 splat
citation_if L139 splat     code_fim L220 splat          code_gen L147 splat
competitive_coding_challenges L306 splat                conversational_tool_use_simulation L1288 splat
critpt L250 splat          cvdp L190 splat (+_verify_objective L288/_verify_subjective L234)
deepswe L430 splat         equivalence_llm_judge L414 → _make_response L334 splat
equivalence_rule L74 splat ether0 L57 → _response L96 splat(exclude)
evalplus L164 splat        example_mcp_weather L126 splat
example_multi_step L87 splat                            example_session_state_mgmt L87 splat
example_single_tool_call L52 splat                      example_tool_call_multireward L136 splat
finance_agent_v2 L928 splat                             finance_sec_search L1213 splat
format_verification L48 splat                           frontierscience_judge L328 splat(exclude)
gdpval L361 → rubric L367 / comparison L477, splat      genrm_compare L207 explicit
google_search L136 splat   gpqa_diamond L60 splat(exclude)                graphwalks L71 splat
gui_coordinate L104 splat(exclude)                      hotpotqa_qa L104 splat
ifbench L174 splat         iheval L722 splat            image_tools L451 → _build L554 explicit
imo_gradingbench L212 splat                             imo_proofbench_judge L437 splat
indirect_prompt_injection L147 splat                    instruction_following L115 splat
inverse_if L270 splat      jailbreak_detection L266 splat                 labbench2_vlm L159 splat
lc_niah L144 splat         legal_agent_bench L101 splat litmus_agent L817 splat
longmemeval L311 splat     longmt_eval L228 splat       math_advanced_calculations L107 splat
math_formal_lean L398 splat                             math_proof_judgement L270 splat
math_with_code L213 splat  math_with_judge L174 splat (+math_with_autograder, physics_judge)
mcqa L340 splat(exclude)   mrcr L64 splat               multichallenge L248 splat
newton_bench L269 splat    ns_tools L404 splat          nvarc L198 splat
omniscience L282 splat(exclude)                         openenv L234 splat
over_refusal_detection L206 splat                       polymath L77 super+explicit
proof_genselect L63 splat  proof_judge L233 splat       proof_verification L189 splat
ragtruth L203 splat        reasoning_gym L53 splat      rolemrc L482 → reference L489 / judge L522
ruler L55 splat            ruler2 L267 splat            scicode L104 splat
simpleqa L265 splat(exclude)                            single_step_tool_use_with_argument_comparison L63 splat
spartqa L267 splat         speed_bench L517 splat       spider2_lite L132 splat+helper
string_match L359 splat(exclude)                        structeval L134 splat
structured_outputs L197 splat                           swe_pivot L573 → _build_response L734 explicit
swebench L262 splat        swerl_gen L96 splat          swerl_llm_judge L176 splat(exclude)
tavily_search L399 splat   terminal_multi_harness L66 splat               text_to_sql L225 splat
toolsandbox L537 splat     ugphysics_judge L181 splat   verifif L745 splat
vlm_eval_kit L85 splat     wmt_translation L430 splat   workplace_assistant L106 splat
xlam_fc L109 splat         xstest L192 splat
```

Near-misses checked and cleared: `text_to_sql:341`, `verifif:937`, `xstest:240`,
`tavily_search:419` return a locally named echoing response; `bbq:384/413`,
`multichallenge:305`, `jailbreak_detection:588/666`, `over_refusal_detection:353`,
`labbench2_vlm:227`, `equivalence_llm_judge:489`, `terminus_judge:543`, `text_to_sql:393`,
`xstest:266` set `response=` inside `JudgeEvaluation` side-records, not the rollout
response; `physics_judge:180` / `ugphysics_judge:319` synthesize a response only to wrap
a judge reply.

### A.4 Agents

**Call `/verify` and return its `response` untouched (25):** `aviary_agent`,
`claude_code_agent` (`:749/:765` — merges `turns_used`, `finished_naturally`,
`ng_agent_observations` at top level only), `cline_agent:727/742`, `codex_agent:665/680`,
`critpt_agent:129/137`, `cvdp_agent:391,795`, `finance_agent:511`, `hermes_agent:509/527`,
`kilocode_agent:546/561`, `langgraph_agent:132/141` (+ `orchestrator_agent:248`,
`parallel_thinking_agent:226`, `reflection_agent:221`, `rewoo_agent:270`),
`non_executing_simple_agent:119`, `openclaw_agent:769/787`, `opencode_agent:807/822`,
`pi_agent:707/722`, `prime_agent:508/523`, `scicode_agent:221/229`,
`simple_agent:337/352`, `speed_bench_agent:240`, `stirrup_agent:1383/1405`,
`tool_simulation_agent:85/95`, `toolsandbox_agent:215/221`, plus success paths of
`browsecomp_agent:799/819` and `remote_agent:426/434`, and `proof_refinement_agent:180/267`.

**Mutate or replace `response` after verify:**

| Agent | Evidence | What happens |
|---|---|---|
| `image_tools_agent` | `:372` verifies `_final_assistant_response(model_response)`; `:390` `verify_response_json["response"] = model_response.model_dump()` | REPLACE — verify scores the final turn only; the persisted row carries the full multi-step response (a pre-existing scored-vs-persisted mismatch) |
| `labbench2_vlm_agent` | `:123-127` + `_strip_image_blocks` (`:71-101`) | MUTATE — drops every `input_image` block from the whole result, `response` included; records a `multimodal_history_redacted` gap |
| `remote_agent` | `:467` | REPLACE on failure path with an empty response |
| `browsecomp_agent` | `:838-860` | REPLACE on agent-error path (last seen or minted-empty response) |
| `opencode_sandboxed_agent` | `:449-452` | echoes `response` but rewrites `responses_create_params.input` (inserts a system turn) |
| `stirrup_agent` | `:1336`, `:1264-1313`, `:1576` | pre-verify strip of `metadata`; judge-only mode substitutes a built response |

**Never call `/verify` — mint the response and reward themselves (12):** `anyswe_agent:654`,
`anyterminal_agent:791`, `gymnasium_agent` (`:226-235` — last envelope kept, output
replaced by merged transcript: id survives), `harbor_agent:320-343`,
`mini_swe_agent:215-228` (default response object), `mini_swe_agent_2:863-877`
(`responses[-1]` with overrides), `osworld_agent:937-972` (synthesized per-step
messages), `pinchbench:903`, `swe_agents` (`:3802-3839` — synthesizes
`id=f"swebench-{instance_id}"`, strips `response.metadata` after harvesting, re-attaches
token fields from `provider_specific_fields`), `tau2:197` (also rewrites
`responses_create_params.input`), `vcqa_agent:291`, `verifiers_agent:341`.

**`simple_agent`'s exact shape** (`:232-233`): `model_response.output = new_outputs;
model_response.usage = usage` — the object sent to verify is the *last call's envelope*
(its `id`, `model`, `created_at` are the final call's) carrying the merged transcript of
all steps. This is why the id witness works for native agents with zero changes.
