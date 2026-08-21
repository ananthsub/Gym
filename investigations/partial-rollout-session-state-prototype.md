# Partial rollout checkpointing: session-state prototype

Working prototype of the Gym-side mechanism behind "resume at a completed-call
boundary when environment state is recoverable" (see
[partial-rollout-checkpointing-high-level.md](partial-rollout-checkpointing-high-level.md)).
A rollout killed mid-flight — trainer restart, node death, preemption — resumes
in fresh processes at its last durable tool boundary instead of restarting from
the prompt.

## Mechanism

- **Identity.** Everything keys off the logical rollout id (`_ng_rollout_id`,
  minted by the caller pre-dispatch). Agents stamp it as a `/ng-rollout/<id>/`
  URL prefix on every downstream call; session middleware derives
  `session_id := rollout id` from the prefix, so the session cookie, the
  in-memory session key, and every storage path are pure functions of the id —
  any worker or process generation re-derives them with no handoff.
- **Durability.** After each tool step, the agent commits a *tool boundary*:
  environment state first, then a `ToolBoundaryRecord` (the step's conversation
  delta + accumulated usage + the model `response_id` anchor) appended to
  `<session_state_dir>/<rollout_id>/boundaries.jsonl`. The record is the commit
  point; a crash between the two writes leaves an orphan snapshot that restore
  never selects. Everything is fsync'd before acknowledgment; any checkpoint
  cut is crash-consistent by construction — nothing is triggered at cut time.
- **Resume.** Redispatch the same run body plus `_ng_resume: true`. The agent
  selects the latest *resumable* boundary (`select_resume_records`), restores
  the environment via `POST /ng-session/restore`, rebuilds the conversation
  from the records, and re-enters its loop. Any restore failure falls back to
  abandon-and-redispatch (status quo).
- **Capability tiers.** Servers declare support via three overridable hooks on
  `SimpleResourcesServer` (`supports_session_state`, `export_session_state`,
  `restore_session_state`). Stateless verify-only servers need nothing —
  conversation rebuild alone resumes them. Sandbox-backed servers (class C:
  export a reconnect descriptor via `AsyncSandbox.serialize/connect`, plus an
  optional provider filesystem-snapshot capability for exact rewind) fit the
  same hooks but are deliberately not part of this branch; they land as a
  follow-up once the sandbox-provider surface settles.

## Ported pairs (each with kill-and-resume tests)

| Pair | State class | Proof |
| --- | --- | --- |
| simple_agent + example_session_state_mgmt | A (trivial) | resume with 1 model call, no tool re-execution, reward 1.0 |
| gymnasium_agent + blackjack | A/B (unseeded RNG) | bit-identical dealer draws after resume, predicted from the durable RNG snapshot |
| simple_agent + workplace_assistant | A (pandas frames) | frame-for-frame fidelity incl. a deletion; snapshot beats replay for logical state |

## Shared-filesystem performance (Lustre)

Small environment states (≤32KB serialized) ride *inline* in the boundary
record: one append + one fsync per boundary, no snapshot file. Store
construction costs zero filesystem operations; store IO runs on a dedicated
bounded executor; `flock` degrades gracefully on mounts without `-o flock`.
`session_state_snapshot_every_n_steps` trades snapshot IO for regenerated
steps on resume (boundary records are still appended every step).

## Reconciling with captured tokens (TransferQueue)

Resume at boundary N means the dead attempt's captured model calls *after* N
are orphans: the resumed attempt regenerates those steps under its own attempt
key. In the native path (token ids on output items) this is automatic — the
final transcript is canonical and orphans never enter it. In the staged/TQ
path, the boundary record's `response_id` anchor partitions captured calls
into kept-prefix / orphans / new-chain; finalization stitches prefix + new
chain and retires orphans (never quarantines them as retry-ambiguity). The
cadence knob widens the orphan window but does not change the mechanism — the
window exists even at cadence 1 (kill between a call's capture and its
boundary commit).

## Known simplifications

Boundary records carry the conversation delta directly (dedupe with the token
capture store is open); record retirement/tombstone lifecycle is left to the
caller; the store factory is not yet config-pluggable
(`SessionStateStore` protocol exists); gate/control-plane adopt-on-resume and
the finalizer-side stitch are RL-side follow-ups.
