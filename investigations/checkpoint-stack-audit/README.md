# Checkpoint stack audit

Repro probes and reports from the review of NVIDIA-NeMo/Gym PRs 2939-2946
(the partial-rollout checkpointing stack, tip `5b189df61` on top of
`pthombre/tokcap/full-stack`).

- [checkpoint-stack-review.html](../checkpoint-stack-review.html): the lifecycle
  as the stack implements it and the 42-case edge catalogue (case ids `E1`-`E42`).
- [checkpoint-fix-plan.html](../checkpoint-fix-plan.html): the nine workstreams
  that close those cases and the blackbox-harness continuation design.

## What the probes are

Each script reproduces one or more findings against the PR stack, not against
this branch. They were written as throwaway probes during the audit; their
module docstrings name the hypothesis they exercise, and the review page's
case entries cite the probe outputs. They are kept here so the findings stay
reproducible until they become strict-xfail characterization tests at the base
of the PR stack, which is the first step of the verification protocol in the
fix plan.

| Directory | Plane | Probes |
| --- | --- | --- |
| `control-model/` | control fence, admission, coordinator, model ledger | fence phases after failed restore and stuck drains; abort under the production middleware stack |
| `agent/` | agent participant, SimpleAgent, GymnasiumAgent | refusal mid-turn losing or corrupting boundaries; execution growth; retire cancelling the wrong task; boundary at `max_steps` |
| `resources/` | resources participant, session middleware, adapters | identity-less callers; revision race at prepare; verify in flight at prepare; lock and tombstone growth; adapter fidelity; pre-restore acceptance |
| `choreography/` | cross-participant seams | failed restore leaves admission closed; checkpoint directories not namespaced per instance; zombie loop after retire |

## Running a probe

Check out the stack tip in a worktree and run a probe from that root with the
stack's environment:

```bash
git fetch upstream pull/2946/head:refs/pr/2946
git worktree add /path/to/gym-stack refs/pr/2946
cd /path/to/gym-stack
GYM_STACK_ROOT=$PWD uv run --extra dev python /path/to/investigations/checkpoint-stack-audit/resources/h4b_inflight_verify_race.py
```

`GYM_STACK_ROOT` defaults to the current directory. A probe prints the
observed behavior; the expected (defective) output is described in the review
page entry it belongs to.
