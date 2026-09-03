# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Throwaway: exercise AgentCheckpointParticipant state-transition holes."""

import asyncio
import os
import sys
import time


sys.path.insert(0, os.environ.get("GYM_STACK_ROOT", os.getcwd()))

from nemo_gym.checkpoint.agent import (  # noqa: E402
    AgentBoundaryRecord,
    AgentCheckpointError,
    AgentCheckpointParticipant,
    AgentExecutionState,
)


def rec(idx, attempt=0, tag="x"):
    return AgentBoundaryRecord(
        rollout_id="r1",
        attempt_index=attempt,
        boundary_index=idx,
        output_items=[{"type": "message", "role": "assistant", "content": tag}],
    )


async def h2_failed_after_boundary_drops_record():
    p = AgentCheckpointParticipant()
    ex = await p.begin("r1", 0, task=asyncio.current_task())
    await p.commit_boundary(rec(1))  # RUNNING, boundary kept
    prep = asyncio.create_task(p.prepare(time.time() + 2))
    await asyncio.sleep(0)
    assert ex.state == AgentExecutionState.PARK_REQUESTED
    # model returns 409 checkpoint_parked -> raise_for_status -> /run wrapper finish(failed)
    await p.finish(ex, outcome="failed")
    report = await prep
    print("H2 prepare report:", {k: report[k] for k in ("running", "parked", "active")})
    print("H2 records_for_commit:", len(p.records_for_commit()), "(boundary 1 existed but is dropped)")


async def h9_retire_does_not_stop_other_task_loop():
    """SimpleAgent: parked coroutine runs in the /v1/responses handler task, not the /run task."""
    p = AgentCheckpointParticipant()
    run_task_holder = {}
    inner_progress = []

    async def run_task():
        ex = await p.begin("r1", 0, task=asyncio.current_task())
        run_task_holder["ex"] = ex
        try:
            await asyncio.Event().wait()  # awaiting the self HTTP call forever
        except asyncio.CancelledError:
            await p.finish(ex, outcome="cancelled")
            raise

    async def inner_loop():
        # the /v1/responses handler: separate task
        await asyncio.sleep(0.01)
        await p.commit_boundary(rec(1))
        inner_progress.append("after boundary 1 (resumed)")
        # loop continues to next model call + boundary under a retired attempt
        await p.commit_boundary(rec(2))
        inner_progress.append("after boundary 2")

    rt = asyncio.create_task(run_task())
    it = asyncio.create_task(inner_loop())
    await asyncio.sleep(0.02)
    prep = asyncio.create_task(p.prepare(time.time() + 2))
    await asyncio.sleep(0.05)
    print("H9 status after park:", p.status()["executions"])
    await p.retire("r1", 0)
    try:
        await rt
    except asyncio.CancelledError:
        pass
    await asyncio.sleep(0.05)
    print("H9 inner loop progress after retire:", inner_progress, "inner done:", it.done())
    await prep
    # begin the same key again after RETIRED (allowed) -> new execution object under same key
    ex2 = await p.begin("r1", 0, task=asyncio.current_task()) if p._accepting else None
    print("H9 begin same key after retire allowed (needs resume first):", ex2 is not None)
    await p.resume()
    ex2 = await p.begin("r1", 0, task=asyncio.current_task())
    print("H9 begin same key after retire+resume allowed:", ex2 is not None, "state", ex2.state)


async def h9_stale_loop_contaminates_new_execution():
    p = AgentCheckpointParticipant()
    await p.begin("r1", 0, task=None)
    await p.commit_boundary(rec(3, tag="old"))
    await p.retire("r1", 0)
    ex_new = await p.begin("r1", 0, task=None)
    # old (still running) inner loop commits its next boundary: lands on ex_new
    await p.commit_boundary(rec(4, tag="old-continued"))
    print(
        "H9 new execution boundary came from old loop:", ex_new.boundary.output_items, ex_new.boundary.boundary_index
    )
    try:
        await p.commit_boundary(rec(1, tag="new"))
    except AgentCheckpointError as e:
        print("H9 new run's own first boundary rejected:", e)


async def h9_retire_after_commit_before_resume(tmp_path):
    from nemo_gym.checkpoint.agent import commit_agent_state, restore_agent_state

    p = AgentCheckpointParticipant()
    ex = await p.begin("r1", 0, task=None)
    prep = asyncio.create_task(p.prepare(time.time() + 2))
    await asyncio.sleep(0)
    park = asyncio.create_task(p.commit_boundary(rec(2)))
    await prep
    commit_agent_state(p, tmp_path, checkpoint_id="c1")
    await p.retire("r1", 0)  # fence allows retire only in IDLE/PREPARING/PREPARED, but participant does not care
    fresh = AgentCheckpointParticipant()
    restore_agent_state(fresh, tmp_path)
    print("H9 restore installs continuation for retired attempt:", fresh.continuation("r1", 1) is not None)
    await park
    del ex


async def h4_memory():
    p = AgentCheckpointParticipant()
    for i in range(5000):
        ex = await p.begin(f"r{i}", 0, task=None)
        await p.commit_boundary(
            AgentBoundaryRecord(
                rollout_id=f"r{i}",
                attempt_index=0,
                boundary_index=1,
                output_items=[{"type": "message", "role": "assistant", "content": "x" * 2000}],
            )
        )
        await p.finish(ex, outcome="completed")
    print(
        "H4 executions retained after 5000 completed runs:",
        len(p._executions),
        "each holding boundary:",
        all(e.boundary is not None for e in p._executions.values()),
    )
    # restored entries never pruned either
    p.install_restored([rec(1)])
    await p.resume()
    ex = await p.begin("r1", 1, task=None)
    await p.finish(ex, outcome="completed")
    print("H4 restored entry still present after continuation completed:", p.continuation("r1", 1) is not None)


async def prepare_failure_leaves_admission_closed():
    p = AgentCheckpointParticipant()
    ex = await p.begin("r1", 0, task=None)
    report = await p.prepare(time.time() + 0.01)
    print("prepare timed out; running:", report["running"], "accepting:", p._accepting, "state:", ex.state)
    # controller gives up (no abort route). Next boundary of this run parks with nobody to resume it.
    park = asyncio.create_task(p.commit_boundary(rec(1)))
    await asyncio.sleep(0.05)
    print("run parked with no active checkpoint:", ex.state, "park task done:", park.done())
    await p.retire("r1", 0)
    try:
        await park
    except BaseException as e:  # noqa: BLE001
        print("park ended with", type(e).__name__)


async def main():
    import tempfile
    from pathlib import Path

    await h2_failed_after_boundary_drops_record()
    await h9_retire_does_not_stop_other_task_loop()
    await h9_stale_loop_contaminates_new_execution()
    with tempfile.TemporaryDirectory() as d:
        await h9_retire_after_commit_before_resume(Path(d))
    await h4_memory()
    await prepare_failure_leaves_admission_closed()


asyncio.run(main())
