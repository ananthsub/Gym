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
"""Throwaway probes for the checkpoint audit. Run from the gym-stack worktree with `uv run python`."""

import asyncio
import tempfile
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.checkpoint import (
    MODEL_ADMISSION_URL_PREFIX,
    MODEL_CHECKPOINT_URL_PREFIX,
    AdmissionLimiter,
    AgentBoundaryRecord,
    AgentCheckpointParticipant,
    ControlCapabilities,
    ControlFence,
    MultiProcessCapability,
    commit_agent_state,
    install_control_plane,
    install_model_admission,
    install_model_checkpoint,
    restore_agent_state,
)
from nemo_gym.token_id_capture.lineage import FileLineageStore


AUTH = {"authorization": "Bearer secret"}


def probe_model_restore_failure_leaves_admission_closed(tmp: Path) -> None:
    app = FastAPI()
    limiter = AdmissionLimiter()
    fence = ControlFence()
    ledger = FileLineageStore(tmp / "ledger")
    install_control_plane(
        app,
        capabilities=ControlCapabilities(
            component="responses_api_models",
            name="policy",
            multi_process=MultiProcessCapability(mode="single_worker", num_workers=1),
            instance_role="policy",
        ),
        fence=fence,
    )
    install_model_admission(app, limiter=limiter, fence=fence, instance_role="policy", auth_token="secret")
    install_model_checkpoint(
        app,
        fence=fence,
        limiter=limiter,
        ledger_provider=lambda: ledger,
        file_ledger_root_provider=lambda: ledger.checkpoint_root,
        instance_role="policy",
        auth_token="secret",
    )
    client = TestClient(app)
    # Restore from a directory with no manifest (torn commit) -> raises ledger_mismatch.
    (tmp / "empty-ckpt").mkdir()
    restore = client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/restore",
        json={"checkpoint_id": "r1", "deadline_ts": 4e9, "checkpoint_dir": str(tmp / "empty-ckpt")},
        headers=AUTH,
    )
    print("[A] restore status", restore.status_code, restore.json()["error"]["code"])
    print("[A] fence phase after failed restore:", fence.phase.value, "| limiter state:", limiter.state.value)
    resume = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/resume",
        json={"checkpoint_id": "r1", "deadline_ts": 4e9},
        headers=AUTH,
    )
    print("[A] resume after failed restore ->", resume.status_code, resume.json())
    gen = client.post("/v1/responses", json={})
    print("[A] data plane after failed restore ->", gen.status_code, gen.json() if gen.status_code == 409 else "")


async def probe_agent_commit_dir_collision(tmp: Path) -> None:
    """Two agent servers sharing one checkpoint_dir: the second commit silently returns the first's manifest."""

    async def parked_participant(rollout: str) -> AgentCheckpointParticipant:
        p = AgentCheckpointParticipant()
        await p.begin(rollout, 0, task=asyncio.current_task())
        prepare = asyncio.create_task(p.prepare(time.time() + 2))
        await asyncio.sleep(0)
        asyncio.create_task(
            p.commit_boundary(
                AgentBoundaryRecord(rollout_id=rollout, attempt_index=0, boundary_index=1, output_items=[])
            )
        )
        await prepare
        return p

    agent_a = await parked_participant("rollout-from-agent-A")
    agent_b = await parked_participant("rollout-from-agent-B")
    ckpt = tmp / "shared-ckpt"
    ra = commit_agent_state(agent_a, ckpt, checkpoint_id="c1")
    rb = commit_agent_state(agent_b, ckpt, checkpoint_id="c1")
    print("[B] agent A commit:", ra)
    print("[B] agent B commit:", rb, "<- same manifest; B's record was never written")
    print("[B] files on disk:", sorted(p.name for p in (ckpt / "agent").iterdir()))
    restored_b = AgentCheckpointParticipant()
    restore_agent_state(restored_b, ckpt)
    print(
        "[B] restored on agent B: continuation for B's rollout =", restored_b.continuation("rollout-from-agent-B", 1)
    )
    print(
        "[B] restored on agent B: continuation for A's rollout =",
        restored_b.continuation("rollout-from-agent-A", 1) is not None,
    )


async def probe_retire_wakes_parked_inner_task() -> None:
    """retire() sets resume_event -> a parked inner /v1/responses task wakes and continues as RETIRED."""
    p = AgentCheckpointParticipant()
    outer_cancelled = asyncio.Event()

    async def outer():
        try:
            await asyncio.sleep(100)
        except asyncio.CancelledError:
            outer_cancelled.set()
            raise

    outer_task = asyncio.create_task(outer())
    execution = await p.begin("r", 0, task=outer_task)
    prepare = asyncio.create_task(p.prepare(time.time() + 2))
    await asyncio.sleep(0)
    steps = []

    async def inner_episode():
        # mimic SimpleAgent._create_episode loop: commit boundary, then continue to next model call
        await p.commit_boundary(
            AgentBoundaryRecord(rollout_id="r", attempt_index=0, boundary_index=1, output_items=[])
        )
        steps.append("continued past boundary 1 -> next model/tool call would be issued here")
        await p.commit_boundary(
            AgentBoundaryRecord(rollout_id="r", attempt_index=0, boundary_index=2, output_items=[])
        )
        steps.append("continued past boundary 2")

    inner = asyncio.create_task(inner_episode())
    await prepare
    print("[C] parked:", p.status()["parked"], "state:", execution.state.value)
    await p.retire("r", 0)
    await asyncio.sleep(0.05)
    print(
        "[C] outer /run task cancelled:",
        outer_cancelled.is_set(),
        "| inner task done:",
        inner.done(),
        "| steps:",
        steps,
    )
    print("[C] execution state:", execution.state.value, "| boundary index now:", execution.boundary.boundary_index)


def main() -> None:
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        probe_model_restore_failure_leaves_admission_closed(tmp)
        asyncio.run(probe_agent_commit_dir_collision(tmp))
        asyncio.run(probe_retire_wakes_parked_inner_task())


if __name__ == "__main__":
    main()
