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
"""Probe H1/H6/H7: phase holes on coordinator and single-worker routes, restore-failure divergence."""

import asyncio
import shutil
import tempfile
from pathlib import Path

import httpx
from fastapi import FastAPI
from starlette.testclient import TestClient

from nemo_gym.checkpoint import (
    MODEL_ADMISSION_URL_PREFIX,
    MODEL_CHECKPOINT_URL_PREFIX,
    AdmissionCoordinator,
    AdmissionLimiter,
    ControlCapabilities,
    ControlFence,
    MultiProcessCapability,
    WorkerAdmissionAgent,
    build_coordinator_control_app,
    install_control_plane,
    install_model_admission,
    install_model_checkpoint,
)
from nemo_gym.token_id_capture.lineage import FileLineageStore


AUTH = {"authorization": "Bearer s"}


async def coordinator_probe() -> None:
    sock_dir = Path(tempfile.mkdtemp(prefix="ngp-"))
    coord = AdmissionCoordinator(sock_dir / "c.sock", expected_workers=2)
    await coord.start()
    workers = []
    for i in range(2):
        lim = AdmissionLimiter()
        ag = WorkerAdmissionAgent(coord.socket_path, f"w{i}", lim)
        await ag.start()
        workers.append((lim, ag))
    await coord.wait_until(lambda s: s["workers"]["live"] == 2, timeout_s=2)
    app = build_coordinator_control_app(
        coord,
        auth_token="s",
        capabilities=ControlCapabilities(
            component="responses_api_models",
            name="p",
            multi_process=MultiProcessCapability(mode="coordinator", num_workers=2),
            instance_role="policy",
        ),
        ack_timeout_s=1.0,
    )
    held = workers[0][0].admit(rollout_id="4-2", attempt_index=0)
    await coord.wait_until(lambda s: s["inflight_total"] == 1, timeout_s=2)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://c", headers=AUTH) as c:
        pause = await c.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "k1", "deadline_ts": 4e9})
        print("H1 coordinator pause while straggler:", pause.status_code, pause.json())
        abort = await c.post(
            f"{MODEL_ADMISSION_URL_PREFIX}/abort_inflight",
            json={"checkpoint_id": "k1", "deadline_ts": 4e9, "rollout_id": "4-2", "attempt_index": 0},
        )
        print("H1 coordinator abort_inflight during PREPARING:", abort.status_code, abort.json())
        resume = await c.post(f"{MODEL_ADMISSION_URL_PREFIX}/resume", json={"checkpoint_id": "k1", "deadline_ts": 4e9})
        print("H1 coordinator resume during PREPARING:", resume.status_code, resume.json())
        pause2 = await c.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "k2", "deadline_ts": 4e9})
        print("H1 coordinator new checkpoint while stuck:", pause2.status_code, pause2.json())
    workers[0][0].release(held)
    for _, ag in workers:
        await ag.stop()
    await coord.stop()
    shutil.rmtree(sock_dir, ignore_errors=True)


def single_worker_probe() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="ngsw-"))
    app = FastAPI()
    limiter = AdmissionLimiter()
    fence = ControlFence()
    ledger = FileLineageStore(tmp / "ledger")
    install_control_plane(
        app,
        capabilities=ControlCapabilities(
            component="responses_api_models",
            name="p",
            multi_process=MultiProcessCapability(mode="single_worker", num_workers=1),
            instance_role="policy",
        ),
        fence=fence,
    )
    install_model_admission(app, limiter=limiter, fence=fence, instance_role="policy", auth_token="s")
    install_model_checkpoint(
        app,
        fence=fence,
        limiter=limiter,
        ledger_provider=lambda: ledger,
        file_ledger_root_provider=lambda: ledger.checkpoint_root,
        instance_role="policy",
        auth_token="s",
    )
    client = TestClient(app)

    # H7: restore from a bad dir -> limiter closed, fence back to IDLE.
    bad = client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/restore",
        json={"checkpoint_id": "r1", "deadline_ts": 4e9, "checkpoint_dir": str(tmp / "nope")},
        headers=AUTH,
    )
    print(
        "H7 restore bad dir:",
        bad.status_code,
        bad.json()["error"]["code"],
        "| fence:",
        fence.phase.value,
        "| limiter:",
        limiter.state.value,
    )
    resume = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/resume", json={"checkpoint_id": "r1", "deadline_ts": 4e9}, headers=AUTH
    )
    print("H7 resume after failed restore:", resume.status_code, resume.json())
    resume2 = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/resume", json={"checkpoint_id": "r2", "deadline_ts": 4e9}, headers=AUTH
    )
    print("H7 resume with new id after failed restore:", resume2.status_code, resume2.json())
    print("H7 limiter state now:", limiter.state.value)

    # Recover via pause+resume with a new id, then do a real commit/restore for H6.
    client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "k0", "deadline_ts": 4e9}, headers=AUTH)
    client.post(f"{MODEL_ADMISSION_URL_PREFIX}/resume", json={"checkpoint_id": "k0", "deadline_ts": 4e9}, headers=AUTH)
    print("recovered limiter:", limiter.state.value, "fence:", fence.phase.value)

    limiter.release(limiter.admit(rollout_id="4-2", attempt_index=0))
    (tmp / "ledger" / "4-2.lineage.jsonl").write_text('{"model_call_id":"c1","staging_key":"s"}\n')
    client.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "k1", "deadline_ts": 4e9}, headers=AUTH)
    commit = client.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
        json={"checkpoint_id": "k1", "deadline_ts": 4e9, "checkpoint_dir": str(tmp / "ck1")},
        headers=AUTH,
    )
    print("commit:", commit.status_code, commit.json(), "fence:", fence.phase.value)
    status = client.get(
        f"{MODEL_ADMISSION_URL_PREFIX}/status", params={"checkpoint_id": "k1", "deadline_ts": 4e9}, headers=AUTH
    )
    print("H6 status in COMMITTED_PAUSED:", status.status_code, status.json())
    abort = client.post(
        f"{MODEL_ADMISSION_URL_PREFIX}/abort_inflight",
        json={"checkpoint_id": "k1", "deadline_ts": 4e9, "rollout_id": "4-2", "attempt_index": 0},
        headers=AUTH,
    )
    print("H6 abort in COMMITTED_PAUSED:", abort.status_code, abort.json())

    # Fresh process restore -> status.
    app2 = FastAPI()
    limiter2 = AdmissionLimiter()
    fence2 = ControlFence()
    ledger2 = FileLineageStore(tmp / "ledger2")
    install_control_plane(
        app2,
        capabilities=ControlCapabilities(
            component="responses_api_models",
            name="p",
            multi_process=MultiProcessCapability(mode="single_worker", num_workers=1),
            instance_role="policy",
        ),
        fence=fence2,
    )
    install_model_admission(app2, limiter=limiter2, fence=fence2, instance_role="policy", auth_token="s")
    install_model_checkpoint(
        app2,
        fence=fence2,
        limiter=limiter2,
        ledger_provider=lambda: ledger2,
        file_ledger_root_provider=lambda: ledger2.checkpoint_root,
        instance_role="policy",
        auth_token="s",
    )
    c2 = TestClient(app2)
    restore = c2.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/restore",
        json={"checkpoint_id": "r9", "deadline_ts": 4e9, "checkpoint_dir": str(tmp / "ck1")},
        headers=AUTH,
    )
    print("restore:", restore.status_code, restore.json(), "fence:", fence2.phase.value)
    status = c2.get(
        f"{MODEL_ADMISSION_URL_PREFIX}/status", params={"checkpoint_id": "r9", "deadline_ts": 4e9}, headers=AUTH
    )
    print("H6 status in RESTORED_PAUSED:", status.status_code, status.json())
    # Second-generation checkpoint on the restored server: is the restored file kept?
    c2.post(f"{MODEL_ADMISSION_URL_PREFIX}/resume", json={"checkpoint_id": "r9", "deadline_ts": 4e9}, headers=AUTH)
    c2.post(f"{MODEL_ADMISSION_URL_PREFIX}/pause", json={"checkpoint_id": "k2", "deadline_ts": 4e9}, headers=AUTH)
    commit2 = c2.post(
        f"{MODEL_CHECKPOINT_URL_PREFIX}/commit",
        json={"checkpoint_id": "k2", "deadline_ts": 4e9, "checkpoint_dir": str(tmp / "ck2")},
        headers=AUTH,
    )
    print("H2/H4 second-generation commit:", commit2.status_code, commit2.json())
    print("ck2 files:", sorted(p.name for p in (tmp / "ck2" / "model-ledger").iterdir()))
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(coordinator_probe())
    print("=" * 60)
    single_worker_probe()
