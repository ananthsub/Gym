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
"""Session checkpointing for workplace_assistant (class-A logical state).

The whole session is pandas DataFrames inside tool containers; export
serializes them generically and restore rebuilds a fresh tool env with the
frames overwritten. The test kills the server (fresh instance, empty memory)
after several mutations across containers and proves the restored environment
is frame-for-frame identical — including a deletion, which pure replay of a
fresh env would also reproduce, but which here needs no replay at all.
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from resources_servers.workplace_assistant.app import (
    WorkbenchResourcesServer,
    WorkbenchResourcesServerConfig,
)


def _frames_as_json(server: WorkbenchResourcesServer, session_id: str) -> dict:
    return {
        name: {
            attr: frame.to_json(orient="split") for attr, frame in vars(container).items() if hasattr(frame, "to_json")
        }
        for name, container in server.session_id_to_tool_env[session_id]["containers"].items()
    }


def _make_client(tmp_path) -> tuple[TestClient, WorkbenchResourcesServer]:
    config = WorkbenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="workplace_assistant",
        session_state_dir=str(tmp_path),
    )
    server = WorkbenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    return TestClient(server.setup_webserver(), raise_server_exceptions=False), server


class TestSessionState:
    def test_export_restore_across_server_restart(self, tmp_path) -> None:
        first, first_server = _make_client(tmp_path)
        assert first.post("/ng-rollout/wa-1/seed_session", json={}).status_code == 200

        # Mutate three containers: send an email, delete an email, create an event.
        response = first.post(
            "/ng-rollout/wa-1/email_send_email",
            json={"recipient": "carlos.gomez@atlas.com", "subject": "Resume test", "body": "checkpoint me"},
        )
        assert "Error" not in str(response.json()["output"])
        emails = first_server.session_id_to_tool_env["wa-1"]["containers"]["email"]._emails
        victim_id = emails["email_id"].iloc[0]
        first.post("/ng-rollout/wa-1/email_delete_email", json={"email_id": victim_id})
        first.post(
            "/ng-rollout/wa-1/calendar_create_event",
            json={
                "event_name": "Resume sync",
                "participant_email": "carlos.gomez@atlas.com",
                "event_start": "2023-12-01 10:00:00",
                "duration": "30",
            },
        )
        assert first.post("/ng-rollout/wa-1/ng-session/export", json={"boundary_index": 3}).json()["exported"] is True
        expected_frames = _frames_as_json(first_server, "wa-1")

        # 'Restart': fresh instance, empty memory, same store.
        second, second_server = _make_client(tmp_path)
        response = second.post("/ng-rollout/wa-1/email_search_emails", json={"query": "Resume test"})
        assert response.status_code == 400  # session not initialized

        assert (
            second.post("/ng-rollout/wa-1/ng-session/restore", json={"boundary_index": 3}).json()["restored"] is True
        )

        # The restored environment answers through the real tool functions.
        response = second.post("/ng-rollout/wa-1/email_search_emails", json={"query": "Resume test"})
        assert response.json()["output"]["emails"][0]["subject"] == "Resume test"
        response = second.post(
            "/ng-rollout/wa-1/email_get_email_information_by_id", json={"email_id": victim_id, "field": "subject"}
        )
        assert response.json()["output"] == "Email not found."
        response = second.post("/ng-rollout/wa-1/calendar_search_events", json={"query": "Resume sync"})
        assert response.json()["output"]["events"][0]["event_name"] == "Resume sync"

        # Frame-for-frame fidelity across every container and attribute.
        # Compare in JSON form: cell-level dict equality would fail on
        # NaN != NaN for missing values (float NaN even under dtype=str).
        assert _frames_as_json(second_server, "wa-1") == expected_frames

    def test_restored_env_stays_mutable(self, tmp_path) -> None:
        # Functions must be rebound to the restored containers, not stale ones.
        first, _ = _make_client(tmp_path)
        first.post("/ng-rollout/wa-2/seed_session", json={})
        first.post("/ng-rollout/wa-2/ng-session/export", json={"boundary_index": 1})

        second, second_server = _make_client(tmp_path)
        second.post("/ng-rollout/wa-2/ng-session/restore", json={"boundary_index": 1})
        second.post(
            "/ng-rollout/wa-2/email_send_email",
            json={"recipient": "a.b@c.com", "subject": "post-restore", "body": "x"},
        )
        emails = second_server.session_id_to_tool_env["wa-2"]["containers"]["email"]._emails
        assert (emails["subject"] == "post-restore").any()

    def test_export_unknown_session_returns_nothing(self, tmp_path) -> None:
        client, _ = _make_client(tmp_path)
        body = client.post("/ng-rollout/wa-3/ng-session/export", json={"boundary_index": 1}).json()
        assert body["supported"] is True and body["exported"] is False
