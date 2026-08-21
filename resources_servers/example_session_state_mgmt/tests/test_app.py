# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from unittest.mock import MagicMock

from fastapi.testclient import TestClient
from httpx import Cookies

from nemo_gym.server_utils import ServerClient
from resources_servers.example_session_state_mgmt.app import (
    StatefulCounterResourcesServer,
    StatefulCounterResourcesServerConfig,
)


class TestApp:
    def test_sanity(self) -> None:
        config = StatefulCounterResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="",
        )
        server = StatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

        app = server.setup_webserver()
        client = TestClient(app)

        class StatelessCookies(Cookies):
            def extract_cookies(self, response):
                pass

        client._cookies = StatelessCookies(client._cookies)

        # Check that we are at 0
        response = client.post("/get_counter_value")
        initial_request_cookies = response.cookies
        assert response.json() == {"count": 0}
        response = client.post("/increment_counter", json={"count": 2}, cookies=initial_request_cookies)
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", cookies=initial_request_cookies)
        assert response.json() == {"count": 2}

        # Start a new session i.e. don't pass cookies
        response = client.post("/increment_counter", json={"count": 4})
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", cookies=response.cookies)
        assert response.json() == {"count": 4}
        response = client.post("/increment_counter", json={"count": 3}, cookies=response.cookies)
        assert response.json() == {"success": True}
        response = client.post("/get_counter_value", cookies=response.cookies)
        assert response.json() == {"count": 7}

        response = client.post("/get_counter_value", cookies=initial_request_cookies)
        assert response.json() == {"count": 2}

    def _make_server(self, session_state_dir=None) -> StatefulCounterResourcesServer:
        config = StatefulCounterResourcesServerConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="counter_server",
            session_state_dir=str(session_state_dir) if session_state_dir else None,
        )
        return StatefulCounterResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))

    def test_rollout_prefix_binds_session_without_cookies(self, tmp_path) -> None:
        # Requests under /ng-rollout/<id>/ derive the session id from the
        # rollout id, so cookie-less requests from different clients (or
        # workers, or a restarted process) share one session.
        client = TestClient(self._make_server(tmp_path).setup_webserver())
        response = client.post("/ng-rollout/7-0/seed_session", json={"initial_count": 5})
        assert response.status_code == 200
        client.cookies.clear()
        response = client.post("/ng-rollout/7-0/increment_counter", json={"count": 2})
        assert response.json() == {"success": True}
        client.cookies.clear()
        response = client.post("/ng-rollout/7-0/get_counter_value")
        assert response.json() == {"count": 7}

    def test_export_restore_across_server_restart(self, tmp_path) -> None:
        # Live server: seed, mutate, export at boundary 1.
        first = TestClient(self._make_server(tmp_path).setup_webserver())
        first.post("/ng-rollout/7-0/seed_session", json={"initial_count": 5})
        first.post("/ng-rollout/7-0/increment_counter", json={"count": 2})
        response = first.post("/ng-rollout/7-0/ng-session/export", json={"boundary_index": 1})
        assert response.json() == {
            "supported": True,
            "exported": True,
            "boundary_index": 1,
            "inline": False,
            "state": None,
            "detail": "",
        }

        # "Restarted" server: a fresh instance with empty memory over the same store.
        second = TestClient(self._make_server(tmp_path).setup_webserver())
        response = second.post("/ng-rollout/7-0/get_counter_value")
        assert response.json() == {"count": 0}

        response = second.post("/ng-rollout/7-0/ng-session/restore", json={"boundary_index": 1})
        assert response.json() == {"supported": True, "restored": True, "detail": ""}
        response = second.post("/ng-rollout/7-0/get_counter_value")
        assert response.json() == {"count": 7}
        response = second.post(
            "/ng-rollout/7-0/verify",
            json={"responses_create_params": {"input": []}, "response": _EMPTY_RESPONSE, "expected_count": 7},
        )
        assert response.json()["reward"] == 1.0

    def test_inline_export_skips_the_snapshot_file(self, tmp_path) -> None:
        # A caller that accepts inline state (inline_ok) gets it in the
        # response and no snapshot file is written — the Lustre-friendly path.
        client = TestClient(self._make_server(tmp_path).setup_webserver())
        client.post("/ng-rollout/7-1/seed_session", json={"initial_count": 5})
        response = client.post("/ng-rollout/7-1/ng-session/export", json={"boundary_index": 1, "inline_ok": True})
        body = response.json()
        assert body["exported"] is True and body["inline"] is True and body["state"] == {"counter": 5}
        assert not (tmp_path / "7-1").exists() or not list((tmp_path / "7-1").glob("*.snapshot.*"))

        # Restore from the inline state on a fresh server, no file involved.
        second = TestClient(self._make_server(tmp_path).setup_webserver())
        response = second.post(
            "/ng-rollout/7-1/ng-session/restore", json={"boundary_index": 1, "state": {"counter": 5}}
        )
        assert response.json()["restored"] is True
        assert second.post("/ng-rollout/7-1/get_counter_value").json() == {"count": 5}

    def test_restore_missing_snapshot_reports_not_restored(self, tmp_path) -> None:
        client = TestClient(self._make_server(tmp_path).setup_webserver())
        response = client.post("/ng-rollout/7-0/ng-session/restore", json={"boundary_index": 3})
        body = response.json()
        assert body["supported"] is True and body["restored"] is False

    def test_session_state_disabled_without_dir(self) -> None:
        client = TestClient(self._make_server(None).setup_webserver())
        response = client.post("/ng-rollout/7-0/ng-session/export", json={"boundary_index": 1})
        assert response.json()["supported"] is False

    def test_session_routes_require_rollout_prefix(self, tmp_path) -> None:
        client = TestClient(self._make_server(tmp_path).setup_webserver())
        response = client.post("/ng-session/export", json={"boundary_index": 1})
        assert response.status_code == 400


_EMPTY_RESPONSE = {
    "id": "resp_1",
    "created_at": 0,
    "model": "test",
    "object": "response",
    "output": [],
    "parallel_tool_calls": True,
    "tool_choice": "auto",
    "tools": [],
}
