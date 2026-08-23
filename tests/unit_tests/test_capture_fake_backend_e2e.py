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
"""End-to-end capture validation against a fake generation backend, no GPU.

A deterministic OpenAI-compatible backend serves real HTTP. The real
``vllm_model`` server sits in front of it, so these tests exercise the code
paths the in-process fakes bypass: request preprocessing, the outbound hop,
token-bundle extraction, the dialect converters, and capture — through to
terminal attribution and delivery.

The fake tokenizer is consistent: an assistant message re-entering a later
prompt tokenizes to exactly the generation ids it was served with, so
multi-turn conversations produce genuine token-prefix chains.
"""

import asyncio
import threading
import zlib
from unittest.mock import MagicMock

import pytest
import uvicorn
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture import TOKEN_FIELDS, TokenCaptureStore
from nemo_gym.token_id_capture.delivery import MASK_SAMPLE_KEY, TOKEN_CAPTURE_KEY, finalize_rollout_token_capture
from nemo_gym.token_id_capture.fingerprint import assistant_fingerprint
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


# --- deterministic fake generation backend ------------------------------------


def _tok(text: str) -> list[int]:
    return [zlib.crc32(word.encode()) % 50000 for word in str(text).split()]


def tokenize_message(message: dict) -> list[int]:
    """One stable token per role plus one per whitespace word of content."""
    content = message.get("content")
    if isinstance(content, list):
        content = " ".join(str(block.get("text", "")) for block in content if isinstance(block, dict))
    return [zlib.crc32(str(message.get("role", "")).encode()) % 50000] + _tok(content or "")


REPLIES = ["I will inspect the task first", "The final answer is 42", "A Concise Title"]


def build_fake_backend() -> FastAPI:
    app = FastAPI()
    app.state.calls = 0

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request) -> dict:
        body = await request.json()
        messages = body.get("messages") or []
        prompt_ids: list[int] = []
        for message in messages:
            prompt_ids.extend(tokenize_message(message))
        reply = REPLIES[app.state.calls % len(REPLIES)]
        app.state.calls += 1
        generation_ids = tokenize_message({"role": "assistant", "content": reply})
        return {
            "id": f"chatcmpl-fake{app.state.calls}",
            "object": "chat.completion",
            "created": 1,
            "model": body.get("model") or "fake-model",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": reply,
                        "prompt_token_ids": prompt_ids,
                        "generation_token_ids": generation_ids,
                        "generation_log_probs": [-0.1] * len(generation_ids),
                    },
                }
            ],
            "usage": {
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(generation_ids),
                "total_tokens": len(prompt_ids) + len(generation_ids),
            },
        }

    return app


@pytest.fixture(autouse=True)
def _seeded_global_config(monkeypatch):
    # The outbound hop's global aiohttp client lazily loads the global config,
    # which parses argv through Hydra on first touch. The CLI seeds it in
    # production; seed it here so pytest's argv never reaches Hydra.
    from omegaconf import DictConfig

    import nemo_gym.global_config as global_config_module
    import nemo_gym.server_utils as server_utils_module

    monkeypatch.setattr(global_config_module, "_GLOBAL_CONFIG_DICT", DictConfig({}))
    # Each TestClient runs its own event loop; the global aiohttp session binds
    # to the loop that created it. Reset per test so a session from an earlier
    # test's closed loop is never reused.
    monkeypatch.setattr(server_utils_module, "_GLOBAL_AIOHTTP_CLIENT", None)


@pytest.fixture(scope="module")
def fake_backend_url():
    server = uvicorn.Server(uvicorn.Config(build_fake_backend(), host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    while not server.started:
        pass
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}/v1"
    server.should_exit = True
    thread.join(timeout=5)


def _gym_client(tmp_path, fake_backend_url) -> TestClient:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8099,
        entrypoint="",
        name="policy_model",
        base_url=fake_backend_url,
        api_key="dummy",
        model="fake-model",
        return_token_id_information=True,
        uses_reasoning_parser=False,
    )
    server = VLLMModel(
        config=config,
        server_client=MagicMock(
            spec=ServerClient,
            global_config_dict={"token_id_capture": {"enabled": True, "dir": str(tmp_path)}},
        ),
    )
    return TestClient(server.setup_webserver())


def _strip_tokens(item: dict) -> dict:
    return {key: value for key, value in item.items() if key not in TOKEN_FIELDS}


def _wire_fingerprint(payload: dict) -> str:
    """Fingerprint the assistant items exactly as the client received them."""
    output = payload.get("output")
    if isinstance(output, list):
        return assistant_fingerprint([item for item in output if isinstance(item, dict)])
    items = []
    for choice in payload.get("choices") or []:
        message = dict((choice or {}).get("message") or {})
        message.setdefault("role", "assistant")
        items.append(message)
    if not items and payload.get("role") == "assistant":
        items = [payload]
    return assistant_fingerprint(items)


# --- the tests -----------------------------------------------------------------


def test_multi_turn_chain_and_aux_call_end_to_end(tmp_path, fake_backend_url):
    rid = "e2e-0-0"
    prefix = f"/ng-rollout/{rid}/training-token-capture"

    # The context manager keeps one event loop for all requests in the test,
    # matching how a served process runs; the global aiohttp session binds to it.
    with _gym_client(tmp_path, fake_backend_url) as client:
        r1 = client.post(f"{prefix}/v1/responses", json={"input": "solve the task"})
        assert r1.status_code == 200, r1.text
        payload1 = r1.json()
        items1 = [_strip_tokens(item) for item in payload1["output"]]

        follow_up = (
            [{"role": "user", "content": "solve the task"}] + items1 + [{"role": "user", "content": "now finish it"}]
        )
        r2 = client.post(f"{prefix}/v1/responses", json={"input": follow_up})
        assert r2.status_code == 200, r2.text
        payload2 = r2.json()

        aux = client.post(f"{prefix}/v1/responses", json={"input": "write a short title"})
        assert aux.status_code == 200, aux.text

    entries = {entry.response_id: entry for entry in TokenCaptureStore(tmp_path).read_entries(rid)}
    assert len(entries) == 3
    entry1, entry2 = entries[payload1["id"]], entries[payload2["id"]]

    # The real converter round-trip produced a genuine token-prefix chain:
    # call 2's prompt extends call 1's cumulative sequence exactly.
    cumulative1 = list(entry1.prompt_token_ids) + list(entry1.generation_token_ids)
    assert entry2.prompt_token_ids[: len(cumulative1)] == cumulative1
    assert len(entry2.prompt_token_ids) > len(cumulative1)

    # The cross-dialect invariant: the wire payload's assistant items and the
    # captured normalized items hash identically.
    for payload in (payload1, payload2):
        assert _wire_fingerprint(payload) == assistant_fingerprint(list(entries[payload["id"]].output_items))

    # A synthesized result: merged token-free transcript on the final envelope.
    # (A response that still carries inline token arrays is the native path and
    # deliberately bypasses attribution — those arrays are the policy's data.)
    merged_output = [_strip_tokens(item) for item in payload1["output"] + payload2["output"]]
    result = {
        "_ng_rollout_id": rid,
        "response": {"id": payload2["id"], "model": "fake-model", "object": "response", "output": merged_output},
        "reward": 1.0,
    }
    built = asyncio.run(finalize_rollout_token_capture(result, TokenCaptureStore(tmp_path)))
    assert built[MASK_SAMPLE_KEY] is False
    attribution = result[TOKEN_CAPTURE_KEY]["terminal_attribution"]
    assert attribution["method"] == "response_id" and attribution["chain"] == "delivered"
    delivered = [item for item in result["response"]["output"] if item.get("generation_token_ids")]
    assert [item["generation_token_ids"] for item in delivered] == [
        list(entry1.generation_token_ids),
        list(entry2.generation_token_ids),
    ]


def test_chat_route_end_to_end_records_backend_envelope_id(tmp_path, fake_backend_url):
    client = _gym_client(tmp_path, fake_backend_url)
    rid = "e2e-1-0"
    resp = client.post(
        f"/ng-rollout/{rid}/training-token-capture/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "solve the task"}]},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    [entry] = TokenCaptureStore(tmp_path).read_entries(rid)
    # The chat dialect passes the backend's envelope id through.
    assert entry.response_id == payload["id"] and payload["id"].startswith("chatcmpl-fake")
    assert _wire_fingerprint(payload) == assistant_fingerprint(list(entry.output_items))


def test_messages_route_end_to_end_serves_the_recorded_id(tmp_path, fake_backend_url):
    client = _gym_client(tmp_path, fake_backend_url)
    rid = "e2e-2-0"
    resp = client.post(
        f"/ng-rollout/{rid}/training-token-capture/v1/messages",
        json={"model": "claude-x", "max_tokens": 32, "messages": [{"role": "user", "content": "hello there"}]},
    )
    assert resp.status_code == 200, resp.text
    payload = resp.json()
    [entry] = TokenCaptureStore(tmp_path).read_entries(rid)
    # The Anthropic envelope reuses the id capture recorded.
    assert entry.response_id == payload["id"]
    # Anthropic content blocks and the captured normalized items hash identically.
    anthropic_turn = {"role": "assistant", "content": payload["content"]}
    assert assistant_fingerprint([anthropic_turn]) == assistant_fingerprint(list(entry.output_items))
