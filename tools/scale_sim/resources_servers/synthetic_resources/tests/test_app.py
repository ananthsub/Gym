# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from unittest.mock import MagicMock

import pytest
from fastapi.responses import JSONResponse

from app import (
    SyntheticEndpointConfig,
    SyntheticResourcesServer,
    SyntheticResourcesServerConfig,
    _build_padding_body,
    _cpu_burn,
)
from nemo_gym.base_resources_server import BaseVerifyRequest
from nemo_gym.openai_utils import NeMoGymResponse
from nemo_gym.server_utils import ServerClient


def _make_server(**overrides) -> SyntheticResourcesServer:
    cfg = SyntheticResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="synthetic_resources",
        **overrides,
    )
    return SyntheticResourcesServer(config=cfg, server_client=MagicMock(spec=ServerClient))


class TestPaddingBody:
    def test_flat_padding_size(self) -> None:
        body = _build_padding_body(target_bytes=1024, shape="flat_padding", chunk_bytes=256)
        assert isinstance(body["data"], str)
        assert len(body["data"]) == 1024

    def test_realistic_messages_shape(self) -> None:
        body = _build_padding_body(target_bytes=4096, shape="realistic_messages", chunk_bytes=256)
        assert "messages" in body
        # Approx target / chunk_bytes chunks, give or take.
        assert 1 <= len(body["messages"]) <= 4096
        for chunk in body["messages"]:
            assert chunk["role"] == "tool"
            assert isinstance(chunk["content"], str)

    def test_zero_size(self) -> None:
        body = _build_padding_body(target_bytes=0, shape="flat_padding", chunk_bytes=256)
        assert body == {"data": ""}


class TestCpuBurn:
    def test_burn_takes_at_least_target_time(self) -> None:
        import time

        target_ms = 50
        start = time.perf_counter()
        _cpu_burn(target_ms)
        elapsed_ms = (time.perf_counter() - start) * 1000
        # Allow some slack on slow CI but make sure we actually burned.
        assert elapsed_ms >= target_ms * 0.8

    def test_zero_burn_returns_immediately(self) -> None:
        _cpu_burn(0)


class TestSyntheticTool:
    @pytest.mark.asyncio
    async def test_returns_json_response(self) -> None:
        server = _make_server(tool=SyntheticEndpointConfig(body_size_bytes=128))
        from app import SyntheticToolRequest

        from fastapi import Request as FastAPIRequest  # noqa: F401  (typing only)

        result = await server.synthetic_tool(MagicMock(), SyntheticToolRequest())
        assert isinstance(result, JSONResponse)
        body = json.loads(result.body)
        assert "data" in body
        assert len(body["data"]) == 128


class TestVerify:
    @pytest.mark.asyncio
    async def test_verify_returns_reward(self) -> None:
        server = _make_server(verify=SyntheticEndpointConfig(body_size_bytes=64))
        request = BaseVerifyRequest(
            responses_create_params={"input": [{"role": "user", "content": "hi"}]},
            response=NeMoGymResponse(
                id="r",
                created_at=0.0,
                model="dummy",
                object="response",
                output=[],
                parallel_tool_calls=True,
                tool_choice="auto",
                tools=[],
            ),
        )
        result = await server.verify(request)
        # When body_size is set we return a JSONResponse for size control.
        assert isinstance(result, JSONResponse)
        body = json.loads(result.body)
        assert body["reward"] == 1.0
        assert "synthetic_padding" in body


class TestFailureInjection:
    @pytest.mark.asyncio
    async def test_inject_500_rate_one_returns_500(self) -> None:
        server = _make_server(inject_500_rate=1.0)
        from app import SyntheticToolRequest

        result = await server.synthetic_tool(MagicMock(), SyntheticToolRequest())
        assert result.status_code == 500
