# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
import random
from unittest.mock import MagicMock

import pytest
from app import (
    SyntheticModel,
    SyntheticModelConfig,
    _build_response_dict,
    _build_text,
    _gen_log_probs,
    _gen_token_ids,
)
from fastapi.responses import JSONResponse

from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


def _make_server(**overrides) -> SyntheticModel:
    cfg = SyntheticModelConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="synthetic_model",
        **overrides,
    )
    return SyntheticModel(config=cfg, server_client=MagicMock(spec=ServerClient))


class TestPayloadBuilders:
    def test_text_length_matches_tokens(self) -> None:
        assert len(_build_text(n_tokens=128, chars_per_token=4)) == 512

    def test_token_ids_in_range(self) -> None:
        ids = _gen_token_ids(random.Random(0), n=100, vocab_size=200_000)
        assert len(ids) == 100
        assert all(0 <= i < 200_000 for i in ids)

    def test_log_probs_negative_or_zero(self) -> None:
        lps = _gen_log_probs(random.Random(0), n=50)
        assert len(lps) == 50
        assert all(-12.0 <= lp <= 0.0 for lp in lps)


class TestBuildResponseDict:
    def test_includes_message_and_function_call(self) -> None:
        cfg = SyntheticModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="synthetic_model",
            output_tokens=64,
        )
        request_body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hi"}])
        d = _build_response_dict(random.Random(0), cfg, request_body)
        types = [item["type"] for item in d["output"]]
        assert "message" in types
        assert "function_call" in types

    def test_token_ids_present_when_training_enabled(self) -> None:
        cfg = SyntheticModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="synthetic_model",
            output_tokens=32,
            prompt_tokens=16,
            include_token_ids_and_log_probs=True,
        )
        d = _build_response_dict(random.Random(0), cfg, NeMoGymResponseCreateParamsNonStreaming(input=[]))
        msg = next(item for item in d["output"] if item["type"] == "message")
        assert len(msg["prompt_token_ids"]) == 16
        assert len(msg["generation_token_ids"]) == 32
        assert len(msg["generation_log_probs"]) == 32

    def test_token_ids_absent_when_training_disabled(self) -> None:
        cfg = SyntheticModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="synthetic_model",
            output_tokens=32,
            include_token_ids_and_log_probs=False,
        )
        d = _build_response_dict(random.Random(0), cfg, NeMoGymResponseCreateParamsNonStreaming(input=[]))
        msg = next(item for item in d["output"] if item["type"] == "message")
        assert "generation_token_ids" not in msg

    def test_reasoning_items_prepend(self) -> None:
        cfg = SyntheticModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="synthetic_model",
            output_tokens=8,
            n_reasoning_items=3,
            reasoning_tokens_per_item=16,
        )
        d = _build_response_dict(random.Random(0), cfg, NeMoGymResponseCreateParamsNonStreaming(input=[]))
        types = [item["type"] for item in d["output"]]
        assert types[:3] == ["reasoning", "reasoning", "reasoning"]
        assert types[-2:] == ["message", "function_call"]

    def test_function_call_uses_configured_tool_name(self) -> None:
        cfg = SyntheticModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="synthetic_model",
            tool_name="my_custom_tool",
        )
        d = _build_response_dict(random.Random(0), cfg, NeMoGymResponseCreateParamsNonStreaming(input=[]))
        fn = next(item for item in d["output"] if item["type"] == "function_call")
        assert fn["name"] == "my_custom_tool"

    def test_usage_block_reflects_token_counts(self) -> None:
        cfg = SyntheticModelConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="synthetic_model",
            prompt_tokens=100,
            output_tokens=200,
            n_reasoning_items=2,
            reasoning_tokens_per_item=50,
        )
        d = _build_response_dict(random.Random(0), cfg, NeMoGymResponseCreateParamsNonStreaming(input=[]))
        u = d["usage"]
        assert u["input_tokens"] == 100
        # output_tokens + reasoning tokens (2*50)
        assert u["output_tokens"] == 200 + 100
        assert u["total_tokens"] == 100 + 200 + 100
        assert u["output_tokens_details"]["reasoning_tokens"] == 100


class TestResponses:
    @pytest.mark.asyncio
    async def test_responses_returns_jsonresponse(self) -> None:
        server = _make_server(async_latency_ms=0, output_tokens=32)
        request_body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hi"}])
        result = await server.responses(MagicMock(), request_body)
        assert isinstance(result, JSONResponse)
        body = json.loads(result.body)
        assert body["object"] == "response"
        types = [item["type"] for item in body["output"]]
        assert "function_call" in types


class TestChatCompletions:
    @pytest.mark.asyncio
    async def test_chat_completions_raises(self) -> None:
        server = _make_server()
        with pytest.raises(NotImplementedError):
            await server.chat_completions(MagicMock())
