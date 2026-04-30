# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Synthetic model server for scale-sim.

Returns a `NeMoGymResponse` shaped like a real RL-training output:

- One `NeMoGymResponseOutputMessage` whose `text` is sized as
  ``output_tokens * chars_per_token``
- ``prompt_token_ids: List[int]`` of length ``prompt_tokens`` (≈6 bytes each in JSON)
- ``generation_token_ids: List[int]`` of length ``output_tokens``
- ``generation_log_probs: List[float]`` of length ``output_tokens`` (≈10 bytes each in JSON)
- Optionally ``n_reasoning_items`` × ``NeMoGymResponseReasoningItem`` of
  ``reasoning_tokens_per_item`` tokens each
- One ``NeMoGymResponseFunctionToolCall`` to drive the agent loop

This matches what production vLLM / OpenAI adapters emit, so the bytes on the wire,
the orjson parse cost, and the per-rollout memory residency in the agent are all
representative — not "one big string field".

We bypass Pydantic on the way out (return JSONResponse with a hand-built dict) so
the wire format includes the training-mode token-id / log-prob fields cleanly,
without fighting Pydantic discriminated-union dispatch on the base classes.

Knob breakdown for back-of-envelope sizing:
    body_bytes ≈
        output_tokens * chars_per_token          # output_text content
      + (prompt_tokens + output_tokens) * 6       # token IDs (JSON int + comma)
      + output_tokens * 10                        # log probs (JSON float + comma)
      + n_reasoning_items * reasoning_tokens_per_item * (chars_per_token + 16)
      + ~1 KB metadata
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any, Dict, List, Literal
from uuid import uuid4

from fastapi import Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from nemo_gym.base_responses_api_model import (
    BaseResponsesAPIModelConfig,
    Body,
    SimpleResponsesAPIModel,
)
from nemo_gym.openai_utils import (
    NeMoGymChatCompletion,
    NeMoGymChatCompletionCreateParamsNonStreaming,
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
)


LatencyDist = Literal["fixed", "lognormal"]


class SyntheticModelConfig(BaseResponsesAPIModelConfig):
    # ---- Latency ----
    async_latency_ms: float = Field(
        default=50.0,
        description="`await asyncio.sleep(...)` — simulates token generation time.",
    )
    cpu_burn_ms: float = Field(
        default=0.0,
        description="`time.perf_counter()` busy-loop — holds the event loop.",
    )
    latency_dist: LatencyDist = Field(default="fixed")
    latency_lognormal_mu: float = 0.0
    latency_lognormal_sigma: float = 0.0

    # ---- Sequence-length-driven payload knobs ----
    prompt_tokens: int = Field(
        default=512,
        description="Length of `prompt_token_ids` list. Drives JSON cost on the input-side payload echo.",
    )
    output_tokens: int = Field(
        default=256,
        description=(
            "Length of `generation_token_ids` and `generation_log_probs`. "
            "Also drives output `text` length via `chars_per_token`."
        ),
    )
    chars_per_token: int = Field(
        default=4,
        description="Average chars per token in the output `text` field. ~4 is typical for English BPE.",
    )
    include_token_ids_and_log_probs: bool = Field(
        default=True,
        description=(
            "If true (RL training default), include `prompt_token_ids`, `generation_token_ids`, "
            "and `generation_log_probs` on the output message. Adds ~16 bytes/token to body size."
        ),
    )

    # ---- Reasoning content ----
    n_reasoning_items: int = Field(
        default=0,
        description="Number of `<think>`-style reasoning items prepended to the output.",
    )
    reasoning_tokens_per_item: int = Field(
        default=0,
        description="Tokens per reasoning item. Each reasoning item gets its own token_ids/log_probs.",
    )

    # ---- Tool calling ----
    tool_name: str = Field(
        default="synthetic_tool",
        description="Function name emitted in the function_call. Must match a route on the resources server.",
    )

    # ---- Failure injection ----
    inject_500_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    seed: int = Field(default=0)

    # ---- Token-ID range ----
    vocab_size: int = Field(
        default=200_000,
        description="Upper bound for randomly-generated token IDs. ~200K matches a realistic SentencePiece vocab.",
    )


def _sample_latency_seconds(cfg: SyntheticModelConfig, rng: random.Random) -> float:
    if cfg.latency_dist == "fixed":
        return cfg.async_latency_ms / 1000.0
    return rng.lognormvariate(cfg.latency_lognormal_mu, cfg.latency_lognormal_sigma) / 1000.0


def _cpu_burn(milliseconds: float) -> None:
    if milliseconds <= 0:
        return
    deadline = time.perf_counter() + (milliseconds / 1000.0)
    while time.perf_counter() < deadline:
        pass


# Per-request random generation of huge lists is prohibitively slow above ~10K
# items (Python comprehension on rng.randrange / rng.uniform). For long sequences
# we use a deterministic fast path that produces lists with the same JSON byte
# distribution as random data (same digit-length distribution for ints, same
# float-formatting length for log probs). The bytes on the wire and the orjson
# parse cost are what we're measuring; the actual ID values don't matter.
_FAST_PATH_THRESHOLD = 10_000


def _gen_token_ids(rng: random.Random, n: int, vocab_size: int) -> List[int]:
    if n <= _FAST_PATH_THRESHOLD:
        return [rng.randrange(vocab_size) for _ in range(n)]
    # Multiplier 7919 (a prime) varies digit lengths across the list. % vocab_size
    # caps the range. Identical JSON-encoded byte count to random for any reasonable
    # vocab_size (every ID fits in 1-6 digits).
    return [(i * 7919) % vocab_size for i in range(n)]


def _gen_log_probs(rng: random.Random, n: int) -> List[float]:
    if n <= _FAST_PATH_THRESHOLD:
        return [rng.uniform(-12.0, 0.0) for _ in range(n)]
    # Deterministic varied negative floats. JSON encodes each as ~10 chars,
    # matching the random case.
    return [-(i % 9999) / 1000.0 - 0.001 for i in range(n)]


def _build_text(n_tokens: int, chars_per_token: int) -> str:
    return "x" * (n_tokens * chars_per_token)


def _build_reasoning_item(
    rng: random.Random,
    n_tokens: int,
    cfg: SyntheticModelConfig,
) -> Dict[str, Any]:
    """Returns the raw dict shape of a NeMoGymResponseReasoningItem (+ training fields if enabled)."""
    item: Dict[str, Any] = {
        "id": f"rzn_{uuid4().hex[:12]}",
        "summary": [{"text": _build_text(n_tokens, cfg.chars_per_token), "type": "summary_text"}],
        "type": "reasoning",
        "encrypted_content": None,
    }
    if cfg.include_token_ids_and_log_probs:
        item["prompt_token_ids"] = []
        item["generation_token_ids"] = _gen_token_ids(rng, n_tokens, cfg.vocab_size)
        item["generation_log_probs"] = _gen_log_probs(rng, n_tokens)
    return item


def _build_output_message(rng: random.Random, cfg: SyntheticModelConfig) -> Dict[str, Any]:
    msg: Dict[str, Any] = {
        "id": f"msg_{uuid4().hex[:12]}",
        "content": [
            {
                "annotations": [],
                "text": _build_text(cfg.output_tokens, cfg.chars_per_token),
                "type": "output_text",
            }
        ],
        "role": "assistant",
        "status": "completed",
        "type": "message",
    }
    if cfg.include_token_ids_and_log_probs:
        msg["prompt_token_ids"] = _gen_token_ids(rng, cfg.prompt_tokens, cfg.vocab_size)
        msg["generation_token_ids"] = _gen_token_ids(rng, cfg.output_tokens, cfg.vocab_size)
        msg["generation_log_probs"] = _gen_log_probs(rng, cfg.output_tokens)
    return msg


def _build_function_call(cfg: SyntheticModelConfig) -> Dict[str, Any]:
    fn_call: Dict[str, Any] = {
        "arguments": "{}",
        "call_id": f"call_{uuid4().hex[:16]}",
        "name": cfg.tool_name,
        "type": "function_call",
        "status": "completed",
    }
    if cfg.include_token_ids_and_log_probs:
        # Tool calls in production typically have a small number of generation tokens
        # for the function name + args. Keep it minimal (~32 tokens worth) so this isn't
        # the dominant cost — that's `output_tokens` and `reasoning_tokens_per_item`.
        fn_call["prompt_token_ids"] = []
        fn_call["generation_token_ids"] = list(range(32))
        fn_call["generation_log_probs"] = [-0.5] * 32
    return fn_call


def _build_response_dict(
    rng: random.Random,
    cfg: SyntheticModelConfig,
    request_body: NeMoGymResponseCreateParamsNonStreaming,
) -> Dict[str, Any]:
    output: List[Dict[str, Any]] = []

    # Reasoning items first (matches production order: think → answer → tool call).
    for _ in range(cfg.n_reasoning_items):
        output.append(_build_reasoning_item(rng, cfg.reasoning_tokens_per_item, cfg))

    output.append(_build_output_message(rng, cfg))
    output.append(_build_function_call(cfg))

    total_reasoning_tokens = cfg.n_reasoning_items * cfg.reasoning_tokens_per_item
    return {
        "id": f"resp_{uuid4().hex[:16]}",
        "created_at": time.time(),
        "model": request_body.model or "synthetic-model",
        "object": "response",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": request_body.tools or [],
        "usage": {
            "input_tokens": cfg.prompt_tokens,
            "output_tokens": cfg.output_tokens + total_reasoning_tokens,
            "total_tokens": cfg.prompt_tokens + cfg.output_tokens + total_reasoning_tokens,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": total_reasoning_tokens},
        },
        "status": "completed",
    }


class SyntheticModel(SimpleResponsesAPIModel):
    config: SyntheticModelConfig

    def model_post_init(self, context: Any) -> None:
        self._rng = random.Random(self.config.seed)
        return super().model_post_init(context)

    async def responses(
        self, request: Request, body: NeMoGymResponseCreateParamsNonStreaming = Body()
    ) -> NeMoGymResponse:
        if self.config.inject_500_rate > 0 and self._rng.random() < self.config.inject_500_rate:
            return JSONResponse(content={"error": "synthetic_500"}, status_code=500)
        latency_s = _sample_latency_seconds(self.config, self._rng)
        if latency_s > 0:
            await asyncio.sleep(latency_s)
        _cpu_burn(self.config.cpu_burn_ms)
        # Build response as a dict and return as JSONResponse to ship the training-mode
        # token-id / log-prob fields without fighting Pydantic Union dispatch.
        return JSONResponse(content=_build_response_dict(self._rng, self.config, body))

    async def chat_completions(
        self, body: NeMoGymChatCompletionCreateParamsNonStreaming = Body()
    ) -> NeMoGymChatCompletion:
        # Not used by simple_agent; provided for API-completeness only.
        raise NotImplementedError(
            "synthetic_model does not implement /v1/chat/completions. "
            "Use simple_agent (which uses /v1/responses) for the scale-sim harness."
        )


if __name__ == "__main__":
    SyntheticModel.run_webserver()
