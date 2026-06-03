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

import orjson
from fastapi import Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import Field

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
from nemo_gym.server_utils import is_nemo_gym_fastapi_entrypoint


LatencyDist = Literal["fixed", "lognormal", "pareto"]


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
    # ---- Pareto / heavy-tail straggler injection ----
    # When latency_dist="pareto", per-request latency is sampled from a Pareto
    # distribution with shape alpha (lower alpha = heavier tail). The minimum
    # (xm) defaults to async_latency_ms — so the no-straggler case still anchors
    # at the same floor as latency_dist="fixed". A hard cap (`pareto_max_ms`)
    # keeps individual cells bounded; uncapped Pareto can produce arbitrarily
    # large samples and dominate the cell's wall-clock budget.
    pareto_alpha: float = Field(
        default=1.5,
        description=(
            "Pareto shape parameter. Lower = heavier tail. alpha=1.5 → "
            "p99 ≈ 8x p50, p999 ≈ 17x p50 (LLM-generation-realistic). "
            "alpha=2.5 → p99 ≈ 3.2x. Only used when latency_dist='pareto'."
        ),
    )
    pareto_min_ms: float = Field(
        default=0.0,
        description=(
            "Pareto distribution minimum (xm). When 0, falls back to "
            "async_latency_ms so the floor matches the 'fixed' setting."
        ),
    )
    pareto_max_ms: float = Field(
        default=120_000.0,
        description="Hard cap on per-request latency. Prevents single samples from dominating cell wall-clock.",
    )

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

    # ---- Chunked / mid-stream-disconnect injection ----
    # When the response is returned as a single Content-Length-known JSON
    # blob (the default JSONResponse path), any TCP RST or server-side close
    # mid-response surfaces on the client as ``ConnectionResetError`` or
    # ``ServerDisconnectedError``. Production sees ``ClientPayloadError``
    # because real vLLM streams tokens via SSE / chunked-transfer encoding
    # and a mid-stream interruption hits aiohttp's payload-framing check
    # before its transport-level read. These knobs let us flip the response
    # path to chunked transfer and inject mid-response disconnects so the
    # harness can produce the exact ``ClientPayloadError`` class string
    # production sees.
    response_mode: Literal["json", "chunked"] = Field(
        default="json",
        description=(
            "'json' = single Content-Length JSONResponse (production-pre-vLLM "
            "shape). 'chunked' = Transfer-Encoding: chunked streaming response "
            "(production vLLM SSE shape; mid-stream disconnects surface as "
            "ClientPayloadError on the client)."
        ),
    )
    chunked_n_chunks: int = Field(
        default=16,
        ge=1,
        description=(
            "Only used when response_mode='chunked'. Number of chunks to split "
            "the response body into. Each chunk is preceded by an "
            "asyncio.sleep(async_latency_ms / chunked_n_chunks) so the total "
            "wall-clock matches the JSON response path."
        ),
    )
    inject_mid_stream_disconnect_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Only used when response_mode='chunked'. Probability per request "
            "of aborting after ~half the chunks have been written so the "
            "client sees a truncated chunked-stream and raises "
            "aiohttp.ClientPayloadError. The way the truncation is induced "
            "is controlled by `disconnect_mode`."
        ),
    )
    disconnect_mode: Literal["raw_truncation", "starlette_raise"] = Field(
        default="raw_truncation",
        description=(
            "How a disconnect is induced when `inject_mid_stream_disconnect_rate` "
            "fires. Two paths exist because they produce different aiohttp "
            "exception classes on the client and we want both reproducible.\n"
            "* 'raw_truncation' (default): the handler writes headers + the "
            "  first half of the body directly through the ASGI `send` "
            "  callable (bypassing Starlette's `StreamingResponse`), then "
            "  raises `asyncio.CancelledError` so uvicorn closes the socket "
            "  WITHOUT sending the terminating `0\\r\\n\\r\\n` chunked frame. "
            "  aiohttp on the client side sees a truncated chunked stream and "
            '  raises `ClientPayloadError("Response payload is not '
            '  completed")` — the exact production class.\n'
            "* 'starlette_raise': the legacy path. Raises a RuntimeError "
            "  inside a `StreamingResponse` generator after half the chunks. "
            "  Starlette's cleanup still emits the terminating chunk, so the "
            "  client receives a well-framed but content-truncated body and "
            "  the gym surfaces `orjson.JSONDecodeError` rather than "
            "  `ClientPayloadError`. Useful for documenting the gap; not "
            "  recommended as the default."
        ),
    )

    # ---- Token-ID range ----
    vocab_size: int = Field(
        default=200_000,
        description="Upper bound for randomly-generated token IDs. ~200K matches a realistic SentencePiece vocab.",
    )


def _sample_latency_seconds(cfg: SyntheticModelConfig, rng: random.Random) -> float:
    if cfg.latency_dist == "fixed":
        return cfg.async_latency_ms / 1000.0
    if cfg.latency_dist == "lognormal":
        return rng.lognormvariate(cfg.latency_lognormal_mu, cfg.latency_lognormal_sigma) / 1000.0
    if cfg.latency_dist == "pareto":
        # Inverse-CDF Pareto sample. Built-in random.paretovariate produces a
        # Pareto(alpha) sample with min=1; scale by xm to set the floor and
        # clamp by pareto_max_ms to bound cell wall-clock.
        xm_ms = cfg.pareto_min_ms if cfg.pareto_min_ms > 0 else cfg.async_latency_ms
        raw_ms = rng.paretovariate(cfg.pareto_alpha) * xm_ms
        return min(raw_ms, cfg.pareto_max_ms) / 1000.0
    raise ValueError(f"Unknown latency_dist={cfg.latency_dist!r}")


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

        if self.config.response_mode == "json":
            # Default path: single Content-Length JSONResponse. Mid-response close
            # surfaces as ConnectionResetError / ServerDisconnectedError on the
            # client (NOT ClientPayloadError).
            latency_s = _sample_latency_seconds(self.config, self._rng)
            if latency_s > 0:
                await asyncio.sleep(latency_s)
            _cpu_burn(self.config.cpu_burn_ms)
            return JSONResponse(content=_build_response_dict(self._rng, self.config, body))

        # Chunked path: Transfer-Encoding: chunked. Each chunk advances by an
        # equal slice of the latency budget. Optionally abort mid-stream to
        # force ClientPayloadError on the client.
        latency_s = _sample_latency_seconds(self.config, self._rng)
        _cpu_burn(self.config.cpu_burn_ms)
        body_bytes = orjson.dumps(_build_response_dict(self._rng, self.config, body))
        n_chunks = max(1, self.config.chunked_n_chunks)
        per_chunk_sleep_s = (latency_s / n_chunks) if n_chunks > 0 else 0.0
        chunk_size = max(1, (len(body_bytes) + n_chunks - 1) // n_chunks)
        cfg = self.config
        rng = self._rng
        # Decide BEFORE writing anything whether this request will be
        # disconnect-injected, so the seed-driven decision is reproducible.
        should_disconnect = (
            cfg.inject_mid_stream_disconnect_rate > 0 and rng.random() < cfg.inject_mid_stream_disconnect_rate
        )

        # raw_truncation path: bypass Starlette's StreamingResponse and write
        # headers + partial body directly through the ASGI `send` callable.
        # Then raise CancelledError (a BaseException, not Exception) so
        # neither FastAPI nor Starlette catches it and emits the terminating
        # chunked frame on cleanup. uvicorn observes the unfinished response
        # and closes the underlying socket. The client's aiohttp sees a
        # truncated chunked stream — no `0\r\n\r\n` terminator before EOF —
        # and raises `ClientPayloadError("Response payload is not
        # completed")`, which is the exact aiohttp exception production
        # reports against vLLM.
        if should_disconnect and cfg.disconnect_mode == "raw_truncation":
            send = request._send
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [
                        (b"content-type", b"application/json"),
                        (b"transfer-encoding", b"chunked"),
                    ],
                }
            )
            stop_at = max(1, n_chunks // 2)
            for chunk_idx in range(stop_at):
                if per_chunk_sleep_s > 0:
                    await asyncio.sleep(per_chunk_sleep_s)
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, len(body_bytes))
                await send(
                    {
                        "type": "http.response.body",
                        "body": body_bytes[start:end],
                        "more_body": True,
                    }
                )
            # No more_body=False ever sent. CancelledError propagates as
            # BaseException through FastAPI/Starlette to uvicorn, which closes
            # the socket without emitting the chunked terminator.
            raise asyncio.CancelledError("synthetic_raw_truncation")

        async def _stream():
            for chunk_idx in range(n_chunks):
                if per_chunk_sleep_s > 0:
                    await asyncio.sleep(per_chunk_sleep_s)
                # Legacy starlette_raise mode: raise from inside the generator
                # after half the chunks. Starlette's cleanup still emits the
                # terminator, so the client receives a well-framed but
                # content-truncated body and the gym surfaces JSONDecodeError
                # rather than ClientPayloadError. Kept so we can re-run the
                # legacy behavior for comparison; not the default.
                if should_disconnect and cfg.disconnect_mode == "starlette_raise" and chunk_idx >= n_chunks // 2:
                    raise RuntimeError("synthetic_starlette_raise")
                start = chunk_idx * chunk_size
                end = min(start + chunk_size, len(body_bytes))
                yield body_bytes[start:end]
                if end >= len(body_bytes):
                    return

        return StreamingResponse(
            _stream(),
            media_type="application/json",
            headers={"Transfer-Encoding": "chunked"},
        )

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
elif is_nemo_gym_fastapi_entrypoint(__file__):
    # uvicorn worker re-import path. When num_workers > 1, the parent process
    # spawns uvicorn with "module:app" so each forked worker imports this
    # module fresh and finds the top-level `app` to bind. Mirrors how
    # `responses_api_models/vllm_model/app.py` exposes its app.
    app = SyntheticModel.run_webserver()  # noqa: F401
