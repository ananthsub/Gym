# synthetic_model

A controllable model server for scale testing. Returns a `NeMoGymResponse`
shaped like a real RL-training output — same fields, same byte distribution,
same orjson parse cost on the agent side.

Exposes:

- `/v1/responses` — emits one output message + (optional) reasoning items + one
  `function_call` to `synthetic_tool`. The function call drives the agent loop;
  combined with `simple_agent`'s `max_steps=n_hops` this gives a controllable
  number of model↔tool round-trips per `/run`.
- `/v1/chat/completions` — raises `NotImplementedError`. The harness uses
  `/v1/responses` exclusively (matching production via `simple_agent`).

## Sequence-length-driven payload

Body size is **derived** from token counts, not set as raw bytes. This matches
how real bodies grow:

| Knob | Meaning |
| --- | --- |
| `prompt_tokens` | Length of the `prompt_token_ids` list on the output message. |
| `output_tokens` | Length of `generation_token_ids` and `generation_log_probs`, and drives output `text` length via `chars_per_token`. |
| `chars_per_token` | Average chars per token in the `text` field. ~4 is typical for English BPE. |
| `include_token_ids_and_log_probs` | RL training default `true`. Adds ~16 bytes/token to the body. Disable to model a non-training serving path. |
| `n_reasoning_items` | Number of `<think>`-style reasoning items prepended to the output. |
| `reasoning_tokens_per_item` | Tokens per reasoning item. Each gets its own token IDs / log probs. |
| `vocab_size` | Upper bound for randomly-generated token IDs. Default 200K (SentencePiece-ish). |

Back-of-envelope body size:

```
body_bytes ≈
    output_tokens * chars_per_token              # output_text content
  + (prompt_tokens + output_tokens) * 6          # JSON int + comma per token id
  + output_tokens * 10                           # JSON float + comma per logprob
  + n_reasoning_items * reasoning_tokens_per_item * (chars_per_token + 16)
  + ~1 KB metadata
```

Example sizes (training mode, no reasoning):

| `prompt_tokens` | `output_tokens` | Body bytes (≈) | Notes |
| --- | --- | --- | --- |
| 512 | 256 | 10 KB | Smoke test scale |
| 2K | 2K | 70 KB | Short-rollout RL |
| 4K | 16K | 430 KB | Typical text rollout today |
| 4K | 64K | 1.7 MB | Long-thinking single-output |
| 4K | 128K | 3.4 MB | Large multimodal-scale body |
| 4K | 256K | 6.8 MB | |
| 4K | 512K | 13.6 MB | |
| 4K | 1M | 27 MB | Approximate next-model upper bound |

With reasoning (each reasoning item adds `reasoning_tokens_per_item × ~20 bytes`):

| Configuration | Body bytes |
| --- | --- |
| `output=16K, n_reasoning=1 × 32K` | ~1.1 MB |
| `output=16K, n_reasoning=1 × 128K` | ~4.5 MB |
| `output=64K, n_reasoning=1 × 1M` | ~28 MB |

## Memory back-of-envelope at high concurrency

The model server holds the response body in memory while the agent reads it.
At C concurrent requests of B bytes each, model-server peak is roughly C × B,
plus aiohttp send buffers, plus orjson scratch, plus Python overhead (~3×):

| `concurrency` | `output_tokens` | Per-rollout body | Model server peak (≈3×) |
| --- | --- | --- | --- |
| 8192 | 16K | 430 KB | 10 GB |
| 8192 | 128K | 3.4 MB | 80 GB |
| 8192 | 1M | 27 MB | **640 GB — won't fit on one host** |
| 1024 | 1M | 27 MB | 80 GB |
| 256 | 1M | 27 MB | 20 GB |
| 64 | 1M | 27 MB | 5 GB |

When sweeping output_tokens up into the 100K-1M range, **scale concurrency
down inversely** so the per-host peak stays bounded. Use `--mode zip` with
matched lists.

## Performance note: deterministic generation for `output_tokens > 10K`

Python list comprehension over `random.randrange` is ~1 μs per element. At
1M tokens that's ~1 s of CPU per request just to populate the list, which
saturates the model server's CPUs at trivial concurrency. For
`output_tokens > 10_000` (and likewise for `reasoning_tokens_per_item`) the
synthetic model uses a deterministic fast path that produces a list with
identical JSON byte distribution to random data — the bytes-on-wire and
orjson parse cost are unchanged, only the synthesis CPU is faster.

If you specifically need *random* IDs (e.g. to defeat orjson interning of
small ints), drop the threshold via `_FAST_PATH_THRESHOLD` in `app.py`.

## Latency knobs

| Knob | What it does |
| --- | --- |
| `async_latency_ms` | `await asyncio.sleep(...)` — yields the event loop. Models I/O-bound generation. |
| `cpu_burn_ms` | `time.perf_counter()` busy-loop — holds the event loop. Models CPU-bound serialization / log-prob extraction. |
| `latency_dist` | `fixed` or `lognormal`. |

## Failure injection

| Knob | What it does |
| --- | --- |
| `inject_500_rate` | P(handler returns HTTP 500). Tests retry layering. |

## Tool calling

| Knob | What it does |
| --- | --- |
| `tool_name` | Name emitted in the `function_call`. Must match a route on the paired resources server. |

## Implementation note: response is built as a dict, not via Pydantic

We construct the response as a plain dict and ship it via `JSONResponse`
rather than constructing a `NeMoGymResponse` object. This bypasses Pydantic's
discriminated-union dispatch — the gym's `NeMoGymResponseInputItem` Union
doesn't include the `*ForTraining` subclasses that carry token IDs / log probs,
so going through Pydantic on the way out would silently drop those fields.

The agent receiving the response does call `NeMoGymResponse.model_validate`,
which may also drop the training fields on input (a separate concern); but the
agent still pays the full orjson parse cost on the way in regardless of which
fields it retains. That's the cost we're measuring.
