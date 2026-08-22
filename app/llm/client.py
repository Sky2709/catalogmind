"""Shared Anthropic-via-Bedrock client, cached per process - same lifecycle idiom
as `app/redis_client.py`. Constructing the client is cheap (holds credentials/config,
no handshake), so a plain `lru_cache` singleton is enough.

Uses `anthropic[bedrock]`'s `AsyncAnthropicBedrockMantle` - confirmed against the
Bedrock user guide the user supplied (`bedrock-ug.pdf`), not guessed: this is the
Messages-API-compatible Bedrock client (as opposed to the lower-level boto3 Invoke/
Converse APIs), which is why the rest of `app/llm/` can use the same
`thinking`/`output_config`/`tools`/`cache_control` shapes the original, pre-Gemini
plan was built around - Bedrock exposes the same Messages API surface, just over a
different transport and auth mechanism (a Bedrock long-term API key via
`AWS_BEARER_TOKEN_BEDROCK`, not `ANTHROPIC_API_KEY`).

`ANTHROPIC_TRANSIENT_ERRORS` mirrors `WeaviateHybridRetriever`'s and the old Gemini
client's reasoning exactly: only genuinely transient failures (5xx-equivalent,
connection/timeout) are retried. `RateLimitError` (429) is deliberately excluded for
the same reason a Gemini 429 was excluded - blind exponential backoff on a rate
limit can compound the problem, and `with_retry` only retries by exception *type*,
not by reading `retry-after`.

**Two-layer timeout, learned the hard way on the Gemini build, applied here from the
start rather than rediscovered**: a live Gemini call once hung for over two minutes
with `200 OK` already received and zero further progress, and the SDK's own
documented timeout setting didn't reliably bound a stall *between* streamed chunks -
only a manually-enforced per-chunk `asyncio.wait_for` (in `app/llm/graph.py`) turned
out to be trustworthy. The constructor `timeout=` below is kept as a first,
best-effort layer (real, not a guess: the SDK is generally well-regarded for
honouring it), but `app/llm/graph.py`'s per-chunk wrapper is the layer this codebase
actually relies on - not re-verified against a live Bedrock stall yet, so treat that
belief as inherited caution, not a confirmed fact, until it's actually been tested
against a real hang here too.
"""

from __future__ import annotations

from functools import lru_cache

import anthropic

from app.config import get_settings

ANTHROPIC_TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (
    anthropic.APIConnectionError,  # covers APITimeoutError too (its subclass)
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    anthropic.ServiceUnavailableError,
    TimeoutError,  # asyncio.wait_for's, see app/llm/graph.py
)

# Generous, not tight: `scripts/bench_chat.py` measured real cold-turn latency up to
# p99=28.1s for a *whole* turn against Gemini (embed + cache-check + one LLM call);
# kept as the starting point here since Bedrock's own latency hasn't been separately
# benchmarked yet. This bounds a single call, not the whole multi-round agent loop.
BEDROCK_CALL_TIMEOUT_SECONDS = 60.0


@lru_cache(maxsize=1)
def get_bedrock_client() -> anthropic.AsyncAnthropicBedrockMantle:
    settings = get_settings()
    return anthropic.AsyncAnthropicBedrockMantle(
        api_key=settings.bedrock_api_key,
        aws_region=settings.aws_region,
        timeout=BEDROCK_CALL_TIMEOUT_SECONDS,
    )


async def dispose_bedrock_client() -> None:
    if get_bedrock_client.cache_info().currsize:
        await get_bedrock_client().close()
        get_bedrock_client.cache_clear()
