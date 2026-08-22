"""Fixed-window rate limiting, backed by Redis so it holds across every API process
sharing one Redis instance - unlike the in-memory ingestion-concurrency semaphore in
`app.ingestion.pipeline`, which is deliberately per-process and answers a different
question ("how much load can this process's own CPU/DB connections take" vs "how much
load is one caller allowed to generate at all").

A fixed window, not a sliding one or a token bucket: `INCR` + `EXPIRE NX` is one
counter per window, no Lua script, no separate cleanup job. Its one known imprecision
- a caller can burst up to ~2x the limit right at a window boundary (once near the end
of one window, once at the start of the next) - is acceptable for "stop a merchant
from hammering the ingest endpoint", which cares about sustained rate, not
millisecond-level fairness. A sliding-window or token-bucket limiter would remove that
imprecision at the cost of real complexity this use case doesn't need.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    count: int
    limit: int
    retry_after_seconds: int


async def check_rate_limit(
    redis: Any, key: str, *, limit: int, window_seconds: int
) -> RateLimitResult:
    """Increments `key`'s counter. The expiry is set only the first time a window's
    counter is created (`count == 1`) - a request that increments an *already
    existing* counter must never push its expiry back out, or a steady trickle of
    requests could keep one window open indefinitely and the limit would never
    actually reset.

    `redis` only ever needs `incr`/`expire`/`ttl` - typed `Any` rather than a
    structural Protocol because `redis.asyncio.Redis`'s own stubs type those as a
    sync/async union (they back both the sync and async clients), which doesn't
    satisfy a strict async Protocol no matter how it's phrased. The test suite's fake
    implements the same three methods without needing to satisfy that typing fight."""
    count = await redis.incr(key)
    if count == 1:
        await redis.expire(key, window_seconds)

    if count <= limit:
        return RateLimitResult(allowed=True, count=count, limit=limit, retry_after_seconds=0)

    ttl = await redis.ttl(key)
    # A missing/expired key reports ttl as -1 or -2 depending on the client/state;
    # never hand back a negative or zero Retry-After.
    retry_after = ttl if ttl and ttl > 0 else window_seconds
    return RateLimitResult(allowed=False, count=count, limit=limit, retry_after_seconds=retry_after)
