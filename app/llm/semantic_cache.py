"""Per-tenant semantic cache for chat answers: skip the LLM call entirely when a
near-duplicate question was already answered recently.

Lookup cost has to stay near-zero to be worth paying before an LLM call, so this
reuses the existing free local embedder (`app.ingestion.embed`, CPU, no network) to
embed the incoming query, then compares it against a small, bounded, per-tenant list
of past (embedding, answer) pairs in Redis - a plain Python dot product per entry
(vectors are already L2-normalised, so cosine similarity *is* the dot product - the
same trick `app/ingestion/embedding_quality.py::cosine_similarity` uses,
reimplemented here as one line rather than importing across the ingestion/chat
layer boundary for a single arithmetic op).

**No vector-index infrastructure** (RedisVL or similar) - nothing in this repo has
that today, and a linear scan over a capped ~200-entry-per-tenant list is the
right-sized solution at this project's demo scale. Documented as a scale limit, not
hidden: past a few hundred entries per tenant this stops being O(1)-ish, and a real
vector index would be the production upgrade.

Uses the shared `get_redis_client()` (`app.redis_client`), same client rate limiting
already runs through - no separate connection.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

MAX_ENTRIES_PER_TENANT = 200
TTL_SECONDS = 24 * 60 * 60


def _key(tenant: str) -> str:
    return f"semcache:{tenant}"


def _cosine(a: list[float], b: list[float]) -> float:
    """Both vectors are already L2-normalised - see module docstring."""
    return sum(x * y for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class CachedAnswer:
    answer: str
    citations: list[dict[str, Any]]
    similarity: float


async def lookup(
    redis: Any, tenant: str, query_embedding: list[float], *, threshold: float
) -> CachedAnswer | None:
    """The best match at or above `threshold`, or `None` on a miss (including an
    empty/nonexistent cache)."""
    raw_entries = await redis.lrange(_key(tenant), 0, -1)
    best: CachedAnswer | None = None
    for raw in raw_entries:
        entry = json.loads(raw)
        similarity = _cosine(query_embedding, entry["embedding"])
        if similarity >= threshold and (best is None or similarity > best.similarity):
            best = CachedAnswer(
                answer=entry["answer"], citations=entry["citations"], similarity=similarity
            )
    return best


async def invalidate(redis: Any, tenant: str) -> None:
    """Drop every cached answer for a tenant - call after a re-ingestion actually
    changes the catalog (a price, a restock, a discontinued SKU).

    A real, previously-unfixed gap: this cache had no invalidation story at all, keyed
    purely on `tenant` with a flat 24h TTL and no catalog-version/content-hash check
    anywhere in it. A merchant re-ingesting their feed could have a shopper served a
    cached answer - and its citations, including price and stock status - from before
    the update for up to 24h, or until 200 newer queries evicted it. Same failure
    shape as the write-without-a-cache-flag bug this module's `store()` caller already
    fixed once (`app/llm/graph.py`'s comment on that fix) - a write path missing a
    guard that lets stale/wrong content sit in the shared cache for real shoppers.
    """
    await redis.delete(_key(tenant))


async def invalidate_all(redis: Any) -> int:
    """Drop every cached answer for every tenant - not a per-tenant re-ingestion
    signal, a blunt "wipe all of it" for exactly one caller: `app/main.py`'s
    startup, gated to `Settings.environment == "local"` only.

    Real problem this exists to solve: Redis is a separate, long-lived
    container (`make up`), untouched by `make dev`'s `--reload` restarts of the
    API process - a stale cached answer survives every dev-server reload,
    including ones where the underlying code/retrieval behaviour genuinely
    changed. A real live bug report ("suit, women" surfacing unrelated
    products) turned out to be exactly this: a stale cache entry, not a fresh
    reproduction of current behaviour, and diagnosing that took a live
    reproduction script specifically because the cache silently survived
    across restarts. Clearing it on every local startup makes "restart the
    dev server" mean what a developer actually expects it to mean - a genuinely
    fresh view of current code, not a lucky/unlucky cache hit from an hour ago.

    Deliberately **not** run in `prod`/`ci`: a real production restart (a
    deploy, a crash-restart, a rolling update) is unrelated to whether search
    results should have changed, and wiping every tenant's cache on every such
    restart would pay the real cost/latency benefit this cache exists for, for
    no reason most of the time. `SCAN`, not `KEYS` - non-blocking, safe even if
    this were ever pointed at a busy Redis instance (it never should be, but
    cheap to get right regardless).
    """
    deleted = 0
    async for key in redis.scan_iter(match="semcache:*"):
        deleted += await redis.delete(key)
    return deleted


async def store(
    redis: Any,
    tenant: str,
    query_embedding: list[float],
    answer: str,
    citations: list[dict[str, Any]],
) -> None:
    """Push the newest entry to the front and cap the list - a fixed-size ring
    buffer per tenant, not an ever-growing one."""
    key = _key(tenant)
    entry = json.dumps(
        {"embedding": query_embedding, "answer": answer, "citations": citations, "ts": time.time()}
    )
    await redis.lpush(key, entry)
    await redis.ltrim(key, 0, MAX_ENTRIES_PER_TENANT - 1)
    await redis.expire(key, TTL_SECONDS)
