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
