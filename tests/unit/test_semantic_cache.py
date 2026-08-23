"""Per-tenant semantic cache - against a tiny in-memory fake Redis, same "fake, not
mock" approach `test_rate_limit.py` uses. Embeddings here are toy 2-D vectors, not
real BGE output - the cosine-similarity arithmetic doesn't care about dimensionality,
and a real embedding call would make this suite slow and non-deterministic for no
benefit.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from app.llm.semantic_cache import MAX_ENTRIES_PER_TENANT, invalidate_all, lookup, store


class _FakeRedis:
    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}
        self.expire_calls: list[tuple[str, int]] = []

    async def lpush(self, key: str, value: str) -> None:
        self.lists.setdefault(key, []).insert(0, value)

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        values = self.lists.get(key, [])
        return values[start : end + 1] if end != -1 else values[start:]

    async def ltrim(self, key: str, start: int, end: int) -> None:
        if key in self.lists:
            self.lists[key] = self.lists[key][start : end + 1]

    async def expire(self, key: str, seconds: int) -> None:
        self.expire_calls.append((key, seconds))

    async def delete(self, key: str) -> int:
        return 1 if self.lists.pop(key, None) is not None else 0

    async def scan_iter(self, match: str) -> AsyncIterator[str]:
        prefix = match.removesuffix("*")
        for key in list(self.lists):
            if key.startswith(prefix):
                yield key


async def test_lookup_on_an_empty_cache_misses() -> None:
    redis = _FakeRedis()
    result = await lookup(redis, "tenant-a", [1.0, 0.0], threshold=0.95)
    assert result is None


async def test_an_identical_query_hits() -> None:
    redis = _FakeRedis()
    await store(redis, "tenant-a", [1.0, 0.0], "It's waterproof.", [{"sku": "BOOT-1"}])

    result = await lookup(redis, "tenant-a", [1.0, 0.0], threshold=0.95)
    assert result is not None
    assert result.answer == "It's waterproof."
    assert result.citations == [{"sku": "BOOT-1"}]
    assert result.similarity == 1.0


async def test_a_dissimilar_query_misses() -> None:
    redis = _FakeRedis()
    await store(redis, "tenant-a", [1.0, 0.0], "It's waterproof.", [])

    result = await lookup(redis, "tenant-a", [0.0, 1.0], threshold=0.95)
    assert result is None


async def test_the_best_of_several_entries_wins() -> None:
    redis = _FakeRedis()
    await store(redis, "tenant-a", [1.0, 0.0], "exact match, stored first", [])
    await store(redis, "tenant-a", [0.99, 0.14107], "close but not exact, stored second", [])

    result = await lookup(redis, "tenant-a", [1.0, 0.0], threshold=0.0)
    assert result is not None
    assert result.answer == "exact match, stored first"
    assert result.similarity == 1.0


async def test_tenants_are_isolated() -> None:
    redis = _FakeRedis()
    await store(redis, "tenant-a", [1.0, 0.0], "tenant a's answer", [])

    result = await lookup(redis, "tenant-b", [1.0, 0.0], threshold=0.95)
    assert result is None


async def test_the_per_tenant_list_is_capped() -> None:
    redis = _FakeRedis()
    for i in range(MAX_ENTRIES_PER_TENANT + 5):
        await store(redis, "tenant-a", [1.0, float(i)], f"answer {i}", [])

    assert len(redis.lists["semcache:tenant-a"]) == MAX_ENTRIES_PER_TENANT


async def test_storing_refreshes_the_ttl() -> None:
    redis = _FakeRedis()
    await store(redis, "tenant-a", [1.0, 0.0], "answer", [])
    assert redis.expire_calls
    key, seconds = redis.expire_calls[-1]
    assert key == "semcache:tenant-a"
    assert seconds > 0


async def test_invalidate_all_clears_every_tenant() -> None:
    redis = _FakeRedis()
    await store(redis, "tenant-a", [1.0, 0.0], "a's answer", [])
    await store(redis, "tenant-b", [1.0, 0.0], "b's answer", [])

    deleted = await invalidate_all(redis)

    assert deleted == 2
    assert await lookup(redis, "tenant-a", [1.0, 0.0], threshold=0.95) is None
    assert await lookup(redis, "tenant-b", [1.0, 0.0], threshold=0.95) is None


async def test_invalidate_all_on_an_empty_cache_is_a_no_op() -> None:
    redis = _FakeRedis()
    assert await invalidate_all(redis) == 0


async def test_invalidate_all_does_not_touch_unrelated_keys() -> None:
    """Only `semcache:*` - a real Redis instance shared with rate limiting
    (see module docstring) will have other key prefixes in it too."""
    redis = _FakeRedis()
    await store(redis, "tenant-a", [1.0, 0.0], "a's answer", [])
    redis.lists["ratelimit:tenant-a"] = ["some-unrelated-value"]

    await invalidate_all(redis)

    assert "ratelimit:tenant-a" in redis.lists
