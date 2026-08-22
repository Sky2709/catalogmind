"""Fixed-window rate limiting, against a tiny in-memory fake Redis - no real Redis
needed, the same "fake, not mock" approach `test_retry.py` uses for its fake clock.
"""

from __future__ import annotations

from app.rate_limit import check_rate_limit


class _FakeRedis:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.ttls: dict[str, int] = {}
        self.expire_calls: list[tuple[str, int]] = []

    async def incr(self, key: str) -> int:
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    async def expire(self, key: str, seconds: int) -> None:
        self.ttls[key] = seconds
        self.expire_calls.append((key, seconds))

    async def ttl(self, key: str) -> int:
        return self.ttls.get(key, -1)


async def test_requests_within_the_limit_are_allowed() -> None:
    redis = _FakeRedis()
    for expected_count in (1, 2, 3):
        result = await check_rate_limit(redis, "k", limit=3, window_seconds=60)
        assert result.allowed is True
        assert result.count == expected_count


async def test_request_exceeding_the_limit_is_rejected() -> None:
    redis = _FakeRedis()
    for _ in range(3):
        await check_rate_limit(redis, "k", limit=3, window_seconds=60)

    result = await check_rate_limit(redis, "k", limit=3, window_seconds=60)
    assert result.allowed is False
    assert result.count == 4
    assert result.limit == 3


async def test_expiry_is_set_once_per_window_not_on_every_request() -> None:
    """A request that increments an existing counter must not push its expiry back
    out - otherwise a steady trickle keeps one window open forever."""
    redis = _FakeRedis()
    for _ in range(5):
        await check_rate_limit(redis, "k", limit=10, window_seconds=60)
    assert redis.expire_calls == [("k", 60)]


async def test_retry_after_uses_the_actual_ttl_when_available() -> None:
    redis = _FakeRedis()
    for _ in range(3):
        await check_rate_limit(redis, "k", limit=3, window_seconds=60)
    redis.ttls["k"] = 42  # simulate Redis reporting 42s left on the real window

    result = await check_rate_limit(redis, "k", limit=3, window_seconds=60)
    assert result.allowed is False
    assert result.retry_after_seconds == 42


async def test_retry_after_falls_back_to_the_window_when_ttl_is_missing() -> None:
    """A missing/expired key reports ttl as -1 or -2 depending on client/state -
    never hand back a negative or zero Retry-After header value."""
    redis = _FakeRedis()
    for _ in range(3):
        await check_rate_limit(redis, "k", limit=3, window_seconds=60)
    redis.ttls["k"] = -1  # no expiry set / key vanished between calls

    result = await check_rate_limit(redis, "k", limit=3, window_seconds=60)
    assert result.retry_after_seconds == 60


async def test_different_keys_are_independent() -> None:
    redis = _FakeRedis()
    for _ in range(3):
        await check_rate_limit(redis, "tenant-a", limit=3, window_seconds=60)

    result = await check_rate_limit(redis, "tenant-b", limit=3, window_seconds=60)
    assert result.allowed is True
    assert result.count == 1
