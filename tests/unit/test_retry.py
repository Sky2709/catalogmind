"""Retry/backoff logic, with a fake `sleep` so tests exercise real retry counts and
delay growth without actually waiting.
"""

from __future__ import annotations

import pytest

from app.retry import with_retry


class _FlakyNetworkError(Exception):
    pass


class _PermanentError(Exception):
    pass


class _FakeClock:
    """Records every requested delay instead of sleeping."""

    def __init__(self) -> None:
        self.delays: list[float] = []

    async def sleep(self, seconds: float) -> None:
        self.delays.append(seconds)


async def test_succeeds_on_the_first_try_without_sleeping() -> None:
    clock = _FakeClock()
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        return "ok"

    result = await with_retry(call, retryable=(_FlakyNetworkError,), sleep=clock.sleep)
    assert result == "ok"
    assert calls == 1
    assert clock.delays == []


async def test_retries_and_eventually_succeeds() -> None:
    clock = _FakeClock()
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise _FlakyNetworkError("connection reset")
        return "ok"

    result = await with_retry(call, retryable=(_FlakyNetworkError,), attempts=5, sleep=clock.sleep)
    assert result == "ok"
    assert calls == 3
    assert len(clock.delays) == 2  # slept between attempt 1->2 and 2->3, not after success


async def test_reraises_after_exhausting_all_attempts() -> None:
    clock = _FakeClock()
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        raise _FlakyNetworkError("still down")

    with pytest.raises(_FlakyNetworkError, match="still down"):
        await with_retry(call, retryable=(_FlakyNetworkError,), attempts=3, sleep=clock.sleep)
    assert calls == 3
    assert len(clock.delays) == 2  # no sleep after the final, doomed attempt


async def test_non_retryable_exception_propagates_immediately() -> None:
    clock = _FakeClock()
    calls = 0

    async def call() -> str:
        nonlocal calls
        calls += 1
        raise _PermanentError("bad schema")

    with pytest.raises(_PermanentError):
        await with_retry(call, retryable=(_FlakyNetworkError,), attempts=5, sleep=clock.sleep)
    assert calls == 1  # a validation-style error must not be retried at all
    assert clock.delays == []


async def test_backoff_grows_and_stays_within_the_configured_max() -> None:
    clock = _FakeClock()

    async def call() -> None:
        raise _FlakyNetworkError()

    with pytest.raises(_FlakyNetworkError):
        await with_retry(
            call,
            retryable=(_FlakyNetworkError,),
            attempts=5,
            base_delay=1.0,
            max_delay=3.0,
            sleep=clock.sleep,
        )
    # base_delay * 2**attempt = 1, 2, 4, 8 before the cap and jitter; capped at 3.0,
    # then up to 25% jitter on top - so each delay must be >= its base and < cap*1.25.
    assert len(clock.delays) == 4
    uncapped = [1.0, 2.0, 3.0, 3.0]  # 4.0 and 8.0 clamp to the 3.0 max
    for delay, base in zip(clock.delays, uncapped, strict=True):
        assert base <= delay <= base * 1.25 + 1e-9
