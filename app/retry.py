"""Retry with exponential backoff, scoped to genuinely transient I/O failures.

Deliberately narrow: only a dropped connection or a timeout gets retried. A schema
or validation error is retried by nothing here, because retrying can't fix it and
would only delay surfacing a real problem - the same reasoning `_embed_and_upsert`
already applies when it refuses to swallow a Weaviate batch failure as a bad row.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

DEFAULT_ATTEMPTS = 3
DEFAULT_BASE_DELAY = 0.5
DEFAULT_MAX_DELAY = 5.0


async def with_retry[T](
    call: Callable[[], Awaitable[T]],
    *,
    retryable: tuple[type[BaseException], ...],
    attempts: int = DEFAULT_ATTEMPTS,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> T:
    """Calls `call()`, retrying on any exception type in `retryable` with exponential
    backoff plus up to 25% jitter (so N concurrent callers retrying the same outage
    don't all hammer the recovering service on the same schedule). Re-raises the
    final attempt's exception if every attempt fails - a caller that wanted a fallback
    instead of a raise needs to provide one itself, this only buys retries.

    `sleep` is injectable so tests exercise the real backoff/retry logic without
    actually waiting."""
    last_exc: BaseException | None = None
    for attempt in range(attempts):
        try:
            return await call()
        except retryable as exc:
            last_exc = exc
            if attempt == attempts - 1:
                break
            delay = min(max_delay, base_delay * (2**attempt)) * (1 + random.random() * 0.25)
            logger.warning(
                "retrying after %s (attempt %s/%s), sleeping %.2fs",
                type(exc).__name__,
                attempt + 1,
                attempts,
                delay,
            )
            await sleep(delay)
    assert last_exc is not None  # loop always sets this before falling through
    raise last_exc
