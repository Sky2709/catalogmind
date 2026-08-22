"""The concurrency limiter itself: proves the semaphore that guards `ingest()`'s
Weaviate/Mongo/embedding work actually caps how many holders run at once. Doesn't
call `ingest()` end to end (that needs the real stack) - exercises the same
module-level semaphore directly, which is the actual mechanism doing the limiting.
"""

from __future__ import annotations

import asyncio

from app.ingestion.pipeline import INGESTION_CONCURRENCY_LIMIT, _ingestion_semaphore


async def test_semaphore_never_admits_more_than_the_configured_limit() -> None:
    concurrent_holders = 0
    max_observed = 0
    lock = asyncio.Lock()

    async def worker() -> None:
        nonlocal concurrent_holders, max_observed
        async with _ingestion_semaphore:
            async with lock:
                concurrent_holders += 1
                max_observed = max(max_observed, concurrent_holders)
            await asyncio.sleep(0.01)  # hold the slot long enough for overlap
            async with lock:
                concurrent_holders -= 1

    # More tasks than the limit, launched together, so admission is actually contested.
    await asyncio.gather(*(worker() for _ in range(INGESTION_CONCURRENCY_LIMIT * 4)))
    assert max_observed == INGESTION_CONCURRENCY_LIMIT


async def test_semaphore_is_fully_released_after_all_workers_finish() -> None:
    async def worker() -> None:
        async with _ingestion_semaphore:
            await asyncio.sleep(0)

    await asyncio.gather(*(worker() for _ in range(INGESTION_CONCURRENCY_LIMIT * 3)))

    # If every acquire was matched by a release, the full limit must be immediately
    # acquirable again - no held-open slot left over from a worker that didn't clean
    # up properly.
    for _ in range(INGESTION_CONCURRENCY_LIMIT):
        await asyncio.wait_for(_ingestion_semaphore.acquire(), timeout=0.1)
    for _ in range(INGESTION_CONCURRENCY_LIMIT):
        _ingestion_semaphore.release()
