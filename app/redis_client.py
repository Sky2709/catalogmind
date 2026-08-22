"""Shared async Redis client, cached per process - same lifecycle pattern as
`app/database.py` and `app/mongo.py`. `app/routers/health.py`'s readiness check
deliberately uses its own ephemeral client instead of this one (see that module) -
this one is for actual runtime use, like rate limiting.
"""

from __future__ import annotations

from functools import lru_cache

import redis.asyncio as aioredis

from app.config import get_settings


@lru_cache(maxsize=1)
def get_redis_client() -> aioredis.Redis:
    return aioredis.from_url(get_settings().redis_url)


async def dispose_redis_client() -> None:
    if get_redis_client.cache_info().currsize:
        await get_redis_client().aclose()
        get_redis_client.cache_clear()
