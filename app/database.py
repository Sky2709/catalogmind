"""Async SQLAlchemy engine and session lifecycle.

One engine per process, created lazily and cached. Sessions are per-request and always
closed, via the `session_scope` FastAPI dependency.

`pool_pre_ping` is on deliberately. This stack runs inside WSL where the VM can be
suspended between requests, which leaves pooled connections dead but still checked-in;
without pre-ping the next request fails with a confusing `ConnectionDoesNotExistError`
rather than transparently reconnecting.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_engine() -> AsyncEngine:
    settings = get_settings()
    return create_async_engine(
        settings.postgres_dsn,
        pool_pre_ping=True,
        pool_size=10,
        max_overflow=20,
        # Recycle before common idle-connection reapers (pgbouncer, cloud proxies) act.
        pool_recycle=1800,
        echo=False,
    )


@lru_cache(maxsize=1)
def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=get_engine(),
        class_=AsyncSession,
        # Objects stay usable after commit. Without this, reading `merchant.tenant`
        # after a commit triggers a lazy refresh on a closed session and blows up.
        expire_on_commit=False,
        autoflush=False,
    )


async def session_scope() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency: one session per request, committed or rolled back.

    Commit is centralised here rather than left to each handler, so a handler that
    forgets cannot silently discard writes, and any exception rolls the whole request
    back as one unit.
    """
    async with get_sessionmaker()() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def dispose_engine() -> None:
    """Close the pool on shutdown so the process exits cleanly."""
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
        logger.info("database engine disposed")
