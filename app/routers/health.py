"""Liveness and readiness.

/health       - is the process up? (never touches dependencies)
/health/ready - is every dependency reachable? This is the Day-0 gate and the
                thing CI waits on before seeding fixtures.
"""

from __future__ import annotations

import asyncio
from typing import Literal

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from app.config import get_settings

router = APIRouter(tags=["health"])


class DependencyStatus(BaseModel):
    name: str
    ok: bool
    detail: str | None = None
    latency_ms: float | None = None


class ReadinessResponse(BaseModel):
    status: Literal["ready", "degraded"]
    dependencies: list[DependencyStatus]


@router.get("/health", summary="Liveness probe")
async def health() -> dict[str, str]:
    settings = get_settings()
    return {"status": "ok", "version": settings.api_version}


# A readiness probe must answer fast enough to be useful to an orchestrator. Client
# libraries disagree wildly about default connect timeouts (Weaviate took 8s here with
# nothing listening), so we impose our own ceiling rather than trusting each driver.
CHECK_TIMEOUT_SECONDS = 3.0


async def _timed(name: str, coro) -> DependencyStatus:
    loop = asyncio.get_running_loop()
    start = loop.time()

    def elapsed() -> float:
        return round((loop.time() - start) * 1000, 2)

    try:
        await asyncio.wait_for(coro, timeout=CHECK_TIMEOUT_SECONDS)
        return DependencyStatus(name=name, ok=True, latency_ms=elapsed())
    except TimeoutError:
        return DependencyStatus(
            name=name,
            ok=False,
            detail=f"timed out after {CHECK_TIMEOUT_SECONDS}s",
            latency_ms=elapsed(),
        )
    except Exception as exc:  # noqa: BLE001 - readiness reports, never raises
        return DependencyStatus(
            name=name,
            ok=False,
            detail=f"{type(exc).__name__}: {exc}",
            latency_ms=elapsed(),
        )


async def _check_weaviate() -> None:
    import weaviate

    settings = get_settings()
    client = weaviate.use_async_with_local(
        host=settings.weaviate_host,
        port=settings.weaviate_port,
        grpc_port=settings.weaviate_grpc_port,
    )
    await client.connect()
    try:
        if not await client.is_ready():
            raise RuntimeError("weaviate reported not ready")
    finally:
        await client.close()


async def _check_postgres() -> None:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine

    engine = create_async_engine(get_settings().postgres_dsn, pool_pre_ping=True)
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    finally:
        await engine.dispose()


async def _check_mongo() -> None:
    from motor.motor_asyncio import AsyncIOMotorClient

    client: AsyncIOMotorClient = AsyncIOMotorClient(
        get_settings().mongo_uri, serverSelectionTimeoutMS=3000
    )
    try:
        await client.admin.command("ping")
    finally:
        client.close()


async def _check_redis() -> None:
    import redis.asyncio as aioredis

    client = aioredis.from_url(get_settings().redis_url)
    try:
        await client.ping()
    finally:
        await client.aclose()


@router.get(
    "/health/ready",
    summary="Readiness probe",
    response_model=ReadinessResponse,
    responses={503: {"description": "One or more dependencies unreachable"}},
)
async def ready(response: Response) -> ReadinessResponse:
    checks = await asyncio.gather(
        _timed("weaviate", _check_weaviate()),
        _timed("postgres", _check_postgres()),
        _timed("mongo", _check_mongo()),
        _timed("redis", _check_redis()),
    )
    all_ok = all(c.ok for c in checks)
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return ReadinessResponse(status="ready" if all_ok else "degraded", dependencies=list(checks))
