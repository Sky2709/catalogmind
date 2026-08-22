"""CatalogMind API entrypoint.

A multi-tenant conversational commerce API: merchants POST a product feed and get
a grounded, streaming shopping assistant scoped strictly to their own catalog.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.database import dispose_engine
from app.ingestion.embed import warm_up as warm_up_embedder
from app.llm.client import dispose_bedrock_client
from app.mongo import dispose_mongo_client
from app.redis_client import dispose_redis_client
from app.retrieval.rerank import warm_up as warm_up_reranker
from app.retrieval.weaviate_client import dispose_shared_client
from app.routers import chat, health, ingestion, merchants, metrics, search, usage

logging.basicConfig(level=get_settings().log_level)
logger = logging.getLogger("catalogmind")

DESCRIPTION = """
Multi-tenant conversational commerce API.

**Tenant isolation is enforced at three layers** — Weaviate native multi-tenancy,
Postgres row-level security, and per-merchant MongoDB collections. A merchant's
API key resolves to exactly one tenant; no endpoint accepts a tenant from the
request body.

Typical flow:

1. `POST /v1/merchants` — provision a merchant, receive an API key
2. `POST /v1/merchants/{id}/catalog:ingest` — upload a product feed
3. `GET  /v1/merchants/{id}/ingestion/{job_id}` — watch it index
4. `POST /v1/merchants/{id}/chat` — talk to the catalog
"""

TAGS_METADATA = [
    {"name": "health", "description": "Liveness and dependency readiness."},
    {"name": "merchants", "description": "Tenant provisioning and API keys."},
    {"name": "ingestion", "description": "Catalog upload, normalisation and indexing."},
    {"name": "search", "description": "Hybrid retrieval with tunable alpha and reranking."},
    {"name": "chat", "description": "Grounded conversational commerce (SSE streaming)."},
    {"name": "usage", "description": "Per-merchant LLM token usage and estimated cost."},
    {"name": "admin", "description": "Tenant lifecycle and operational controls."},
    {"name": "observability", "description": "Prometheus scrape endpoint."},
]


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    # Crash now rather than serve prod with a development admin token.
    settings.validate_production()
    logger.info(
        "starting catalogmind env=%s weaviate=%s",
        settings.environment,
        settings.weaviate_http_url,
    )
    # Both models are lazily loaded on first use (`lru_cache`) and search needs both
    # (the embedder unconditionally, the reranker whenever a merchant's default or a
    # request asks for it) - paying that multi-second load here means the first real
    # shopper request never eats it. Loaded concurrently: they're independent model
    # loads, and neither saturates every core by itself during weight loading.
    t0 = time.perf_counter()
    await asyncio.gather(asyncio.to_thread(warm_up_embedder), asyncio.to_thread(warm_up_reranker))
    logger.info("warmed embedding + reranker models in %.2fs", time.perf_counter() - t0)
    try:
        yield
    finally:
        await dispose_engine()
        dispose_mongo_client()
        await dispose_redis_client()
        await dispose_shared_client()
        await dispose_bedrock_client()
        logger.info("shutting down catalogmind")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.api_title,
        version=settings.api_version,
        description=DESCRIPTION,
        openapi_tags=TAGS_METADATA,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.include_router(health.router)
    app.include_router(merchants.router)
    app.include_router(ingestion.router)
    app.include_router(search.router)
    app.include_router(chat.router)
    app.include_router(usage.router)
    app.include_router(metrics.router)

    # Minimal SSE chat page lives here; Swagger remains the primary interface.
    app.mount("/ui", StaticFiles(directory="static", html=True), name="ui")

    return app


app = create_app()
