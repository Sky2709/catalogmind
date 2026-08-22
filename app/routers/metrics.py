"""Prometheus scrape endpoint.

Unauthenticated, like `/health` - it carries no tenant data (see `app.obs.metrics`'s
module docstring for why nothing here is labelled by tenant), and a Prometheus scraper
living inside the cluster network is the normal way this gets protected in practice,
not an API key.
"""

from __future__ import annotations

from fastapi import APIRouter, Response

from app.obs.metrics import render_latest

router = APIRouter(tags=["observability"])


@router.get("/metrics", summary="Prometheus scrape endpoint", include_in_schema=False)
async def metrics() -> Response:
    body, content_type = render_latest()
    return Response(content=body, media_type=content_type)
