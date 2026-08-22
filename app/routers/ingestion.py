"""Catalog upload: accept a feed, hand it to the pipeline, report progress.

Ingestion is scheduled as a FastAPI `BackgroundTask`, not awaited inline - the caller
gets `job_id` back immediately (202) and polls the status endpoint, matching the
`POST ...:ingest` / `GET .../ingestion/{job_id}` split. See `app.ingestion.pipeline`
for why that background task is not yet a real queue.
"""

from __future__ import annotations

import csv
import io
import logging
from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, Form, HTTPException, UploadFile, status

from app.deps import DbSession, ScopedMerchant
from app.ingestion.pipeline import run_ingestion_job
from app.models.db import IngestionJob, JobStatus
from app.rate_limit import check_rate_limit
from app.redis_client import get_redis_client
from app.schemas import IngestionAccepted, IngestionJobOut

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1/merchants", tags=["ingestion"])

# Generous for a CSV feed, small enough that reading it whole into memory is never a
# concern on this project's scale. A feed too large for this belongs behind a queue
# that streams from object storage, not this endpoint.
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# A real catalog upload is not a per-second operation - a merchant re-uploading their
# whole feed 10 times in a minute is either a broken retry loop or someone testing how
# hard they can hit this endpoint, not normal use. Keyed by tenant (from the API key,
# never the request), so one merchant's uploads can never count against another's.
INGEST_RATE_LIMIT = 10
INGEST_RATE_WINDOW_SECONDS = 60


async def enforce_ingest_rate_limit(merchant: ScopedMerchant) -> None:
    redis = get_redis_client()
    result = await check_rate_limit(
        redis,
        f"ratelimit:ingest:{merchant.tenant}",
        limit=INGEST_RATE_LIMIT,
        window_seconds=INGEST_RATE_WINDOW_SECONDS,
    )
    if not result.allowed:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Rate limit exceeded: {INGEST_RATE_LIMIT} uploads per "
                f"{INGEST_RATE_WINDOW_SECONDS}s. Retry after {result.retry_after_seconds}s."
            ),
            headers={"Retry-After": str(result.retry_after_seconds)},
        )


def _parse_csv(raw: bytes) -> list[dict[str, str]]:
    # utf-8-sig eats an Excel-exported BOM; a plain utf-8 decode leaves it glued to the
    # first header name and silently breaks that column's mapping.
    text = raw.decode("utf-8-sig")
    return [dict(row) for row in csv.DictReader(io.StringIO(text))]


@router.post(
    "/{tenant}/catalog:ingest",
    response_model=IngestionAccepted,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Upload a product feed for indexing",
    dependencies=[Depends(enforce_ingest_rate_limit)],
    responses={
        400: {"description": "Empty or unparseable upload"},
        429: {"description": "Too many uploads for this merchant in the current window"},
    },
)
async def ingest_catalog(
    merchant: ScopedMerchant,
    session: DbSession,
    background_tasks: BackgroundTasks,
    file: UploadFile,
    full_sync: Annotated[
        bool,
        Form(
            description=(
                "If true, delete any product already in this tenant but absent from "
                "this feed. Only safe when the upload is the merchant's complete, "
                "current catalog - never for a partial/delta upload, which would "
                "otherwise get every product it didn't mention deleted."
            )
        ),
    ] = False,
) -> IngestionAccepted:
    """Accept a CSV feed, create its job record, and start indexing in the background.

    The row count in the response is exact (the file is fully read to get it); the
    per-row parse outcome is not known yet - that is what the job status endpoint is
    for.
    """
    raw = await file.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "Feed exceeds the 50MB upload limit."
        )
    if not raw.strip():
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "Uploaded file is empty.")

    try:
        rows = _parse_csv(raw)
    except (csv.Error, UnicodeDecodeError) as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, f"Could not parse CSV: {exc}") from exc

    job = IngestionJob(
        merchant_id=merchant.id,
        status=JobStatus.PENDING,
        source_filename=file.filename,
        rows_total=len(rows),
    )
    session.add(job)
    # Committed here, not left to the `session_scope` dependency's post-response
    # commit: BackgroundTasks run before that cleanup code executes, so the worker's
    # own session would otherwise look up a job row that is not committed yet.
    await session.commit()
    job_id = job.id

    background_tasks.add_task(run_ingestion_job, job_id, rows, full_sync=full_sync)
    logger.info(
        "scheduled ingestion job %s merchant=%s rows=%s full_sync=%s",
        job_id,
        merchant.tenant,
        len(rows),
        full_sync,
    )

    return IngestionAccepted(job_id=job_id, status=job.status, rows_total=len(rows))


@router.get(
    "/{tenant}/ingestion/{job_id}",
    response_model=IngestionJobOut,
    summary="Check an ingestion job's progress",
    responses={404: {"description": "No such job for this merchant"}},
)
async def get_ingestion_job(
    job_id: int, merchant: ScopedMerchant, session: DbSession
) -> IngestionJobOut:
    """Scoped by `merchant_id` as well as `job_id`, so guessing another merchant's
    job id returns 404 rather than their ingestion progress."""
    job = await session.get(IngestionJob, job_id)
    if job is None or job.merchant_id != merchant.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No such ingestion job.")
    return IngestionJobOut.model_validate(job)
