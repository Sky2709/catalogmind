"""Turns one merchant's feed into indexed Weaviate objects.

Ties together what `app.ingestion.adapters` and `app.ingestion.embed` deliberately
leave to a caller: parse the feed, keep the last row per SKU, work out which products
actually changed, embed only those, upsert them, and keep an unmodified copy of every
accepted row in Mongo for replay.

**Runs as a FastAPI background task today, not a queue.** Catalogs at this project's
scale finish in low single-digit minutes fully embedded on CPU - measured seeding the
three demo catalogs (~34K rows total) - so a dedicated worker process would be
infrastructure the current scope doesn't need yet. `run_ingestion_job` takes only a
job id and the rows already read off the upload - exactly the signature an `arq` task
would have - so switching to a real queue later is a call-site change
(`background_tasks.add_task` -> `redis.enqueue_job`), not a rewrite of this module.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from pymongo import ReplaceOne
from pymongo.errors import ConnectionFailure

from app.database import get_sessionmaker
from app.ingestion.adapters.base import ColumnMapping, FeedAdapter, ParseStats
from app.ingestion.embed import aembed_documents
from app.llm import semantic_cache
from app.models.db import IngestionJob, JobStatus, Merchant
from app.models.product import IngestionError, Product
from app.mongo import raw_products_collection
from app.redis_client import get_redis_client
from app.retrieval import weaviate_client as wv
from app.retrieval.weaviate_client import WEAVIATE_TRANSIENT_ERRORS
from app.retry import with_retry

logger = logging.getLogger(__name__)

MONGO_TRANSIENT_ERRORS = (ConnectionFailure,)

EMBED_BATCH_SIZE = 128
"""The Weaviate `insert_many` batch size (see `_embed_and_upsert`) - how many products
go into one write call. Re-measured with real `Product` properties (not synthetic
stand-ins) via `scripts/bench_ingestion.py`: fresh-insert throughput slightly favoured
64, upsert-into-an-existing-tenant slightly favoured 128, and neither shape was ever
badly served by 128. Bigger batches (256, 512) were consistently worse in both shapes -
Weaviate saturates on per-object HNSW insert work rather than benefiting from batching
the way a network round trip does.

This is a *different* tuning problem from the CPU embedding batch size below, even
though they used to share this constant - see `EMBED_MODEL_BATCH_SIZE`."""

EMBED_MODEL_BATCH_SIZE = 16
"""The batch size passed to `aembed_documents` - i.e. how many texts go into one
`sentence_transformers.encode()` call. Was silently inheriting `embed.py`'s
`DEFAULT_BATCH_SIZE` (64), unexamined, until measured directly against the real,
mixed-length text this pipeline actually embeds (72-3,065 chars across the three demo
catalogs, not a fixed synthetic length): throughput at batch size 64 was ~51.5
docs/sec vs ~60.9 at 16 - a real ~18% loss that had been sitting unnoticed. Smaller
batches win here because a batch pads every sequence to its longest member - one long
description in a batch of 64 taxes 63 short ones with it. Tried explicit
length-bucketing (sorting texts by length before batching) as the "proper" fix at both
batch size 16 and 256; it barely moved the needle either way (<1%) - the simple fix
(smaller batches) captured the whole win and length-bucketing wasn't worth the added
complexity. Deliberately kept separate from `EMBED_BATCH_SIZE`: measured independently,
this pipeline's embedding calls should tune independently of Weaviate's, not share a
number by accident.

**Concurrency caveat, measured, not assumed:** running N of these concurrently (as
separate merchants' background ingestion jobs would) does not parallelise cleanly -
2 concurrent full jobs showed no aggregate improvement over 1, and 4 concurrent jobs
each took ~3.3x longer than running alone. `sentence_transformers` already tries to
use every CPU core inside one call; concurrent calls compete for the same cores rather
than scaling. If ingestion jobs need to run for multiple merchants at once, this is
the spot that needs a concurrency limiter (e.g. a semaphore around embedding calls),
not a bigger batch size - not implemented here since it's a behaviour change beyond
this scope, but worth flagging before this hits real concurrent traffic."""

MONGO_BULK_BATCH_SIZE = 2000
"""Re-measured with 10 reps (not 3) across two real shapes - fresh insert and upsert
into an already-populated collection - using real Myntra rows including the large
multi-URL image blob, the biggest documents this pipeline actually writes. 2,000
showed a consistent, reproducible ~25% throughput edge over 1,000 in *both* shapes
independently (not a one-off - a previous, less rigorous pass had picked 1,000).
Concurrency, also measured: 2 simultaneous tenants writing at once roughly doubles
aggregate throughput, but 4 simultaneous tenants barely improves on 2 (~10% more)
while each tenant's own effective rate drops by more than half versus writing alone -
this Mongo container saturates at ~2 concurrent writers on this dev machine. Nowhere
near pymongo's real ceiling either way (100,000 ops / 48MB per message - see
`pymongo.common`), so there is headroom before the batch size itself needs revisiting."""

MAX_ERROR_SAMPLE = 50
"""Job rows keep a capped sample of failures, not all of them - a 50,000-row feed with
400 bad rows should tell the merchant that, not hand back a 400-entry JSON blob."""

INGESTION_CONCURRENCY_LIMIT = 2
"""How many `ingest()` calls may run their Weaviate/Mongo/embedding work at once, in
this one process. Measured, not guessed: `scripts/bench_ingestion.py` found all three
resources - Mongo writes, Weaviate writes, and CPU embedding - saturate at ~2
concurrent users on this dev stack, with aggregate throughput barely improving beyond
that while each individual job's own completion time roughly doubles or worse. Two
merchants ingesting at once is the point past which a third just makes everyone
slower without doing more total work, so it is throttled here rather than left to
degrade silently. This is a per-process semaphore - if this API ever runs as more
than one worker process, the real limit is `INGESTION_CONCURRENCY_LIMIT * n_workers`,
which is a real gap a future queue-based worker (see the module docstring) would need
to account for with a shared (e.g. Redis-backed) limiter instead of this in-memory one."""
_ingestion_semaphore = asyncio.Semaphore(INGESTION_CONCURRENCY_LIMIT)


# --- pure parsing + delta decision: no I/O, fully unit-testable ------------------


@dataclass
class ParsedFeed:
    """Output of one pass over a feed's rows."""

    products: list[Product]
    """The winning product per SKU, in first-appearance order."""
    raw_by_sku: dict[str, Mapping[str, Any]]
    """Each winning product's own source row, unmodified - for the Mongo replay copy."""
    stats: ParseStats
    errors: list[IngestionError]


def parse_and_dedupe(adapter: FeedAdapter, rows: Iterable[Mapping[str, Any]]) -> ParsedFeed:
    """Map every row once, keeping the last occurrence of each SKU.

    Combines what `adapters.base.deduplicate` does with tracking each surviving
    product's raw row - `deduplicate` alone discards that context, and the Mongo
    replay copy needs it. Pure and synchronous so it is testable without a stack.
    """
    stats = ParseStats()
    errors: list[IngestionError] = []
    products_by_sku: dict[str, Product] = {}
    raw_by_sku: dict[str, Mapping[str, Any]] = {}
    order: list[str] = []

    for row_number, row in enumerate(rows, start=1):
        result = adapter.parse_row(row, row_number)
        if isinstance(result, IngestionError):
            stats.record_failure(result.reason)
            errors.append(result)
            continue

        if result.sku in products_by_sku:
            stats.record_duplicate()
        else:
            order.append(result.sku)
        stats.record_ok()
        products_by_sku[result.sku] = result
        raw_by_sku[result.sku] = row

    return ParsedFeed(
        products=[products_by_sku[sku] for sku in order],
        raw_by_sku=raw_by_sku,
        stats=stats,
        errors=errors,
    )


def partition_by_delta(
    products: Sequence[Product], existing: Mapping[str, str]
) -> tuple[list[Product], int]:
    """Split parsed products into "needs (re-)embedding" and "unchanged - skip".

    `existing` is `sku -> content_hash` for what a tenant already has indexed. A SKU
    missing from it is new; a SKU present with a different hash changed; a SKU present
    with the same hash is exactly what it was last ingestion and is not touched.
    """
    changed: list[Product] = []
    unchanged = 0
    for product in products:
        if existing.get(product.sku) == product.content_hash():
            unchanged += 1
        else:
            changed.append(product)
    return changed, unchanged


# --- orchestration: the I/O half ---------------------------------------------------


@dataclass
class IngestOutcome:
    """What one run did, before it is written onto the job row."""

    parse: ParseStats = field(default_factory=ParseStats)
    indexed: int = 0
    unchanged: int = 0
    deleted: int = 0
    """Removed because absent from a `full_sync` feed. Always 0 unless `full_sync`
    was requested - a partial/delta upload must never lose products it simply didn't
    mention."""
    errors: list[IngestionError] = field(default_factory=list)


async def _embed_and_upsert(
    client: Any, tenant: str, products: Sequence[Product], *, batch_size: int = EMBED_BATCH_SIZE
) -> None:
    """Embed and upsert in fixed-size batches, so one giant feed never holds every
    vector for the whole catalog in memory at once."""
    for start in range(0, len(products), batch_size):
        batch = products[start : start + batch_size]
        vectors = await aembed_documents(
            [p.embedding_text() for p in batch], batch_size=EMBED_MODEL_BATCH_SIZE
        )

        async def _upsert(
            batch: Sequence[Product] = batch, vectors: list[list[float]] = vectors
        ) -> Any:
            return await wv.upsert_products(client, tenant, batch, vectors)

        result = await with_retry(_upsert, retryable=WEAVIATE_TRANSIENT_ERRORS)
        if result.errors:
            # These rows already passed adapter validation - a failure here is
            # Weaviate/network trouble, not a bad row, and must not be swallowed as one.
            raise RuntimeError(f"weaviate batch upsert failed: {result.errors}")


async def _store_raw_rows(tenant: str, raw_by_sku: Mapping[str, Mapping[str, Any]]) -> None:
    """Keep an unmodified copy of every accepted row, upserted by SKU so re-ingesting
    the same feed updates the stored row instead of accumulating duplicates.

    One `bulk_write` per chunk, not one round trip per document. A serial loop of
    `replace_one` calls was the actual bottleneck the first time this ran end to end -
    seeding the three demo catalogs took ~9 minutes for 12,491 rows with the loop,
    dwarfing the CPU embedding pass it sits next to.
    """
    if not raw_by_sku:
        return
    collection = raw_products_collection(tenant)
    items = list(raw_by_sku.items())
    for start in range(0, len(items), MONGO_BULK_BATCH_SIZE):
        chunk = items[start : start + MONGO_BULK_BATCH_SIZE]
        operations = [
            ReplaceOne({"_id": sku}, {"_id": sku, **dict(raw)}, upsert=True) for sku, raw in chunk
        ]

        async def _bulk_write(operations: list[ReplaceOne[Any]] = operations) -> Any:
            return await collection.bulk_write(operations, ordered=False)

        await with_retry(_bulk_write, retryable=MONGO_TRANSIENT_ERRORS)


async def ingest(
    merchant: Merchant,
    rows: Iterable[Mapping[str, Any]],
    *,
    adapter: FeedAdapter | None = None,
    mapping: ColumnMapping | None = None,
    batch_size: int = EMBED_BATCH_SIZE,
    full_sync: bool = False,
) -> IngestOutcome:
    """Parse, dedupe, delta-detect, embed and upsert one feed for one tenant.

    `adapter` and `mapping` are mutually exclusive escape hatches, in order of
    escalation (see `app.ingestion.adapters.base`): most callers pass neither and get
    a plain `FeedAdapter` built from the merchant's stored `column_mapping` - the "no
    code" path. A caller with an unusual mapping but no other need can pass `mapping`
    directly. A feed a mapping cannot express (no id column, a field that needs
    reshaping) passes a full `FeedAdapter` subclass instance as `adapter`.

    `full_sync` controls whether a SKU that exists in the tenant but is *absent* from
    this feed gets deleted. Defaults to False deliberately: delta detection tells you
    what's new or changed, but "missing" is genuinely ambiguous - a merchant sending
    today's 50 price updates did not just discontinue the other 49,950 products. Only
    a caller that knows this feed is the merchant's complete, current catalog should
    opt in.
    """
    if adapter is not None and mapping is not None:
        raise ValueError("pass either `adapter` or `mapping`, not both")
    if adapter is None:
        adapter = FeedAdapter(mapping or ColumnMapping.from_dict(merchant.column_mapping or {}))
    parsed = parse_and_dedupe(adapter, rows)

    # Mongo, Weaviate and CPU embedding all measurably saturate under concurrent
    # ingestion (see `INGESTION_CONCURRENCY_LIMIT`) - a second job queues here rather
    # than piling straight onto already-contended resources.
    async with _ingestion_semaphore:
        async with wv.weaviate_client() as client:
            existing = await with_retry(
                lambda: wv.existing_content_hashes(client, merchant.tenant),
                retryable=WEAVIATE_TRANSIENT_ERRORS,
            )
            to_upsert, unchanged = partition_by_delta(parsed.products, existing)
            if to_upsert:
                await _embed_and_upsert(client, merchant.tenant, to_upsert, batch_size=batch_size)

            deleted = 0
            if full_sync:
                current_skus = {p.sku for p in parsed.products}
                stale_skus = [sku for sku in existing if sku not in current_skus]
                if stale_skus:
                    deleted = await with_retry(
                        lambda: wv.delete_products_by_sku(client, merchant.tenant, stale_skus),
                        retryable=WEAVIATE_TRANSIENT_ERRORS,
                    )

        await _store_raw_rows(merchant.tenant, parsed.raw_by_sku)

    # A merchant's re-ingest that actually changed something (a price, a restock, a
    # discontinued SKU) must not leave the chat semantic cache serving a shopper an
    # answer built from the old catalog state - see `semantic_cache.invalidate`'s
    # docstring. Skipped when nothing changed (`to_upsert`/`deleted` both empty) to
    # avoid a pointless Redis write on every no-op re-ingest.
    if to_upsert or deleted:
        await semantic_cache.invalidate(get_redis_client(), merchant.tenant)

    return IngestOutcome(
        parse=parsed.stats,
        indexed=len(to_upsert),
        unchanged=unchanged,
        deleted=deleted,
        errors=parsed.errors,
    )


async def run_ingestion_job(
    job_id: int, rows: Sequence[Mapping[str, Any]], *, full_sync: bool = False
) -> None:
    """Execute one ingestion job end-to-end and record the outcome on its row.

    Owns its own DB session rather than reusing the request's - by the time this runs
    (as a background task, after the response was sent) the request's session is
    already closed. `rows` are passed in already read off the upload for the same
    reason: the original `UploadFile` does not survive past the request either.
    """
    async with get_sessionmaker()() as session:
        job = await session.get(IngestionJob, job_id)
        if job is None:
            logger.error("ingestion job %s vanished before it could run", job_id)
            return
        merchant = await session.get(Merchant, job.merchant_id)
        assert merchant is not None  # FK guarantees this

        job.status = JobStatus.RUNNING
        await session.commit()

        try:
            outcome = await ingest(merchant, rows, full_sync=full_sync)
        except Exception as exc:  # noqa: BLE001 - a crashed job must resolve, not hang forever
            logger.exception("ingestion job %s failed", job_id)
            job.status = JobStatus.FAILED
            job.error_message = f"{type(exc).__name__}: {exc}"
            job.finished_at = datetime.now(UTC)
            await session.commit()
            return

        job.rows_total = outcome.parse.total
        job.rows_indexed = outcome.indexed
        job.rows_failed = outcome.parse.failed
        job.rows_skipped = outcome.unchanged
        job.rows_deleted = outcome.deleted
        job.errors = {
            "reasons": outcome.parse.reasons,
            "duplicates": outcome.parse.duplicates,
            "sample": [e.model_dump(mode="json") for e in outcome.errors[:MAX_ERROR_SAMPLE]],
            "sample_truncated": len(outcome.errors) > MAX_ERROR_SAMPLE,
        }
        job.status = JobStatus.PARTIAL if outcome.parse.failed else JobStatus.SUCCEEDED
        job.finished_at = datetime.now(UTC)
        await session.commit()

        logger.info(
            "ingestion job %s merchant=%s indexed=%s unchanged=%s deleted=%s failed=%s",
            job_id,
            merchant.tenant,
            outcome.indexed,
            outcome.unchanged,
            outcome.deleted,
            outcome.parse.failed,
        )
