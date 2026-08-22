"""Provision the three demo merchants and ingest their catalogs from cold.

`make seed` is the one command that takes a fresh `make up` stack to "ready to
search" - the Day 2 demo gate. Talks to the database and Weaviate directly through the
same functions the app uses, rather than over HTTP, so it works without `make dev`
running and needs no admin token.

Idempotent: re-running it reuses an existing merchant (no second API key - the first
is the only one ever shown) and re-ingesting the same files is a no-op past the first
run, courtesy of the pipeline's own content-hash delta detection.

See `data/SOURCES.md` for where each feed came from and why it's shaped the way it is.
"""

from __future__ import annotations

import asyncio
import csv
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select

from app.database import dispose_engine, get_sessionmaker
from app.ingestion.adapters.base import FeedAdapter
from app.ingestion.adapters.demo_catalogs import (
    AmazonElectronicsAdapter,
    MyntraFashionAdapter,
    SheinHomeGoodsAdapter,
)
from app.ingestion.pipeline import ingest
from app.models.db import ApiKey, Merchant, display_prefix, generate_api_key, hash_api_key
from app.mongo import dispose_mongo_client
from app.retrieval import weaviate_client as wv

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s %(message)s")
logger = logging.getLogger("seed")

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass(frozen=True)
class DemoCatalog:
    tenant: str
    name: str
    default_currency: str
    files: tuple[Path, ...]
    adapter_factory: Callable[[], FeedAdapter]


def _catalogs() -> list[DemoCatalog]:
    home_dir = DATA_DIR / "home-shein"
    return [
        DemoCatalog(
            tenant="demo-fashion-in",
            name="Demo: Indian Fashion (Myntra)",
            default_currency="INR",
            files=(DATA_DIR / "fashion-myntra" / "Myntra_fashion_products.csv",),
            adapter_factory=MyntraFashionAdapter,
        ),
        DemoCatalog(
            tenant="demo-electronics-in",
            name="Demo: Electronics (Amazon India)",
            default_currency="INR",
            files=(DATA_DIR / "electronics-amazon" / "electronics_product.csv",),
            adapter_factory=AmazonElectronicsAdapter,
        ),
        DemoCatalog(
            tenant="demo-home-goods",
            name="Demo: Home Goods (deliberately messy)",
            default_currency="USD",
            files=(
                home_dir / "us-shein-home_and_kitchen-3719.csv",
                home_dir / "us-shein-home_textile-3883.csv",
                home_dir / "us-shein-tools_and_home_improvement-3903.csv",
            ),
            adapter_factory=lambda: SheinHomeGoodsAdapter(default_category="Home Goods"),
        ),
    ]


def _read_csv(path: Path) -> list[dict[str, str]]:
    # utf-8-sig eats an Excel-exported BOM, same as the upload endpoint.
    with path.open(encoding="utf-8-sig", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


async def _get_or_create_merchant(catalog: DemoCatalog) -> tuple[Merchant, str | None]:
    """Returns the merchant and, only on first creation, its plaintext API key."""
    async with get_sessionmaker()() as session:
        existing = await session.scalar(select(Merchant).where(Merchant.tenant == catalog.tenant))
        if existing is not None:
            return existing, None

        merchant = Merchant(
            tenant=catalog.tenant,
            name=catalog.name,
            default_currency=catalog.default_currency,
        )
        session.add(merchant)

        raw_key = generate_api_key()
        session.add(
            ApiKey(
                merchant=merchant,
                key_hash=hash_api_key(raw_key),
                key_prefix=display_prefix(raw_key),
                label="seed",
            )
        )
        await session.commit()

    async with wv.weaviate_client() as client:
        await wv.ensure_schema(client)
        if not await wv.tenant_exists(client, catalog.tenant):
            await wv.create_tenant(client, catalog.tenant)

    logger.info("provisioned merchant tenant=%s", catalog.tenant)
    return merchant, raw_key


async def _seed_one(catalog: DemoCatalog) -> None:
    for path in catalog.files:
        if not path.exists():
            logger.error("missing feed file %s - see data/SOURCES.md for where it comes from", path)
            raise SystemExit(1)

    merchant, raw_key = await _get_or_create_merchant(catalog)
    if raw_key:
        logger.info(
            "  %s api key: %s  (save this - it will not be shown again)", catalog.tenant, raw_key
        )

    rows: list[dict[str, str]] = []
    for path in catalog.files:
        rows.extend(_read_csv(path))

    logger.info("ingesting %s rows for %s ...", len(rows), catalog.tenant)
    outcome = await ingest(merchant, rows, adapter=catalog.adapter_factory())
    logger.info(
        "%s: indexed=%s unchanged=%s failed=%s duplicates=%s",
        catalog.tenant,
        outcome.indexed,
        outcome.unchanged,
        outcome.parse.failed,
        outcome.parse.duplicates,
    )
    if outcome.parse.failed:
        logger.info("  failure reasons: %s", outcome.parse.reasons)


async def main() -> None:
    for catalog in _catalogs():
        await _seed_one(catalog)
    await dispose_engine()
    dispose_mongo_client()


if __name__ == "__main__":
    asyncio.run(main())
