"""Rigorous, rerunnable benchmark for the three batching decisions in the ingestion
pipeline: Mongo `bulk_write` chunk size, Weaviate `insert_many` batch size, and the
CPU embedding batch size.

Why this exists as a committed script and not a one-off scratchpad experiment: the
first pass at this (done inline in a session, not committed) had real methodological
gaps - synthetic Weaviate objects instead of real `Product` properties, only the
fresh-insert shape, no concurrent load, no tail latency, only 3 repetitions. This
version fixes all of that:

* Uses real `Product` objects parsed from the actual seeded catalogs (via the real
  adapters), not synthetic stand-ins - so payload size is realistic.
* Tests both fresh-insert (empty collection/tenant) and upsert-into-existing (the more
  common real path: re-ingesting into an already-populated catalog).
* Tests concurrent load - multiple tenants/collections written to at once - because a
  chunk size that is fastest alone can behave differently under contention.
* Reports p50/p95/p99 of per-chunk latency, not just mean throughput.
* Uses the real, mixed-length text distribution across all three demo catalogs for
  the embedding sweep, after an earlier check found throughput varies ~10x with text
  length alone (title-only ~590 docs/sec vs a full description ~55 docs/sec) - a fixed
  synthetic length would have hidden that entirely.

Run: `.venv/bin/python -m scripts.bench_ingestion [mongo|weaviate|embedding|all]`
Requires `make up` (Mongo, Weaviate) and the three demo catalogs downloaded into
`data/raw/` (see `data/SOURCES.md`) - it does not need them seeded, it reads the raw
CSVs itself.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import ReplaceOne

from app.ingestion.adapters.demo_catalogs import (
    AmazonElectronicsAdapter,
    MyntraFashionAdapter,
    SheinHomeGoodsAdapter,
)
from app.ingestion.pipeline import parse_and_dedupe
from app.models.product import Product
from app.retrieval import weaviate_client as wv
from scripts.bench_utils import percentiles
from scripts.seed import _read_csv

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
MONGO_REPS = 10
WEAVIATE_REPS = 10
EMBED_REPS = 5  # CPU embedding is ~100x more expensive per trial than the other two


# --- shared: build real Product objects from the real seeded feeds ----------------


def load_real_products() -> dict[str, list[Product]]:
    """Parse every demo feed through its real adapter. Same code path `scripts.seed`
    uses - these are exactly the objects that get embedded and upserted in production,
    not stand-ins."""
    catalogs = {
        "fashion": (
            DATA_DIR / "fashion-myntra" / "Myntra_fashion_products.csv",
            MyntraFashionAdapter(),
        ),
        "electronics": (
            DATA_DIR / "electronics-amazon" / "electronics_product.csv",
            AmazonElectronicsAdapter(),
        ),
    }
    out: dict[str, list[Product]] = {}
    for name, (path, adapter) in catalogs.items():
        rows = _read_csv(path)
        out[name] = parse_and_dedupe(adapter, rows).products

    home_rows: list[dict[str, str]] = []
    for f in (
        "us-shein-home_and_kitchen-3719.csv",
        "us-shein-home_textile-3883.csv",
        "us-shein-tools_and_home_improvement-3903.csv",
    ):
        home_rows.extend(_read_csv(DATA_DIR / "home-shein" / f))
    out["home"] = parse_and_dedupe(
        SheinHomeGoodsAdapter(default_category="Home Goods"), home_rows
    ).products
    return out


def report(
    label: str, sizes: list[int], per_chunk_latencies: dict[int, list[float]], n_items: int
) -> None:
    """docs/sec uses actual items-per-chunk (`min(size, n_items)`), not the nominal
    `size` - otherwise an oversized chunk that only ever covers `n_items` in one call
    would be scored as if it moved `size` items, inflating its throughput."""
    print(f"\n{label}")
    print(
        f"{'size':>7}  {'n_chunks':>8}  {'p50 ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}  {'docs/sec':>10}"
    )
    for size in sizes:
        vals = per_chunk_latencies[size]
        p50, p95, p99 = percentiles(vals)
        actual_chunk_len = min(size, n_items)
        rate = actual_chunk_len / statistics.mean(vals) if vals else 0.0
        print(
            f"{size:>7}  {len(vals):>8}  {p50 * 1000:>8.1f}  {p95 * 1000:>8.1f}  "
            f"{p99 * 1000:>8.1f}  {rate:>10.0f}"
        )


# --- Mongo bulk_write ---------------------------------------------------------------


async def _mongo_trial(coll: Any, items: list[tuple[str, dict]], size: int) -> list[float]:
    """One trial: write `items` in chunks of `size`. Returns per-chunk latencies."""
    latencies = []
    for start in range(0, len(items), size):
        chunk = items[start : start + size]
        ops = [ReplaceOne({"_id": s}, {"_id": s, **r}, upsert=True) for s, r in chunk]
        t0 = time.perf_counter()
        await coll.bulk_write(ops, ordered=False)
        latencies.append(time.perf_counter() - t0)
    return latencies


async def bench_mongo(products_by_catalog: dict[str, list[Product]]) -> None:
    # Real rows including the Myntra images blob - the biggest documents we actually have.
    rows = _read_csv(DATA_DIR / "fashion-myntra" / "Myntra_fashion_products.csv")[:3000]
    items = [(r["sku"], r) for r in rows]

    client: AsyncIOMotorClient = AsyncIOMotorClient("mongodb://localhost:27017")
    coll = client["bench_scratch"]["raw_products_bench"]
    sizes = [250, 500, 1000, 2000, 5000]

    print("=" * 70)
    print("MONGO bulk_write - fresh insert into an empty collection")
    fresh: dict[int, list[float]] = {s: [] for s in sizes}
    for size in sizes:
        for _ in range(MONGO_REPS):
            await coll.drop()
            fresh[size].extend(await _mongo_trial(coll, items, size))
    report("fresh insert (per-chunk latency)", sizes, fresh, len(items))

    print("\n" + "=" * 70)
    print(
        "MONGO bulk_write - upsert into an already-populated collection (real re-ingestion shape)"
    )
    await coll.drop()
    await coll.bulk_write(
        [ReplaceOne({"_id": s}, {"_id": s, **r}, upsert=True) for s, r in items], ordered=False
    )
    existing: dict[int, list[float]] = {s: [] for s in sizes}
    for size in sizes:
        for _ in range(MONGO_REPS):
            existing[size].extend(await _mongo_trial(coll, items, size))
    report(
        "upsert into existing 3,000-doc collection (per-chunk latency)", sizes, existing, len(items)
    )

    print("\n" + "=" * 70)
    print("MONGO bulk_write - CONCURRENT tenants writing at once (chunk size 1000)")
    for k in (1, 2, 4):
        colls = [client["bench_scratch"][f"raw_products_bench_{i}"] for i in range(k)]
        for c in colls:
            await c.drop()
        start = time.perf_counter()
        await asyncio.gather(*(_mongo_trial(c, items, 1000) for c in colls))
        elapsed = time.perf_counter() - start
        total_docs = len(items) * k
        print(
            f"  concurrency={k}: {elapsed:.2f}s total, "
            f"{total_docs / elapsed:.0f} docs/sec aggregate, "
            f"{len(items) / elapsed:.0f} docs/sec per-tenant-equivalent"
        )
        for c in colls:
            await c.drop()

    client.close()


# --- Weaviate insert_many ------------------------------------------------------------


async def _weaviate_trial(collection: Any, objects: list[Any], size: int) -> list[float]:
    latencies = []
    for start in range(0, len(objects), size):
        batch = objects[start : start + size]
        t0 = time.perf_counter()
        res = await collection.data.insert_many(batch)
        latencies.append(time.perf_counter() - t0)
        if res.errors:
            raise RuntimeError(res.errors)
    return latencies


async def bench_weaviate(products_by_catalog: dict[str, list[Product]]) -> None:
    from weaviate.collections.classes.data import DataObject
    from weaviate.util import generate_uuid5

    # Real fashion Products - the ones with the richest properties (description,
    # brand, attributes) of the three catalogs.
    products = products_by_catalog["fashion"][:1500]
    vector = [0.01] * 384
    sizes = [64, 128, 256, 512]

    async with wv.weaviate_client() as client:
        await wv.ensure_schema(client)
        tenant = "bench-scratch-weaviate"
        if not await wv.tenant_exists(client, tenant):
            await wv.create_tenant(client, tenant)

        def build_objects(rep: int) -> list[Any]:
            return [
                DataObject(
                    properties=wv.properties_from_product(p),
                    uuid=generate_uuid5(f"{p.sku}-r{rep}", tenant),
                    vector=vector,
                )
                for p in products
            ]

        print("=" * 70)
        print("WEAVIATE insert_many - fresh insert into an empty tenant (real Product properties)")
        fresh: dict[int, list[float]] = {s: [] for s in sizes}
        for size in sizes:
            for rep in range(WEAVIATE_REPS):
                await wv.delete_tenant(client, tenant)
                await wv.create_tenant(client, tenant)
                collection = wv.product_collection(client, tenant)
                fresh[size].extend(await _weaviate_trial(collection, build_objects(rep), size))
        report("fresh insert (per-batch latency)", sizes, fresh, len(products))

        print("\n" + "=" * 70)
        print(
            "WEAVIATE insert_many - inserting NEW objects into a tenant with 3,000 pre-existing ones"
        )
        print("  (each rep reloads a fresh 3,000-object background, then times genuinely new")
        print("  objects - NOT re-writing the same ones, which would let Weaviate skip HNSW")
        print("  graph work on a byte-identical overwrite and understate real insert cost)")
        background = products_by_catalog["electronics"][:3000]
        background_objects = [
            DataObject(
                properties=wv.properties_from_product(p),
                uuid=generate_uuid5(p.sku, "background"),
                vector=vector,
            )
            for p in background
        ]
        existing: dict[int, list[float]] = {s: [] for s in sizes}
        for size in sizes:
            for rep in range(WEAVIATE_REPS):
                await wv.delete_tenant(client, tenant)
                await wv.create_tenant(client, tenant)
                collection = wv.product_collection(client, tenant)
                for start in range(0, len(background_objects), 500):
                    await collection.data.insert_many(background_objects[start : start + 500])
                # genuinely new objects, distinct from both the background and every
                # other rep's uuids
                new_objects = build_objects(rep=2000 + rep)
                existing[size].extend(await _weaviate_trial(collection, new_objects, size))
        report(
            "insert new objects on top of a 3,000-object tenant (per-batch latency)",
            sizes,
            existing,
            len(products),
        )

        print("\n" + "=" * 70)
        print("WEAVIATE insert_many - CONCURRENT tenants writing at once (batch size 128)")
        for k in (1, 2, 4):
            tenants = [f"bench-scratch-weaviate-{i}" for i in range(k)]
            for t in tenants:
                if not await wv.tenant_exists(client, t):
                    await wv.create_tenant(client, t)
            collections = [wv.product_collection(client, t) for t in tenants]
            t0 = time.perf_counter()
            await asyncio.gather(
                *(_weaviate_trial(c, build_objects(rep=i), 128) for i, c in enumerate(collections))
            )
            elapsed = time.perf_counter() - t0
            total = len(products) * k
            print(
                f"  concurrency={k}: {elapsed:.2f}s total, "
                f"{total / elapsed:.0f} objects/sec aggregate, "
                f"{len(products) / elapsed:.0f} objects/sec per-tenant-equivalent"
            )
            for t in tenants:
                await wv.delete_tenant(client, t)

        await wv.delete_tenant(client, tenant)


# --- CPU embedding batch size --------------------------------------------------------


def _build_embedding_corpus(products_by_catalog: dict[str, list[Product]], n: int) -> list[str]:
    """Sample proportionally across all three catalogs, not just the first one in
    dict order. Taking `all_products[:n]` after concatenating dict values silently
    became 100% fashion text in an earlier run of this script, since fashion alone has
    more than `n` products - that measured a real number, but a mislabelled one (a
    fashion-only worst case, not a "mixed corpus")."""
    per_catalog = max(1, n // len(products_by_catalog))
    products = [p for plist in products_by_catalog.values() for p in plist[:per_catalog]]
    return [p.embedding_text() for p in products[:n]]


def bench_embedding(products_by_catalog: dict[str, list[Product]]) -> tuple[list[str], int]:
    """Runs the synchronous parts (batch-size sweep, length bucketing). Returns the
    corpus and the best batch size found, so the caller can run the concurrency test -
    which needs its own event loop - separately."""
    from app.ingestion.embed import embed_documents, warm_up

    warm_up()

    texts = _build_embedding_corpus(products_by_catalog, 2048)
    lengths = [len(t) for t in texts]
    print("=" * 70)
    print(
        f"EMBEDDING corpus (proportional across all 3 catalogs): n={len(texts)}, char length "
        f"min={min(lengths)} p50={statistics.median(lengths)} max={max(lengths)}"
    )

    sizes = [8, 16, 32, 64, 128, 256]
    print(f"\n{'batch_size':>10}  {'mean s':>8}  {'stdev':>7}  {'docs/sec':>9}")
    results: dict[int, list[float]] = {s: [] for s in sizes}
    for size in sizes:
        for _ in range(EMBED_REPS):
            start = time.perf_counter()
            embed_documents(texts, batch_size=size)
            results[size].append(time.perf_counter() - start)
        mean = statistics.mean(results[size])
        stdev = statistics.stdev(results[size]) if len(results[size]) > 1 else 0.0
        print(f"{size:>10}  {mean:>8.3f}  {stdev:>7.3f}  {len(texts) / mean:>9.1f}")

    best_size = min(sizes, key=lambda s: statistics.mean(results[s]))

    # Bucketing should matter most for a LARGE batch (more room for one outlier-length
    # text to force padding on everyone else in it) - test both the best size and the
    # largest, not just whichever happened to win the raw-throughput sweep.
    for bucket_size in sorted({best_size, sizes[-1]}):
        print(f"\nLENGTH BUCKETING at batch_size={bucket_size}: sorted-by-length vs natural order")
        natural_times = []
        sorted_times = []
        texts_sorted = sorted(texts, key=len)
        for _ in range(EMBED_REPS):
            t0 = time.perf_counter()
            embed_documents(texts, batch_size=bucket_size)
            natural_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            embed_documents(texts_sorted, batch_size=bucket_size)
            sorted_times.append(time.perf_counter() - t0)
        natural_mean = statistics.mean(natural_times)
        sorted_mean = statistics.mean(sorted_times)
        improvement = (natural_mean - sorted_mean) / natural_mean * 100
        print(f"  natural order:      mean {natural_mean:.3f}s")
        print(f"  sorted by length:   mean {sorted_mean:.3f}s  ({improvement:+.1f}%)")

    return texts, best_size


async def bench_embedding_concurrency(texts: list[str], best_size: int) -> None:
    from app.ingestion.embed import embed_documents

    print(
        f"\nCONCURRENCY at batch_size={best_size}: N simultaneous embedding calls (asyncio.to_thread)"
    )

    # Each concurrent caller gets its OWN full 512-doc job, not a slice of one shared
    # job - matching how the Mongo/Weaviate concurrency tests modelled "N merchants
    # each ingesting their own catalog at once", not "N threads splitting one catalog".
    per_caller = texts[:512]
    for k in (1, 2, 4):
        t0 = time.perf_counter()
        await asyncio.gather(
            *(
                asyncio.to_thread(embed_documents, per_caller, batch_size=best_size)
                for _ in range(k)
            )
        )
        elapsed = time.perf_counter() - t0
        total_docs = len(per_caller) * k
        print(
            f"  concurrency={k}: {elapsed:.2f}s total, "
            f"{total_docs / elapsed:.1f} docs/sec aggregate, "
            f"{len(per_caller) / elapsed:.1f} docs/sec per-caller (each did its own {len(per_caller)}-doc job)"
        )


async def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("loading real Product objects from the seeded catalogs...")
    products_by_catalog = load_real_products()
    for name, plist in products_by_catalog.items():
        print(f"  {name}: {len(plist)} products")

    if target in ("mongo", "all"):
        await bench_mongo(products_by_catalog)
    if target in ("weaviate", "all"):
        await bench_weaviate(products_by_catalog)
    if target in ("embedding", "all"):
        texts, best_size = bench_embedding(products_by_catalog)
        await bench_embedding_concurrency(texts, best_size)


if __name__ == "__main__":
    asyncio.run(main())
