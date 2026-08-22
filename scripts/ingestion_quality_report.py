"""Data-quality report for the seeded catalogs: coverage, field completeness,
distribution sanity, category degeneracy, storage round-trip consistency, and a
human-readable spot-check sample.

Answers "did ingestion parse and store this catalog correctly" - a narrower and
earlier question than `eval/`'s "does search return the right SKUs for a labelled
query", which needs a working search endpoint and a golden query set that don't exist
yet. This is checkable today, directly against whatever `make seed` already put into
Weaviate. See `app/ingestion/quality.py` for the pure checks this orchestrates.

Run: `.venv/bin/python -m scripts.ingestion_quality_report [tenant substring]`
Requires `make up` (Weaviate) and the catalogs already seeded via `make seed` - the
storage-consistency check reads back what's actually stored, not just what parsing
would produce.
"""

from __future__ import annotations

import asyncio
import sys
from decimal import Decimal
from typing import Any

from app.ingestion.pipeline import ParsedFeed, parse_and_dedupe
from app.ingestion.quality import (
    QualityFlag,
    category_diversity,
    detect_anomalies,
    duplicate_rate_flags,
    even_sample,
    field_completeness,
    numeric_summary,
)
from app.models.product import Product
from app.retrieval import weaviate_client as wv
from scripts.seed import DemoCatalog, _catalogs, _read_csv

SPOT_CHECK_SIZE = 5


async def fetch_all_stored_properties(client: Any, tenant: str) -> dict[str, dict[str, Any]]:
    """Every object currently stored for a tenant, keyed by SKU - one iterator pass
    (server-side paginated), not one round trip per SKU. The same reason
    `existing_content_hashes` in `weaviate_client.py` does it this way: N individual
    fetches doesn't scale, and exhaustive coverage means N is every product, not a
    sample."""
    collection = wv.product_collection(client, tenant)
    stored: dict[str, dict[str, Any]] = {}
    async for obj in collection.iterator():
        sku = obj.properties.get("sku")
        if sku is not None:
            stored[str(sku)] = dict(obj.properties)
    return stored


def diff_stored_properties(product: Product, stored: dict[str, Any] | None) -> list[str]:
    """Re-derive expected Weaviate properties from the source and diff against what is
    actually stored. Pure (no I/O) so it's unit-testable and reusable for both the
    exhaustive and the ad-hoc single-SKU case."""
    if stored is None:
        return [f"{product.sku}: not found in Weaviate"]
    expected = wv.properties_from_product(product)
    issues: list[str] = []
    for name, expected_value in expected.items():
        if name == "updated_at":
            continue  # a timestamp, expected to differ run to run
        stored_value = stored.get(name)
        if isinstance(expected_value, float) and isinstance(stored_value, int | float):
            if abs(expected_value - stored_value) > 1e-6:
                issues.append(
                    f"{product.sku}.{name}: expected {expected_value!r}, stored {stored_value!r}"
                )
        elif stored_value != expected_value:
            issues.append(
                f"{product.sku}.{name}: expected {expected_value!r}, stored {stored_value!r}"
            )
    return issues


async def check_storage_consistency(client: Any, tenant: str, products: list[Product]) -> list[str]:
    """Every product in the catalog, not a sample - diffed against one full iterator
    pass over the tenant. Catches drift a parsing-only check cannot: a partially
    failed batch, a stale upsert, a SKU whose derivation changed since it was last
    ingested - and exhaustively, so an issue confined to some other subset of rows
    than a 20-item sample happened to draw cannot hide."""
    stored_by_sku = await fetch_all_stored_properties(client, tenant)
    issues: list[str] = []
    for product in products:
        issues.extend(diff_stored_properties(product, stored_by_sku.get(product.sku)))
    return issues


def print_catalog_report(
    catalog: DemoCatalog, parsed: ParsedFeed, *, peer_duplicate_flag: QualityFlag | None = None
) -> list[QualityFlag]:
    products = parsed.products
    completeness = field_completeness(products)
    price = numeric_summary([float(p.price) for p in products if p.price is not None])
    rating = numeric_summary([p.rating for p in products if p.rating is not None])
    desc_len = numeric_summary([float(len(p.description)) for p in products if p.description])
    category = category_diversity(products)
    flags = detect_anomalies(
        parse_stats=parsed.stats,
        completeness=completeness,
        price=price,
        rating=rating,
        category=category,
    )
    if peer_duplicate_flag is not None:
        flags.append(peer_duplicate_flag)

    print("=" * 72)
    print(f"{catalog.name}  (tenant={catalog.tenant})")
    print(
        f"  rows: total={parsed.stats.total} ok={parsed.stats.ok} "
        f"failed={parsed.stats.failed} duplicates={parsed.stats.duplicates}"
    )
    if parsed.stats.reasons:
        print(f"  failure reasons: {parsed.stats.reasons}")
    print(f"  products after dedup: {len(products)}")

    print("\n  field completeness:")
    for name, frac in sorted(completeness.items(), key=lambda kv: kv[1]):
        print(f"    {name:<16} {frac:>6.1%}")

    print(
        f"\n  price:  min={price.min} p25={price.p25} median={price.median} p75={price.p75} "
        f"p95={price.p95} max={price.max}  (n={price.count}, null={price.null_count})"
    )
    print(
        f"  rating: min={rating.min} median={rating.median} max={rating.max}  "
        f"(n={rating.count}, null={rating.null_count})"
    )
    print(
        f"  description length (chars): median={desc_len.median} p95={desc_len.p95} "
        f"max={desc_len.max}"
    )

    print(
        f"\n  category: {category.distinct_categories} distinct value(s) across "
        f"{category.products_with_category} categorised products  (HHI={category.hhi:.2f})"
    )
    for label, count in category.top():
        print(f"    {count:>6}  {label}")

    if flags:
        print("\n  FLAGS:")
        for flag in flags:
            marker = "!!" if flag.severity == "critical" else " !"
            print(f"    {marker} [{flag.severity}] {flag.message}")
        if any("junk value" in f.message for f in flags):
            print_price_evidence(products)
    else:
        print("\n  no flags.")

    return flags


def print_price_evidence(products: list[Product], n: int = 3) -> None:
    """A price-outlier flag on its own can't distinguish a junk value from the top of
    a real secondary cluster (a $750 Garmin smartwatch in an otherwise sub-$50
    clothing catalog is legitimate; a $888,888 placeholder row is not) - both look
    identical to a purely statistical test with no product-type context. Printing the
    actual highest-priced products is what makes that distinction fast for a human,
    instead of requiring a trip back to the raw CSV to find out which it was."""

    def _price(p: Product) -> Decimal:
        return p.price if p.price is not None else Decimal(0)

    priced = sorted((p for p in products if p.price is not None), key=_price, reverse=True)
    print("\n  highest-priced products (is this a real premium tier, or junk?):")
    for product in priced[:n]:
        print(f"    {product.price} {product.currency}  {product.brand or ''} {product.title[:70]}")


def print_spot_check(products: list[Product], n: int = SPOT_CHECK_SIZE) -> None:
    print(f"\n  spot check ({n} products, evenly sampled - read these yourself):")
    for product in even_sample(products, n):
        desc = product.description[:120] + ("..." if len(product.description) > 120 else "")
        print(f"    SKU {product.sku}")
        print(f"      title:    {product.title[:90]}")
        print(f"      brand:    {product.brand!r}   price: {product.price} {product.currency}")
        print(f"      category: {product.category_path}")
        print(f"      desc:     {desc!r}")


async def main() -> None:
    target = sys.argv[1] if len(sys.argv) > 1 else None
    catalogs = _catalogs()
    if target:
        catalogs = [c for c in catalogs if target in c.tenant]
        if not catalogs:
            print(f"no catalog with tenant matching {target!r}")
            return

    # Phase 1: parse everything first. Duplicate-rate flagging needs every catalog's
    # rate at once to self-calibrate (see `duplicate_rate_flags`) - it can't be
    # decided one catalog at a time the way the other checks can.
    parsed_by_catalog: dict[str, ParsedFeed] = {}
    for catalog in catalogs:
        rows: list[dict[str, str]] = []
        for path in catalog.files:
            rows.extend(_read_csv(path))
        parsed_by_catalog[catalog.tenant] = parse_and_dedupe(catalog.adapter_factory(), rows)

    duplicate_rates = {
        tenant: (parsed.stats.duplicates / parsed.stats.total if parsed.stats.total else 0.0)
        for tenant, parsed in parsed_by_catalog.items()
    }
    peer_flags = duplicate_rate_flags(duplicate_rates)

    all_flags: dict[str, list[QualityFlag]] = {}

    async with wv.weaviate_client() as client:
        for catalog in catalogs:
            parsed = parsed_by_catalog[catalog.tenant]

            all_flags[catalog.tenant] = print_catalog_report(
                catalog, parsed, peer_duplicate_flag=peer_flags.get(catalog.tenant)
            )
            print_spot_check(parsed.products)

            print(
                f"\n  storage round-trip consistency (ALL {len(parsed.products)} products vs live Weaviate):"
            )
            issues = await check_storage_consistency(client, catalog.tenant, parsed.products)
            if issues:
                for issue in issues[:50]:
                    print(f"    !! {issue}")
                if len(issues) > 50:
                    print(f"    ... and {len(issues) - 50} more (truncated for display)")
            else:
                print("    OK - every product matches what is actually stored")
            print()

    print("=" * 72)
    n_critical = sum(1 for flags in all_flags.values() for f in flags if f.severity == "critical")
    n_warning = sum(1 for flags in all_flags.values() for f in flags if f.severity == "warning")
    print(
        f"SUMMARY: {n_critical} critical, {n_warning} warning, across {len(all_flags)} catalog(s)"
    )


if __name__ == "__main__":
    asyncio.run(main())
