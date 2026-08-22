"""The pure half of the ingestion pipeline: parsing/dedup and delta detection.

Both `parse_and_dedupe` and `partition_by_delta` are synchronous and I/O-free by
design (see `app/ingestion/pipeline.py`), so this suite never touches Weaviate,
Mongo or Postgres - it is the same guarantee `test_adapters.py` and
`test_normalize.py` already give the layers below this one.
"""

from __future__ import annotations

from app.ingestion.adapters.base import ColumnMapping, FeedAdapter
from app.ingestion.pipeline import parse_and_dedupe, partition_by_delta
from app.models.product import IngestionError, Product

MAPPING = ColumnMapping(sku="id", title="name", price="price")
ADAPTER = FeedAdapter(MAPPING)


# --- parse_and_dedupe --------------------------------------------------------------


def test_parses_every_row_into_a_product() -> None:
    rows = [
        {"id": "A1", "name": "Widget", "price": "9.99"},
        {"id": "A2", "name": "Gadget", "price": "19.99"},
    ]
    parsed = parse_and_dedupe(ADAPTER, rows)

    assert [p.sku for p in parsed.products] == ["A1", "A2"]
    assert parsed.stats.total == 2
    assert parsed.stats.ok == 2
    assert parsed.stats.failed == 0
    assert parsed.errors == []


def test_bad_rows_become_errors_and_do_not_block_good_ones() -> None:
    rows = [
        {"id": "", "name": "Missing SKU"},
        {"id": "A1", "name": "Widget", "price": "9.99"},
        {"id": "A2", "name": ""},  # missing title
    ]
    parsed = parse_and_dedupe(ADAPTER, rows)

    assert [p.sku for p in parsed.products] == ["A1"]
    assert parsed.stats.ok == 1
    assert parsed.stats.failed == 2
    assert all(isinstance(e, IngestionError) for e in parsed.errors)
    assert {e.row_number for e in parsed.errors} == {1, 3}


def test_duplicate_sku_keeps_last_occurrence_and_its_own_raw_row() -> None:
    rows = [
        {"id": "A1", "name": "Widget v1", "price": "9.99"},
        {"id": "A1", "name": "Widget v2", "price": "12.99"},
    ]
    parsed = parse_and_dedupe(ADAPTER, rows)

    assert len(parsed.products) == 1
    assert parsed.products[0].title == "Widget v2"
    assert parsed.stats.duplicates == 1
    # The raw row stored for replay must be the *winning* occurrence's own row.
    assert parsed.raw_by_sku["A1"]["name"] == "Widget v2"


def test_dedupe_preserves_first_appearance_order() -> None:
    rows = [
        {"id": "B", "name": "Second"},
        {"id": "A", "name": "First"},
        {"id": "B", "name": "Second, updated"},
    ]
    parsed = parse_and_dedupe(ADAPTER, rows)

    assert [p.sku for p in parsed.products] == ["B", "A"]


def test_raw_by_sku_only_contains_successfully_parsed_rows() -> None:
    rows = [
        {"id": "A1", "name": "Widget"},
        {"id": "", "name": "No SKU"},
    ]
    parsed = parse_and_dedupe(ADAPTER, rows)

    assert set(parsed.raw_by_sku) == {"A1"}


def test_empty_feed_is_not_an_error() -> None:
    parsed = parse_and_dedupe(ADAPTER, [])

    assert parsed.products == []
    assert parsed.stats.total == 0


# --- partition_by_delta -------------------------------------------------------------


def _product(sku: str, title: str = "T") -> Product:
    return Product(sku=sku, title=title)


def test_new_sku_is_not_unchanged() -> None:
    products = [_product("A1")]
    changed, unchanged = partition_by_delta(products, existing={})

    assert changed == products
    assert unchanged == 0


def test_matching_content_hash_is_skipped() -> None:
    product = _product("A1")
    changed, unchanged = partition_by_delta([product], existing={"A1": product.content_hash()})

    assert changed == []
    assert unchanged == 1


def test_changed_content_hash_is_reembedded() -> None:
    product = _product("A1", title="New title")
    changed, unchanged = partition_by_delta([product], existing={"A1": "stale-hash"})

    assert changed == [product]
    assert unchanged == 0


def test_partition_only_reports_what_it_was_given() -> None:
    """A SKU present in `existing` but absent from the feed is simply not mentioned -
    delta detection here is not responsible for deciding what to delete."""
    changed, unchanged = partition_by_delta([], existing={"gone": "some-hash"})

    assert changed == []
    assert unchanged == 0


def test_mixed_batch_splits_correctly() -> None:
    new = _product("NEW")
    unchanged_product = _product("SAME")
    changed_product = _product("CHANGED", title="v2")

    existing = {
        "SAME": unchanged_product.content_hash(),
        "CHANGED": "old-hash-before-the-title-changed",
    }
    changed, unchanged_count = partition_by_delta(
        [new, unchanged_product, changed_product], existing
    )

    assert {p.sku for p in changed} == {"NEW", "CHANGED"}
    assert unchanged_count == 1
