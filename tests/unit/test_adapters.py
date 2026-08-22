"""Feed adapters and the declarative column mapping.

The requirement these tests defend: onboarding merchant N+1 must not require a code
change. A new merchant whose CSV merely uses different column names should be
configuration, not a deploy.
"""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from app.ingestion.adapters.base import (
    ColumnMapping,
    FeedAdapter,
    ParseStats,
    deduplicate,
)
from app.models.product import IngestionError, Product

RICH_ROW = {
    "pid": "FLP-001",
    "name": "<b>Cotton</b> Shirt",
    "desc": "Soft &amp; light",
    "cat": "Clothing >> Men >> Shirts",
    "mrp": "Rs. 1,299.00",
    "sale": "999/-",
    "stock": "in stock",
    "stars": "4.5",
    "reviews": "1,234 reviews",
    "colour": "Blue",
    "fabric": "Cotton",
    "internal_note": "not mapped",
}

MAPPING = ColumnMapping(
    sku="pid",
    title="name",
    description="desc",
    category="cat",
    price="sale",
    original_price="mrp",
    in_stock="stock",
    rating="stars",
    review_count="reviews",
    attribute_columns=("colour", "fabric"),
    default_currency="INR",
)


@pytest.fixture
def adapter() -> FeedAdapter:
    return FeedAdapter(MAPPING)


# --- mapping ---------------------------------------------------------------------


def test_maps_every_field(adapter: FeedAdapter) -> None:
    p = adapter.parse_row(RICH_ROW, 1)
    assert isinstance(p, Product)
    assert p.sku == "FLP-001"
    assert p.title == "Cotton Shirt"  # tags stripped
    assert p.description == "Soft & light"  # entity resolved
    assert p.category_path == ["Clothing", "Men", "Shirts"]
    assert p.price == Decimal("999")
    assert p.original_price == Decimal("1299.00")
    assert p.currency == "INR"
    assert p.in_stock is True
    assert p.rating == 4.5
    assert p.review_count == 1234
    assert p.attributes == {"colour": "Blue", "fabric": "Cotton"}


def test_unmapped_columns_are_ignored(adapter: FeedAdapter) -> None:
    """A merchant's internal columns must not leak into the index."""
    p = adapter.parse_row(RICH_ROW, 1)
    assert isinstance(p, Product)
    assert "internal_note" not in p.attributes


def test_a_different_schema_needs_only_a_different_mapping() -> None:
    """The core claim: new merchant, zero code."""
    other = FeedAdapter(
        ColumnMapping(sku="item_id", title="product_name", price="cost", category=None)
    )
    p = other.parse_row({"item_id": "X1", "product_name": "Widget", "cost": "$4.99"}, 1)
    assert isinstance(p, Product)
    assert p.sku == "X1"
    assert p.title == "Widget"
    assert p.price == Decimal("4.99")


# --- rejection -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"pid": "  ", "name": "No SKU"}, "SKU"),
        ({"pid": None, "name": "No SKU"}, "SKU"),
        ({"name": "No SKU column"}, "SKU"),
        ({"pid": "A1", "name": ""}, "title"),
        ({"pid": "A1", "name": "   "}, "title"),
        ({"pid": "A1"}, "title"),
    ],
)
def test_unusable_rows_become_errors_not_exceptions(row: dict, expected: str) -> None:
    result = FeedAdapter(MAPPING).parse_row(row, 7)
    assert isinstance(result, IngestionError)
    assert expected in result.reason
    assert result.row_number == 7
    assert result.raw == row  # merchant gets the offending row back


def test_one_bad_row_does_not_abort_the_feed(adapter: FeedAdapter) -> None:
    """The whole point of yielding errors rather than raising."""
    rows = [RICH_ROW, {"pid": "", "name": "bad"}, {"pid": "B2", "name": "Good"}]
    results = list(adapter.parse(rows))
    assert [type(r).__name__ for r in results] == ["Product", "IngestionError", "Product"]


def test_row_numbers_are_one_based_and_sequential(adapter: FeedAdapter) -> None:
    rows = [{"pid": "", "name": "x"}] * 3
    errors = list(adapter.parse(rows))
    assert [e.row_number for e in errors] == [1, 2, 3]  # type: ignore[union-attr]


def test_parse_is_lazy(adapter: FeedAdapter) -> None:
    """A 5M-row feed must never be materialised in memory."""
    import types

    assert isinstance(adapter.parse([RICH_ROW]), types.GeneratorType)


# --- currency precedence ------------------------------------------------------------


def test_explicit_currency_column_wins() -> None:
    a = FeedAdapter(
        ColumnMapping(sku="s", title="t", price="p", currency="cur", default_currency="INR")
    )
    p = a.parse_row({"s": "1", "t": "x", "p": "$10", "cur": "gbp"}, 1)
    assert isinstance(p, Product)
    assert p.currency == "GBP"


def test_adapter_default_beats_symbol_sniffing() -> None:
    """'$' cannot distinguish USD from CAD; the adapter knows the source, so it wins."""
    a = FeedAdapter(ColumnMapping(sku="s", title="t", price="p", default_currency="CAD"))
    p = a.parse_row({"s": "1", "t": "x", "p": "$10"}, 1)
    assert isinstance(p, Product)
    assert p.currency == "CAD"


def test_symbol_sniffing_is_the_last_resort() -> None:
    a = FeedAdapter(ColumnMapping(sku="s", title="t", price="p"))
    p = a.parse_row({"s": "1", "t": "x", "p": "Rs. 10"}, 1)
    assert isinstance(p, Product)
    assert p.currency == "INR"


def test_rating_scale_comes_from_the_mapping() -> None:
    a = FeedAdapter(ColumnMapping(sku="s", title="t", rating="r", rating_scale=10.0))
    p = a.parse_row({"s": "1", "t": "x", "r": "8"}, 1)
    assert isinstance(p, Product)
    assert p.rating == 4.0


def test_missing_stock_column_uses_the_default() -> None:
    a = FeedAdapter(ColumnMapping(sku="s", title="t", default_in_stock=False))
    p = a.parse_row({"s": "1", "t": "x"}, 1)
    assert isinstance(p, Product)
    assert p.in_stock is False


# --- ColumnMapping.from_dict --------------------------------------------------------


def test_from_dict_ignores_unknown_keys() -> None:
    """A mapping written by a newer version must not break an older worker."""
    m = ColumnMapping.from_dict({"sku": "id", "title": "name", "some_future_field": "whatever"})
    assert m.sku == "id"
    assert m.title == "name"


def test_from_dict_coerces_attribute_columns_to_tuple() -> None:
    m = ColumnMapping.from_dict({"attribute_columns": ["a", "b"]})
    assert m.attribute_columns == ("a", "b")


# --- deduplication -------------------------------------------------------------------


def test_duplicate_skus_collapse_last_wins() -> None:
    """Feeds are commonly appended to, so a later row is the more recent truth.

    Left alone, two objects with the same SKU compete for the same query and split
    their own relevance.
    """
    a = FeedAdapter(ColumnMapping(sku="s", title="t"))
    rows = [{"s": "A", "t": "first"}, {"s": "B", "t": "other"}, {"s": "A", "t": "second"}]
    out = list(deduplicate(a.parse(rows)))
    titles = {p.sku: p.title for p in out if isinstance(p, Product)}
    assert titles == {"A": "second", "B": "other"}


def test_deduplicate_preserves_first_appearance_order() -> None:
    a = FeedAdapter(ColumnMapping(sku="s", title="t"))
    rows = [{"s": "A", "t": "1"}, {"s": "B", "t": "2"}, {"s": "A", "t": "3"}]
    out = [p.sku for p in deduplicate(a.parse(rows)) if isinstance(p, Product)]
    assert out == ["A", "B"]


def test_deduplicate_passes_errors_through() -> None:
    a = FeedAdapter(ColumnMapping(sku="s", title="t"))
    rows = [{"s": "", "t": "bad"}, {"s": "A", "t": "good"}]
    out = list(deduplicate(a.parse(rows)))
    assert sum(isinstance(r, IngestionError) for r in out) == 1


def test_deduplicate_counts_duplicates() -> None:
    a = FeedAdapter(ColumnMapping(sku="s", title="t"))
    stats = ParseStats()
    list(deduplicate(a.parse([{"s": "A", "t": "1"}, {"s": "A", "t": "2"}]), stats))
    assert stats.duplicates == 1


# --- ParseStats ----------------------------------------------------------------------


def test_stats_bucket_reasons_by_prefix() -> None:
    """400 failures must aggregate into a few actionable buckets, not 400 strings."""
    stats = ParseStats()
    stats.record_failure("missing or empty SKU (column 'pid')")
    stats.record_failure("missing or empty SKU (column 'id')")
    stats.record_failure("missing or empty title (column 'name')")
    assert stats.reasons == {"missing or empty SKU": 2, "missing or empty title": 1}


def test_success_rate() -> None:
    stats = ParseStats()
    stats.record_ok()
    stats.record_ok()
    stats.record_failure("x")
    assert stats.total == 3
    assert stats.success_rate == pytest.approx(2 / 3)


def test_success_rate_of_empty_feed_is_zero_not_a_crash() -> None:
    assert ParseStats().success_rate == 0.0


# --- delta detection guarantee --------------------------------------------------------


def test_content_hash_ignores_ingestion_timestamp(adapter: FeedAdapter) -> None:
    """The guarantee delta ingestion rests on.

    Two parses of an identical row happen at different wall-clock times and therefore
    carry different `updated_at`. If that leaked into the hash, every re-ingestion
    would look like a full catalog change and re-embed everything.
    """
    a = adapter.parse_row(RICH_ROW, 1)
    b = adapter.parse_row(RICH_ROW, 1)
    assert isinstance(a, Product) and isinstance(b, Product)
    # Force the timestamps apart so the test proves the exclusion rather than relying
    # on two parses happening to land in different clock ticks.
    b = b.model_copy(update={"updated_at": datetime(2020, 1, 1, tzinfo=UTC)})
    assert a.updated_at != b.updated_at
    assert a.content_hash() == b.content_hash()


def test_content_hash_changes_when_content_changes(adapter: FeedAdapter) -> None:
    a = adapter.parse_row(RICH_ROW, 1)
    b = adapter.parse_row({**RICH_ROW, "sale": "888"}, 1)
    assert isinstance(a, Product) and isinstance(b, Product)
    assert a.content_hash() != b.content_hash()


# --- subclass hooks --------------------------------------------------------------------


def test_preprocess_can_reshape_a_nested_row() -> None:
    """The escape hatch for feeds declarative mapping cannot express."""

    class NestedAdapter(FeedAdapter):
        name = "nested"

        def preprocess(self, row):  # type: ignore[no-untyped-def]
            return {"s": row["id"], "t": row["attrs"]["name"]}

    p = NestedAdapter(ColumnMapping(sku="s", title="t")).parse_row(
        {"id": "N1", "attrs": {"name": "Nested Widget"}}, 1
    )
    assert isinstance(p, Product)
    assert p.sku == "N1"
    assert p.title == "Nested Widget"


def test_postprocess_can_adjust_the_mapped_product() -> None:
    class BrandStamping(FeedAdapter):
        def postprocess(self, product, row):  # type: ignore[no-untyped-def]
            return product.model_copy(update={"brand": "HouseBrand"})

    p = BrandStamping(ColumnMapping(sku="s", title="t")).parse_row({"s": "1", "t": "x"}, 1)
    assert isinstance(p, Product)
    assert p.brand == "HouseBrand"


def test_exception_in_a_hook_becomes_an_error_row() -> None:
    class Broken(FeedAdapter):
        def preprocess(self, row):  # type: ignore[no-untyped-def]
            raise ValueError("boom")

    result = Broken(MAPPING).parse_row(RICH_ROW, 3)
    assert isinstance(result, IngestionError)
    assert "ValueError: boom" in result.reason
