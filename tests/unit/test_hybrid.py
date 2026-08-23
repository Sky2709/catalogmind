"""Pure-function pieces of `app.retrieval.hybrid` - filter building, the rerank text
projection, and hit assembly. None of this needs a live Weaviate; the query itself is
exercised in `tests/integration/test_search.py`.
"""

from __future__ import annotations

from decimal import Decimal

from weaviate.collections.classes.filters import _Filters

from app.retrieval.base import SearchFilters
from app.retrieval.hybrid import _build_filters, hit_from_properties, rerank_text


def test_no_filters_means_no_filter_object() -> None:
    assert _build_filters(SearchFilters()) is None


def test_a_single_filter_builds_a_filter_object() -> None:
    result = _build_filters(SearchFilters(in_stock_only=True))
    assert isinstance(result, _Filters)


def test_multiple_filters_combine_into_one_object() -> None:
    """Doesn't inspect the combined filter's internals (private API) - just proves
    several active filters don't crash the AND-combination step."""
    result = _build_filters(
        SearchFilters(
            min_price=Decimal("10"),
            max_price=Decimal("100"),
            brands=["Nike", "Adidas"],
            categories=["Shoes"],
            in_stock_only=True,
        )
    )
    assert isinstance(result, _Filters)


def test_rerank_text_orders_title_first() -> None:
    text = rerank_text(
        {
            "title": "Waterproof Hiking Boots",
            "brand": "Trailblazer",
            "category_path": ["Footwear", "Hiking"],
            "description": "Grippy sole, ankle support.",
            "attributes_text": "colour: brown\nsize: 10",
        }
    )
    lines = text.split("\n")
    assert lines[0] == "Waterproof Hiking Boots"
    assert "Trailblazer" in lines


def test_rerank_text_excludes_category_path() -> None:
    """A real, measured regression, not a hypothesis - see
    `Product.embedding_text()`'s docstring: once `category_path` held real
    per-product text, letting it feed BM25/rerank text diluted the
    discriminating title/description terms broadly enough to drop overall
    nDCG@10 from 0.9021 to 0.8355 across all 170 golden queries. Filterable
    Weaviate property only now, same design as `gender`."""
    text = rerank_text(
        {"title": "Waterproof Trail Boots", "category_path": ["Footwear", "Outdoor Gear"]}
    )
    assert "Footwear" not in text
    assert "Outdoor Gear" not in text


def test_rerank_text_tolerates_missing_optional_fields() -> None:
    """Home-goods-style rows: no brand, no description, empty category."""
    text = rerank_text({"title": "Mystery Item"})
    assert text == "Mystery Item"


def test_hit_from_properties_parses_attributes_json() -> None:
    hit = hit_from_properties(
        {
            "sku": "SKU-1",
            "title": "A Product",
            "price": 19.99,
            "attributes_json": '{"colour": "red"}',
        },
        score=0.83,
        rank_before_rerank=2,
    )
    assert hit.sku == "SKU-1"
    assert hit.price == Decimal("19.99")
    assert hit.attributes == {"colour": "red"}
    assert hit.rank_before_rerank == 2
    assert hit.score == 0.83


def test_hit_from_properties_tolerates_malformed_attributes_json() -> None:
    """A storage-layer bug should degrade to an empty attribute bag, not a 500."""
    hit = hit_from_properties(
        {"sku": "SKU-2", "title": "A Product", "attributes_json": "not json"},
        score=0.5,
        rank_before_rerank=0,
    )
    assert hit.attributes == {}


def test_hit_from_properties_defaults_missing_price_to_none() -> None:
    hit = hit_from_properties(
        {"sku": "SKU-3", "title": "A Product"}, score=0.1, rank_before_rerank=0
    )
    assert hit.price is None
    assert hit.category_path == []
