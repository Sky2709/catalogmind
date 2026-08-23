"""Locked per-tenant category taxonomy (`app/ingestion/taxonomy.py`) and the
flattened filter-value view of it the chat agent's tool schema uses
(`app/llm/prompting.py::_price_brand_filter_properties`).
"""

from __future__ import annotations

import pytest

from app.ingestion.taxonomy import (
    TAXONOMY_BY_TENANT,
    UNCATEGORIZED,
    category_filter_values,
    taxonomy_for,
)


@pytest.mark.parametrize("tenant", sorted(TAXONOMY_BY_TENANT))
def test_every_taxonomy_leaf_has_exactly_one_separator(tenant: str) -> None:
    """Every leaf must be a clean "Category > Subcategory" pair - a malformed
    leaf (missing separator, or more than one) would silently produce a wrong
    `category_path` split (`Product._split_category`) or a wrong flattened
    filter value."""
    for leaf in taxonomy_for(tenant):
        assert leaf.count(" > ") == 1, f"{tenant}: malformed leaf {leaf!r}"


@pytest.mark.parametrize("tenant", sorted(TAXONOMY_BY_TENANT))
def test_taxonomy_includes_uncategorized_fallback(tenant: str) -> None:
    assert UNCATEGORIZED in taxonomy_for(tenant)


def test_taxonomy_for_unknown_tenant_raises() -> None:
    """`scripts/enrich_categories.py` is only ever run by hand against a known
    tenant - a typo should fail loudly, not silently skip classification."""
    with pytest.raises(KeyError):
        taxonomy_for("not-a-real-tenant")


def test_category_filter_values_flattens_category_and_subcategory() -> None:
    values = category_filter_values("demo-fashion-in")
    assert values is not None
    # A subcategory leaf ...
    assert "Skincare" in values
    # ... and its parent category, since `contains_any` matches either level.
    assert "Beauty & Personal Care" in values


def test_category_filter_values_deduplicates_self_referential_leaves() -> None:
    """ "Watches > Watches" (a category with no real subcategory split) must
    contribute "Watches" to the flattened set exactly once, not error or
    duplicate."""
    values = category_filter_values("demo-fashion-in")
    assert values is not None
    assert values.count("Watches") == 1


def test_category_filter_values_is_sorted_and_deduplicated() -> None:
    values = category_filter_values("demo-fashion-in")
    assert values is not None
    assert list(values) == sorted(set(values))


def test_category_filter_values_none_for_unknown_tenant() -> None:
    """A real merchant beyond the three demo catalogs has never had
    `scripts/enrich_categories.py` run against it - the live chat path must
    degrade gracefully (no `category` enum constraint at all), not crash or
    build a filter that matches nothing. `None`, not an empty tuple, is the
    signal `app/llm/prompting.py` checks for this."""
    assert category_filter_values("some-future-merchant") is None


@pytest.mark.parametrize("tenant", sorted(TAXONOMY_BY_TENANT))
def test_category_filter_values_covers_every_taxonomy_piece(tenant: str) -> None:
    values = category_filter_values(tenant)
    assert values is not None
    expected: set[str] = set()
    for leaf in taxonomy_for(tenant):
        category, subcategory = leaf.split(" > ")
        expected.add(category)
        expected.add(subcategory)
    assert set(values) == expected
