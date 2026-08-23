"""Locked per-tenant category taxonomy (`app/ingestion/taxonomy.py`) and the
flattened filter-value view of it the chat agent's tool schema uses
(`app/llm/prompting.py::_price_brand_filter_properties`).
"""

from __future__ import annotations

import json
import os

import pytest

from app.ingestion import taxonomy as taxonomy_module
from app.ingestion.taxonomy import (
    TAXONOMY_BY_TENANT,
    UNCATEGORIZED,
    category_filter_values,
    load_enrichment_map,
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


# --- load_enrichment_map: mtime-aware reload, not a bare lru_cache -------------


def _write_jsonl(path, classifications: dict[str, str]) -> None:
    path.write_text(json.dumps({"classifications": classifications}) + "\n", encoding="utf-8")


def test_load_enrichment_map_missing_file_returns_empty_dict(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(taxonomy_module, "ENRICHMENT_DIR", tmp_path)
    assert load_enrichment_map("no-such-tenant") == {}


def test_load_enrichment_map_reflects_a_changed_file_without_restarting(
    tmp_path, monkeypatch
) -> None:
    """The real, previously-unfixed gap this pins: a bare `lru_cache` never
    notices the underlying file changed - re-running the enrichment script for
    an already-cached tenant left a live process silently serving the old map.
    `os.utime` forces a real mtime change rather than relying on wall-clock
    timing between the two writes."""
    monkeypatch.setattr(taxonomy_module, "ENRICHMENT_DIR", tmp_path)
    monkeypatch.setattr(taxonomy_module, "_enrichment_cache", {})
    path = tmp_path / "demo-test-tenant.jsonl"

    _write_jsonl(path, {"Old Title": "Old Category > Old Subcategory"})
    os.utime(path, (1_000_000, 1_000_000))
    assert load_enrichment_map("demo-test-tenant") == {
        "Old Title": "Old Category > Old Subcategory"
    }

    _write_jsonl(path, {"New Title": "New Category > New Subcategory"})
    os.utime(path, (2_000_000, 2_000_000))
    assert load_enrichment_map("demo-test-tenant") == {
        "New Title": "New Category > New Subcategory"
    }


def test_load_enrichment_map_reuses_cache_when_file_is_unchanged(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(taxonomy_module, "ENRICHMENT_DIR", tmp_path)
    monkeypatch.setattr(taxonomy_module, "_enrichment_cache", {})
    path = tmp_path / "demo-test-tenant.jsonl"
    _write_jsonl(path, {"Title": "Category > Subcategory"})

    first = load_enrichment_map("demo-test-tenant")
    second = load_enrichment_map("demo-test-tenant")
    assert first is second  # same object - proves the second call didn't re-read the file


def test_load_enrichment_map_scales_past_three_tenants(tmp_path, monkeypatch) -> None:
    """The bare `lru_cache(maxsize=3)` this replaced silently thrashed past
    exactly 3 live tenants - this project's own three demo catalogs, which is
    exactly why it never surfaced on its own. A 4th and 5th tenant must both
    still resolve correctly, with no cap to size."""
    monkeypatch.setattr(taxonomy_module, "ENRICHMENT_DIR", tmp_path)
    monkeypatch.setattr(taxonomy_module, "_enrichment_cache", {})
    tenants = [f"tenant-{i}" for i in range(5)]
    for i, tenant in enumerate(tenants):
        _write_jsonl(tmp_path / f"{tenant}.jsonl", {"Title": f"Category {i} > Sub {i}"})

    for i, tenant in enumerate(tenants):
        assert load_enrichment_map(tenant) == {"Title": f"Category {i} > Sub {i}"}
    # Re-check the first tenant again - still correct, not evicted-and-wrong.
    assert load_enrichment_map(tenants[0]) == {"Title": "Category 0 > Sub 0"}
