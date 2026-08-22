"""The pure half of the storage round-trip check: diffing a re-derived expected
properties dict against what a tenant actually returned. No Weaviate needed - the
I/O-bound fetch (`fetch_all_stored_properties`) is exercised for real in
`tests/integration/test_ingestion.py`-style coverage instead.
"""

from __future__ import annotations

from app.models.product import Product
from scripts.ingestion_quality_report import diff_stored_properties


def _product(**overrides: object) -> Product:
    defaults: dict[str, object] = {"sku": "A1", "title": "Widget", "price": "9.99"}
    defaults.update(overrides)
    return Product(**defaults)


def test_missing_from_storage_is_one_clear_issue() -> None:
    issues = diff_stored_properties(_product(), stored=None)
    assert issues == ["A1: not found in Weaviate"]


def test_matching_storage_has_no_issues() -> None:
    product = _product()
    from app.retrieval import weaviate_client as wv

    stored = wv.properties_from_product(product)
    assert diff_stored_properties(product, stored) == []


def test_updated_at_mismatch_is_ignored() -> None:
    """A timestamp is expected to differ run to run - it must not be treated as drift."""
    product = _product()
    from app.retrieval import weaviate_client as wv

    stored = wv.properties_from_product(product)
    stored["updated_at"] = "2000-01-01T00:00:00Z"
    assert diff_stored_properties(product, stored) == []


def test_real_field_drift_is_reported() -> None:
    product = _product(title="New Title")
    from app.retrieval import weaviate_client as wv

    stored = wv.properties_from_product(product)
    stored["title"] = "Stale Title"  # simulate a stale/partial write
    issues = diff_stored_properties(product, stored)
    assert len(issues) == 1
    assert "title" in issues[0]
    assert "New Title" in issues[0]
    assert "Stale Title" in issues[0]


def test_float_comparison_tolerates_representation_noise() -> None:
    """Prices round-trip through Weaviate's own float representation - a comparison
    that requires bit-exact equality would false-positive on harmless float noise."""
    product = _product(price="9.99")
    from app.retrieval import weaviate_client as wv

    stored = wv.properties_from_product(product)
    stored["price"] = 9.990000000001  # representation noise, not real drift
    assert diff_stored_properties(product, stored) == []
