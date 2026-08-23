"""The chat agent's tool schemas (`app/llm/prompting.py`) - specifically the
`category` filter's per-tenant enum, added 2026-08-23 so Claude has a reliable,
machine-checkable list of valid category/subcategory values instead of having
to guess a string that happens to match `category_path`'s stored content.
"""

from __future__ import annotations

from typing import Any, cast

from app.llm.prompting import get_catalog_stats_tool, search_catalog_tool


def _category_schema(tool: Any) -> dict[str, Any]:
    return cast(dict[str, Any], tool["input_schema"]["properties"]["category"])


def test_search_catalog_category_has_enum_for_a_known_tenant() -> None:
    schema = _category_schema(search_catalog_tool("demo-fashion-in"))
    assert "enum" in schema
    assert "Skincare" in schema["enum"]
    assert "Beauty & Personal Care" in schema["enum"]


def test_get_catalog_stats_category_has_enum_for_a_known_tenant() -> None:
    schema = _category_schema(get_catalog_stats_tool("demo-fashion-in"))
    assert "enum" in schema
    assert "Skincare" in schema["enum"]


def test_category_enum_is_tenant_specific() -> None:
    fashion = set(_category_schema(search_catalog_tool("demo-fashion-in"))["enum"])
    electronics = set(_category_schema(search_catalog_tool("demo-electronics-in"))["enum"])
    assert "Skincare" in fashion
    assert "Skincare" not in electronics
    assert "Smartphones" in electronics
    assert "Smartphones" not in fashion


def test_category_has_no_enum_for_a_tenant_without_a_taxonomy() -> None:
    """The live chat path must degrade gracefully for any real merchant beyond
    the three demo catalogs - a plain, unconstrained string field, exactly its
    behaviour before this enum existed, not a crash and not an enum that would
    make `category` an unsatisfiable filter."""
    schema = _category_schema(search_catalog_tool("some-future-merchant"))
    assert "enum" not in schema
    assert schema["type"] == "string"


def test_other_filter_properties_unaffected_by_the_category_enum() -> None:
    tool: Any = search_catalog_tool("demo-fashion-in")
    props = tool["input_schema"]["properties"]
    assert props["brand"] == {"type": "string"}
    assert props["min_price"]["type"] == "number"
    assert props["in_stock_only"]["default"] is False


def test_search_catalog_tool_still_requires_query() -> None:
    tool = search_catalog_tool("demo-fashion-in")
    assert tool["input_schema"]["required"] == ["query"]


def test_get_catalog_stats_tool_still_has_no_required_fields() -> None:
    tool = get_catalog_stats_tool("demo-fashion-in")
    assert tool["input_schema"]["required"] == []
