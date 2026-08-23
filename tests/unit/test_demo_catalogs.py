"""Adapters for the three demo catalogs (`app/ingestion/adapters/demo_catalogs.py`).

Started as `MyntraFashionAdapter`-only coverage for its `preprocess` hook -
picking the wrong image out of a `"~"`-delimited blob, or crashing on a row
that lacks one, is exactly the kind of thing that stays invisible until
someone asks "why don't I see images." Grew to cover all three adapters'
category-enrichment wiring (`scripts/enrich_categories.py`'s offline
classification, looked up by title or ASIN depending on the catalog) once
that landed - see PROGRESS.md.
"""

from __future__ import annotations

from app.ingestion.adapters import demo_catalogs
from app.ingestion.adapters.demo_catalogs import (
    AmazonElectronicsAdapter,
    MyntraFashionAdapter,
    SheinHomeGoodsAdapter,
)
from app.ingestion.taxonomy import UNCATEGORIZED
from app.models.product import IngestionError, Product

ROW = {
    "sku": "10015819",
    "name": "DKNY Unisex Black Trolley Bag",
    "mpn": "10015819",
    "price": "9583",
    "in_stock": "TRUE",
    "currency": "INR",
    "brand": "DKNY",
    "description": "A sturdy trolley bag.",
    "images": "",
    "gender": "Unisex",
}


def _parse(images: str) -> Product:
    row = {**ROW, "images": images}
    result = MyntraFashionAdapter().parse_row(row, row_number=1)
    assert isinstance(result, Product), result
    return result


def test_first_image_taken_from_delimited_list() -> None:
    product = _parse("http://a.jpg ~ http://b.jpg ~ http://c.jpg")
    assert product.image_url == "http://a.jpg"


def test_single_image_with_no_delimiter() -> None:
    product = _parse("http://only.jpg")
    assert product.image_url == "http://only.jpg"


def test_missing_images_value_yields_no_image_not_a_crash() -> None:
    row = {k: v for k, v in ROW.items() if k != "images"}
    result = MyntraFashionAdapter().parse_row(row, row_number=1)
    assert isinstance(result, Product), result
    assert result.image_url is None


def test_empty_images_value_yields_no_image() -> None:
    assert _parse("").image_url is None


def test_whitespace_and_bare_delimiters_yield_no_image() -> None:
    assert _parse("   ").image_url is None
    assert _parse("~~~").image_url is None


def test_ragged_whitespace_around_delimiter_is_trimmed() -> None:
    product = _parse("  http://a.jpg~http://b.jpg  ")
    assert product.image_url == "http://a.jpg"


def test_other_fields_still_map_correctly_alongside_the_image_fix() -> None:
    product = _parse("http://a.jpg ~ http://b.jpg")
    assert product.sku == "10015819"
    assert product.title == "DKNY Unisex Black Trolley Bag"
    assert product.brand == "DKNY"
    assert product.currency == "INR"
    assert product.in_stock is True
    assert product.attributes == {"gender": "Unisex"}


def test_row_with_no_sku_still_fails_like_before() -> None:
    row = {**ROW, "sku": "", "images": "http://a.jpg"}
    result = MyntraFashionAdapter().parse_row(row, row_number=7)
    assert isinstance(result, IngestionError)


# --- category enrichment: fashion (looked up by title) ------------------------


def test_fashion_category_comes_from_enrichment_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        demo_catalogs,
        "load_enrichment_map",
        lambda tenant: {"DKNY Unisex Black Trolley Bag": "Bags & Luggage > Luggage & Trolleys"},
    )
    product = _parse("http://a.jpg")
    assert product.category_path == ["Bags & Luggage", "Luggage & Trolleys"]


def test_fashion_category_falls_back_to_uncategorized_when_not_in_enrichment(
    monkeypatch,
) -> None:
    monkeypatch.setattr(demo_catalogs, "load_enrichment_map", lambda tenant: {})
    product = _parse("http://a.jpg")
    assert product.category_path == UNCATEGORIZED.split(" > ")


# --- category enrichment: electronics (looked up by ASIN, not title) ----------

ELECTRONICS_ROW = {
    "name": "OnePlus Bullets Z2 Bluetooth Wireless in Ear Earphones",
    "main_category": "tv, audio & cameras",
    "sub_category": "All Electronics",
    "link": "https://www.amazon.in/dp/B08DR5X5XR/ref=something",
    "discount_price": "₹1,999",
    "actual_price": "₹2,299",
    "ratings": "4.2",
    "no_of_ratings": "90,304",
    "image": "http://img.jpg",
}


def test_electronics_category_looked_up_by_asin_not_title(monkeypatch) -> None:
    monkeypatch.setattr(
        demo_catalogs,
        "load_enrichment_map",
        lambda tenant: {
            "B08DR5X5XR": "Audio > Headphones & Earbuds",
            # A different key under the (truncated, unreliable) title text must
            # never be consulted - electronics is deliberately keyed by ASIN.
            ELECTRONICS_ROW["name"]: "Other > Uncategorized",
        },
    )
    result = AmazonElectronicsAdapter().parse_row(ELECTRONICS_ROW, row_number=1)
    assert isinstance(result, Product), result
    assert result.category_path == ["Audio", "Headphones & Earbuds"]


def test_electronics_category_falls_back_when_asin_not_in_enrichment(monkeypatch) -> None:
    monkeypatch.setattr(demo_catalogs, "load_enrichment_map", lambda tenant: {})
    result = AmazonElectronicsAdapter().parse_row(ELECTRONICS_ROW, row_number=1)
    assert isinstance(result, Product), result
    assert result.category_path == UNCATEGORIZED.split(" > ")


# --- category enrichment: home-goods (must never affect the SKU digest) -------

HOME_GOODS_ROW = {
    "goods-title-link": "1pc Multifunctional 9-hole Hanger For Home Wardrobe",
    "rank-sub": "in Closet Organizers",
    "price": "$12.99",
}


def test_home_goods_category_comes_from_enrichment_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        demo_catalogs,
        "load_enrichment_map",
        lambda tenant: {
            HOME_GOODS_ROW["goods-title-link"]: "Storage & Organization > Closet & Wardrobe"
        },
    )
    result = SheinHomeGoodsAdapter(default_category="Home Goods").parse_row(
        HOME_GOODS_ROW, row_number=1
    )
    assert isinstance(result, Product), result
    assert result.category_path == ["Storage & Organization", "Closet & Wardrobe"]


def test_home_goods_sku_is_unaffected_by_the_enrichment_lookup(monkeypatch) -> None:
    """The real bug this pins: `_sku` is a hash of the `rank-sub`-derived
    `_category` (shipped earlier to stop cross-category title collisions from
    silently overwriting each other) - the *separate*, LLM-derived
    `_display_category` used for search/filtering must never feed that hash,
    or every home-goods SKU would change a second time depending on which
    enrichment happened to be loaded when a row was ingested."""

    def _sku_for(enrichment: dict[str, str]) -> str:
        monkeypatch.setattr(demo_catalogs, "load_enrichment_map", lambda tenant: enrichment)
        result = SheinHomeGoodsAdapter(default_category="Home Goods").parse_row(
            HOME_GOODS_ROW, row_number=1
        )
        assert isinstance(result, Product), result
        return result.sku

    sku_with_no_enrichment = _sku_for({})
    sku_with_enrichment = _sku_for(
        {HOME_GOODS_ROW["goods-title-link"]: "Storage & Organization > Closet & Wardrobe"}
    )
    assert sku_with_no_enrichment == sku_with_enrichment
