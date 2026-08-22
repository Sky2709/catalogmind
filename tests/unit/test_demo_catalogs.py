"""`MyntraFashionAdapter`'s `preprocess` hook - the one piece of real code behind
Myntra's escalation from a plain `ColumnMapping` to a full adapter (see
`app/ingestion/adapters/demo_catalogs.py`'s module docstring). Picking the wrong
image out of a `"~"`-delimited blob, or crashing on a row that lacks one, is exactly
the kind of thing that stays invisible until someone asks "why don't I see images."
"""

from __future__ import annotations

from app.ingestion.adapters.demo_catalogs import MyntraFashionAdapter
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
