"""The Product collection schema contract.

Schema mistakes are expensive: they surface as quietly bad relevance rather than as
errors, and fixing one means a full re-index. These tests pin the decisions that are
easy to break by accident.
"""

from __future__ import annotations

import pytest
from weaviate.classes.config import DataType, Tokenization

from app.models.product import Product
from app.retrieval.weaviate_client import PRODUCT_COLLECTION, product_properties


@pytest.fixture(scope="module")
def props() -> dict:
    return {p.name: p for p in product_properties()}


def test_collection_name_is_stable() -> None:
    """Renaming the collection orphans every stored vector."""
    assert PRODUCT_COLLECTION == "Product"


def test_no_duplicate_properties(props: dict) -> None:
    assert len(props) == len(product_properties())


def test_every_product_field_is_represented(props: dict) -> None:
    """Nothing in the canonical model may silently fail to reach the index."""
    # `attributes` is flattened into two columns; `updated_at` maps 1:1.
    represented = set(props) | {"attributes"}
    missing = set(Product.model_fields) - represented
    assert not missing, f"Product fields absent from the Weaviate schema: {missing}"


def test_sku_is_word_tokenized_and_searchable(props: dict) -> None:
    """This is the IDENTIFIER query class's whole contract.

    WORD tokenisation splits "DW-4402B" into ["dw", "4402b"]; a shopper typing the same
    string produces the same tokens, so BM25 scores an exact hit at low alpha. FIELD
    tokenisation would store one opaque token and break identifier search.
    """
    sku = props["sku"]
    assert sku.tokenization is Tokenization.WORD
    assert sku.indexSearchable is True
    assert sku.indexFilterable is True


def test_text_signal_fields_are_searchable(props: dict) -> None:
    """BM25 must see everything a shopper might phrase their query in terms of."""
    for name in ("title", "description", "brand", "category_path", "attributes_text"):
        assert props[name].indexSearchable is True, f"{name} is invisible to BM25"


def test_filterable_fields_support_structured_prefilters(props: dict) -> None:
    """SearchFilters (price/brand/category/stock) needs these filterable."""
    for name in ("price", "brand", "category_path", "in_stock", "rating"):
        assert props[name].indexFilterable is True, f"{name} cannot be pre-filtered"


def test_content_hash_is_filterable_but_not_searchable(props: dict) -> None:
    """Delta detection queries it; BM25 matching hex digests would only add noise."""
    ch = props["content_hash"]
    assert ch.indexFilterable is True
    assert ch.indexSearchable is False


def test_payload_fields_are_not_indexed(props: dict) -> None:
    """URLs and raw JSON are returned, never scored. Indexing them wastes memory."""
    for name in ("image_url", "product_url", "attributes_json"):
        p = props[name]
        assert p.indexSearchable is False, f"{name} should not be searchable"
        assert p.indexFilterable is False, f"{name} should not be filterable"


def test_numeric_and_temporal_types(props: dict) -> None:
    assert props["price"].dataType is DataType.NUMBER
    assert props["review_count"].dataType is DataType.INT
    assert props["in_stock"].dataType is DataType.BOOL
    assert props["updated_at"].dataType is DataType.DATE
    assert props["category_path"].dataType is DataType.TEXT_ARRAY


def test_currency_is_field_tokenized(props: dict) -> None:
    """'INR' is a code, not prose - it should match exactly or not at all."""
    assert props["currency"].tokenization is Tokenization.FIELD
    assert props["currency"].indexSearchable is False
