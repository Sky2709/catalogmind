"""`app/llm/price_text.py` - the price-mention regex shared by production
(`app/llm/claims.py`) and eval (`eval/generation_metrics.py`). Mirrors the cases
`eval/generation_metrics.py`'s own tests already covered before the extraction.
"""

from __future__ import annotations

from decimal import Decimal

from app.llm.price_text import extract_price_mentions


def test_extracts_a_rupee_symbol_price() -> None:
    assert extract_price_mentions("It's available for ₹1,299.") == [Decimal("1299")]


def test_extracts_a_dollar_price() -> None:
    assert extract_price_mentions("This one is $45.99.") == [Decimal("45.99")]


def test_extracts_a_currency_code_suffix() -> None:
    assert extract_price_mentions("Priced at 1299 INR.") == [Decimal("1299")]


def test_extracts_multiple_mentions_in_order() -> None:
    assert extract_price_mentions("₹1,299 or $45.99") == [Decimal("1299"), Decimal("45.99")]


def test_no_mention_returns_empty_list() -> None:
    assert extract_price_mentions("A great waterproof boot.") == []


def test_bare_number_without_currency_marker_is_not_a_price_mention() -> None:
    assert extract_price_mentions("It comes in size 10, ships in 2 days.") == []
