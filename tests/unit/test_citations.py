"""The post-hoc hallucinated-citation detector - the second, honest-limitation layer
of grounding described in `app/llm/citations.py`'s module docstring.

Retired this session (2026-08-22): guessing which substrings in free prose are a
SKU citation (a whole file of carve-outs for model numbers, barcodes, "3-in-1"
idioms, price ranges, and the shopper's own echoed words). Claude now declares a
citation explicitly via `[[SKU:...]]` (`app/llm/markers.py`), so detection is exact
set membership against what was actually retrieved - these tests are correspondingly
much smaller than the file they replace.
"""

from __future__ import annotations

from app.llm.citations import find_hallucinated_citations

RETRIEVED = [
    {"sku": "BOOT-WP-10", "title": "Waterproof Hiking Boot", "brand": "TrailMax"},
    {"sku": "SHIRT-CTN-M", "title": "Cotton Crew Shirt, Medium", "brand": None},
    {"sku": "10015819", "title": "Raymond Men Maroon Slim Fit Formal Shirt", "brand": "Raymond"},
]


def test_a_marker_citing_a_retrieved_sku_flags_nothing() -> None:
    answer = "I'd recommend the [[SKU:BOOT-WP-10]] - it's waterproof and well reviewed."
    assert find_hallucinated_citations(answer, RETRIEVED) == []


def test_a_marker_citing_an_unretrieved_sku_is_flagged() -> None:
    answer = "The [[SKU:BOOT-WP-10]] is great, or try the [[SKU:JACKET-RAIN-99]] instead."
    assert find_hallucinated_citations(answer, RETRIEVED) == ["JACKET-RAIN-99"]


def test_bare_numeric_sku_marker_is_recognised() -> None:
    """Mirrors fashion's Myntra-style bare-numeric SKUs (no letters at all)."""
    answer = "[[SKU:10015819]] matches, but [[SKU:99999999]] does not exist in this catalog."
    assert find_hallucinated_citations(answer, RETRIEVED) == ["99999999"]


def test_text_outside_any_marker_is_never_scanned() -> None:
    """The whole point of the marker protocol: a SKU-shaped-looking number or
    hyphenated word in ordinary prose - the entire false-positive surface the old
    regex needed a dozen carve-outs for - is never even considered unless it's
    inside a [[SKU:...]] marker."""
    answer = (
        "This boot is a well-known, long-lasting brand. It's a 3-in-1 cleaning "
        "kit, ships as a 2-Pack, comes in size 10, and the barcode is 8903705152451."
    )
    assert find_hallucinated_citations(answer, RETRIEVED) == []


def test_matching_is_case_insensitive() -> None:
    answer = "Try the [[SKU:boot-wp-10]], it's a great match."
    assert find_hallucinated_citations(answer, RETRIEVED) == []


def test_a_hallucinated_sku_repeated_is_reported_once() -> None:
    answer = "Consider [[SKU:JACKET-RAIN-99]]. Again, [[SKU:JACKET-RAIN-99]] is a solid pick."
    assert find_hallucinated_citations(answer, RETRIEVED) == ["JACKET-RAIN-99"]


def test_no_markers_at_all_flags_nothing() -> None:
    answer = "I found a few options that might work for you."
    assert find_hallucinated_citations(answer, RETRIEVED) == []
