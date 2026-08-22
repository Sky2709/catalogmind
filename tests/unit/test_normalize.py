"""Feed normalisation.

The cases here are drawn from what public retail datasets actually contain. Two of
them exist because the first implementation got them wrong: negative prices passed the
sign check (the regex discarded the minus before validation), and out-of-range ratings
were halved rather than clamped, silently turning a corrupt 6.0 into a believable 3.0.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.ingestion.normalize import (
    clean_text,
    detect_currency,
    fix_mojibake,
    normalise_sku,
    parse_bool,
    parse_int,
    parse_price,
    parse_rating,
    split_categories,
    strip_html,
)

# --- price -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1299", Decimal("1299")),
        ("1,299", Decimal("1299")),  # three trailing digits -> thousands
        ("1,29", Decimal("1.29")),  # two trailing digits -> decimal
        ("1,299.00", Decimal("1299.00")),
        ("1.299,00", Decimal("1299.00")),  # European
        ("1 299,50", Decimal("1299.50")),  # French, thin space
        ("Rs. 1,299.00", Decimal("1299.00")),
        ("$49.99", Decimal("49.99")),
        ("999/-", Decimal("999")),
        ("1,299 (25% off)", Decimal("1299")),
        ("MRP 1299", Decimal("1299")),
        (1299, Decimal("1299")),
        (12.5, Decimal("12.5")),
        (Decimal("7.25"), Decimal("7.25")),
    ],
)
def test_parse_price_accepts(raw: object, expected: Decimal) -> None:
    assert parse_price(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "   ",
        "N/A",
        "n/a",
        "Price not available",
        "TBD",
        "-",
        "---",
        "no digits here",
        True,  # a bool is not a price
    ],
)
def test_parse_price_rejects(raw: object) -> None:
    assert parse_price(raw) is None


@pytest.mark.parametrize("raw", ["-5", "Rs. -5", "-1,299.00", -5, -0.01, Decimal("-3")])
def test_negative_prices_are_rejected(raw: object) -> None:
    """Regression: the sign must survive tokenisation to be checked.

    The first implementation's number regex did not capture a leading minus, so "-5"
    matched as "5", passed the >= 0 guard, and a corrupt row poisoned price filters.
    """
    assert parse_price(raw) is None


def test_price_range_takes_the_first_value() -> None:
    """'1,299 - 1,499' is a range; the lower bound is the defensible reading."""
    assert parse_price("1,299 - 1,499") == Decimal("1299")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Rs. 1299", "INR"),
        ("₹1299", "INR"),
        ("INR 1299", "INR"),
        ("$49.99", "USD"),
        ("USD 49", "USD"),
        ("£10", "GBP"),
        ("€10", "EUR"),
        ("1299", None),
        ("", None),
        (None, None),
    ],
)
def test_detect_currency(raw: str | None, expected: str | None) -> None:
    assert detect_currency(raw) == expected


def test_longest_currency_token_wins() -> None:
    """'rs.' must beat 'rs', and 'us$' must beat '$'."""
    assert detect_currency("Rs. 100") == "INR"
    assert detect_currency("US$ 100") == "USD"


# --- text ------------------------------------------------------------------------


def test_strip_html_removes_tags() -> None:
    assert clean_text("<p>Nice <b>shirt</b></p>") == "Nice shirt"


def test_entities_are_resolved_before_tags() -> None:
    """Escaped markup is literal text, not a tag to strip."""
    assert clean_text("&lt;b&gt;not a tag&lt;/b&gt;") == "not a tag"
    assert clean_text("caf&eacute;  au   lait") == "café au lait"


def test_mojibake_round_trip_repair() -> None:
    """UTF-8 bytes misread as cp1252 are reversed, not pattern-matched."""
    assert fix_mojibake("Itâ€™s great") == "It’s great"
    assert fix_mojibake("â‚¹1,299") == "₹1,299"


def test_clean_text_leaves_clean_text_alone() -> None:
    """The mojibake guard must not corrupt text that was never broken."""
    for text in ("It's great", "Café au lait", "naïve", "₹1,299", "日本語"):
        assert fix_mojibake(text) == text


def test_mojibake_repair_is_safe_on_unencodable_text() -> None:
    """A marker plus a character cp1252 cannot represent must not raise."""
    assert fix_mojibake("â ₹ 日本") == "â ₹ 日本"


def test_nfkc_folds_fullwidth() -> None:
    """Full-width and ASCII forms must produce identical tokens."""
    assert clean_text("Ｎｉｋｅ") == "Nike"


def test_whitespace_is_collapsed() -> None:
    assert clean_text("  a \n\t b  ") == "a b"


def test_clean_text_handles_none_and_truncates() -> None:
    assert clean_text(None) == ""
    assert clean_text("abcdefghij", max_length=4) == "abcd"


def test_strip_html_is_not_a_sanitiser() -> None:
    """Documented behaviour: it extracts text, it does not neutralise scripts."""
    assert "alert" in strip_html("<script>alert(1)</script>")


# --- categories --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Clothing >> Men >> Shirts", ["Clothing", "Men", "Shirts"]),
        ("Clothing > Men > Shirts", ["Clothing", "Men", "Shirts"]),
        ("clothing/men/shirts", ["clothing", "men", "shirts"]),
        ("A | B | C", ["A", "B", "C"]),
        ("A -> B", ["A", "B"]),
        (["A ", " B"], ["A", "B"]),
        ("Solo", ["Solo"]),
        ("", []),
        (None, []),
        ("A >> >> B", ["A", "B"]),  # empty segments dropped
    ],
)
def test_split_categories(raw: object, expected: list[str]) -> None:
    assert split_categories(raw) == expected


def test_double_angle_is_not_split_twice() -> None:
    """'>>' must match before '>', or every segment gains an empty neighbour."""
    assert split_categories("A >> B") == ["A", "B"]


# --- booleans ----------------------------------------------------------------------


@pytest.mark.parametrize("raw", ["Y", "yes", "TRUE", "1", "in stock", "Available", 1, True])
def test_parse_bool_true(raw: object) -> None:
    assert parse_bool(raw) is True


@pytest.mark.parametrize("raw", ["N", "no", "FALSE", "0", "out of stock", "sold out", 0, False])
def test_parse_bool_false(raw: object) -> None:
    assert parse_bool(raw) is False


def test_parse_bool_unknown_returns_default() -> None:
    assert parse_bool("maybe") is None
    assert parse_bool("maybe", default=True) is True
    assert parse_bool(None, default=False) is False


# --- ratings -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "scale", "expected"),
    [
        ("4.5", 5.0, 4.5),
        ("0", 5.0, 0.0),
        ("9/10", 5.0, 4.5),  # value declares its own denominator
        ("4/5", 5.0, 4.0),
        ("8", 10.0, 4.0),  # adapter declares the source scale
        ("3,5", 5.0, 3.5),  # comma decimal
    ],
)
def test_parse_rating(raw: str, scale: float, expected: float) -> None:
    assert parse_rating(raw, source_scale=scale) == pytest.approx(expected)


def test_out_of_range_rating_is_clamped_not_rescaled() -> None:
    """Regression: 6.0 on a five-point scale is corrupt, and must clamp to 5.0.

    The first implementation halved anything above the max, turning 6.0 into 3.0 - a
    value that looks entirely plausible and is completely fabricated.
    """
    assert parse_rating("6.0", source_scale=5.0) == 5.0
    assert parse_rating("-1", source_scale=5.0) == 0.0


@pytest.mark.parametrize("raw", ["abc", None, "", True])
def test_parse_rating_rejects(raw: object) -> None:
    assert parse_rating(raw) is None


def test_rating_output_is_always_on_a_five_point_scale() -> None:
    for raw, scale in [("10", 10.0), ("5", 5.0), ("100", 100.0)]:
        assert parse_rating(raw, source_scale=scale) == 5.0


# --- ints and skus -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,234 reviews", 1234),
        ("12", 12),
        (7, 7),
        (7.9, 7),
        ("none", None),
        (None, None),
        (True, None),
    ],
)
def test_parse_int(raw: object, expected: int | None) -> None:
    assert parse_int(raw) == expected


def test_normalise_sku_preserves_case() -> None:
    """SKUs are identifiers; uppercasing could merge two genuinely distinct products."""
    assert normalise_sku("  ab-1  ") == "ab-1"
    assert normalise_sku("AB-1") == "AB-1"


def test_normalise_sku_rejects_empty() -> None:
    assert normalise_sku("") is None
    assert normalise_sku("   ") is None
    assert normalise_sku(None) is None
