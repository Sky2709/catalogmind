"""Field-level normalisation for merchant feeds.

Real catalogs are hostile. Across the public datasets this project ingests you will
find prices written as ``"Rs. 1,299.00"``, ``"INR 1299"``, ``"1,299"``,
``"1,299 (25% off)"`` and ``"Price not available"``; descriptions containing raw HTML
and Windows-1252 mojibake; categories as ``"Clothing >> Men >> Shirts"`` or
``"clothing/men/shirts"``; stock flags as ``"Y"``, ``"in stock"``, ``"1"``, ``"TRUE"``
and ``"available"``.

Every function here is total: it returns ``None`` rather than raising, because one
malformed row must never abort a 50,000-row ingestion. The row-level decision about
whether a ``None`` is fatal belongs to the pipeline, which records it as an
``IngestionError`` the merchant can act on.

This module is deliberately dependency-free and pure, so it is exhaustively testable
without a database, a network, or a running stack.
"""

from __future__ import annotations

import html
import re
import unicodedata
from decimal import Decimal, InvalidOperation

# --- currency ------------------------------------------------------------------

# Symbol/prefix -> ISO 4217. Matched longest-first so "rs." beats "rs".
_CURRENCY_TOKENS: dict[str, str] = {
    "₹": "INR",  # rupee sign
    "rs.": "INR",
    "rs": "INR",
    "inr": "INR",
    "us$": "USD",
    "usd": "USD",
    "$": "USD",
    "£": "GBP",  # pound sign
    "gbp": "GBP",
    "€": "EUR",  # euro sign
    "eur": "EUR",
    "¥": "JPY",  # yen sign
    "jpy": "JPY",
    "aed": "AED",
}

# Trailing noise that follows a price in scraped feeds: "(25% off)", "onwards", "/-"
_PRICE_NOISE = re.compile(
    r"\(.*?\)|\b(?:onwards?|only|approx\.?|starting(?:\s+at)?|mrp|incl\.?|excl\.?)\b|/-",
    re.IGNORECASE,
)

_NOT_A_PRICE = re.compile(
    r"^\s*(?:n/?a|nil|none|null|-+|price\s*not\s*available|out\s*of\s*stock|tbd|\?+)\s*$",
    re.IGNORECASE,
)

# A number with optional thousands separators and optional decimals.
# Handles 1,299.00 - 1299 - 1.299,00 (European) - 1 299,00 (French) - -5
# The sign MUST be captured: without it '-5' matches as '5', and a corrupt
# negative price silently passes the >= 0 guard in parse_price.
_NUMBER = re.compile(r"-?\d[\d\s.,' ]*\d|-?\d")


def detect_currency(value: str | None) -> str | None:
    """ISO 4217 code implied by a price string, if any.

    ``$`` is genuinely ambiguous (USD/CAD/AUD/SGD...). We resolve it to USD and note
    the assumption rather than silently dropping the field; an adapter should override
    when the source's currency is known out of band, which is the reliable path.
    """
    if not value:
        return None
    lowered = value.strip().lower()
    for token in sorted(_CURRENCY_TOKENS, key=len, reverse=True):
        if token in lowered:
            return _CURRENCY_TOKENS[token]
    return None


def _normalise_number_text(text: str) -> str | None:
    """Resolve thousands/decimal separators into a plain decimal string.

    The ambiguous case is a single separator: ``1,299`` is 1299 in India/US but 1.299
    in Europe. We use the widely-used heuristic - if exactly three digits follow the
    final separator and there is no other separator, treat it as a thousands
    separator. So ``1,29`` reads as 1.29 and ``1,299`` as 1299.
    """
    cleaned = text.replace(" ", "").replace(" ", "").replace("'", "")
    if not cleaned:
        return None

    last_comma = cleaned.rfind(",")
    last_dot = cleaned.rfind(".")

    if last_comma == -1 and last_dot == -1:
        return cleaned

    if last_comma > last_dot:
        decimal_sep, thousands_sep = ",", "."
        sep_index = last_comma
    else:
        decimal_sep, thousands_sep = ".", ","
        sep_index = last_dot

    tail = cleaned[sep_index + 1 :]
    if len(tail) == 3 and cleaned.count(decimal_sep) == 1 and thousands_sep not in cleaned:
        # Exactly three trailing digits and no other separator: thousands, not decimals.
        return cleaned.replace(decimal_sep, "")

    return cleaned.replace(thousands_sep, "").replace(decimal_sep, ".")


def parse_price(value: object) -> Decimal | None:
    """Best-effort price extraction. Returns None for anything unusable.

    Negative values are rejected: a negative price is corrupt data, not a discount,
    and letting one through would poison range filters.
    """
    if value is None:
        return None
    if isinstance(value, Decimal):
        return value if value >= 0 else None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            dec = Decimal(str(value))
        except InvalidOperation:
            return None
        return dec if dec >= 0 else None

    text = str(value).strip()
    if not text or _NOT_A_PRICE.match(text):
        return None

    text = _PRICE_NOISE.sub(" ", text)
    match = _NUMBER.search(text)
    if not match:
        return None

    normalised = _normalise_number_text(match.group(0))
    if normalised is None:
        return None

    try:
        dec = Decimal(normalised)
    except InvalidOperation:
        return None
    return dec if dec >= 0 else None


# --- text ------------------------------------------------------------------------

_TAG = re.compile(r"<[^>]+>")
_WHITESPACE = re.compile(r"\s+")

# Characters that only appear when UTF-8 bytes were decoded as cp1252/latin-1.
_MOJIBAKE_MARKERS = ("Ã", "â", "Â")


def strip_html(value: str) -> str:
    """Remove tags and resolve entities. Not a sanitiser - a text extractor."""
    # Entities first, so "&lt;b&gt;" becomes literal text rather than a stripped tag.
    text = html.unescape(value)
    return _TAG.sub(" ", text)


def fix_mojibake(value: str) -> str:
    """Repair UTF-8 bytes that were decoded as cp1252 and stored that way.

    The classic scraped-retail corruption: an apostrophe becomes "a-euro-trademark".
    Rather than maintain a lookup table of corrupted sequences, reverse the mistake
    directly - encode back to cp1252 bytes, then decode as UTF-8 as originally
    intended. If that round-trip fails, the text was not mojibake; return it unchanged.

    Guarded by a marker check so clean text (the overwhelming majority of rows) skips
    two encode/decode passes.
    """
    if not any(marker in value for marker in _MOJIBAKE_MARKERS):
        return value
    try:
        return value.encode("cp1252").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def clean_text(value: object, *, max_length: int | None = None) -> str:
    """Normalise arbitrary feed text into something safe to index.

    Order matters: entities and tags come out before whitespace collapsing, or the
    space introduced in place of a tag would not be collapsed.
    """
    if value is None:
        return ""
    text = fix_mojibake(str(value))
    text = strip_html(text)
    # NFKC folds full-width and compatibility characters to ASCII equivalents, so
    # full-width "Nike" and plain "Nike" produce the same tokens.
    text = unicodedata.normalize("NFKC", text)
    text = _WHITESPACE.sub(" ", text).strip()
    if max_length is not None and len(text) > max_length:
        text = text[:max_length].rstrip()
    return text


# --- categories --------------------------------------------------------------------

# Longest first so ">>" is not split twice by ">".
_CATEGORY_SEPARATORS = (">>", "->", "»", ">", "|", "/", "\\")


def split_categories(value: object) -> list[str]:
    """Split a category path on whichever separator the feed happens to use."""
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [c for c in (clean_text(v) for v in value) if c]

    text = clean_text(value)
    if not text:
        return []

    for sep in _CATEGORY_SEPARATORS:
        if sep in text:
            return [p for p in (clean_text(part) for part in text.split(sep)) if p]
    return [text]


# --- booleans and numbers -----------------------------------------------------------

_TRUE = {"1", "true", "t", "yes", "y", "in stock", "instock", "available", "in_stock"}
_FALSE = {"0", "false", "f", "no", "n", "out of stock", "outofstock", "unavailable", "sold out"}


def parse_bool(value: object, *, default: bool | None = None) -> bool | None:
    """Interpret the many ways feeds spell a boolean."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = clean_text(value).lower()
    if not text:
        return default
    if text in _TRUE:
        return True
    if text in _FALSE:
        return False
    return default


def parse_rating(value: object, *, source_scale: float = 5.0) -> float | None:
    """Rating rescaled to 0-5, or None if unparseable.

    The scale is never guessed from the magnitude of an individual value. Halving
    anything above 5 would map a corrupt ``6.0`` on a five-point scale to a
    plausible-looking ``3.0`` — inventing data instead of flagging it, and doing so
    invisibly. Out-of-range values are clamped, which is the honest degradation.

    Scale comes from one of two places, both authoritative:

    * The value itself, when the feed declares it — ``"9/10"`` carries its own
      denominator and is read as 4.5.
    * ``source_scale``, supplied by the adapter, which knows what the source publishes.
    """
    if value is None or isinstance(value, bool):
        return None

    text = str(value).strip()
    scale = source_scale

    # "9/10" declares its own denominator; trust the data over the caller's default.
    if "/" in text:
        numerator, _, denominator = text.partition("/")
        try:
            declared = float(denominator.strip().replace(",", "."))
            if declared > 0:
                scale = declared
                text = numerator
        except ValueError:
            text = numerator

    try:
        rating = float(text.strip().replace(",", "."))
    except ValueError:
        return None
    if rating != rating:  # NaN
        return None
    if scale <= 0:
        return None

    clamped = max(0.0, min(rating, scale))
    return round(clamped * 5.0 / scale, 3)


def parse_int(value: object) -> int | None:
    """Integer from noisy text: '1,234 reviews' -> 1234."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if value == value else None

    match = _NUMBER.search(str(value))
    if not match:
        return None
    normalised = _normalise_number_text(match.group(0))
    if normalised is None:
        return None
    try:
        return int(Decimal(normalised))
    except (InvalidOperation, ValueError):
        return None


def normalise_sku(value: object) -> str | None:
    """Canonical SKU: trimmed, internal whitespace collapsed, case preserved.

    Case is preserved deliberately. SKUs are identifiers and some merchants really do
    distinguish ``ab-1`` from ``AB-1``; uppercasing would silently merge two products.
    Weaviate's WORD tokenisation lowercases at index time anyway, so keyword search
    stays case-insensitive regardless.
    """
    if value is None:
        return None
    return clean_text(value) or None
