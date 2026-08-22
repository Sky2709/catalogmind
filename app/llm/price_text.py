"""Extracting a price figure from generated prose - used by
`eval/generation_metrics.py::find_price_mismatch` (checking a cited product's own
price against prose, an eval-only check with no production equivalent).

`app/llm/claims.py::find_stat_claim_mismatch` used to share this module too (its
own `get_catalog_stats`-claim check), but retired its regex-extraction step this
session (2026-08-22) in favour of the `[[STAT:N]]` marker protocol
(`app/llm/markers.py`) - a production claim is now read from an explicit marker,
not guessed at from a price-shaped substring in prose. Kept as its own module
rather than duplicated, in case a second prose-price consumer shows up again.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

# One currency-marked price mention: a symbol/code before or after a digit group,
# optionally comma-grouped and/or decimal ("₹1,299", "$45.99", "1299 INR", "Rs. 500").
PRICE_MENTION = re.compile(
    r"(?:₹|\$|Rs\.?|INR|USD)\s*([\d,]+(?:\.\d+)?)|([\d,]+(?:\.\d+)?)\s*(?:INR|USD)",
    re.IGNORECASE,
)


def extract_price_mentions(text: str) -> list[Decimal]:
    values: list[Decimal] = []
    for match in PRICE_MENTION.finditer(text):
        raw = (match.group(1) or match.group(2) or "").replace(",", "")
        if not raw:
            continue
        try:
            values.append(Decimal(raw))
        except InvalidOperation:
            continue
    return values
