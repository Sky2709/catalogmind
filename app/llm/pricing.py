"""Bedrock $/token pricing, for per-merchant cost tracking (`app/llm/cost_tracking.py`).

No internal pricing doc exists anywhere in this repo or in the reference Bedrock user
guide's static text - the guide documents API shape, not live commercial rates. Rates
below are web-search-confirmed against AWS/Anthropic's own pricing pages (2026-08-22),
the same "best available source, dated, and flagged" discipline `alpha_router.py`'s
`PRIOR_ALPHA` already uses for a different kind of number that can't be pinned down
from first principles.

**`anthropic.claude-sonnet-5`'s rate is promotional and expires 2026-08-31** (then
$3/$10M input, $15/$10M output). If this module still reads $2/$10 well after that
date, it is stale - re-check against AWS's Bedrock pricing page before trusting a cost
figure this table produces for anything beyond order-of-magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal

_MTOK = Decimal(1_000_000)

# Standard Anthropic prompt-cache multipliers, applied to a model's *input* rate.
# This project only ever uses the default 5-minute cache TTL (never requests the
# 1-hour tier), so only that multiplier is modelled.
_CACHE_READ_MULTIPLIER = Decimal("0.1")
_CACHE_WRITE_MULTIPLIER = Decimal("1.25")


@dataclass(frozen=True)
class ModelPricing:
    input_per_mtok: Decimal
    output_per_mtok: Decimal


PRICING: dict[str, ModelPricing] = {
    "anthropic.claude-sonnet-5": ModelPricing(
        input_per_mtok=Decimal("2.00"), output_per_mtok=Decimal("10.00")
    ),
    "anthropic.claude-haiku-4-5": ModelPricing(
        input_per_mtok=Decimal("1.00"), output_per_mtok=Decimal("5.00")
    ),
}


def estimate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> Decimal:
    """Dollar cost of one Claude call from `response.usage`'s fields.

    Raises `KeyError` for an unpriced model rather than returning `Decimal(0)` -
    a silent $0 would corrupt the ledger every `app/config.py` `MODEL_REASONING`/
    `MODEL_FAST` change forgets to pair with a `PRICING` update, and that failure
    mode is exactly what `tests/unit/test_pricing.py`'s
    `test_settings_models_are_priced` exists to catch before it ships.
    """
    pricing = PRICING[model]
    cost = (
        Decimal(input_tokens) * pricing.input_per_mtok
        + Decimal(output_tokens) * pricing.output_per_mtok
        + Decimal(cache_read_tokens) * pricing.input_per_mtok * _CACHE_READ_MULTIPLIER
        + Decimal(cache_creation_tokens) * pricing.input_per_mtok * _CACHE_WRITE_MULTIPLIER
    ) / _MTOK
    return cost
