"""Cost math for Day 6's per-merchant ledger.

A wrong $/token here silently mis-bills every merchant's usage summary - these
run as plain unit tests (no I/O) because the arithmetic itself is what needs
pinning, not any real Bedrock call.
"""

from __future__ import annotations

from decimal import Decimal

import pytest

from app.config import get_settings
from app.llm.pricing import PRICING, estimate_cost_usd


def test_zero_tokens_cost_nothing() -> None:
    assert estimate_cost_usd("anthropic.claude-haiku-4-5", 0, 0) == Decimal(0)


def test_input_and_output_priced_independently() -> None:
    # Haiku: $1/$5 per million. 1M input alone -> $1; 1M output alone -> $5.
    assert estimate_cost_usd("anthropic.claude-haiku-4-5", 1_000_000, 0) == Decimal("1.00000000")
    assert estimate_cost_usd("anthropic.claude-haiku-4-5", 0, 1_000_000) == Decimal("5.00000000")


def test_cache_read_is_a_tenth_of_the_input_rate() -> None:
    # 1M cache-read tokens at Haiku's $1/M input rate * 0.1 = $0.10, on top of the
    # baseline $1 for the (separate) 1M `input_tokens`.
    cost = estimate_cost_usd(
        "anthropic.claude-haiku-4-5", 1_000_000, 0, cache_read_tokens=1_000_000
    )
    assert cost == Decimal("1.10000000")


def test_cache_creation_is_1_25x_the_input_rate() -> None:
    cost = estimate_cost_usd(
        "anthropic.claude-haiku-4-5", 1_000_000, 0, cache_creation_tokens=1_000_000
    )
    assert cost == Decimal("2.25000000")


def test_sonnet_and_haiku_have_different_rates() -> None:
    sonnet = estimate_cost_usd("anthropic.claude-sonnet-5", 1_000_000, 1_000_000)
    haiku = estimate_cost_usd("anthropic.claude-haiku-4-5", 1_000_000, 1_000_000)
    assert sonnet > haiku


def test_unknown_model_raises_instead_of_returning_zero() -> None:
    # A silent $0 would look like "this call was free," not "this model isn't
    # priced" - fail loud instead.
    with pytest.raises(KeyError):
        estimate_cost_usd("some-model-nobody-priced", 100, 100)


def test_settings_models_are_priced() -> None:
    """Guards the one silent-failure mode `record_llm_usage`'s `try/except`
    deliberately allows: if `MODEL_REASONING`/`MODEL_FAST` (`app/config.py`) ever
    change without a matching `PRICING` update, chat keeps working but cost
    tracking quietly stops recording for that model - every write logs a warning
    and no row lands. This turns that into a loud, immediate CI failure instead."""
    settings = get_settings()
    assert settings.model_reasoning in PRICING
    assert settings.model_fast in PRICING
