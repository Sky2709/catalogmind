"""Chat model routing - cheap by default, escalate only on a real signal. Mirrors
`test_alpha_router.py`'s structure for the retrieval-side router.
"""

from __future__ import annotations

import pytest

from app.llm.model_router import ESCALATE_AFTER_ROUNDS, classify_complexity

FAST_MESSAGES = [
    "do you have waterproof hiking boots",
    "what's the price of the trailhead jacket",
    "is the cotton shirt in stock",
]

REASONING_MESSAGES = [
    "compare the trailhead jacket vs the acme jacket",
    "which is better, the wireless headphones or the wired ones",
    "what's the difference between these two water bottles",
    "I need something waterproof and warm and under $50 and available in size 10",
]


@pytest.mark.parametrize("message", FAST_MESSAGES)
def test_simple_messages_stay_on_the_fast_tier(message: str) -> None:
    assert classify_complexity(message).tier == "fast"


@pytest.mark.parametrize("message", REASONING_MESSAGES)
def test_comparison_and_multi_constraint_messages_escalate(message: str) -> None:
    assert classify_complexity(message).tier == "reasoning"


def test_a_second_unresolved_tool_call_round_forces_escalation() -> None:
    """Even a trivially simple message escalates once the agent's own loop shows
    the easy path already failed once - a concrete signal, not a lexical guess."""
    decision = classify_complexity("boots", tool_call_rounds=ESCALATE_AFTER_ROUNDS)
    assert decision.tier == "reasoning"
    assert "round" in decision.reasons[0]


def test_first_round_does_not_escalate_on_its_own() -> None:
    decision = classify_complexity("boots", tool_call_rounds=ESCALATE_AFTER_ROUNDS - 1)
    assert decision.tier == "fast"


def test_reasons_are_reported_for_observability() -> None:
    decision = classify_complexity("compare boots vs sneakers")
    assert decision.reasons
    assert decision.tier == "reasoning"
