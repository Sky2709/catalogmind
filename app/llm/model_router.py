"""Cheap-vs-strong model routing for the chat agent, same discipline as
`app/retrieval/alpha_router.py`: lexical heuristics only, no LLM call spent
deciding which LLM to call - that would defeat the point of having a cheap tier at
all, and it's the same "no LLM call on the hot path" reasoning the retrieval router
is built around, extended here to the generation side.

Defaults to `model_fast` (`anthropic.claude-haiku-4-5`). Escalates to
`model_reasoning` (`anthropic.claude-sonnet-5`) only when there's a real signal the
turn needs multi-constraint reasoning: explicit comparison language, several
distinct constraints mentioned at once, or the agent's own tool-call loop already
going a second round without resolving (a concrete sign the easy path didn't work,
not a guess).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

ModelTier = Literal["fast", "reasoning"]

# "vs"/"versus" as whole words (not inside "advertisement"), plus explicit
# comparison/superiority phrasing.
_COMPARISON_CUE = re.compile(
    r"\b(vs\.?|versus|compare[sd]?|comparison|better than|worse than|"
    r"difference between|which (?:is|one)|pros and cons|trade-?offs?)\b",
    re.IGNORECASE,
)

# Multiple explicit constraints joined together - "under $50 AND waterproof AND
# size 10" reads as harder than any one constraint alone.
_CONSTRAINT_JOIN = re.compile(r"\b(and|but|also|as well as)\b", re.IGNORECASE)

_TOKEN = re.compile(r"[\w$%]+")

# A second tool-call round means the first `search_catalog` call didn't settle the
# question - a concrete signal of difficulty, not a lexical guess.
ESCALATE_AFTER_ROUNDS = 2

# Long messages tend to bundle more constraints than a short one ever can -
# calibrated loosely against typical shopping-query length, not a golden set (this
# router doesn't have one; it's a cost/latency lever, not a quality-scored
# classifier like `alpha_router.classify`).
_LONG_MESSAGE_TOKENS = 25


@dataclass(frozen=True)
class RoutingDecision:
    tier: ModelTier
    reasons: tuple[str, ...]


def classify_complexity(message: str, *, tool_call_rounds: int = 0) -> RoutingDecision:
    if tool_call_rounds >= ESCALATE_AFTER_ROUNDS:
        return RoutingDecision(
            "reasoning", (f"tool-call loop already at round {tool_call_rounds}",)
        )

    reasons: list[str] = []
    if _COMPARISON_CUE.search(message):
        reasons.append("comparison language")

    constraint_joins = len(_CONSTRAINT_JOIN.findall(message))
    if constraint_joins >= 2:
        reasons.append(f"{constraint_joins} constraint joins ('and'/'but'/...)")

    n_tokens = len(_TOKEN.findall(message))
    if n_tokens >= _LONG_MESSAGE_TOKENS:
        reasons.append(f"long message ({n_tokens} tokens)")

    if reasons:
        return RoutingDecision("reasoning", tuple(reasons))
    return RoutingDecision("fast", ("no escalation signal",))
