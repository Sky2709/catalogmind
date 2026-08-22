"""Verifying numeric claims against what `get_catalog_stats` actually returned
this turn - the same "check the model against real tool output" discipline
`app/llm/citations.py` already proved out for SKU citations, applied to a different
failure mode: not a fabricated product, but a confidently wrong catalog-wide claim.

Kept as a separate module rather than folded into `citations.py`: different marker
(`[[STAT:N]]`, not `[[SKU:X]]`), different failure mode (a false "nothing
matches"/"the highest is X" rather than an invented product), and `citations.py`'s
whole history is the SKU-citation story specifically.

Real production bug this module exists to catch (2026-08-22, confirmed live, not
hypothetical): asked "what's the highest-priced item for men", the chat agent
answered "nothing above ₹2,499" using only `search_catalog`'s 5-result window - the
real answer was a ₹58,854 item, and 94 products in the catalog exceeded ₹10,000.
`get_catalog_stats` (`app/retrieval/hybrid.py::WeaviateHybridRetriever.stats`) exists
to make that claim answerable correctly; this module exists to catch it if the model
asserts a number anyway without the tool result backing it up.

**Retired the free-text quantifiable-negation regex this session (2026-08-22)**:
the old `_QUANTIFIABLE_NEGATION_CUE`/`has_quantifiable_negation` tried to guess
whether a sentence *meant* "nothing meets this threshold" from its surface
phrasing - the same fundamentally-incomplete guess `app/llm/citations.py`'s old
SKU-shaped-token regex made, just for a narrower claim shape. Claude now wraps any
`get_catalog_stats`-backed figure in a `[[STAT:N]]` marker (`app/llm/markers.py`)
regardless of how the surrounding sentence is phrased - "nothing above X", "the
max is X", "everything is under X" all reduce to the same checkable `[[STAT:X]]`,
so there is no phrasing left to enumerate. See `PROGRESS.md`'s dated entry for the
full account.

Detection, not prevention - by the time these run, the answer has already streamed
to the client (same documented limitation as `find_hallucinated_citations`).
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from decimal import Decimal, InvalidOperation
from typing import Any

from app.llm.markers import extract_stat_markers

# Matches eval.generation_metrics.find_price_mismatch's existing tolerance - a
# model saying "about ₹58,900" for a real ₹58,854 max is a rounding, not a
# hallucination.
_PRICE_TOLERANCE_FRACTION = Decimal("0.02")


def _stat_markers_as_decimals(answer: str) -> list[Decimal]:
    values = []
    for raw in extract_stat_markers(answer):
        try:
            values.append(Decimal(raw))
        except InvalidOperation:
            continue  # a malformed marker is not this function's problem to solve
    return values


def find_stat_claim_mismatch(
    answer: str, stats_evidence: Collection[Mapping[str, Any]]
) -> bool | None:
    """`True` if any `[[STAT:N]]` marker in `answer` doesn't match the one
    `metric=="price"` stats call's min/max/mean this turn; `False` if every
    marker matches; `None` if the shape is too ambiguous to score (more than one
    stats call, no price-metric call, or no marker emitted at all) - same
    "undefined isn't zero" discipline as `eval.generation_metrics.find_price_mismatch`.

    Deliberately does not attempt to verify rating/review_count claims yet - a
    scoped MVP boundary, not a silently ignored one.
    """
    markers = _stat_markers_as_decimals(answer)
    if not markers:
        return None

    price_entries = [e for e in stats_evidence if e.get("metric") == "price"]
    if len(price_entries) != 1:
        return None

    entry = price_entries[0]
    known = {
        Decimal(v)
        for v in (entry.get("minimum"), entry.get("maximum"), entry.get("mean"))
        if v is not None
    }
    if not known:
        return None

    return any(
        not any(abs(marker - value) / value <= _PRICE_TOLERANCE_FRACTION for value in known if value != 0)
        for marker in markers
    )


def find_unverified_quantitative_refusal(
    answer: str, stats_evidence: Collection[Mapping[str, Any]]
) -> bool:
    """`True` if `answer` contains a `[[STAT:N]]` marker with no
    `get_catalog_stats` evidence at all this turn to back it - a confident
    numeric claim asserted without the one tool call that could verify it.

    Unlike the old text-cue version, this needs no separate "does this sentence
    sound like a threshold negation" check: any stats-backed figure is wrapped in
    the same marker regardless of phrasing, so the mere absence of stats evidence
    is enough once a marker is present.
    """
    if not extract_stat_markers(answer):
        return False
    return not any(e.get("kind") == "stats" for e in stats_evidence)


# Cue for "this question needed get_catalog_stats and might not have gotten it" -
# fires on the *shopper's* message (was this the kind of question that needed the
# tool), not the answer - a smaller, bounded input space than free-form generated
# prose (real shoppers reuse a handful of recognisable phrasings; an LLM's own
# prose does not), and purely an observability leading-indicator (confirmed with
# the user as "signal only, not a gate"). That boundedness is exactly why this
# heuristic was kept as a lexical cue rather than folded into the marker-protocol
# rework retiring the *other* free-text heuristics in this module - see that
# rework's plan file for the full reasoning on why the two are different in kind.
#
# Measured against an independent LLM judge (2026-08-22,
# `eval/measure_superlative_heuristic.py`) rather than left an unvalidated guess,
# the same discipline `alpha_router.py`'s cue lists already get. First pass:
# 8/14 agreement, 0 false positives, 6/6 confirmed real misses on a hand-authored
# probe set designed to test paraphrases of the same underlying need. Extended
# with the confirmed, unambiguous gaps: "priciest" (a `cheapest`/`most expensive`
# synonym), "average" (a real `get_catalog_stats` metric with no cue at all
# before this), "total number of" (a `how many` synonym). Deliberately not
# adding "mean" as a synonym for "average" alongside these: it was never one of
# the measured probes, and a bare `\bmean\b` would false-positive on ordinary
# usage ("what do you mean by slim fit?") - exactly the kind of unmeasured,
# speculative addition this whole exercise is arguing against making. **Two
# confirmed
# gaps deliberately left unfixed**, named rather than silently accepted: a
# spelled-out number threshold ("over ten thousand rupees" - would need a
# number-word parser, not a keyword, to catch generally) and a vague
# distribution question ("what price range do most products fall into" - a
# recognisable phrase would risk flagging looser "what's your price range"
# questions that don't always need an exact aggregate, trading one failure
# direction for the other rather than fixing anything).
_SUPERLATIVE_CUE = re.compile(
    r"\b(highest|lowest|most expensive|priciest|cheapest|how many|count of|"
    r"total number of|average|"
    r"above [₹$]?\d+|over [₹$]?\d+|under [₹$]?\d+|below [₹$]?\d+|"
    r"more than \d+|at least \d+|at most \d+)\b",
    re.IGNORECASE,
)


def has_superlative_language(message: str) -> bool:
    return bool(_SUPERLATIVE_CUE.search(message))
