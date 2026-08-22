"""Generation-quality scoring: pure functions only, no network/LLM calls - the same
split `eval/metrics.py` keeps for retrieval. `eval/generation_eval.py` is the only
module here that spends real money; everything in this file is free to run and
unit-test (`tests/unit/test_generation_metrics.py`).

Three mechanical signals, each a lexical/arithmetic heuristic rather than an LLM
judge - consistent with this project's existing discipline
(`app/retrieval/alpha_router.py`, `app/llm/model_router.py`) of preferring a cheap,
inspectable rule over spending another model call to grade the first one:

1. **Groundedness** - reuses `app.llm.citations.find_hallucinated_citations`
   directly, the exact function the live chat graph runs on every real turn
   (`app/llm/graph.py::validate_and_store`). Scoring with the same function
   production uses, not a reimplementation, is deliberate - a second, drifted
   copy could pass this eval while the real thing regresses.
2. **Attribute (price) hallucination** - a mechanical, narrowly-scoped check: if
   the answer mentions exactly one price-shaped figure and exactly one real SKU
   was cited with a known price, do they match (within rounding tolerance)? Any
   messier shape (multiple prices, multiple cited products, no price mentioned at
   all) is left **unscored** (`None`), not guessed at - same "undefined isn't
   zero" rule `eval/metrics.py` documents for queries with no relevant judgments.
   This deliberately does not attempt the harder, more general "is every claimed
   attribute correct" question (color, material, size...) - that needs either an
   LLM judge or a much larger structured-attribute ground truth, neither built
   here. Price is the one attribute already present, structured, and numeric in
   every cited product (`app/llm/prompting.py::_citable`), which is what makes it
   checkable without either.
3. **Refusal correctness** - for a scenario built with no real answer
   (`ChatScenario(kind="refusal")`), correct behaviour is "didn't fabricate a
   citation" *and* "said so in words a shopper would recognise as a no". Reuses
   `app.llm.refusal_text.refuses` directly - the same lexical-cue detector
   `app/llm/graph.py::validate_and_store` runs on every real turn (to decide
   whether to display product cards alongside a refusal), for the same
   "score with what production actually uses" reason `find_hallucinated_citations`
   above is shared rather than reimplemented. See that module's docstring for
   the lexical-cue limitation (misses an unusual phrasing; can also fire on a
   helpful answer that hedges before still recommending something).
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from typing import Any

from app.llm.citations import find_hallucinated_citations
from app.llm.markers import is_no_match
from app.llm.price_text import extract_price_mentions
from app.llm.refusal_text import refuses

__all__ = [
    "aggregate_generation",
    "find_price_mismatch",
    "refuses",
    "score_scenario",
]

# Within this fraction of the real price counts as "the same number, differently
# rounded" rather than a genuine mismatch - a model saying "about ₹1,300" for a
# ₹1,299 item is not a hallucination.
_PRICE_TOLERANCE_FRACTION = 0.02


def find_price_mismatch(answer: str, cited_products: list[dict]) -> bool | None:
    """`True` if the one price mentioned in `answer` doesn't match the one cited
    product's real price; `False` if it matches; `None` if the shape is too
    ambiguous to score (see module docstring - multiple prices, multiple priced
    products cited, or no price mentioned at all)."""
    priced_products = [p for p in cited_products if p.get("price") is not None]
    if len(priced_products) != 1:
        return None

    mentions = extract_price_mentions(answer)
    if len(mentions) != 1:
        return None

    try:
        real_price = Decimal(str(priced_products[0]["price"]))
    except InvalidOperation:
        return None

    if real_price == 0:
        return None

    return abs(mentions[0] - real_price) / real_price > Decimal(str(_PRICE_TOLERANCE_FRACTION))


@dataclass(frozen=True)
class ScenarioResult:
    scenario_id: str
    kind: str
    grounded: bool
    """No hallucinated SKU citation - `find_hallucinated_citations` came back empty."""
    hallucinated_citations: tuple[str, ...]
    answer_correct: bool | None
    """Grounded scenarios only: does the answer *text* actually name an expected
    SKU (not merely: was one retrieved this turn - see `score_scenario`)? `None`
    for refusal scenarios, where there is no expected SKU to check against."""
    price_mismatch: bool | None
    """`None` = unscored (see `find_price_mismatch`); only meaningful when not None."""
    refused: bool | None
    """Refusal scenarios only: did the answer read as a refusal? `None` for
    grounded scenarios."""


def score_scenario(
    *,
    scenario_id: str,
    kind: str,
    expected_skus: frozenset[str],
    answer: str,
    cited_products: list[dict],
) -> ScenarioResult:
    hallucinated = tuple(find_hallucinated_citations(answer, cited_products))

    answer_correct = None
    refused = None
    if kind == "grounded":
        # Checked against the answer *text*, not `cited_skus` (the retrieved
        # products) - the expected SKU being retrieved this turn is a retrieval-
        # quality fact already measured by `eval/retrieval_eval.py`; what this
        # eval exists to check is whether the model actually said so. Still a
        # plain substring check (not a `[[SKU:...]]`-marker check) deliberately:
        # `[[SKU:BOOT-WP-10]]` contains `BOOT-WP-10` as a plain substring, so this
        # needs no change for the marker protocol to keep working.
        folded_answer = answer.casefold()
        answer_correct = any(sku.casefold() in folded_answer for sku in expected_skus)
    elif kind == "refusal":
        # `is_no_match` first - the real primary signal production now uses
        # (`app/llm/graph.py::validate_and_store`) - `refuses()` only as the same
        # backstop production keeps. Scoring with what production actually uses,
        # not a parallel reimplementation that could drift.
        refused = is_no_match(answer) or refuses(answer)

    return ScenarioResult(
        scenario_id=scenario_id,
        kind=kind,
        grounded=not hallucinated,
        hallucinated_citations=hallucinated,
        answer_correct=answer_correct,
        price_mismatch=find_price_mismatch(answer, cited_products),
        refused=refused,
    )


def aggregate_generation(results: list[ScenarioResult]) -> dict[str, Any]:
    """Macro rates across every scenario, mirroring `eval.metrics.aggregate`'s
    "None means undefined, report how many contributed" convention."""
    n = len(results)
    groundedness_rate = sum(r.grounded for r in results) / n if n else None

    grounded_results = [r for r in results if r.kind == "grounded"]
    answer_hit_rate = (
        sum(bool(r.answer_correct) for r in grounded_results) / len(grounded_results)
        if grounded_results
        else None
    )

    refusal_results = [r for r in results if r.kind == "refusal"]
    refusal_correctness = (
        sum(bool(r.refused) and r.grounded for r in refusal_results) / len(refusal_results)
        if refusal_results
        else None
    )

    scored_price_checks = [r.price_mismatch for r in results if r.price_mismatch is not None]
    hallucinated_attribute_rate = (
        sum(scored_price_checks) / len(scored_price_checks) if scored_price_checks else None
    )

    return {
        "n_scenarios": n,
        "groundedness_rate": groundedness_rate,
        "answer_hit_rate": answer_hit_rate,
        "answer_hit_rate_n": len(grounded_results),
        "refusal_correctness": refusal_correctness,
        "refusal_correctness_n": len(refusal_results),
        "hallucinated_attribute_rate": hallucinated_attribute_rate,
        "hallucinated_attribute_rate_n": len(scored_price_checks),
    }
