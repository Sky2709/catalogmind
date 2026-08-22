"""Detecting a refusal-shaped answer in plain English - shared by production
(`app/llm/graph.py::validate_and_store`, deciding whether to display product cards
for a turn that just declined to recommend anything) and eval
(`eval/generation_metrics.py`, scoring `kind="refusal"` scenarios). Kept as its own
module and imported from production, not the other way around, matching
`price_text.py`'s existing split.

Deliberately broad across common phrasings for "I don't have that" - not tuned
against a labelled set (there isn't one), just sampled from how a refusal reads in
plain English. The `(?:\\s+\\w+){0,2}` gap between the negation and the verb is a
real fix, not initial caution: the original contiguous "don't have"/"don't carry"
version missed real refusals a live eval run actually produced ("we don't appear to
carry smartphones", "I don't see smartphones in our catalog") - a shopping
assistant hedges more than a short synthetic test answer would suggest.

Widened again after the full 218-scenario eval run (2026-08-21): "I wasn't able to
find any 4K smart Android TVs... I don't want to recommend any of these items" is a
clear, plain-English refusal that missed every existing branch at the time.

This is a lexical cue list, not a semantic judge - it will miss a refusal phrased
unusually, and (the risk that matters most for its production use below) it can
also fire on a genuinely helpful answer that hedges with a refusal-shaped phrase
while still recommending something ("I don't have an exact match, but this similar
item might work: SKU-X") - accepted the same way `citations.py`'s and `claims.py`'s
own heuristics document their false-positive/negative tradeoffs, not silently
assumed precise.
"""

from __future__ import annotations

import re

_REFUSAL_CUE = re.compile(
    r"\b(don'?t(?:\s+\w+){0,2}\s+(?:have|sell|carry|stock|currently|see|offer|"
    r"want to (?:recommend|guess|suggest|point you))|"
    r"doesn'?t(?:\s+\w+){0,2}\s+(?:have|carry|stock|offer)|"
    r"do not(?:\s+\w+){0,2}\s+(?:have|sell|carry|stock|see|offer)|"
    r"no (?:matching|such|products?|results?)|"
    r"not (?:available|carried|something we|able to find|carry)|"
    r"couldn'?t find|unable to find|wasn'?t able to find|"
    r"sorry|unfortunately|apologi[sz]e|"
    r"outside (?:of )?(?:our|this) catalog|"
    r"isn'?t something (?:we|this) )\b",
    re.IGNORECASE,
)


def refuses(answer: str) -> bool:
    """Whether `answer` reads as a refusal in plain English. See module docstring
    for the lexical-cue limitation."""
    return bool(_REFUSAL_CUE.search(answer))
