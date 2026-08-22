"""Dynamic alpha selection.

Weaviate's hybrid search blends a BM25 keyword score with a vector score using a
single knob, `alpha` (0.0 = pure keyword, 1.0 = pure vector). Almost every
deployment picks one value and ships it. That is leaving relevance on the table,
because the *right* alpha depends on the shape of the query, not the merchant:

    "something for a beach wedding"   -> no lexical overlap with the catalog, so
                                         BM25 contributes noise. Go vector-heavy.
    "waterproof hiking boots size 10" -> both signals carry information.
    "DW-4402B"                        -> the embedding of a part number lands near
                                         other part numbers, which is useless. Go
                                         keyword-heavy.

So: classify the query, then pick alpha per class.

Two rules govern this module.

1. **No LLM call on the hot path.** Classification runs on every search. It must cost
   microseconds, not a network round trip. Heuristics only.
2. **The alpha values are measured, not guessed.** `PRIOR_ALPHA` below are starting
   points. `eval/sweep_alpha.py` sweeps alpha per class against the golden sets and
   writes the winners to `eval/results/tuned_alpha.json`, which this module loads at
   import. If that file is absent (fresh clone, before the first sweep) we fall back
   to the priors and say so.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.retrieval.base import QueryClass

logger = logging.getLogger(__name__)

TUNED_ALPHA_PATH = Path("eval/results/tuned_alpha.json")

# Starting points only - replaced by measured values after the first sweep.
PRIOR_ALPHA: dict[QueryClass, float] = {
    QueryClass.EXPLORATORY: 0.75,
    QueryClass.ATTRIBUTE: 0.50,
    QueryClass.IDENTIFIER: 0.15,
}

# --- signals -------------------------------------------------------------------

# Mixed letters+digits, or a short prefix followed by digits: DW-4402B, A1502, 55X900H.
# The third alternative is pure-numeric SKUs (Myntra-style: "10015819") - real product
# codes with no letters at all, caught the hard way: `eval/retrieval_eval.py`'s
# rerank-quality comparison showed fashion's raw-SKU identifier queries still
# collapsing under reranking even after the has_identifier_shaped_token() skip
# existed, because every one of those SKUs is all-digit and the first two
# alternatives both require a letter. 6+ digits, not fewer, so this doesn't fire on
# ordinary quantities/measurements ("500ml", "size 10", a 4-digit price).
_IDENTIFIER_TOKEN = re.compile(
    r"^(?=.*\d)(?=.*[a-z])[a-z0-9]+(?:[-_/][a-z0-9]+)*$"
    r"|^[a-z]{1,4}[-_]?\d{2,}[a-z0-9-]*$"
    r"|^\d{6,}$",
    re.IGNORECASE,
)

# Spec-like measurements: size 10, 500ml, 32 inch, 4gb, 1.5 ton, 120x60
_MEASUREMENT = re.compile(
    r"\b\d+(?:\.\d+)?\s?(?:ml|l|g|kg|mm|cm|m|in|inch|inches|ft|gb|tb|mb|w|watt|"
    r"v|volt|hz|mah|mp|ton|oz|lb|pack|pcs|yr|year)\b"
    r"|\bsize\s+\w+\b"
    r"|\b\d{2,3}\s?x\s?\d{2,3}\b",
    re.IGNORECASE,
)

# Intent / occasion / recommendation language with no catalog vocabulary.
#
# Deliberately excludes "casual"/"formal": a real, measured false positive caught
# while building `eval/golden/demo_fashion_in.py` - a live `classify()` call on
# "Raymond formal shirt" and "Indian Terrain men slim fit casual shirt" put both in
# EXPLORATORY (alpha 0.75, vector-heavy) purely off that one word, when in fashion
# e-commerce "casual"/"formal" overwhelmingly name a garment *style* attribute
# ("formal shirt", "casual shoes"), not an occasion the way "wedding"/"party"/"gift"
# genuinely are. The misroute was not cosmetic: one query in that golden set went from
# a near-perfect ranking at alpha 0.75 to nDCG@10 = 0.0 at the alpha its (wrong) class
# actually got. Moved to `_ATTRIBUTE_CUE` below instead.
_EXPLORATORY_CUE = re.compile(
    r"\b(something|anything|ideas?|suggest\w*|recommend\w*|looking for|"
    r"need\s+(?:a|an|some)|what should|help me|good for|best for|suitable|"
    r"gift|present|occasion|wedding|party|birthday|anniversary|festival|diwali|"
    r"christmas|vacation|holiday|trip|beach|trend\w*|style|outfit|"
    r"goes with|match\w*)\b",
    re.IGNORECASE,
)

_QUESTION_CUE = re.compile(r"^\s*(what|which|how|can|do|does|is|are|any)\b|\?\s*$", re.IGNORECASE)

# Attribute adjectives that constrain but do not identify. "casual"/"formal" moved
# here from `_EXPLORATORY_CUE` - see that regex's comment for why.
_ATTRIBUTE_CUE = re.compile(
    r"\b(waterproof|wireless|cotton|leather|stainless|organic|slim|regular|oversized|"
    r"under|below|above|cheap|budget|premium|red|blue|black|white|green|yellow|pink|"
    r"small|medium|large|xl|xxl|mens|womens|kid|kids|unisex|rechargeable|portable|"
    r"noise|cancelling|nonstick|induction|smart|4k|hd|casual|formal)\b",
    re.IGNORECASE,
)

_QUOTED = re.compile(r"[\"’']([^\"’']{2,})[\"’']")

_TOKEN = re.compile(r"[\w\-/]+")


@dataclass(frozen=True)
class Classification:
    query_class: QueryClass
    confidence: float
    reasons: tuple[str, ...]


def has_identifier_shaped_token(query: str) -> bool:
    """True when the query contains an actual SKU/model-number-shaped token
    (`_IDENTIFIER_TOKEN`, e.g. 'DW-4402B', 'A1502', or a bare numeric SKU like
    '10015819') - deliberately narrower than
    ``classify(query).query_class == QueryClass.IDENTIFIER``.

    The IDENTIFIER class has several triggers, and only one of them is genuine
    product-code text: a short query with no competing exploratory/attribute cue
    (`"hiking boots"`) or a lone ALLCAPS token can win the class outright with zero
    identifier-shaped tokens present. A caller deciding "can text-similarity
    reranking possibly help here" needs that distinction - `rerank_text()` never
    includes the SKU (see `app/retrieval/hybrid.py`), so a real product code gets no
    benefit from reranking, but a short natural-language query like "hiking boots"
    still can and should be reranked.
    """
    return any(_IDENTIFIER_TOKEN.match(t) for t in _TOKEN.findall(query.strip()))


def classify(query: str) -> Classification:
    """Assign a query to a class using cheap lexical signals.

    Scores each class, returns the winner with a normalised confidence. Confidence is
    reported (and logged) so the eval harness can measure classifier accuracy
    separately from retrieval quality - when nDCG drops it matters whether the
    classifier or the retriever regressed.
    """
    q = query.strip()
    if not q:
        return Classification(QueryClass.EXPLORATORY, 0.0, ("empty query",))

    tokens = _TOKEN.findall(q)
    n_tokens = len(tokens)
    scores: dict[QueryClass, float] = {
        QueryClass.EXPLORATORY: 0.0,
        QueryClass.ATTRIBUTE: 0.0,
        QueryClass.IDENTIFIER: 0.0,
    }
    reasons: list[str] = []

    # --- identifier signals ---
    id_tokens = [t for t in tokens if _IDENTIFIER_TOKEN.match(t)]
    if id_tokens:
        # Weight by how much of the query is identifier-shaped.
        scores[QueryClass.IDENTIFIER] += 2.5 * (len(id_tokens) / max(n_tokens, 1)) + 1.0
        reasons.append(f"identifier-shaped token(s): {id_tokens[:3]}")

    if _QUOTED.search(q):
        scores[QueryClass.IDENTIFIER] += 1.0
        reasons.append("quoted exact phrase")

    if n_tokens <= 3 and not _EXPLORATORY_CUE.search(q):
        scores[QueryClass.IDENTIFIER] += 0.6
        reasons.append("very short query")

    # An ALLCAPS token that is not a common word reads as a model or brand code.
    caps = [t for t in tokens if len(t) > 2 and t.isupper()]
    if caps:
        scores[QueryClass.IDENTIFIER] += 0.5
        reasons.append(f"uppercase token(s): {caps[:3]}")

    # --- attribute signals ---
    if _MEASUREMENT.search(q):
        scores[QueryClass.ATTRIBUTE] += 1.8
        reasons.append("measurement/spec present")

    attr_hits = _ATTRIBUTE_CUE.findall(q)
    if attr_hits:
        scores[QueryClass.ATTRIBUTE] += 0.7 * min(len(attr_hits), 3)
        reasons.append(f"attribute cue(s): {attr_hits[:3]}")

    if 3 < n_tokens <= 8:
        scores[QueryClass.ATTRIBUTE] += 0.5
        reasons.append("mid-length query")

    # --- exploratory signals ---
    if _EXPLORATORY_CUE.search(q):
        scores[QueryClass.EXPLORATORY] += 2.0
        reasons.append("intent/occasion language")

    if _QUESTION_CUE.search(q):
        scores[QueryClass.EXPLORATORY] += 0.8
        reasons.append("question form")

    if n_tokens > 8:
        scores[QueryClass.EXPLORATORY] += 0.8
        reasons.append("long natural-language query")

    if not any(c.isdigit() for c in q) and n_tokens >= 4:
        scores[QueryClass.EXPLORATORY] += 0.4
        reasons.append("no numerics")

    # --- resolve ---
    if max(scores.values()) == 0.0:
        # Nothing fired. ATTRIBUTE is the safe middle: a mid alpha degrades gracefully
        # in both directions, whereas guessing an extreme is actively harmful.
        return Classification(QueryClass.ATTRIBUTE, 0.0, ("no signal; defaulted",))

    winner = max(scores, key=lambda k: scores[k])
    total = sum(scores.values())
    confidence = scores[winner] / total if total else 0.0
    return Classification(winner, round(confidence, 3), tuple(reasons))


# --- tuned alpha loading -------------------------------------------------------


def _load_tuned_alpha() -> tuple[dict[QueryClass, float], bool]:
    """Load measured alphas produced by `eval/sweep_alpha.py`.

    Returns (mapping, is_tuned). Falls back to priors when the sweep has not run.
    """
    if not TUNED_ALPHA_PATH.exists():
        logger.info(
            "no tuned alpha at %s - using priors. Run `make sweep` to measure.",
            TUNED_ALPHA_PATH,
        )
        return dict(PRIOR_ALPHA), False

    try:
        raw = json.loads(TUNED_ALPHA_PATH.read_text(encoding="utf-8"))
        mapping = dict(PRIOR_ALPHA)
        for key, value in raw.get("alpha_by_class", {}).items():
            mapping[QueryClass(key)] = float(value)
        logger.info("loaded tuned alpha from %s: %s", TUNED_ALPHA_PATH, mapping)
        return mapping, True
    except Exception as exc:  # noqa: BLE001 - never let a bad artefact break startup
        logger.warning("could not read %s (%s); using priors", TUNED_ALPHA_PATH, exc)
        return dict(PRIOR_ALPHA), False


class AlphaRouter:
    """Chooses the hybrid alpha for a query."""

    def __init__(self, enabled: bool = True, default_alpha: float = 0.5) -> None:
        self.enabled = enabled
        self.default_alpha = default_alpha
        self.alpha_by_class, self.is_tuned = _load_tuned_alpha()

    def resolve(self, query: str, override: float | None = None) -> tuple[float, Classification]:
        """Return (alpha, classification).

        Precedence: explicit override > dynamic routing > static default. The
        classification is returned even when it was not used, so `/search` can report
        what the router *would* have chosen - useful when comparing against a fixed
        alpha baseline.
        """
        classification = classify(query)

        if override is not None:
            return override, classification
        if not self.enabled:
            return self.default_alpha, classification
        return self.alpha_by_class[classification.query_class], classification
