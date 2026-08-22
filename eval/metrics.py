"""Retrieval quality metrics.

Every published number in the README flows through this module, so the maths is kept
explicit and hand-verifiable rather than delegated to a library. Two conventions are
worth stating up front, because papers and libraries disagree and a silent mismatch
makes results incomparable:

**Gain function.** Discounted Cumulative Gain comes in two flavours::

    linear       gain = rel
    exponential  gain = 2**rel - 1

With binary relevance they rank identically (2**1 - 1 == 1). With graded relevance they
do not: exponential gain sharply rewards putting a *highly* relevant item first, which
is the behaviour we want from a shopping assistant — surfacing the perfect product at
rank 1 matters far more than surfacing a merely acceptable one. We default to
**exponential** and say so in the report. `sklearn.metrics.ndcg_score` defaults to
linear, so numbers here will not match a naive sklearn comparison; that is expected.

**Queries with no relevant items.** Recall and nDCG are undefined when the golden set
lists nothing relevant. We return ``None`` rather than 0.0, and ``aggregate`` skips
them. Scoring them as zero would silently drag every average down in proportion to how
sloppy the labelling was, which is precisely backwards.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from statistics import mean

# A judgement maps document id -> graded relevance. 0 means irrelevant.
Relevance = Mapping[str, int]


def _dedupe(ranking: Sequence[str]) -> list[str]:
    """Drop repeated ids, keeping first occurrence.

    A retriever returning the same SKU twice would otherwise be able to inflate recall
    by padding its result list.
    """
    seen: set[str] = set()
    out: list[str] = []
    for doc_id in ranking:
        if doc_id not in seen:
            seen.add(doc_id)
            out.append(doc_id)
    return out


def _gain(relevance: int, exponential: bool) -> float:
    return (2.0**relevance - 1.0) if exponential else float(relevance)


def dcg_at_k(
    ranking: Sequence[str],
    judgements: Relevance,
    k: int,
    *,
    exponential: bool = True,
) -> float:
    """Discounted cumulative gain over the top k.

    Discount is ``log2(rank + 1)`` with rank starting at 1, so position 1 is undiscounted
    (log2(2) == 1).
    """
    total = 0.0
    for index, doc_id in enumerate(_dedupe(ranking)[:k], start=1):
        rel = judgements.get(doc_id, 0)
        if rel:
            total += _gain(rel, exponential) / math.log2(index + 1)
    return total


def ndcg_at_k(
    ranking: Sequence[str],
    judgements: Relevance,
    k: int,
    *,
    exponential: bool = True,
) -> float | None:
    """Normalised DCG in [0, 1], or None when nothing relevant exists."""
    ideal_order = sorted((r for r in judgements.values() if r > 0), reverse=True)
    if not ideal_order:
        return None

    idcg = sum(
        _gain(rel, exponential) / math.log2(index + 1)
        for index, rel in enumerate(ideal_order[:k], start=1)
    )
    if idcg == 0:
        return None
    return dcg_at_k(ranking, judgements, k, exponential=exponential) / idcg


def recall_at_k(ranking: Sequence[str], judgements: Relevance, k: int) -> float | None:
    """Fraction of all relevant items that appear in the top k."""
    relevant = {doc_id for doc_id, rel in judgements.items() if rel > 0}
    if not relevant:
        return None
    retrieved = set(_dedupe(ranking)[:k])
    return len(retrieved & relevant) / len(relevant)


def precision_at_k(ranking: Sequence[str], judgements: Relevance, k: int) -> float | None:
    """Fraction of the top k that is relevant.

    Denominator is k, not len(ranking): a retriever that returns three results for
    precision@10 is penalised for the seven it failed to fill, which is the honest
    reading when the catalog contains more relevant items.
    """
    if k <= 0:
        return None
    relevant = {doc_id for doc_id, rel in judgements.items() if rel > 0}
    if not relevant:
        return None
    hits = sum(1 for doc_id in _dedupe(ranking)[:k] if doc_id in relevant)
    return hits / k


def reciprocal_rank(ranking: Sequence[str], judgements: Relevance) -> float | None:
    """1 / rank of the first relevant hit; 0.0 if none was retrieved at all.

    Note the deliberate asymmetry with the None cases above: "there was something to
    find and you missed it" is a genuine zero, whereas "there was nothing to find" is
    undefined.
    """
    relevant = {doc_id for doc_id, rel in judgements.items() if rel > 0}
    if not relevant:
        return None
    for index, doc_id in enumerate(_dedupe(ranking), start=1):
        if doc_id in relevant:
            return 1.0 / index
    return 0.0


def hit_rate_at_k(ranking: Sequence[str], judgements: Relevance, k: int) -> float | None:
    """1.0 if any relevant item made the top k, else 0.0."""
    relevant = {doc_id for doc_id, rel in judgements.items() if rel > 0}
    if not relevant:
        return None
    return 1.0 if relevant & set(_dedupe(ranking)[:k]) else 0.0


# --- per-query and aggregate ----------------------------------------------------


@dataclass(frozen=True)
class QueryScores:
    """All metrics for a single query. None means 'undefined', not 'zero'."""

    query_id: str
    ndcg_at_10: float | None
    recall_at_10: float | None
    recall_at_50: float | None
    precision_at_10: float | None
    reciprocal_rank: float | None
    hit_rate_at_10: float | None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "ndcg@10": self.ndcg_at_10,
            "recall@10": self.recall_at_10,
            "recall@50": self.recall_at_50,
            "precision@10": self.precision_at_10,
            "mrr": self.reciprocal_rank,
            "hit_rate@10": self.hit_rate_at_10,
        }


def score_query(
    query_id: str,
    ranking: Sequence[str],
    judgements: Relevance,
    *,
    exponential: bool = True,
) -> QueryScores:
    return QueryScores(
        query_id=query_id,
        ndcg_at_10=ndcg_at_k(ranking, judgements, 10, exponential=exponential),
        recall_at_10=recall_at_k(ranking, judgements, 10),
        recall_at_50=recall_at_k(ranking, judgements, 50),
        precision_at_10=precision_at_k(ranking, judgements, 10),
        reciprocal_rank=reciprocal_rank(ranking, judgements),
        hit_rate_at_10=hit_rate_at_k(ranking, judgements, 10),
    )


def aggregate(scores: Iterable[QueryScores]) -> dict[str, float | None]:
    """Macro-average each metric across queries, skipping undefined values.

    Macro (mean of per-query scores) rather than micro (pooled counts): every query
    gets equal weight, so a handful of queries with large golden sets cannot dominate
    the headline number.

    Also reports ``n_queries`` and, per metric, how many queries actually contributed —
    a metric averaged over three of sixty queries is not a result, and the report must
    make that visible rather than printing a confident-looking float.
    """
    materialised = list(scores)
    if not materialised:
        return {"n_queries": 0}

    out: dict[str, float | None] = {"n_queries": len(materialised)}
    keys = materialised[0].as_dict().keys()

    for key in keys:
        values = [value for score in materialised if (value := score.as_dict()[key]) is not None]
        out[key] = round(mean(values), 4) if values else None
        out[f"{key}_n"] = len(values)

    return out
