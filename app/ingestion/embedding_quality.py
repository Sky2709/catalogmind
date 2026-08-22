"""Intrinsic embedding-quality checks: does the vector space this pipeline produces
actually carry semantic signal, without needing a search endpoint or a labelled query
set (Day 3/4 - see `app/ingestion/quality.py`'s module docstring for the same
reasoning applied to raw data quality instead of embeddings).

Three checks, all standard practice for sanity-checking an embedding space before
building a retrieval system on top of it:

* **Group contrast** - products sharing a real attribute (brand, category) should be
  more similar to each other, on average, than two random products. If they are not,
  the embedding is not capturing that signal, whatever else it might be capturing.
* **Average pairwise similarity** - a sanity ceiling. If unrelated products score
  very close to 1.0 on average, the embedding space has collapsed (everything looks
  the same), which no downstream reranking or alpha-tuning can fix.
* **Nearest-neighbour spot check** - for a human to actually read. A number saying
  "0.62" means nothing on its own; seeing that a shirt's nearest neighbours are other
  shirts is what makes a similarity score interpretable.

Vectors are assumed already L2-normalised (true for everything `app.ingestion.embed`
produces), so cosine similarity is a plain dot product - no need to pass norms around.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    """Assumes both vectors are already L2-normalised - see module docstring."""
    return sum(x * y for x, y in zip(a, b, strict=True))


def nearest_neighbors(
    query_vector: Sequence[float],
    corpus: Sequence[tuple[str, Sequence[float]]],
    k: int,
    *,
    exclude: str | None = None,
) -> list[tuple[str, float]]:
    """Top-k (label, similarity) pairs, highest first. `exclude` drops a label (an
    anchor's own id) from its own neighbour list."""
    scored = [
        (label, cosine_similarity(query_vector, vec)) for label, vec in corpus if label != exclude
    ]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:k]


def average_pairwise_similarity(
    vectors: Sequence[Sequence[float]], *, n_pairs: int = 2000, seed: int = 0
) -> float:
    """Mean cosine similarity over `n_pairs` random distinct pairs (all pairs, if the
    corpus is smaller than that). A sanity ceiling: real, diverse product text should
    land well below 1.0 on average - if it doesn't, the embedding space has collapsed,
    not just "the products happen to be similar"."""
    n = len(vectors)
    if n < 2:
        return 0.0
    rng = random.Random(seed)
    total_possible = n * (n - 1) // 2
    pairs: list[tuple[int, int]]
    if total_possible <= n_pairs:
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        seen: set[tuple[int, int]] = set()
        while len(seen) < n_pairs:
            i, j = rng.randrange(n), rng.randrange(n)
            if i != j:
                seen.add((min(i, j), max(i, j)))
        pairs = list(seen)
    return statistics.mean(cosine_similarity(vectors[i], vectors[j]) for i, j in pairs)


@dataclass(frozen=True)
class GroupContrastResult:
    """Mean within-group similarity vs mean across-group (random pair) similarity,
    for a grouping variable that is supposed to correlate with the embedded text
    (e.g. brand, category) but was never used to *produce* the embedding - so a real
    gap here is evidence the embedding independently captures that signal."""

    grouping_field: str
    groups_compared: int
    within_group_mean: float
    across_group_mean: float
    n_within_pairs: int
    n_across_pairs: int

    @property
    def contrast(self) -> float:
        """within - across. Positive and not tiny means the embedding discriminates
        this grouping; near zero means it doesn't (which may mean the grouping itself
        has no semantic footprint in the text, not necessarily that the embedding is
        bad - see the module docstring's caveat about this needing human judgement)."""
        return self.within_group_mean - self.across_group_mean


def group_contrast[T](
    items: Sequence[T],
    vectors: Sequence[Sequence[float]],
    group_key: Callable[[T], str | None],
    *,
    grouping_field: str,
    max_pairs_per_side: int = 2000,
    seed: int = 0,
) -> GroupContrastResult | None:
    """Returns None if there are fewer than 2 groups with >= 2 members each - no
    within-group pair is possible below that, and the comparison is meaningless."""
    if len(items) != len(vectors):
        raise ValueError("items and vectors must be the same length and order")

    by_group: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        key = group_key(item)
        if key:
            by_group.setdefault(key, []).append(idx)

    groups_with_pairs = [idxs for idxs in by_group.values() if len(idxs) >= 2]
    if len(groups_with_pairs) < 2:
        return None

    rng = random.Random(seed)

    within_pairs: list[tuple[int, int]] = []
    for idxs in groups_with_pairs:
        all_pairs = [(idxs[a], idxs[b]) for a in range(len(idxs)) for b in range(a + 1, len(idxs))]
        if len(all_pairs) > max_pairs_per_side:
            all_pairs = rng.sample(all_pairs, max_pairs_per_side)
        within_pairs.extend(all_pairs)

    all_grouped_idxs = [i for idxs in groups_with_pairs for i in idxs]
    across_pairs: set[tuple[int, int]] = set()
    attempts = 0
    while len(across_pairs) < max_pairs_per_side and attempts < max_pairs_per_side * 20:
        attempts += 1
        i, j = rng.choice(all_grouped_idxs), rng.choice(all_grouped_idxs)
        if i == j:
            continue
        key_i = group_key(items[i])
        key_j = group_key(items[j])
        if key_i != key_j:
            across_pairs.add((min(i, j), max(i, j)))

    within_sims = [cosine_similarity(vectors[i], vectors[j]) for i, j in within_pairs]
    across_sims = [cosine_similarity(vectors[i], vectors[j]) for i, j in across_pairs]
    if not within_sims or not across_sims:
        return None

    return GroupContrastResult(
        grouping_field=grouping_field,
        groups_compared=len(groups_with_pairs),
        within_group_mean=statistics.mean(within_sims),
        across_group_mean=statistics.mean(across_sims),
        n_within_pairs=len(within_sims),
        n_across_pairs=len(across_sims),
    )
