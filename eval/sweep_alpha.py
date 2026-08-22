"""The headline experiment: does dynamic alpha actually beat any single fixed
alpha, measured against the 170-query golden set - not argued from first
principles.

Sweeps `alpha` from 0.0 to 1.0 in steps of 0.1, for each of the three query classes,
across all 170 golden queries from all three demo tenants (grouped by each query's
own labelled `query_class` - the golden set's design-time category, not whatever
`classify()` happens to guess live; the sweep is about which alpha best serves a
given *kind* of query, independent of classifier accuracy, which is its own,
separate question). Reranking is deliberately off throughout: the sweep is about the
hybrid-blend stage, and `scripts/bench_search.py` already measured reranking at
~8.8s/call at `retrieve_top_k=50` - 3 classes x 11 alphas x ~57 queries each with
reranking on would be hours, not the minutes this actually takes without it.

The claim actually being tested, precisely: is there one single fixed alpha that
matches what each class's own best alpha achieves? If not - and a router that could
always pick each query's class's best alpha is exactly what "dynamic alpha" is -
dynamic alpha has a real, measured advantage over every fixed alternative, not just
over whichever fixed value we felt like comparing against.

Writes:
- `eval/results/alpha_sweep.json` - the full (alpha, class) -> nDCG@10 grid
- `eval/results/alpha_sweep.png` - one curve per class vs alpha, with the best-fixed
  and dynamic reference points marked
- `eval/results/tuned_alpha.json` - the winning alpha per class, in the exact shape
  `app/retrieval/alpha_router.py::_load_tuned_alpha` already knows how to read. This
  is the file that doesn't exist yet today, so the router runs on `PRIOR_ALPHA`
  guesses until this script has been run once.

Run: `.venv/bin/python -m eval.sweep_alpha`
Requires `make up` and the three demo catalogs seeded (`make seed`).
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from app.retrieval.base import QueryClass, SearchFilters, SearchRequest
from app.retrieval.hybrid import get_retriever
from app.retrieval.weaviate_client import dispose_shared_client
from eval.golden import GoldenQuery, load_golden_set
from eval.metrics import ndcg_at_k

TENANTS = ("demo-fashion-in", "demo-electronics-in", "demo-home-goods")
ALPHAS = [round(a * 0.1, 1) for a in range(11)]  # 0.0, 0.1, ..., 1.0

SWEEP_JSON_PATH = Path("eval/results/alpha_sweep.json")
SWEEP_CHART_PATH = Path("eval/results/alpha_sweep.png")
TUNED_ALPHA_PATH = Path("eval/results/tuned_alpha.json")


def _load_all_queries() -> dict[QueryClass, list[tuple[str, GoldenQuery]]]:
    """Every golden query, grouped by its own labelled class, tagged with its
    tenant (a query only knows its own tenant's SKUs, so the tenant has to travel
    with it)."""
    by_class: dict[QueryClass, list[tuple[str, GoldenQuery]]] = {c: [] for c in QueryClass}
    for tenant in TENANTS:
        for query in load_golden_set(tenant):
            by_class[query.query_class].append((tenant, query))
    return by_class


async def _ndcg_for(tenant: str, query: GoldenQuery, alpha: float) -> float | None:
    request = SearchRequest(
        query=query.query,
        tenant=tenant,
        limit=10,
        alpha=alpha,
        rerank=False,
        filters=SearchFilters(),
    )
    response = await get_retriever().search(request)
    ranking = [hit.sku for hit in response.hits]
    return ndcg_at_k(ranking, query.judgments, 10)


async def _sweep_class(
    query_class: QueryClass, queries: list[tuple[str, GoldenQuery]]
) -> dict[float, float]:
    """Mean nDCG@10 at every alpha, for one class's queries."""
    scores_by_alpha: dict[float, float] = {}
    for alpha in ALPHAS:
        values = [
            v
            for v in await asyncio.gather(
                *(_ndcg_for(tenant, query, alpha) for tenant, query in queries)
            )
            if v is not None
        ]
        scores_by_alpha[alpha] = sum(values) / len(values) if values else 0.0
    return scores_by_alpha


def _best_fixed_alpha_overall(grid: dict[QueryClass, dict[float, float]]) -> tuple[float, float]:
    """Across every candidate alpha, the one that does best *averaged over all three
    classes equally* - the single-alpha alternative dynamic routing is up against."""
    best_alpha, best_score = ALPHAS[0], -1.0
    for alpha in ALPHAS:
        macro = sum(grid[c][alpha] for c in QueryClass) / len(QueryClass)
        if macro > best_score:
            best_alpha, best_score = alpha, macro
    return best_alpha, best_score


def _dynamic_score(
    grid: dict[QueryClass, dict[float, float]],
) -> tuple[dict[QueryClass, float], float]:
    """Each class gets its own best alpha - exactly what a correctly tuned dynamic
    router achieves. Returns (winning alpha per class, macro-averaged score)."""
    best_alpha_per_class = {c: max(grid[c], key=lambda a: grid[c][a]) for c in QueryClass}
    macro = sum(grid[c][best_alpha_per_class[c]] for c in QueryClass) / len(QueryClass)
    return best_alpha_per_class, macro


def _write_chart(grid: dict[QueryClass, dict[float, float]], best_fixed_alpha: float) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for query_class in QueryClass:
        ys = [grid[query_class][a] for a in ALPHAS]
        ax.plot(ALPHAS, ys, marker="o", label=query_class.value)
        best_a = max(grid[query_class], key=lambda a: grid[query_class][a])
        ax.scatter([best_a], [grid[query_class][best_a]], zorder=5, s=80, edgecolor="black")

    ax.axvline(
        best_fixed_alpha, color="grey", linestyle="--", linewidth=1, label="best single fixed alpha"
    )
    ax.set_xlabel("alpha (0 = pure keyword, 1 = pure vector)")
    ax.set_ylabel("mean nDCG@10")
    ax.set_title("Alpha sweep: no single fixed alpha serves every query class")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    SWEEP_CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(SWEEP_CHART_PATH, dpi=150)


def _write_json_outputs(
    grid: dict[QueryClass, dict[float, float]],
    best_alpha_per_class: dict[QueryClass, float],
    dynamic_macro: float,
    best_fixed_alpha: float,
    best_fixed_macro: float,
) -> None:
    SWEEP_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_JSON_PATH.write_text(
        json.dumps(
            {
                "alphas": ALPHAS,
                "ndcg_by_class": {c.value: grid[c] for c in QueryClass},
                "best_alpha_per_class": {c.value: best_alpha_per_class[c] for c in QueryClass},
                "dynamic_macro_ndcg": dynamic_macro,
                "best_single_fixed_alpha": best_fixed_alpha,
                "best_single_fixed_macro_ndcg": best_fixed_macro,
                "dynamic_advantage": dynamic_macro - best_fixed_macro,
            },
            indent=2,
        )
    )
    TUNED_ALPHA_PATH.write_text(
        json.dumps(
            {"alpha_by_class": {c.value: best_alpha_per_class[c] for c in QueryClass}}, indent=2
        )
    )


async def main() -> None:
    by_class = _load_all_queries()
    for query_class in QueryClass:
        print(f"{query_class.value}: {len(by_class[query_class])} queries")

    grid: dict[QueryClass, dict[float, float]] = {}
    t0 = time.perf_counter()
    for query_class in QueryClass:
        print(f"\nsweeping {query_class.value}...")
        grid[query_class] = await _sweep_class(query_class, by_class[query_class])
        row = "  ".join(f"a={a:.1f}:{grid[query_class][a]:.3f}" for a in ALPHAS)
        print(f"  {row}")
    print(f"\nsweep took {time.perf_counter() - t0:.1f}s")

    best_alpha_per_class, dynamic_macro = _dynamic_score(grid)
    best_fixed_alpha, best_fixed_macro = _best_fixed_alpha_overall(grid)

    print("\nBest alpha per class (what a tuned dynamic router would use):")
    for query_class in QueryClass:
        a = best_alpha_per_class[query_class]
        print(f"  {query_class.value:>12}: alpha={a:.1f}  nDCG@10={grid[query_class][a]:.4f}")

    print(
        f"\nBest SINGLE fixed alpha across all 3 classes: {best_fixed_alpha:.1f}  "
        f"(macro nDCG@10={best_fixed_macro:.4f})"
    )
    print(f"Dynamic (each class its own best alpha):        macro nDCG@10={dynamic_macro:.4f}")
    print(
        f"Dynamic advantage over the best possible single fixed alpha: "
        f"{dynamic_macro - best_fixed_macro:+.4f}"
    )

    _write_chart(grid, best_fixed_alpha)
    _write_json_outputs(
        grid, best_alpha_per_class, dynamic_macro, best_fixed_alpha, best_fixed_macro
    )
    print(f"\nWrote {SWEEP_JSON_PATH}, {SWEEP_CHART_PATH}, {TUNED_ALPHA_PATH}")

    await dispose_shared_client()


if __name__ == "__main__":
    asyncio.run(main())
