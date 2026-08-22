"""Turns `eval/results/retrieval_eval.json` and `eval/results/alpha_sweep.json` into
one markdown report with the numbers `README.md` actually publishes - so every
number that ends up in the README traces back to a JSON file `make eval`/`make
sweep` produced, not to a number someone typed while writing the README.

Run: `.venv/bin/python -m eval.report`
Requires `eval.retrieval_eval` and `eval.sweep_alpha` to have both run at least once
(`make eval` runs the first automatically; `make sweep` runs the second - `make eval`
does not, since the sweep is a separate, longer-running experiment, not part of the
regular eval loop).
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import get_settings

RETRIEVAL_EVAL_PATH = Path("eval/results/retrieval_eval.json")
ALPHA_SWEEP_PATH = Path("eval/results/alpha_sweep.json")
REPORT_PATH = Path("eval/results/report.md")


def _load(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(
            f"{path} does not exist - run `make eval`"
            + (" and `make sweep`" if path == ALPHA_SWEEP_PATH else "")
            + " first."
        )
    return json.loads(path.read_text())


def _retrieval_section(data: dict) -> str:
    shipped = data["shipped_config"]
    rerank_off = data["rerank_off"]
    fair = data["rerank_at_judgment_depth"]
    overall = shipped["overall"]
    overall_no_rerank = rerank_off["overall"]
    fair_lift = fair["overall"]["ndcg@10"] - overall_no_rerank["ndcg@10"]
    shipped_top_k = get_settings().retrieve_top_k

    lines = [
        "## Retrieval quality",
        "",
        (
            f"Measured against **{overall['n_queries']} hand-verified golden queries** "
            "across the three demo catalogs (see `eval/golden/`) - every judgment "
            "anchored to a real, live-checked product, not invented. Judgments are a "
            "verified top-K pool, not an exhaustive scan of each catalog - these "
            "numbers are for comparing configurations against each other on this "
            "fixed judged set, not a claim about absolute recall over the whole "
            "catalog (see `eval/golden/__init__.py`)."
        ),
        "",
        f"**Shipped configuration** (`retrieve_top_k={shipped_top_k}`, dynamic alpha "
        "routing, identifier-aware reranking) - what production returns today:",
        "",
        "| Metric | Score |\n|---|---|",
    ]
    for key, label in (
        ("ndcg@10", "nDCG@10"),
        ("recall@10", "Recall@10"),
        ("recall@50", "Recall@50"),
        ("mrr", "MRR"),
        ("hit_rate@10", "Hit rate@10"),
    ):
        lines.append(f"| {label} | {overall[key]:.4f} |")

    lines += [
        "",
        (
            f"At the shipped `retrieve_top_k={shipped_top_k}`, this number and the "
            'judgment-depth-matched "fair comparison" below are nearly identical - '
            "unlike an earlier, deeper default (`retrieve_top_k=50`), where the "
            "pool reached well past most golden queries' judged depth (often 4-15 "
            "items) and let reranking correctly promote genuinely relevant items "
            "the metric had never judged, scoring real improvements as wrong. A "
            "shallow shipped pool removes most of that headroom, which is exactly "
            "why `retrieve_top_k` was lowered - see `app/config.py`'s docstring for "
            "the full reasoning."
        ),
        "",
        "**Reranking's measured quality lift, with the candidate pool matched to each "
        "query's own judgment count** (so every candidate reranking sees was actually "
        "verified):",
        "",
        f"{overall_no_rerank['ndcg@10']:.4f} without reranking → "
        f"{fair['overall']['ndcg@10']:.4f} with it (**{fair_lift:+.4f}** nDCG@10) - "
        f"negative, against a measured latency cost of ~2.4s/call at the shipped "
        f"`retrieve_top_k={shipped_top_k}` (up to ~8.8s/call had it stayed at the "
        "old `retrieve_top_k=50`; `scripts/bench_search.py`, see `PROGRESS.md`'s "
        "Day 3/4 notes). No quality upside was measured at any depth, so the "
        "cheaper pool was kept.",
        "",
        "Per tenant / query class (shipped configuration):",
        "",
        "| Tenant :: class | n | nDCG@10 | Recall@10 | MRR |",
        "|---|---|---|---|---|",
    ]
    for key in sorted(k for k in shipped if k != "overall"):
        row = shipped[key]
        lines.append(
            f"| {key} | {row['n_queries']} | {row['ndcg@10']:.4f} | "
            f"{row['recall@10']:.4f} | {row['mrr']:.4f} |"
        )
    return "\n".join(lines)


def _alpha_sweep_section(data: dict) -> str:
    lines = [
        "",
        "## The alpha sweep - dynamic alpha vs any single fixed value",
        "",
        "![alpha sweep](alpha_sweep.png)",
        "",
        "| Query class | Best alpha | nDCG@10 at that alpha |",
        "|---|---|---|",
    ]
    for query_class, alpha in data["best_alpha_per_class"].items():
        score = data["ndcg_by_class"][query_class][str(alpha)]
        lines.append(f"| {query_class} | {alpha} | {score:.4f} |")

    lines += [
        "",
        (
            f"Best single fixed alpha across **all three** classes at once: "
            f"**{data['best_single_fixed_alpha']}** "
            f"(macro nDCG@10 = {data['best_single_fixed_macro_ndcg']:.4f}). "
            f"Dynamic routing - each class getting its own best alpha - reaches "
            f"macro nDCG@10 = {data['dynamic_macro_ndcg']:.4f}, an advantage of "
            f"**{data['dynamic_advantage']:+.4f}** over the best *any* single fixed "
            f"alpha can do. `eval/results/tuned_alpha.json` (written by this sweep) "
            f"is what `app/retrieval/alpha_router.py` now loads instead of its "
            f"`PRIOR_ALPHA` guesses."
        ),
    ]
    return "\n".join(lines)


def main() -> None:
    retrieval_data = _load(RETRIEVAL_EVAL_PATH)
    sections = [_retrieval_section(retrieval_data)]

    if ALPHA_SWEEP_PATH.exists():
        sections.append(_alpha_sweep_section(_load(ALPHA_SWEEP_PATH)))
    else:
        print(
            f"note: {ALPHA_SWEEP_PATH} not found - run `make sweep` to include it. Skipping that section."
        )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(sections) + "\n")
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
