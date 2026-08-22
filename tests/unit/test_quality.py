"""Pure data-quality checks: field completeness, distributions, category concentration,
anomaly flags, deterministic sampling. No stack required - see the module docstring
in `app/ingestion/quality.py` for why these are kept separate from `eval/`.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, TypedDict

from app.ingestion.adapters.base import ParseStats
from app.ingestion.quality import (
    CategoryDiversity,
    NumericSummary,
    category_diversity,
    detect_anomalies,
    duplicate_rate_flags,
    even_sample,
    field_completeness,
    numeric_summary,
)
from app.models.product import Product


def _product(**overrides: Any) -> Product:
    defaults: dict[str, Any] = {"sku": "A1", "title": "Widget"}
    defaults.update(overrides)
    return Product(**defaults)


class _AnomalyInputs(TypedDict):
    parse_stats: ParseStats
    completeness: dict[str, float]
    price: NumericSummary
    rating: NumericSummary
    category: CategoryDiversity


# --- field_completeness --------------------------------------------------------------


def test_completeness_of_empty_catalog_is_zero_everywhere() -> None:
    result = field_completeness([])
    assert result["brand"] == 0.0
    assert result["price"] == 0.0


def test_fully_populated_product_scores_full_completeness() -> None:
    product = _product(
        brand="Acme",
        category_path=["A", "B"],
        price="9.99",
        original_price="12.99",
        currency="USD",
        rating=4.5,
        review_count=10,
        image_url="http://x/1.jpg",
        product_url="http://x/p/1",
        description="A widget.",
        attributes={"colour": "red"},
    )
    result = field_completeness([product])
    assert all(v == 1.0 for v in result.values())


def test_missing_fields_score_zero_not_partial() -> None:
    product = _product()  # only sku/title
    result = field_completeness([product])
    assert result["brand"] == 0.0
    assert result["price"] == 0.0
    assert result["category_path"] == 0.0


def test_zero_price_counts_as_populated_not_missing() -> None:
    """A real (if unusual) price of 0 must not be miscounted as absent - `is not None`
    is the correct check here, not truthiness."""
    product = _product(price="0")
    result = field_completeness([product])
    assert result["price"] == 1.0


def test_completeness_is_the_fraction_across_the_catalog() -> None:
    products = [
        _product(brand="Acme"),
        _product(brand=None),
        _product(brand=None),
        _product(brand=None),
    ]
    result = field_completeness(products)
    assert result["brand"] == 0.25


# --- numeric_summary -------------------------------------------------------------------


def test_numeric_summary_of_no_values() -> None:
    summary = numeric_summary([])
    assert summary == NumericSummary(count=0, null_count=0)


def test_numeric_summary_excludes_nulls_from_stats_but_counts_them() -> None:
    summary = numeric_summary([10.0, None, 20.0, None])
    assert summary.count == 2
    assert summary.null_count == 2
    assert summary.min == 10.0
    assert summary.max == 20.0


def test_numeric_summary_percentiles_on_a_known_sequence() -> None:
    summary = numeric_summary(list(range(1, 101)))  # 1..100
    assert summary.count == 100
    assert summary.min == 1
    assert summary.max == 100
    assert summary.median == 50 or summary.median == 51  # midpoint, index-based


# --- category_diversity ---------------------------------------------------------------


def test_category_diversity_with_no_categories() -> None:
    result = category_diversity([_product(), _product()])
    assert result.products_with_category == 0
    assert result.distinct_categories == 0
    assert result.top_share == 0.0


def test_category_diversity_counts_distinct_paths() -> None:
    products = [
        _product(category_path=["Shoes"]),
        _product(category_path=["Shoes"]),
        _product(category_path=["Bags"]),
    ]
    result = category_diversity(products)
    assert result.products_with_category == 3
    assert result.distinct_categories == 2
    assert result.top()[0] == ("Shoes", 2)


def test_category_diversity_hhi_is_one_when_perfectly_concentrated() -> None:
    products = [_product(category_path=["Shoes"]) for _ in range(50)]
    result = category_diversity(products)
    assert result.hhi == 1.0


def test_category_diversity_hhi_is_low_when_evenly_spread() -> None:
    products = [_product(category_path=[f"Cat{i}"]) for i in range(50)]
    result = category_diversity(products)
    assert result.hhi == 1 / 50  # each of 50 categories has an equal 1/50 share


def test_category_diversity_top_share_is_a_fraction_of_categorised_products_only() -> None:
    """Products with NO category at all must not dilute the degeneracy signal - a
    catalog where every categorised product shares one value is degenerate even if
    half the catalog has no category at all."""
    products = [
        _product(category_path=["Shoes"]),
        _product(category_path=["Shoes"]),
        _product(),  # no category
        _product(),  # no category
    ]
    result = category_diversity(products)
    assert result.top_share == 1.0


# --- detect_anomalies ------------------------------------------------------------------


def _clean_inputs() -> _AnomalyInputs:
    stats = ParseStats()
    stats.record_ok()
    stats.record_ok()
    return {
        "parse_stats": stats,
        "completeness": {"description": 1.0},
        "price": numeric_summary([10.0, 20.0]),
        "rating": numeric_summary([4.0, 5.0]),
        "category": CategoryDiversity(
            products_with_category=2, distinct_categories=2, counts=Counter({"A": 1, "B": 1})
        ),
    }


def test_clean_catalog_raises_no_flags() -> None:
    assert detect_anomalies(**_clean_inputs()) == []


def test_high_failure_rate_is_flagged() -> None:
    stats = ParseStats()
    for _ in range(90):
        stats.record_ok()
    for _ in range(10):
        stats.record_failure("missing title")
    inputs = _clean_inputs()
    inputs["parse_stats"] = stats
    flags = detect_anomalies(**inputs)
    assert any("failed to parse" in f.message for f in flags)


def test_concentrated_category_is_flagged_only_above_the_sample_floor() -> None:
    tiny = CategoryDiversity(
        products_with_category=5, distinct_categories=1, counts=Counter({"X": 5})
    )
    inputs = _clean_inputs()
    inputs["category"] = tiny
    assert detect_anomalies(**inputs) == []  # too small a sample to trust, HHI=1.0 notwithstanding

    large_concentrated = CategoryDiversity(
        products_with_category=1000, distinct_categories=1, counts=Counter({"X": 1000})
    )
    inputs["category"] = large_concentrated
    flags = detect_anomalies(**inputs)
    assert any("highly" in f.message and "concentrated" in f.message for f in flags)


def test_moderately_concentrated_category_gets_the_moderate_label() -> None:
    """HHI in the DOJ 0.15-0.25 band should say "moderately", not "highly" - the
    graduated bands are the point, not just a single cutoff."""
    # One category at 45%, remaining 55% spread across 11 others at 5% each:
    # HHI = 0.45^2 + 11*0.05^2 = 0.2025 + 0.0275 = 0.23 - moderate, not high.
    counts = Counter({"Big": 450})
    counts.update({f"Small{i}": 50 for i in range(11)})
    category = CategoryDiversity(products_with_category=1000, distinct_categories=12, counts=counts)
    inputs = _clean_inputs()
    inputs["category"] = category
    flags = detect_anomalies(**inputs)
    assert any("moderately" in f.message and "concentrated" in f.message for f in flags)
    assert not any("highly" in f.message for f in flags)


def test_impossible_price_is_a_critical_flag() -> None:
    inputs = _clean_inputs()
    inputs["price"] = numeric_summary([-5.0, 10.0])
    flags = detect_anomalies(**inputs)
    assert any(f.severity == "critical" and "minimum price" in f.message for f in flags)


def test_extreme_price_outlier_is_flagged() -> None:
    """A $888,888 junk row found in the real home-goods feed, among otherwise
    dollar-and-cents priced items, is exactly the case this guards against. A
    continuous 1..99 spread (not repeated-value buckets) keeps Q1/Q3 genuinely
    distinct, so Tukey's fence has a real (non-zero) IQR to work with, the same way
    real continuously-varying prices do."""
    values = list(range(1, 100)) + [888888]  # 1..99 plus one wild outlier
    inputs = _clean_inputs()
    inputs["price"] = numeric_summary([float(v) for v in values])
    flags = detect_anomalies(**inputs)
    assert any("junk value" in f.message for f in flags)


def test_wide_but_legitimate_price_range_is_not_flagged() -> None:
    """A catalog can legitimately have a long tail (electronics: cheap accessories up
    to laptops in the real seeded catalog) without being junk data. A smooth
    geometric progression from 10 to 91,000 (the real electronics min/max) has no
    discontinuous jump anywhere - every step is proportionally the same size - which
    is what makes a wide range "legitimate" rather than "one point is way off"."""
    steps = 99
    values = [10.0 * (91000.0 / 10.0) ** (i / steps) for i in range(steps + 1)]
    inputs = _clean_inputs()
    inputs["price"] = numeric_summary(values)
    flags = detect_anomalies(**inputs)
    assert not any("junk value" in f.message for f in flags)


# --- duplicate_rate_flags (cross-catalog, self-calibrating) ------------------------


def test_duplicate_rate_needs_no_flag_when_all_catalogs_are_similar() -> None:
    """G=(0.06-0.05)/stdev~1.41 at n=4, below the Grubbs critical value of 1.481."""
    rates = {"a": 0.05, "b": 0.06, "c": 0.04, "d": 0.05}
    assert duplicate_rate_flags(rates) == {}


def test_duplicate_rate_outlier_at_real_project_scale_is_flagged() -> None:
    """This is the case that motivated switching to Grubbs' test in the first place:
    at n=3 (this project's actual catalog count), the real observed rates - fashion
    0%, electronics ~0.9%, home-goods ~13.8% - give G~1.41, which a flat z>2
    threshold could never flag (mathematically capped below 1.42 at n=3) but Grubbs'
    n=3 critical value of 1.155 correctly does."""
    rates = {"fashion": 0.0, "electronics": 88 / 9600, "home": 1587 / 11503}
    flags = duplicate_rate_flags(rates)
    assert set(flags) == {"home"}


def test_duplicate_rate_outlier_with_more_peers_is_flagged() -> None:
    """Six catalogs cluster at 5%, one is at 40% - a clear peer-relative outlier
    (G~2.24 vs the n=6 critical value of 1.887)."""
    rates = {"a": 0.05, "b": 0.05, "c": 0.05, "d": 0.05, "e": 0.05, "f": 0.40}
    flags = duplicate_rate_flags(rates)
    assert set(flags) == {"f"}


def test_duplicate_rate_falls_back_to_absolute_threshold_below_three_catalogs() -> None:
    """With only 2 catalogs, Grubbs' test is undefined - falls back to the documented
    (honestly arbitrary) absolute threshold instead."""
    rates = {"a": 0.05, "b": 0.50}
    flags = duplicate_rate_flags(rates)
    assert "b" in flags
    assert "a" not in flags


def test_duplicate_rate_identical_across_all_catalogs_flags_none() -> None:
    """Zero variance among peers must not produce a division-by-zero or a spurious
    flag - every catalog behaving identically is exactly the case with nothing to
    single out."""
    rates = {"a": 0.10, "b": 0.10, "c": 0.10}
    assert duplicate_rate_flags(rates) == {}


def test_impossible_rating_is_a_critical_flag() -> None:
    inputs = _clean_inputs()
    inputs["rating"] = numeric_summary([4.0, 7.0])
    flags = detect_anomalies(**inputs)
    assert any(f.severity == "critical" and "rating above 5" in f.message for f in flags)


def test_mostly_empty_descriptions_are_flagged() -> None:
    inputs = _clean_inputs()
    inputs["completeness"] = {"description": 0.1}
    flags = detect_anomalies(**inputs)
    assert any("no description" in f.message for f in flags)


# --- even_sample -----------------------------------------------------------------------


def test_even_sample_of_empty_sequence() -> None:
    assert even_sample([], 5) == []


def test_even_sample_returns_everything_when_n_exceeds_length() -> None:
    assert even_sample([1, 2, 3], 10) == [1, 2, 3]


def test_even_sample_is_deterministic() -> None:
    items = list(range(100))
    assert even_sample(items, 10) == even_sample(items, 10)


def test_even_sample_spreads_across_the_whole_sequence() -> None:
    items = list(range(100))
    sample = even_sample(items, 5)
    assert sample[0] < 20  # near the start
    assert sample[-1] >= 80  # near the end
    assert sample == sorted(sample)  # preserves order
