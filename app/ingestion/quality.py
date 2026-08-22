"""Data-quality checks for an ingested catalog, independent of retrieval quality.

This answers a narrower question than `eval/` does. `eval/` asks "does search return
the right SKUs for a labelled query" - it needs a working search endpoint and a golden
query set, neither of which exist yet. This module asks "did the feed get parsed and
stored the way it should have" - answerable today, directly against whatever is
already ingested, and a necessary (not sufficient) condition for the retrieval numbers
to mean anything later: a degenerate category field or a silently-empty description
can't be diagnosed by nDCG, because nDCG only tells you search got worse, not why.

Everything here is pure and synchronous - no Weaviate, no Mongo - so it is unit
testable the same way `app.ingestion.pipeline`'s parsing functions are. The one
I/O-bound check that belongs with this (comparing what's stored in Weaviate against
what re-parsing the source would produce) lives in `scripts/ingestion_quality_report.py`
instead, for that reason.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from app.ingestion.adapters.base import ParseStats
from app.models.product import Product

# --- field completeness -------------------------------------------------------------

# What "populated" means per field - a Decimal of 0 or an empty string are both
# "not populated" for this purpose, distinct from Python truthiness only for the
# numeric fields, where `is not None` is deliberately used instead of truthiness so a
# real (if unusual) price/rating of 0 is not miscounted as missing.
FIELD_CHECKS: dict[str, Callable[[Product], bool]] = {
    "brand": lambda p: bool(p.brand),
    "category_path": lambda p: bool(p.category_path),
    "price": lambda p: p.price is not None,
    "original_price": lambda p: p.original_price is not None,
    "currency": lambda p: bool(p.currency),
    "rating": lambda p: p.rating is not None,
    "review_count": lambda p: p.review_count is not None,
    "image_url": lambda p: bool(p.image_url),
    "product_url": lambda p: bool(p.product_url),
    "description": lambda p: bool(p.description),
    "attributes": lambda p: bool(p.attributes),
}


def field_completeness(products: Sequence[Product]) -> dict[str, float]:
    """Fraction of products with each field populated. A catalog missing a field
    entirely (no brand column at all) shows up here as 0.0, same as a catalog that
    has the column but failed to parse it - this alone can't tell those two apart,
    which is exactly why the spot-check sample in the report exists."""
    if not products:
        return dict.fromkeys(FIELD_CHECKS, 0.0)
    n = len(products)
    return {name: sum(1 for p in products if check(p)) / n for name, check in FIELD_CHECKS.items()}


# --- numeric distributions -----------------------------------------------------------


@dataclass(frozen=True)
class NumericSummary:
    """Percentiles, not just mean/stdev - a parsing bug that corrupts a handful of
    values (a price off by 100x) hides in a mean but stands out at p99/max."""

    count: int
    null_count: int
    min: float | None = None
    p25: float | None = None
    median: float | None = None
    p75: float | None = None
    p95: float | None = None
    max: float | None = None
    mean: float | None = None


def numeric_summary(values: Sequence[float | None]) -> NumericSummary:
    present = sorted(v for v in values if v is not None)
    null_count = len(values) - len(present)
    if not present:
        return NumericSummary(count=0, null_count=null_count)

    def pct(p: float) -> float:
        idx = min(len(present) - 1, max(0, round(p * (len(present) - 1))))
        return present[idx]

    return NumericSummary(
        count=len(present),
        null_count=null_count,
        min=present[0],
        p25=pct(0.25),
        median=pct(0.5),
        p75=pct(0.75),
        p95=pct(0.95),
        max=present[-1],
        mean=statistics.mean(present),
    )


# --- category degeneracy --------------------------------------------------------------


@dataclass(frozen=True)
class CategoryDiversity:
    """Whether `category_path` is actually a useful signal or just parsed successfully.
    Both of those are different things - the electronics catalog does the latter
    without the former: every one of its 9,600 rows carries the identical single
    category, which is valid data but useless as a filter or a ranking signal.

    Stores the *full* count distribution, not just the top few - `hhi` needs every
    value's share, not an approximation from a truncated top-N, to be an honest
    concentration measure rather than a lower bound on one."""

    products_with_category: int
    distinct_categories: int
    counts: Counter[str] = field(default_factory=Counter)

    def top(self, n: int = 5) -> list[tuple[str, int]]:
        return self.counts.most_common(n)

    @property
    def top_share(self) -> float:
        if not self.products_with_category:
            return 0.0
        top = self.counts.most_common(1)
        if not top:
            return 0.0
        return top[0][1] / self.products_with_category

    @property
    def hhi(self) -> float:
        """Herfindahl-Hirschman Index: sum of squared market shares, 0 (perfectly
        diverse - every product its own category) to 1 (every product the same
        category). Not a threshold invented for this project - it's the standard
        concentration measure behind the US DOJ/FTC Horizontal Merger Guidelines,
        which publish bands: HHI < 0.15 unconcentrated, 0.15-0.25 moderately
        concentrated, > 0.25 highly concentrated. Using that external, sourced
        convention in place of a hand-picked "one value is >90% of the catalog" cutoff
        - see `detect_anomalies`."""
        if not self.products_with_category:
            return 0.0
        return sum((count / self.products_with_category) ** 2 for count in self.counts.values())


def category_diversity(products: Sequence[Product]) -> CategoryDiversity:
    labels = [" > ".join(p.category_path) for p in products if p.category_path]
    counts = Counter(labels)
    return CategoryDiversity(
        products_with_category=len(labels),
        distinct_categories=len(counts),
        counts=counts,
    )


# --- quality flags: thresholds turning the numbers above into "look at this" -------


@dataclass(frozen=True)
class QualityFlag:
    severity: str  # "warning" | "critical"
    message: str


FAILURE_RATE_WARN = 0.05
EMPTY_DESCRIPTION_WARN = 0.50

CATEGORY_DEGENERACY_MIN_SAMPLE = 20
CATEGORY_HHI_HIGH = 0.25
CATEGORY_HHI_MODERATE = 0.15
"""Herfindahl-Hirschman Index bands. Not picked by eyeballing these catalogs - these
are the published bands from the US DOJ/FTC Horizontal Merger Guidelines for market
concentration: HHI < 0.15 unconcentrated, 0.15-0.25 moderately concentrated, > 0.25
highly concentrated. HHI is the standard way to quantify "how concentrated is this
categorical distribution" in a single number that accounts for the whole distribution,
not just the largest bucket's share."""

PRICE_OUTLIER_IQR_MULTIPLIER = 3.0
"""Tukey's "far out" (extreme outlier) fence: flag a value beyond Q3 + k*IQR. k=3 is
Tukey's own convention for far-out points, distinct from the more familiar k=1.5 used
for the whiskers on an ordinary box plot - k=3 is deliberately the stricter of the
two, chosen so this does not fire on a catalog's ordinary long tail (a legitimately
expensive item in an otherwise-cheap catalog). Textbook robust-statistics practice
(Tukey, *Exploratory Data Analysis*, 1977), not a ratio invented for this project.

Applied on the **log** of price, not the raw value - real prices are right-skewed
(a few expensive items, many cheap ones; roughly log-normal), and Tukey's fence is a
linear rule. Verified this matters: a smooth, entirely legitimate geometric spread
from 10 to 91,000 (mirroring the real electronics catalog's min/max) false-positives
on *raw* Tukey - the top end of any long-tailed distribution eventually clears a
linear fence no matter how gradually it grows. Log-transforming first is the standard
fix for skewed data in outlier-detection literature, and percentiles are preserved
under a monotonic transform like log, so this needs no raw values - just `log10` of
the percentiles already computed."""

GRUBBS_CRITICAL_VALUES: dict[int, float] = {
    3: 1.155,
    4: 1.481,
    5: 1.715,
    6: 1.887,
    7: 2.020,
    8: 2.126,
    9: 2.215,
    10: 2.290,
    11: 2.355,
    12: 2.412,
    13: 2.462,
    14: 2.507,
    15: 2.549,
}
"""Grubbs' test critical values, two-sided alpha=0.05, for sample size n (source:
NIST-derived table, chem.libretexts.org "Critical Values for Grubbs' Test"). This is
the textbook-correct fix for the problem a flat z-score threshold has at small n: the
*maximum possible* z-score for one differing value among n peers is mathematically
capped at sqrt(n-1) - for n=3 that ceiling is ~1.41, meaning a flat "z > 2.0" threshold
can *never* fire with only 3 catalogs, no matter how extreme the outlier is. Grubbs'
critical values are calibrated per sample size specifically to avoid that trap. Table
stops at n=15 because this project has no realistic need for more; beyond it, the
largest tabulated value is reused, which makes the test *more* conservative (harder to
trigger) for a case denser than anything actually tested here - a safe direction."""

DUPLICATE_RATE_ABSOLUTE_FALLBACK = 0.20
"""Used only when fewer than 3 catalogs are being compared in one run - Grubbs' test
is undefined below n=3. This is honestly just as arbitrary as the threshold it
replaces; there is no way around that with too few catalogs to compare, which is
itself worth knowing rather than hiding behind false confidence."""


def detect_anomalies(
    *,
    parse_stats: ParseStats,
    completeness: dict[str, float],
    price: NumericSummary,
    rating: NumericSummary,
    category: CategoryDiversity,
) -> list[QualityFlag]:
    """Threshold checks over already-computed stats. Deliberately conservative and
    catalog-agnostic - no hardcoded expectations like "fashion should have 100% brand
    coverage", because that bakes in today's catalogs and silently stops meaning
    anything the day a fourth, differently-shaped catalog is onboarded. Only flags
    things that are anomalous by structure (a degenerate field, an impossible value),
    not by comparison to some other catalog.

    Duplicate-rate is deliberately NOT checked here - see `duplicate_rate_flags`,
    which needs every catalog in the run at once to be self-calibrating rather than
    an arbitrary fixed percentage."""
    flags: list[QualityFlag] = []

    if parse_stats.total and parse_stats.failed / parse_stats.total > FAILURE_RATE_WARN:
        rate = parse_stats.failed / parse_stats.total
        flags.append(
            QualityFlag(
                "warning",
                f"{parse_stats.failed}/{parse_stats.total} rows ({rate:.0%}) failed to parse",
            )
        )

    if (
        category.products_with_category >= CATEGORY_DEGENERACY_MIN_SAMPLE
        and category.hhi > CATEGORY_HHI_MODERATE
    ):
        band = "highly" if category.hhi > CATEGORY_HHI_HIGH else "moderately"
        (top_label, top_count), *_ = category.top(1)
        flags.append(
            QualityFlag(
                "warning",
                f"category field is {band} concentrated (HHI={category.hhi:.2f}; DOJ "
                f"merger-guideline bands: >0.25 high, 0.15-0.25 moderate): {top_label!r} "
                f"alone is {top_count / category.products_with_category:.0%} of "
                "categorised products",
            )
        )

    if price.count and price.min is not None and price.min <= 0:
        flags.append(QualityFlag("critical", f"minimum price is {price.min} - should be > 0"))

    if (
        price.count
        and price.p25 is not None
        and price.p75 is not None
        and price.max is not None
        and price.p25 > 0  # log undefined at/below 0; a non-positive price is
        # already the critical flag above, not this one
    ):
        log_p25, log_p75, log_max = (
            math.log10(price.p25),
            math.log10(price.p75),
            math.log10(price.max),
        )
        iqr = log_p75 - log_p25
        fence = log_p75 + PRICE_OUTLIER_IQR_MULTIPLIER * iqr
        if iqr > 0 and log_max > fence:
            flags.append(
                QualityFlag(
                    "warning",
                    f"max price ({price.max}) is beyond Tukey's far-out fence on a log "
                    f"scale (log10 Q3={log_p75:.2f} + {PRICE_OUTLIER_IQR_MULTIPLIER}xIQR"
                    f"[{iqr:.2f}] = {fence:.2f}, vs log10(max)={log_max:.2f}) - could be a "
                    "junk value, or the legitimate top of a distinct premium-item cluster "
                    "this test has no product-type context to tell apart from one; check "
                    "the highest-priced items directly",
                )
            )

    if rating.count and rating.max is not None and rating.max > 5:
        flags.append(QualityFlag("critical", f"a rating above 5 slipped through: max={rating.max}"))

    empty_description = 1 - completeness.get("description", 0.0)
    if empty_description > EMPTY_DESCRIPTION_WARN:
        flags.append(
            QualityFlag(
                "warning",
                f"{empty_description:.0%} of products have no description - their "
                "embeddings rely almost entirely on the title",
            )
        )

    return flags


def duplicate_rate_flags(rates_by_catalog: dict[str, float]) -> dict[str, QualityFlag]:
    """Flags a catalog's duplicate rate only when it is a statistical outlier
    *relative to its peers in this run*, not against a fixed percentage.

    Unlike the price and category checks above, there is no universal external
    standard for "how many duplicate SKUs is too many" - it depends entirely on how
    SKUs are derived, which this function cannot see, only the resulting numbers. A
    catalog with a real unique-ID column should have near-zero duplicates; a catalog
    synthesising SKUs from a title hash (home-goods) structurally has more, because
    two rows with an identical title collide by design, not by bug.

    The best available validated answer given that is self-calibration via Grubbs'
    test for a single outlier (see `GRUBBS_CRITICAL_VALUES`) against every catalog
    actually being examined in this run, testing only the highest rate - a low
    duplicate rate is never the problem. An earlier version of this used a flat
    z-score threshold, which turned out to be silently broken at this project's real
    scale: with only 3 catalogs, the maximum possible z-score for one differing value
    is mathematically capped below 1.42, so a "z > 2" rule could never fire no matter
    how extreme the outlier was - it looked validated but was untestable in practice.
    Grubbs' per-n critical values (1.155 at n=3) exist specifically to avoid that
    trap. Below n=3, Grubbs is undefined and this falls back to
    `DUPLICATE_RATE_ABSOLUTE_FALLBACK`, which is honestly just as arbitrary as the
    number it replaces - there is no way around that with too few peers to compare
    against, and pretending otherwise would be dishonest."""
    if len(rates_by_catalog) < 3:
        return {
            name: QualityFlag(
                "warning",
                f"duplicate rate {rate:.0%} exceeds the fallback threshold "
                f"({DUPLICATE_RATE_ABSOLUTE_FALLBACK:.0%}) - fewer than 3 catalogs in "
                "this run, so there is no peer baseline to self-calibrate against instead",
            )
            for name, rate in rates_by_catalog.items()
            if rate > DUPLICATE_RATE_ABSOLUTE_FALLBACK
        }

    values = list(rates_by_catalog.values())
    mean = statistics.mean(values)
    stdev = statistics.pstdev(values)
    if stdev == 0:
        return {}

    # Grubbs' test identifies the single most extreme point, not every point
    # independently - here, specifically the highest rate.
    high_name = max(rates_by_catalog, key=lambda name: rates_by_catalog[name])
    high_rate = rates_by_catalog[high_name]
    g = (high_rate - mean) / stdev
    critical = GRUBBS_CRITICAL_VALUES.get(
        len(values), GRUBBS_CRITICAL_VALUES[max(GRUBBS_CRITICAL_VALUES)]
    )
    if g > critical:
        return {
            high_name: QualityFlag(
                "warning",
                f"duplicate rate {high_rate:.0%} is a Grubbs-test outlier among this "
                f"run's {len(values)} catalogs (G={g:.2f} > critical {critical:.3f} at "
                f"alpha=0.05; peer mean {mean:.0%}) - check SKU derivation",
            )
        }
    return {}


# --- sampling for human spot-checks -------------------------------------------------


def even_sample[T](items: Sequence[T], n: int) -> list[T]:
    """`n` items spread evenly across the sequence, not the first `n` or a random
    draw. Deterministic (same input -> same sample every run, useful for a report
    someone re-runs and diffs) and less likely than "first n" to accidentally sample
    only one section of a feed that was scraped in, say, alphabetical or
    category-grouped order."""
    if n <= 0 or not items:
        return []
    if n >= len(items):
        return list(items)
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]
