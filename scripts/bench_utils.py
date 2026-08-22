"""Shared plumbing for the benchmark scripts (`bench_ingestion.py`, `bench_search.py`).

Kept tiny and dependency-free on purpose - these are the only two functions both
scripts need, and the reporting format is what makes their outputs comparable.
"""

from __future__ import annotations

import statistics


def percentiles(values: list[float]) -> tuple[float, float, float]:
    """(p50, p95, p99) via nearest-rank on the sorted sample - no interpolation, so a
    result is always a value that was actually observed, never an invented one."""
    s = sorted(values)
    n = len(s)

    def pct(p: float) -> float:
        idx = min(n - 1, max(0, round(p * (n - 1))))
        return s[idx]

    return pct(0.50), pct(0.95), pct(0.99)


def print_latency_table(rows: list[tuple[str, list[float]]]) -> None:
    """One row per label: p50/p95/p99 in milliseconds, from a list of second-valued
    latency samples."""
    print(f"{'label':>28}  {'n':>5}  {'p50 ms':>8}  {'p95 ms':>8}  {'p99 ms':>8}  {'mean ms':>8}")
    for label, values in rows:
        if not values:
            continue
        p50, p95, p99 = percentiles(values)
        print(
            f"{label:>28}  {len(values):>5}  {p50 * 1000:>8.1f}  {p95 * 1000:>8.1f}  "
            f"{p99 * 1000:>8.1f}  {statistics.mean(values) * 1000:>8.1f}"
        )
