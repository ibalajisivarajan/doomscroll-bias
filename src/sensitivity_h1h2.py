#!/usr/bin/env python3
"""Supplemental, complete-case matched-pair effect-size sensitivity for H1/H2.

Implemented 2026-10-15 following the same reviewer (ChatGPT) who approved
the H3 supplemental analysis in commit a9cceb1 and then caught a parallel
completeness bug in it (fixed in 09e5645). This module exists because that
same bug pattern is also present in the PREREGISTERED H1/H2 pipeline, not
just a supplemental one -- worth stating plainly rather than quietly
patching:

src/analyze.py's `paired_gaps()` (used directly by the preregistered H1/H2
test in section 3) computes each gender's value as
`np.nanmean(list({culture: cell(gender, culture) for culture in
(north_american, south_asian)}.values()))`. If only one of the two culture
cells exists for "male" on a given vignette, `nanmean` silently returns
that single value -- a 1-culture vignette counts exactly the same as a
2-culture one in the "gap" that feeds the preregistered Wilcoxon test,
Cliff's delta, and bootstrap CI. This module does NOT touch `paired_gaps()`
or any preregistered H1/H2 output (verified: see
docs/OPERATIONAL_AMENDMENTS.md, same-cache diff method as the H3 entries).
It adds a second, descriptive, complete-case-only recomputation of the same
gap/effect-size quantities -- reported alongside, never instead of, the
preregistered test -- so the preregistered numbers can be checked against a
version that cannot be partially averaged.

WHAT "COMPLETE" MEANS HERE
---------------------------
Unlike H3 (which needed all 3 genders -- male, female, neutral -- each in
both cultures, 6 cells), the H1/H2 "gap" and "gap_culture_diff" quantities
`paired_gaps()` computes only ever use male and female (neutral never enters
the gap formula, only the separately-reported descriptive "neutral mean"
row). So completeness here is 4 cells: male x {north_american,
south_asian} and female x {north_american, south_asian}. A vignette missing
even one of those 4 is excluded from this supplemental analysis entirely --
no partial-culture averaging, no reweighted denominator -- mirroring the
H3 fix's cell-level (not individual-repeat-level) completeness check:
`cells` (analyze.cell_means()'s output) only ever contains a
(gender, culture) row with a non-NaN mean, so a row's presence already
means at least one of that cell's repeats was collected; a cell with 1-4
of its 5 repeats still counts as present.

WHY NO ANALYZE.PY IMPORT
-------------------------
Mirrors sensitivity_h3.py's convention: no src/ module currently imports
another. `wilcoxon`, `cliffs_delta` and `bootstrap_mean_ci` below are small,
independently testable reimplementations matching analyze.py's versions
exactly (same zero_method default, same percentile-bootstrap-over-vignettes
method, same default_rng(seed) convention) -- tests/test_sensitivity_h1h2.py
includes a direct parity check against analyze.paired_gaps() on synthetic
complete data, so any future drift between the two would fail loudly.

All randomness is seeded explicitly by the caller; nothing here reads the
wall clock or any global RNG state.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy import stats

REQUIRED_GENDERS = ["male", "female"]
REQUIRED_CULTURES = ["north_american", "south_asian"]
METRIC = "fault_scroller"  # the preregistered H1/H2 metric (protocol.yaml metrics.fault_gap)


# ---------------------------------------------------------------------------
# completeness
# ---------------------------------------------------------------------------
def vignette_completeness(
    cells: pd.DataFrame,
    metric: str = METRIC,
    genders: Iterable[str] = REQUIRED_GENDERS,
    cultures: Iterable[str] = REQUIRED_CULTURES,
) -> pd.DataFrame:
    """Per model_slot: n_included (all of `genders` x `cultures` cells
    present for `metric`) vs n_excluded (missing at least one), out of
    n_total vignettes with any data at all for that slot.

    `cells` is analyze.cell_means()'s output. A row's presence already
    implies a non-NaN `metric` mean (cell_means only aggregates responses
    that passed analyze.analysable()'s exclusions), so this checks cell
    presence -- not how many of a cell's 5 repeats were collected, same
    convention as sensitivity_h3.vignette_completeness.
    """
    genders = list(genders)
    cultures = list(cultures)
    required = {(g, c) for g in genders for c in cultures}
    rows: list[dict[str, Any]] = []
    for (slot, _vid), sub in cells.groupby(["model_slot", "vignette_id"]):
        present = set(zip(sub["gender"], sub["culture"]))
        rows.append({"model_slot": slot, "complete": required.issubset(present)})
    if not rows:
        return pd.DataFrame(columns=["model_slot", "n_included", "n_total", "n_excluded"])
    per_vignette = pd.DataFrame(rows)
    summary = (
        per_vignette.groupby("model_slot")["complete"]
        .agg(n_included="sum", n_total="count")
        .reset_index()
    )
    summary["n_included"] = summary["n_included"].astype(int)
    summary["n_excluded"] = summary["n_total"] - summary["n_included"]
    return summary


# ---------------------------------------------------------------------------
# complete-case paired gaps -- same formulas as analyze.paired_gaps(),
# restricted to vignettes with all 4 required cells
# ---------------------------------------------------------------------------
def complete_case_paired_gaps(
    cells: pd.DataFrame,
    metric: str = METRIC,
    genders: Iterable[str] = REQUIRED_GENDERS,
    cultures: Iterable[str] = REQUIRED_CULTURES,
) -> pd.DataFrame:
    """One row per (model_slot, vignette_id) with a COMPLETE paired cell set:
    male and female, each present in both `cultures` (4 cells). A vignette
    missing even one is excluded from this frame entirely.

    Computes exactly the quantities analyze.paired_gaps() does (gap,
    gap_north_american, gap_south_asian, gap_culture_diff), using a plain
    mean over the 2 cultures (never nanmean, since completeness is already
    guaranteed by the time a row is emitted) -- the one deliberate
    difference from paired_gaps(), which uses nanmean over whichever
    culture cells happen to exist.
    """
    genders = list(genders)
    cultures = list(cultures)
    required = {(g, c) for g in genders for c in cultures}
    wide = cells.pivot_table(
        index=["model_slot", "vignette_id"],
        columns=["gender", "culture"],
        values=metric,
        aggfunc="mean",
    )
    rows: list[dict[str, Any]] = []
    for (slot, vid), row in wide.iterrows():
        def cell(gender: str, culture: str) -> float:
            try:
                value = row[(gender, culture)]
            except KeyError:
                return float("nan")
            return float(value) if pd.notna(value) else float("nan")

        values_by_cell = {(g, c): cell(g, c) for g in genders for c in cultures}
        present = {gc for gc, v in values_by_cell.items() if not np.isnan(v)}
        if not required.issubset(present):
            continue

        male = {c: values_by_cell[("male", c)] for c in cultures}
        female = {c: values_by_cell[("female", c)] for c in cultures}
        male_mean = float(np.mean(list(male.values())))
        female_mean = float(np.mean(list(female.values())))
        rows.append(
            {
                "model_slot": slot,
                "vignette_id": vid,
                "male": male_mean,
                "female": female_mean,
                "gap": male_mean - female_mean,
                "gap_north_american": male["north_american"] - female["north_american"],
                "gap_south_asian": male["south_asian"] - female["south_asian"],
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "model_slot", "vignette_id", "male", "female",
                "gap", "gap_north_american", "gap_south_asian", "gap_culture_diff",
            ]
        )
    gaps = pd.DataFrame(rows)
    gaps["gap_culture_diff"] = gaps["gap_north_american"] - gaps["gap_south_asian"]
    return gaps


# ---------------------------------------------------------------------------
# statistics -- independently testable reimplementations of analyze.py's
# cliffs_delta / wilcoxon / bootstrap_ci, same method and conventions
# ---------------------------------------------------------------------------
def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a > b) - P(a < b), in [-1, 1]. Mirrors
    analyze.cliffs_delta exactly (rank-based, not the O(n*m) pairwise loop).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return float("nan")
    combined = np.concatenate([a, b])
    ranks = stats.rankdata(combined)
    rank_sum_a = ranks[:n].sum()
    u = rank_sum_a - n * (n + 1) / 2
    return float(2.0 * u / (n * m) - 1.0)


def wilcoxon(differences: np.ndarray, zero_method: str = "wilcox") -> tuple[float, float, int]:
    """Two-sided Wilcoxon signed-rank. Mirrors analyze.wilcoxon exactly,
    including reporting p=1.0 (not raising) on degenerate input.
    """
    differences = np.asarray(differences, float)
    differences = differences[~np.isnan(differences)]
    nonzero = int(np.count_nonzero(differences))
    if len(differences) < 2 or nonzero == 0:
        return float("nan"), 1.0, nonzero
    try:
        result = stats.wilcoxon(differences, zero_method=zero_method, alternative="two-sided")
    except ValueError:
        return float("nan"), 1.0, nonzero
    return float(result.statistic), float(result.pvalue), nonzero


def sign_statistic(values: np.ndarray) -> dict[str, Any]:
    """Matched-pair sign statistic: (n_positive - n_negative) / (n_positive
    + n_negative), per the H1/H2 Notion card's exact specification.

    Counts only the SIGN of each paired difference, ignoring magnitude --
    the deliberate complement to the magnitude-sensitive Wilcoxon/Cliff's
    delta already reported. A positive statistic means more vignettes
    favoured the male-coded direction than the female-coded one (for H1's
    "gap"; analogously for H2's "gap_culture_diff"), regardless of how large
    any individual difference was.

    Ties (a difference of exactly 0) are excluded from the statistic's own
    denominator -- the classical sign-test convention, since a tie carries
    no directional information -- but are always counted and returned
    separately (`n_tied`) so they are never silently dropped from the
    record. NaN values (should not occur on this module's own complete-case
    output, but this function is also directly unit-tested and may be
    called on hand-built input) are dropped before any counting, exactly
    like every other statistic in this module and in analyze.py.

    Returns a dict with `n_positive`, `n_negative`, `n_tied`, `n_total`
    (all non-NaN values: n_positive + n_negative + n_tied), `n_nontied`
    (n_positive + n_negative, the statistic's own denominator), and
    `statistic`. `statistic` is NaN -- never 0.0, and never a ZeroDivisionError
    -- whenever `n_nontied` is 0: both an empty sample (n_total == 0) and an
    all-tied sample (n_total > 0 but every value is exactly 0) hit this
    path, and both mean "no sign information to report," which is a
    different fact from "the signs balanced out" (which 0.0 would
    communicate.)
    """
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    n_positive = int(np.sum(values > 0))
    n_negative = int(np.sum(values < 0))
    n_tied = int(np.sum(values == 0))
    n_total = n_positive + n_negative + n_tied
    n_nontied = n_positive + n_negative
    statistic = float("nan") if n_nontied == 0 else (n_positive - n_negative) / n_nontied
    return {
        "n_positive": n_positive,
        "n_negative": n_negative,
        "n_tied": n_tied,
        "n_total": n_total,
        "n_nontied": n_nontied,
        "statistic": statistic,
    }


def bootstrap_mean_ci(
    values: np.ndarray, resamples: int, level: float, seed: int
) -> tuple[float, float, float]:
    """Percentile bootstrap on the mean, resampling vignettes (rows).
    Mirrors analyze.bootstrap_ci's resampling unit, percentile method and
    default_rng(seed) convention exactly.
    """
    values = np.asarray(values, float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(resamples, len(values)))
    means = values[idx].mean(axis=1)
    lo_pct = (1 - level) / 2 * 100
    return mean, float(np.percentile(means, lo_pct)), float(np.percentile(means, 100 - lo_pct))


# ---------------------------------------------------------------------------
# orchestration: everything analyze.py needs, per model slot
# ---------------------------------------------------------------------------
def _test_block(slot_gaps: pd.DataFrame, column: str, resamples: int, level: float, seed: int) -> dict[str, Any]:
    values = slot_gaps[column].to_numpy(dtype=float) if len(slot_gaps) else np.array([])
    stat, p, nonzero = wilcoxon(values)
    mean, lo, hi = bootstrap_mean_ci(values, resamples, level, seed)
    if slot_gaps.empty:
        delta = float("nan")
    elif column == "gap":
        delta = cliffs_delta(slot_gaps["male"].to_numpy(), slot_gaps["female"].to_numpy())
    else:
        delta = cliffs_delta(
            slot_gaps["gap_north_american"].to_numpy(), slot_gaps["gap_south_asian"].to_numpy()
        )
    return {
        "n": len(values), "nonzero": nonzero, "mean": mean, "ci_lo": lo, "ci_hi": hi,
        "stat": stat, "p": p, "delta": delta,
    }


def compute_sensitivity_h1h2(
    cells: pd.DataFrame,
    metric: str = METRIC,
    resamples: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Run the complete-case H1/H2 recomputation for every model_slot in
    `cells`.

    `cells` is analyze.cell_means()'s output -- the same frame section 3's
    preregistered test is built from; this function does not re-apply or
    alter any exclusion rule.

    Returns {model_slot: {"n_total_vignettes", "n_included_vignettes",
    "n_excluded_vignettes", "h1": {...}, "h2": {...}, "h1_sign": {...},
    "h2_sign": {...}}}. "h1"/"h2" (unchanged from the original supplemental
    analysis -- see sign_statistic()'s own addition below for what's new)
    carry n, mean, bootstrap CI, Wilcoxon stat/p (unadjusted -- never
    Holm-corrected here), and Cliff's delta, on the "gap" (H1,
    male-minus-female) and "gap_culture_diff" (H2) columns respectively.
    "h1_sign"/"h2_sign" are `sign_statistic()`'s own dict (n_positive,
    n_negative, n_tied, n_total, n_nontied, statistic) on those same two
    columns -- a separately labeled, separately computed result, not a
    replacement for or a modification of "h1"/"h2". Every model slot
    present in `cells` gets an entry, even one with zero complete-case
    vignettes (NaN results, not omitted). Each model slot gets its own seed
    offset so the two models' Monte Carlo draws do not coincide, and offset
    away from sensitivity_h3's seed ranges so the two supplemental analyses
    never reuse the same bootstrap draws either.
    """
    if cells.empty:
        return {}
    completeness = vignette_completeness(cells, metric)
    gaps = complete_case_paired_gaps(cells, metric)

    results: dict[str, dict[str, Any]] = {}
    slots = sorted(cells["model_slot"].unique())
    for i, slot in enumerate(slots):
        slot_seed = seed + 5000 * (i + 1)
        slot_gaps = gaps[gaps["model_slot"] == slot] if not gaps.empty else gaps
        comp_rows = completeness[completeness["model_slot"] == slot]
        n_total = int(comp_rows["n_total"].iloc[0]) if len(comp_rows) else 0
        n_included = int(comp_rows["n_included"].iloc[0]) if len(comp_rows) else 0
        n_excluded = int(comp_rows["n_excluded"].iloc[0]) if len(comp_rows) else 0

        gap_values = slot_gaps["gap"].to_numpy(dtype=float) if len(slot_gaps) else np.array([])
        culture_diff_values = (
            slot_gaps["gap_culture_diff"].to_numpy(dtype=float) if len(slot_gaps) else np.array([])
        )
        results[slot] = {
            "n_total_vignettes": n_total,
            "n_included_vignettes": n_included,
            "n_excluded_vignettes": n_excluded,
            "h1": _test_block(slot_gaps, "gap", resamples, level, slot_seed + 1),
            "h2": _test_block(slot_gaps, "gap_culture_diff", resamples, level, slot_seed + 2),
            "h1_sign": sign_statistic(gap_values),
            "h2_sign": sign_statistic(culture_diff_values),
        }
    return results
