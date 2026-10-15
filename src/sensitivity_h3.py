#!/usr/bin/env python3
"""Supplemental, paired-aware sensitivity analysis for H3 (exploratory).

Spec source: Notion card "Specify a paired-aware H3 sensitivity analysis,
clearly labeled supplemental/post-lock" (statistical design authored by
ChatGPT, 2026-10-15; implemented 2026-10-15 per the handover in
docs/OPERATIONAL_AMENDMENTS.md).

WHY THIS EXISTS, AND WHAT IT IS NOT
------------------------------------
The preregistered H3 test in src/analyze.py (section 5) is a chi-square of
`action_category` by `culture` computed directly on individual responses. It
treats the 5 repeated runs of a cell as 5 independent observations. They are
not: 5 runs of the same (model, vignette, condition) cell measure decoding
instability at temperature 0, not 5 independent draws -- exactly the same
reason analyze.py collapses repeats to a cell mean before testing H1/H2. A
reviewer could reasonably flag the H3 chi-square's implicit n-inflation.

This module does NOT replace that chi-square, does NOT touch its code, and
is not folded into its reported p-value or the secondary-family Holm
correction. It is a separate, additive, descriptive robustness check: does a
paired-and-repeat-aware version of the same question point the same way?
Both results are reported in results.md; the preregistered one is the one
that counts against the preregistered plan.

THE FOUR STEPS (see docs/OPERATIONAL_AMENDMENTS.md 2026-10-15 for the
original written spec this implements)
----------------------------------------------------------------------
1. aggregate_proportions: collapse each (model, vignette, gender, culture)
   cell's 5 repeats into one proportion vector over the action categories.
2. paired_vignette_differences: for each vignette, the north_american minus
   south_asian proportion difference per category, averaged over the three
   gender conditions -- one signed number per (model, vignette, category).
3. category_bootstrap_summary: percentile bootstrap CI on the mean of that
   per-vignette difference, resampling whole vignettes (never individual
   responses) -- mirrors analyze.bootstrap_ci's resampling unit and percentile
   method exactly. This module does not import analyze.py (no src/ module
   currently imports another; bootstrap_mean_ci is a small, independently
   testable reimplementation of the same approach).
4. omnibus_permutation_test: a single joint statistic combining all
   categories, with a permutation null built by independently sign-flipping
   each vignette's entire diff vector (equivalent to swapping which culture
   is labelled north_american/south_asian for that vignette, preserving the
   within-vignette pairing across all three genders at once).

A note on the omnibus statistic's definition, because the written spec's
wording is ambiguous and worth being explicit about: the statistic is
sum_over_categories( (mean_over_vignettes of diff_category) ** 2 ), i.e.
square the *across-vignette mean* for each category, then sum the squares.
The alternative reading -- square each vignette's own per-category
difference and average *that* over vignettes -- is invariant under the
per-vignette sign flips used for the permutation null (a single vignette's
squared contribution does not depend on its own sign), which would make
every permutation reproduce the observed statistic exactly and the test
degenerate (Monte Carlo p ~= 1 always, by construction, regardless of the
data). Squaring the across-vignette mean first is what makes the statistic
respond to the permutation and gives a real null distribution.

All randomness is seeded explicitly by the caller; nothing here reads the
wall clock or any global RNG state.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

DEFAULT_CATEGORIES = ["seek_help", "boundary", "accommodate", "negotiate", "unclassified"]
DEFAULT_GENDERS = ["male", "female", "neutral"]
CULTURE_A = "north_american"
CULTURE_B = "south_asian"


# ---------------------------------------------------------------------------
# step 1: proportion-vector aggregation
# ---------------------------------------------------------------------------
def aggregate_proportions(
    df: pd.DataFrame, categories: Iterable[str] = DEFAULT_CATEGORIES
) -> pd.DataFrame:
    """One row per (model_slot, vignette_id, gender, culture): the proportion
    of that cell's repeats falling in each action category (sums to 1 when
    n_runs > 0). `df` should already be the analysable, exclusion-filtered
    frame (the same one analyze.py's existing H3 chi-square uses), so this
    collapses exactly the repeats that section 5 counts -- no more, no fewer.
    """
    categories = list(categories)
    rows: list[dict[str, Any]] = []
    for (slot, vid, gender, culture), sub in df.groupby(
        ["model_slot", "vignette_id", "gender", "culture"], dropna=False
    ):
        n = len(sub)
        counts = sub["action_category"].value_counts()
        row: dict[str, Any] = {
            "model_slot": slot,
            "vignette_id": vid,
            "gender": gender,
            "culture": culture,
            "n_runs": n,
        }
        for cat in categories:
            row[cat] = float(counts.get(cat, 0)) / n if n else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# step 2: within-vignette name-cue contrast, averaged over genders
# ---------------------------------------------------------------------------
def paired_vignette_differences(
    prop_df: pd.DataFrame,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    genders: Iterable[str] = DEFAULT_GENDERS,
    culture_a: str = CULTURE_A,
    culture_b: str = CULTURE_B,
) -> pd.DataFrame:
    """One row per (model_slot, vignette_id): for each category, the mean
    over available genders of (proportion in `culture_a` minus proportion in
    `culture_b`). A vignette/category is NaN only when neither culture's cell
    exists for any gender -- same missing-cell semantics as the rest of the
    pipeline, never silently dropped versus silently zeroed.
    """
    categories = list(categories)
    genders = list(genders)
    rows: list[dict[str, Any]] = []
    for (slot, vid), sub in prop_df.groupby(["model_slot", "vignette_id"]):
        by_gc = {(r["gender"], r["culture"]): r for r in sub.to_dict("records")}
        row: dict[str, Any] = {"model_slot": slot, "vignette_id": vid}
        for cat in categories:
            diffs = []
            for gender in genders:
                a = by_gc.get((gender, culture_a))
                b = by_gc.get((gender, culture_b))
                if a is not None and b is not None:
                    diffs.append(a[cat] - b[cat])
            row[f"diff_{cat}"] = float(np.mean(diffs)) if diffs else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# step 3: bootstrap CIs, resampling whole vignettes
# ---------------------------------------------------------------------------
def bootstrap_mean_ci(
    values: np.ndarray, resamples: int, level: float, seed: int
) -> tuple[float, float, float]:
    """Percentile bootstrap on the mean. Resamples *rows* of `values` --
    i.e. whole vignettes, since each row here is already one vignette's
    gender-averaged, culture-paired difference. Mirrors
    analyze.bootstrap_ci's resampling unit, percentile method and
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


def category_bootstrap_summary(
    diffs_df: pd.DataFrame,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    resamples: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> pd.DataFrame:
    """Per category: n vignettes contributing, mean paired difference, CI.

    `diffs_df` is expected to already be restricted to one model_slot (the
    spec runs this "for each model_slot, independently"). Each category gets
    its own seed offset (seed + category index), the same pattern
    analyze.py uses to avoid reusing one bootstrap draw across tests.
    """
    categories = list(categories)
    rows: list[dict[str, Any]] = []
    for i, cat in enumerate(categories):
        values = diffs_df[f"diff_{cat}"].to_numpy(dtype=float)
        mean, lo, hi = bootstrap_mean_ci(values, resamples, level, seed + i)
        n = int(np.count_nonzero(~np.isnan(values)))
        rows.append(
            {"action_category": cat, "n_vignettes": n, "mean_diff": mean, "ci_lo": lo, "ci_hi": hi}
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# step 4: omnibus permutation check
# ---------------------------------------------------------------------------
def omnibus_statistic(diffs_df: pd.DataFrame, categories: Iterable[str]) -> float:
    """sum_over_categories( (mean_over_vignettes of diff_category) ** 2 ).

    See the module docstring for why this reading (square the across-vignette
    mean, not the per-vignette value) is the one that makes the permutation
    test non-degenerate.
    """
    total = 0.0
    for cat in categories:
        values = diffs_df[f"diff_{cat}"].to_numpy(dtype=float)
        values = values[~np.isnan(values)]
        if len(values) == 0:
            continue
        total += float(np.mean(values)) ** 2
    return total


def omnibus_permutation_test(
    diffs_df: pd.DataFrame,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, np.ndarray]:
    """Monte Carlo permutation test on `omnibus_statistic`.

    Planned null (stated in words): within each vignette, which culture
    condition is labelled north_american versus south_asian is exchangeable
    -- i.e. the paired difference for that vignette is as likely to have
    come out with either sign. Each permutation independently flips the sign
    of every category's diff for a given vignette with probability 0.5
    (equivalent to swapping the two culture labels for all three gender
    variants of that vignette at once, since diff_category is already the
    gender-averaged contrast), recomputes the omnibus statistic, and the
    Monte Carlo p-value is the fraction of permuted statistics at least as
    extreme as observed, with the standard +1/+1 correction so a finite
    number of resamples never reports exactly p = 0:

        p = (1 + #{permuted statistic >= observed}) / (1 + resamples)

    Returns (observed_statistic, p_value, permutation_distribution).
    """
    categories = list(categories)
    cat_cols = [f"diff_{c}" for c in categories]
    observed = omnibus_statistic(diffs_df, categories)

    values = diffs_df[cat_cols].to_numpy(dtype=float)  # (n_vignettes, n_categories)
    valid_rows = ~np.isnan(values).any(axis=1)
    values = values[valid_rows]
    n_vignettes = values.shape[0]
    if n_vignettes == 0:
        return observed, float("nan"), np.array([])

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, n_vignettes))
    # (resamples, n_vignettes, n_categories): each vignette's whole diff
    # vector flips together, never individual categories independently.
    flipped = values[None, :, :] * signs[:, :, None]
    cat_means = flipped.mean(axis=1)  # (resamples, n_categories)
    perm_stats = (cat_means**2).sum(axis=1)  # (resamples,)

    p = float((1 + np.count_nonzero(perm_stats >= observed)) / (1 + resamples))
    return observed, p, perm_stats


# ---------------------------------------------------------------------------
# orchestration: everything analyze.py needs, per model slot
# ---------------------------------------------------------------------------
def compute_sensitivity_h3(
    good: pd.DataFrame,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    resamples: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Run steps 1-4 for every model_slot present in `good`.

    `good` is the same analysable, exclusion-filtered frame analyze.py
    already builds (df -> analysable(df)) and passes into the existing H3
    chi-square block; this function does not re-apply or alter any
    exclusion rule.

    Returns {model_slot: {"n_vignettes", "category_summary" (DataFrame),
    "omnibus_observed", "omnibus_p"}}. Each model_slot gets its own seed
    offset so the two models' Monte Carlo draws do not coincide.
    """
    categories = list(categories)
    prop_df = aggregate_proportions(good, categories)
    diffs_df = paired_vignette_differences(prop_df, categories)

    results: dict[str, dict[str, Any]] = {}
    for i, (slot, sub) in enumerate(diffs_df.groupby("model_slot")):
        slot_seed = seed + 1000 * (i + 1)
        category_summary = category_bootstrap_summary(
            sub, categories, resamples, level, slot_seed
        )
        observed, p, _ = omnibus_permutation_test(sub, categories, resamples, slot_seed + 500)
        results[slot] = {
            "n_vignettes": int(sub["vignette_id"].nunique()),
            "category_summary": category_summary,
            "omnibus_observed": observed,
            "omnibus_p": p,
        }
    return results
