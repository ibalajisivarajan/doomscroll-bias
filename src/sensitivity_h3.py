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
   Complete-case only (OPERATIONAL AMENDMENT, 2026-10-16, ChatGPT review of
   commit a9cceb1): a vignette must have all three genders present in BOTH
   culture conditions (6 cells) to be included at all. The original
   implementation averaged over however many gender pairs happened to
   exist, which could silently reweight a vignette with 1 available gender
   pair the same as one with 3. `vignette_completeness` reports exactly how
   many vignettes were included versus excluded, per model slot, so the
   exclusion is never silent.
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
   within-vignette pairing across all three genders at once). Both the
   observed and the permuted statistics are computed from the exact same
   complete-case `values` matrix, established once by `complete_case_matrix`
   before either is touched (OPERATIONAL AMENDMENT, 2026-10-16: the original
   implementation computed the observed statistic by filtering each
   category's NaNs independently, before the complete-case row mask used
   for the permutation existed -- for data produced by this module's own
   paired_vignette_differences the two happened to select the same rows
   (a vignette's NaN-ness never differs by category there), but a diffs_df
   built or edited any other way could silently disagree, and an empty
   input returned 0.0 -- "no shift detected" -- instead of NaN -- "nothing
   to measure").

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
    """One row per (model_slot, vignette_id) with a COMPLETE paired cell set:
    every gender in `genders` present in BOTH `culture_a` and `culture_b`
    (6 cells for the default 3 genders). A vignette missing even one of
    those cells is excluded from this frame entirely -- no partial-gender
    averaging and no reweighted denominator. Call `vignette_completeness`
    for the per-model-slot included/excluded counts this produces.

    For each category, the included row's value is the mean over all
    `genders` of (proportion in `culture_a` minus proportion in
    `culture_b`).
    """
    categories = list(categories)
    genders = list(genders)
    required = {(g, culture_a) for g in genders} | {(g, culture_b) for g in genders}
    rows: list[dict[str, Any]] = []
    for (slot, vid), sub in prop_df.groupby(["model_slot", "vignette_id"]):
        by_gc = {(r["gender"], r["culture"]): r for r in sub.to_dict("records")}
        if not required.issubset(by_gc.keys()):
            continue
        row: dict[str, Any] = {"model_slot": slot, "vignette_id": vid}
        for cat in categories:
            diffs = [
                by_gc[(gender, culture_a)][cat] - by_gc[(gender, culture_b)][cat]
                for gender in genders
            ]
            row[f"diff_{cat}"] = float(np.mean(diffs))
        rows.append(row)
    if not rows:
        return pd.DataFrame(columns=["model_slot", "vignette_id"] + [f"diff_{c}" for c in categories])
    return pd.DataFrame(rows)


def vignette_completeness(
    prop_df: pd.DataFrame,
    genders: Iterable[str] = DEFAULT_GENDERS,
    culture_a: str = CULTURE_A,
    culture_b: str = CULTURE_B,
) -> pd.DataFrame:
    """Per model_slot: n_included (all 6 required cells present) vs
    n_excluded (missing at least one), out of n_total vignettes with any
    data at all for that slot. Exists so `paired_vignette_differences`'s
    complete-case exclusion is never silent.
    """
    genders = list(genders)
    required = {(g, culture_a) for g in genders} | {(g, culture_b) for g in genders}
    rows: list[dict[str, Any]] = []
    for (slot, _vid), sub in prop_df.groupby(["model_slot", "vignette_id"]):
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
def complete_case_matrix(diffs_df: pd.DataFrame, categories: Iterable[str]) -> np.ndarray:
    """(n_vignettes, n_categories) array restricted to rows where every one
    of `categories`' diff column is non-NaN.

    This is the single source of truth for which vignettes count toward the
    omnibus statistic -- established once, before either the observed or
    the permuted statistic is computed, so the two can never silently
    disagree on the vignette population (OPERATIONAL AMENDMENT, 2026-10-16:
    see the module docstring's step 4 note). `paired_vignette_differences`
    already returns a complete-case frame with no NaNs left in it, so in the
    normal pipeline this is a no-op filter -- it exists so these functions
    are also correct in isolation, against any diffs_df built by hand.
    """
    categories = list(categories)
    cat_cols = [f"diff_{c}" for c in categories]
    if diffs_df.empty or not set(cat_cols).issubset(diffs_df.columns):
        return np.empty((0, len(categories)))
    values = diffs_df[cat_cols].to_numpy(dtype=float)
    valid = ~np.isnan(values).any(axis=1)
    return values[valid]


def omnibus_statistic(values: np.ndarray) -> float:
    """sum_over_categories( (mean_over_vignettes of diff_category) ** 2 ),
    computed on an already complete-case `values` matrix of shape
    (n_vignettes, n_categories) -- see `complete_case_matrix`.

    NaN when there are zero complete-case rows, never 0.0: "no complete
    cases to measure from" is a different fact from "measured a zero
    shift", and conflating them (the original implementation returned 0.0
    for an empty input) would misreport an empty sensitivity analysis as a
    null result.

    See the module docstring for why this reading (square the across-vignette
    mean, not the per-vignette value) is the one that makes the permutation
    test non-degenerate.
    """
    if values.shape[0] == 0:
        return float("nan")
    cat_means = values.mean(axis=0)
    return float(np.sum(cat_means**2))


def omnibus_permutation_test(
    diffs_df: pd.DataFrame,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    resamples: int = 10000,
    seed: int = 0,
) -> tuple[float, float, np.ndarray, int]:
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

    The complete-case vignette set is established once (`complete_case_matrix`)
    and used for both the observed statistic and every permutation -- they
    are never computed over different vignette populations.

    Returns (observed_statistic, p_value, permutation_distribution,
    n_complete_vignettes) -- the last so a caller can report the exact
    sample size the statistic was actually computed from.
    """
    categories = list(categories)
    values = complete_case_matrix(diffs_df, categories)
    n_vignettes = values.shape[0]
    observed = omnibus_statistic(values)
    if n_vignettes == 0:
        return observed, float("nan"), np.array([]), 0

    rng = np.random.default_rng(seed)
    signs = rng.choice(np.array([-1.0, 1.0]), size=(resamples, n_vignettes))
    # (resamples, n_vignettes, n_categories): each vignette's whole diff
    # vector flips together, never individual categories independently.
    flipped = values[None, :, :] * signs[:, :, None]
    cat_means = flipped.mean(axis=1)  # (resamples, n_categories)
    perm_stats = (cat_means**2).sum(axis=1)  # (resamples,)

    p = float((1 + np.count_nonzero(perm_stats >= observed)) / (1 + resamples))
    return observed, p, perm_stats, n_vignettes


# ---------------------------------------------------------------------------
# orchestration: everything analyze.py needs, per model slot
# ---------------------------------------------------------------------------
def compute_sensitivity_h3(
    good: pd.DataFrame,
    categories: Iterable[str] = DEFAULT_CATEGORIES,
    genders: Iterable[str] = DEFAULT_GENDERS,
    resamples: int = 10000,
    level: float = 0.95,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Run steps 1-4 for every model_slot present in `good`.

    `good` is the same analysable, exclusion-filtered frame analyze.py
    already builds (df -> analysable(df)) and passes into the existing H3
    chi-square block; this function does not re-apply or alter any
    exclusion rule.

    Returns {model_slot: {"n_total_vignettes", "n_included_vignettes",
    "n_excluded_vignettes", "category_summary" (DataFrame),
    "omnibus_observed", "omnibus_p", "omnibus_n_complete"}}. Every model
    slot present in `good` gets an entry, even one with zero complete-case
    vignettes (reported as NaN bootstrap/omnibus results, not omitted --
    see the handover's "no complete paired vignettes" test case). Each
    model_slot gets its own seed offset so the two models' Monte Carlo
    draws do not coincide.
    """
    categories = list(categories)
    genders = list(genders)
    prop_df = aggregate_proportions(good, categories)
    completeness = vignette_completeness(prop_df, genders)
    diffs_df = paired_vignette_differences(prop_df, categories, genders)

    results: dict[str, dict[str, Any]] = {}
    if prop_df.empty:
        return results
    slots = sorted(prop_df["model_slot"].unique())
    for i, slot in enumerate(slots):
        slot_seed = seed + 1000 * (i + 1)
        slot_diffs = diffs_df[diffs_df["model_slot"] == slot]
        comp_rows = completeness[completeness["model_slot"] == slot]
        n_total = int(comp_rows["n_total"].iloc[0]) if len(comp_rows) else 0
        n_included = int(comp_rows["n_included"].iloc[0]) if len(comp_rows) else 0
        n_excluded = int(comp_rows["n_excluded"].iloc[0]) if len(comp_rows) else 0

        category_summary = category_bootstrap_summary(
            slot_diffs, categories, resamples, level, slot_seed
        )
        observed, p, _, n_complete = omnibus_permutation_test(
            slot_diffs, categories, resamples, slot_seed + 500
        )
        results[slot] = {
            "n_total_vignettes": n_total,
            "n_included_vignettes": n_included,
            "n_excluded_vignettes": n_excluded,
            "category_summary": category_summary,
            "omnibus_observed": observed,
            "omnibus_p": p,
            "omnibus_n_complete": n_complete,
        }
    return results
