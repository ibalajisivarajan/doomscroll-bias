"""Unit tests for the H3 paired-aware sensitivity analysis (supplemental).

Synthetic data only -- these exercise the statistics in src/sensitivity_h3.py
in isolation and never touch results/tables/scored.csv. Run with:

    python -m unittest tests.test_sensitivity_h3 -v

or, from the repo root, `python -m unittest discover -s tests`. No pytest
dependency: the rest of this repo has none, so this stays runnable with only
what's already in requirements.txt (pandas, numpy via pandas/scipy).

Covers the ChatGPT review of commit a9cceb1 (2026-10-16 correction):
  1. complete paired data producing the expected contrasts
  2. missing one name-cue condition for one gender
  3. missing an entire gender pair
  4. missing/unclassifiable action-category values
  5. no complete paired vignettes
  6. observed and permutation statistics operating on identical vignette sets
     (regression tests that fail against the pre-correction implementation)
  7. deterministic results with a fixed seed
  8. existing preregistered H3 outputs remaining unchanged (by construction:
     compute_sensitivity_h3 must not mutate its input frame)
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import sensitivity_h3 as sh3  # noqa: E402

CATEGORIES = ["seek_help", "boundary", "accommodate", "negotiate", "unclassified"]
GENDERS = ["male", "female", "neutral"]
CULTURES = ["north_american", "south_asian"]


def make_responses(
    n_vignettes: int,
    category_fn,
    runs_per_cell: int = 5,
    model_slot: str = "A",
    omit_cells: set[tuple] = frozenset(),
) -> pd.DataFrame:
    """Build a synthetic analysable-style frame: one row per response.

    `category_fn(vignette_id, gender, culture, run_index) -> str` decides the
    action_category for that single response. `omit_cells` is a set of
    (vignette_index, gender, culture) tuples to leave entirely out of the
    frame, simulating a (gender, culture) cell with zero collected
    responses -- the missingness cases the review asks for.
    """
    rows = []
    for v in range(n_vignettes):
        vid = f"V{v:03d}"
        for gender in GENDERS:
            for culture in CULTURES:
                if (v, gender, culture) in omit_cells:
                    continue
                for run_index in range(1, runs_per_cell + 1):
                    rows.append(
                        {
                            "model_slot": model_slot,
                            "vignette_id": vid,
                            "gender": gender,
                            "culture": culture,
                            "run_index": run_index,
                            "action_category": category_fn(vid, gender, culture, run_index),
                        }
                    )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. aggregate_proportions
# ---------------------------------------------------------------------------
class AggregateProportionsTest(unittest.TestCase):
    def test_proportions_sum_to_one_and_match_counts(self):
        def cat_fn(vid, gender, culture, run_index):
            return "boundary" if run_index <= 3 else "negotiate"

        df = make_responses(n_vignettes=2, category_fn=cat_fn)
        prop = sh3.aggregate_proportions(df, CATEGORIES)

        self.assertEqual(len(prop), 2 * len(GENDERS) * len(CULTURES))
        row = prop.iloc[0]
        self.assertEqual(row["n_runs"], 5)
        self.assertAlmostEqual(row["boundary"], 3 / 5)
        self.assertAlmostEqual(row["negotiate"], 2 / 5)
        self.assertAlmostEqual(sum(row[cat] for cat in CATEGORIES), 1.0)

    def test_unused_category_is_zero_not_missing(self):
        def cat_fn(vid, gender, culture, run_index):
            return "seek_help"

        df = make_responses(n_vignettes=1, category_fn=cat_fn, runs_per_cell=5)
        prop = sh3.aggregate_proportions(df, CATEGORIES)
        row = prop.iloc[0]
        for cat in CATEGORIES:
            self.assertAlmostEqual(row[cat], 1.0 if cat == "seek_help" else 0.0)

    def test_missing_or_unclassifiable_action_category_is_excluded_from_numerator_not_denominator(self):
        # 4: "missing action-category values". score.py's own default is
        # "unclassified" (action_category is never truly null downstream),
        # but aggregate_proportions should still behave sensibly if some rows
        # carry a value outside `categories` (e.g. None/NaN from a hand-built
        # frame): it should not inflate any category's numerator, while still
        # counting those rows in the denominator (n_runs), since they are
        # real collected responses, just not attributable to a category.
        def cat_fn(vid, gender, culture, run_index):
            return None if run_index <= 2 else "boundary"

        df = make_responses(n_vignettes=1, category_fn=cat_fn, runs_per_cell=5)
        prop = sh3.aggregate_proportions(df, CATEGORIES)
        row = prop.iloc[0]
        self.assertEqual(row["n_runs"], 5)
        self.assertAlmostEqual(row["boundary"], 3 / 5)
        for cat in CATEGORIES:
            if cat != "boundary":
                self.assertAlmostEqual(row[cat], 0.0)
        # Proportions legitimately do not sum to 1 here: 2/5 of the repeats
        # carried no attributable category at all.
        self.assertAlmostEqual(sum(row[cat] for cat in CATEGORIES), 3 / 5)


# ---------------------------------------------------------------------------
# 2/3. paired_vignette_differences -- complete-case enforcement
# ---------------------------------------------------------------------------
class PairedVignetteDifferencesTest(unittest.TestCase):
    def test_complete_paired_data_produces_expected_contrasts(self):
        # Case 1: all 6 cells present for every vignette.
        def cat_fn(vid, gender, culture, run_index):
            return "boundary" if culture == "north_american" else "negotiate"

        df = make_responses(n_vignettes=3, category_fn=cat_fn)
        prop = sh3.aggregate_proportions(df, CATEGORIES)
        diffs = sh3.paired_vignette_differences(prop, CATEGORIES, GENDERS)

        self.assertEqual(len(diffs), 3)  # nothing excluded
        for _, row in diffs.iterrows():
            self.assertAlmostEqual(row["diff_boundary"], 1.0)
            self.assertAlmostEqual(row["diff_negotiate"], -1.0)
            self.assertAlmostEqual(row["diff_seek_help"], 0.0)

    def test_missing_one_name_cue_condition_for_one_gender_excludes_the_vignette(self):
        # Case 2: vignette 0 is missing south_asian for "male" only -- one of
        # six required cells. The whole vignette must be excluded, not
        # averaged over the remaining 2 genders.
        def cat_fn(vid, gender, culture, run_index):
            return "boundary"

        df = make_responses(
            n_vignettes=2, category_fn=cat_fn,
            omit_cells={(0, "male", "south_asian")},
        )
        prop = sh3.aggregate_proportions(df, CATEGORIES)
        diffs = sh3.paired_vignette_differences(prop, CATEGORIES, GENDERS)

        self.assertEqual(set(diffs["vignette_id"]), {"V001"})
        self.assertEqual(len(diffs), 1)

    def test_missing_entire_gender_pair_excludes_the_vignette(self):
        # Case 3: vignette 0 is missing BOTH culture cells for "neutral" --
        # an entire gender pair gone, not just one side of it.
        def cat_fn(vid, gender, culture, run_index):
            return "boundary"

        df = make_responses(
            n_vignettes=2, category_fn=cat_fn,
            omit_cells={(0, "neutral", "north_american"), (0, "neutral", "south_asian")},
        )
        prop = sh3.aggregate_proportions(df, CATEGORIES)
        diffs = sh3.paired_vignette_differences(prop, CATEGORIES, GENDERS)

        self.assertEqual(set(diffs["vignette_id"]), {"V001"})
        self.assertEqual(len(diffs), 1)

    def test_no_complete_paired_vignettes_returns_empty_frame_with_columns(self):
        # Case 5: every vignette is missing something -> 0 included rows,
        # but the frame must still have the expected columns (not a bare
        # empty DataFrame that would KeyError downstream).
        def cat_fn(vid, gender, culture, run_index):
            return "boundary"

        df = make_responses(
            n_vignettes=2, category_fn=cat_fn,
            omit_cells={(0, "male", "south_asian"), (1, "female", "north_american")},
        )
        prop = sh3.aggregate_proportions(df, CATEGORIES)
        diffs = sh3.paired_vignette_differences(prop, CATEGORIES, GENDERS)

        self.assertEqual(len(diffs), 0)
        self.assertIn("diff_boundary", diffs.columns)
        self.assertIn("model_slot", diffs.columns)


class VignetteCompletenessTest(unittest.TestCase):
    def test_reports_included_and_excluded_counts_per_model(self):
        def cat_fn(vid, gender, culture, run_index):
            return "boundary"

        df_a = make_responses(
            n_vignettes=3, category_fn=cat_fn, model_slot="A",
            omit_cells={(0, "male", "south_asian")},
        )
        df_b = make_responses(n_vignettes=2, category_fn=cat_fn, model_slot="B")
        good = pd.concat([df_a, df_b], ignore_index=True)

        prop = sh3.aggregate_proportions(good, CATEGORIES)
        comp = sh3.vignette_completeness(prop, GENDERS)
        comp = comp.set_index("model_slot")

        self.assertEqual(comp.loc["A", "n_total"], 3)
        self.assertEqual(comp.loc["A", "n_included"], 2)
        self.assertEqual(comp.loc["A", "n_excluded"], 1)
        self.assertEqual(comp.loc["B", "n_total"], 2)
        self.assertEqual(comp.loc["B", "n_included"], 2)
        self.assertEqual(comp.loc["B", "n_excluded"], 0)


# ---------------------------------------------------------------------------
# bootstrap CI
# ---------------------------------------------------------------------------
class BootstrapMeanCiTest(unittest.TestCase):
    def test_symmetric_data_ci_contains_point_estimate(self):
        rng = np.random.default_rng(42)
        values = rng.normal(loc=0.1, scale=0.05, size=200)
        mean, lo, hi = sh3.bootstrap_mean_ci(values, resamples=2000, level=0.95, seed=7)
        self.assertAlmostEqual(mean, float(np.mean(values)))
        self.assertLess(lo, mean)
        self.assertGreater(hi, mean)
        self.assertLess(lo, hi)

    def test_too_few_values_returns_nan_bounds_not_a_crash(self):
        mean, lo, hi = sh3.bootstrap_mean_ci(np.array([0.3]), resamples=100, level=0.95, seed=1)
        self.assertAlmostEqual(mean, 0.3)
        self.assertTrue(np.isnan(lo))
        self.assertTrue(np.isnan(hi))

    def test_deterministic_given_same_seed(self):
        values = np.linspace(-1, 1, 50)
        a = sh3.bootstrap_mean_ci(values, resamples=500, level=0.95, seed=99)
        b = sh3.bootstrap_mean_ci(values, resamples=500, level=0.95, seed=99)
        self.assertEqual(a, b)


# ---------------------------------------------------------------------------
# 6. omnibus: observed and permutation must share the exact same vignette set
# ---------------------------------------------------------------------------
class CompleteCaseMatrixRegressionTest(unittest.TestCase):
    """Regression tests that fail against the pre-2026-10-16 implementation,
    which computed the observed statistic from each category's NaNs filtered
    independently, before the row mask used for permutation existed.
    """

    def _mixed_nan_diffs_df(self) -> pd.DataFrame:
        # Vignette V000 is missing diff_boundary only (not diff_seek_help).
        # paired_vignette_differences (post-fix) can never itself produce a
        # diffs_df like this -- a vignette's cells are either all present or
        # the whole row is excluded -- but omnibus_statistic and
        # omnibus_permutation_test must still behave correctly if a diffs_df
        # built or edited any other way disagrees between categories.
        return pd.DataFrame(
            {
                "model_slot": ["A", "A", "A"],
                "vignette_id": ["V000", "V001", "V002"],
                "diff_boundary": [np.nan, 0.2, -0.2],
                "diff_seek_help": [0.5, 0.1, -0.1],
            }
        )

    def test_complete_case_matrix_drops_a_row_with_any_nan_category(self):
        diffs_df = self._mixed_nan_diffs_df()
        values = sh3.complete_case_matrix(diffs_df, ["boundary", "seek_help"])
        # V000 has a NaN in diff_boundary -> the whole row drops, leaving 2.
        self.assertEqual(values.shape, (2, 2))

    def test_observed_and_permutation_use_identical_vignette_count(self):
        diffs_df = self._mixed_nan_diffs_df()
        observed, p, perm_stats, n_complete = sh3.omnibus_permutation_test(
            diffs_df, ["boundary", "seek_help"], resamples=200, seed=1
        )
        # Pre-fix: observed was computed from omnibus_statistic(diffs_df, ...),
        # which drops NaNs PER CATEGORY -- diff_boundary would use {0.2, -0.2}
        # (n=2) while diff_seek_help would use all 3 values -- different
        # vignette populations contributing to one "observed" number, while
        # the permutation null used the complete-case n=2. Post-fix, both
        # must agree: n_complete reflects exactly the 2 complete-case rows,
        # and `observed` is computed from that same 2-row matrix.
        self.assertEqual(n_complete, 2)
        values = sh3.complete_case_matrix(diffs_df, ["boundary", "seek_help"])
        self.assertEqual(values.shape[0], n_complete)
        expected_observed = sh3.omnibus_statistic(values)
        self.assertAlmostEqual(observed, expected_observed)

    def test_pre_fix_style_per_category_filtering_would_have_disagreed(self):
        # Explicit demonstration of the bug this correction removes: filtering
        # each category's NaNs independently (the old `omnibus_statistic`
        # behaviour) computes a DIFFERENT observed value than the corrected,
        # single-complete-case-set version, on this exact mixed-NaN input.
        diffs_df = self._mixed_nan_diffs_df()

        def pre_fix_observed(diffs_df, categories):
            total = 0.0
            for cat in categories:
                values = diffs_df[f"diff_{cat}"].to_numpy(dtype=float)
                values = values[~np.isnan(values)]
                if len(values) == 0:
                    continue
                total += float(np.mean(values)) ** 2
            return total

        old_style = pre_fix_observed(diffs_df, ["boundary", "seek_help"])
        values = sh3.complete_case_matrix(diffs_df, ["boundary", "seek_help"])
        corrected = sh3.omnibus_statistic(values)
        self.assertNotAlmostEqual(old_style, corrected)

    def test_no_complete_cases_returns_nan_not_zero(self):
        # Pre-fix: an empty/all-NaN diffs_df returned observed = 0.0 (a
        # silently-summed empty total), misreporting "no shift" instead of
        # "nothing to measure". Post-fix it must be NaN.
        diffs_df = pd.DataFrame(
            {
                "model_slot": ["A", "A"],
                "vignette_id": ["V000", "V001"],
                "diff_boundary": [np.nan, np.nan],
                "diff_seek_help": [np.nan, np.nan],
            }
        )
        observed, p, perm_stats, n_complete = sh3.omnibus_permutation_test(
            diffs_df, ["boundary", "seek_help"], resamples=200, seed=1
        )
        self.assertEqual(n_complete, 0)
        self.assertTrue(np.isnan(observed))
        self.assertTrue(np.isnan(p))
        self.assertEqual(len(perm_stats), 0)

    def test_empty_values_matrix_omnibus_statistic_is_nan_not_zero(self):
        self.assertTrue(np.isnan(sh3.omnibus_statistic(np.empty((0, 3)))))


class OmnibusPermutationTest(unittest.TestCase):
    def test_null_distribution_centers_near_zero_under_exchangeability(self):
        rng = np.random.default_rng(123)
        n = 60
        diffs_df = pd.DataFrame(
            {
                "model_slot": ["A"] * n,
                "vignette_id": [f"V{i:03d}" for i in range(n)],
                **{f"diff_{cat}": rng.normal(loc=0.0, scale=0.02, size=n) for cat in CATEGORIES},
            }
        )
        observed, p, perm_stats, n_complete = sh3.omnibus_permutation_test(
            diffs_df, CATEGORIES, resamples=5000, seed=3
        )
        self.assertEqual(n_complete, n)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        self.assertGreater(np.mean(perm_stats), 0.0)
        self.assertLess(p, 1.0)

    def test_strong_consistent_signal_is_detected(self):
        n = 40
        diffs_df = pd.DataFrame(
            {
                "model_slot": ["A"] * n,
                "vignette_id": [f"V{i:03d}" for i in range(n)],
                "diff_boundary": [0.3] * n,
                "diff_seek_help": [0.0] * n,
                "diff_accommodate": [0.0] * n,
                "diff_negotiate": [0.0] * n,
                "diff_unclassified": [0.0] * n,
            }
        )
        observed, p, perm_stats, n_complete = sh3.omnibus_permutation_test(
            diffs_df, CATEGORIES, resamples=2000, seed=11
        )
        self.assertEqual(n_complete, n)
        self.assertAlmostEqual(observed, 0.3**2)
        self.assertLess(p, 0.01)
        self.assertLess(float(np.mean(perm_stats)), observed)

    def test_degenerate_per_vignette_reading_would_be_sign_invariant(self):
        values = np.array([0.4, 0.3, 0.2, 0.1])
        per_vignette_then_mean = np.mean(values**2)
        signs = np.array([1.0, 1.0, -1.0, -1.0])
        flipped_per_vignette_then_mean = np.mean((values * signs) ** 2)
        self.assertAlmostEqual(per_vignette_then_mean, flipped_per_vignette_then_mean)

        mean_then_square = np.mean(values) ** 2
        flipped_mean_then_square = np.mean(values * signs) ** 2
        self.assertNotAlmostEqual(mean_then_square, flipped_mean_then_square)

    def test_deterministic_given_same_seed(self):
        n = 20
        diffs_df = pd.DataFrame(
            {
                "model_slot": ["A"] * n,
                "vignette_id": [f"V{i:03d}" for i in range(n)],
                **{f"diff_{cat}": np.linspace(-0.1, 0.1, n) for cat in CATEGORIES},
            }
        )
        a = sh3.omnibus_permutation_test(diffs_df, CATEGORIES, resamples=500, seed=42)
        b = sh3.omnibus_permutation_test(diffs_df, CATEGORIES, resamples=500, seed=42)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])
        self.assertTrue(np.array_equal(a[2], b[2]))
        self.assertEqual(a[3], b[3])


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
class ComputeSensitivityH3Test(unittest.TestCase):
    def test_end_to_end_on_synthetic_two_model_cache(self):
        def cat_fn(vid, gender, culture, run_index):
            return "boundary" if culture == "north_american" else "negotiate"

        df_a = make_responses(n_vignettes=10, category_fn=cat_fn, model_slot="A")
        df_b = make_responses(n_vignettes=10, category_fn=cat_fn, model_slot="B")
        good = pd.concat([df_a, df_b], ignore_index=True)

        results = sh3.compute_sensitivity_h3(
            good, CATEGORIES, GENDERS, resamples=500, level=0.95, seed=20260914
        )

        self.assertEqual(set(results.keys()), {"A", "B"})
        for slot, res in results.items():
            self.assertEqual(res["n_total_vignettes"], 10)
            self.assertEqual(res["n_included_vignettes"], 10)
            self.assertEqual(res["n_excluded_vignettes"], 0)
            self.assertEqual(res["omnibus_n_complete"], 10)
            summary = res["category_summary"]
            self.assertEqual(list(summary["action_category"]), CATEGORIES)
            boundary_row = summary[summary["action_category"] == "boundary"].iloc[0]
            self.assertAlmostEqual(boundary_row["mean_diff"], 1.0)
            self.assertTrue(0.0 <= res["omnibus_p"] <= 1.0)

    def test_no_complete_paired_vignettes_reports_nan_not_a_crash(self):
        # Case 5 at the orchestration level: every vignette for this model
        # is missing at least one cell -> n_included == 0, and the category
        # summary / omnibus results must be NaN, not an exception or a
        # silently-wrong zero.
        def cat_fn(vid, gender, culture, run_index):
            return "boundary"

        df = make_responses(
            n_vignettes=2, category_fn=cat_fn, model_slot="A",
            omit_cells={(0, "male", "south_asian"), (1, "female", "north_american")},
        )
        results = sh3.compute_sensitivity_h3(
            df, CATEGORIES, GENDERS, resamples=200, level=0.95, seed=1
        )
        res = results["A"]
        self.assertEqual(res["n_total_vignettes"], 2)
        self.assertEqual(res["n_included_vignettes"], 0)
        self.assertEqual(res["n_excluded_vignettes"], 2)
        self.assertEqual(res["omnibus_n_complete"], 0)
        self.assertTrue(np.isnan(res["omnibus_observed"]))
        self.assertTrue(np.isnan(res["omnibus_p"]))
        for row in res["category_summary"].itertuples():
            self.assertEqual(row.n_vignettes, 0)
            self.assertTrue(np.isnan(row.mean_diff))

    def test_does_not_mutate_input_good_frame(self):
        # 8: the preregistered H3 chi-square in analyze.py reads from the
        # same `good` DataFrame object this function is given. If
        # compute_sensitivity_h3 mutated it in place, the chi-square's
        # output could change even though its own code never ran
        # differently -- so this must be provably side-effect-free.
        def cat_fn(vid, gender, culture, run_index):
            return "boundary" if culture == "north_american" else "negotiate"

        good = make_responses(n_vignettes=5, category_fn=cat_fn)
        before = good.copy(deep=True)

        sh3.compute_sensitivity_h3(good, CATEGORIES, GENDERS, resamples=200, seed=1)

        pd.testing.assert_frame_equal(good, before)

    def test_deterministic_across_repeated_calls(self):
        def cat_fn(vid, gender, culture, run_index):
            return "boundary" if culture == "north_american" else "negotiate"

        good = make_responses(n_vignettes=6, category_fn=cat_fn)
        r1 = sh3.compute_sensitivity_h3(good, CATEGORIES, GENDERS, resamples=300, seed=7)
        r2 = sh3.compute_sensitivity_h3(good, CATEGORIES, GENDERS, resamples=300, seed=7)
        self.assertEqual(r1["A"]["omnibus_observed"], r2["A"]["omnibus_observed"])
        self.assertEqual(r1["A"]["omnibus_p"], r2["A"]["omnibus_p"])
        pd.testing.assert_frame_equal(
            r1["A"]["category_summary"], r2["A"]["category_summary"]
        )


if __name__ == "__main__":
    unittest.main()
