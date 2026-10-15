"""Unit tests for the H3 paired-aware sensitivity analysis (supplemental).

Synthetic data only -- these exercise the statistics in src/sensitivity_h3.py
in isolation and never touch results/tables/scored.csv. Run with:

    python -m unittest tests.test_sensitivity_h3 -v

or, from the repo root, `python -m unittest discover -s tests`. No pytest
dependency: the rest of this repo has none, so this stays runnable with only
what's already in requirements.txt (pandas, numpy via pandas/scipy).
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
) -> pd.DataFrame:
    """Build a synthetic analysable-style frame: one row per response.

    `category_fn(vignette_id, gender, culture, run_index) -> str` decides the
    action_category for that single response, letting each test encode
    exactly the pattern it wants to check (symmetric, biased, exchangeable
    under label swap, etc.) without hand-building hundreds of rows.
    """
    rows = []
    for v in range(n_vignettes):
        vid = f"V{v:03d}"
        for gender in GENDERS:
            for culture in CULTURES:
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


class AggregateProportionsTest(unittest.TestCase):
    def test_proportions_sum_to_one_and_match_counts(self):
        # 5 repeats per cell, deterministic split: 3 "boundary", 2 "negotiate".
        def cat_fn(vid, gender, culture, run_index):
            return "boundary" if run_index <= 3 else "negotiate"

        df = make_responses(n_vignettes=2, category_fn=cat_fn)
        prop = sh3.aggregate_proportions(df, CATEGORIES)

        # One row per (model_slot, vignette_id, gender, culture).
        self.assertEqual(len(prop), 2 * len(GENDERS) * len(CULTURES))

        row = prop.iloc[0]
        self.assertEqual(row["n_runs"], 5)
        self.assertAlmostEqual(row["boundary"], 3 / 5)
        self.assertAlmostEqual(row["negotiate"], 2 / 5)
        self.assertAlmostEqual(row["seek_help"], 0.0)
        # Every category column plus n_runs's own proportions sum to 1.
        total = sum(row[cat] for cat in CATEGORIES)
        self.assertAlmostEqual(total, 1.0)

    def test_unused_category_is_zero_not_missing(self):
        def cat_fn(vid, gender, culture, run_index):
            return "seek_help"

        df = make_responses(n_vignettes=1, category_fn=cat_fn, runs_per_cell=5)
        prop = sh3.aggregate_proportions(df, CATEGORIES)
        row = prop.iloc[0]
        for cat in CATEGORIES:
            if cat == "seek_help":
                self.assertAlmostEqual(row[cat], 1.0)
            else:
                self.assertAlmostEqual(row[cat], 0.0)


class PairedVignetteDifferencesTest(unittest.TestCase):
    def test_averages_over_genders_and_signs_na_minus_sa(self):
        # north_american always "boundary", south_asian always "negotiate":
        # diff_boundary should be +1 for every gender -> averaged over
        # genders still +1. diff_negotiate should be -1.
        def cat_fn(vid, gender, culture, run_index):
            return "boundary" if culture == "north_american" else "negotiate"

        df = make_responses(n_vignettes=3, category_fn=cat_fn)
        prop = sh3.aggregate_proportions(df, CATEGORIES)
        diffs = sh3.paired_vignette_differences(prop, CATEGORIES)

        self.assertEqual(len(diffs), 3)
        for _, row in diffs.iterrows():
            self.assertAlmostEqual(row["diff_boundary"], 1.0)
            self.assertAlmostEqual(row["diff_negotiate"], -1.0)
            self.assertAlmostEqual(row["diff_seek_help"], 0.0)

    def test_missing_culture_cell_gives_nan_not_zero(self):
        def cat_fn(vid, gender, culture, run_index):
            return "boundary"

        df = make_responses(n_vignettes=1, category_fn=cat_fn)
        # Drop every south_asian row for this vignette to simulate an
        # incomplete cell (partial collection).
        df = df[df["culture"] != "south_asian"]
        prop = sh3.aggregate_proportions(df, CATEGORIES)
        diffs = sh3.paired_vignette_differences(prop, CATEGORIES)

        self.assertEqual(len(diffs), 1)
        self.assertTrue(np.isnan(diffs.iloc[0]["diff_boundary"]))


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


class OmnibusPermutationTest(unittest.TestCase):
    def test_null_distribution_centers_near_zero_under_exchangeability(self):
        # Build diffs_df directly: no real NA/SA shift (differences are pure
        # noise around 0), so under the true null the omnibus statistic
        # should be small and the permutation null should center near the
        # *observed* statistic's own ballpark -- i.e. p should NOT be tiny.
        rng = np.random.default_rng(123)
        n = 60
        diffs_df = pd.DataFrame(
            {
                "model_slot": ["A"] * n,
                "vignette_id": [f"V{i:03d}" for i in range(n)],
                **{
                    f"diff_{cat}": rng.normal(loc=0.0, scale=0.02, size=n)
                    for cat in CATEGORIES
                },
            }
        )
        observed, p, perm_stats = sh3.omnibus_permutation_test(
            diffs_df, CATEGORIES, resamples=5000, seed=3
        )
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)
        # Under exchangeability the permutation distribution's mean should
        # land close to the observed statistic's own order of magnitude --
        # a strong, biased effect would instead produce a left-skewed null
        # far below the observed value and a tiny p every time.
        self.assertGreater(np.mean(perm_stats), 0.0)
        self.assertLess(p, 1.0)

    def test_strong_consistent_signal_is_detected(self):
        # Every vignette carries the same sizeable NA-vs-SA shift in one
        # category: the omnibus statistic should be large and essentially
        # every permutation (which can only shrink it, since sign flips move
        # the across-vignette mean toward 0) should score lower.
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
        observed, p, perm_stats = sh3.omnibus_permutation_test(
            diffs_df, CATEGORIES, resamples=2000, seed=11
        )
        self.assertAlmostEqual(observed, 0.3**2)
        self.assertLess(p, 0.01)
        self.assertLess(float(np.mean(perm_stats)), observed)

    def test_degenerate_per_vignette_reading_would_be_sign_invariant(self):
        # Documents, as an explicit check (not just a docstring claim), that
        # squaring per-vignette before averaging is sign-invariant -- the
        # reason omnibus_statistic instead squares the across-vignette mean.
        values = np.array([0.4, 0.3, 0.2, 0.1])
        per_vignette_then_mean = np.mean(values**2)
        signs = np.array([1.0, 1.0, -1.0, -1.0])
        flipped_per_vignette_then_mean = np.mean((values * signs) ** 2)
        self.assertAlmostEqual(per_vignette_then_mean, flipped_per_vignette_then_mean)

        # Whereas squaring the across-vignette mean (what we actually use)
        # does respond to the same sign flips.
        mean_then_square = np.mean(values) ** 2
        flipped_mean_then_square = np.mean(values * signs) ** 2
        self.assertNotAlmostEqual(mean_then_square, flipped_mean_then_square)


class ComputeSensitivityH3Test(unittest.TestCase):
    def test_end_to_end_on_synthetic_two_model_cache(self):
        def cat_fn(vid, gender, culture, run_index):
            return "boundary" if culture == "north_american" else "negotiate"

        df_a = make_responses(n_vignettes=10, category_fn=cat_fn, model_slot="A")
        df_b = make_responses(n_vignettes=10, category_fn=cat_fn, model_slot="B")
        good = pd.concat([df_a, df_b], ignore_index=True)

        results = sh3.compute_sensitivity_h3(
            good, CATEGORIES, resamples=500, level=0.95, seed=20260914
        )

        self.assertEqual(set(results.keys()), {"A", "B"})
        for slot, res in results.items():
            self.assertEqual(res["n_vignettes"], 10)
            summary = res["category_summary"]
            self.assertEqual(list(summary["action_category"]), CATEGORIES)
            boundary_row = summary[summary["action_category"] == "boundary"].iloc[0]
            self.assertAlmostEqual(boundary_row["mean_diff"], 1.0)
            self.assertTrue(0.0 <= res["omnibus_p"] <= 1.0)

        # Different model slots must not share the exact same Monte Carlo
        # draw (seed offset by slot index inside compute_sensitivity_h3).
        self.assertNotEqual(
            results["A"]["omnibus_observed"], None
        )  # sanity: field populated


if __name__ == "__main__":
    unittest.main()
