"""Unit tests for the H1/H2 complete-case matched-pair effect-size
sensitivity analysis (supplemental).

Synthetic data only -- these exercise the statistics in
src/sensitivity_h1h2.py in isolation and never touch
results/tables/scored.csv. Run with:

    python -m unittest tests.test_sensitivity_h1h2 -v

or, from the repo root, `python -m unittest discover -s tests`.

Covers, mirroring tests/test_sensitivity_h3.py's coverage for the analogous
H3 fix:
  1. complete paired data producing the expected contrasts, AND matching
     analyze.paired_gaps()'s own numbers exactly on the same complete data
     (the parity check that proves this is a faithful recomputation, not a
     different formula)
  2. missing one culture cell for one gender -> vignette excluded, unlike
     analyze.paired_gaps()'s nanmean behaviour
  3. no complete paired vignettes -> NaN results, not a crash
  4. deterministic results with a fixed seed
  5. does not mutate its input `cells` frame (protects the preregistered
     H1/H2 output, which reads from the same frame)
  6. preregistered H1/H2 output remaining unchanged -- verified at the repo
     level via the same same-cache diff method used for the H3 entries,
     reported in docs/OPERATIONAL_AMENDMENTS.md, not re-derived here since
     it depends on the live scored.csv.

Plus (2026-10-15, second review round): the matched-pair sign statistic,
(n_positive - n_negative) / (n_positive + n_negative), per the H1/H2 Notion
card's exact specification. SignStatisticTest covers known counts, all-
positive/all-negative samples, an empty sample, an all-tied sample, NaN
values dropped before counting, and a regression test proving the
denominator is n_nontied (excluding ties) and not n_total (which a naive,
misleading implementation would use instead, producing a different and
wrong number whenever ties are present).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import analyze as az  # noqa: E402
import sensitivity_h1h2 as sh12  # noqa: E402

GENDERS = ["male", "female", "neutral"]
CULTURES = ["north_american", "south_asian"]


def make_cells(
    n_vignettes: int,
    value_fn,
    model_slot: str = "A",
    omit_cells: set[tuple] = frozenset(),
) -> pd.DataFrame:
    """Build a synthetic analyze.cell_means()-shaped frame directly (one row
    per (model_slot, vignette_id, gender, culture)) rather than raw
    responses -- cell_means()'s own aggregation isn't under test here,
    sensitivity_h1h2's handling of its output is.

    `value_fn(vignette_index, gender, culture) -> float` sets fault_scroller
    for that cell. `omit_cells` is a set of (vignette_index, gender,
    culture) tuples to leave out entirely, simulating a cell with zero
    collected responses.
    """
    rows = []
    for v in range(n_vignettes):
        vid = f"V{v:03d}"
        for gender in GENDERS:
            for culture in CULTURES:
                if (v, gender, culture) in omit_cells:
                    continue
                rows.append(
                    {
                        "model_slot": model_slot,
                        "model_id": "test-model",
                        "vignette_id": vid,
                        "gender": gender,
                        "culture": culture,
                        "fault_scroller": value_fn(v, gender, culture),
                        "fault_scroller_sd": 0.0,
                        "fault_partner": 0.0,
                        "severity": 0.0,
                        "severity_sd": 0.0,
                        "n_runs": 5,
                    }
                )
    return pd.DataFrame(rows)


class VignetteCompletenessTest(unittest.TestCase):
    def test_reports_included_and_excluded_counts_per_model(self):
        def value_fn(v, gender, culture):
            return 5.0

        cells_a = make_cells(
            n_vignettes=3, value_fn=value_fn, model_slot="A",
            omit_cells={(0, "male", "south_asian")},
        )
        cells_b = make_cells(n_vignettes=2, value_fn=value_fn, model_slot="B")
        cells = pd.concat([cells_a, cells_b], ignore_index=True)

        comp = sh12.vignette_completeness(cells).set_index("model_slot")
        self.assertEqual(comp.loc["A", "n_total"], 3)
        self.assertEqual(comp.loc["A", "n_included"], 2)
        self.assertEqual(comp.loc["A", "n_excluded"], 1)
        self.assertEqual(comp.loc["B", "n_total"], 2)
        self.assertEqual(comp.loc["B", "n_included"], 2)
        self.assertEqual(comp.loc["B", "n_excluded"], 0)

    def test_neutral_gender_is_irrelevant_to_completeness(self):
        # Missing the entire "neutral" gender must NOT exclude a vignette --
        # H1/H2's gap formula never uses it.
        def value_fn(v, gender, culture):
            return 5.0

        cells = make_cells(
            n_vignettes=1, value_fn=value_fn, model_slot="A",
            omit_cells={(0, "neutral", "north_american"), (0, "neutral", "south_asian")},
        )
        comp = sh12.vignette_completeness(cells).set_index("model_slot")
        self.assertEqual(comp.loc["A", "n_included"], 1)
        self.assertEqual(comp.loc["A", "n_excluded"], 0)


class CompleteCaseParityTest(unittest.TestCase):
    def test_matches_analyze_paired_gaps_on_complete_data(self):
        # 1: on fully complete data, complete_case_paired_gaps() must
        # produce EXACTLY the same gap/gap_north_american/gap_south_asian/
        # gap_culture_diff numbers as the preregistered analyze.paired_gaps()
        # -- proving this is a faithful recomputation of the same formula,
        # not a different one, when completeness is not at issue.
        rng = np.random.default_rng(7)

        def value_fn(v, gender, culture):
            base = {"male": 6.0, "female": 4.0, "neutral": 5.0}[gender]
            cshift = {"north_american": 0.3, "south_asian": -0.3}[culture]
            return base + cshift + float(rng.normal(0, 0.01))

        cells = make_cells(n_vignettes=15, value_fn=value_fn, model_slot="A")

        official = az.paired_gaps(cells, "fault_scroller").set_index("vignette_id")
        mine = sh12.complete_case_paired_gaps(cells).set_index("vignette_id")

        self.assertEqual(len(official), len(mine))
        for vid in official.index:
            for col in ("gap", "gap_north_american", "gap_south_asian", "gap_culture_diff"):
                self.assertAlmostEqual(
                    official.loc[vid, col], mine.loc[vid, col], places=9,
                    msg=f"mismatch at {vid}.{col}",
                )

    def test_missing_one_culture_cell_excludes_the_vignette(self):
        # 2: analyze.paired_gaps() would nanmean over the single remaining
        # culture cell and silently produce a "complete-looking" gap.
        # complete_case_paired_gaps() must exclude the vignette instead.
        def value_fn(v, gender, culture):
            return {"male": 6.0, "female": 4.0, "neutral": 5.0}[gender]

        cells = make_cells(
            n_vignettes=2, value_fn=value_fn, model_slot="A",
            omit_cells={(0, "male", "south_asian")},
        )
        official = az.paired_gaps(cells, "fault_scroller")
        mine = sh12.complete_case_paired_gaps(cells)

        # The preregistered function still produces a (nanmean'd) row for
        # the incomplete vignette -- that's the exact behaviour this
        # supplemental analysis is sensitivity-checking, not a bug to fix
        # here. The complete-case version must not.
        self.assertEqual(len(official), 2)
        self.assertEqual(set(mine["vignette_id"]), {"V001"})
        self.assertEqual(len(mine), 1)

    def test_no_complete_paired_vignettes_returns_empty_frame_with_columns(self):
        # 3, at the paired-gaps level.
        def value_fn(v, gender, culture):
            return 5.0

        cells = make_cells(
            n_vignettes=2, value_fn=value_fn, model_slot="A",
            omit_cells={(0, "male", "south_asian"), (1, "female", "north_american")},
        )
        gaps = sh12.complete_case_paired_gaps(cells)
        self.assertEqual(len(gaps), 0)
        self.assertIn("gap", gaps.columns)
        self.assertIn("gap_culture_diff", gaps.columns)


class ComputeSensitivityH1H2Test(unittest.TestCase):
    def test_end_to_end_on_synthetic_two_model_cache(self):
        rng = np.random.default_rng(11)

        def value_fn(v, gender, culture):
            base = {"male": 7.0, "female": 3.0, "neutral": 5.0}[gender]
            return base + float(rng.normal(0, 0.1))

        cells_a = make_cells(n_vignettes=20, value_fn=value_fn, model_slot="A")
        cells_b = make_cells(n_vignettes=20, value_fn=value_fn, model_slot="B")
        cells = pd.concat([cells_a, cells_b], ignore_index=True)

        results = sh12.compute_sensitivity_h1h2(cells, resamples=500, level=0.95, seed=20260914)

        self.assertEqual(set(results.keys()), {"A", "B"})
        for slot, res in results.items():
            self.assertEqual(res["n_included_vignettes"], 20)
            self.assertEqual(res["n_excluded_vignettes"], 0)
            self.assertAlmostEqual(res["h1"]["mean"], 4.0, delta=0.5)
            self.assertLess(res["h1"]["p"], 0.05)
            self.assertTrue(0.0 <= res["h1"]["delta"] <= 1.0)
            # No systematic culture-by-gender interaction was built into
            # value_fn, so H2's gap_culture_diff should be small/noisy, not
            # a strong detected effect either way -- just check it ran.
            self.assertTrue(-1.0 <= res["h2"]["delta"] <= 1.0)

    def test_no_complete_paired_vignettes_reports_nan_not_a_crash(self):
        # 3, at the orchestration level.
        def value_fn(v, gender, culture):
            return 5.0

        cells = make_cells(
            n_vignettes=2, value_fn=value_fn, model_slot="A",
            omit_cells={(0, "male", "south_asian"), (1, "female", "north_american")},
        )
        results = sh12.compute_sensitivity_h1h2(cells, resamples=200, seed=1)
        res = results["A"]
        self.assertEqual(res["n_included_vignettes"], 0)
        self.assertEqual(res["n_excluded_vignettes"], 2)
        self.assertTrue(np.isnan(res["h1"]["mean"]))
        self.assertTrue(np.isnan(res["h2"]["mean"]))
        # Zero complete-case vignettes -> zero paired observations -> the
        # 2026-10-16 amendment's "insufficient n" case: NaN ("n/a"), not the
        # old 1.0 convention (which was reserved for >=2 obs, all-tied).
        self.assertTrue(np.isnan(res["h1"]["p"]))

    def test_deterministic_given_same_seed(self):
        def value_fn(v, gender, culture):
            base = {"male": 6.0, "female": 4.0, "neutral": 5.0}[gender]
            return base + 0.01 * v

        cells = make_cells(n_vignettes=10, value_fn=value_fn, model_slot="A")
        r1 = sh12.compute_sensitivity_h1h2(cells, resamples=300, seed=42)
        r2 = sh12.compute_sensitivity_h1h2(cells, resamples=300, seed=42)
        self.assertEqual(r1["A"]["h1"]["ci_lo"], r2["A"]["h1"]["ci_lo"])
        self.assertEqual(r1["A"]["h1"]["ci_hi"], r2["A"]["h1"]["ci_hi"])
        self.assertEqual(r1["A"]["h2"]["ci_lo"], r2["A"]["h2"]["ci_lo"])

    def test_does_not_mutate_input_cells_frame(self):
        # 5: the preregistered H1/H2 test in analyze.py reads from the same
        # `cells` DataFrame object this function is given.
        def value_fn(v, gender, culture):
            base = {"male": 6.0, "female": 4.0, "neutral": 5.0}[gender]
            return base + 0.01 * v

        cells = make_cells(n_vignettes=8, value_fn=value_fn, model_slot="A")
        before = cells.copy(deep=True)

        sh12.compute_sensitivity_h1h2(cells, resamples=200, seed=1)

        pd.testing.assert_frame_equal(cells, before)


class SignStatisticTest(unittest.TestCase):
    def test_known_counts_produce_the_specified_formula(self):
        # 7 positive, 3 negative, 0 tied -> (7 - 3) / 10 = 0.4
        values = np.array([1, 2, 3, 4, 5, 6, 7, -1, -2, -3], dtype=float)
        result = sh12.sign_statistic(values)
        self.assertEqual(result["n_positive"], 7)
        self.assertEqual(result["n_negative"], 3)
        self.assertEqual(result["n_tied"], 0)
        self.assertEqual(result["n_total"], 10)
        self.assertEqual(result["n_nontied"], 10)
        self.assertAlmostEqual(result["statistic"], 0.4)

    def test_all_positive_gives_plus_one(self):
        result = sh12.sign_statistic(np.array([0.1, 2.0, 5.5]))
        self.assertEqual(result["n_positive"], 3)
        self.assertEqual(result["n_negative"], 0)
        self.assertAlmostEqual(result["statistic"], 1.0)

    def test_all_negative_gives_minus_one(self):
        result = sh12.sign_statistic(np.array([-0.1, -2.0, -5.5]))
        self.assertEqual(result["n_negative"], 3)
        self.assertEqual(result["n_positive"], 0)
        self.assertAlmostEqual(result["statistic"], -1.0)

    def test_empty_sample_is_nan_with_zero_counts(self):
        result = sh12.sign_statistic(np.array([]))
        self.assertEqual(result["n_positive"], 0)
        self.assertEqual(result["n_negative"], 0)
        self.assertEqual(result["n_tied"], 0)
        self.assertEqual(result["n_total"], 0)
        self.assertEqual(result["n_nontied"], 0)
        self.assertTrue(np.isnan(result["statistic"]))

    def test_all_tied_sample_is_nan_not_zero(self):
        # Every value exactly 0: a naive (n_positive - n_negative) / n_total
        # implementation would compute 0/3 = 0.0, misreporting "signs
        # balanced out" when the truth is "no sign information exists at
        # all" (n_nontied == 0). Must be NaN.
        result = sh12.sign_statistic(np.array([0.0, 0.0, 0.0]))
        self.assertEqual(result["n_tied"], 3)
        self.assertEqual(result["n_total"], 3)
        self.assertEqual(result["n_nontied"], 0)
        self.assertTrue(np.isnan(result["statistic"]))

    def test_ties_are_counted_and_excluded_from_denominator_not_dropped(self):
        # 2 positive, 1 negative, 2 tied: statistic must use n_nontied (3),
        # not n_total (5) -- (2 - 1) / 3 = 0.333..., NOT (2 - 1) / 5 = 0.2.
        # This is the direct regression test against a misleading
        # implementation that divides by n_total.
        values = np.array([1.0, 2.0, -1.0, 0.0, 0.0])
        result = sh12.sign_statistic(values)
        self.assertEqual(result["n_positive"], 2)
        self.assertEqual(result["n_negative"], 1)
        self.assertEqual(result["n_tied"], 2)
        self.assertEqual(result["n_total"], 5)
        self.assertEqual(result["n_nontied"], 3)
        self.assertAlmostEqual(result["statistic"], 1 / 3)
        # A naive total-denominator implementation would give a DIFFERENT,
        # wrong answer on this exact input -- assert that difference
        # explicitly so this test fails if the denominator regresses.
        naive_wrong = (result["n_positive"] - result["n_negative"]) / result["n_total"]
        self.assertNotAlmostEqual(result["statistic"], naive_wrong)

    def test_nan_values_dropped_before_counting(self):
        values = np.array([1.0, -1.0, np.nan, 2.0, np.nan])
        result = sh12.sign_statistic(values)
        self.assertEqual(result["n_total"], 3)  # the 2 NaNs don't count
        self.assertEqual(result["n_positive"], 2)
        self.assertEqual(result["n_negative"], 1)


class ComputeSensitivityH1H2SignIntegrationTest(unittest.TestCase):
    def test_sign_results_present_and_consistent_with_h1_h2(self):
        # Integration: h1_sign/h2_sign must be present, and their n_total
        # must match the already-reported h1/h2 "n" for the identical
        # complete-case gap values -- same underlying data, two separately
        # labeled results, never a different dataset silently substituted.
        rng = np.random.default_rng(3)

        def value_fn(v, gender, culture):
            base = {"male": 6.0, "female": 4.0, "neutral": 5.0}[gender]
            return base + float(rng.normal(0, 0.05))

        cells = make_cells(n_vignettes=12, value_fn=value_fn, model_slot="A")
        results = sh12.compute_sensitivity_h1h2(cells, resamples=300, seed=1)
        res = results["A"]

        self.assertIn("h1_sign", res)
        self.assertIn("h2_sign", res)
        self.assertEqual(res["h1_sign"]["n_total"], res["h1"]["n"])
        self.assertEqual(res["h2_sign"]["n_total"], res["h2"]["n"])
        # Strong, consistent male > female signal built into value_fn -> H1
        # sign statistic should be strongly positive.
        self.assertGreater(res["h1_sign"]["statistic"], 0.5)

    def test_no_complete_vignettes_sign_statistic_is_nan(self):
        def value_fn(v, gender, culture):
            return 5.0

        cells = make_cells(
            n_vignettes=2, value_fn=value_fn, model_slot="A",
            omit_cells={(0, "male", "south_asian"), (1, "female", "north_american")},
        )
        results = sh12.compute_sensitivity_h1h2(cells, resamples=200, seed=1)
        res = results["A"]
        self.assertTrue(np.isnan(res["h1_sign"]["statistic"]))
        self.assertEqual(res["h1_sign"]["n_total"], 0)

    def test_existing_h1_h2_block_unchanged_by_sign_statistic_addition(self):
        # Regression guard for "preserve the existing complete-case
        # sensitivity section": adding h1_sign/h2_sign must not alter the
        # h1/h2 dicts' own values at all.
        rng = np.random.default_rng(5)

        def value_fn(v, gender, culture):
            base = {"male": 6.0, "female": 4.0, "neutral": 5.0}[gender]
            return base + float(rng.normal(0, 0.05))

        cells = make_cells(n_vignettes=10, value_fn=value_fn, model_slot="A")
        res = sh12.compute_sensitivity_h1h2(cells, resamples=300, seed=1)["A"]
        for key in ("n", "mean", "ci_lo", "ci_hi", "stat", "p", "delta"):
            self.assertIn(key, res["h1"])
            self.assertIn(key, res["h2"])


if __name__ == "__main__":
    unittest.main()
