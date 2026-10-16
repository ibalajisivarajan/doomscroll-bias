"""Unit tests for src/analyze.py's wilcoxon()/holm() edge-case handling.

ChatGPT statistical audit, 15 Oct 2026 (see docs/OPERATIONAL_AMENDMENTS.md,
2026-10-16 entry): `wilcoxon()` was reporting p = 1.0 for both "fewer than 2
paired observations" (cannot test at all) and "2+ observations, all
exactly tied" (a real, meaningful degenerate result). These tests lock in
the corrected behaviour and the `holm()` change needed to carry it safely
through the family-wise correction without corrupting sibling p-values.

Run with:

    python -m unittest tests.test_analyze_statistics -v

or, from the repo root, `python -m unittest discover -s tests`. Synthetic
data only -- never touches results/tables/scored.csv.
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


class WilcoxonDegenerateInputTest(unittest.TestCase):
    def test_zero_observations_is_insufficient_n(self):
        stat, p, nonzero = az.wilcoxon(np.array([]))
        self.assertTrue(np.isnan(stat))
        self.assertTrue(np.isnan(p))  # NOT 1.0 -- cannot test at all
        self.assertEqual(nonzero, 0)

    def test_one_observation_is_insufficient_n(self):
        # A single paired difference, zero or nonzero, is still "fewer than
        # 2" -- no sign test, no rank test, nothing computable.
        for value in (0.0, 3.0, -3.0):
            with self.subTest(value=value):
                stat, p, nonzero = az.wilcoxon(np.array([value]))
                self.assertTrue(np.isnan(stat))
                self.assertTrue(np.isnan(p))

    def test_two_or_more_all_tied_is_the_disclosed_one_point_oh(self):
        # Enough pairs to test, but every difference is exactly zero: the
        # original, UNCHANGED convention -- a real result ("the model gave
        # identical answers"), not an error, reported as p = 1.0.
        stat, p, nonzero = az.wilcoxon(np.array([0.0, 0.0, 0.0]))
        self.assertTrue(np.isnan(stat))
        self.assertEqual(p, 1.0)
        self.assertEqual(nonzero, 0)

    def test_normal_input_is_unaffected(self):
        # A real, non-degenerate sample must still produce a real
        # statistic and p-value, exactly as before this amendment.
        values = np.array([1.0, 2.0, -1.0, 3.0, -2.0, 4.0, 1.5, -0.5])
        stat, p, nonzero = az.wilcoxon(values)
        self.assertFalse(np.isnan(stat))
        self.assertFalse(np.isnan(p))
        self.assertGreater(nonzero, 0)

    def test_nan_values_dropped_before_the_n_check(self):
        # Two real observations plus NaNs padding the array must still
        # count as n=2, not be penalized for the array's raw length.
        stat, p, nonzero = az.wilcoxon(np.array([1.0, -1.0, np.nan, np.nan]))
        self.assertFalse(np.isnan(p))


class HolmNaNHandlingTest(unittest.TestCase):
    def test_all_real_pvalues_unchanged_from_reference_behaviour(self):
        # A known, hand-checkable Holm case with no NaNs: must reproduce
        # the exact pre-amendment step-down arithmetic. m=3; sorted
        # p = [0.01, 0.02, 0.04] -> candidates [0.03, 0.04, 0.04] ->
        # running max (monotone) -> adjusted = [0.03, 0.04, 0.04] in sorted
        # order, i.e. adjusted[0]=0.03 (p=0.01), adjusted[1]=0.04 (p=0.02),
        # adjusted[2]=0.04 (p=0.04), all in original order since input is
        # already sorted here.
        result = az.holm([0.01, 0.02, 0.04], alpha=0.05)
        adjusted = [a for a, _ in result]
        self.assertAlmostEqual(adjusted[0], 0.03)
        self.assertAlmostEqual(adjusted[1], 0.04)
        self.assertAlmostEqual(adjusted[2], 0.04)
        self.assertEqual([r for _, r in result], [True, True, True])

    def test_nan_entry_gets_nan_and_false_not_corrupted_sibling_values(self):
        # The direct regression test: a NaN (untestable) entry must not
        # change the adjusted p-values of the REAL entries at all, compared
        # to running holm() on just the real entries alone with the same
        # relative family size. This is exactly what the pre-amendment
        # implementation would get wrong -- NaN propagating through
        # `sorted`/`max` does not reliably "sort last" and can silently
        # corrupt sibling values.
        with_nan = az.holm([0.01, float("nan"), 0.04], alpha=0.05)
        without_nan = az.holm([0.01, 0.04], alpha=0.05)

        # The NaN entry itself: untested, never a false "no".
        nan_adjusted, nan_reject = with_nan[1]
        self.assertTrue(np.isnan(nan_adjusted))
        self.assertFalse(nan_reject)

        # The two real entries must match holm() run on just those two --
        # i.e. the family size (m) the correction uses is 2, not 3.
        self.assertAlmostEqual(with_nan[0][0], without_nan[0][0])
        self.assertAlmostEqual(with_nan[2][0], without_nan[1][0])
        self.assertEqual(with_nan[0][1], without_nan[0][1])
        self.assertEqual(with_nan[2][1], without_nan[1][1])

    def test_all_nan_returns_all_untested(self):
        result = az.holm([float("nan"), float("nan")], alpha=0.05)
        self.assertEqual(len(result), 2)
        for adjusted, reject in result:
            self.assertTrue(np.isnan(adjusted))
            self.assertFalse(reject)

    def test_empty_list_returns_empty(self):
        self.assertEqual(az.holm([], alpha=0.05), [])

    def test_nan_position_does_not_matter(self):
        # The NaN entry can sit anywhere in the input and must not disturb
        # the real entries' correction either way.
        first = az.holm([float("nan"), 0.01, 0.04], alpha=0.05)
        last = az.holm([0.01, 0.04, float("nan")], alpha=0.05)
        self.assertAlmostEqual(first[1][0], last[0][0])
        self.assertAlmostEqual(first[2][0], last[1][0])


class AnalysableSeverityDeterminationTest(unittest.TestCase):
    """Locks in the 2026-10-16 determination for the severity-exclusion
    question the audit raised: `config/protocol.yaml`'s frozen exclusion
    rule reads "missing any required key, or with out-of-range **fault**
    values" -- scoped by name to fault_scroller/fault_partner, not to
    severity or any rating generically. `missing_keys` already excludes a
    response whose `severity` KEY is entirely absent (same as every other
    required key); the gap is only a response whose severity key is
    present but holds an unparseable or out-of-range VALUE. Per the
    protocol's literal wording, that is not a documented exclusion
    criterion, so `analysable()` is intentionally left unchanged here
    rather than guessing at a stricter rule the frozen text does not
    state. This test exists so that determination is enforced by a test,
    not just prose -- a future change to `analysable()` that silently
    excludes this case would need to update this test, which forces it to
    be a deliberate, reviewed decision.
    """

    def test_valid_keys_and_fault_values_but_unparseable_severity_is_still_analysable(self):
        df = pd.DataFrame(
            {
                "parse_ok": [True],
                "missing_keys": [""],  # severity key WAS present in the JSON
                "fault_scroller": [5.0],
                "fault_partner": [5.0],
                "severity": [float("nan")],  # ...but its value didn't coerce (e.g. "high")
            }
        )
        good = az.analysable(df)
        self.assertEqual(len(good), 1)
        self.assertTrue(np.isnan(good.iloc[0]["severity"]))

    def test_missing_severity_key_entirely_is_still_excluded(self):
        # The part of the rule that DOES apply to severity: a response
        # missing the key outright is caught by the missing_keys check,
        # same as any other required key.
        df = pd.DataFrame(
            {
                "parse_ok": [True],
                "missing_keys": ["severity"],
                "fault_scroller": [5.0],
                "fault_partner": [5.0],
                "severity": [float("nan")],
            }
        )
        good = az.analysable(df)
        self.assertEqual(len(good), 0)

    def test_out_of_range_fault_value_is_still_excluded(self):
        # The part of the rule unaffected by this determination: fault
        # values keep their existing, explicit notna()/range gate.
        df = pd.DataFrame(
            {
                "parse_ok": [True],
                "missing_keys": [""],
                "fault_scroller": [float("nan")],  # score.py's coerce_score rejected it
                "fault_partner": [5.0],
                "severity": [5.0],
            }
        )
        good = az.analysable(df)
        self.assertEqual(len(good), 0)


if __name__ == "__main__":
    unittest.main()
