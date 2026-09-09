from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.selection_bias import (
    ARGMAX,
    ESTIMATORS,
    MATCHED,
    PLATEAU,
    common_length,
    compare_estimators,
    delta_at,
    exposure_gap,
    off_metric_delta_at_selection,
    selected_iteration,
)


def make_curve(average_precision, auc=None):
    """A learning curve frame with the columns the trainers persist."""
    average_precision = np.asarray(average_precision, dtype=float)
    if auc is None:
        auc = np.full_like(average_precision, 0.9)
    return pd.DataFrame(
        {
            "iteration": np.arange(1, len(average_precision) + 1),
            "validation_average_precision": average_precision,
            "validation_auc": np.asarray(auc, dtype=float),
        }
    )


class ExposureGapTests(unittest.TestCase):
    def test_reports_which_arm_trained_longer(self):
        reference = make_curve(np.linspace(0.5, 0.6, 100))
        variant = make_curve(np.linspace(0.5, 0.6, 250))
        gap = exposure_gap(reference, variant)
        self.assertEqual(gap["reference_rounds"], 100)
        self.assertEqual(gap["variant_rounds"], 250)
        self.assertEqual(gap["round_gap"], 150)
        self.assertEqual(gap["absolute_round_gap"], 150)
        self.assertEqual(gap["longer_trained"], "variant")
        self.assertEqual(gap["common_rounds"], 100)
        self.assertAlmostEqual(gap["round_ratio"], 2.5)

    def test_negative_gap_when_the_reference_trained_longer(self):
        gap = exposure_gap(make_curve(np.zeros(300)), make_curve(np.zeros(120)))
        self.assertEqual(gap["round_gap"], -180)
        self.assertEqual(gap["longer_trained"], "reference")

    def test_equal_lengths_are_labelled_equal(self):
        gap = exposure_gap(make_curve(np.zeros(50)), make_curve(np.zeros(50)))
        self.assertEqual(gap["round_gap"], 0)
        self.assertEqual(gap["longer_trained"], "equal")
        self.assertAlmostEqual(gap["round_ratio"], 1.0)

    def test_common_length_is_the_shorter_arm(self):
        self.assertEqual(common_length(make_curve(np.zeros(40)), make_curve(np.zeros(90))), 40)


class DeltaEstimatorTests(unittest.TestCase):
    def test_argmax_uses_each_arm_full_curve(self):
        reference = make_curve([0.10, 0.20, 0.30])
        variant = make_curve([0.10, 0.20, 0.30, 0.90])
        self.assertAlmostEqual(delta_at(reference, variant, ARGMAX), 0.60)

    def test_matched_scores_both_arms_over_a_common_budget(self):
        # The variant's 0.90 lies beyond the reference's last round, so a
        # matched budget must not count it.
        reference = make_curve([0.10, 0.20, 0.30])
        variant = make_curve([0.10, 0.20, 0.30, 0.90])
        self.assertAlmostEqual(delta_at(reference, variant, MATCHED), 0.0)

    def test_matched_removes_an_advantage_that_is_only_extra_rounds(self):
        # Identical curves; the variant simply keeps going. The argmax rewards
        # it for the extra draws, matching does not.
        shared = list(np.linspace(0.50, 0.60, 500))
        reference = make_curve(shared)
        variant = make_curve(shared + list(np.linspace(0.60, 0.66, 300)))
        self.assertGreater(delta_at(reference, variant, ARGMAX), 0.05)
        self.assertAlmostEqual(delta_at(reference, variant, MATCHED), 0.0)

    def test_plateau_reads_the_settled_level_not_the_best_draw(self):
        # Both arms settle at the same level, but the variant has one lucky
        # spike. The argmax sees the spike; the plateau does not.
        reference = make_curve(np.full(400, 0.50))
        spiked = np.full(400, 0.50)
        spiked[123] = 0.80
        variant = make_curve(spiked)
        self.assertAlmostEqual(delta_at(reference, variant, ARGMAX), 0.30)
        plateau = delta_at(reference, variant, PLATEAU, plateau_window=100)
        self.assertAlmostEqual(plateau, 0.0)

    def test_plateau_window_is_clipped_to_the_common_range(self):
        reference = make_curve(np.full(30, 0.4))
        variant = make_curve(np.full(30, 0.5))
        self.assertAlmostEqual(delta_at(reference, variant, PLATEAU, plateau_window=10_000), 0.10)

    def test_unknown_estimator_is_rejected(self):
        curve = make_curve(np.zeros(10))
        with self.assertRaises(ValueError):
            delta_at(curve, curve, "best_of_three")

    def test_missing_metric_column_is_rejected(self):
        curve = make_curve(np.zeros(10)).drop(columns=["validation_auc"])
        with self.assertRaises(ValueError):
            delta_at(curve, curve, ARGMAX, metric_column="validation_auc")

    def test_non_finite_curve_is_rejected(self):
        curve = make_curve([0.1, np.nan, 0.3])
        with self.assertRaises(ValueError):
            delta_at(curve, curve, ARGMAX)

    def test_empty_curve_is_rejected(self):
        with self.assertRaises(ValueError):
            delta_at(make_curve([]), make_curve([0.1]), ARGMAX)


class SelectedIterationTests(unittest.TestCase):
    def test_selects_the_round_maximising_the_stopping_metric(self):
        curve = make_curve([0.1, 0.7, 0.3, 0.2])
        self.assertEqual(selected_iteration(curve), 2)

    def test_budget_restricts_the_search(self):
        curve = make_curve([0.1, 0.7, 0.3, 0.9])
        self.assertEqual(selected_iteration(curve), 4)
        self.assertEqual(selected_iteration(curve, budget=3), 2)

    def test_zero_budget_is_rejected(self):
        with self.assertRaises(ValueError):
            selected_iteration(make_curve([0.1, 0.2]), budget=0)


class OffMetricTests(unittest.TestCase):
    def test_off_metric_is_read_at_the_selected_round_of_each_arm(self):
        # Reference peaks on AP at round 1, variant at round 3. The off-metric
        # must be read at those rounds, not at its own maximum.
        reference = make_curve([0.9, 0.1, 0.1], auc=[0.50, 0.99, 0.99])
        variant = make_curve([0.1, 0.1, 0.9], auc=[0.99, 0.99, 0.40])
        self.assertAlmostEqual(off_metric_delta_at_selection(reference, variant), 0.40 - 0.50)

    def test_off_metric_can_disagree_with_the_stopping_metric(self):
        # The pattern the ticket turns on: variant wins on the stopping metric
        # and loses on the metric the stopping rule does not watch.
        reference = make_curve([0.50, 0.51], auc=[0.90, 0.91])
        variant = make_curve([0.50, 0.60], auc=[0.90, 0.85])
        self.assertGreater(delta_at(reference, variant, ARGMAX), 0)
        self.assertLess(off_metric_delta_at_selection(reference, variant), 0)


class CompareEstimatorsTests(unittest.TestCase):
    def test_reports_every_estimator_and_the_exposure_screen(self):
        reference = make_curve(np.linspace(0.50, 0.60, 300))
        variant = make_curve(np.linspace(0.50, 0.62, 800))
        result = compare_estimators(reference, variant, plateau_window=100)
        for key in (
            "delta_argmax",
            "delta_matched",
            "delta_plateau",
            "argmax_minus_matched",
            "off_metric_delta_at_selection",
            "round_gap",
            "common_rounds",
        ):
            self.assertIn(key, result)
        self.assertEqual(len(ESTIMATORS), 3)

    def test_equal_length_arms_have_identical_argmax_and_matched_deltas(self):
        # The unexposed case: nothing to correct, whatever the delta's size.
        reference = make_curve(np.linspace(0.50, 0.60, 400))
        variant = make_curve(np.linspace(0.50, 0.65, 400))
        result = compare_estimators(reference, variant, noise_floor=0.0005)
        self.assertEqual(result["round_gap"], 0)
        self.assertAlmostEqual(result["delta_argmax"], result["delta_matched"])
        self.assertAlmostEqual(result["argmax_minus_matched"], 0.0)
        self.assertTrue(result["estimators_agree"])

    def test_extra_rounds_break_agreement_and_are_measured_in_noise_floors(self):
        shared = list(np.linspace(0.50, 0.60, 500))
        reference = make_curve(shared)
        variant = make_curve(shared + list(np.linspace(0.60, 0.66, 400)))
        result = compare_estimators(reference, variant, noise_floor=0.0005)
        self.assertFalse(result["estimators_agree"])
        self.assertGreater(result["argmax_minus_matched"], 0)
        self.assertGreater(result["shift_in_noise_floors"], 1.0)

    def test_sign_flip_under_matching_is_flagged(self):
        # Variant wins on the full curve only because it ran longer; over the
        # common budget the reference is ahead.
        reference = make_curve([0.50, 0.70])
        variant = make_curve([0.50, 0.60, 0.90])
        result = compare_estimators(reference, variant, plateau_window=2)
        self.assertGreater(result["delta_argmax"], 0)
        self.assertLess(result["delta_matched"], 0)
        self.assertTrue(result["sign_flips_under_matching"])

    def test_noise_floor_is_optional(self):
        curve = make_curve(np.linspace(0.5, 0.6, 50))
        result = compare_estimators(curve, curve)
        self.assertNotIn("estimators_agree", result)

    def test_non_positive_noise_floor_is_rejected(self):
        curve = make_curve(np.linspace(0.5, 0.6, 50))
        with self.assertRaises(ValueError):
            compare_estimators(curve, curve, noise_floor=0.0)

    def test_identical_curves_give_zero_under_every_estimator(self):
        curve = make_curve(np.linspace(0.50, 0.61, 600))
        result = compare_estimators(curve, curve, noise_floor=0.0005)
        self.assertAlmostEqual(result["delta_argmax"], 0.0)
        self.assertAlmostEqual(result["delta_matched"], 0.0)
        self.assertAlmostEqual(result["delta_plateau"], 0.0)
        self.assertFalse(result["sign_flips_under_matching"])


if __name__ == "__main__":
    unittest.main()
