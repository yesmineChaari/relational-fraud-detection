from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.calibration_and_operating_points import (
    ALERTS_PER_DAY,
    COST_ASSUMPTIONS,
    COST_RATIOS,
    RECALIBRATION_IS_OPTIMISTIC,
    alert_budget_row,
    build_cost_sweep,
    expected_calibration_error,
    recalibrate,
    validation_span_days,
)


class ValidationSpanTests(unittest.TestCase):
    def test_span_is_read_from_the_split_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split_metadata.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "boundary_transaction_dt": {
                            "validation_min": 0,
                            "validation_max": 86_400 * 10,
                        }
                    },
                    handle,
                )
            with patch("src.models.calibration_and_operating_points.SPLIT_METADATA_PATH", path):
                self.assertAlmostEqual(validation_span_days(), 10.0)

    def test_non_positive_span_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "split_metadata.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"boundary_transaction_dt": {"validation_min": 5, "validation_max": 5}},
                    handle,
                )
            with patch("src.models.calibration_and_operating_points.SPLIT_METADATA_PATH", path):
                with self.assertRaises(ValueError):
                    validation_span_days()


class CalibrationErrorTests(unittest.TestCase):
    def test_a_perfectly_calibrated_score_has_near_zero_error(self):
        rng = np.random.default_rng(0)
        scores = rng.uniform(0, 1, 200_000)
        y_true = (rng.uniform(0, 1, 200_000) < scores).astype(np.int8)
        ece, mce, _ = expected_calibration_error(y_true, scores)
        self.assertLess(ece, 0.01)
        self.assertLess(mce, 0.05)

    def test_a_systematically_overconfident_score_is_detected(self):
        # Every score claims 0.9; nothing is ever positive.
        y_true = np.zeros(1000, dtype=np.int8)
        scores = np.full(1000, 0.9)
        ece, mce, _ = expected_calibration_error(y_true, scores)
        self.assertAlmostEqual(ece, 0.9, places=6)
        self.assertAlmostEqual(mce, 0.9, places=6)

    def test_underprediction_produces_a_negative_gap(self):
        # The direction actually observed in this project: predicted below
        # observed. The sign of `gap` must make that readable.
        y_true = np.concatenate([np.ones(300, dtype=np.int8), np.zeros(700, dtype=np.int8)])
        scores = np.full(1000, 0.05)
        _, _, curve = expected_calibration_error(y_true, scores)
        occupied = curve[curve["count"] > 0]
        self.assertEqual(len(occupied), 1)
        self.assertLess(float(occupied["gap"].iloc[0]), 0)

    def test_empty_bins_are_reported_rather_than_dropped(self):
        y_true = np.zeros(100, dtype=np.int8)
        scores = np.full(100, 0.02)
        _, _, curve = expected_calibration_error(y_true, scores, n_bins=20)
        self.assertEqual(len(curve), 20)
        self.assertEqual(int((curve["count"] == 0).sum()), 19)

    def test_all_bins_empty_is_rejected(self):
        with self.assertRaises(ValueError):
            expected_calibration_error(np.zeros(0, dtype=np.int8), np.zeros(0), n_bins=5)


class RecalibrationTests(unittest.TestCase):
    def test_isotonic_fit_on_its_own_rows_is_perfectly_calibrated_in_sample(self):
        # The reason the optimism caveat exists: an in-sample isotonic fit can
        # drive calibration error to zero and still say nothing about held-out
        # rows. If this ever stops being true the caveat should be revisited.
        rng = np.random.default_rng(1)
        scores = rng.uniform(0, 1, 5000)
        y_true = (rng.uniform(0, 1, 5000) < scores * 0.3).astype(np.int8)
        adjusted = recalibrate(y_true, scores)["isotonic"]
        ece, _, _ = expected_calibration_error(y_true, adjusted)
        self.assertLess(ece, 1e-9)

    def test_both_calibrators_return_values_in_the_unit_interval(self):
        rng = np.random.default_rng(2)
        scores = rng.uniform(0, 1, 2000)
        y_true = (rng.uniform(0, 1, 2000) < 0.2).astype(np.int8)
        for adjusted in recalibrate(y_true, scores).values():
            self.assertGreaterEqual(float(np.min(adjusted)), 0.0)
            self.assertLessEqual(float(np.max(adjusted)), 1.0)

    def test_recalibration_preserves_row_count(self):
        rng = np.random.default_rng(3)
        scores = rng.uniform(0, 1, 500)
        y_true = (rng.uniform(0, 1, 500) < 0.1).astype(np.int8)
        for adjusted in recalibrate(y_true, scores).values():
            self.assertEqual(len(adjusted), 500)


class AlertBudgetTests(unittest.TestCase):
    def setUp(self):
        # 10 rows, 3 positives, ranked so the positives sit at the top.
        self.y_true = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8)
        self.order = np.arange(10)

    def test_a_budget_covering_only_true_positives_is_perfectly_precise(self):
        row = alert_budget_row(self.y_true, self.order, 3)
        self.assertEqual(row["true_positives"], 3)
        self.assertEqual(row["false_positives"], 0)
        self.assertAlmostEqual(row["precision"], 1.0)
        self.assertAlmostEqual(row["recall"], 1.0)

    def test_a_partial_budget_trades_recall_for_nothing_gained(self):
        row = alert_budget_row(self.y_true, self.order, 2)
        self.assertAlmostEqual(row["precision"], 1.0)
        self.assertAlmostEqual(row["recall"], 2 / 3)
        self.assertEqual(row["false_negatives"], 1)

    def test_a_budget_beyond_the_positives_dilutes_precision(self):
        row = alert_budget_row(self.y_true, self.order, 6)
        self.assertAlmostEqual(row["precision"], 0.5)
        self.assertAlmostEqual(row["recall"], 1.0)
        self.assertEqual(row["false_positives"], 3)

    def test_a_budget_larger_than_the_population_is_clipped(self):
        row = alert_budget_row(self.y_true, self.order, 10_000)
        self.assertEqual(row["alerts"], 10)
        self.assertEqual(row["false_negatives"], 0)

    def test_confusion_counts_are_internally_consistent(self):
        for n in range(1, 11):
            row = alert_budget_row(self.y_true, self.order, n)
            self.assertEqual(row["true_positives"] + row["false_positives"], row["alerts"])
            self.assertEqual(row["true_positives"] + row["false_negatives"], 3)


class CostSweepTests(unittest.TestCase):
    def make_budget(self):
        return pd.DataFrame(
            [
                {
                    "variant": "B0",
                    "alerts_per_day": 10,
                    "false_positives": 10,
                    "false_negatives": 100,
                },
                {
                    "variant": "B0",
                    "alerts_per_day": 100,
                    "false_positives": 900,
                    "false_negatives": 20,
                },
            ]
        )

    def test_cost_is_false_alerts_plus_weighted_misses(self):
        sweep = build_cost_sweep(self.make_budget())
        row = sweep[
            (sweep.alerts_per_day == 10) & (sweep.cost_ratio_missed_fraud_to_false_alert == 10)
        ].iloc[0]
        self.assertEqual(row["total_cost_in_false_alert_units"], 10 + 10 * 100)

    def test_a_low_ratio_prefers_the_tight_budget(self):
        # Misses cheap: 10 + 5*100 = 510 beats 900 + 5*20 = 1000, so tolerating
        # misses to avoid false alerts wins.
        sweep = build_cost_sweep(self.make_budget())
        row = sweep[sweep.cost_ratio_missed_fraud_to_false_alert == 5].iloc[0]
        self.assertEqual(row["cost_minimising_alerts_per_day"], 10)

    def test_a_high_ratio_prefers_the_wide_budget(self):
        # Misses expensive: 900 + 250*20 = 5900 beats 10 + 250*100 = 25010, so
        # the same data flips the recommendation. The chosen budget is a
        # function of an assumption, not of the model.
        sweep = build_cost_sweep(self.make_budget())
        row = sweep[sweep.cost_ratio_missed_fraud_to_false_alert == 250].iloc[0]
        self.assertEqual(row["cost_minimising_alerts_per_day"], 100)

    def test_every_cost_row_is_labelled_illustrative(self):
        sweep = build_cost_sweep(self.make_budget())
        self.assertTrue(sweep["costs_are_illustrative"].all())

    def test_every_cost_row_carries_the_ratio_that_produced_it(self):
        sweep = build_cost_sweep(self.make_budget())
        self.assertFalse(sweep["cost_ratio_missed_fraud_to_false_alert"].isna().any())
        self.assertEqual(
            sorted(sweep["cost_ratio_missed_fraud_to_false_alert"].unique()),
            sorted(COST_RATIOS),
        )


class DisclosureTests(unittest.TestCase):
    def test_the_optimism_caveat_names_the_specific_failure(self):
        self.assertIn("fit on the validation rows", RECALIBRATION_IS_OPTIMISTIC)
        self.assertIn("exactly zero", RECALIBRATION_IS_OPTIMISTIC)

    def test_cost_assumptions_disclose_the_absence_of_amount_weighting(self):
        self.assertIn("not amount-weighted", COST_ASSUMPTIONS)

    def test_budgets_span_more_than_one_order_of_magnitude(self):
        self.assertGreater(max(ALERTS_PER_DAY) / min(ALERTS_PER_DAY), 10)


if __name__ == "__main__":
    unittest.main()
