from __future__ import annotations

import unittest

import pandas as pd

from src.models.rederive_ablation_at_equal_budget import (
    FEATURE_KEYS,
    LEAVE_ONE_OUT,
    OUTCOME_ADDITIVE,
    OUTCOME_NOT_LOCALISED,
    OUTCOME_REDUNDANT,
    OUTCOME_SINGLE_FEATURE,
    SINGLETON,
    classify_panel,
    metrics_paths,
    model_paths,
)


def cell(mode, feature, delta, excludes_zero):
    return {
        "ablation_mode": mode,
        "feature": feature,
        "delta_pr_auc": delta,
        "excludes_zero": excludes_zero,
    }


def panel(rows):
    return pd.DataFrame(rows)


class ClassifyPanelTests(unittest.TestCase):
    def test_any_significant_leave_one_out_cost_makes_the_panel_additive(self):
        carriers, non_redundant, outcome = classify_panel(
            panel(
                [
                    cell(SINGLETON, "a", 0.009, True),
                    cell(SINGLETON, "b", 0.008, True),
                    cell(LEAVE_ONE_OUT, "a", -0.004, True),
                    cell(LEAVE_ONE_OUT, "b", 0.001, False),
                ]
            )
        )
        self.assertEqual(carriers, ["a", "b"])
        self.assertEqual(non_redundant, ["a"])
        self.assertEqual(outcome, OUTCOME_ADDITIVE)

    def test_carriers_with_no_cost_to_dropping_any_is_redundancy(self):
        _, non_redundant, outcome = classify_panel(
            panel(
                [
                    cell(SINGLETON, "a", 0.009, True),
                    cell(SINGLETON, "b", 0.008, True),
                    cell(LEAVE_ONE_OUT, "a", 0.0005, False),
                    cell(LEAVE_ONE_OUT, "b", -0.0009, False),
                ]
            )
        )
        self.assertEqual(non_redundant, [])
        self.assertEqual(outcome, OUTCOME_REDUNDANT)

    def test_exactly_one_carrier_and_no_cost_is_a_single_feature(self):
        carriers, _, outcome = classify_panel(
            panel(
                [
                    cell(SINGLETON, "a", 0.009, True),
                    cell(SINGLETON, "b", 0.001, False),
                    cell(LEAVE_ONE_OUT, "a", 0.0002, False),
                ]
            )
        )
        self.assertEqual(carriers, ["a"])
        self.assertEqual(outcome, OUTCOME_SINGLE_FEATURE)

    def test_no_carrier_and_no_cost_is_not_localised(self):
        carriers, non_redundant, outcome = classify_panel(
            panel(
                [
                    cell(SINGLETON, "a", 0.001, False),
                    cell(LEAVE_ONE_OUT, "a", 0.0002, False),
                ]
            )
        )
        self.assertEqual(carriers, [])
        self.assertEqual(non_redundant, [])
        self.assertEqual(outcome, OUTCOME_NOT_LOCALISED)

    def test_a_significant_singleton_in_the_wrong_direction_does_not_carry(self):
        # An interval excluding zero on the *negative* side means adding the
        # feature hurt; it must never be counted as carrying the gain.
        carriers, _, outcome = classify_panel(panel([cell(SINGLETON, "a", -0.004, True)]))
        self.assertEqual(carriers, [])
        self.assertEqual(outcome, OUTCOME_NOT_LOCALISED)

    def test_a_significant_leave_one_out_gain_is_not_non_redundancy(self):
        # Dropping the feature *helped* significantly. That is not evidence the
        # feature was needed, so it must not be recorded as non-redundant.
        _, non_redundant, outcome = classify_panel(
            panel(
                [
                    cell(SINGLETON, "a", 0.009, True),
                    cell(LEAVE_ONE_OUT, "a", 0.005, True),
                ]
            )
        )
        self.assertEqual(non_redundant, [])
        self.assertEqual(outcome, OUTCOME_SINGLE_FEATURE)

    def test_additive_wins_over_carrier_counting(self):
        # Even with a single carrier, a real cost to dropping something means
        # the summaries combine rather than one standing in for the rest.
        _, non_redundant, outcome = classify_panel(
            panel(
                [
                    cell(SINGLETON, "a", 0.009, True),
                    cell(LEAVE_ONE_OUT, "b", -0.004, True),
                ]
            )
        )
        self.assertEqual(non_redundant, ["b"])
        self.assertEqual(outcome, OUTCOME_ADDITIVE)

    def test_reproduces_the_published_equal_budget_outcome(self):
        # The measured panel at 6,011 trees, as committed to the report.
        carriers, non_redundant, outcome = classify_panel(
            panel(
                [
                    cell(SINGLETON, "prior_count", 0.00697, True),
                    cell(SINGLETON, "prior_count_24h", 0.00434, True),
                    cell(SINGLETON, "prior_count_7d", 0.00088, False),
                    cell(SINGLETON, "time_since_previous_hours", -0.00196, False),
                    cell(LEAVE_ONE_OUT, "prior_count", -0.00429, True),
                    cell(LEAVE_ONE_OUT, "prior_count_24h", 0.00080, False),
                    cell(LEAVE_ONE_OUT, "prior_count_7d", -0.00380, True),
                    cell(LEAVE_ONE_OUT, "time_since_previous_hours", -0.00385, True),
                ]
            )
        )
        self.assertEqual(carriers, ["prior_count", "prior_count_24h"])
        self.assertEqual(
            non_redundant,
            ["prior_count", "prior_count_7d", "time_since_previous_hours"],
        )
        self.assertEqual(outcome, OUTCOME_ADDITIVE)


class PanelPathTests(unittest.TestCase):
    def test_every_ablation_cell_has_a_model_and_a_metrics_file(self):
        models, metrics = model_paths(), metrics_paths()
        self.assertEqual(set(models), set(metrics))
        for mode in (SINGLETON, LEAVE_ONE_OUT):
            for key in FEATURE_KEYS:
                self.assertIn(f"{mode}_{key}", models)
        # Eight cells plus the two converged references.
        self.assertEqual(len(models), 10)

    def test_paths_are_distinct_per_cell(self):
        models = model_paths()
        self.assertEqual(len(set(models.values())), len(models))


if __name__ == "__main__":
    unittest.main()
