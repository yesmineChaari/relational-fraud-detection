from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from src.config.paths import ROOT_DIR
from src.models.compare_fixed_budget_ablation import (
    ARGMAX_OUTCOME,
    EQUAL_BUDGET_OUTCOME,
    INTERPRETATION_RULE,
    VERDICT_MATCHES_ARGMAX,
    VERDICT_MATCHES_EQUAL_BUDGET,
    VERDICT_MATCHES_NEITHER,
    assert_panel_complete,
    summarize,
)
from src.models.rederive_ablation_at_equal_budget import LEAVE_ONE_OUT, SINGLETON


def cell(mode, feature, delta, excludes_zero, roc=None):
    return {
        "ablation_mode": mode,
        "feature": feature,
        "delta_pr_auc": delta,
        "excludes_zero": excludes_zero,
        "delta_roc_auc": delta if roc is None else roc,
    }


def redundant_panel():
    """Carriers, but nothing costs anything to drop."""
    return pd.DataFrame(
        [
            cell(SINGLETON, "a", 0.009, True),
            cell(SINGLETON, "b", 0.008, True),
            cell(LEAVE_ONE_OUT, "a", 0.0005, False),
            cell(LEAVE_ONE_OUT, "b", -0.0009, False),
        ]
    )


def additive_panel():
    """Dropping a feature costs something significant."""
    return pd.DataFrame(
        [
            cell(SINGLETON, "a", 0.009, True),
            cell(SINGLETON, "b", 0.001, False),
            cell(LEAVE_ONE_OUT, "a", -0.004, True),
            cell(LEAVE_ONE_OUT, "b", -0.005, True),
        ]
    )


def not_localised_panel():
    """No carrier and no cost anywhere."""
    return pd.DataFrame(
        [
            cell(SINGLETON, "a", 0.001, False),
            cell(LEAVE_ONE_OUT, "a", 0.0002, False),
        ]
    )


class VerdictMappingTests(unittest.TestCase):
    """Each pre-registered branch must map to the verdict it was promised to."""

    def test_matching_the_argmax_outcome_confirms_the_published_verdict(self):
        summary = summarize(redundant_panel())
        self.assertEqual(summary["outcome_code"], ARGMAX_OUTCOME)
        self.assertEqual(summary["verdict"], VERDICT_MATCHES_ARGMAX)
        self.assertIn("stands", summary["conclusion"])

    def test_matching_the_equal_budget_outcome_supersedes_it(self):
        summary = summarize(additive_panel())
        self.assertEqual(summary["outcome_code"], EQUAL_BUDGET_OUTCOME)
        self.assertEqual(summary["verdict"], VERDICT_MATCHES_EQUAL_BUDGET)
        self.assertIn("superseded", summary["conclusion"])

    def test_matching_neither_is_reported_as_undecidable(self):
        summary = summarize(not_localised_panel())
        self.assertNotIn(summary["outcome_code"], (ARGMAX_OUTCOME, EQUAL_BUDGET_OUTCOME))
        self.assertEqual(summary["verdict"], VERDICT_MATCHES_NEITHER)
        self.assertIn("not localisable", summary["conclusion"])

    def test_the_three_verdicts_are_distinct(self):
        self.assertEqual(
            len({VERDICT_MATCHES_ARGMAX, VERDICT_MATCHES_EQUAL_BUDGET, VERDICT_MATCHES_NEITHER}),
            3,
        )

    def test_the_two_prior_protocols_disagreed(self):
        # The premise of this whole exercise. If they ever match, this module's
        # framing no longer describes the situation.
        self.assertNotEqual(ARGMAX_OUTCOME, EQUAL_BUDGET_OUTCOME)


class SummaryContentTests(unittest.TestCase):
    def test_the_protocol_is_recorded_as_fixed_budget_without_early_stopping(self):
        summary = summarize(redundant_panel())
        self.assertIn("no_early_stopping", summary["protocol"])
        self.assertFalse(summary["early_stopping_used"])
        self.assertEqual(summary["fixed_budget_trees"], 10_000)

    def test_both_prior_protocols_are_carried_in_the_summary(self):
        summary = summarize(redundant_panel())
        self.assertEqual(summary["prior_protocols"]["argmax"], ARGMAX_OUTCOME)
        self.assertEqual(summary["prior_protocols"]["equal_budget_6011"], EQUAL_BUDGET_OUTCOME)

    def test_the_rule_is_declared_imported_rather_than_restated(self):
        summary = summarize(redundant_panel())
        self.assertIn("verbatim", summary["rule_source"])
        self.assertIn("carries the gain", INTERPRETATION_RULE)

    def test_no_test_evaluation_is_claimed(self):
        self.assertFalse(summarize(redundant_panel())["test_evaluated"])

    def test_a_significant_cell_whose_off_metric_disagrees_is_flagged(self):
        # PR-AUC says the variant wins, ROC-AUC says it loses: the signature of
        # a conclusion about the stopping rule rather than the predictors.
        panel = pd.DataFrame(
            [
                cell(SINGLETON, "a", 0.009, True, roc=-0.004),
                cell(LEAVE_ONE_OUT, "a", 0.0005, False, roc=0.0005),
            ]
        )
        self.assertEqual(
            summarize(panel)["significant_cells_where_roc_auc_disagrees"], ["singleton_a"]
        )

    def test_an_agreeing_off_metric_is_not_flagged(self):
        self.assertEqual(
            summarize(redundant_panel())["significant_cells_where_roc_auc_disagrees"], []
        )

    def test_a_non_significant_cell_is_never_flagged(self):
        # Only cells the rule actually acts on are worth contradicting.
        panel = pd.DataFrame([cell(SINGLETON, "a", 0.009, False, roc=-0.004)])
        self.assertEqual(summarize(panel)["significant_cells_where_roc_auc_disagrees"], [])


class PanelCompletenessTests(unittest.TestCase):
    def test_an_incomplete_panel_refuses_to_report(self):
        # Reporting a partial panel would compare cells trained under different
        # amounts of the experiment having finished.
        with patch("src.models.compare_fixed_budget_ablation.run_is_complete", return_value=False):
            with self.assertRaises(SystemExit) as caught:
                assert_panel_complete()
            self.assertIn("not finished", str(caught.exception))

    def test_the_refusal_names_the_command_that_fixes_it(self):
        with patch("src.models.compare_fixed_budget_ablation.run_is_complete", return_value=False):
            with self.assertRaises(SystemExit) as caught:
                assert_panel_complete()
            self.assertIn("train_ablation_fixed_budget", str(caught.exception))

    def test_a_complete_panel_passes(self):
        with patch("src.models.compare_fixed_budget_ablation.run_is_complete", return_value=True):
            assert_panel_complete()


class RuleProvenanceTests(unittest.TestCase):
    def test_the_rule_is_imported_not_reimplemented(self):
        # Three protocols must differ in exactly one place. A retyped rule could
        # drift and make a protocol difference look like a rule difference.
        source = (ROOT_DIR / "src" / "models" / "compare_fixed_budget_ablation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("classify_panel", source)
        self.assertIn("from src.models.rederive_ablation_at_equal_budget import", source)
        self.assertNotIn("def classify_panel", source)


if __name__ == "__main__":
    unittest.main()
