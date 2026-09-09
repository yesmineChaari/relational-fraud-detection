"""Unit tests for the B1 cross-relation comparison report's significance handling."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.models.compare_b1_variants import build_comparison_row, load_significance

FAKE_SIGNIFICANCE = {
    "comparisons": {
        "b1_card1_vs_b0": {
            "ci_lower_95": 0.002,
            "ci_upper_95": 0.010,
            "excludes_zero": True,
        },
        "b1_card1_card2_vs_b0": {
            "ci_lower_95": -0.004,
            "ci_upper_95": 0.006,
            "excludes_zero": False,
        },
        "b1_card1_vs_b1_card1_card2": {
            "ci_lower_95": -0.001,
            "ci_upper_95": 0.008,
            "excludes_zero": False,
        },
    }
}


class BuildComparisonRowSignificanceTests(unittest.TestCase):
    def test_a_new_variant_carries_its_ci_against_b0(self) -> None:
        metrics = {"pr_auc": 0.655, "roc_auc": 0.928, "best_iteration": 5999}
        row = build_comparison_row(
            "B1-card1", "card1", "new", metrics, {}, 0.649, 0.925, FAKE_SIGNIFICANCE
        )
        self.assertEqual(row["delta_pr_auc_vs_b0_ci_lower_95"], 0.002)
        self.assertEqual(row["delta_pr_auc_vs_b0_ci_upper_95"], 0.010)
        self.assertTrue(row["delta_pr_auc_vs_b0_excludes_zero"])

    def test_the_baseline_row_has_no_ci_against_itself(self) -> None:
        metrics = {"pr_auc": 0.649, "roc_auc": 0.925, "best_iteration": 5821}
        row = build_comparison_row(
            "B0", "none", "baseline", metrics, {}, 0.649, 0.925, FAKE_SIGNIFICANCE
        )
        self.assertIsNone(row["delta_pr_auc_vs_b0_ci_lower_95"])
        self.assertIsNone(row["delta_pr_auc_vs_b0_excludes_zero"])

    def test_a_variant_with_no_significance_entry_gets_none(self) -> None:
        metrics = {"pr_auc": 0.644, "roc_auc": 0.920, "best_iteration": 4000}
        row = build_comparison_row(
            "B1-card_core_addr1",
            "card_core_addr1",
            "original",
            metrics,
            {},
            0.649,
            0.925,
            FAKE_SIGNIFICANCE,
        )
        self.assertIsNone(row["delta_pr_auc_vs_b0_ci_lower_95"])


class LoadSignificanceTests(unittest.TestCase):
    def test_a_missing_file_raises_with_an_actionable_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "b1_significance.json"
            with self.assertRaises(FileNotFoundError) as ctx:
                load_significance(missing)
        self.assertIn("compare_b1_significance", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
