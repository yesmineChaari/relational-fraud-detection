from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd

from src.models.compare_stages import (
    BASELINE_ROLE,
    CONVERGED,
    FROZEN,
    OUTCOME_BEATS_INCUMBENT,
    OUTCOME_FAILS_BASELINE,
    OUTCOME_MATCHES_INCUMBENT,
    OUTCOME_UNDECIDED,
    OUTCOME_UNDERPERFORMS_INCUMBENT,
    ROOT_DIR,
    STAGES,
    build_rows,
    classify_stages,
)

PREVIOUS_REPORT = ROOT_DIR / "reports" / "b1" / "b1_cross_relation_comparison.csv"


def row(variant, order, pr_auc, delta, excludes_zero, role="new"):
    return {
        "variant": variant,
        "stage": variant,
        "stage_order": order,
        "role": role,
        "validation_pr_auc": pr_auc,
        "delta_pr_auc_vs_b0": delta,
        "delta_pr_auc_vs_b0_excludes_zero": excludes_zero,
    }


def baseline_row():
    return row("B0", 0, 0.649139, 0.0, None, role=BASELINE_ROLE)


def significance(delta, excludes_zero):
    return {"observed_delta": delta, "excludes_zero": excludes_zero}


class ReproducesPreviousReportTests(unittest.TestCase):
    """The acceptance criterion: existing rows keep their exact values."""

    def setUp(self):
        if not PREVIOUS_REPORT.exists():
            self.skipTest(f"Previous cross-relation report not present: {PREVIOUS_REPORT}")
        self.previous = pd.read_csv(PREVIOUS_REPORT).set_index("variant")
        self.frozen = pd.DataFrame(build_rows(STAGES)).set_index("variant")

    def test_every_previous_variant_is_still_present(self):
        for variant in self.previous.index:
            self.assertIn(variant, self.frozen.index)

    def test_pr_auc_and_roc_auc_are_unchanged(self):
        for variant in self.previous.index:
            for column in ("validation_pr_auc", "validation_roc_auc"):
                self.assertAlmostEqual(
                    float(self.frozen.loc[variant, column]),
                    float(self.previous.loc[variant, column]),
                    places=12,
                    msg=f"{variant}.{column} drifted",
                )

    def test_deltas_against_the_baseline_are_unchanged(self):
        for variant in self.previous.index:
            for column in ("delta_pr_auc_vs_b0", "delta_roc_auc_vs_b0"):
                self.assertAlmostEqual(
                    float(self.frozen.loc[variant, column]),
                    float(self.previous.loc[variant, column]),
                    places=12,
                    msg=f"{variant}.{column} drifted",
                )

    def test_intervals_are_unchanged_where_they_existed(self):
        for variant in self.previous.index:
            previous_lower = self.previous.loc[variant, "delta_pr_auc_vs_b0_ci_lower_95"]
            if pd.isna(previous_lower):
                continue
            self.assertAlmostEqual(
                float(self.frozen.loc[variant, "delta_pr_auc_vs_b0_ci_lower_95"]),
                float(previous_lower),
                places=12,
            )

    def test_g1_is_the_row_the_previous_report_lacked(self):
        self.assertNotIn("G1-card1", self.previous.index)
        self.assertIn("G1-card1", self.frozen.index)


class BuiltRowsTests(unittest.TestCase):
    def setUp(self):
        self.rows = build_rows(STAGES)

    def test_no_stage_has_been_evaluated_on_test(self):
        self.assertFalse(any(r["final_test_evaluated"] for r in self.rows))

    def test_selection_provenance_is_carried_on_every_row(self):
        self.assertTrue(all(r["candidate_selection_used_validation"] for r in self.rows))

    def test_frozen_rows_record_that_the_cap_was_reached(self):
        # The convergence check's finding, visible in the table rather than
        # only in prose: every frozen run stopped at its cap.
        self.assertTrue(all(r["estimator_cap_reached"] for r in self.rows))

    def test_a_delta_is_never_reported_without_its_interval_being_addressed(self):
        for r in self.rows:
            if r["role"] == BASELINE_ROLE:
                continue
            has_interval = r["delta_pr_auc_vs_b0_ci_lower_95"] is not None
            self.assertIn(r["delta_pr_auc_vs_b0_excludes_zero"], (True, False, None))
            if has_interval:
                self.assertIsNotNone(r["delta_pr_auc_vs_b0_excludes_zero"])

    def test_every_row_declares_its_protocol(self):
        self.assertTrue(all(r["protocol"] == FROZEN for r in self.rows))

    def test_missing_baseline_for_a_protocol_is_rejected(self):
        orphan = [dict(stage, protocol="invented_protocol") for stage in STAGES[2:3]]
        with self.assertRaises(ValueError):
            build_rows(orphan)


class ClassifyStagesTests(unittest.TestCase):
    def test_newest_stage_beating_the_incumbent(self):
        rows = [
            baseline_row(),
            row("B1", 1, 0.6550, 0.0059, True),
            row("G1", 2, 0.6700, 0.0209, True),
        ]
        result = classify_stages(rows, significance(0.0150, True))
        self.assertEqual(result["outcome_code"], OUTCOME_BEATS_INCUMBENT)
        self.assertEqual(result["incumbent_variant"], "B1")

    def test_newest_stage_matching_the_incumbent(self):
        rows = [
            baseline_row(),
            row("B1", 1, 0.6550, 0.0059, True),
            row("G1", 2, 0.6555, 0.0064, True),
        ]
        result = classify_stages(rows, significance(0.0005, False))
        self.assertEqual(result["outcome_code"], OUTCOME_MATCHES_INCUMBENT)

    def test_newest_stage_underperforming_while_still_beating_the_baseline(self):
        rows = [
            baseline_row(),
            row("B1", 1, 0.6550, 0.0059, True),
            row("G1", 2, 0.6520, 0.0029, True),
        ]
        result = classify_stages(rows, significance(-0.0030, True))
        self.assertEqual(result["outcome_code"], OUTCOME_UNDERPERFORMS_INCUMBENT)

    def test_failing_the_baseline_takes_precedence_over_the_incumbent_comparison(self):
        # G1 loses to B0 outright. Ranking it against B1 would understate that.
        rows = [
            baseline_row(),
            row("B1", 1, 0.6550, 0.0059, True),
            row("G1", 2, 0.6264, -0.0227, True),
        ]
        result = classify_stages(rows, significance(-0.0286, True))
        self.assertEqual(result["outcome_code"], OUTCOME_FAILS_BASELINE)

    def test_a_missing_baseline_interval_leaves_the_outcome_undecided(self):
        rows = [
            baseline_row(),
            row("B1", 1, 0.6550, 0.0059, True),
            row("G1", 2, 0.6700, 0.0209, None),
        ]
        result = classify_stages(rows, significance(0.0150, True))
        self.assertEqual(result["outcome_code"], OUTCOME_UNDECIDED)

    def test_a_missing_incumbent_interval_leaves_the_outcome_undecided(self):
        rows = [
            baseline_row(),
            row("B1", 1, 0.6550, 0.0059, True),
            row("G1", 2, 0.6700, 0.0209, True),
        ]
        result = classify_stages(rows, incumbent_significance=None)
        self.assertEqual(result["outcome_code"], OUTCOME_UNDECIDED)

    def test_the_best_variant_within_the_newest_stage_represents_it(self):
        rows = [
            baseline_row(),
            row("B1", 1, 0.6550, 0.0059, True),
            row("G1-a", 2, 0.6300, -0.0191, True),
            row("G1-b", 2, 0.6700, 0.0209, True),
        ]
        result = classify_stages(rows, significance(0.0150, True))
        self.assertEqual(result["latest_variant"], "G1-b")

    def test_the_incumbent_is_the_best_earlier_stage_not_the_most_recent(self):
        rows = [
            baseline_row(),
            row("B1-weak", 1, 0.6400, -0.0091, True),
            row("B1-strong", 1, 0.6550, 0.0059, True),
            row("G1", 2, 0.6700, 0.0209, True),
        ]
        result = classify_stages(rows, significance(0.0150, True))
        self.assertEqual(result["incumbent_variant"], "B1-strong")

    def test_registering_a_further_stage_needs_no_classifier_change(self):
        # The third acceptance criterion, exercised directly: a stage the
        # classifier has never heard of is ranked purely from stage_order.
        rows = [
            baseline_row(),
            row("B1", 1, 0.6550, 0.0059, True),
            row("G1", 2, 0.6264, -0.0227, True),
            row("G2-transformer", 3, 0.6800, 0.0309, True),
        ]
        result = classify_stages(rows, significance(0.0250, True))
        self.assertEqual(result["latest_variant"], "G2-transformer")
        self.assertEqual(result["incumbent_variant"], "B1")
        self.assertEqual(result["outcome_code"], OUTCOME_BEATS_INCUMBENT)

    def test_a_single_stage_with_no_predecessor_is_handled(self):
        rows = [baseline_row(), row("B1", 1, 0.6550, 0.0059, True)]
        result = classify_stages(rows, incumbent_significance=None)
        self.assertIsNone(result["incumbent_variant"])
        self.assertEqual(result["outcome_code"], OUTCOME_BEATS_INCUMBENT)

    def test_empty_rows_are_rejected(self):
        with self.assertRaises(ValueError):
            classify_stages([])

    def test_rows_without_a_baseline_are_rejected(self):
        with self.assertRaises(ValueError):
            classify_stages([row("B1", 1, 0.6550, 0.0059, True)])

    def test_every_outcome_code_carries_a_conclusion(self):
        from src.models.compare_stages import CONCLUSIONS

        for code in (
            OUTCOME_BEATS_INCUMBENT,
            OUTCOME_MATCHES_INCUMBENT,
            OUTCOME_UNDERPERFORMS_INCUMBENT,
            OUTCOME_FAILS_BASELINE,
            OUTCOME_UNDECIDED,
        ):
            self.assertIn(code, CONCLUSIONS)
            self.assertTrue(CONCLUSIONS[code].strip())


class ProtocolSeparationTests(unittest.TestCase):
    def test_the_two_protocols_are_distinct_labels(self):
        self.assertNotEqual(FROZEN, CONVERGED)

    def test_each_protocol_gets_its_own_baseline_delta(self):
        # A converged variant measured against a cap-truncated baseline would
        # reintroduce the confound the convergence work removed, so deltas are
        # computed per protocol rather than against one global reference.
        from src.models.compare_stages import CONVERGED_STAGES

        converged = pd.DataFrame(build_rows(CONVERGED_STAGES)).set_index("variant")
        frozen = pd.DataFrame(build_rows(STAGES)).set_index("variant")
        self.assertAlmostEqual(float(converged.loc["B0", "delta_pr_auc_vs_b0"]), 0.0)
        self.assertAlmostEqual(float(frozen.loc["B0", "delta_pr_auc_vs_b0"]), 0.0)
        # Same model, different protocol, genuinely different numbers.
        self.assertNotAlmostEqual(
            float(converged.loc["G1-card1", "validation_pr_auc"]),
            float(frozen.loc["G1-card1", "validation_pr_auc"]),
            places=6,
        )

    def test_converged_runs_did_not_reach_their_cap(self):
        from src.models.compare_stages import CONVERGED_STAGES

        rows = build_rows(CONVERGED_STAGES)
        self.assertTrue(all(r["estimator_cap_reached"] is False for r in rows))


if __name__ == "__main__":
    unittest.main()
