from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.compare_selection_bias import (
    EXPOSED_DIRECTION_HOLDS,
    EXPOSED_DIRECTION_IN_DOUBT,
    NOISE_FLOOR_KEY_PATH,
    UNEXPOSED,
    build_registry,
    classify,
    load_noise_floor,
    summarize,
)


def write_summary(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def nested(value):
    """A seed-variance summary carrying `value` at the fixed key path."""
    payload: dict = {"std": value}
    for key in reversed(NOISE_FLOOR_KEY_PATH[:-1]):
        payload = {key: payload}
    return payload


class NoiseFloorTests(unittest.TestCase):
    def test_reads_the_clean_stratum_standard_deviation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed_variance_summary.json"
            write_summary(path, nested(0.000514200988964303))
            with (
                patch("src.models.compare_selection_bias.SEED_VARIANCE_SUMMARY_PATH", path),
                patch("src.models.compare_selection_bias.ROOT_DIR", Path(tmp)),
            ):
                floor, source = load_noise_floor()
        self.assertAlmostEqual(floor, 0.000514200988964303)
        self.assertIn("seed_variance_summary.json", source)

    def test_does_not_fall_back_to_another_standard_deviation(self):
        # The contaminated stratum's sd is ~16x larger; picking it up by
        # accident would mark every comparison unexposed.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed_variance_summary.json"
            write_summary(
                path,
                {
                    "paired_delta_pr_auc_spread": {"std": 0.006145709420919689},
                    "early_stopping_stratification": {
                        "contaminated_delta_pr_auc": {"std": 0.008553723971667784}
                    },
                },
            )
            with patch("src.models.compare_selection_bias.SEED_VARIANCE_SUMMARY_PATH", path):
                with self.assertRaises(KeyError):
                    load_noise_floor()

    def test_non_positive_floor_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed_variance_summary.json"
            write_summary(path, nested(0.0))
            with patch("src.models.compare_selection_bias.SEED_VARIANCE_SUMMARY_PATH", path):
                with self.assertRaises(ValueError):
                    load_noise_floor()


class ClassifyTests(unittest.TestCase):
    def test_agreement_within_the_floor_is_unexposed(self):
        result = {
            "estimators_agree": True,
            "delta_argmax": 0.0063,
            "delta_matched": 0.0063,
            "delta_plateau": 0.0049,
        }
        self.assertEqual(classify(result), UNEXPOSED)

    def test_unexposed_regardless_of_how_large_the_delta_is(self):
        # A big effect measured on equally-selected arms is not exposed.
        result = {
            "estimators_agree": True,
            "delta_argmax": -0.5,
            "delta_matched": -0.5,
            "delta_plateau": -0.5,
        }
        self.assertEqual(classify(result), UNEXPOSED)

    def test_shifted_but_same_sign_holds_direction(self):
        result = {
            "estimators_agree": False,
            "delta_argmax": 0.00931,
            "delta_matched": 0.00723,
            "delta_plateau": 0.00703,
        }
        self.assertEqual(classify(result), EXPOSED_DIRECTION_HOLDS)

    def test_sign_change_puts_direction_in_doubt(self):
        result = {
            "estimators_agree": False,
            "delta_argmax": 0.00242,
            "delta_matched": -0.00134,
            "delta_plateau": -0.00240,
        }
        self.assertEqual(classify(result), EXPOSED_DIRECTION_IN_DOUBT)

    def test_exact_zero_does_not_count_as_a_sign_disagreement(self):
        result = {
            "estimators_agree": False,
            "delta_argmax": 0.004,
            "delta_matched": 0.0,
            "delta_plateau": 0.002,
        }
        self.assertEqual(classify(result), EXPOSED_DIRECTION_HOLDS)


class RegistryTests(unittest.TestCase):
    def test_covers_both_headlines_the_ablation_panel_and_the_seed_panel(self):
        registry = build_registry()
        comparisons = {entry["comparison"] for entry in registry}
        self.assertIn("b1_card1_vs_b0", comparisons)
        self.assertIn("g1_card1_vs_b0", comparisons)
        for key in ("prior_count", "prior_count_24h", "prior_count_7d"):
            self.assertIn(f"singleton_{key}_vs_b0", comparisons)
            self.assertIn(f"loo_{key}_vs_b1_card1", comparisons)
        self.assertEqual(sum(entry["family"] == "seed_panel" for entry in registry), 5)

    def test_every_entry_names_a_reference_and_a_variant(self):
        for entry in build_registry():
            self.assertIn("reference_dir", entry)
            self.assertIn("variant_dir", entry)
            self.assertNotEqual(entry["reference_dir"], entry["variant_dir"])

    def test_comparison_labels_are_unique(self):
        registry = build_registry()
        labels = [entry["comparison"] for entry in registry]
        self.assertEqual(len(labels), len(set(labels)))

    def test_singletons_reference_b0_and_leave_one_out_references_b1(self):
        for entry in build_registry():
            if entry["family"] == "ablation_singleton":
                self.assertIn("b0", entry["reference_dir"].as_posix())
            if entry["family"] == "ablation_loo":
                self.assertIn("b1_card1", entry["reference_dir"].as_posix())


class SummarizeTests(unittest.TestCase):
    def make_table(self):
        return pd.DataFrame(
            [
                {
                    "family": "headline",
                    "comparison": "unexposed_one",
                    "classification": UNEXPOSED,
                    "delta_argmax": 0.0063,
                    "delta_matched": 0.0063,
                    "delta_plateau": 0.0049,
                    "argmax_minus_matched": 0.0,
                    "off_metric_delta_at_selection": 0.0034,
                    "round_gap": 76,
                    "absolute_round_gap": 76,
                },
                {
                    "family": "ablation_loo",
                    "comparison": "doubtful_one",
                    "classification": EXPOSED_DIRECTION_IN_DOUBT,
                    "delta_argmax": 0.0004,
                    "delta_matched": -0.0036,
                    "delta_plateau": -0.0036,
                    "argmax_minus_matched": 0.0040,
                    "off_metric_delta_at_selection": -0.0022,
                    "round_gap": 1616,
                    "absolute_round_gap": 1616,
                },
            ]
        )

    def test_partitions_comparisons_by_classification(self):
        summary = summarize(self.make_table(), 0.000514, "reports/x.json")
        self.assertEqual(summary["n_comparisons"], 2)
        self.assertEqual(summary["unexposed_comparisons"], ["unexposed_one"])
        self.assertEqual(summary["exposed_comparisons"], ["doubtful_one"])
        self.assertEqual(summary["direction_in_doubt_comparisons"], ["doubtful_one"])

    def test_flags_where_the_off_metric_contradicts_the_stopping_metric(self):
        summary = summarize(self.make_table(), 0.000514, "reports/x.json")
        self.assertEqual(summary["off_metric_contradicts_stopping_metric"], ["doubtful_one"])

    def test_carries_the_noise_floor_and_its_source_through(self):
        summary = summarize(self.make_table(), 0.000514, "reports/seed.json")
        self.assertAlmostEqual(summary["clean_stratum_noise_floor"], 0.000514)
        self.assertEqual(summary["noise_floor_source"], "reports/seed.json")
        self.assertFalse(summary["test_evaluated"])

    def test_records_the_rule_and_the_estimator_decision(self):
        summary = summarize(self.make_table(), 0.000514, "reports/seed.json")
        self.assertIn("noise floor", summary["classification_rule"])
        self.assertIn("lower bound", summary["estimator_decision"])


if __name__ == "__main__":
    unittest.main()
