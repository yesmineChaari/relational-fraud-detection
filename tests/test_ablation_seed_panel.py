from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.models.summarize_ablation_seed_panel import (
    MAX_BEST_ITERATION_RATIO,
    OUTCOME_PANEL_TOO_SMALL,
    OUTCOME_SIGN_STABLE,
    OUTCOME_SIGN_UNSTABLE,
    PANEL_SEEDS,
    REFERENCE_CONFIG,
    build_paired_table,
    classify,
    discover_paired_seeds,
    resolve_reference_paths,
    stratify,
)
from src.models.train_lightgbm_ablation import (
    ABLATION_FEATURES,
    FEATURE_KEYS,
    LEAVE_ONE_OUT,
    MODES,
    SINGLETON,
    STOP_METRIC,
)
from src.models.train_lightgbm_baseline import RANDOM_SEED
from src.models.train_lightgbm_convergence_check import (
    DEFAULT_MAX_ESTIMATORS as CONVERGED_MAX_ESTIMATORS,
)

VARIANT_FEATURE = ABLATION_FEATURES[1]


def paired_row(
    seed: int,
    delta: float,
    variant_iterations: int = 8_000,
    reference_iterations: int = 7_000,
    roc_delta: float = 0.001,
) -> dict:
    reference_pr_auc = 0.6555
    ratio = max(variant_iterations, reference_iterations) / min(
        variant_iterations, reference_iterations
    )
    return {
        "seed": seed,
        "variant_pr_auc": reference_pr_auc + delta,
        "variant_roc_auc": 0.9250 + roc_delta,
        "variant_best_iteration": variant_iterations,
        "reference_pr_auc": reference_pr_auc,
        "reference_roc_auc": 0.9250,
        "reference_best_iteration": reference_iterations,
        "delta_pr_auc": delta,
        "delta_roc_auc": roc_delta,
        "best_iteration_ratio": ratio,
        "early_stopping_clean": bool(ratio <= MAX_BEST_ITERATION_RATIO),
        "longer_trained": (
            "variant" if variant_iterations > reference_iterations else "reference"
        ),
    }


def write_metrics(path: Path, pr_auc: float, roc_auc: float, best_iteration: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            {"pr_auc": pr_auc, "roc_auc": roc_auc, "best_iteration": best_iteration}, handle
        )


class ReferenceResolutionTests(unittest.TestCase):
    def test_leave_one_out_reads_against_the_converged_four_feature_model(self) -> None:
        self.assertEqual(REFERENCE_CONFIG[LEAVE_ONE_OUT], "b1_card1")
        paths = resolve_reference_paths(LEAVE_ONE_OUT, 202)
        posix = paths["metrics"].as_posix()
        self.assertIn("b1_card1", posix)
        self.assertIn(f"stop_{STOP_METRIC}", posix)
        self.assertIn(f"cap{CONVERGED_MAX_ESTIMATORS}_seed202", posix)

    def test_singleton_reads_against_the_converged_baseline(self) -> None:
        self.assertEqual(REFERENCE_CONFIG[SINGLETON], "b0")
        self.assertIn("b0", resolve_reference_paths(SINGLETON, 42)["metrics"].as_posix())

    def test_every_mode_has_a_reference(self) -> None:
        self.assertEqual(sorted(REFERENCE_CONFIG), sorted(MODES))

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_reference_paths("pairs", 42)

    def test_each_seed_reads_its_own_reference(self) -> None:
        """Pairing across seeds would mix the two sources of refit noise."""
        first = resolve_reference_paths(LEAVE_ONE_OUT, 202)["metrics"]
        second = resolve_reference_paths(LEAVE_ONE_OUT, 707)["metrics"]
        self.assertNotEqual(first, second)


class StratificationTests(unittest.TestCase):
    def test_a_balanced_pair_is_clean(self) -> None:
        table = pd.DataFrame([paired_row(42, 0.005, 8_000, 7_000)])
        self.assertTrue(bool(table.loc[0, "early_stopping_clean"]))

    def test_a_lopsided_pair_is_contaminated(self) -> None:
        table = pd.DataFrame([paired_row(707, 0.015, 8_115, 3_644)])
        self.assertFalse(bool(table.loc[0, "early_stopping_clean"]))

    def test_the_threshold_is_inclusive(self) -> None:
        table = pd.DataFrame([paired_row(42, 0.005, 8_000, 4_000)])
        self.assertEqual(table.loc[0, "best_iteration_ratio"], MAX_BEST_ITERATION_RATIO)
        self.assertTrue(bool(table.loc[0, "early_stopping_clean"]))

    def test_the_ratio_is_direction_agnostic(self) -> None:
        """A reference that trained far longer contaminates just as much."""
        variant_longer = pd.DataFrame([paired_row(1, 0.005, 9_000, 3_000)])
        reference_longer = pd.DataFrame([paired_row(2, 0.005, 3_000, 9_000)])
        self.assertFalse(bool(variant_longer.loc[0, "early_stopping_clean"]))
        self.assertFalse(bool(reference_longer.loc[0, "early_stopping_clean"]))

    def test_both_strata_are_reported(self) -> None:
        table = pd.DataFrame(
            [
                paired_row(42, 0.0055, 8_219, 6_087),
                paired_row(202, 0.0014, 8_486, 7_011),
                paired_row(707, 0.0149, 8_115, 3_644),
                paired_row(1337, 0.0015, 6_807, 9_310),
            ]
        )
        result = stratify(table)
        self.assertEqual(result["n_clean"], 3)
        self.assertEqual(result["n_contaminated"], 1)
        self.assertEqual(result["contaminated_seeds"], [707])
        self.assertIn("clean_delta_pr_auc", result)

    def test_the_stratification_rule_and_its_provenance_are_recorded(self) -> None:
        table = pd.DataFrame([paired_row(42, 0.005), paired_row(202, 0.004)])
        result = stratify(table)
        self.assertEqual(result["max_best_iteration_ratio"], MAX_BEST_ITERATION_RATIO)
        for field in ("rule", "why_not_the_inherited_rule", "threshold_fixed_when"):
            self.assertTrue(result[field].strip())

    def test_roc_auc_is_reported_across_every_seed(self) -> None:
        """ROC-AUC is not the stopping metric, so contamination does not apply to it."""
        table = pd.DataFrame(
            [paired_row(42, 0.005), paired_row(707, 0.015, 8_115, 3_644)]
        )
        result = stratify(table)
        self.assertEqual(result["roc_auc_delta_all_seeds"]["n"], 2)
        self.assertTrue(result["roc_auc_is_not_the_stopping_metric"])


class ClassificationTests(unittest.TestCase):
    def test_a_sign_stable_delta_is_reported_as_stable(self) -> None:
        table = pd.DataFrame(
            [
                paired_row(42, 0.005484),
                paired_row(202, 0.001364),
                paired_row(1337, 0.001463),
                paired_row(2024, 0.005200),
            ]
        )
        outcome = classify(table, stratify(table))
        self.assertEqual(outcome["outcome_code"], OUTCOME_SIGN_STABLE)
        self.assertTrue(outcome["clean_delta_sign_stable"])
        self.assertAlmostEqual(outcome["clean_delta_mean"], 0.00337775, places=6)

    def test_a_sign_flipping_delta_is_reported_as_unstable(self) -> None:
        table = pd.DataFrame(
            [
                paired_row(42, 0.005),
                paired_row(202, -0.004),
                paired_row(1337, 0.002),
                paired_row(2024, -0.003),
            ]
        )
        outcome = classify(table, stratify(table))
        self.assertEqual(outcome["outcome_code"], OUTCOME_SIGN_UNSTABLE)
        self.assertFalse(outcome["clean_delta_sign_stable"])

    def test_a_consistently_negative_delta_is_also_sign_stable(self) -> None:
        table = pd.DataFrame(
            [paired_row(42, -0.005), paired_row(202, -0.004), paired_row(1337, -0.006)]
        )
        outcome = classify(table, stratify(table))
        self.assertEqual(outcome["outcome_code"], OUTCOME_SIGN_STABLE)

    def test_a_single_clean_seed_cannot_support_a_verdict(self) -> None:
        table = pd.DataFrame(
            [paired_row(42, 0.005), paired_row(707, 0.015, 8_115, 3_644)]
        )
        outcome = classify(table, stratify(table))
        self.assertEqual(outcome["outcome_code"], OUTCOME_PANEL_TOO_SMALL)

    def test_contaminated_seeds_are_excluded_from_the_verdict(self) -> None:
        """The lopsided seed carries the largest delta; it must not drive the mean."""
        table = pd.DataFrame(
            [
                paired_row(42, 0.005484),
                paired_row(202, 0.001364),
                paired_row(707, 0.014851, 8_115, 3_644),
                paired_row(1337, 0.001463),
            ]
        )
        outcome = classify(table, stratify(table))
        self.assertLess(outcome["clean_delta_mean"], 0.005)


class PairedTableTests(unittest.TestCase):
    def test_only_seeds_with_both_halves_are_paired(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            variant_dir = root / "variant"
            reference_dir = root / "reference"

            def fake_variant(mode, feature, seed=RANDOM_SEED):
                return {"metrics": variant_dir / f"seed{seed}.json"}

            def fake_reference(mode, seed):
                return {"metrics": reference_dir / f"seed{seed}.json"}

            write_metrics(variant_dir / "seed42.json", 0.6610, 0.9251, 8_219)
            write_metrics(reference_dir / "seed42.json", 0.6555, 0.9248, 6_087)
            # Seed 202 has a variant run but no reference yet.
            write_metrics(variant_dir / "seed202.json", 0.6578, 0.9248, 8_486)

            with patch(
                "src.models.summarize_ablation_seed_panel.resolve_run_paths", fake_variant
            ), patch(
                "src.models.summarize_ablation_seed_panel.resolve_reference_paths",
                fake_reference,
            ):
                self.assertEqual(
                    discover_paired_seeds(LEAVE_ONE_OUT, VARIANT_FEATURE), [42]
                )

    def test_a_panel_with_no_paired_seed_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp)

            def fake(mode, feature=None, seed=RANDOM_SEED):
                return {"metrics": missing / "absent.json"}

            with patch(
                "src.models.summarize_ablation_seed_panel.resolve_run_paths",
                lambda mode, feature, seed=RANDOM_SEED: fake(mode, feature, seed),
            ), patch(
                "src.models.summarize_ablation_seed_panel.resolve_reference_paths",
                lambda mode, seed: fake(mode, None, seed),
            ):
                with self.assertRaises(FileNotFoundError):
                    discover_paired_seeds(LEAVE_ONE_OUT, VARIANT_FEATURE)

    def test_the_delta_is_variant_minus_reference(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            def fake_variant(mode, feature, seed=RANDOM_SEED):
                return {"metrics": root / f"v{seed}.json"}

            def fake_reference(mode, seed):
                return {"metrics": root / f"r{seed}.json"}

            write_metrics(root / "v42.json", 0.6610, 0.9251, 8_219)
            write_metrics(root / "r42.json", 0.6555, 0.9248, 6_087)

            with patch(
                "src.models.summarize_ablation_seed_panel.resolve_run_paths", fake_variant
            ), patch(
                "src.models.summarize_ablation_seed_panel.resolve_reference_paths",
                fake_reference,
            ):
                table = build_paired_table(LEAVE_ONE_OUT, VARIANT_FEATURE, [42])
            self.assertAlmostEqual(table.loc[0, "delta_pr_auc"], 0.0055, places=6)
            self.assertEqual(table.loc[0, "longer_trained"], "variant")


class PanelConstantTests(unittest.TestCase):
    def test_the_panel_uses_the_project_seed_set(self) -> None:
        self.assertIn(RANDOM_SEED, PANEL_SEEDS)
        self.assertEqual(len(PANEL_SEEDS), len(set(PANEL_SEEDS)))

    def test_the_variant_under_test_is_a_real_feature(self) -> None:
        self.assertIn(VARIANT_FEATURE, FEATURE_KEYS)


if __name__ == "__main__":
    unittest.main()
