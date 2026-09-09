from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.compare_permuted_null import (
    OUTCOME_AT_OR_BELOW_B0,
    OUTCOME_RECOVERS_GAIN,
    classify_permuted_outcome,
    discover_runs,
    spread,
)
from src.models.train_lightgbm_baseline import RANDOM_SEED, build_lightgbm_model
from src.models.train_lightgbm_convergence_check import (
    DEFAULT_MAX_ESTIMATORS as CONVERGED_MAX_ESTIMATORS,
)
from src.models.train_lightgbm_g1 import B1_CARD1_PROTECTED_PATHS
from src.models.train_lightgbm_permuted_null import (
    ALL_PROTECTED_PATHS,
    B0_CONVERGED_PATHS,
    B1_CARD1_CONVERGED_PATHS,
    PERMUTATION_SEEDS,
    RELATION,
    STOP_METRIC,
    permute_card1,
    resolve_run_paths,
    validate_permuted_null_configuration,
)
from src.models.train_lightgbm_relational import B0_PROTECTED_PATHS, MAX_ESTIMATORS

SCALE_POS_WEIGHT = 27.434310083918007
OUTPUT_KEYS = [
    "model",
    "metrics",
    "metadata",
    "feature_importance",
    "validation_predictions",
    "learning_curve",
    "permuted_relational_features",
    "permuted_relational_metadata",
]


def frozen_metadata(**overrides) -> dict:
    model = build_lightgbm_model(SCALE_POS_WEIGHT, n_estimators=MAX_ESTIMATORS)
    metadata = {
        "lightgbm_parameters": model.get_params(deep=False),
        "scale_pos_weight": SCALE_POS_WEIGHT,
    }
    metadata.update(overrides)
    return metadata


def permuted_null_model(**overrides):
    model = build_lightgbm_model(SCALE_POS_WEIGHT, n_estimators=CONVERGED_MAX_ESTIMATORS)
    parameters = {"random_state": RANDOM_SEED}
    parameters.update(overrides)
    model.set_params(**parameters)
    return model


def source_frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1, dtype="int64"),
            "TransactionDT": np.sort(rng.integers(0, 10_000_000, size=n)).astype("int64"),
            "split": ["train"] * (n // 2) + ["validation"] * (n - n // 2),
            "card1": rng.integers(1000, 1050, size=n).astype("int64"),
        }
    )


def comparison_row(
    seed: int,
    delta_vs_b0: float,
    ci_lower: float,
    ci_upper: float,
) -> dict:
    return {
        "permutation_seed": seed,
        "pr_auc": 0.649 + delta_vs_b0,
        "delta_pr_auc_vs_b0": delta_vs_b0,
        "delta_pr_auc_vs_b0_ci_lower_95": ci_lower,
        "delta_pr_auc_vs_b0_ci_upper_95": ci_upper,
        "delta_pr_auc_vs_b0_excludes_zero": bool(ci_lower > 0.0 or ci_upper < 0.0),
        "recovered_fraction_of_real_gain": delta_vs_b0 / 0.006304055057609892,
    }


def write_run(root: Path, seed: int, pr_auc: float = 0.6458) -> None:
    run_dir = root / f"permseed_{seed}" / f"cap{CONVERGED_MAX_ESTIMATORS}"
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"pr_auc": pr_auc, "permutation_seed": seed}, handle)
    pd.DataFrame({"TransactionID": [1, 2], "isFraud": [0, 1], "prediction": [0.1, 0.9]}).to_parquet(
        run_dir / "validation_predictions.parquet", index=False
    )


class RunPathTests(unittest.TestCase):
    def test_every_panel_seed_resolves_under_the_permuted_null_tree(self) -> None:
        for seed in PERMUTATION_SEEDS:
            paths = resolve_run_paths(seed)
            posix = paths["run_dir"].as_posix()
            self.assertIn("permuted_null", posix)
            self.assertIn(RELATION, posix)
            self.assertIn(f"permseed_{seed}", posix)
            self.assertIn(f"cap{CONVERGED_MAX_ESTIMATORS}", posix)

    def test_negative_permutation_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_paths(-1)

    def test_runs_do_not_collide_across_permutation_seeds(self) -> None:
        seen: set[str] = set()
        for seed in PERMUTATION_SEEDS:
            paths = resolve_run_paths(seed)
            for key in OUTPUT_KEYS:
                posix = paths[key].as_posix()
                self.assertNotIn(posix, seen, f"{key} collides across permutation seeds")
                seen.add(posix)

    def test_the_permuted_feature_table_never_shadows_the_real_relation(self) -> None:
        """The real relational_features_card1.parquet must not be overwritten."""
        for seed in PERMUTATION_SEEDS:
            permuted = resolve_run_paths(seed)["permuted_relational_features"]
            self.assertIn("permuted_null", permuted.as_posix())
            self.assertIn(f"permseed{seed}", permuted.name)
            self.assertNotEqual(permuted.name, f"relational_features_{RELATION}.parquet")


class ProtectedArtifactTests(unittest.TestCase):
    """No permuted-null run may write to anything it reads or is measured against."""

    def test_frozen_b0_artifacts_are_protected(self) -> None:
        for path in B0_PROTECTED_PATHS:
            self.assertIn(path, ALL_PROTECTED_PATHS)

    def test_frozen_b1_card1_artifacts_are_protected(self) -> None:
        for path in B1_CARD1_PROTECTED_PATHS:
            self.assertIn(path, ALL_PROTECTED_PATHS)

    def test_converged_references_are_protected(self) -> None:
        for paths in (B0_CONVERGED_PATHS, B1_CARD1_CONVERGED_PATHS):
            for key in ("metrics", "validation_predictions"):
                self.assertIn(paths[key], ALL_PROTECTED_PATHS)

    def test_the_real_card1_relation_is_protected(self) -> None:
        """The permutation rebuilds features; it must never touch the real ones."""
        protected = {path.as_posix() for path in ALL_PROTECTED_PATHS}
        self.assertTrue(
            any(path.endswith(f"relational_features_{RELATION}.parquet") for path in protected),
            "the real card1 feature table is not hash-pinned",
        )

    def test_no_run_writes_to_a_protected_path(self) -> None:
        protected = set(ALL_PROTECTED_PATHS)
        for seed in PERMUTATION_SEEDS:
            paths = resolve_run_paths(seed)
            for key in OUTPUT_KEYS:
                self.assertNotIn(paths[key], protected, f"{key} would overwrite a protected file")

    def test_no_run_lands_under_a_frozen_report_tree(self) -> None:
        for seed in PERMUTATION_SEEDS:
            posix = resolve_run_paths(seed)["run_dir"].as_posix()
            for frozen_tree in ("/reports/baseline", "/reports/b1/", "/reports/g1/"):
                self.assertNotIn(frozen_tree, posix)


class ReferenceConfigurationTests(unittest.TestCase):
    """The control is read against converged references, never the capped ones.

    Comparing a permuted run at cap 15,000 against a cap-6,000 reference would
    reintroduce exactly the early-stopping confound the convergence check
    removed, which is why this control was blocked until that check landed.
    """

    def test_references_are_the_converged_cap_15000_runs(self) -> None:
        for paths in (B0_CONVERGED_PATHS, B1_CARD1_CONVERGED_PATHS):
            posix = paths["metrics"].as_posix()
            self.assertIn("convergence_check", posix)
            self.assertIn(f"cap{CONVERGED_MAX_ESTIMATORS}_seed{RANDOM_SEED}", posix)

    def test_references_use_average_precision_patience(self) -> None:
        self.assertEqual(STOP_METRIC, "average_precision")
        for paths in (B0_CONVERGED_PATHS, B1_CARD1_CONVERGED_PATHS):
            self.assertIn(f"stop_{STOP_METRIC}", paths["metrics"].as_posix())

    def test_references_are_not_the_frozen_capped_artifacts(self) -> None:
        frozen = {path.as_posix() for path in [*B0_PROTECTED_PATHS, *B1_CARD1_PROTECTED_PATHS]}
        for paths in (B0_CONVERGED_PATHS, B1_CARD1_CONVERGED_PATHS):
            self.assertNotIn(paths["metrics"].as_posix(), frozen)
            self.assertNotIn(paths["validation_predictions"].as_posix(), frozen)
        self.assertNotEqual(CONVERGED_MAX_ESTIMATORS, MAX_ESTIMATORS)


class PermutationTests(unittest.TestCase):
    """The permutation must destroy the entity assignment and nothing else."""

    def test_the_entity_size_marginal_is_preserved_exactly(self) -> None:
        source = source_frame()
        permuted = permute_card1(source, permutation_seed=1)
        before = source["card1"].value_counts().sort_index()
        after = permuted["card1"].value_counts().sort_index()
        pd.testing.assert_series_equal(before, after)

    def test_timestamps_are_untouched_row_by_row(self) -> None:
        source = source_frame()
        permuted = permute_card1(source, permutation_seed=1)
        pd.testing.assert_series_equal(source["TransactionDT"], permuted["TransactionDT"])

    def test_every_column_other_than_card1_is_untouched(self) -> None:
        source = source_frame()
        permuted = permute_card1(source, permutation_seed=1)
        for column in source.columns:
            if column == "card1":
                continue
            pd.testing.assert_series_equal(source[column], permuted[column], obj=column)

    def test_the_entity_assignment_actually_changes(self) -> None:
        source = source_frame()
        permuted = permute_card1(source, permutation_seed=1)
        changed = (source["card1"].to_numpy() != permuted["card1"].to_numpy()).sum()
        self.assertGreater(changed, 0, "the permutation left every row on its own entity")

    def test_the_source_frame_is_not_mutated(self) -> None:
        source = source_frame()
        before = source["card1"].to_numpy(copy=True)
        permute_card1(source, permutation_seed=1)
        np.testing.assert_array_equal(before, source["card1"].to_numpy())

    def test_a_permutation_seed_is_deterministic(self) -> None:
        source = source_frame()
        first = permute_card1(source, permutation_seed=7)["card1"].to_numpy()
        second = permute_card1(source, permutation_seed=7)["card1"].to_numpy()
        np.testing.assert_array_equal(first, second)

    def test_different_permutation_seeds_give_different_assignments(self) -> None:
        source = source_frame()
        first = permute_card1(source, permutation_seed=1)["card1"].to_numpy()
        second = permute_card1(source, permutation_seed=2)["card1"].to_numpy()
        self.assertTrue((first != second).any())

    def test_missing_entity_values_are_carried_through_the_multiset(self) -> None:
        source = source_frame()
        source.loc[:9, "card1"] = np.nan
        permuted = permute_card1(source, permutation_seed=1)
        self.assertEqual(
            int(source["card1"].isna().sum()), int(permuted["card1"].isna().sum())
        )


class FrozenConfigurationTests(unittest.TestCase):
    """Only the entity assignment may differ from the converged B1-card1 run."""

    def test_the_converged_cap_and_frozen_seed_are_accepted(self) -> None:
        validate_permuted_null_configuration(permuted_null_model(), frozen_metadata())

    def test_a_changed_learning_rate_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_permuted_null_configuration(
                permuted_null_model(learning_rate=0.05), frozen_metadata()
            )

    def test_a_changed_num_leaves_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_permuted_null_configuration(
                permuted_null_model(num_leaves=128), frozen_metadata()
            )

    def test_a_changed_training_seed_is_rejected(self) -> None:
        """random_state stays pinned: only the permutation seed varies."""
        with self.assertRaises(ValueError):
            validate_permuted_null_configuration(
                permuted_null_model(random_state=202), frozen_metadata()
            )

    def test_the_frozen_capped_estimator_count_is_rejected(self) -> None:
        model = build_lightgbm_model(SCALE_POS_WEIGHT, n_estimators=MAX_ESTIMATORS)
        model.set_params(random_state=RANDOM_SEED)
        with self.assertRaises(ValueError):
            validate_permuted_null_configuration(model, frozen_metadata())

    def test_a_changed_class_weight_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_permuted_null_configuration(
                permuted_null_model(), frozen_metadata(scale_pos_weight=20.0)
            )

    def test_missing_frozen_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_permuted_null_configuration(permuted_null_model(), {"scale_pos_weight": 1.0})


class VerdictTests(unittest.TestCase):
    """The rule was fixed before any permuted run was scored."""

    def test_a_positive_point_estimate_inside_the_interval_does_not_beat_b0(self) -> None:
        """The seed-1 case: +0.00004 on an interval spanning zero is not a win."""
        table = pd.DataFrame([comparison_row(1, 0.000036, -0.003597, 0.003596)])
        outcome = classify_permuted_outcome(table)
        self.assertEqual(outcome["outcome_code"], OUTCOME_AT_OR_BELOW_B0)
        self.assertFalse(outcome["any_permuted_variant_significantly_beats_b0"])

    def test_a_significant_positive_delta_recovers_the_gain(self) -> None:
        table = pd.DataFrame([comparison_row(1, 0.005, 0.001, 0.009)])
        outcome = classify_permuted_outcome(table)
        self.assertEqual(outcome["outcome_code"], OUTCOME_RECOVERS_GAIN)
        self.assertTrue(outcome["any_permuted_variant_significantly_beats_b0"])

    def test_a_significant_negative_delta_supports_the_null(self) -> None:
        """A permuted variant losing to B0 is evidence for entity history."""
        table = pd.DataFrame([comparison_row(2, -0.005968, -0.009592, -0.002395)])
        outcome = classify_permuted_outcome(table)
        self.assertEqual(outcome["outcome_code"], OUTCOME_AT_OR_BELOW_B0)

    def test_one_significant_seed_out_of_three_flips_the_verdict(self) -> None:
        table = pd.DataFrame(
            [
                comparison_row(1, 0.000036, -0.003597, 0.003596),
                comparison_row(2, 0.005, 0.001, 0.009),
                comparison_row(3, -0.003429, -0.007101, 0.000160),
            ]
        )
        outcome = classify_permuted_outcome(table)
        self.assertEqual(outcome["outcome_code"], OUTCOME_RECOVERS_GAIN)

    def test_the_published_panel_reproduces_the_published_verdict(self) -> None:
        table = pd.DataFrame(
            [
                comparison_row(1, 0.000036, -0.003597, 0.003596),
                comparison_row(2, -0.005968, -0.009592, -0.002395),
                comparison_row(3, -0.003429, -0.007101, 0.000160),
            ]
        )
        outcome = classify_permuted_outcome(table)
        self.assertEqual(outcome["outcome_code"], OUTCOME_AT_OR_BELOW_B0)
        # Recorded, not hidden: seed 1's point estimate sits a hair above B0,
        # so the strict at-or-below flag is False while the verdict stands.
        self.assertFalse(outcome["all_permuted_variants_at_or_below_b0"])

    def test_an_empty_panel_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            classify_permuted_outcome(pd.DataFrame(columns=["delta_pr_auc_vs_b0"]))


class SpreadTests(unittest.TestCase):
    def test_a_single_run_reports_no_standard_deviation(self) -> None:
        result = spread(np.array([0.005]))
        self.assertEqual(result["n"], 1)
        self.assertIsNone(result["std"])

    def test_a_panel_reports_the_sample_standard_deviation(self) -> None:
        values = np.array([0.000036, -0.005968, -0.003429])
        result = spread(values)
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["std"], float(values.std(ddof=1)))
        self.assertAlmostEqual(result["min"], -0.005968)
        self.assertAlmostEqual(result["max"], 0.000036)


class DiscoverRunsTests(unittest.TestCase):
    def _discover_in(self, root: Path) -> list[int]:
        with patch("src.models.compare_permuted_null.REPORT_DIR", root), patch(
            "src.models.train_lightgbm_permuted_null.REPORT_DIR", root
        ):
            return discover_runs()

    def test_completed_runs_are_collected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, 1)
            write_run(root, 2)
            self.assertEqual(self._discover_in(root), [1, 2])

    def test_a_run_outside_the_default_panel_is_still_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, 99)
            self.assertEqual(self._discover_in(root), [99])

    def test_an_incomplete_run_directory_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, 1)
            (root / "permseed_2" / f"cap{CONVERGED_MAX_ESTIMATORS}").mkdir(parents=True)
            self.assertEqual(self._discover_in(root), [1])

    def test_a_non_numeric_directory_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, 1)
            (root / "permseed_draft").mkdir(parents=True)
            self.assertEqual(self._discover_in(root), [1])

    def test_an_empty_tree_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                self._discover_in(Path(tmp))


if __name__ == "__main__":
    unittest.main()
