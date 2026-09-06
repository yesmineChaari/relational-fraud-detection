from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.summarize_seed_variance import (
    CANDIDATE_CONFIG,
    REFERENCE_CONFIG,
    build_interpretation,
    discover_runs,
    paired_seed_deltas,
    spread,
    stratify_by_early_stopping,
)
from src.models.train_lightgbm_baseline import build_lightgbm_model
from src.models.train_lightgbm_relational import (
    B0_PROTECTED_PATHS,
    FROZEN_PARAMETER_NAMES,
    MAX_ESTIMATORS,
)
from src.models.train_seed_variants import (
    CONFIGURATIONS,
    DEFAULT_SEEDS,
    PROTECTED_PATHS,
    RANDOM_SEED,
    check_frozen_reproduction,
    resolve_run_paths,
    validate_seed_variant_configuration,
)

SCALE_POS_WEIGHT = 27.5


def frozen_metadata(**overrides) -> dict:
    """A stand-in for the frozen B0 metadata, built from the frozen builder."""
    model = build_lightgbm_model(SCALE_POS_WEIGHT, n_estimators=MAX_ESTIMATORS)
    parameters = model.get_params(deep=False)
    metadata = {
        "lightgbm_parameters": parameters,
        "scale_pos_weight": SCALE_POS_WEIGHT,
    }
    metadata.update(overrides)
    return metadata


def seed_variant_model(seed: int, **overrides):
    model = build_lightgbm_model(SCALE_POS_WEIGHT, n_estimators=MAX_ESTIMATORS)
    model.set_params(random_state=seed, **overrides)
    return model


def write_run(root: Path, config: str, seed: int, **overrides) -> None:
    run_dir = root / config / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "configuration": config,
        "random_seed": seed,
        "pr_auc": 0.65,
        "roc_auc": 0.92,
        "best_iteration": 5_900,
        "actual_stopping_iteration": 6_000,
        "maximum_estimators": 6_000,
        "early_stopping_triggered": False,
        "estimator_cap_reached": True,
        "test_evaluated": False,
    }
    metrics.update(overrides)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle)


class RunPathTests(unittest.TestCase):
    def test_every_configuration_and_default_seed_resolves(self) -> None:
        for config in CONFIGURATIONS:
            for seed in DEFAULT_SEEDS:
                paths = resolve_run_paths(config, seed)
                self.assertIn(f"seed_{seed}", paths["run_dir"].as_posix())
                self.assertIn("seed_variance", paths["run_dir"].as_posix())

    def test_unknown_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_paths("b1_nonexistent", 42)

    def test_negative_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_paths("b0", -1)

    def test_five_seeds_are_planned(self) -> None:
        """FRM's requirement is at least five seeds per configuration."""
        self.assertGreaterEqual(len(DEFAULT_SEEDS), 5)
        self.assertEqual(len(set(DEFAULT_SEEDS)), len(DEFAULT_SEEDS))

    def test_the_frozen_seed_is_in_the_panel(self) -> None:
        """Seed 42 doubles as the harness fidelity check against the frozen runs."""
        self.assertIn(RANDOM_SEED, DEFAULT_SEEDS)


class ProtectedArtifactTests(unittest.TestCase):
    """The frozen B0 and B1 artifacts must survive every seed run untouched."""

    def test_frozen_b0_artifacts_are_protected(self) -> None:
        for path in B0_PROTECTED_PATHS:
            self.assertIn(path, PROTECTED_PATHS)

    def test_frozen_b1_card1_artifacts_are_protected(self) -> None:
        protected = {p.as_posix() for p in PROTECTED_PATHS}
        self.assertTrue(
            any("lightgbm_b1_card1.txt" in p for p in protected),
            "The frozen B1-card1 model is not hash-pinned.",
        )
        self.assertTrue(
            any(p.endswith("reports/b1/card1/metrics.json") for p in protected),
            "The frozen B1-card1 metrics are not hash-pinned.",
        )

    def test_no_seed_run_writes_to_a_protected_path(self) -> None:
        protected = {p.resolve() for p in PROTECTED_PATHS}
        for config in CONFIGURATIONS:
            for seed in DEFAULT_SEEDS:
                for role, path in resolve_run_paths(config, seed).items():
                    self.assertNotIn(
                        path.resolve(),
                        protected,
                        f"{config}/seed_{seed} would overwrite a frozen artifact via {role}.",
                    )

    def test_seed_runs_do_not_collide_with_each_other(self) -> None:
        seen: set[Path] = set()
        for config in CONFIGURATIONS:
            for seed in DEFAULT_SEEDS:
                path = resolve_run_paths(config, seed)["model"]
                self.assertNotIn(path, seen, f"Model path collision at {path}.")
                seen.add(path)


class FrozenConfigurationTests(unittest.TestCase):
    """random_state is the only parameter allowed to move."""

    def test_a_seed_only_change_is_accepted(self) -> None:
        for seed in DEFAULT_SEEDS:
            validate_seed_variant_configuration(
                seed_variant_model(seed), frozen_metadata(), seed, MAX_ESTIMATORS
            )

    def test_random_state_is_not_pinned_to_the_frozen_seed(self) -> None:
        """The guard would be self-defeating if it forced random_state back to 42."""
        self.assertIn("random_state", FROZEN_PARAMETER_NAMES)
        validate_seed_variant_configuration(
            seed_variant_model(2024), frozen_metadata(), 2024, MAX_ESTIMATORS
        )

    def test_a_changed_learning_rate_is_rejected(self) -> None:
        model = seed_variant_model(202, learning_rate=0.05)
        with self.assertRaises(ValueError) as caught:
            validate_seed_variant_configuration(
                model, frozen_metadata(), 202, MAX_ESTIMATORS
            )
        self.assertIn("learning_rate", str(caught.exception))

    def test_a_changed_subsample_is_rejected(self) -> None:
        """Subsampling is what makes a seed change bite; it must stay frozen."""
        model = seed_variant_model(202, subsample=0.5)
        with self.assertRaises(ValueError):
            validate_seed_variant_configuration(
                model, frozen_metadata(), 202, MAX_ESTIMATORS
            )

    def test_a_seed_that_disagrees_with_the_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_seed_variant_configuration(
                seed_variant_model(202), frozen_metadata(), 707, MAX_ESTIMATORS
            )

    def test_a_changed_class_weight_is_rejected(self) -> None:
        model = seed_variant_model(202)
        with self.assertRaises(ValueError):
            validate_seed_variant_configuration(
                model,
                frozen_metadata(scale_pos_weight=SCALE_POS_WEIGHT + 1.0),
                202,
                MAX_ESTIMATORS,
            )

    def test_the_estimator_cap_cannot_be_lowered_at_the_frozen_cap(self) -> None:
        model = seed_variant_model(202)
        model.set_params(n_estimators=3_000)
        with self.assertRaises(ValueError):
            validate_seed_variant_configuration(
                model, frozen_metadata(), 202, MAX_ESTIMATORS
            )


class FrozenReproductionTests(unittest.TestCase):
    def test_the_check_only_applies_to_the_frozen_seed(self) -> None:
        self.assertFalse(check_frozen_reproduction("b0", 2024, 0.65)["applicable"])
        self.assertTrue(
            check_frozen_reproduction("b0", RANDOM_SEED, 0.65)["applicable"]
        )


class SpreadTests(unittest.TestCase):
    def test_sample_standard_deviation_is_used(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = spread(values)
        self.assertAlmostEqual(result["std"], float(values.std(ddof=1)))
        self.assertNotAlmostEqual(result["std"], float(values.std(ddof=0)))

    def test_range_and_bounds_are_reported(self) -> None:
        result = spread(np.array([0.64, 0.65, 0.66]))
        self.assertEqual(result["n_seeds"], 3)
        self.assertAlmostEqual(result["min"], 0.64)
        self.assertAlmostEqual(result["max"], 0.66)
        self.assertAlmostEqual(result["range"], 0.02)

    def test_a_single_run_cannot_produce_a_spread(self) -> None:
        with self.assertRaises(ValueError):
            spread(np.array([0.65]))


class PairedDeltaTests(unittest.TestCase):
    def make_runs(self, rows: list[tuple[str, int, float]]) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "configuration": config,
                    "seed": seed,
                    "pr_auc": pr_auc,
                    "roc_auc": 0.92,
                    "best_iteration": 5_900,
                    "early_stopping_triggered": False,
                }
                for config, seed, pr_auc in rows
            ]
        )

    def test_deltas_are_paired_by_seed(self) -> None:
        runs = self.make_runs(
            [
                (REFERENCE_CONFIG, 42, 0.6400),
                (REFERENCE_CONFIG, 202, 0.6300),
                (CANDIDATE_CONFIG, 42, 0.6500),
                (CANDIDATE_CONFIG, 202, 0.6350),
            ]
        )
        deltas = paired_seed_deltas(runs).set_index("seed")
        self.assertAlmostEqual(deltas.loc[42, "delta_pr_auc"], 0.0100)
        self.assertAlmostEqual(deltas.loc[202, "delta_pr_auc"], 0.0050)

    def test_unpaired_seeds_are_dropped_rather_than_mismatched(self) -> None:
        """A half-finished panel must not pair seed 42's B0 with seed 202's B1."""
        runs = self.make_runs(
            [
                (REFERENCE_CONFIG, 42, 0.6400),
                (REFERENCE_CONFIG, 202, 0.6300),
                (CANDIDATE_CONFIG, 42, 0.6500),
            ]
        )
        deltas = paired_seed_deltas(runs)
        self.assertEqual(list(deltas["seed"]), [42])

    def test_no_shared_seed_is_an_error(self) -> None:
        runs = self.make_runs(
            [(REFERENCE_CONFIG, 42, 0.64), (CANDIDATE_CONFIG, 202, 0.65)]
        )
        with self.assertRaises(ValueError):
            paired_seed_deltas(runs)


def make_paired_runs(rows: list[tuple[int, float, float, bool, bool]]) -> pd.DataFrame:
    """(seed, b0_pr_auc, b1_pr_auc, b0_stopped_early, b1_stopped_early)."""
    records = []
    for seed, b0_pr, b1_pr, b0_stop, b1_stop in rows:
        records.append(
            {
                "configuration": REFERENCE_CONFIG,
                "seed": seed,
                "pr_auc": b0_pr,
                "roc_auc": 0.9200,
                "best_iteration": 3_648 if b0_stop else 5_900,
                "early_stopping_triggered": b0_stop,
            }
        )
        records.append(
            {
                "configuration": CANDIDATE_CONFIG,
                "seed": seed,
                "pr_auc": b1_pr,
                "roc_auc": 0.9226,
                "best_iteration": 3_644 if b1_stop else 5_950,
                "early_stopping_triggered": b1_stop,
            }
        )
    return pd.DataFrame(records)


class EarlyStoppingStratificationTests(unittest.TestCase):
    def test_seeds_are_split_by_whether_either_model_stopped_early(self) -> None:
        runs = make_paired_runs(
            [
                (42, 0.6491, 0.6550, False, False),
                (202, 0.6401, 0.6541, True, False),
                (707, 0.6448, 0.6426, False, True),
                (1337, 0.6428, 0.6494, False, False),
            ]
        )
        result = stratify_by_early_stopping(paired_seed_deltas(runs))
        self.assertEqual(result["n_clean"], 2)
        self.assertEqual(result["n_contaminated"], 2)
        self.assertEqual(result["clean_seeds"], [42, 1337])
        self.assertEqual(result["contaminated_seeds"], [202, 707])

    def test_which_model_stopped_is_recorded_per_seed(self) -> None:
        runs = make_paired_runs(
            [
                (42, 0.6491, 0.6550, False, False),
                (202, 0.6401, 0.6541, True, False),
                (707, 0.6448, 0.6426, False, True),
            ]
        )
        status = stratify_by_early_stopping(paired_seed_deltas(runs))[
            "which_stopped_per_seed"
        ]
        self.assertEqual(status[42], "neither")
        self.assertEqual(status[202], REFERENCE_CONFIG)
        self.assertEqual(status[707], CANDIDATE_CONFIG)

    def test_roc_auc_is_reported_across_every_seed(self) -> None:
        """ROC-AUC is not the stopping criterion, so no seed is excluded from it."""
        runs = make_paired_runs(
            [
                (42, 0.6491, 0.6550, False, False),
                (202, 0.6401, 0.6541, True, False),
                (707, 0.6448, 0.6426, False, True),
            ]
        )
        result = stratify_by_early_stopping(paired_seed_deltas(runs))
        self.assertEqual(result["roc_auc_delta_all_seeds"]["n_seeds"], 3)
        self.assertTrue(result["roc_auc_delta_sign_stable"])

    def test_a_single_clean_seed_yields_no_clean_spread(self) -> None:
        """One run cannot establish a spread, so the stratum is left unreported."""
        runs = make_paired_runs(
            [
                (42, 0.6491, 0.6550, False, False),
                (202, 0.6401, 0.6541, True, False),
            ]
        )
        result = stratify_by_early_stopping(paired_seed_deltas(runs))
        self.assertEqual(result["n_clean"], 1)
        self.assertNotIn("clean_delta_pr_auc", result)


class InterpretationTests(unittest.TestCase):
    def make_deltas(self, values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "seed": list(range(len(values))),
                "delta_pr_auc": values,
                "delta_roc_auc": values,
            }
        )

    def test_a_sign_flip_across_seeds_is_called_unstable(self) -> None:
        deltas = self.make_deltas([0.006, -0.004, 0.002, -0.001, 0.003])
        result = build_interpretation(
            spread(deltas["delta_pr_auc"].to_numpy()), deltas, {"available": False}
        )
        self.assertEqual(result["verdict"], "unstable")
        self.assertFalse(result["paired_delta_sign_stable_across_seeds"])

    def test_a_tight_positive_delta_is_called_stable(self) -> None:
        deltas = self.make_deltas([0.0059, 0.0061, 0.0058, 0.0060, 0.0057])
        result = build_interpretation(
            spread(deltas["delta_pr_auc"].to_numpy()), deltas, {"available": False}
        )
        self.assertEqual(result["verdict"], "stable")
        self.assertTrue(result["paired_delta_positive_in_every_seed"])

    def test_a_positive_but_noisy_delta_is_called_within_seed_noise(self) -> None:
        deltas = self.make_deltas([0.0010, 0.0090, 0.0020, 0.0085, 0.0015])
        result = build_interpretation(
            spread(deltas["delta_pr_auc"].to_numpy()), deltas, {"available": False}
        )
        self.assertEqual(result["verdict"], "within_seed_noise")
        self.assertTrue(result["paired_delta_sign_stable_across_seeds"])

    def test_seed_variance_exceeding_the_bootstrap_is_reported(self) -> None:
        deltas = self.make_deltas([0.0010, 0.0090, 0.0020, 0.0085, 0.0015])
        result = build_interpretation(
            spread(deltas["delta_pr_auc"].to_numpy()),
            deltas,
            {"available": True, "bootstrap_std_delta": 0.0005},
        )
        self.assertTrue(result["seed_variance_dominates_sampling_variance"])
        self.assertIn("understates the true uncertainty", result["plain_language"])

    def test_a_sign_flip_confined_to_early_stopped_seeds_is_called_confounded(
        self,
    ) -> None:
        """The real panel's shape: unstable overall, tight where the contest is fair."""
        runs = make_paired_runs(
            [
                (42, 0.64913857, 0.65503913, False, False),
                (202, 0.64012433, 0.65414788, True, False),
                (707, 0.64480743, 0.64264537, False, True),
                (1337, 0.64279500, 0.64942275, False, False),
                (2024, 0.64860778, 0.64974079, False, True),
            ]
        )
        deltas = paired_seed_deltas(runs)
        stratification = stratify_by_early_stopping(deltas)
        result = build_interpretation(
            spread(deltas["delta_pr_auc"].to_numpy()),
            deltas,
            {"available": False},
            stratification,
        )
        self.assertEqual(result["verdict"], "confounded_by_early_stopping")
        self.assertFalse(result["paired_delta_sign_stable_across_seeds"])
        self.assertTrue(result["early_stopping_confound_explains_instability"])
        self.assertIn("relational gain is real", result["plain_language"])

    def test_a_sign_flip_among_clean_seeds_stays_unstable(self) -> None:
        """If the fair comparisons themselves disagree, the confound is no excuse."""
        runs = make_paired_runs(
            [
                (42, 0.6491, 0.6550, False, False),
                (202, 0.6500, 0.6440, False, False),
                (707, 0.6448, 0.6426, False, True),
            ]
        )
        deltas = paired_seed_deltas(runs)
        result = build_interpretation(
            spread(deltas["delta_pr_auc"].to_numpy()),
            deltas,
            {"available": False},
            stratify_by_early_stopping(deltas),
        )
        self.assertEqual(result["verdict"], "unstable")
        self.assertFalse(result["early_stopping_confound_explains_instability"])

    def test_a_dominant_bootstrap_is_reported_as_not_optimistic(self) -> None:
        deltas = self.make_deltas([0.0059, 0.0061, 0.0058, 0.0060, 0.0057])
        result = build_interpretation(
            spread(deltas["delta_pr_auc"].to_numpy()),
            deltas,
            {"available": True, "bootstrap_std_delta": 0.0030},
        )
        self.assertFalse(result["seed_variance_dominates_sampling_variance"])
        self.assertIn("not materially optimistic", result["plain_language"])


class DiscoverRunsTests(unittest.TestCase):
    def test_completed_runs_are_collected_across_configurations(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, REFERENCE_CONFIG, 42, pr_auc=0.6400)
            write_run(root, CANDIDATE_CONFIG, 42, pr_auc=0.6500)
            with patch("src.models.summarize_seed_variance.REPORT_DIR", root):
                runs = discover_runs()
        self.assertEqual(len(runs), 2)
        self.assertEqual(set(runs["configuration"]), {REFERENCE_CONFIG, CANDIDATE_CONFIG})

    def test_an_incomplete_run_directory_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, REFERENCE_CONFIG, 42)
            (root / REFERENCE_CONFIG / "seed_202").mkdir(parents=True)
            with patch("src.models.summarize_seed_variance.REPORT_DIR", root):
                runs = discover_runs()
        self.assertEqual(list(runs["seed"]), [42])

    def test_a_test_evaluated_run_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, REFERENCE_CONFIG, 42, test_evaluated=True)
            with patch("src.models.summarize_seed_variance.REPORT_DIR", root):
                with self.assertRaises(ValueError):
                    discover_runs()

    def test_a_misfiled_run_is_refused(self) -> None:
        """A metrics file whose configuration disagrees with its directory."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, REFERENCE_CONFIG, 42, configuration=CANDIDATE_CONFIG)
            with patch("src.models.summarize_seed_variance.REPORT_DIR", root):
                with self.assertRaises(ValueError):
                    discover_runs()

    def test_an_empty_tree_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.models.summarize_seed_variance.REPORT_DIR", Path(tmp)):
                with self.assertRaises(FileNotFoundError):
                    discover_runs()


if __name__ == "__main__":
    unittest.main()
