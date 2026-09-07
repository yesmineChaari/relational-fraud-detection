from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.summarize_convergence_check import (
    VERDICT_CAP_STILL_BINDING,
    VERDICT_CONVERGED_CONCLUSIONS_SURVIVE,
    VERDICT_HEADLINE_REVISION_REQUIRED,
    build_cap_decision,
)
from src.models.train_lightgbm_baseline import build_lightgbm_model
from src.models.train_lightgbm_convergence_check import (
    ALL_PROTECTED_PATHS,
    CONFIGURATIONS,
    STOP_METRICS,
    resolve_run_paths,
    summarize_learning_curve_for_stop_metric,
    validate_convergence_configuration,
)
from src.models.train_lightgbm_g1_controls import G1_PROTECTED_PATHS
from src.models.train_lightgbm_relational import B0_PROTECTED_PATHS, MAX_ESTIMATORS
from src.models.train_lightgbm_g1 import B1_CARD1_PROTECTED_PATHS

SCALE_POS_WEIGHT = 27.5
DEFAULT_MAX_ESTIMATORS = 15_000


def frozen_metadata(**overrides) -> dict:
    model = build_lightgbm_model(SCALE_POS_WEIGHT, n_estimators=MAX_ESTIMATORS)
    metadata = {
        "lightgbm_parameters": model.get_params(deep=False),
        "scale_pos_weight": SCALE_POS_WEIGHT,
    }
    metadata.update(overrides)
    return metadata


def convergence_model(seed: int, n_estimators: int, **overrides):
    model = build_lightgbm_model(SCALE_POS_WEIGHT, n_estimators=n_estimators)
    model.set_params(random_state=seed, **overrides)
    return model


class RunPathTests(unittest.TestCase):
    def test_every_configuration_and_stop_metric_resolves(self) -> None:
        for config in CONFIGURATIONS:
            for stop_metric in STOP_METRICS:
                paths = resolve_run_paths(config, stop_metric, DEFAULT_MAX_ESTIMATORS, 42)
                posix = paths["run_dir"].as_posix()
                self.assertIn("convergence_check", posix)
                self.assertIn(config, posix)
                self.assertIn(f"stop_{stop_metric}", posix)
                self.assertIn(f"cap{DEFAULT_MAX_ESTIMATORS}_seed42", posix)

    def test_unknown_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_paths("b1_nonexistent", "average_precision", DEFAULT_MAX_ESTIMATORS, 42)

    def test_unknown_stop_metric_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_paths("b0", "logloss", DEFAULT_MAX_ESTIMATORS, 42)

    def test_negative_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_paths("b0", "average_precision", DEFAULT_MAX_ESTIMATORS, -1)

    def test_non_positive_cap_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_paths("b0", "average_precision", 0, 42)


class ProtectedArtifactTests(unittest.TestCase):
    """No convergence-check run may write to a frozen artifact this investigation reads."""

    def test_frozen_b0_artifacts_are_protected(self) -> None:
        for path in B0_PROTECTED_PATHS:
            self.assertIn(path, ALL_PROTECTED_PATHS)

    def test_frozen_b1_card1_artifacts_are_protected(self) -> None:
        for path in B1_CARD1_PROTECTED_PATHS:
            self.assertIn(path, ALL_PROTECTED_PATHS)

    def test_frozen_g1_artifacts_are_protected(self) -> None:
        for path in G1_PROTECTED_PATHS:
            self.assertIn(path, ALL_PROTECTED_PATHS)

    def test_no_run_writes_to_a_protected_path(self) -> None:
        protected = {p.resolve() for p in ALL_PROTECTED_PATHS}
        for config in CONFIGURATIONS:
            for stop_metric in STOP_METRICS:
                for seed in (42, 202):
                    paths = resolve_run_paths(config, stop_metric, DEFAULT_MAX_ESTIMATORS, seed)
                    for role, path in paths.items():
                        if role == "run_dir":
                            continue
                        self.assertNotIn(
                            path.resolve(),
                            protected,
                            f"{config}/{stop_metric}/seed{seed} would overwrite a "
                            f"frozen artifact via {role}.",
                        )

    def test_runs_do_not_collide_with_each_other_across_stop_metric_or_cap(self) -> None:
        seen: set = set()
        for config in CONFIGURATIONS:
            for stop_metric in STOP_METRICS:
                for cap in (6_000, DEFAULT_MAX_ESTIMATORS):
                    for seed in (42, 202):
                        path = resolve_run_paths(config, stop_metric, cap, seed)["model"]
                        self.assertNotIn(path, seen, f"Model path collision at {path}.")
                        seen.add(path)

    def test_no_run_lands_under_a_frozen_report_tree(self) -> None:
        forbidden_fragments = (
            "reports/baseline",
            "reports/b1/",
            "reports/g1/",
            "reports/seed_variance",
            "models/seed_variance",
        )
        for config in CONFIGURATIONS:
            for stop_metric in STOP_METRICS:
                paths = resolve_run_paths(config, stop_metric, DEFAULT_MAX_ESTIMATORS, 42)
                for role, path in paths.items():
                    posix = path.as_posix()
                    for fragment in forbidden_fragments:
                        self.assertNotIn(fragment, posix, f"{role} path lands under {fragment}: {posix}")


class FrozenConfigurationTests(unittest.TestCase):
    """Only random_state and n_estimators may differ from frozen B0."""

    def test_a_cap_and_seed_only_change_is_accepted(self) -> None:
        for seed in (42, 202):
            validate_convergence_configuration(
                convergence_model(seed, DEFAULT_MAX_ESTIMATORS),
                frozen_metadata(),
                seed,
                DEFAULT_MAX_ESTIMATORS,
            )

    def test_a_cap_only_change_at_the_frozen_seed_is_accepted(self) -> None:
        validate_convergence_configuration(
            convergence_model(42, DEFAULT_MAX_ESTIMATORS),
            frozen_metadata(),
            42,
            DEFAULT_MAX_ESTIMATORS,
        )

    def test_a_changed_learning_rate_is_rejected(self) -> None:
        model = convergence_model(42, DEFAULT_MAX_ESTIMATORS, learning_rate=0.05)
        with self.assertRaises(ValueError) as caught:
            validate_convergence_configuration(model, frozen_metadata(), 42, DEFAULT_MAX_ESTIMATORS)
        self.assertIn("learning_rate", str(caught.exception))

    def test_a_changed_subsample_is_rejected(self) -> None:
        model = convergence_model(42, DEFAULT_MAX_ESTIMATORS, subsample=0.5)
        with self.assertRaises(ValueError):
            validate_convergence_configuration(model, frozen_metadata(), 42, DEFAULT_MAX_ESTIMATORS)

    def test_a_changed_class_weight_is_rejected(self) -> None:
        model = convergence_model(42, DEFAULT_MAX_ESTIMATORS)
        with self.assertRaises(ValueError):
            validate_convergence_configuration(
                model,
                frozen_metadata(scale_pos_weight=SCALE_POS_WEIGHT + 1.0),
                42,
                DEFAULT_MAX_ESTIMATORS,
            )

    def test_a_seed_that_disagrees_with_the_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_convergence_configuration(
                convergence_model(42, DEFAULT_MAX_ESTIMATORS),
                frozen_metadata(),
                202,
                DEFAULT_MAX_ESTIMATORS,
            )

    def test_a_cap_that_disagrees_with_the_model_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_convergence_configuration(
                convergence_model(42, DEFAULT_MAX_ESTIMATORS),
                frozen_metadata(),
                42,
                20_000,
            )


class LearningCurveGeneralizationTests(unittest.TestCase):
    def make_curve(self, ap_values: list[float], auc_values: list[float]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "iteration": np.arange(1, len(ap_values) + 1),
                "validation_average_precision": ap_values,
                "validation_auc": auc_values,
            }
        )

    def test_average_precision_stop_metric_matches_the_existing_helper(self) -> None:
        curve = self.make_curve([0.60, 0.65, 0.70, 0.68, 0.66], [0.90, 0.91, 0.92, 0.93, 0.94])
        result = summarize_learning_curve_for_stop_metric(
            curve, best_iteration=3, maximum_estimators=10, stop_metric="average_precision"
        )
        self.assertEqual(result["best_iteration"], 3)
        self.assertTrue(result["early_stopping_triggered"])

    def test_auc_stop_metric_checks_against_the_auc_argmax(self) -> None:
        """PR-AUC peaks at round 3, AUC peaks at round 5 -- decoupled patience follows AUC."""
        curve = self.make_curve([0.60, 0.65, 0.70, 0.68, 0.66], [0.90, 0.91, 0.92, 0.93, 0.95])
        result = summarize_learning_curve_for_stop_metric(
            curve, best_iteration=5, maximum_estimators=5, stop_metric="auc"
        )
        self.assertEqual(result["best_iteration"], 5)
        self.assertAlmostEqual(result["best_validation_average_precision"], 0.66)
        self.assertAlmostEqual(result["best_validation_auc"], 0.95)

    def test_auc_stop_metric_rejects_a_best_iteration_off_the_auc_argmax(self) -> None:
        curve = self.make_curve([0.60, 0.65, 0.70, 0.68, 0.66], [0.90, 0.91, 0.92, 0.93, 0.95])
        with self.assertRaises(AssertionError):
            summarize_learning_curve_for_stop_metric(
                curve, best_iteration=3, maximum_estimators=5, stop_metric="auc"
            )


def _convergence(config: str, triggered: bool) -> dict:
    return {
        "extended_maximum_estimators": 15_000,
        "extended_best_iteration": 9_000 if triggered else 15_000,
        "extended_early_stopping_triggered": triggered,
        "extended_pr_auc": 0.65,
        "extended_roc_auc": 0.93,
        "frozen_maximum_estimators": 6_000,
        "frozen_best_iteration": 5_900,
        "frozen_pr_auc": 0.649,
        "frozen_roc_auc": 0.925,
        "best_iteration_gap": 3_100 if triggered else 9_100,
    }


def _comparison(delta_shift: float) -> dict:
    return {
        "reference": "b0",
        "candidate": "b1_card1",
        "capped_delta_pr_auc": 0.0059,
        "converged_delta_pr_auc": 0.0059 + delta_shift,
        "delta_shift": delta_shift,
    }


class CapDecisionTests(unittest.TestCase):
    NOISE_FLOOR = 0.00051  # the panel's published clean-stratum std

    def test_missing_convergence_points_are_not_decidable(self) -> None:
        result = build_cap_decision(
            {"b0": _convergence("b0", True)}, None, None, self.NOISE_FLOOR
        )
        self.assertFalse(result["decidable"])
        self.assertIsNone(result["verdict"])

    def test_any_config_still_capped_is_cap_still_binding(self) -> None:
        convergence = {
            "b0": _convergence("b0", True),
            "b1_card1": _convergence("b1_card1", False),
            "g1_card1": _convergence("g1_card1", True),
        }
        result = build_cap_decision(
            convergence, _comparison(0.0), _comparison(0.0), self.NOISE_FLOOR
        )
        self.assertEqual(result["verdict"], VERDICT_CAP_STILL_BINDING)

    def test_converged_deltas_within_noise_survive(self) -> None:
        convergence = {name: _convergence(name, True) for name in CONFIGURATIONS}
        small_shift = self.NOISE_FLOOR  # within 2x the noise floor
        result = build_cap_decision(
            convergence, _comparison(small_shift), _comparison(-small_shift), self.NOISE_FLOOR
        )
        self.assertEqual(result["verdict"], VERDICT_CONVERGED_CONCLUSIONS_SURVIVE)

    def test_a_large_converged_shift_requires_revision(self) -> None:
        convergence = {name: _convergence(name, True) for name in CONFIGURATIONS}
        large_shift = 10.0 * self.NOISE_FLOOR
        result = build_cap_decision(
            convergence, _comparison(large_shift), _comparison(0.0), self.NOISE_FLOOR
        )
        self.assertEqual(result["verdict"], VERDICT_HEADLINE_REVISION_REQUIRED)


if __name__ == "__main__":
    unittest.main()
