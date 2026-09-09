from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models.compare_ablation import (
    OUTCOME_ADDITIVE,
    OUTCOME_NOT_LOCALISED,
    OUTCOME_PANEL_INCOMPLETE,
    OUTCOME_REDUNDANT,
    OUTCOME_SINGLE_FEATURE,
    classify_ablation_outcome,
    discover_runs,
    load_noise_floor,
    rank_singletons,
    spread,
)
from src.models.train_lightgbm_ablation import (
    ABLATION_FEATURES,
    ALL_PROTECTED_PATHS,
    B0_CONVERGED_PATHS,
    B1_CARD1_CONVERGED_PATHS,
    FEATURE_KEYS,
    KEY_TO_FEATURE,
    LEAVE_ONE_OUT,
    MODES,
    MODE_REFERENCE,
    RELATION,
    SINGLETON,
    STOP_METRIC,
    build_ablation_feature_manifest,
    feature_subset,
    main as ablation_main,
    planned_runs,
    resolve_feature,
    resolve_run_paths,
    validate_ablation_configuration,
)
from src.models.train_lightgbm_baseline import RANDOM_SEED, build_lightgbm_model
from src.models.train_lightgbm_convergence_check import (
    DEFAULT_MAX_ESTIMATORS as CONVERGED_MAX_ESTIMATORS,
)
from src.models.train_lightgbm_g1 import B1_CARD1_PROTECTED_PATHS
from src.models.train_lightgbm_relational import (
    B0_PROTECTED_PATHS,
    EXPECTED_B0_FEATURE_COUNT,
    MAX_ESTIMATORS,
)

SCALE_POS_WEIGHT = 27.434310083918007
NOISE_FLOOR = 0.000514200988964303
REAL_GAIN = 0.006304055057609892

OUTPUT_KEYS = [
    "model",
    "metrics",
    "metadata",
    "feature_importance",
    "validation_predictions",
    "learning_curve",
]


def b0_features(count: int = EXPECTED_B0_FEATURE_COUNT) -> list[str]:
    return [f"f{i}" for i in range(count)]


def frozen_metadata(**overrides) -> dict:
    model = build_lightgbm_model(SCALE_POS_WEIGHT, n_estimators=MAX_ESTIMATORS)
    metadata = {
        "lightgbm_parameters": model.get_params(deep=False),
        "scale_pos_weight": SCALE_POS_WEIGHT,
    }
    metadata.update(overrides)
    return metadata


def ablation_model(**overrides):
    model = build_lightgbm_model(SCALE_POS_WEIGHT, n_estimators=CONVERGED_MAX_ESTIMATORS)
    parameters = {"random_state": RANDOM_SEED}
    parameters.update(overrides)
    model.set_params(**parameters)
    return model


def comparison_row(
    mode: str,
    feature: str,
    delta: float,
    ci_lower: float,
    ci_upper: float,
) -> dict:
    reference_pr_auc = 0.649193708652322 if mode == SINGLETON else 0.6554977637099318
    return {
        "ablation_mode": mode,
        "feature": feature,
        "feature_key": FEATURE_KEYS[feature],
        "pr_auc": reference_pr_auc + delta,
        "reference_pr_auc": reference_pr_auc,
        "delta_pr_auc_vs_reference": delta,
        "delta_pr_auc_vs_reference_ci_lower_95": ci_lower,
        "delta_pr_auc_vs_reference_ci_upper_95": ci_upper,
        "delta_pr_auc_vs_reference_excludes_zero": bool(ci_lower > 0.0 or ci_upper < 0.0),
        "delta_exceeds_noise_floor": bool(abs(delta) > NOISE_FLOOR),
        "fraction_of_real_gain": delta / REAL_GAIN,
    }


def null_panel() -> list[dict]:
    """Eight runs, none of them significant in either direction."""
    return [
        comparison_row(mode, feature, 0.00002, -0.0011, 0.0013)
        for mode in MODES
        for feature in ABLATION_FEATURES
    ]


def write_run(
    root: Path, mode: str, feature: str, complete: bool = True, seed: int = RANDOM_SEED
) -> None:
    run_dir = (
        root / f"{mode}_{FEATURE_KEYS[feature]}" / f"cap{CONVERGED_MAX_ESTIMATORS}_seed{seed}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"pr_auc": 0.6512, "ablation_mode": mode, "ablated_feature": feature}, handle)
    if complete:
        pd.DataFrame(
            {"TransactionID": [1, 2], "isFraud": [0, 1], "prediction": [0.1, 0.9]}
        ).to_parquet(run_dir / "validation_predictions.parquet", index=False)


class RunPathTests(unittest.TestCase):
    def test_every_planned_run_resolves_under_the_ablation_tree(self) -> None:
        planned = planned_runs()
        self.assertEqual(len(planned), len(MODES) * len(ABLATION_FEATURES))
        for mode, feature in planned:
            paths = resolve_run_paths(mode, feature)
            posix = paths["run_dir"].as_posix()
            self.assertIn("ablation", posix)
            self.assertIn(RELATION, posix)
            self.assertIn(f"{mode}_{FEATURE_KEYS[feature]}", posix)
            self.assertIn(f"cap{CONVERGED_MAX_ESTIMATORS}", posix)

    def test_every_planned_run_has_a_distinct_output_path(self) -> None:
        seen: set[str] = set()
        for mode, feature in planned_runs():
            paths = resolve_run_paths(mode, feature)
            for key in OUTPUT_KEYS:
                posix = paths[key].as_posix()
                self.assertNotIn(posix, seen, f"{mode}/{feature} reuses an output path: {posix}")
                seen.add(posix)

    def test_short_key_and_full_column_name_resolve_to_the_same_run(self) -> None:
        for feature in ABLATION_FEATURES:
            key = FEATURE_KEYS[feature]
            self.assertEqual(resolve_feature(key), feature)
            self.assertEqual(resolve_feature(feature), feature)
            self.assertEqual(
                resolve_run_paths(SINGLETON, key), resolve_run_paths(SINGLETON, feature)
            )

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_paths("halves", ABLATION_FEATURES[0])

    def test_unknown_feature_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_paths(SINGLETON, "card1_prior_count_30d")

    def test_no_ablation_output_path_is_a_protected_artifact(self) -> None:
        protected = {path.as_posix() for path in ALL_PROTECTED_PATHS}
        for mode, feature in planned_runs():
            paths = resolve_run_paths(mode, feature)
            for key in OUTPUT_KEYS:
                self.assertNotIn(paths[key].as_posix(), protected)


class SeedDimensionTests(unittest.TestCase):
    def test_the_default_seed_is_the_frozen_seed(self) -> None:
        default = resolve_run_paths(SINGLETON, ABLATION_FEATURES[0])
        explicit = resolve_run_paths(SINGLETON, ABLATION_FEATURES[0], RANDOM_SEED)
        self.assertEqual(default, explicit)
        self.assertIn(f"seed{RANDOM_SEED}", default["run_dir"].as_posix())

    def test_a_different_seed_lands_in_a_different_run(self) -> None:
        frozen = resolve_run_paths(LEAVE_ONE_OUT, ABLATION_FEATURES[1], RANDOM_SEED)
        other = resolve_run_paths(LEAVE_ONE_OUT, ABLATION_FEATURES[1], 202)
        for key in OUTPUT_KEYS:
            self.assertNotEqual(frozen[key], other[key])
        self.assertIn("seed202", other["run_dir"].as_posix())
        self.assertIn("seed202", other["model"].name)

    def test_negative_seed_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_run_paths(SINGLETON, ABLATION_FEATURES[0], -1)

    def test_a_seed_panel_run_must_carry_its_own_random_state(self) -> None:
        manifest = build_ablation_feature_manifest(b0_features(), [ABLATION_FEATURES[0]])
        validate_ablation_configuration(
            ablation_model(random_state=202),
            frozen_metadata(),
            manifest,
            [ABLATION_FEATURES[0]],
            202,
        )

    def test_a_seed_panel_run_carrying_the_frozen_seed_is_rejected(self) -> None:
        """Asking for seed 202 and training under 42 would silently duplicate a panel cell."""
        manifest = build_ablation_feature_manifest(b0_features(), [ABLATION_FEATURES[0]])
        with self.assertRaises(ValueError):
            validate_ablation_configuration(
                ablation_model(random_state=RANDOM_SEED),
                frozen_metadata(),
                manifest,
                [ABLATION_FEATURES[0]],
                202,
            )

    def test_every_other_frozen_parameter_still_binds_under_a_seed_panel(self) -> None:
        manifest = build_ablation_feature_manifest(b0_features(), [ABLATION_FEATURES[0]])
        with self.assertRaises(ValueError):
            validate_ablation_configuration(
                ablation_model(random_state=202, num_leaves=32),
                frozen_metadata(),
                manifest,
                [ABLATION_FEATURES[0]],
                202,
            )


class CommandLineTests(unittest.TestCase):
    """The CLI must actually reach the trainer.

    An earlier revision advertised --seed on the parser but never passed it
    through main(), so every seed-panel run silently trained under the frozen
    seed and overwrote nothing. These assert the wiring, not just the flag.
    """

    def run_main(self, argv: list[str]) -> list[tuple]:
        calls: list[tuple] = []

        def record(mode, feature, seed=RANDOM_SEED, skip_existing=False):
            calls.append((mode, feature, seed, skip_existing))
            return {}

        with patch("src.models.train_lightgbm_ablation.run_ablation", record):
            ablation_main(argv)
        return calls

    def test_a_requested_seed_reaches_the_trainer(self) -> None:
        calls = self.run_main(
            ["--mode", "loo", "--feature", "prior_count_24h", "--seed", "202"]
        )
        self.assertEqual(len(calls), 1)
        mode, feature, seed, _ = calls[0]
        self.assertEqual((mode, feature, seed), (LEAVE_ONE_OUT, KEY_TO_FEATURE["prior_count_24h"], 202))

    def test_several_seeds_expand_into_several_runs(self) -> None:
        calls = self.run_main(
            [
                "--mode", "loo",
                "--feature", "prior_count_24h",
                "--seed", "202",
                "--seed", "707",
                "--seed", "1337",
            ]
        )
        self.assertEqual([seed for _, _, seed, _ in calls], [202, 707, 1337])

    def test_omitting_the_seed_uses_the_frozen_seed(self) -> None:
        calls = self.run_main(["--mode", "singleton", "--feature", "prior_count"])
        self.assertEqual([seed for _, _, seed, _ in calls], [RANDOM_SEED])

    def test_the_default_invocation_plans_the_whole_panel_at_the_frozen_seed(self) -> None:
        calls = self.run_main([])
        self.assertEqual(len(calls), len(MODES) * len(ABLATION_FEATURES))
        self.assertEqual({seed for _, _, seed, _ in calls}, {RANDOM_SEED})

    def test_skip_existing_reaches_the_trainer(self) -> None:
        calls = self.run_main(
            ["--mode", "loo", "--feature", "prior_count_24h", "--skip-existing"]
        )
        self.assertTrue(all(skip for _, _, _, skip in calls))


class FeatureSubsetTests(unittest.TestCase):
    def test_singleton_carries_exactly_the_named_feature(self) -> None:
        for feature in ABLATION_FEATURES:
            self.assertEqual(feature_subset(SINGLETON, feature), [feature])

    def test_leave_one_out_carries_the_other_three_in_canonical_order(self) -> None:
        for feature in ABLATION_FEATURES:
            subset = feature_subset(LEAVE_ONE_OUT, feature)
            self.assertEqual(len(subset), len(ABLATION_FEATURES) - 1)
            self.assertNotIn(feature, subset)
            self.assertEqual(subset, [f for f in ABLATION_FEATURES if f != feature])

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            feature_subset("pairs", ABLATION_FEATURES[0])


class FeatureManifestTests(unittest.TestCase):
    def test_singleton_manifest_adds_exactly_one_predictor(self) -> None:
        base = b0_features()
        manifest = build_ablation_feature_manifest(base, [ABLATION_FEATURES[0]])
        self.assertEqual(len(manifest), EXPECTED_B0_FEATURE_COUNT + 1)
        self.assertEqual(manifest[: len(base)], base)
        self.assertEqual(manifest[len(base) :], [ABLATION_FEATURES[0]])

    def test_leave_one_out_manifest_adds_exactly_three_predictors(self) -> None:
        base = b0_features()
        for feature in ABLATION_FEATURES:
            subset = feature_subset(LEAVE_ONE_OUT, feature)
            manifest = build_ablation_feature_manifest(base, subset)
            self.assertEqual(len(manifest), EXPECTED_B0_FEATURE_COUNT + 3)
            self.assertNotIn(feature, manifest)

    def test_subset_order_is_canonical_regardless_of_input_order(self) -> None:
        base = b0_features()
        subset = list(reversed(ABLATION_FEATURES[:3]))
        manifest = build_ablation_feature_manifest(base, subset)
        self.assertEqual(manifest[len(base) :], ABLATION_FEATURES[:3])

    def test_empty_subset_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_ablation_feature_manifest(b0_features(), [])

    def test_full_four_feature_subset_is_rejected_as_b1_itself(self) -> None:
        with self.assertRaises(ValueError):
            build_ablation_feature_manifest(b0_features(), list(ABLATION_FEATURES))

    def test_duplicate_subset_entry_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_ablation_feature_manifest(
                b0_features(), [ABLATION_FEATURES[0], ABLATION_FEATURES[0]]
            )

    def test_unknown_subset_feature_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_ablation_feature_manifest(b0_features(), ["card1_prior_count_30d"])

    def test_relational_feature_already_in_b0_is_rejected(self) -> None:
        base = [*b0_features(), ABLATION_FEATURES[0]]
        with self.assertRaises(ValueError):
            build_ablation_feature_manifest(base, [ABLATION_FEATURES[0]])

    def test_duplicate_b0_manifest_is_rejected(self) -> None:
        base = b0_features()
        with self.assertRaises(ValueError):
            build_ablation_feature_manifest([*base, base[0]], [ABLATION_FEATURES[0]])


class ConfigurationTests(unittest.TestCase):
    def test_matching_singleton_configuration_is_accepted(self) -> None:
        manifest = build_ablation_feature_manifest(b0_features(), [ABLATION_FEATURES[0]])
        validate_ablation_configuration(
            ablation_model(), frozen_metadata(), manifest, [ABLATION_FEATURES[0]]
        )

    def test_matching_leave_one_out_configuration_is_accepted(self) -> None:
        subset = feature_subset(LEAVE_ONE_OUT, ABLATION_FEATURES[0])
        manifest = build_ablation_feature_manifest(b0_features(), subset)
        validate_ablation_configuration(ablation_model(), frozen_metadata(), manifest, subset)

    def test_capped_run_is_rejected(self) -> None:
        model = build_lightgbm_model(SCALE_POS_WEIGHT, n_estimators=MAX_ESTIMATORS)
        manifest = build_ablation_feature_manifest(b0_features(), [ABLATION_FEATURES[0]])
        with self.assertRaises(ValueError):
            validate_ablation_configuration(
                model, frozen_metadata(), manifest, [ABLATION_FEATURES[0]]
            )

    def test_unpinned_training_seed_is_rejected(self) -> None:
        manifest = build_ablation_feature_manifest(b0_features(), [ABLATION_FEATURES[0]])
        with self.assertRaises(ValueError):
            validate_ablation_configuration(
                ablation_model(random_state=7),
                frozen_metadata(),
                manifest,
                [ABLATION_FEATURES[0]],
            )

    def test_drifted_learning_rate_is_rejected(self) -> None:
        manifest = build_ablation_feature_manifest(b0_features(), [ABLATION_FEATURES[0]])
        with self.assertRaises(ValueError):
            validate_ablation_configuration(
                ablation_model(learning_rate=0.05),
                frozen_metadata(),
                manifest,
                [ABLATION_FEATURES[0]],
            )

    def test_drifted_class_weight_is_rejected(self) -> None:
        manifest = build_ablation_feature_manifest(b0_features(), [ABLATION_FEATURES[0]])
        with self.assertRaises(ValueError):
            validate_ablation_configuration(
                ablation_model(),
                frozen_metadata(scale_pos_weight=SCALE_POS_WEIGHT + 1.0),
                manifest,
                [ABLATION_FEATURES[0]],
            )

    def test_manifest_disagreeing_with_the_subset_is_rejected(self) -> None:
        """A four-feature manifest paired with a singleton subset must not pass."""
        manifest = [*b0_features(), *ABLATION_FEATURES]
        with self.assertRaises(ValueError):
            validate_ablation_configuration(
                ablation_model(), frozen_metadata(), manifest, [ABLATION_FEATURES[0]]
            )


class ProtectedArtifactTests(unittest.TestCase):
    def test_frozen_and_converged_references_are_all_protected(self) -> None:
        protected = {path.as_posix() for path in ALL_PROTECTED_PATHS}
        for path in [*B0_PROTECTED_PATHS, *B1_CARD1_PROTECTED_PATHS]:
            self.assertIn(path.as_posix(), protected)
        for paths in (B0_CONVERGED_PATHS, B1_CARD1_CONVERGED_PATHS):
            for key, path in paths.items():
                if key == "run_dir":
                    continue
                self.assertIn(path.as_posix(), protected)

    def test_the_real_card1_relational_artifacts_are_protected(self) -> None:
        protected = {path.as_posix() for path in ALL_PROTECTED_PATHS}
        self.assertTrue(
            any(f"relational_features_{RELATION}.parquet" in p for p in protected),
            "The card1 relational feature parquet must be protected from overwrite.",
        )

    def test_converged_references_use_the_average_precision_protocol(self) -> None:
        for paths in (B0_CONVERGED_PATHS, B1_CARD1_CONVERGED_PATHS):
            posix = paths["metrics"].as_posix()
            self.assertIn(f"stop_{STOP_METRIC}", posix)
            self.assertIn(f"cap{CONVERGED_MAX_ESTIMATORS}_seed{RANDOM_SEED}", posix)

    def test_each_mode_names_the_reference_it_is_read_against(self) -> None:
        self.assertEqual(MODE_REFERENCE[SINGLETON], "b0_converged")
        self.assertEqual(MODE_REFERENCE[LEAVE_ONE_OUT], "b1_card1_converged")
        self.assertEqual(sorted(MODE_REFERENCE), sorted(MODES))


class NoiseFloorTests(unittest.TestCase):
    def test_the_floor_is_read_from_the_seed_variance_panel(self) -> None:
        self.assertAlmostEqual(load_noise_floor(), NOISE_FLOOR, places=12)

    def test_a_summary_without_the_clean_stratum_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed_variance_summary.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump({"paired_delta_pr_auc_spread": {"std": 0.006}}, handle)
            with self.assertRaises(KeyError):
                load_noise_floor(path)

    def test_a_non_positive_floor_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "seed_variance_summary.json"
            with path.open("w", encoding="utf-8") as handle:
                json.dump(
                    {"early_stopping_stratification": {"clean_delta_pr_auc": {"std": 0.0}}}, handle
                )
            with self.assertRaises(ValueError):
                load_noise_floor(path)


class OutcomeClassificationTests(unittest.TestCase):
    def test_no_significant_result_in_either_direction_is_not_localised(self) -> None:
        outcome = classify_ablation_outcome(pd.DataFrame(null_panel()), NOISE_FLOOR)
        self.assertEqual(outcome["outcome_code"], OUTCOME_NOT_LOCALISED)
        self.assertEqual(outcome["features_carrying_the_gain"], [])
        self.assertEqual(outcome["non_redundant_features"], [])

    def test_one_carrier_that_is_also_non_redundant_localises_the_gain(self) -> None:
        target = ABLATION_FEATURES[0]
        rows = []
        for feature in ABLATION_FEATURES:
            if feature == target:
                rows.append(comparison_row(SINGLETON, feature, 0.0061, 0.0033, 0.0089))
                rows.append(comparison_row(LEAVE_ONE_OUT, feature, -0.0058, -0.0086, -0.0030))
            else:
                rows.append(comparison_row(SINGLETON, feature, 0.00004, -0.0011, 0.0012))
                rows.append(comparison_row(LEAVE_ONE_OUT, feature, -0.00003, -0.0012, 0.0011))
        outcome = classify_ablation_outcome(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertEqual(outcome["outcome_code"], OUTCOME_SINGLE_FEATURE)
        self.assertEqual(outcome["features_carrying_the_gain"], [target])
        self.assertEqual(outcome["non_redundant_features"], [target])

    def test_several_carriers_and_no_non_redundant_feature_is_redundancy(self) -> None:
        rows = []
        for feature in ABLATION_FEATURES:
            rows.append(comparison_row(SINGLETON, feature, 0.0059, 0.0031, 0.0087))
            rows.append(comparison_row(LEAVE_ONE_OUT, feature, -0.00006, -0.0013, 0.0012))
        outcome = classify_ablation_outcome(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertEqual(outcome["outcome_code"], OUTCOME_REDUNDANT)
        self.assertEqual(outcome["features_carrying_the_gain"], sorted(ABLATION_FEATURES))
        self.assertEqual(outcome["non_redundant_features"], [])

    def test_two_non_redundant_features_are_additive(self) -> None:
        rows = []
        for index, feature in enumerate(ABLATION_FEATURES):
            rows.append(comparison_row(SINGLETON, feature, 0.0040, 0.0018, 0.0062))
            if index < 2:
                rows.append(comparison_row(LEAVE_ONE_OUT, feature, -0.0031, -0.0059, -0.0008))
            else:
                rows.append(comparison_row(LEAVE_ONE_OUT, feature, -0.00002, -0.0012, 0.0011))
        outcome = classify_ablation_outcome(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertEqual(outcome["outcome_code"], OUTCOME_ADDITIVE)
        self.assertEqual(outcome["non_redundant_features"], sorted(ABLATION_FEATURES[:2]))

    def test_a_non_redundant_feature_without_any_carrier_is_additive(self) -> None:
        rows = []
        for index, feature in enumerate(ABLATION_FEATURES):
            rows.append(comparison_row(SINGLETON, feature, 0.00003, -0.0011, 0.0012))
            if index == 0:
                rows.append(comparison_row(LEAVE_ONE_OUT, feature, -0.0033, -0.0061, -0.0009))
            else:
                rows.append(comparison_row(LEAVE_ONE_OUT, feature, -0.00002, -0.0012, 0.0011))
        outcome = classify_ablation_outcome(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertEqual(outcome["outcome_code"], OUTCOME_ADDITIVE)

    def test_a_positive_leave_one_out_interval_is_not_read_as_non_redundant(self) -> None:
        """Dropping a feature and scoring significantly *higher* is not evidence it was needed."""
        rows = []
        for index, feature in enumerate(ABLATION_FEATURES):
            rows.append(comparison_row(SINGLETON, feature, 0.0059, 0.0031, 0.0087))
            if index == 0:
                rows.append(comparison_row(LEAVE_ONE_OUT, feature, 0.0033, 0.0009, 0.0061))
            else:
                rows.append(comparison_row(LEAVE_ONE_OUT, feature, -0.00002, -0.0012, 0.0011))
        outcome = classify_ablation_outcome(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertEqual(outcome["non_redundant_features"], [])
        self.assertEqual(outcome["outcome_code"], OUTCOME_REDUNDANT)

    def test_a_negative_singleton_interval_is_not_read_as_carrying_the_gain(self) -> None:
        rows = [comparison_row(SINGLETON, ABLATION_FEATURES[0], -0.0061, -0.0089, -0.0033)]
        rows += [
            comparison_row(SINGLETON, feature, 0.00003, -0.0011, 0.0012)
            for feature in ABLATION_FEATURES[1:]
        ]
        rows += [
            comparison_row(LEAVE_ONE_OUT, feature, -0.00002, -0.0012, 0.0011)
            for feature in ABLATION_FEATURES
        ]
        outcome = classify_ablation_outcome(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertEqual(outcome["features_carrying_the_gain"], [])
        self.assertEqual(outcome["outcome_code"], OUTCOME_NOT_LOCALISED)

    def test_a_partial_panel_is_flagged_as_incomplete(self) -> None:
        rows = [comparison_row(SINGLETON, feature, 0.0001, -0.001, 0.001) for feature in ABLATION_FEATURES]
        outcome = classify_ablation_outcome(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertFalse(outcome["panel_complete"])
        self.assertEqual(outcome["outcome_code"], OUTCOME_PANEL_INCOMPLETE)

    def test_a_partial_panel_never_emits_a_substantive_outcome(self) -> None:
        """A missing direction must not be read as a measured null.

        With no leave-one-out run trained, `non_redundant` is empty for the
        same reason it would be if every leave-one-out variant came back null
        -- so no code that depends on that emptiness may be emitted.
        """
        rows = [comparison_row(SINGLETON, ABLATION_FEATURES[0], 0.0093, 0.0057, 0.0129)]
        rows += [
            comparison_row(SINGLETON, feature, 0.00003, -0.0011, 0.0012)
            for feature in ABLATION_FEATURES[1:]
        ]
        outcome = classify_ablation_outcome(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertEqual(outcome["outcome_code"], OUTCOME_PANEL_INCOMPLETE)
        self.assertNotIn(
            outcome["outcome_code"],
            {OUTCOME_REDUNDANT, OUTCOME_SINGLE_FEATURE, OUTCOME_ADDITIVE, OUTCOME_NOT_LOCALISED},
        )

    def test_a_partial_panel_still_reports_the_carriers_it_measured(self) -> None:
        target = ABLATION_FEATURES[0]
        rows = [comparison_row(SINGLETON, target, 0.0093, 0.0057, 0.0129)]
        outcome = classify_ablation_outcome(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertEqual(outcome["features_carrying_the_gain"], [target])
        self.assertEqual(outcome["non_redundant_features"], [])
        self.assertEqual(outcome["outcome_code"], OUTCOME_PANEL_INCOMPLETE)

    def test_a_single_missing_leave_one_out_run_still_blocks_a_verdict(self) -> None:
        rows = [
            comparison_row(SINGLETON, feature, 0.0059, 0.0031, 0.0087)
            for feature in ABLATION_FEATURES
        ]
        rows += [
            comparison_row(LEAVE_ONE_OUT, feature, -0.00006, -0.0013, 0.0012)
            for feature in ABLATION_FEATURES[:-1]
        ]
        outcome = classify_ablation_outcome(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertEqual(outcome["outcome_code"], OUTCOME_PANEL_INCOMPLETE)

    def test_a_full_panel_is_flagged_as_complete(self) -> None:
        outcome = classify_ablation_outcome(pd.DataFrame(null_panel()), NOISE_FLOOR)
        self.assertTrue(outcome["panel_complete"])

    def test_an_empty_table_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            classify_ablation_outcome(pd.DataFrame(columns=["ablation_mode"]), NOISE_FLOOR)


class SingletonRankingTests(unittest.TestCase):
    def test_singletons_are_ranked_by_descending_pr_auc(self) -> None:
        rows = [
            comparison_row(SINGLETON, ABLATION_FEATURES[0], 0.0010, -0.001, 0.003),
            comparison_row(SINGLETON, ABLATION_FEATURES[1], 0.0060, 0.003, 0.009),
            comparison_row(SINGLETON, ABLATION_FEATURES[2], 0.0030, 0.001, 0.005),
            comparison_row(SINGLETON, ABLATION_FEATURES[3], 0.0020, 0.000, 0.004),
        ]
        ranking = rank_singletons(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertEqual(
            [entry["feature"] for entry in ranking],
            [
                ABLATION_FEATURES[1],
                ABLATION_FEATURES[2],
                ABLATION_FEATURES[3],
                ABLATION_FEATURES[0],
            ],
        )
        self.assertEqual([entry["rank"] for entry in ranking], [1, 2, 3, 4])

    def test_a_margin_below_the_noise_floor_is_not_interpretable(self) -> None:
        rows = [
            comparison_row(SINGLETON, ABLATION_FEATURES[0], 0.0060, 0.003, 0.009),
            comparison_row(SINGLETON, ABLATION_FEATURES[1], 0.0060 - NOISE_FLOOR / 2, 0.003, 0.009),
        ]
        ranking = rank_singletons(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertFalse(ranking[0]["margin_exceeds_noise_floor"])

    def test_a_margin_above_the_noise_floor_is_interpretable(self) -> None:
        rows = [
            comparison_row(SINGLETON, ABLATION_FEATURES[0], 0.0060, 0.003, 0.009),
            comparison_row(SINGLETON, ABLATION_FEATURES[1], 0.0060 - NOISE_FLOOR * 4, 0.003, 0.009),
        ]
        ranking = rank_singletons(pd.DataFrame(rows), NOISE_FLOOR)
        self.assertTrue(ranking[0]["margin_exceeds_noise_floor"])

    def test_the_last_ranked_singleton_has_no_margin(self) -> None:
        ranking = rank_singletons(pd.DataFrame(null_panel()), NOISE_FLOOR)
        self.assertIsNone(ranking[-1]["margin_over_next"])
        self.assertIsNone(ranking[-1]["margin_exceeds_noise_floor"])

    def test_leave_one_out_runs_are_excluded_from_the_singleton_ranking(self) -> None:
        ranking = rank_singletons(pd.DataFrame(null_panel()), NOISE_FLOOR)
        self.assertEqual(len(ranking), len(ABLATION_FEATURES))


class DiscoverRunsTests(unittest.TestCase):
    def test_completed_runs_are_discovered_across_both_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, SINGLETON, ABLATION_FEATURES[0])
            write_run(root, LEAVE_ONE_OUT, ABLATION_FEATURES[2])
            with patch("src.models.train_lightgbm_ablation.REPORT_DIR", root):
                self.assertEqual(
                    discover_runs(),
                    [
                        (SINGLETON, ABLATION_FEATURES[0]),
                        (LEAVE_ONE_OUT, ABLATION_FEATURES[2]),
                    ],
                )

    def test_a_run_missing_its_predictions_is_treated_as_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, SINGLETON, ABLATION_FEATURES[0])
            write_run(root, SINGLETON, ABLATION_FEATURES[1], complete=False)
            with patch("src.models.train_lightgbm_ablation.REPORT_DIR", root):
                self.assertEqual(discover_runs(), [(SINGLETON, ABLATION_FEATURES[0])])

    def test_an_empty_report_tree_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.models.train_lightgbm_ablation.REPORT_DIR", Path(tmp)):
                with self.assertRaises(FileNotFoundError):
                    discover_runs()

    def test_a_seed_panel_run_is_not_mistaken_for_a_panel_cell(self) -> None:
        """The verdict covers the frozen-seed panel; other seeds must not leak into it."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_run(root, SINGLETON, ABLATION_FEATURES[0])
            write_run(root, LEAVE_ONE_OUT, ABLATION_FEATURES[1], seed=202)
            with patch("src.models.train_lightgbm_ablation.REPORT_DIR", root):
                self.assertEqual(discover_runs(), [(SINGLETON, ABLATION_FEATURES[0])])


class SpreadTests(unittest.TestCase):
    def test_a_single_value_reports_no_standard_deviation(self) -> None:
        result = spread(np.array([0.0061]))
        self.assertEqual(result["n"], 1)
        self.assertIsNone(result["std"])

    def test_several_values_report_the_sample_standard_deviation(self) -> None:
        values = np.array([0.001, 0.002, 0.003])
        result = spread(values)
        self.assertEqual(result["n"], 3)
        self.assertAlmostEqual(result["std"], float(values.std(ddof=1)))
        self.assertAlmostEqual(result["min"], 0.001)
        self.assertAlmostEqual(result["max"], 0.003)


class FeatureKeyTests(unittest.TestCase):
    def test_keys_are_unique_and_round_trip_to_their_columns(self) -> None:
        self.assertEqual(len(KEY_TO_FEATURE), len(ABLATION_FEATURES))
        for feature, key in FEATURE_KEYS.items():
            self.assertNotIn(RELATION, key)
            self.assertEqual(KEY_TO_FEATURE[key], feature)

    def test_the_panel_covers_all_four_summaries_in_both_directions(self) -> None:
        planned = planned_runs()
        for mode in MODES:
            covered = sorted(feature for run_mode, feature in planned if run_mode == mode)
            self.assertEqual(covered, sorted(ABLATION_FEATURES))


if __name__ == "__main__":
    unittest.main()
