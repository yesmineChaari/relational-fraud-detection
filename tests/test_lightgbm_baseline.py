from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.data.build_model_dataset import add_time_features, merge_identity_data
from src.models.train_lightgbm_baseline import (
    MISSING_TOKEN,
    UNKNOWN_TOKEN,
    RunConfig,
    apply_category_mapping,
    assert_no_forbidden_features,
    build_artifact_paths,
    build_lightgbm_model,
    fit_category_mapping,
    get_feature_columns,
    precision_recall_at_fraction,
    resolve_scale_pos_weight,
)


class DatasetBuilderTests(unittest.TestCase):
    def test_identity_merge_is_left_join_and_tracks_presence(self) -> None:
        transactions = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "TransactionDT": [86_400, 90_000, 176_400],
                "isFraud": [0, 1, 0],
            }
        )
        identity = pd.DataFrame(
            {
                "TransactionID": [2],
                "DeviceType": ["mobile"],
            }
        )

        merged = merge_identity_data(transactions, identity)

        self.assertEqual(len(merged), 3)
        self.assertTrue(merged["TransactionID"].is_unique)
        self.assertEqual(merged["has_identity"].tolist(), [False, True, False])
        self.assertTrue(pd.isna(merged.loc[0, "DeviceType"]))

    def test_elapsed_time_features_do_not_infer_calendar_dates(self) -> None:
        frame = pd.DataFrame(
            {
                "TransactionDT": [86_400, 90_000, 176_400],
            }
        )

        result = add_time_features(frame)

        np.testing.assert_allclose(result["elapsed_days"], [1.0, 1.04166667, 2.04166667])
        self.assertEqual(result["hour_in_day"].tolist(), [0, 1, 1])
        self.assertEqual(result["day_in_week_cycle"].tolist(), [1, 1, 2])


class BaselinePreprocessingTests(unittest.TestCase):
    def test_mapping_distinguishes_missing_unknown_and_known(self) -> None:
        train = pd.Series(["known", None, "other", "known"], dtype="string")
        validation = pd.Series(["known", "new", None], dtype="string")

        mapping = fit_category_mapping(train)
        encoded = apply_category_mapping(validation, mapping)

        self.assertEqual(mapping[MISSING_TOKEN], 0)
        self.assertEqual(mapping[UNKNOWN_TOKEN], 1)
        self.assertNotIn("new", mapping)
        self.assertEqual(encoded.tolist(), [mapping["known"], 1, 0])

    def test_mapping_is_deterministic(self) -> None:
        first = fit_category_mapping(pd.Series(["b", "a", None]))
        second = fit_category_mapping(pd.Series([None, "a", "b", "a"]))
        self.assertEqual(first, second)

    def test_forbidden_features_raise(self) -> None:
        with self.assertRaises(AssertionError):
            assert_no_forbidden_features(["TransactionAmt", "isFraud"])

    def test_accidental_merge_features_raise(self) -> None:
        with self.assertRaises(AssertionError):
            assert_no_forbidden_features(["TransactionAmt", "TransactionID_y"])

    def test_relational_features_are_allowed(self) -> None:
        feature_columns = get_feature_columns(["TransactionID", "TransactionAmt", "graph_degree"])
        self.assertEqual(feature_columns, ["TransactionAmt", "graph_degree"])


class FinalizationConfigurationTests(unittest.TestCase):
    def test_weighted_config_uses_train_only_class_ratio(self) -> None:
        labels = pd.Series([0, 0, 0, 1], dtype="int8")
        self.assertEqual(resolve_scale_pos_weight("weighted", labels), 3.0)

    def test_unweighted_config_uses_scale_pos_weight_one(self) -> None:
        labels = pd.Series([0, 0, 0, 1], dtype="int8")
        self.assertEqual(resolve_scale_pos_weight("unweighted", labels), 1.0)

    def test_estimator_cap_is_configurable(self) -> None:
        model = build_lightgbm_model(1.0, n_estimators=6_000)
        self.assertEqual(model.get_params()["n_estimators"], 6_000)

    def test_early_stopping_rounds_are_configurable(self) -> None:
        config = RunConfig(
            run_name="weighted_6000",
            weighting="weighted",
            n_estimators=6_000,
            early_stopping_rounds=200,
        )
        config.validate()
        self.assertEqual(config.early_stopping_rounds, 200)

    def test_experiment_artifacts_do_not_overwrite_each_other(self) -> None:
        weighted = build_artifact_paths("weighted_6000")
        unweighted = build_artifact_paths("unweighted_6000")
        self.assertNotEqual(weighted.report_dir, unweighted.report_dir)
        self.assertNotEqual(weighted.model, unweighted.model)

    def test_unsafe_run_name_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            build_artifact_paths("../canonical")


class RankingMetricTests(unittest.TestCase):
    def test_precision_recall_at_fraction(self) -> None:
        labels = np.array([0, 1, 0, 1, 0], dtype=np.int8)
        probabilities = np.array([0.1, 0.9, 0.2, 0.8, 0.3])

        result = precision_recall_at_fraction(labels, probabilities, 0.4)

        self.assertEqual(result["k"], 2)
        self.assertEqual(result["frauds_found"], 2)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["recall"], 1.0)

    def test_precision_recall_rejects_invalid_fraction(self) -> None:
        with self.assertRaises(ValueError):
            precision_recall_at_fraction(
                np.array([0, 1]),
                np.array([0.1, 0.9]),
                0.0,
            )


if __name__ == "__main__":
    unittest.main()
