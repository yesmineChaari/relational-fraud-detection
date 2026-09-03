from __future__ import annotations

import copy
import unittest

import numpy as np
import pandas as pd

from src.features.build_relational_features import (
    OUTPUT_COLUMNS as RELATIONAL_OUTPUT_COLUMNS,
    RELATIONAL_FEATURES,
)
from src.models.train_lightgbm_baseline import (
    CATEGORY_MAPPINGS_PATH,
    build_lightgbm_model,
    evaluate_validation,
)
from src.models.train_lightgbm_relational import (
    EARLY_STOPPING_ROUNDS,
    EVALUATION_SPLIT,
    MAX_ESTIMATORS,
    TEST_EVALUATED,
    apply_frozen_category_mappings,
    attach_relational_features,
    build_b1_feature_manifest,
    build_comparison_to_b0,
    file_sha256,
    load_frozen_b0_metadata,
    load_frozen_category_mappings,
    validate_frozen_lightgbm_configuration,
    validate_model_columns_against_frozen_b0,
    validate_relational_feature_dtypes,
    validate_relational_merge,
)


def make_relational_table(transaction_ids: list[int]) -> pd.DataFrame:
    row_count = len(transaction_ids)
    return pd.DataFrame(
        {
            "TransactionID": transaction_ids,
            RELATIONAL_FEATURES[0]: np.arange(row_count, dtype=np.int64),
            RELATIONAL_FEATURES[1]: np.arange(row_count, dtype=np.int64),
            RELATIONAL_FEATURES[2]: np.arange(row_count, dtype=np.int64),
            RELATIONAL_FEATURES[3]: np.arange(row_count, dtype=np.float64),
        },
        columns=RELATIONAL_OUTPUT_COLUMNS,
    )


class RelationalMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_index = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "split": ["train", "validation", "test"],
            }
        )

    def test_one_to_one_merge_preserves_model_order_and_splits(self) -> None:
        relational = make_relational_table([3, 1, 2])

        merged = validate_relational_merge(
            self.model_index, relational, RELATIONAL_FEATURES
        )

        self.assertEqual(merged["TransactionID"].tolist(), [1, 2, 3])
        self.assertEqual(merged["split"].tolist(), ["train", "validation", "test"])
        self.assertEqual(list(merged.columns), ["TransactionID", "split", *RELATIONAL_FEATURES])

    def test_duplicate_relational_ids_fail(self) -> None:
        relational = make_relational_table([1, 1, 3])

        with self.assertRaises(ValueError):
            validate_relational_merge(
                self.model_index, relational, RELATIONAL_FEATURES
            )

    def test_missing_relational_ids_fail(self) -> None:
        relational = make_relational_table([1, 2])

        with self.assertRaises(ValueError):
            validate_relational_merge(
                self.model_index, relational, RELATIONAL_FEATURES
            )

    def test_extra_relational_ids_fail(self) -> None:
        relational = make_relational_table([1, 2, 3, 4])

        with self.assertRaises(ValueError):
            validate_relational_merge(
                self.model_index, relational, RELATIONAL_FEATURES
            )

    def test_partition_attachment_preserves_rows_and_split(self) -> None:
        merged = validate_relational_merge(
            self.model_index,
            make_relational_table([1, 2, 3]),
            RELATIONAL_FEATURES,
        )
        train = pd.DataFrame(
            {
                "TransactionID": [1],
                "split": pd.Series(["train"], dtype="category"),
                "isFraud": [0],
                "TransactionAmt": [10.0],
            }
        )

        result = attach_relational_features(
            train, merged, "train", RELATIONAL_FEATURES
        )

        self.assertEqual(len(result), 1)
        self.assertEqual(result["TransactionID"].tolist(), [1])
        self.assertEqual(result["split"].astype("string").tolist(), ["train"])
        self.assertTrue(set(RELATIONAL_FEATURES).issubset(result.columns))


class ControlledManifestTests(unittest.TestCase):
    def test_b1_manifest_is_b0_plus_exactly_four_features(self) -> None:
        b0_features = ["TransactionDT", "TransactionAmt", "ProductCD"]

        b1_features = build_b1_feature_manifest(b0_features, RELATIONAL_FEATURES)

        self.assertEqual(b1_features, [*b0_features, *RELATIONAL_FEATURES])
        self.assertEqual(len(b1_features) - len(b0_features), 4)

    def test_current_base_columns_must_match_frozen_manifest(self) -> None:
        b0_features = ["TransactionDT", "TransactionAmt"]
        model_columns = [
            "TransactionID",
            *b0_features,
            "isFraud",
            "split",
            "has_identity",
            *RELATIONAL_FEATURES,
        ]

        validate_model_columns_against_frozen_b0(
            model_columns, b0_features, RELATIONAL_FEATURES
        )

        with self.assertRaises(ValueError):
            validate_model_columns_against_frozen_b0(
                [*model_columns, "unexpected_base_feature"],
                b0_features,
                RELATIONAL_FEATURES,
            )

    def test_categorical_manifest_is_identical_to_b0(self) -> None:
        metadata = load_frozen_b0_metadata()
        b1_features = build_b1_feature_manifest(
            metadata["feature_columns"], RELATIONAL_FEATURES
        )
        categorical_columns = metadata["categorical_feature_columns"]

        self.assertTrue(set(categorical_columns).issubset(b1_features))
        self.assertFalse(set(categorical_columns) & set(RELATIONAL_FEATURES))
        self.assertEqual(len(categorical_columns), 31)

    def test_all_four_relational_features_are_numeric(self) -> None:
        relational = make_relational_table([1, 2, 3])

        validate_relational_feature_dtypes(relational, RELATIONAL_FEATURES)

        for column in RELATIONAL_FEATURES:
            self.assertTrue(pd.api.types.is_numeric_dtype(relational[column]))


class FrozenPreprocessingAndConfigurationTests(unittest.TestCase):
    def test_canonical_category_mapping_artifact_remains_unchanged(self) -> None:
        metadata = load_frozen_b0_metadata()
        categorical_columns = metadata["categorical_feature_columns"]
        hash_before = file_sha256(CATEGORY_MAPPINGS_PATH)

        mappings, mapping_hash = load_frozen_category_mappings(categorical_columns)

        self.assertEqual(mapping_hash, hash_before)
        self.assertEqual(file_sha256(CATEGORY_MAPPINGS_PATH), hash_before)
        self.assertEqual(list(mappings), categorical_columns)

    def test_applying_frozen_mappings_does_not_mutate_them(self) -> None:
        mappings = {
            "category": {
                "__MISSING__": 0,
                "__UNKNOWN__": 1,
                "known": 2,
            }
        }
        original = copy.deepcopy(mappings)
        train = pd.DataFrame({"category": ["known", None]})
        validation = pd.DataFrame({"category": ["new", None]})

        apply_frozen_category_mappings(
            train,
            validation,
            ["category"],
            mappings,
        )

        self.assertEqual(mappings, original)
        self.assertEqual(train["category"].tolist(), [2, 0])
        self.assertEqual(validation["category"].tolist(), [1, 0])

    def test_class_weight_matches_frozen_b0_policy(self) -> None:
        metadata = load_frozen_b0_metadata()
        negatives = metadata["training_rows"] - metadata["training_fraud_count"]
        expected_weight = negatives / metadata["training_fraud_count"]

        self.assertEqual(expected_weight, metadata["scale_pos_weight"])
        self.assertEqual(expected_weight, 27.434310083918007)

    def test_lightgbm_configuration_matches_frozen_b0(self) -> None:
        metadata = load_frozen_b0_metadata()
        model = build_lightgbm_model(
            metadata["scale_pos_weight"],
            n_estimators=MAX_ESTIMATORS,
        )

        validate_frozen_lightgbm_configuration(model, metadata)

        params = model.get_params(deep=False)
        self.assertEqual(params["learning_rate"], 0.03)
        self.assertEqual(params["num_leaves"], 64)
        self.assertEqual(params["n_estimators"], 6_000)
        self.assertEqual(EARLY_STOPPING_ROUNDS, 200)

    def test_metric_comparison_uses_b0_semantics(self) -> None:
        labels = pd.Series([0, 1, 0, 1], dtype="int8")
        b0_metrics = evaluate_validation(labels, np.array([0.1, 0.8, 0.2, 0.7]))
        b1_metrics = evaluate_validation(labels, np.array([0.1, 0.9, 0.2, 0.8]))

        comparison = build_comparison_to_b0(b0_metrics, b1_metrics)

        self.assertEqual(
            comparison["metric"].tolist(),
            [
                "PR-AUC",
                "ROC-AUC",
                "Precision@0.5%",
                "Recall@0.5%",
                "Precision@1%",
                "Recall@1%",
                "Precision@2%",
                "Recall@2%",
                "Precision@5%",
                "Recall@5%",
            ],
        )
        np.testing.assert_allclose(
            comparison["delta_b1_minus_b0"],
            comparison["b1_value"] - comparison["b0_value"],
        )

    def test_training_path_does_not_evaluate_final_test(self) -> None:
        self.assertEqual(EVALUATION_SPLIT, "validation")
        self.assertFalse(TEST_EVALUATED)


if __name__ == "__main__":
    unittest.main()
