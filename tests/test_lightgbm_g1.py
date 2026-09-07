from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.models.train_lightgbm_baseline import evaluate_validation
from src.models.train_lightgbm_g1 import (
    EMBEDDING_DIM,
    EXPECTED_G1_FEATURE_COUNT,
    TEST_EVALUATED,
    build_comparison_table,
    build_g1_feature_manifest,
    embedding_feature_names,
    embedding_gain_ranks,
    load_embedding_metadata,
    load_frozen_b0_metadata,
    load_frozen_b1_card1_metadata,
    validate_embedding_feature_dtypes,
    validate_embedding_merge,
)


def make_embedding_table(transaction_ids: list[int], splits: list[str]) -> pd.DataFrame:
    feat_names = embedding_feature_names()
    row_count = len(transaction_ids)
    data = {
        "TransactionID": transaction_ids,
        "split": pd.Series(splits, dtype="string"),
    }
    rng = np.random.default_rng(0)
    for name in feat_names:
        data[name] = rng.normal(size=row_count).astype(np.float32)
    return pd.DataFrame(data)


class EmbeddingFeatureNamesTests(unittest.TestCase):
    def test_names_are_zero_padded_and_ordered(self) -> None:
        names = embedding_feature_names(4)
        self.assertEqual(names, ["embedding_00", "embedding_01", "embedding_02", "embedding_03"])

    def test_default_width_matches_the_encoder(self) -> None:
        self.assertEqual(len(embedding_feature_names()), EMBEDDING_DIM)


class EmbeddingDtypeValidationTests(unittest.TestCase):
    def test_non_float_column_is_rejected(self) -> None:
        feat_names = ["embedding_00"]
        df = pd.DataFrame({"embedding_00": [1, 2, 3]})
        with self.assertRaises(TypeError):
            validate_embedding_feature_dtypes(df, feat_names)

    def test_non_finite_values_are_rejected(self) -> None:
        feat_names = ["embedding_00"]
        df = pd.DataFrame({"embedding_00": np.array([1.0, np.nan, 3.0], dtype=np.float32)})
        with self.assertRaises(ValueError):
            validate_embedding_feature_dtypes(df, feat_names)

    def test_finite_float_column_passes(self) -> None:
        feat_names = ["embedding_00"]
        df = pd.DataFrame({"embedding_00": np.array([1.0, 2.0, 3.0], dtype=np.float32)})
        validate_embedding_feature_dtypes(df, feat_names)


class EmbeddingMergeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_index = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "split": ["train", "validation", "test"],
            }
        )

    def test_one_to_one_merge_preserves_model_order_and_splits(self) -> None:
        embeddings = make_embedding_table([3, 1, 2], ["test", "train", "validation"])

        merged = validate_embedding_merge(self.model_index, embeddings, embedding_feature_names())

        self.assertEqual(merged["TransactionID"].tolist(), [1, 2, 3])
        self.assertEqual(merged["split"].tolist(), ["train", "validation", "test"])

    def test_duplicate_embedding_ids_fail(self) -> None:
        embeddings = make_embedding_table([1, 1, 3], ["train", "train", "test"])
        with self.assertRaises(ValueError):
            validate_embedding_merge(self.model_index, embeddings, embedding_feature_names())

    def test_missing_embedding_ids_fail(self) -> None:
        embeddings = make_embedding_table([1, 2], ["train", "validation"])
        with self.assertRaises(ValueError):
            validate_embedding_merge(self.model_index, embeddings, embedding_feature_names())

    def test_extra_embedding_ids_fail(self) -> None:
        embeddings = make_embedding_table([1, 2, 3, 4], ["train", "validation", "test", "train"])
        with self.assertRaises(ValueError):
            validate_embedding_merge(self.model_index, embeddings, embedding_feature_names())

    def test_split_disagreement_fails(self) -> None:
        embeddings = make_embedding_table([1, 2, 3], ["train", "train", "test"])
        with self.assertRaises(AssertionError):
            validate_embedding_merge(self.model_index, embeddings, embedding_feature_names())


class ControlledManifestTests(unittest.TestCase):
    def test_g1_manifest_is_b0_plus_exactly_embedding_dim_features(self) -> None:
        b0_features = ["TransactionDT", "TransactionAmt", "ProductCD"]
        feat_names = embedding_feature_names()

        g1_features = build_g1_feature_manifest(b0_features, feat_names)

        self.assertEqual(g1_features, [*b0_features, *feat_names])
        self.assertEqual(len(g1_features) - len(b0_features), EMBEDDING_DIM)

    def test_overlap_between_b0_and_embedding_names_fails(self) -> None:
        with self.assertRaises(ValueError):
            build_g1_feature_manifest(["embedding_00", "TransactionAmt"], embedding_feature_names())

    def test_expected_feature_count_is_435_plus_embedding_dim(self) -> None:
        self.assertEqual(EXPECTED_G1_FEATURE_COUNT, 435 + EMBEDDING_DIM)


class EmbeddingGainRankTests(unittest.TestCase):
    def test_ranks_are_computed_for_every_embedding_column(self) -> None:
        feat_names = ["embedding_00", "embedding_01"]
        feature_importance = pd.DataFrame(
            {
                "feature": ["a", "embedding_00", "b", "embedding_01"],
                "importance_gain": [100.0, 50.0, 10.0, 5.0],
                "importance_split": [40, 20, 8, 2],
            }
        )

        ranks = embedding_gain_ranks(feature_importance, feat_names)

        self.assertEqual(ranks["total_features_in_model"], 4)
        self.assertEqual(ranks["embedding_gain_ranks"], [2, 4])
        self.assertEqual(ranks["best_embedding_gain_rank"], 2)
        self.assertEqual(ranks["worst_embedding_gain_rank"], 4)

    def test_missing_embedding_column_raises(self) -> None:
        feature_importance = pd.DataFrame(
            {"feature": ["a"], "importance_gain": [1.0], "importance_split": [1]}
        )
        with self.assertRaises(AssertionError):
            embedding_gain_ranks(feature_importance, ["embedding_00"])


class ComparisonTableTests(unittest.TestCase):
    def test_columns_are_relabeled_for_the_given_reference(self) -> None:
        labels = pd.Series([0, 1, 0, 1], dtype="int8")
        reference_metrics = evaluate_validation(labels, np.array([0.1, 0.6, 0.2, 0.7]))
        candidate_metrics = evaluate_validation(labels, np.array([0.1, 0.9, 0.2, 0.8]))

        table = build_comparison_table(reference_metrics, candidate_metrics, "b0", "g1")

        self.assertIn("b0_value", table.columns)
        self.assertIn("g1_value", table.columns)
        self.assertIn("delta_g1_minus_b0", table.columns)
        np.testing.assert_allclose(
            table["delta_g1_minus_b0"], table["g1_value"] - table["b0_value"]
        )


class FrozenReferenceIntegrationTests(unittest.TestCase):
    """These exercise the real, already-frozen repository artifacts."""

    def test_b0_metadata_loads_and_has_435_predictors(self) -> None:
        metadata = load_frozen_b0_metadata()
        self.assertEqual(len(metadata["feature_columns"]), 435)

    def test_b1_card1_metadata_loads_and_has_439_predictors(self) -> None:
        metadata = load_frozen_b1_card1_metadata()
        self.assertEqual(metadata["b1_feature_count"], 439)
        self.assertFalse(metadata["test_evaluated"])

    def test_embedding_metadata_declares_no_test_label_usage(self) -> None:
        metadata = load_embedding_metadata("card1")
        self.assertFalse(metadata["test_labels_used"])
        self.assertEqual(metadata["architecture"]["embedding_dim"], EMBEDDING_DIM)

    def test_g1_never_evaluates_test(self) -> None:
        self.assertFalse(TEST_EVALUATED)


if __name__ == "__main__":
    unittest.main()
