"""Unit tests for the shared paired-bootstrap significance module.

The bootstrap and its supporting helpers used to live inside train_lightgbm_g1.py,
reachable only by importing a training entrypoint. These tests exercise the
module in isolation: no trainer, no graph encoder, no LightGBM fit.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models import significance
from src.models.significance import (
    compare_variants,
    load_validation_predictions,
    paired_bootstrap_pr_auc_delta,
)


class PairedBootstrapTests(unittest.TestCase):
    def test_identical_scores_give_a_zero_delta_confidence_interval(self) -> None:
        rng = np.random.default_rng(1)
        n = 500
        y_true = (rng.random(n) < 0.1).astype(np.int8)
        scores = rng.random(n)

        result = paired_bootstrap_pr_auc_delta(y_true, scores, scores, n_resamples=200, seed=0)

        self.assertEqual(result["observed_delta"], 0.0)
        self.assertAlmostEqual(result["ci_lower_95"], 0.0, places=6)
        self.assertAlmostEqual(result["ci_upper_95"], 0.0, places=6)
        self.assertFalse(result["excludes_zero"])

    def test_a_clearly_better_candidate_excludes_zero(self) -> None:
        rng = np.random.default_rng(2)
        n = 2_000
        y_true = (rng.random(n) < 0.1).astype(np.int8)
        # Candidate scores are strongly correlated with the label; reference is pure noise.
        candidate_scores = y_true.astype(np.float64) + rng.random(n) * 0.1
        reference_scores = rng.random(n)

        result = paired_bootstrap_pr_auc_delta(
            y_true, candidate_scores, reference_scores, n_resamples=200, seed=0
        )

        self.assertGreater(result["observed_delta"], 0.0)
        self.assertTrue(result["excludes_zero"])
        self.assertGreater(result["ci_lower_95"], 0.0)

    def test_mismatched_lengths_raise(self) -> None:
        y_true = np.array([0, 1, 0], dtype=np.int8)
        with self.assertRaises(ValueError):
            paired_bootstrap_pr_auc_delta(y_true, np.array([0.1, 0.2]), np.array([0.1, 0.2, 0.3]))

    def test_zero_resamples_raise(self) -> None:
        y_true = np.array([0, 1, 0], dtype=np.int8)
        scores = np.array([0.1, 0.2, 0.3])
        with self.assertRaises(ValueError):
            paired_bootstrap_pr_auc_delta(y_true, scores, scores, n_resamples=0)

    def test_same_seed_is_deterministic(self) -> None:
        rng = np.random.default_rng(3)
        n = 300
        y_true = (rng.random(n) < 0.2).astype(np.int8)
        candidate_scores = rng.random(n)
        reference_scores = rng.random(n)

        first = paired_bootstrap_pr_auc_delta(
            y_true, candidate_scores, reference_scores, n_resamples=150, seed=7
        )
        second = paired_bootstrap_pr_auc_delta(
            y_true, candidate_scores, reference_scores, n_resamples=150, seed=7
        )

        self.assertEqual(first["bootstrap_mean_delta"], second["bootstrap_mean_delta"])
        self.assertEqual(first["ci_lower_95"], second["ci_lower_95"])
        self.assertEqual(first["ci_upper_95"], second["ci_upper_95"])


def _write_predictions(
    path: Path, transaction_ids: list[int], labels: list[int], scores: list[float]
) -> None:
    df = pd.DataFrame(
        {"TransactionID": transaction_ids, "isFraud": labels, "prediction": scores}
    )
    df.to_parquet(path, index=False)


class LoadValidationPredictionsTests(unittest.TestCase):
    def test_row_count_mismatch_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.parquet"
            _write_predictions(path, [1, 2, 3], [0, 1, 0], [0.1, 0.9, 0.2])
            with patch.dict(significance.EXPECTED_SPLIT_COUNTS, {"validation": 4}):
                with self.assertRaises(ValueError):
                    load_validation_predictions(path, "prediction_column")

    def test_prediction_column_is_renamed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "predictions.parquet"
            _write_predictions(path, [1, 2, 3], [0, 1, 0], [0.1, 0.9, 0.2])
            with patch.dict(significance.EXPECTED_SPLIT_COUNTS, {"validation": 3}):
                df = load_validation_predictions(path, "candidate_prediction")
        self.assertIn("candidate_prediction", df.columns)
        self.assertNotIn("prediction", df.columns)


class CompareVariantsTests(unittest.TestCase):
    def test_generalises_to_an_arbitrary_pair_of_variants(self) -> None:
        rng = np.random.default_rng(4)
        n = 400
        transaction_ids = list(range(n))
        labels = (rng.random(n) < 0.15).astype(int).tolist()
        candidate_scores = (np.array(labels) + rng.random(n) * 0.2).tolist()
        reference_scores = rng.random(n).tolist()

        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = Path(tmp) / "candidate.parquet"
            reference_path = Path(tmp) / "reference.parquet"
            _write_predictions(candidate_path, transaction_ids, labels, candidate_scores)
            _write_predictions(reference_path, transaction_ids, labels, reference_scores)

            with patch.dict(significance.EXPECTED_SPLIT_COUNTS, {"validation": n}):
                result = compare_variants(
                    "variant_x", candidate_path, "variant_y", reference_path, n_resamples=200
                )

        self.assertEqual(result["candidate"], "variant_x")
        self.assertEqual(result["reference"], "variant_y")
        self.assertEqual(result["validation_row_count"], n)
        self.assertGreater(result["observed_delta"], 0.0)
        self.assertTrue(result["excludes_zero"])

    def test_mismatched_labels_raise(self) -> None:
        n = 50
        transaction_ids = list(range(n))
        with tempfile.TemporaryDirectory() as tmp:
            candidate_path = Path(tmp) / "candidate.parquet"
            reference_path = Path(tmp) / "reference.parquet"
            labels_a = [0] * n
            labels_b = [0] * (n - 1) + [1]
            scores = [0.5] * n
            _write_predictions(candidate_path, transaction_ids, labels_a, scores)
            _write_predictions(reference_path, transaction_ids, labels_b, scores)

            with patch.dict(significance.EXPECTED_SPLIT_COUNTS, {"validation": n}):
                with self.assertRaises(AssertionError):
                    compare_variants("a", candidate_path, "b", reference_path, n_resamples=50)


if __name__ == "__main__":
    unittest.main()
