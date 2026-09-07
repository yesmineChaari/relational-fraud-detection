"""Paired bootstrap significance testing over persisted validation predictions.

Originally defined inside train_lightgbm_g1.py, where it was reachable only by
importing a training entrypoint (and, transitively, the graph encoder and every
other G1 training dependency). This module has no trainer dependency: it reads
already-written validation_predictions.parquet files and compares two score
vectors over the same row index, so any stage comparison can call it directly.

Every function here is pure and side-effect free; callers own writing whatever
report artifact the comparison produces.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from src.models.train_lightgbm_baseline import EXPECTED_SPLIT_COUNTS, RANDOM_SEED

DEFAULT_N_RESAMPLES = 10_000


def _fast_average_precision(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Binary average precision, numerically identical to sklearn's but without
    sklearn's generic multiclass dispatch overhead -- needed because the paired
    bootstrap below calls this thousands of times over 88,581-row resamples."""
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    tp_cumsum = np.cumsum(y_sorted)
    total_positives = tp_cumsum[-1]
    ranks = np.arange(1, len(y_true) + 1)
    precision_at_rank = tp_cumsum / ranks
    return float(np.sum(precision_at_rank * y_sorted) / total_positives)


def paired_bootstrap_pr_auc_delta(
    y_true: np.ndarray,
    candidate_scores: np.ndarray,
    reference_scores: np.ndarray,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """95% CI on PR-AUC(candidate) - PR-AUC(reference) over paired bootstrap resamples.

    Resamples row indices with replacement (same indices applied to both score
    vectors, so the comparison is paired rather than two independent bootstraps).
    Matches the project's documented statistical-validation methodology: a
    paired bootstrap over the validation set, reusing already-trained predictions.
    """
    n = len(y_true)
    if len(candidate_scores) != n or len(reference_scores) != n:
        raise ValueError("y_true, candidate_scores and reference_scores must have equal length.")
    if n_resamples <= 0:
        raise ValueError("n_resamples must be positive.")

    observed_delta = float(
        average_precision_score(y_true, candidate_scores)
        - average_precision_score(y_true, reference_scores)
    )

    y_true_float = y_true.astype(np.float64, copy=False)
    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_labels = y_true_float[idx]
        while resampled_labels.sum() == 0 or resampled_labels.sum() == n:
            idx = rng.integers(0, n, size=n)
            resampled_labels = y_true_float[idx]
        candidate_pr_auc = _fast_average_precision(resampled_labels, candidate_scores[idx])
        reference_pr_auc = _fast_average_precision(resampled_labels, reference_scores[idx])
        deltas[i] = candidate_pr_auc - reference_pr_auc

    ci_lower, ci_upper = (float(v) for v in np.percentile(deltas, [2.5, 97.5]))
    return {
        "metric": "pr_auc",
        "n_resamples": int(n_resamples),
        "observed_delta": observed_delta,
        "bootstrap_mean_delta": float(deltas.mean()),
        "bootstrap_std_delta": float(deltas.std()),
        "ci_lower_95": ci_lower,
        "ci_upper_95": ci_upper,
        "excludes_zero": bool(ci_lower > 0.0 or ci_upper < 0.0),
        "random_seed": int(seed),
    }


def load_validation_predictions(path: Path, prediction_column: str) -> pd.DataFrame:
    """Read a persisted validation_predictions.parquet, renaming its prediction
    column so several variants can be merged on TransactionID without collision."""
    df = pd.read_parquet(path, columns=["TransactionID", "isFraud", "prediction"])
    if len(df) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise ValueError(f"Expected {EXPECTED_SPLIT_COUNTS['validation']:,} validation rows in {path}.")
    return df.rename(columns={"prediction": prediction_column})


def compare_variants(
    candidate_label: str,
    candidate_predictions_path: Path,
    reference_label: str,
    reference_predictions_path: Path,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Paired-bootstrap PR-AUC comparison between any two persisted
    validation_predictions.parquet files sharing the same TransactionID index.

    Generic over the variant pair, so a new stage comparison needs no new
    bootstrap code -- only the two prediction-file paths and their labels.
    """
    candidate_df = load_validation_predictions(candidate_predictions_path, "candidate_prediction")
    reference_df = load_validation_predictions(reference_predictions_path, "reference_prediction")

    merged = candidate_df.merge(
        reference_df[["TransactionID", "isFraud", "reference_prediction"]].rename(
            columns={"isFraud": "isFraud_reference"}
        ),
        on="TransactionID",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise AssertionError(
            f"Significance merge lost or duplicated validation rows for "
            f"{candidate_label} vs {reference_label}."
        )
    if not merged["isFraud"].equals(merged["isFraud_reference"]):
        raise AssertionError(
            f"isFraud labels disagree between {candidate_label} and {reference_label} "
            "validation predictions."
        )

    y_true = merged["isFraud"].to_numpy(dtype=np.int8)
    candidate_scores = merged["candidate_prediction"].to_numpy(dtype=np.float64)
    reference_scores = merged["reference_prediction"].to_numpy(dtype=np.float64)

    result = paired_bootstrap_pr_auc_delta(
        y_true, candidate_scores, reference_scores, n_resamples=n_resamples, seed=seed
    )
    result["candidate"] = candidate_label
    result["reference"] = reference_label
    result["validation_row_count"] = int(len(merged))
    return result
