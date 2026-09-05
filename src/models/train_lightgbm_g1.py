"""G1 LightGBM trainer: frozen B0 predictors plus GraphSAGE per-transaction embeddings.

Same controlled-experiment discipline as B1 (src/models/train_lightgbm_relational.py):
the classifier configuration is frozen and re-validated against B0 before every run,
so a measured change is attributable to the representation, not to a retuned model.
Where B1 substituted four scalar relational summaries for the frozen predictors, G1
substitutes the 32-dimensional GraphSAGE embedding (src/graph/train_graphsage_encoder.py)
computed on the same card1 relation. Validation-only: test_evaluated stays false.

Both frozen references this experiment is measured against -- B0 and B1-card1 -- are
hash-pinned before and after the run with the same protection mechanism B1 uses for B0,
so no run of this script can silently drift the baselines its own comparison depends on.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from pandas.api.types import is_float_dtype, is_numeric_dtype
from sklearn.metrics import average_precision_score

from src.graph.train_graphsage_encoder import (
    EMBEDDING_DIM,
    EMBEDDINGS_PATH as GRAPH_EMBEDDINGS_PATH,
    METADATA_PATH as GRAPH_METADATA_PATH,
)
from src.models.train_lightgbm_baseline import (
    CATEGORY_MAPPINGS_PATH,
    EXPECTED_ROWS,
    EXPECTED_SPLIT_COUNTS,
    MODEL_DATASET_PATH,
    RANDOM_SEED,
    assert_supported_model_dtypes,
    build_feature_importance,
    build_learning_curve,
    build_lightgbm_model,
    calculate_scale_pos_weight,
    evaluate_validation,
    identify_categorical_columns,
    load_model_dataset,
    repository_relative,
    summarize_learning_curve,
    validate_split_counts,
    write_json,
)
from src.models.train_lightgbm_relational import (
    B0_PROTECTED_PATHS,
    EARLY_STOPPING_ROUNDS,
    EVALUATION_SPLIT,
    MAX_ESTIMATORS,
    TEST_EVALUATED,
    apply_frozen_category_mappings,
    assert_protected_artifacts_unchanged,
    attach_relational_features,
    build_comparison_to_b0,
    load_frozen_b0_metadata,
    load_frozen_category_mappings,
    read_json,
    snapshot_protected_artifacts,
    validate_frozen_lightgbm_configuration,
    validate_model_columns_against_frozen_b0,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

RELATION = "card1"
EXPECTED_B0_FEATURE_COUNT = 435
EXPECTED_G1_FEATURE_COUNT = EXPECTED_B0_FEATURE_COUNT + EMBEDDING_DIM
SIGNIFICANCE_RESAMPLES = 10_000

B1_CARD1_MODEL_PATH = ROOT_DIR / "models" / "lightgbm_b1_card1.txt"
B1_CARD1_REPORT_DIR = ROOT_DIR / "reports" / "b1" / "card1"
B1_CARD1_METRICS_PATH = B1_CARD1_REPORT_DIR / "metrics.json"
B1_CARD1_METADATA_PATH = B1_CARD1_REPORT_DIR / "metadata.json"
B1_CARD1_FEATURE_IMPORTANCE_PATH = B1_CARD1_REPORT_DIR / "feature_importance.csv"
B1_CARD1_VALIDATION_PREDICTIONS_PATH = B1_CARD1_REPORT_DIR / "validation_predictions.parquet"

# The B1-card1 reference this experiment reports against is frozen the same way B0 is.
B1_CARD1_PROTECTED_PATHS = [
    B1_CARD1_MODEL_PATH,
    B1_CARD1_METRICS_PATH,
    B1_CARD1_METADATA_PATH,
    B1_CARD1_FEATURE_IMPORTANCE_PATH,
]


def _resolve_g1_paths(relation: str) -> dict[str, Path]:
    model_path = ROOT_DIR / "models" / f"lightgbm_g1_{relation}.txt"
    report_dir = ROOT_DIR / "reports" / "g1" / relation
    return {
        "model": model_path,
        "report_dir": report_dir,
        "metrics": report_dir / "metrics.json",
        "metadata": report_dir / "metadata.json",
        "feature_importance": report_dir / "feature_importance.csv",
        "validation_predictions": report_dir / "validation_predictions.parquet",
        "learning_curve": report_dir / "learning_curve.csv",
        "comparison_to_b0": report_dir / "comparison_to_b0.csv",
        "comparison_to_b1_card1": report_dir / "comparison_to_b1_card1.csv",
        "significance": report_dir / "significance.json",
    }


def embedding_feature_names(embedding_dim: int = EMBEDDING_DIM) -> list[str]:
    return [f"embedding_{i:02d}" for i in range(embedding_dim)]


def load_frozen_b1_card1_metadata(path: Path = B1_CARD1_METADATA_PATH) -> dict[str, Any]:
    metadata = read_json(path)
    if metadata.get("relation_name") != "card1":
        raise ValueError("B1-card1 metadata has the wrong relation name.")
    if metadata.get("test_evaluated") is not False:
        raise ValueError("Frozen B1-card1 metadata violates final-test discipline.")
    if metadata.get("b1_feature_count") != EXPECTED_B0_FEATURE_COUNT + 4:
        raise ValueError("Frozen B1-card1 predictor count is not the expected 439.")
    return metadata


def load_embedding_metadata(
    relation: str,
    embedding_dim: int = EMBEDDING_DIM,
    path: Path = GRAPH_METADATA_PATH,
) -> dict[str, Any]:
    metadata = read_json(path)
    if metadata.get("relation_name") != relation:
        raise ValueError(f"GraphSAGE embedding metadata has the wrong relation (expected {relation!r}).")
    if metadata.get("architecture", {}).get("embedding_dim") != embedding_dim:
        raise ValueError("GraphSAGE embedding dimension does not match the encoder's declared width.")
    if metadata.get("test_labels_used") is not False:
        raise ValueError("GraphSAGE embeddings were trained using test labels.")
    if metadata.get("target_labels_used_in_node_features") is not False:
        raise ValueError("GraphSAGE node features leaked the fraud label.")
    if metadata.get("embedding_has_no_target_column") is not True:
        raise ValueError("GraphSAGE embedding artifact metadata does not attest to being label-free.")
    if metadata.get("split_row_counts") != EXPECTED_SPLIT_COUNTS:
        raise ValueError("GraphSAGE embedding split counts do not match the frozen split manifest.")
    return metadata


def load_embeddings(
    relation: str,
    feat_names: list[str],
    path: Path = GRAPH_EMBEDDINGS_PATH,
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"GraphSAGE embeddings not found for relation {relation!r}. "
            f"Run: python -m src.graph.train_graphsage_encoder\nExpected: {path}"
        )
    embeddings_df = pd.read_parquet(path)
    expected_cols = ["TransactionID", "split", *feat_names]
    if list(embeddings_df.columns) != expected_cols:
        raise ValueError(
            f"Embedding columns must be exactly {expected_cols}; got {list(embeddings_df.columns)}."
        )
    if len(embeddings_df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} embedding rows; got {len(embeddings_df):,}.")
    if "isFraud" in embeddings_df.columns:
        raise AssertionError("isFraud must never appear in the embedding artifact.")
    return embeddings_df


def validate_embedding_feature_dtypes(embeddings_df: pd.DataFrame, feat_names: list[str]) -> None:
    for column in feat_names:
        if not is_float_dtype(embeddings_df[column].dtype):
            raise TypeError(f"{column} must use a floating dtype.")
        values = embeddings_df[column].to_numpy()
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains non-finite embedding values.")


def validate_embedding_merge(
    model_index: pd.DataFrame,
    embeddings_df: pd.DataFrame,
    feat_names: list[str],
) -> pd.DataFrame:
    required_index_columns = {"TransactionID", "split"}
    missing = required_index_columns - set(model_index.columns)
    if missing:
        raise ValueError(f"Model index is missing columns: {sorted(missing)}.")
    if model_index["TransactionID"].isna().any():
        raise ValueError("Model index contains missing TransactionID values.")
    if embeddings_df["TransactionID"].isna().any():
        raise ValueError("Embeddings contain missing TransactionID values.")
    if not model_index["TransactionID"].is_unique:
        raise ValueError("Model index contains duplicate TransactionID values.")
    if not embeddings_df["TransactionID"].is_unique:
        raise ValueError("Embeddings contain duplicate TransactionID values.")
    validate_embedding_feature_dtypes(embeddings_df, feat_names)

    membership = model_index[["TransactionID"]].merge(
        embeddings_df[["TransactionID"]],
        on="TransactionID",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    missing_emb = int(membership["_merge"].eq("left_only").sum())
    extra_emb = int(membership["_merge"].eq("right_only").sum())
    if missing_emb or extra_emb:
        raise ValueError(
            f"Embedding/model TransactionID membership mismatch: "
            f"missing embedding IDs={missing_emb}, extra embedding IDs={extra_emb}."
        )

    split_check = model_index[["TransactionID", "split"]].merge(
        embeddings_df[["TransactionID", "split"]],
        on="TransactionID",
        suffixes=("_model", "_embedding"),
        validate="one_to_one",
    )
    if not split_check["split_model"].astype("string").equals(
        split_check["split_embedding"].astype("string")
    ):
        raise AssertionError("Embedding split assignment disagrees with the model index.")

    merged = model_index[["TransactionID", "split"]].merge(
        embeddings_df[["TransactionID", *feat_names]],
        on="TransactionID",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if len(merged) != len(model_index):
        raise AssertionError("Embedding merge changed the model-index row count.")
    if not np.array_equal(
        merged["TransactionID"].to_numpy(),
        model_index["TransactionID"].to_numpy(),
    ):
        raise AssertionError("Embedding merge changed TransactionID order.")
    return merged


def load_g1_datasets(
    relation: str = RELATION,
    embeddings_path: Path = GRAPH_EMBEDDINGS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feat_names = embedding_feature_names()
    train_df, validation_df, dataset_summary = load_model_dataset()
    embeddings_df = load_embeddings(relation, feat_names, path=embeddings_path)
    model_index = pd.read_parquet(MODEL_DATASET_PATH, columns=["TransactionID", "split"])
    validate_split_counts(model_index, "model_dataset.parquet G1 index")
    merged_index = validate_embedding_merge(model_index, embeddings_df, feat_names)

    train_with_embeddings = attach_relational_features(train_df, merged_index, "train", feat_names)
    validation_with_embeddings = attach_relational_features(
        validation_df, merged_index, "validation", feat_names
    )
    if len(train_with_embeddings) != EXPECTED_SPLIT_COUNTS["train"]:
        raise AssertionError("G1 train row count changed.")
    if len(validation_with_embeddings) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise AssertionError("G1 validation row count changed.")
    return train_with_embeddings, validation_with_embeddings, dataset_summary


def build_g1_feature_manifest(b0_features: list[str], feat_names: list[str]) -> list[str]:
    if not b0_features or len(b0_features) != len(set(b0_features)):
        raise ValueError("B0 feature manifest is empty or contains duplicates.")
    overlap = sorted(set(b0_features) & set(feat_names))
    if overlap:
        raise ValueError(f"B0 feature manifest already contains embedding features: {overlap}.")
    g1_features = [*b0_features, *feat_names]
    if len(g1_features) != len(b0_features) + len(feat_names):
        raise AssertionError("G1 must contain exactly one additional predictor per embedding dim.")
    return g1_features


def embedding_gain_ranks(feature_importance: pd.DataFrame, feat_names: list[str]) -> dict[str, Any]:
    ranked = feature_importance.copy()
    ranked["rank_gain"] = ranked["importance_gain"].rank(ascending=False, method="min").astype(int)
    ranked["rank_split"] = ranked["importance_split"].rank(ascending=False, method="min").astype(int)
    embedding_rows = ranked[ranked["feature"].isin(feat_names)].set_index("feature")
    missing = set(feat_names) - set(embedding_rows.index)
    if missing:
        raise AssertionError(f"Embedding features missing from feature importance: {sorted(missing)}.")
    gain_ranks = embedding_rows.loc[feat_names, "rank_gain"].tolist()
    split_ranks = embedding_rows.loc[feat_names, "rank_split"].tolist()
    return {
        "total_features_in_model": int(len(ranked)),
        "embedding_gain_ranks": [int(r) for r in gain_ranks],
        "embedding_split_ranks": [int(r) for r in split_ranks],
        "best_embedding_gain_rank": int(min(gain_ranks)),
        "worst_embedding_gain_rank": int(max(gain_ranks)),
        "median_embedding_gain_rank": float(np.median(gain_ranks)),
    }


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
    n_resamples: int = SIGNIFICANCE_RESAMPLES,
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
    df = pd.read_parquet(path, columns=["TransactionID", "isFraud", "prediction"])
    if len(df) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise ValueError(f"Expected {EXPECTED_SPLIT_COUNTS['validation']:,} validation rows in {path}.")
    return df.rename(columns={"prediction": prediction_column})


def compute_significance(
    g1_validation_predictions: pd.DataFrame,
) -> dict[str, Any]:
    b0_predictions = load_validation_predictions(
        ROOT_DIR / "reports" / "baseline" / "validation_predictions.parquet", "b0_prediction"
    )
    b1_predictions = load_validation_predictions(
        B1_CARD1_VALIDATION_PREDICTIONS_PATH, "b1_card1_prediction"
    )
    g1_predictions = g1_validation_predictions[["TransactionID", "isFraud", "prediction"]].rename(
        columns={"prediction": "g1_prediction"}
    )

    merged = g1_predictions.merge(
        b0_predictions[["TransactionID", "isFraud", "b0_prediction"]].rename(
            columns={"isFraud": "isFraud_b0"}
        ),
        on="TransactionID",
        validate="one_to_one",
    ).merge(
        b1_predictions[["TransactionID", "isFraud", "b1_card1_prediction"]].rename(
            columns={"isFraud": "isFraud_b1"}
        ),
        on="TransactionID",
        validate="one_to_one",
    )
    if len(merged) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise AssertionError("Significance merge lost or duplicated validation rows.")
    if not merged["isFraud"].equals(merged["isFraud_b0"]) or not merged["isFraud"].equals(
        merged["isFraud_b1"]
    ):
        raise AssertionError("isFraud labels disagree across G1/B0/B1-card1 validation predictions.")

    y_true = merged["isFraud"].to_numpy(dtype=np.int8)
    g1_scores = merged["g1_prediction"].to_numpy(dtype=np.float64)
    b0_scores = merged["b0_prediction"].to_numpy(dtype=np.float64)
    b1_scores = merged["b1_card1_prediction"].to_numpy(dtype=np.float64)

    print(f"Running paired bootstrap (G1 vs B0, {SIGNIFICANCE_RESAMPLES:,} resamples)...")
    g1_vs_b0 = paired_bootstrap_pr_auc_delta(y_true, g1_scores, b0_scores)
    print(f"Running paired bootstrap (G1 vs B1-card1, {SIGNIFICANCE_RESAMPLES:,} resamples)...")
    g1_vs_b1_card1 = paired_bootstrap_pr_auc_delta(y_true, g1_scores, b1_scores)

    return {
        "method": (
            "paired_bootstrap_pr_auc_delta: resample validation-row indices with "
            "replacement (identical indices applied to both score vectors per "
            "resample), n_resamples draws, 95% CI from the empirical percentiles "
            "of the resampled PR-AUC delta"
        ),
        "validation_row_count": int(len(merged)),
        "g1_vs_b0": g1_vs_b0,
        "g1_vs_b1_card1": g1_vs_b1_card1,
    }


def build_comparison_table(
    reference_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    reference_label: str,
    candidate_label: str,
) -> pd.DataFrame:
    raw = build_comparison_to_b0(reference_metrics, candidate_metrics)
    return raw.rename(
        columns={
            "b0_value": f"{reference_label}_value",
            "b1_value": f"{candidate_label}_value",
            "delta_b1_minus_b0": f"delta_{candidate_label}_minus_{reference_label}",
        }
    )


def build_g1_metadata(
    *,
    relation: str,
    model,
    b0_metadata: dict[str, Any],
    b1_card1_metadata: dict[str, Any],
    embedding_metadata: dict[str, Any],
    g1_features: list[str],
    categorical_columns: list[str],
    category_mappings_sha256: str,
    scale_pos_weight: float,
    train_row_count: int,
    validation_row_count: int,
    learning_curve_summary: dict[str, Any],
    validation_metrics: dict[str, Any],
    b0_artifact_hashes: dict[str, str],
    embedding_ranks: dict[str, Any],
    significance: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    feat_names = embedding_feature_names()
    numeric_columns = [col for col in g1_features if col not in categorical_columns]
    return {
        "experiment_name": f"G1_graphsage_lightgbm_{relation}",
        "model_family": "LightGBM",
        "controlled_experiment_definition": (
            "G1 = frozen B0 feature set + GraphSAGE per-transaction embeddings, "
            "in place of the four B1 relational scalar features"
        ),
        "relation_name": relation,
        "embedding_source_path": repository_relative(GRAPH_EMBEDDINGS_PATH),
        "embedding_metadata_path": repository_relative(GRAPH_METADATA_PATH),
        "embedding_dim": EMBEDDING_DIM,
        "embedding_feature_names": feat_names,
        "embedding_encoder_architecture": embedding_metadata["architecture"],
        "b0_reference_metrics_path": repository_relative(
            ROOT_DIR / "reports" / "baseline" / "lightgbm_metrics.json"
        ),
        "b0_reference_metadata_path": repository_relative(
            ROOT_DIR / "reports" / "baseline" / "baseline_metadata.json"
        ),
        "b1_card1_reference_metrics_path": repository_relative(B1_CARD1_METRICS_PATH),
        "b1_card1_reference_metadata_path": repository_relative(B1_CARD1_METADATA_PATH),
        "frozen_reference_artifact_sha256": b0_artifact_hashes,
        "b0_feature_count": len(b0_metadata["feature_columns"]),
        "b1_card1_feature_count": b1_card1_metadata["b1_feature_count"],
        "g1_feature_count": len(g1_features),
        "feature_columns": g1_features,
        "categorical_feature_count": len(categorical_columns),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_columns": categorical_columns,
        "numeric_feature_columns": numeric_columns,
        "categorical_mapping_reference": repository_relative(CATEGORY_MAPPINGS_PATH),
        "categorical_mappings_sha256": category_mappings_sha256,
        "categorical_preprocessing_policy": (
            "Reuse the frozen B0 train-only mappings unchanged; missing and unknown "
            "category behavior is identical to B0."
        ),
        "class_weighting_policy": (
            "scale_pos_weight = negative_train_count / positive_train_count"
        ),
        "scale_pos_weight": float(scale_pos_weight),
        "lightgbm_parameters": model.get_params(deep=False),
        "max_estimators": MAX_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "early_stopping_metric": "average_precision",
        "best_iteration": int(model.best_iteration_),
        "actual_stopping_iteration": int(learning_curve_summary["actual_stopping_iteration"]),
        "early_stopping_triggered": bool(learning_curve_summary["early_stopping_triggered"]),
        "train_row_count": int(train_row_count),
        "validation_row_count": int(validation_row_count),
        "evaluation_split": EVALUATION_SPLIT,
        "validation_pr_auc": float(validation_metrics["pr_auc"]),
        "validation_roc_auc": float(validation_metrics["roc_auc"]),
        "validation_ranking_metrics": validation_metrics["ranking_metrics"],
        "test_evaluated": TEST_EVALUATED,
        "embedding_gain_importance": embedding_ranks,
        "statistical_significance": significance,
        "statistical_significance_path": repository_relative(paths["significance"]),
        "model_path": repository_relative(paths["model"]),
        "metrics_path": repository_relative(paths["metrics"]),
        "feature_importance_path": repository_relative(paths["feature_importance"]),
        "validation_predictions_path": repository_relative(paths["validation_predictions"]),
        "learning_curve_path": repository_relative(paths["learning_curve"]),
        "comparison_to_b0_path": repository_relative(paths["comparison_to_b0"]),
        "comparison_to_b1_card1_path": repository_relative(paths["comparison_to_b1_card1"]),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "random_seed": RANDOM_SEED,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_g1(relation: str = RELATION) -> None:
    feat_names = embedding_feature_names()
    paths = _resolve_g1_paths(relation)

    all_protected = [*B0_PROTECTED_PATHS, *B1_CARD1_PROTECTED_PATHS]
    reference_hashes_before = snapshot_protected_artifacts(all_protected)

    b0_metadata = load_frozen_b0_metadata()
    b1_card1_metadata = load_frozen_b1_card1_metadata()
    b1_card1_metrics = read_json(B1_CARD1_METRICS_PATH)
    embedding_metadata = load_embedding_metadata(relation)

    print(f"[G1-{relation}] Loading frozen B0 train/validation data: {MODEL_DATASET_PATH}")
    print(f"[G1-{relation}] Loading GraphSAGE embeddings: {GRAPH_EMBEDDINGS_PATH}")
    train_df, validation_df, _ = load_g1_datasets(relation)

    b0_features = list(b0_metadata["feature_columns"])
    validate_model_columns_against_frozen_b0(list(train_df.columns), b0_features, feat_names)
    validate_model_columns_against_frozen_b0(list(validation_df.columns), b0_features, feat_names)
    g1_features = build_g1_feature_manifest(b0_features, feat_names)
    if len(g1_features) != EXPECTED_G1_FEATURE_COUNT:
        raise ValueError(
            f"G1 predictor-count sanity check failed: "
            f"expected {EXPECTED_G1_FEATURE_COUNT}, got {len(g1_features)}."
        )

    categorical_columns = list(b0_metadata["categorical_feature_columns"])
    raw_train_cat = identify_categorical_columns(train_df[g1_features])
    raw_val_cat = identify_categorical_columns(validation_df[g1_features])
    if raw_train_cat != categorical_columns:
        raise TypeError("G1 categorical features differ from frozen B0.")
    if raw_val_cat != categorical_columns:
        raise TypeError("G1 validation categorical features differ from frozen B0.")
    for col in feat_names:
        if not is_numeric_dtype(train_df[col].dtype):
            raise TypeError(f"G1 embedding predictor {col} is not numeric.")

    validation_metadata = validation_df[["TransactionID", "TransactionDT", "isFraud"]].copy()
    y_train = train_df["isFraud"].astype("int8").copy()
    y_validation = validation_df["isFraud"].astype("int8").copy()
    X_train = train_df[g1_features].copy()
    X_validation = validation_df[g1_features].copy()
    del train_df, validation_df

    mappings, mapping_sha256 = load_frozen_category_mappings(categorical_columns)
    print(f"[G1-{relation}] Applying frozen B0 categorical mappings (no fitting)...")
    apply_frozen_category_mappings(X_train, X_validation, categorical_columns, mappings)
    assert_supported_model_dtypes(X_train, "G1 training predictors")
    assert_supported_model_dtypes(X_validation, "G1 validation predictors")

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    if not np.isclose(scale_pos_weight, float(b0_metadata["scale_pos_weight"]), rtol=0.0, atol=0.0):
        raise ValueError("G1 train-only class weight differs from frozen B0.")
    model = build_lightgbm_model(scale_pos_weight, n_estimators=MAX_ESTIMATORS)
    validate_frozen_lightgbm_configuration(model, b0_metadata)

    print(f"[G1-{relation}] Train rows: {len(X_train):,}")
    print(f"[G1-{relation}] Validation rows: {len(X_validation):,}")
    print(f"[G1-{relation}] B0 predictors: {len(b0_features):,}")
    print(f"[G1-{relation}] G1 predictors: {len(g1_features):,}")
    print(f"[G1-{relation}] Embedding predictors: {len(feat_names):,}")
    print(f"[G1-{relation}] scale_pos_weight: {scale_pos_weight:.15f}")

    evaluation_results: dict[str, dict[str, list[float]]] = {}
    callbacks = [
        lightgbm.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, first_metric_only=True),
        lightgbm.record_evaluation(evaluation_results),
        lightgbm.log_evaluation(period=50),
    ]
    print(f"[G1-{relation}] Training G1 LightGBM...")
    model.fit(
        X_train,
        y_train,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_names=[EVALUATION_SPLIT],
        eval_metric=["average_precision", "auc"],
        categorical_feature=categorical_columns,
        callbacks=callbacks,
    )
    if not model.best_iteration_ or model.best_iteration_ <= 0:
        raise RuntimeError("LightGBM did not report a valid best iteration.")

    learning_curve = build_learning_curve(evaluation_results)
    learning_curve_summary = summarize_learning_curve(
        learning_curve,
        best_iteration=int(model.best_iteration_),
        maximum_estimators=MAX_ESTIMATORS,
    )
    validation_scores = model.predict_proba(
        X_validation, num_iteration=model.best_iteration_
    )[:, 1]
    if len(validation_scores) != EXPECTED_SPLIT_COUNTS[EVALUATION_SPLIT]:
        raise AssertionError("G1 validation prediction count is incorrect.")
    if not np.isfinite(validation_scores).all():
        raise AssertionError("G1 validation predictions contain non-finite values.")

    metrics = evaluate_validation(y_validation, validation_scores)
    metrics.update(
        {
            "model": f"G1_graphsage_lightgbm_{relation}",
            "relation": relation,
            "weighting": "weighted",
            "scale_pos_weight": float(scale_pos_weight),
            "maximum_estimators": MAX_ESTIMATORS,
            "actual_stopping_iteration": int(learning_curve_summary["actual_stopping_iteration"]),
            "best_iteration": int(model.best_iteration_),
            "best_validation_average_precision": float(
                learning_curve_summary["best_validation_average_precision"]
            ),
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "early_stopping_triggered": bool(learning_curve_summary["early_stopping_triggered"]),
            "estimator_cap_reached": bool(learning_curve_summary["estimator_cap_reached"]),
        }
    )

    b0_metrics = read_json(ROOT_DIR / "reports" / "baseline" / "lightgbm_metrics.json")
    comparison_to_b0 = build_comparison_table(b0_metrics, metrics, "b0", "g1")
    comparison_to_b1_card1 = build_comparison_table(b1_card1_metrics, metrics, "b1_card1", "g1")

    feature_importance = build_feature_importance(model)
    if len(feature_importance) != len(g1_features):
        raise AssertionError("G1 feature importance does not match its manifest.")
    ranks = embedding_gain_ranks(feature_importance, feat_names)

    validation_predictions = validation_metadata.copy()
    validation_predictions["prediction"] = validation_scores

    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(paths["model"]), num_iteration=model.best_iteration_)
    write_json(paths["metrics"], metrics)
    feature_importance.to_csv(paths["feature_importance"], index=False)
    validation_predictions.to_parquet(
        paths["validation_predictions"], index=False, engine="pyarrow", compression="snappy"
    )
    learning_curve.to_csv(paths["learning_curve"], index=False)
    comparison_to_b0.to_csv(paths["comparison_to_b0"], index=False)
    comparison_to_b1_card1.to_csv(paths["comparison_to_b1_card1"], index=False)

    significance = compute_significance(validation_predictions)
    write_json(paths["significance"], significance)

    g1_meta = build_g1_metadata(
        relation=relation,
        model=model,
        b0_metadata=b0_metadata,
        b1_card1_metadata=b1_card1_metadata,
        embedding_metadata=embedding_metadata,
        g1_features=g1_features,
        categorical_columns=categorical_columns,
        category_mappings_sha256=mapping_sha256,
        scale_pos_weight=scale_pos_weight,
        train_row_count=len(X_train),
        validation_row_count=len(X_validation),
        learning_curve_summary=learning_curve_summary,
        validation_metrics=metrics,
        b0_artifact_hashes=reference_hashes_before,
        embedding_ranks=ranks,
        significance=significance,
        paths=paths,
    )
    write_json(paths["metadata"], g1_meta)
    assert_protected_artifacts_unchanged(
        reference_hashes_before, all_protected, label="frozen B0/B1-card1"
    )

    expected_artifacts = [
        paths["model"], paths["metrics"], paths["metadata"], paths["feature_importance"],
        paths["validation_predictions"], paths["learning_curve"], paths["comparison_to_b0"],
        paths["comparison_to_b1_card1"], paths["significance"],
    ]
    missing = [str(p) for p in expected_artifacts if not p.exists()]
    if missing:
        raise OSError(f"G1 artifacts were not created: {missing}.")

    print(f"[G1-{relation}] Best iteration: {model.best_iteration_:,}")
    print(f"[G1-{relation}] Actual stopping iteration: {learning_curve_summary['actual_stopping_iteration']:,}")
    print(f"[G1-{relation}] Early stopping triggered: {'YES' if learning_curve_summary['early_stopping_triggered'] else 'NO'}")
    print(f"[G1-{relation}] Validation PR-AUC:  {metrics['pr_auc']:.12f}")
    print(f"[G1-{relation}] Validation ROC-AUC: {metrics['roc_auc']:.12f}")
    print(f"[G1-{relation}] Embedding gain-rank range: {ranks['best_embedding_gain_rank']}-{ranks['worst_embedding_gain_rank']} of {ranks['total_features_in_model']}")
    print(f"[G1-{relation}] G1 vs B0 PR-AUC delta 95% CI: [{significance['g1_vs_b0']['ci_lower_95']:+.5f}, {significance['g1_vs_b0']['ci_upper_95']:+.5f}] excludes zero: {significance['g1_vs_b0']['excludes_zero']}")
    print(f"[G1-{relation}] G1 vs B1-card1 PR-AUC delta 95% CI: [{significance['g1_vs_b1_card1']['ci_lower_95']:+.5f}, {significance['g1_vs_b1_card1']['ci_upper_95']:+.5f}] excludes zero: {significance['g1_vs_b1_card1']['excludes_zero']}")
    print(f"[G1-{relation}] Model saved: {paths['model']}")
    print(f"[G1-{relation}] Reports saved: {paths['report_dir']}")
    print(f"[G1-{relation}] Frozen B0/B1-card1 artifacts unchanged: YES")
    print(f"[G1-{relation}] Final test evaluated: NO")


def main() -> None:
    run_g1(RELATION)


if __name__ == "__main__":
    main()
