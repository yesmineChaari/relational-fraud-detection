from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype

from src.features.build_relational_features import (
    RELATION_REGISTRY,
    _feature_names,
    _output_path as _rel_output_path,
    _metadata_path as _rel_metadata_path,
)
from src.models.train_lightgbm_baseline import (
    CATEGORY_MAPPINGS_PATH,
    EXPECTED_ROWS,
    EXPECTED_SPLIT_COUNTS,
    FORBIDDEN_FEATURE_COLUMNS,
    METADATA_PATH as B0_METADATA_PATH,
    METRICS_PATH as B0_METRICS_PATH,
    MODEL_DATASET_PATH,
    MODEL_PATH as B0_MODEL_PATH,
    RANDOM_SEED,
    apply_category_mapping,
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

ROOT_DIR = Path(__file__).resolve().parents[2]

EXPECTED_B0_FEATURE_COUNT = 435
EXPECTED_B1_FEATURE_COUNT = 439
MAX_ESTIMATORS = 6_000
EARLY_STOPPING_ROUNDS = 200
LOG_EVALUATION_PERIOD = 50
EVALUATION_SPLIT = "validation"
TEST_EVALUATED = False
FROZEN_PARAMETER_NAMES = [
    "objective",
    "learning_rate",
    "num_leaves",
    "max_depth",
    "min_child_samples",
    "subsample",
    "subsample_freq",
    "colsample_bytree",
    "reg_alpha",
    "reg_lambda",
    "random_state",
    "deterministic",
    "force_col_wise",
]

# Always-protected artifacts that must never be overwritten by any B1 variant.
B0_PROTECTED_PATHS = [
    B0_MODEL_PATH,
    B0_METADATA_PATH,
    B0_METRICS_PATH,
    CATEGORY_MAPPINGS_PATH,
]
# The original card_core_addr1 B1 is also frozen once created.
B1_CARD_CORE_ADDR1_PROTECTED = [
    ROOT_DIR / "models" / "lightgbm_b1_card_core_addr1.txt",
    ROOT_DIR / "reports" / "b1" / "card_core_addr1" / "metrics.json",
    ROOT_DIR / "reports" / "b1" / "card_core_addr1" / "metadata.json",
    ROOT_DIR / "reports" / "b1" / "card_core_addr1" / "feature_importance.csv",
]


def _resolve_relation_paths(relation: str) -> tuple[Path, Path, Path, Path, Path, Path, Path]:
    """Return (model_path, report_dir, metrics, metadata, feat_imp, val_pred, learning_curve, comparison)."""
    model_path = ROOT_DIR / "models" / f"lightgbm_b1_{relation}.txt"
    report_dir = ROOT_DIR / "reports" / "b1" / relation
    return (
        model_path,
        report_dir,
        report_dir / "metrics.json",
        report_dir / "metadata.json",
        report_dir / "feature_importance.csv",
        report_dir / "validation_predictions.parquet",
        report_dir / "learning_curve.csv",
        report_dir / "comparison_to_b0.csv",
    )


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required frozen artifact not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def file_sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_protected_artifacts(paths: list[Path]) -> dict[str, str]:
    return {
        repository_relative(p): file_sha256(p)
        for p in paths
        if p.exists()
    }


def assert_protected_artifacts_unchanged(
    reference_hashes: dict[str, str],
    paths: list[Path],
    label: str = "protected",
) -> None:
    current = snapshot_protected_artifacts(paths)
    changed = sorted(
        p for p in set(reference_hashes) | set(current)
        if reference_hashes.get(p) != current.get(p)
    )
    if changed:
        raise AssertionError(f"B1 variant modified {label} artifacts: {changed}.")


def load_frozen_b0_metadata(path: Path = B0_METADATA_PATH) -> dict[str, Any]:
    metadata = read_json(path)
    if metadata.get("baseline_frozen") is not True:
        raise ValueError("B0 metadata is not marked as frozen.")
    if metadata.get("selected_as_b0") is not True:
        raise ValueError("B0 metadata is not marked as the selected baseline.")
    if metadata.get("test_evaluated") is not False:
        raise ValueError("Frozen B0 metadata violates final-test discipline.")
    feature_columns = metadata.get("feature_columns")
    categorical_columns = metadata.get("categorical_feature_columns")
    numeric_columns = metadata.get("numeric_feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Frozen B0 feature manifest is missing or invalid.")
    if len(feature_columns) != len(set(feature_columns)):
        raise ValueError("Frozen B0 feature manifest contains duplicates.")
    if len(feature_columns) != EXPECTED_B0_FEATURE_COUNT:
        raise ValueError(
            f"Frozen B0 predictor-count sanity check failed: "
            f"expected {EXPECTED_B0_FEATURE_COUNT}, got {len(feature_columns)}."
        )
    if not isinstance(categorical_columns, list) or not isinstance(numeric_columns, list):
        raise ValueError("Frozen B0 dtype manifests are missing or invalid.")
    if set(categorical_columns) & set(numeric_columns):
        raise ValueError("Frozen B0 categorical and numeric manifests overlap.")
    if set(categorical_columns) | set(numeric_columns) != set(feature_columns):
        raise ValueError("Frozen B0 dtype manifests do not cover its predictors.")
    if metadata.get("number_of_predictors") != len(feature_columns):
        raise ValueError("Frozen B0 predictor count disagrees with its manifest.")
    return metadata


def load_feature_builder_metadata(relation: str) -> dict[str, Any]:
    meta_path = _rel_metadata_path(relation)
    metadata = read_json(meta_path)
    group_columns = RELATION_REGISTRY[relation]
    feat_names = _feature_names(relation)
    if metadata.get("relation_name") != relation:
        raise ValueError(f"Relational feature metadata has the wrong relation name (expected {relation}).")
    if metadata.get("group_columns") != group_columns:
        raise ValueError("Relational feature metadata has the wrong group definition.")
    if metadata.get("feature_names") != feat_names:
        raise ValueError("Relational feature metadata has the wrong feature manifest.")
    if metadata.get("target_labels_used") is not False:
        raise ValueError("Relational feature builder must not use target labels.")
    if metadata.get("row_count") != EXPECTED_ROWS:
        raise ValueError("Relational feature metadata has the wrong row count.")
    return metadata


def validate_relational_feature_dtypes(
    relational_df: pd.DataFrame,
    feat_names: list[str],
) -> None:
    for column in feat_names[:3]:
        if not is_integer_dtype(relational_df[column].dtype):
            raise TypeError(f"{column} must use an integer dtype.")
        if relational_df[column].isna().any():
            raise ValueError(f"{column} contains missing counts.")
        if relational_df[column].lt(0).any():
            raise ValueError(f"{column} contains negative counts.")
    recency_column = feat_names[3]
    if not is_float_dtype(relational_df[recency_column].dtype):
        raise TypeError(f"{recency_column} must use a floating dtype.")
    if relational_df[recency_column].dropna().lt(0).any():
        raise ValueError(f"{recency_column} contains negative recencies.")


def validate_relational_merge(
    model_index: pd.DataFrame,
    relational_df: pd.DataFrame,
    feat_names: list[str],
) -> pd.DataFrame:
    expected_cols = ["TransactionID", *feat_names]
    if list(relational_df.columns) != expected_cols:
        raise ValueError(
            f"Relational feature columns must be exactly {expected_cols}; "
            f"got {list(relational_df.columns)}."
        )
    required_index_columns = {"TransactionID", "split"}
    missing = required_index_columns - set(model_index.columns)
    if missing:
        raise ValueError(f"Model index is missing columns: {sorted(missing)}.")
    if model_index["TransactionID"].isna().any():
        raise ValueError("Model index contains missing TransactionID values.")
    if relational_df["TransactionID"].isna().any():
        raise ValueError("Relational features contain missing TransactionID values.")
    if not model_index["TransactionID"].is_unique:
        raise ValueError("Model index contains duplicate TransactionID values.")
    if not relational_df["TransactionID"].is_unique:
        raise ValueError("Relational features contain duplicate TransactionID values.")
    validate_relational_feature_dtypes(relational_df, feat_names)

    membership = model_index[["TransactionID"]].merge(
        relational_df[["TransactionID"]],
        on="TransactionID",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    missing_rel = int(membership["_merge"].eq("left_only").sum())
    extra_rel = int(membership["_merge"].eq("right_only").sum())
    if missing_rel or extra_rel:
        raise ValueError(
            f"Relational/model TransactionID membership mismatch: "
            f"missing relational IDs={missing_rel}, extra relational IDs={extra_rel}."
        )

    merged = model_index[["TransactionID", "split"]].merge(
        relational_df,
        on="TransactionID",
        how="left",
        sort=False,
        validate="one_to_one",
    )
    if len(merged) != len(model_index):
        raise AssertionError("Relational merge changed the model-index row count.")
    if not np.array_equal(
        merged["TransactionID"].to_numpy(),
        model_index["TransactionID"].to_numpy(),
    ):
        raise AssertionError("Relational merge changed TransactionID order.")
    return merged


def attach_relational_features(
    partition_df: pd.DataFrame,
    merged_index: pd.DataFrame,
    split_name: str,
    feat_names: list[str],
) -> pd.DataFrame:
    if any(col in partition_df.columns for col in feat_names):
        raise ValueError("Model partition already contains B1 relational features.")
    original_ids = partition_df["TransactionID"].to_numpy(copy=True)
    original_splits = partition_df["split"].astype("string").copy()
    relation_partition = merged_index.loc[
        merged_index["split"].astype("string").eq(split_name),
        ["TransactionID", *feat_names],
    ]
    aligned = partition_df[["TransactionID"]].merge(
        relation_partition,
        on="TransactionID",
        how="left",
        sort=False,
        indicator=True,
        validate="one_to_one",
    )
    if not aligned["_merge"].eq("both").all():
        raise ValueError(f"{split_name} contains rows without relational features.")
    aligned = aligned.drop(columns="_merge")
    if not np.array_equal(aligned["TransactionID"].to_numpy(), original_ids):
        raise AssertionError(f"{split_name} relational attachment changed row order.")
    result = partition_df.copy(deep=False)
    for col in feat_names:
        result[col] = aligned[col].to_numpy(copy=False)
    if len(result) != len(partition_df):
        raise AssertionError(f"{split_name} row count changed after relational merge.")
    if not result["split"].astype("string").equals(original_splits):
        raise AssertionError(f"{split_name} split assignments changed.")
    return result


def load_b1_datasets(relation: str = "card_core_addr1") -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    feat_names = _feature_names(relation)
    rel_path = _rel_output_path(relation)
    train_df, validation_df, dataset_summary = load_model_dataset()
    if not rel_path.exists():
        raise FileNotFoundError(
            f"Relational features not found for relation {relation!r}. "
            f"Run: python -m src.features.build_relational_features --relation {relation}\n"
            f"Expected: {rel_path}"
        )
    model_index = pd.read_parquet(MODEL_DATASET_PATH, columns=["TransactionID", "split"])
    validate_split_counts(model_index, "model_dataset.parquet B1 index")
    relational_df = pd.read_parquet(rel_path)
    if len(relational_df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} relational rows; got {len(relational_df):,}."
        )
    merged_index = validate_relational_merge(model_index, relational_df, feat_names)

    train_with_relations = attach_relational_features(train_df, merged_index, "train", feat_names)
    validation_with_relations = attach_relational_features(
        validation_df, merged_index, "validation", feat_names
    )
    if len(train_with_relations) != EXPECTED_SPLIT_COUNTS["train"]:
        raise AssertionError("B1 train row count changed.")
    if len(validation_with_relations) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise AssertionError("B1 validation row count changed.")
    return train_with_relations, validation_with_relations, dataset_summary


def build_b1_feature_manifest(
    b0_features: list[str],
    feat_names: list[str],
) -> list[str]:
    if not b0_features or len(b0_features) != len(set(b0_features)):
        raise ValueError("B0 feature manifest is empty or contains duplicates.")
    overlap = sorted(set(b0_features) & set(feat_names))
    if overlap:
        raise ValueError(f"B0 feature manifest already contains B1 features: {overlap}.")
    b1_features = [*b0_features, *feat_names]
    if len(b1_features) != len(b0_features) + 4:
        raise AssertionError("B1 must contain exactly four additional predictors.")
    return b1_features


def validate_model_columns_against_frozen_b0(
    model_columns: list[str],
    b0_features: list[str],
    feat_names: list[str],
) -> None:
    actual_b0_features = [
        col for col in model_columns
        if col not in FORBIDDEN_FEATURE_COLUMNS and col not in feat_names
    ]
    if actual_b0_features != b0_features:
        missing = sorted(set(b0_features) - set(actual_b0_features))
        extra = sorted(set(actual_b0_features) - set(b0_features))
        raise ValueError(
            f"Current model dataset does not match the frozen B0 manifest: "
            f"missing={missing}, extra={extra}."
        )


def load_frozen_category_mappings(
    categorical_columns: list[str],
    path: Path = CATEGORY_MAPPINGS_PATH,
) -> tuple[dict[str, dict[str, int]], str]:
    payload = read_json(path)
    if payload.get("fit_split") != "train":
        raise ValueError("Canonical categorical mappings were not fitted on train.")
    if payload.get("missing_token") != "__MISSING__":
        raise ValueError("Canonical categorical missing token changed.")
    if payload.get("unknown_token") != "__UNKNOWN__":
        raise ValueError("Canonical categorical unknown token changed.")
    mappings = payload.get("columns")
    if not isinstance(mappings, dict):
        raise ValueError("Canonical categorical mappings are missing.")
    if list(mappings) != categorical_columns:
        raise ValueError("Canonical mapping columns differ from the frozen categorical manifest.")
    for column, mapping in mappings.items():
        if not isinstance(mapping, dict) or not all(isinstance(v, int) for v in mapping.values()):
            raise TypeError(f"Invalid categorical mapping for {column}.")
    return mappings, file_sha256(path)


def apply_frozen_category_mappings(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    categorical_columns: list[str],
    mappings: dict[str, dict[str, int]],
) -> None:
    if list(mappings) != categorical_columns:
        raise ValueError("Categorical mapping manifest does not match B0.")
    for column in categorical_columns:
        if column not in X_train or column not in X_validation:
            raise ValueError(f"Categorical predictor {column} is missing from B1.")
        X_train[column] = apply_category_mapping(X_train[column], mappings[column])
        X_validation[column] = apply_category_mapping(X_validation[column], mappings[column])


def _parameter_values_match(left: Any, right: Any) -> bool:
    if isinstance(left, (float, int)) and isinstance(right, (float, int)):
        return bool(np.isclose(left, right, rtol=0.0, atol=0.0))
    return left == right


def validate_frozen_lightgbm_configuration(
    model: LGBMClassifier,
    b0_metadata: dict[str, Any],
) -> None:
    reference_parameters = b0_metadata.get("lightgbm_parameters")
    if not isinstance(reference_parameters, dict):
        raise ValueError("Frozen B0 LightGBM parameters are missing.")
    actual_parameters = model.get_params(deep=False)
    for name in FROZEN_PARAMETER_NAMES:
        if name not in reference_parameters:
            raise ValueError(f"Frozen B0 parameter is missing: {name}.")
        if not _parameter_values_match(actual_parameters.get(name), reference_parameters[name]):
            raise ValueError(
                f"B1 LightGBM parameter {name} differs from frozen B0: "
                f"B1={actual_parameters.get(name)!r}, B0={reference_parameters[name]!r}."
            )
    if actual_parameters.get("n_estimators") != MAX_ESTIMATORS:
        raise ValueError("B1 estimator cap must remain 6000.")
    if b0_metadata.get("maximum_estimators") != MAX_ESTIMATORS:
        raise ValueError("Frozen B0 estimator cap is not 6000.")
    if b0_metadata.get("early_stopping_rounds") != EARLY_STOPPING_ROUNDS:
        raise ValueError("Frozen B0 early-stopping patience is not 200.")
    if not _parameter_values_match(
        actual_parameters.get("scale_pos_weight"),
        b0_metadata.get("scale_pos_weight"),
    ):
        raise ValueError("B1 class weight differs from frozen B0.")


def _flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flattened = {
        "PR-AUC": float(metrics["pr_auc"]),
        "ROC-AUC": float(metrics["roc_auc"]),
    }
    labels = {
        "top_0.5_pct": "0.5%",
        "top_1_pct": "1%",
        "top_2_pct": "2%",
        "top_5_pct": "5%",
    }
    ranking = metrics.get("ranking_metrics")
    if not isinstance(ranking, dict) or set(ranking) != set(labels):
        raise ValueError("Metrics do not contain the exact B0 ranking metrics.")
    for key, display in labels.items():
        flattened[f"Precision@{display}"] = float(ranking[key]["precision"])
        flattened[f"Recall@{display}"] = float(ranking[key]["recall"])
    return flattened


def build_comparison_to_b0(
    b0_metrics: dict[str, Any],
    b1_metrics: dict[str, Any],
) -> pd.DataFrame:
    if b0_metrics.get("test_evaluated") is not False:
        raise ValueError("B0 comparison artifact violates final-test discipline.")
    if b1_metrics.get("test_evaluated") is not False:
        raise ValueError("B1 comparison artifact violates final-test discipline.")
    b0_values = _flatten_metrics(b0_metrics)
    b1_values = _flatten_metrics(b1_metrics)
    if list(b0_values) != list(b1_values):
        raise AssertionError("B0 and B1 metric manifests differ.")
    return pd.DataFrame(
        [
            {
                "metric": metric,
                "b0_value": b0_values[metric],
                "b1_value": b1_values[metric],
                "delta_b1_minus_b0": b1_values[metric] - b0_values[metric],
            }
            for metric in b0_values
        ]
    )


def build_b1_metadata(
    *,
    relation: str,
    model: LGBMClassifier,
    b0_metadata: dict[str, Any],
    feature_builder_metadata: dict[str, Any],
    b1_features: list[str],
    categorical_columns: list[str],
    category_mappings_sha256: str,
    scale_pos_weight: float,
    train_row_count: int,
    validation_row_count: int,
    learning_curve_summary: dict[str, Any],
    validation_metrics: dict[str, Any],
    b0_artifact_hashes: dict[str, str],
    model_path: Path,
    metrics_path: Path,
    feat_imp_path: Path,
    val_pred_path: Path,
    learning_curve_path: Path,
    comparison_path: Path,
    report_dir: Path,
) -> dict[str, Any]:
    group_columns = RELATION_REGISTRY[relation]
    feat_names = _feature_names(relation)
    numeric_columns = [col for col in b1_features if col not in categorical_columns]
    return {
        "experiment_name": f"B1_relational_lightgbm_{relation}",
        "model_family": "LightGBM",
        "controlled_experiment_definition": (
            "B1 = frozen B0 feature set + four project-engineered relational features"
        ),
        "relation_name": relation,
        "relation_group_columns": group_columns,
        "relational_feature_names": feat_names,
        "relational_feature_source_path": repository_relative(_rel_output_path(relation)),
        "relational_feature_metadata_path": repository_relative(_rel_metadata_path(relation)),
        "b0_reference_metrics_path": repository_relative(B0_METRICS_PATH),
        "b0_reference_metadata_path": repository_relative(B0_METADATA_PATH),
        "b0_artifact_sha256": b0_artifact_hashes,
        "b0_feature_count": len(b0_metadata["feature_columns"]),
        "b1_feature_count": len(b1_features),
        "feature_columns": b1_features,
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
        "feature_builder_target_labels_used": bool(
            feature_builder_metadata["target_labels_used"]
        ),
        "original_ieee_features_removed_for_b1": False,
        "model_path": repository_relative(model_path),
        "metrics_path": repository_relative(metrics_path),
        "feature_importance_path": repository_relative(feat_imp_path),
        "validation_predictions_path": repository_relative(val_pred_path),
        "learning_curve_path": repository_relative(learning_curve_path),
        "comparison_to_b0_path": repository_relative(comparison_path),
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


def run_for_relation(relation: str) -> None:
    """Full B1 training pipeline for a named relation."""
    if relation not in RELATION_REGISTRY:
        raise ValueError(
            f"Unknown relation: {relation!r}. Supported: {sorted(RELATION_REGISTRY)}."
        )
    feat_names = _feature_names(relation)

    (
        model_path,
        report_dir,
        metrics_path,
        metadata_path,
        feat_imp_path,
        val_pred_path,
        learning_curve_path,
        comparison_path,
    ) = _resolve_relation_paths(relation)

    # The frozen B0 and the archived B1-card_core_addr1 are both hash-pinned
    # before training and re-checked afterwards, so no run of this script can
    # silently replace an artifact another experiment is compared against.
    all_protected = [*B0_PROTECTED_PATHS, *B1_CARD_CORE_ADDR1_PROTECTED]
    b0_hashes_before = snapshot_protected_artifacts(all_protected)
    if relation == "card_core_addr1" and any(
        p.exists() for p in B1_CARD_CORE_ADDR1_PROTECTED
    ):
        raise FileExistsError(
            "B1-card_core_addr1 is frozen and archived as the original relational "
            "experiment; delete its artifacts deliberately before regenerating them."
        )
    b0_metadata = load_frozen_b0_metadata()
    feature_builder_metadata = load_feature_builder_metadata(relation)
    b0_metrics = read_json(B0_METRICS_PATH)

    print(f"[{relation}] Loading frozen B0 train/validation data: {MODEL_DATASET_PATH}")
    print(f"[{relation}] Loading relational features: {_rel_output_path(relation)}")
    train_df, validation_df, _ = load_b1_datasets(relation)

    b0_features = list(b0_metadata["feature_columns"])
    validate_model_columns_against_frozen_b0(list(train_df.columns), b0_features, feat_names)
    validate_model_columns_against_frozen_b0(list(validation_df.columns), b0_features, feat_names)
    b1_features = build_b1_feature_manifest(b0_features, feat_names)
    if len(b1_features) != EXPECTED_B1_FEATURE_COUNT:
        raise ValueError(
            f"B1 predictor-count sanity check failed: "
            f"expected {EXPECTED_B1_FEATURE_COUNT}, got {len(b1_features)}."
        )

    categorical_columns = list(b0_metadata["categorical_feature_columns"])
    raw_train_cat = identify_categorical_columns(train_df[b1_features])
    raw_val_cat = identify_categorical_columns(validation_df[b1_features])
    if raw_train_cat != categorical_columns:
        raise TypeError("B1 categorical features differ from frozen B0.")
    if raw_val_cat != categorical_columns:
        raise TypeError("B1 validation categorical features differ from frozen B0.")
    for col in feat_names:
        if not is_numeric_dtype(train_df[col].dtype):
            raise TypeError(f"B1 relational predictor {col} is not numeric.")

    validation_metadata = validation_df[["TransactionID", "TransactionDT", "isFraud"]].copy()
    y_train = train_df["isFraud"].astype("int8").copy()
    y_validation = validation_df["isFraud"].astype("int8").copy()
    X_train = train_df[b1_features].copy()
    X_validation = validation_df[b1_features].copy()
    del train_df, validation_df

    mappings, mapping_sha256 = load_frozen_category_mappings(categorical_columns)
    print(f"[{relation}] Applying frozen B0 categorical mappings (no fitting)...")
    apply_frozen_category_mappings(X_train, X_validation, categorical_columns, mappings)
    assert_supported_model_dtypes(X_train, "B1 training predictors")
    assert_supported_model_dtypes(X_validation, "B1 validation predictors")

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    if not np.isclose(scale_pos_weight, float(b0_metadata["scale_pos_weight"]), rtol=0.0, atol=0.0):
        raise ValueError("B1 train-only class weight differs from frozen B0.")
    model = build_lightgbm_model(scale_pos_weight, n_estimators=MAX_ESTIMATORS)
    validate_frozen_lightgbm_configuration(model, b0_metadata)

    print(f"[{relation}] Train rows: {len(X_train):,}")
    print(f"[{relation}] Validation rows: {len(X_validation):,}")
    print(f"[{relation}] B0 predictors: {len(b0_features):,}")
    print(f"[{relation}] B1 predictors: {len(b1_features):,}")
    print(f"[{relation}] Categorical predictors: {len(categorical_columns):,}")
    print(f"[{relation}] scale_pos_weight: {scale_pos_weight:.15f}")

    evaluation_results: dict[str, dict[str, list[float]]] = {}
    callbacks = [
        lightgbm.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, first_metric_only=True),
        lightgbm.record_evaluation(evaluation_results),
        lightgbm.log_evaluation(period=LOG_EVALUATION_PERIOD),
    ]
    print(f"[{relation}] Training B1 LightGBM...")
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
        raise AssertionError("B1 validation prediction count is incorrect.")
    if not np.isfinite(validation_scores).all():
        raise AssertionError("B1 validation predictions contain non-finite values.")

    metrics = evaluate_validation(y_validation, validation_scores)
    metrics.update(
        {
            "model": f"B1_relational_lightgbm_{relation}",
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
    comparison = build_comparison_to_b0(b0_metrics, metrics)
    feature_importance = build_feature_importance(model)
    if len(feature_importance) != len(b1_features):
        raise AssertionError("B1 feature importance does not match its manifest.")
    validation_predictions = validation_metadata.copy()
    validation_predictions["prediction"] = validation_scores

    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(model_path), num_iteration=model.best_iteration_)
    write_json(metrics_path, metrics)
    feature_importance.to_csv(feat_imp_path, index=False)
    validation_predictions.to_parquet(
        val_pred_path, index=False, engine="pyarrow", compression="snappy"
    )
    learning_curve.to_csv(learning_curve_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    b1_meta = build_b1_metadata(
        relation=relation,
        model=model,
        b0_metadata=b0_metadata,
        feature_builder_metadata=feature_builder_metadata,
        b1_features=b1_features,
        categorical_columns=categorical_columns,
        category_mappings_sha256=mapping_sha256,
        scale_pos_weight=scale_pos_weight,
        train_row_count=len(X_train),
        validation_row_count=len(X_validation),
        learning_curve_summary=learning_curve_summary,
        validation_metrics=metrics,
        b0_artifact_hashes=b0_hashes_before,
        model_path=model_path,
        metrics_path=metrics_path,
        feat_imp_path=feat_imp_path,
        val_pred_path=val_pred_path,
        learning_curve_path=learning_curve_path,
        comparison_path=comparison_path,
        report_dir=report_dir,
    )
    write_json(metadata_path, b1_meta)
    assert_protected_artifacts_unchanged(b0_hashes_before, all_protected, label="frozen B0/B1")

    expected_artifacts = [
        model_path, metrics_path, metadata_path, feat_imp_path,
        val_pred_path, learning_curve_path, comparison_path,
    ]
    missing = [str(p) for p in expected_artifacts if not p.exists()]
    if missing:
        raise OSError(f"B1 artifacts were not created: {missing}.")

    print(f"[{relation}] Best iteration: {model.best_iteration_:,}")
    print(f"[{relation}] Actual stopping iteration: {learning_curve_summary['actual_stopping_iteration']:,}")
    print(f"[{relation}] Early stopping triggered: {'YES' if learning_curve_summary['early_stopping_triggered'] else 'NO'}")
    print(f"[{relation}] Validation PR-AUC:  {metrics['pr_auc']:.12f}")
    print(f"[{relation}] Validation ROC-AUC: {metrics['roc_auc']:.12f}")
    print(f"[{relation}] Model saved: {model_path}")
    print(f"[{relation}] Reports saved: {report_dir}")
    print(f"[{relation}] Frozen B0 artifacts unchanged: YES")
    print(f"[{relation}] Final test evaluated: NO")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train B1 relational LightGBM for a given entity relation."
    )
    parser.add_argument(
        "--relation",
        default="card_core_addr1",
        choices=sorted(RELATION_REGISTRY),
        help="Relation to train (default: card_core_addr1).",
    )
    args = parser.parse_args()
    run_for_relation(args.relation)


if __name__ == "__main__":
    main()
