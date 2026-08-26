from __future__ import annotations

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
    GROUP_COLUMNS,
    METADATA_PATH as RELATIONAL_METADATA_PATH,
    OUTPUT_COLUMNS as RELATIONAL_OUTPUT_COLUMNS,
    OUTPUT_PATH as RELATIONAL_FEATURES_PATH,
    RELATION_NAME,
    RELATIONAL_FEATURES,
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
MODEL_PATH = ROOT_DIR / "models" / "lightgbm_b1_card_core_addr1.txt"
REPORT_DIR = ROOT_DIR / "reports" / "b1" / "card_core_addr1"
METRICS_PATH = REPORT_DIR / "metrics.json"
METADATA_PATH = REPORT_DIR / "metadata.json"
FEATURE_IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"
VALIDATION_PREDICTIONS_PATH = REPORT_DIR / "validation_predictions.parquet"
LEARNING_CURVE_PATH = REPORT_DIR / "learning_curve.csv"
COMPARISON_PATH = REPORT_DIR / "comparison_to_b0.csv"

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
B0_PROTECTED_PATHS = [
    B0_MODEL_PATH,
    B0_METADATA_PATH,
    B0_METRICS_PATH,
    CATEGORY_MAPPINGS_PATH,
]


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


def snapshot_b0_artifacts() -> dict[str, str]:
    return {
        repository_relative(path): file_sha256(path)
        for path in B0_PROTECTED_PATHS
    }


def assert_b0_artifacts_unchanged(reference_hashes: dict[str, str]) -> None:
    current_hashes = snapshot_b0_artifacts()
    if current_hashes != reference_hashes:
        changed = sorted(
            path
            for path in set(reference_hashes) | set(current_hashes)
            if reference_hashes.get(path) != current_hashes.get(path)
        )
        raise AssertionError(f"B1 modified frozen B0 artifacts: {changed}.")


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
            "Frozen B0 predictor-count sanity check failed: "
            f"expected {EXPECTED_B0_FEATURE_COUNT}, got {len(feature_columns)}."
        )
    if not isinstance(categorical_columns, list) or not isinstance(
        numeric_columns, list
    ):
        raise ValueError("Frozen B0 dtype manifests are missing or invalid.")
    if set(categorical_columns) & set(numeric_columns):
        raise ValueError("Frozen B0 categorical and numeric manifests overlap.")
    if set(categorical_columns) | set(numeric_columns) != set(feature_columns):
        raise ValueError("Frozen B0 dtype manifests do not cover its predictors.")
    if metadata.get("number_of_predictors") != len(feature_columns):
        raise ValueError("Frozen B0 predictor count disagrees with its manifest.")
    return metadata


def load_feature_builder_metadata(
    path: Path = RELATIONAL_METADATA_PATH,
) -> dict[str, Any]:
    metadata = read_json(path)
    if metadata.get("relation_name") != RELATION_NAME:
        raise ValueError("Relational feature metadata has the wrong relation name.")
    if metadata.get("group_columns") != GROUP_COLUMNS:
        raise ValueError("Relational feature metadata has the wrong group definition.")
    if metadata.get("feature_names") != RELATIONAL_FEATURES:
        raise ValueError("Relational feature metadata has the wrong feature manifest.")
    if metadata.get("target_labels_used") is not False:
        raise ValueError("Relational feature builder must not use target labels.")
    if metadata.get("row_count") != EXPECTED_ROWS:
        raise ValueError("Relational feature metadata has the wrong row count.")
    return metadata


def validate_relational_feature_dtypes(relational_df: pd.DataFrame) -> None:
    for column in RELATIONAL_FEATURES[:3]:
        if not is_integer_dtype(relational_df[column].dtype):
            raise TypeError(f"{column} must use an integer dtype.")
        if relational_df[column].isna().any():
            raise ValueError(f"{column} contains missing counts.")
        if relational_df[column].lt(0).any():
            raise ValueError(f"{column} contains negative counts.")
    recency_column = RELATIONAL_FEATURES[3]
    if not is_float_dtype(relational_df[recency_column].dtype):
        raise TypeError(f"{recency_column} must use a floating dtype.")
    if relational_df[recency_column].dropna().lt(0).any():
        raise ValueError(f"{recency_column} contains negative recencies.")


def validate_relational_merge(
    model_index: pd.DataFrame,
    relational_df: pd.DataFrame,
) -> pd.DataFrame:
    if list(relational_df.columns) != RELATIONAL_OUTPUT_COLUMNS:
        raise ValueError(
            "Relational feature columns must be exactly "
            f"{RELATIONAL_OUTPUT_COLUMNS}; got {list(relational_df.columns)}."
        )
    required_index_columns = {"TransactionID", "split"}
    missing_index_columns = required_index_columns - set(model_index.columns)
    if missing_index_columns:
        raise ValueError(
            f"Model index is missing columns: {sorted(missing_index_columns)}."
        )
    if model_index["TransactionID"].isna().any():
        raise ValueError("Model index contains missing TransactionID values.")
    if relational_df["TransactionID"].isna().any():
        raise ValueError("Relational features contain missing TransactionID values.")
    if not model_index["TransactionID"].is_unique:
        raise ValueError("Model index contains duplicate TransactionID values.")
    if not relational_df["TransactionID"].is_unique:
        raise ValueError("Relational features contain duplicate TransactionID values.")
    validate_relational_feature_dtypes(relational_df)

    membership = model_index[["TransactionID"]].merge(
        relational_df[["TransactionID"]],
        on="TransactionID",
        how="outer",
        indicator=True,
        validate="one_to_one",
    )
    missing_relational = int(membership["_merge"].eq("left_only").sum())
    extra_relational = int(membership["_merge"].eq("right_only").sum())
    if missing_relational or extra_relational:
        raise ValueError(
            "Relational/model TransactionID membership mismatch: "
            f"missing relational IDs={missing_relational}, "
            f"extra relational IDs={extra_relational}."
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
    if not merged["split"].astype("string").equals(
        model_index["split"].astype("string")
    ):
        raise AssertionError("Relational merge changed split assignments.")
    return merged


def attach_relational_features(
    partition_df: pd.DataFrame,
    merged_index: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    if any(column in partition_df.columns for column in RELATIONAL_FEATURES):
        raise ValueError("Model partition already contains B1 relational features.")
    original_ids = partition_df["TransactionID"].to_numpy(copy=True)
    original_splits = partition_df["split"].astype("string").copy()
    relation_partition = merged_index.loc[
        merged_index["split"].astype("string").eq(split_name),
        ["TransactionID", *RELATIONAL_FEATURES],
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
    for column in RELATIONAL_FEATURES:
        result[column] = aligned[column].to_numpy(copy=False)
    if len(result) != len(partition_df):
        raise AssertionError(f"{split_name} row count changed after relational merge.")
    if not result["split"].astype("string").equals(original_splits):
        raise AssertionError(f"{split_name} split assignments changed.")
    return result


def load_b1_datasets() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    train_df, validation_df, dataset_summary = load_model_dataset()
    if not RELATIONAL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Relational features not found. Run "
            "python -m src.features.build_relational_features first: "
            f"{RELATIONAL_FEATURES_PATH}"
        )
    model_index = pd.read_parquet(
        MODEL_DATASET_PATH,
        columns=["TransactionID", "split"],
    )
    validate_split_counts(model_index, "model_dataset.parquet B1 index")
    relational_df = pd.read_parquet(RELATIONAL_FEATURES_PATH)
    if len(relational_df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} relational rows; got {len(relational_df):,}."
        )
    merged_index = validate_relational_merge(model_index, relational_df)

    train_with_relations = attach_relational_features(
        train_df,
        merged_index,
        "train",
    )
    validation_with_relations = attach_relational_features(
        validation_df,
        merged_index,
        "validation",
    )
    if len(train_with_relations) != EXPECTED_SPLIT_COUNTS["train"]:
        raise AssertionError("B1 train row count changed.")
    if len(validation_with_relations) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise AssertionError("B1 validation row count changed.")
    return train_with_relations, validation_with_relations, dataset_summary


def build_b1_feature_manifest(b0_features: list[str]) -> list[str]:
    if not b0_features or len(b0_features) != len(set(b0_features)):
        raise ValueError("B0 feature manifest is empty or contains duplicates.")
    overlap = sorted(set(b0_features) & set(RELATIONAL_FEATURES))
    if overlap:
        raise ValueError(f"B0 feature manifest already contains B1 features: {overlap}.")
    b1_features = [*b0_features, *RELATIONAL_FEATURES]
    if b1_features[len(b0_features) :] != RELATIONAL_FEATURES:
        raise AssertionError("B1 did not append exactly the four relational features.")
    if len(b1_features) != len(b0_features) + 4:
        raise AssertionError("B1 must contain exactly four additional predictors.")
    return b1_features


def validate_model_columns_against_frozen_b0(
    model_columns: list[str],
    b0_features: list[str],
) -> None:
    actual_b0_features = [
        column
        for column in model_columns
        if column not in FORBIDDEN_FEATURE_COLUMNS
        and column not in RELATIONAL_FEATURES
    ]
    if actual_b0_features != b0_features:
        missing = sorted(set(b0_features) - set(actual_b0_features))
        extra = sorted(set(actual_b0_features) - set(b0_features))
        raise ValueError(
            "Current model dataset does not match the frozen B0 manifest: "
            f"missing={missing}, extra={extra}, "
            f"order_matches={actual_b0_features == b0_features}."
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
        raise ValueError(
            "Canonical mapping columns differ from the frozen categorical manifest."
        )
    for column, mapping in mappings.items():
        if not isinstance(mapping, dict) or not all(
            isinstance(value, int) for value in mapping.values()
        ):
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
        X_validation[column] = apply_category_mapping(
            X_validation[column],
            mappings[column],
        )


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
        if not _parameter_values_match(
            actual_parameters.get(name),
            reference_parameters[name],
        ):
            raise ValueError(
                f"B1 LightGBM parameter {name} differs from frozen B0: "
                f"B1={actual_parameters.get(name)!r}, "
                f"B0={reference_parameters[name]!r}."
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
) -> dict[str, Any]:
    numeric_columns = [
        column for column in b1_features if column not in categorical_columns
    ]
    return {
        "experiment_name": "B1_relational_lightgbm_card_core_addr1",
        "model_family": "LightGBM",
        "controlled_experiment_definition": (
            "B1 = frozen B0 feature set + four project-engineered relational features"
        ),
        "relation_name": RELATION_NAME,
        "relation_group_columns": GROUP_COLUMNS,
        "relational_feature_names": RELATIONAL_FEATURES,
        "relational_feature_source_path": repository_relative(
            RELATIONAL_FEATURES_PATH
        ),
        "relational_feature_metadata_path": repository_relative(
            RELATIONAL_METADATA_PATH
        ),
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
        "categorical_mapping_reference": repository_relative(
            CATEGORY_MAPPINGS_PATH
        ),
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
        "actual_stopping_iteration": int(
            learning_curve_summary["actual_stopping_iteration"]
        ),
        "early_stopping_triggered": bool(
            learning_curve_summary["early_stopping_triggered"]
        ),
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
        "model_path": repository_relative(MODEL_PATH),
        "metrics_path": repository_relative(METRICS_PATH),
        "feature_importance_path": repository_relative(FEATURE_IMPORTANCE_PATH),
        "validation_predictions_path": repository_relative(
            VALIDATION_PREDICTIONS_PATH
        ),
        "learning_curve_path": repository_relative(LEARNING_CURVE_PATH),
        "comparison_to_b0_path": repository_relative(COMPARISON_PATH),
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


def main() -> None:
    b0_hashes_before = snapshot_b0_artifacts()
    b0_metadata = load_frozen_b0_metadata()
    feature_builder_metadata = load_feature_builder_metadata()
    b0_metrics = read_json(B0_METRICS_PATH)

    print(f"Loading frozen B0 train/validation data: {MODEL_DATASET_PATH}")
    print(f"Loading relational features: {RELATIONAL_FEATURES_PATH}")
    train_df, validation_df, _ = load_b1_datasets()

    b0_features = list(b0_metadata["feature_columns"])
    validate_model_columns_against_frozen_b0(
        list(train_df.columns),
        b0_features,
    )
    validate_model_columns_against_frozen_b0(
        list(validation_df.columns),
        b0_features,
    )
    b1_features = build_b1_feature_manifest(b0_features)
    if len(b1_features) != EXPECTED_B1_FEATURE_COUNT:
        raise ValueError(
            "B1 predictor-count sanity check failed: "
            f"expected {EXPECTED_B1_FEATURE_COUNT}, got {len(b1_features)}."
        )

    categorical_columns = list(b0_metadata["categorical_feature_columns"])
    raw_train_categorical = identify_categorical_columns(train_df[b1_features])
    raw_validation_categorical = identify_categorical_columns(
        validation_df[b1_features]
    )
    if raw_train_categorical != categorical_columns:
        raise TypeError("B1 categorical features differ from frozen B0.")
    if raw_validation_categorical != categorical_columns:
        raise TypeError("B1 validation categorical features differ from frozen B0.")
    for column in RELATIONAL_FEATURES:
        if not is_numeric_dtype(train_df[column].dtype):
            raise TypeError(f"B1 relational predictor {column} is not numeric.")

    validation_metadata = validation_df[
        ["TransactionID", "TransactionDT", "isFraud"]
    ].copy()
    y_train = train_df["isFraud"].astype("int8").copy()
    y_validation = validation_df["isFraud"].astype("int8").copy()
    X_train = train_df[b1_features].copy()
    X_validation = validation_df[b1_features].copy()
    del train_df, validation_df

    mappings, mapping_sha256 = load_frozen_category_mappings(
        categorical_columns
    )
    print("Applying frozen B0 categorical mappings (no fitting)...")
    apply_frozen_category_mappings(
        X_train,
        X_validation,
        categorical_columns,
        mappings,
    )
    assert_supported_model_dtypes(X_train, "B1 training predictors")
    assert_supported_model_dtypes(X_validation, "B1 validation predictors")

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    if not np.isclose(
        scale_pos_weight,
        float(b0_metadata["scale_pos_weight"]),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("B1 train-only class weight differs from frozen B0.")
    model = build_lightgbm_model(
        scale_pos_weight,
        n_estimators=MAX_ESTIMATORS,
    )
    validate_frozen_lightgbm_configuration(model, b0_metadata)

    print(f"Train rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_validation):,}")
    print(f"B0 predictors: {len(b0_features):,}")
    print(f"B1 predictors: {len(b1_features):,}")
    print(f"Categorical predictors: {len(categorical_columns):,}")
    print(f"scale_pos_weight: {scale_pos_weight:.15f}")

    evaluation_results: dict[str, dict[str, list[float]]] = {}
    callbacks = [
        lightgbm.early_stopping(
            stopping_rounds=EARLY_STOPPING_ROUNDS,
            first_metric_only=True,
        ),
        lightgbm.record_evaluation(evaluation_results),
        lightgbm.log_evaluation(period=LOG_EVALUATION_PERIOD),
    ]
    print("Training B1 LightGBM...")
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
        X_validation,
        num_iteration=model.best_iteration_,
    )[:, 1]
    if len(validation_scores) != EXPECTED_SPLIT_COUNTS[EVALUATION_SPLIT]:
        raise AssertionError("B1 validation prediction count is incorrect.")
    if not np.isfinite(validation_scores).all():
        raise AssertionError("B1 validation predictions contain non-finite values.")

    metrics = evaluate_validation(y_validation, validation_scores)
    metrics.update(
        {
            "model": "B1_relational_lightgbm",
            "relation": RELATION_NAME,
            "weighting": "weighted",
            "scale_pos_weight": float(scale_pos_weight),
            "maximum_estimators": MAX_ESTIMATORS,
            "actual_stopping_iteration": int(
                learning_curve_summary["actual_stopping_iteration"]
            ),
            "best_iteration": int(model.best_iteration_),
            "best_validation_average_precision": float(
                learning_curve_summary["best_validation_average_precision"]
            ),
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "early_stopping_triggered": bool(
                learning_curve_summary["early_stopping_triggered"]
            ),
            "estimator_cap_reached": bool(
                learning_curve_summary["estimator_cap_reached"]
            ),
        }
    )
    comparison = build_comparison_to_b0(b0_metrics, metrics)
    feature_importance = build_feature_importance(model)
    if len(feature_importance) != len(b1_features):
        raise AssertionError("B1 feature importance does not match its manifest.")
    validation_predictions = validation_metadata.copy()
    validation_predictions["prediction"] = validation_scores

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH), num_iteration=model.best_iteration_)
    write_json(METRICS_PATH, metrics)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    validation_predictions.to_parquet(
        VALIDATION_PREDICTIONS_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )
    learning_curve.to_csv(LEARNING_CURVE_PATH, index=False)
    comparison.to_csv(COMPARISON_PATH, index=False)

    metadata = build_b1_metadata(
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
    )
    write_json(METADATA_PATH, metadata)
    assert_b0_artifacts_unchanged(b0_hashes_before)

    expected_artifacts = [
        MODEL_PATH,
        METRICS_PATH,
        METADATA_PATH,
        FEATURE_IMPORTANCE_PATH,
        VALIDATION_PREDICTIONS_PATH,
        LEARNING_CURVE_PATH,
        COMPARISON_PATH,
    ]
    missing_artifacts = [str(path) for path in expected_artifacts if not path.exists()]
    if missing_artifacts:
        raise OSError(f"B1 artifacts were not created: {missing_artifacts}.")

    print(f"Best iteration: {model.best_iteration_:,}")
    print(
        "Actual stopping iteration: "
        f"{learning_curve_summary['actual_stopping_iteration']:,}"
    )
    print(
        "Early stopping triggered: "
        f"{'YES' if learning_curve_summary['early_stopping_triggered'] else 'NO'}"
    )
    print(f"Validation PR-AUC: {metrics['pr_auc']:.12f}")
    print(f"Validation ROC-AUC: {metrics['roc_auc']:.12f}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Reports saved: {REPORT_DIR}")
    print("Frozen B0 artifacts unchanged: YES")
    print("Final test evaluated: NO")


if __name__ == "__main__":
    main()
