from __future__ import annotations

import argparse
import hashlib
import json
import platform
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMClassifier
from pandas.api.types import (
    is_bool_dtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)
from sklearn.metrics import average_precision_score, roc_auc_score


ROOT_DIR = Path(__file__).resolve().parents[2]

MODEL_DATASET_PATH = ROOT_DIR / "data" / "processed" / "model_dataset.parquet"
SPLIT_MANIFEST_PATH = (
    ROOT_DIR / "data" / "processed" / "split_assignment.parquet"
)
MODEL_PATH = ROOT_DIR / "models" / "lightgbm_baseline.txt"
REPORT_DIR = ROOT_DIR / "reports" / "baseline"
METRICS_PATH = REPORT_DIR / "lightgbm_metrics.json"
FEATURE_IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"
VALIDATION_PREDICTIONS_PATH = REPORT_DIR / "validation_predictions.parquet"
METADATA_PATH = REPORT_DIR / "baseline_metadata.json"
CATEGORY_MAPPINGS_PATH = REPORT_DIR / "categorical_mappings.json"
EXPERIMENTS_DIR = REPORT_DIR / "experiments"

EXPECTED_ROWS = 590_540
EXPECTED_SPLIT_COUNTS = {
    "train": 413_378,
    "validation": 88_581,
    "test": 88_581,
}
RANDOM_SEED = 42
TOP_FRACTIONS = [0.005, 0.01, 0.02, 0.05]
DERIVED_TIME_FEATURES = [
    "elapsed_days",
    "hour_in_day",
    "day_in_week_cycle",
]
MISSING_TOKEN = "__MISSING__"
UNKNOWN_TOKEN = "__UNKNOWN__"

# has_identity is retained only so identity coverage remains auditable.
FORBIDDEN_FEATURE_COLUMNS = {
    "TransactionID",
    "isFraud",
    "split",
    "has_identity",
}
ACCIDENTAL_MERGE_COLUMNS = {
    "TransactionID_x",
    "TransactionID_y",
    "TransactionDT_x",
    "TransactionDT_y",
    "isFraud_x",
    "isFraud_y",
    "split_x",
    "split_y",
}
CategoryMapping = dict[str, int]
CategoryMappings = dict[str, CategoryMapping]


@dataclass(frozen=True)
class RunConfig:
    """The only model settings configurable in the controlled comparison."""

    run_name: str
    weighting: str
    n_estimators: int = 6_000
    early_stopping_rounds: int = 200
    log_evaluation_period: int = 50

    def validate(self) -> None:
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.run_name):
            raise ValueError(
                "run_name must contain only lowercase letters, numbers, "
                "underscores, or hyphens."
            )
        if self.weighting not in {"weighted", "unweighted"}:
            raise ValueError("weighting must be 'weighted' or 'unweighted'.")
        if self.n_estimators <= 0:
            raise ValueError("n_estimators must be positive.")
        if self.early_stopping_rounds <= 0:
            raise ValueError("early_stopping_rounds must be positive.")
        if self.log_evaluation_period <= 0:
            raise ValueError("log_evaluation_period must be positive.")


@dataclass(frozen=True)
class ArtifactPaths:
    report_dir: Path
    model: Path
    metrics: Path
    feature_importance: Path
    validation_predictions: Path
    metadata: Path
    learning_curve: Path


def build_artifact_paths(run_name: str) -> ArtifactPaths:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", run_name):
        raise ValueError(f"Unsafe run name: {run_name!r}.")
    report_dir = EXPERIMENTS_DIR / run_name
    return ArtifactPaths(
        report_dir=report_dir,
        model=ROOT_DIR / "models" / f"lightgbm_baseline_{run_name}.txt",
        metrics=report_dir / "metrics.json",
        feature_importance=report_dir / "feature_importance.csv",
        validation_predictions=report_dir / "validation_predictions.parquet",
        metadata=report_dir / "metadata.json",
        learning_curve=report_dir / "learning_curve.csv",
    )


def repository_relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT_DIR).as_posix()


def split_counts(split_series: pd.Series) -> dict[str, int]:
    return {
        str(name): int(count)
        for name, count in split_series.astype("string").value_counts().items()
    }


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {sorted(missing)}."
        )


def validate_split_counts(df: pd.DataFrame, source_name: str) -> None:
    require_columns(df, {"TransactionID", "split"}, source_name)
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"{source_name} must contain {EXPECTED_ROWS:,} rows; "
            f"got {len(df):,}."
        )
    if df["TransactionID"].isna().any():
        raise ValueError(f"{source_name} contains missing TransactionID values.")
    if not df["TransactionID"].is_unique:
        raise ValueError(f"{source_name} contains duplicate TransactionID values.")
    if df["split"].isna().any():
        raise ValueError(f"{source_name} contains missing split assignments.")

    actual_counts = split_counts(df["split"])
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Unexpected split counts in {source_name}. "
            f"Expected {EXPECTED_SPLIT_COUNTS}, got {actual_counts}."
        )


def validate_dataset_against_manifest(dataset_index: pd.DataFrame) -> None:
    if not SPLIT_MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Frozen split manifest not found: {SPLIT_MANIFEST_PATH}"
        )

    manifest = pd.read_parquet(
        SPLIT_MANIFEST_PATH,
        columns=["TransactionID", "split"],
    )
    validate_split_counts(manifest, "split_assignment.parquet")

    left = dataset_index[["TransactionID", "split"]].copy()
    right = manifest[["TransactionID", "split"]].copy()
    left["split"] = left["split"].astype("string")
    right["split"] = right["split"].astype("string")
    comparison = left.merge(
        right,
        on="TransactionID",
        how="outer",
        suffixes=("_dataset", "_manifest"),
        indicator=True,
        validate="one_to_one",
    )

    membership_mismatches = int((comparison["_merge"] != "both").sum())
    split_mismatches = int(
        (comparison["split_dataset"] != comparison["split_manifest"]).sum()
    )
    if membership_mismatches or split_mismatches:
        raise ValueError(
            "Model dataset does not match the authoritative split manifest: "
            f"TransactionID mismatches={membership_mismatches:,}, "
            f"split mismatches={split_mismatches:,}."
        )


def load_model_dataset() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Load only train/validation predictors; never load final-test labels."""

    if not MODEL_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Model dataset not found. Run src/data/build_model_dataset.py first: "
            f"{MODEL_DATASET_PATH}"
        )

    dataset_index = pd.read_parquet(
        MODEL_DATASET_PATH,
        columns=["TransactionID", "split", "has_identity"],
    )
    validate_split_counts(dataset_index, "model_dataset.parquet")
    validate_dataset_against_manifest(dataset_index)

    if dataset_index["has_identity"].isna().any():
        raise ValueError("model_dataset.parquet contains missing has_identity values.")
    identity_matched = int(dataset_index["has_identity"].astype(bool).sum())
    dataset_summary = {
        "n_rows": int(len(dataset_index)),
        "n_identity_matched": identity_matched,
        "n_identity_missing": int(len(dataset_index) - identity_matched),
        "identity_coverage_pct": float(identity_matched / len(dataset_index) * 100.0),
    }
    del dataset_index

    train_df = pd.read_parquet(
        MODEL_DATASET_PATH,
        filters=[("split", "==", "train")],
    ).reset_index(drop=True)
    validation_df = pd.read_parquet(
        MODEL_DATASET_PATH,
        filters=[("split", "==", "validation")],
    ).reset_index(drop=True)

    if len(train_df) != EXPECTED_SPLIT_COUNTS["train"]:
        raise ValueError(
            "Filtered train partition has an unexpected row count: "
            f"{len(train_df):,}."
        )
    if len(validation_df) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise ValueError(
            "Filtered validation partition has an unexpected row count: "
            f"{len(validation_df):,}."
        )
    if not train_df["split"].astype("string").eq("train").all():
        raise ValueError("Filtered train data contains non-train rows.")
    if not validation_df["split"].astype("string").eq("validation").all():
        raise ValueError("Filtered validation data contains non-validation rows.")

    required = {
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "split",
        "has_identity",
        *DERIVED_TIME_FEATURES,
    }
    require_columns(train_df, required, "train model partition")
    require_columns(validation_df, required, "validation model partition")
    if list(train_df.columns) != list(validation_df.columns):
        raise ValueError("Train and validation model columns are not identical.")

    return train_df, validation_df, dataset_summary


def get_feature_columns(columns: list[str]) -> list[str]:
    feature_columns = [
        column for column in columns if column not in FORBIDDEN_FEATURE_COLUMNS
    ]
    assert_no_forbidden_features(feature_columns)
    return feature_columns


def assert_no_forbidden_features(feature_columns: list[str]) -> None:
    leaked = FORBIDDEN_FEATURE_COLUMNS & set(feature_columns)
    if leaked:
        raise AssertionError(
            f"Forbidden columns present in model features: {sorted(leaked)}."
        )

    accidental = ACCIDENTAL_MERGE_COLUMNS & set(feature_columns)
    if accidental:
        raise AssertionError(
            f"Accidental merge columns present in model features: {sorted(accidental)}."
        )

def is_categorical_dtype(dtype: object) -> bool:
    return (
        is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def identify_categorical_columns(df: pd.DataFrame) -> list[str]:
    return [
        column
        for column in df.columns
        if is_categorical_dtype(df[column].dtype)
    ]


def fit_category_mapping(train_series: pd.Series) -> CategoryMapping:
    """Fit a deterministic vocabulary using one training series only."""

    values = train_series.astype("string")
    train_categories = sorted(str(value) for value in values.dropna().unique())
    reserved_collisions = {MISSING_TOKEN, UNKNOWN_TOKEN} & set(train_categories)
    if reserved_collisions:
        raise ValueError(
            "Training categories collide with reserved preprocessing tokens: "
            f"{sorted(reserved_collisions)}."
        )

    mapping: CategoryMapping = {
        MISSING_TOKEN: 0,
        UNKNOWN_TOKEN: 1,
    }
    for category in train_categories:
        mapping[category] = len(mapping)
    return mapping


def apply_category_mapping(
    series: pd.Series,
    mapping: CategoryMapping,
) -> pd.Series:
    if MISSING_TOKEN not in mapping or UNKNOWN_TOKEN not in mapping:
        raise ValueError("Category mapping lacks reserved missing/unknown codes.")

    values = series.astype("string").fillna(MISSING_TOKEN)
    return (
        values.map(mapping)
        .fillna(mapping[UNKNOWN_TOKEN])
        .astype("int32")
    )


def fit_and_apply_categorical_mappings(
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    categorical_columns: list[str],
) -> CategoryMappings:
    """Fit each mapping on train, then reuse it unchanged on validation."""

    mappings: CategoryMappings = {}
    for column in categorical_columns:
        mapping = fit_category_mapping(X_train[column])
        mappings[column] = mapping
        X_train[column] = apply_category_mapping(X_train[column], mapping)
        X_validation[column] = apply_category_mapping(
            X_validation[column], mapping
        )
    return mappings


def assert_supported_model_dtypes(df: pd.DataFrame, source_name: str) -> None:
    unprocessed = [
        column
        for column in df.columns
        if is_categorical_dtype(df[column].dtype)
    ]
    if unprocessed:
        raise TypeError(
            f"Unprocessed categorical columns remain in {source_name}: {unprocessed}."
        )

    unsupported = [
        column
        for column in df.columns
        if not is_numeric_dtype(df[column].dtype)
        and not is_bool_dtype(df[column].dtype)
    ]
    if unsupported:
        raise TypeError(
            f"Unsupported model dtypes remain in {source_name}: {unsupported}."
        )


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    n_positive = int(y_train.sum())
    n_negative = int(len(y_train) - n_positive)
    if n_positive <= 0 or n_negative <= 0:
        raise ValueError("Training labels must contain both binary classes.")
    return float(n_negative / n_positive)


def resolve_scale_pos_weight(
    weighting: str,
    y_train: pd.Series,
) -> float:
    if weighting == "weighted":
        return calculate_scale_pos_weight(y_train)
    if weighting == "unweighted":
        return 1.0
    raise ValueError("weighting must be 'weighted' or 'unweighted'.")


def build_lightgbm_model(
    scale_pos_weight: float,
    n_estimators: int = 3_000,
) -> LGBMClassifier:
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")
    if scale_pos_weight <= 0:
        raise ValueError("scale_pos_weight must be positive.")
    return LGBMClassifier(
        objective="binary",
        learning_rate=0.03,
        n_estimators=n_estimators,
        num_leaves=64,
        max_depth=-1,
        min_child_samples=50,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
        n_jobs=-1,
        deterministic=True,
        force_col_wise=True,
        verbosity=-1,
    )


def precision_recall_at_fraction(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    fraction: float,
) -> dict[str, float | int]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")

    labels = np.asarray(y_true)
    scores = np.asarray(probabilities)
    if labels.ndim != 1 or scores.ndim != 1:
        raise ValueError("y_true and probabilities must be one-dimensional.")
    if len(labels) == 0 or len(labels) != len(scores):
        raise ValueError("y_true and probabilities must have equal non-zero length.")
    if not np.isfinite(scores).all():
        raise ValueError("probabilities contain non-finite values.")

    k = max(1, int(np.ceil(len(labels) * fraction)))
    order = np.argsort(scores, kind="mergesort")[::-1]
    top_labels = labels[order[:k]]
    frauds_found = int(top_labels.sum())
    total_frauds = int(labels.sum())
    return {
        "fraction": float(fraction),
        "k": int(k),
        "frauds_found": frauds_found,
        "precision": float(frauds_found / k),
        "recall": float(frauds_found / total_frauds if total_frauds else 0.0),
    }


def fraction_metric_name(fraction: float) -> str:
    percentage = fraction * 100
    formatted = f"{percentage:g}"
    return f"top_{formatted}_pct"


def evaluate_validation(
    y_validation: pd.Series,
    probabilities: np.ndarray,
    run_name: str | None = None,
) -> dict[str, Any]:
    labels = y_validation.to_numpy(dtype=np.int8, copy=False)
    pr_auc = float(average_precision_score(labels, probabilities))
    roc_auc = float(roc_auc_score(labels, probabilities))
    if not np.isfinite(pr_auc) or not np.isfinite(roc_auc):
        raise ValueError("Validation PR-AUC or ROC-AUC is not finite.")

    ranking_metrics = {
        fraction_metric_name(fraction): precision_recall_at_fraction(
            labels,
            probabilities,
            fraction,
        )
        for fraction in TOP_FRACTIONS
    }
    if set(ranking_metrics) != {
        "top_0.5_pct",
        "top_1_pct",
        "top_2_pct",
        "top_5_pct",
    }:
        raise AssertionError("Precision@K metric keys were not generated correctly.")

    metrics = {
        "model": "lightgbm_tabular_baseline",
        "evaluation_split": "validation",
        "n_transactions": int(len(labels)),
        "n_fraud": int(labels.sum()),
        "fraud_rate": float(labels.mean()),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "ranking_metrics": ranking_metrics,
        "test_evaluated": False,
    }
    if run_name is not None:
        metrics["run_name"] = run_name
    return metrics


def build_feature_importance(model: LGBMClassifier) -> pd.DataFrame:
    booster = model.booster_
    importance = pd.DataFrame(
        {
            "feature": booster.feature_name(),
            "importance_gain": booster.feature_importance(
                importance_type="gain"
            ),
            "importance_split": booster.feature_importance(
                importance_type="split"
            ),
        }
    )
    return importance.sort_values(
        ["importance_gain", "importance_split", "feature"],
        ascending=[False, False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def save_category_mappings(
    mappings: CategoryMappings,
    path: Path = CATEGORY_MAPPINGS_PATH,
) -> None:
    payload: dict[str, Any] = {
        "fit_split": "train",
        "missing_token": MISSING_TOKEN,
        "unknown_token": UNKNOWN_TOKEN,
        "columns": mappings,
    }
    write_json(path, payload)


def validate_category_mappings_match_reference(
    mappings: CategoryMappings,
    path: Path = CATEGORY_MAPPINGS_PATH,
) -> str:
    """Ensure finalization runs reuse the exact established preprocessing."""

    if not path.exists():
        raise FileNotFoundError(
            f"Reference categorical mappings not found: {path}"
        )
    with path.open(encoding="utf-8") as handle:
        reference = json.load(handle)

    if reference.get("fit_split") != "train":
        raise ValueError("Reference category mappings were not fitted on train.")
    if reference.get("missing_token") != MISSING_TOKEN:
        raise ValueError("Reference categorical missing token changed.")
    if reference.get("unknown_token") != UNKNOWN_TOKEN:
        raise ValueError("Reference categorical unknown token changed.")
    if reference.get("columns") != mappings:
        raise ValueError(
            "Train-fitted categorical mappings differ from the reference baseline."
        )

    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_learning_curve(
    evaluation_results: dict[str, dict[str, list[float]]],
) -> pd.DataFrame:
    if set(evaluation_results) != {"validation"}:
        raise ValueError(
            "Expected exactly one validation evaluation history; got "
            f"{sorted(evaluation_results)}."
        )
    history = evaluation_results["validation"]
    required_metrics = {"average_precision", "auc"}
    missing = required_metrics - set(history)
    if missing:
        raise ValueError(
            f"Validation learning curve lacks metrics: {sorted(missing)}."
        )

    lengths = {len(values) for values in history.values()}
    if len(lengths) != 1 or not lengths or next(iter(lengths)) <= 0:
        raise ValueError("Validation learning-curve metric lengths are invalid.")
    n_iterations = next(iter(lengths))
    curve = pd.DataFrame({"iteration": np.arange(1, n_iterations + 1)})
    for metric_name, values in history.items():
        curve[f"validation_{metric_name}"] = values
    return curve


def summarize_learning_curve(
    learning_curve: pd.DataFrame,
    best_iteration: int,
    maximum_estimators: int,
) -> dict[str, Any]:
    actual_stopping_iteration = int(learning_curve["iteration"].iloc[-1])
    ap = learning_curve["validation_average_precision"]
    best_logged_iteration = int(ap.idxmax() + 1)
    if best_logged_iteration != best_iteration:
        raise AssertionError(
            "LightGBM best_iteration differs from the recorded validation-AP "
            f"maximum: model={best_iteration}, history={best_logged_iteration}."
        )

    def gain_over_window(window: int) -> float | None:
        if len(ap) <= window:
            return None
        return float(ap.iloc[-1] - ap.iloc[-window - 1])

    estimator_cap_reached = actual_stopping_iteration == maximum_estimators
    return {
        "maximum_estimators": int(maximum_estimators),
        "actual_stopping_iteration": actual_stopping_iteration,
        "best_iteration": int(best_iteration),
        "best_validation_average_precision": float(ap.iloc[best_iteration - 1]),
        "final_logged_validation_average_precision": float(ap.iloc[-1]),
        "estimator_cap_reached": estimator_cap_reached,
        "early_stopping_triggered": not estimator_cap_reached,
        "validation_ap_gain_last_200_rounds": gain_over_window(200),
        "validation_ap_gain_last_500_rounds": gain_over_window(500),
    }


def build_metadata(
    *,
    config: RunConfig,
    paths: ArtifactPaths,
    model: LGBMClassifier,
    dataset_summary: dict[str, Any],
    train_rows: int,
    validation_rows: int,
    train_fraud: int,
    validation_fraud: int,
    feature_columns: list[str],
    categorical_columns: list[str],
    scale_pos_weight: float,
    category_mappings_sha256: str,
    learning_curve_summary: dict[str, Any],
    validation_metrics: dict[str, Any],
) -> dict[str, Any]:
    numeric_columns = [
        column for column in feature_columns if column not in categorical_columns
    ]
    return {
        "model_name": "lightgbm_tabular_baseline_candidate",
        "run_name": config.run_name,
        "weighting": config.weighting,
        "random_seed": RANDOM_SEED,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "source_model_dataset_path": repository_relative(MODEL_DATASET_PATH),
        "source_split_manifest_path": repository_relative(SPLIT_MANIFEST_PATH),
        "model_dataset_rows": int(dataset_summary["n_rows"]),
        "model_dataset_columns": int(len(feature_columns) + len(FORBIDDEN_FEATURE_COLUMNS)),
        "split_counts": EXPECTED_SPLIT_COUNTS,
        "training_rows": int(train_rows),
        "validation_rows": int(validation_rows),
        "training_fraud_count": int(train_fraud),
        "training_fraud_rate": float(train_fraud / train_rows),
        "validation_fraud_count": int(validation_fraud),
        "validation_fraud_rate": float(validation_fraud / validation_rows),
        "identity_coverage": {
            "n_identity_matched": int(dataset_summary["n_identity_matched"]),
            "n_identity_missing": int(dataset_summary["n_identity_missing"]),
            "identity_coverage_pct": float(dataset_summary["identity_coverage_pct"]),
            "has_identity_used_as_predictor": False,
        },
        "number_of_predictors": int(len(feature_columns)),
        "number_of_numeric_predictors": int(len(numeric_columns)),
        "number_of_categorical_predictors": int(len(categorical_columns)),
        "feature_columns": feature_columns,
        "numeric_feature_columns": numeric_columns,
        "categorical_feature_columns": categorical_columns,
        "categorical_preprocessing_policy": (
            "Mappings are fitted on TRAIN only. Missing values map to "
            f"{MISSING_TOKEN}; unseen validation values map to {UNKNOWN_TOKEN}. "
            "Encoded columns are declared categorical to LightGBM."
        ),
        "categorical_mappings_path": repository_relative(CATEGORY_MAPPINGS_PATH),
        "categorical_mappings_sha256": category_mappings_sha256,
        "numeric_missing_value_policy": (
            "Numerical NaN values are preserved for native LightGBM handling; "
            "no global imputation is applied."
        ),
        "transaction_dt_policy": (
            "Raw TransactionDT is retained as elapsed seconds and may capture "
            "temporal drift; no real calendar origin is inferred."
        ),
        "scale_pos_weight": float(scale_pos_weight),
        "lightgbm_parameters": model.get_params(deep=False),
        "evaluation_metrics": ["average_precision", "auc"],
        "n_estimators_cap": int(config.n_estimators),
        "maximum_estimators": int(config.n_estimators),
        "early_stopping_rounds": int(config.early_stopping_rounds),
        "early_stopping_metric": "average_precision",
        "early_stopping_first_metric_only": True,
        "best_iteration": int(model.best_iteration_),
        "actual_stopping_iteration": int(
            learning_curve_summary["actual_stopping_iteration"]
        ),
        "best_validation_average_precision": float(
            learning_curve_summary["best_validation_average_precision"]
        ),
        "estimator_cap_reached": bool(
            learning_curve_summary["estimator_cap_reached"]
        ),
        "early_stopping_triggered": bool(
            learning_curve_summary["early_stopping_triggered"]
        ),
        "learning_curve_summary": learning_curve_summary,
        "validation_pr_auc": float(validation_metrics["pr_auc"]),
        "validation_roc_auc": float(validation_metrics["roc_auc"]),
        "validation_ranking_metrics": validation_metrics["ranking_metrics"],
        "derived_time_features": DERIVED_TIME_FEATURES,
        "forbidden_feature_columns": sorted(FORBIDDEN_FEATURE_COLUMNS),
        "relational_or_graph_features_used": False,
        "validation_metrics_path": repository_relative(paths.metrics),
        "validation_predictions_path": repository_relative(
            paths.validation_predictions
        ),
        "feature_importance_path": repository_relative(paths.feature_importance),
        "learning_curve_path": repository_relative(paths.learning_curve),
        "model_path": repository_relative(paths.model),
        "selected_as_b0": False,
        "test_evaluated": False,
        "test_policy": (
            "Final test labels were not loaded or used during baseline training, "
            "preprocessing, class weighting, model selection, or development evaluation."
        ),
    }


def print_training_summary(
    *,
    train_rows: int,
    validation_rows: int,
    train_fraud: int,
    validation_fraud: int,
    n_predictors: int,
    n_categorical: int,
    scale_pos_weight: float,
) -> None:
    print(f"\nTrain rows: {train_rows:,}")
    print(f"Validation rows: {validation_rows:,}")
    print(f"\nTrain fraud: {train_fraud:,} / {train_rows:,}")
    print(f"Train fraud rate: {train_fraud / train_rows:.6%}")
    print(
        f"Validation fraud: {validation_fraud:,} / {validation_rows:,}"
    )
    print(
        f"Validation fraud rate: {validation_fraud / validation_rows:.6%}"
    )
    print(f"\nPredictors: {n_predictors:,}")
    print(f"Categorical predictors: {n_categorical:,}")
    print(f"Numeric predictors: {n_predictors - n_categorical:,}")
    print(f"scale_pos_weight: {scale_pos_weight:.10f}")


def print_validation_results(
    metrics: dict[str, Any],
    learning_curve_summary: dict[str, Any],
) -> None:
    best_iteration = int(learning_curve_summary["best_iteration"])
    print(f"\nBest iteration: {best_iteration:,}")
    print(
        "Actual stopping iteration: "
        f"{learning_curve_summary['actual_stopping_iteration']:,}"
    )
    print(
        "Estimator cap reached: "
        f"{'YES' if learning_curve_summary['estimator_cap_reached'] else 'NO'}"
    )
    print(
        "Early stopping triggered: "
        f"{'YES' if learning_curve_summary['early_stopping_triggered'] else 'NO'}"
    )
    print("\nValidation results:")
    print(f"PR-AUC: {metrics['pr_auc']:.8f}")
    print(f"ROC-AUC: {metrics['roc_auc']:.8f}")

    for name, result in metrics["ranking_metrics"].items():
        label = name.removeprefix("top_").removesuffix("_pct")
        print(f"\nTop {label}%:")
        print(f"  k = {result['k']:,}")
        print(f"  frauds found = {result['frauds_found']:,}")
        print(f"  precision = {result['precision']:.8f}")
        print(f"  recall = {result['recall']:.8f}")


def parse_args(argv: list[str] | None = None) -> RunConfig:
    parser = argparse.ArgumentParser(
        description=(
            "Train one controlled LightGBM baseline-finalization candidate."
        )
    )
    parser.add_argument("--run-name", required=True)
    parser.add_argument(
        "--weighting",
        required=True,
        choices=["weighted", "unweighted"],
        help=(
            "weighted uses the train-only negatives/positives ratio; "
            "unweighted uses scale_pos_weight=1.0"
        ),
    )
    parser.add_argument("--n-estimators", type=int, default=6_000)
    parser.add_argument("--early-stopping-rounds", type=int, default=200)
    parser.add_argument("--log-evaluation-period", type=int, default=50)
    args = parser.parse_args(argv)
    config = RunConfig(
        run_name=args.run_name,
        weighting=args.weighting,
        n_estimators=args.n_estimators,
        early_stopping_rounds=args.early_stopping_rounds,
        log_evaluation_period=args.log_evaluation_period,
    )
    config.validate()
    return config


def main(argv: list[str] | None = None) -> None:
    config = parse_args(argv)
    paths = build_artifact_paths(config.run_name)
    print(f"Loading modeling dataset: {MODEL_DATASET_PATH}")
    print(
        f"Controlled run: {config.run_name} "
        f"({config.weighting}, cap={config.n_estimators:,}, "
        f"patience={config.early_stopping_rounds:,})"
    )
    train_df, validation_df, dataset_summary = load_model_dataset()

    feature_columns = get_feature_columns(list(train_df.columns))
    validation_metadata = validation_df[
        ["TransactionID", "TransactionDT", "isFraud"]
    ].copy()
    y_train = train_df["isFraud"].astype("int8").copy()
    y_validation = validation_df["isFraud"].astype("int8").copy()
    if set(y_train.unique()) != {0, 1} or set(y_validation.unique()) != {0, 1}:
        raise ValueError("Train and validation labels must each contain both classes.")

    X_train = train_df.drop(columns=sorted(FORBIDDEN_FEATURE_COLUMNS))
    X_validation = validation_df.drop(columns=sorted(FORBIDDEN_FEATURE_COLUMNS))
    del train_df, validation_df

    if list(X_train.columns) != feature_columns:
        raise AssertionError("Training feature order changed during target separation.")
    if list(X_validation.columns) != feature_columns:
        raise AssertionError("Validation feature order differs from training.")
    assert_no_forbidden_features(list(X_train.columns))

    categorical_columns = identify_categorical_columns(X_train)
    validation_categorical_columns = identify_categorical_columns(X_validation)
    if categorical_columns != validation_categorical_columns:
        raise TypeError("Train and validation categorical dtypes do not agree.")

    print("Fitting categorical vocabularies on TRAIN only...")
    category_mappings = fit_and_apply_categorical_mappings(
        X_train,
        X_validation,
        categorical_columns,
    )
    category_mappings_sha256 = validate_category_mappings_match_reference(
        category_mappings
    )
    assert_supported_model_dtypes(X_train, "training predictors")
    assert_supported_model_dtypes(X_validation, "validation predictors")

    scale_pos_weight = resolve_scale_pos_weight(config.weighting, y_train)
    train_fraud = int(y_train.sum())
    validation_fraud = int(y_validation.sum())
    if config.weighting == "weighted":
        expected_weight = (len(y_train) - train_fraud) / train_fraud
        if not np.isclose(scale_pos_weight, expected_weight):
            raise AssertionError("Weighted run did not use the train-only class ratio.")
    elif scale_pos_weight != 1.0:
        raise AssertionError("Unweighted run must use scale_pos_weight=1.0.")
    print_training_summary(
        train_rows=len(X_train),
        validation_rows=len(X_validation),
        train_fraud=train_fraud,
        validation_fraud=validation_fraud,
        n_predictors=len(feature_columns),
        n_categorical=len(categorical_columns),
        scale_pos_weight=scale_pos_weight,
    )

    model = build_lightgbm_model(
        scale_pos_weight,
        n_estimators=config.n_estimators,
    )
    evaluation_results: dict[str, dict[str, list[float]]] = {}
    callbacks = [
        # Weighted binary log-loss is useful to log, but scale_pos_weight makes
        # it a poor stopping criterion for this ranking baseline. Stop on the
        # first metric below, the brief's primary metric: average precision.
        lightgbm.early_stopping(
            stopping_rounds=config.early_stopping_rounds,
            first_metric_only=True,
        ),
        lightgbm.record_evaluation(evaluation_results),
        lightgbm.log_evaluation(period=config.log_evaluation_period),
    ]
    print("\nTraining LightGBM...")
    model.fit(
        X_train,
        y_train,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_names=["validation"],
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
        maximum_estimators=config.n_estimators,
    )

    validation_probabilities = model.predict_proba(
        X_validation,
        num_iteration=model.best_iteration_,
    )[:, 1]
    if len(validation_probabilities) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise AssertionError(
            "Unexpected validation prediction count: "
            f"{len(validation_probabilities):,}."
        )
    if not np.isfinite(validation_probabilities).all():
        raise AssertionError("Validation predictions contain non-finite values.")
    if not (
        (validation_probabilities >= 0).all()
        and (validation_probabilities <= 1).all()
    ):
        raise AssertionError("Validation predictions are outside [0, 1].")

    metrics = evaluate_validation(
        y_validation,
        validation_probabilities,
        run_name=config.run_name,
    )
    metrics.update(
        {
            "weighting": config.weighting,
            "scale_pos_weight": float(scale_pos_weight),
            "maximum_estimators": int(config.n_estimators),
            "actual_stopping_iteration": int(
                learning_curve_summary["actual_stopping_iteration"]
            ),
            "best_iteration": int(model.best_iteration_),
            "best_validation_average_precision": float(
                learning_curve_summary["best_validation_average_precision"]
            ),
            "early_stopping_rounds": int(config.early_stopping_rounds),
            "early_stopping_triggered": bool(
                learning_curve_summary["early_stopping_triggered"]
            ),
            "estimator_cap_reached": bool(
                learning_curve_summary["estimator_cap_reached"]
            ),
        }
    )
    feature_importance = build_feature_importance(model)
    if len(feature_importance) != len(feature_columns):
        raise AssertionError("Feature-importance rows do not match predictors.")

    validation_predictions = validation_metadata.copy()
    validation_predictions["prediction"] = validation_probabilities
    if len(validation_predictions) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise AssertionError("Validation-prediction metadata is misaligned.")

    paths.model.parent.mkdir(parents=True, exist_ok=True)
    paths.report_dir.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(
        str(paths.model),
        num_iteration=model.best_iteration_,
    )
    write_json(paths.metrics, metrics)
    feature_importance.to_csv(paths.feature_importance, index=False)
    validation_predictions.to_parquet(
        paths.validation_predictions,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )
    learning_curve.to_csv(paths.learning_curve, index=False)

    metadata = build_metadata(
        config=config,
        paths=paths,
        model=model,
        dataset_summary=dataset_summary,
        train_rows=len(X_train),
        validation_rows=len(X_validation),
        train_fraud=train_fraud,
        validation_fraud=validation_fraud,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        scale_pos_weight=scale_pos_weight,
        category_mappings_sha256=category_mappings_sha256,
        learning_curve_summary=learning_curve_summary,
        validation_metrics=metrics,
    )
    write_json(paths.metadata, metadata)

    expected_artifacts = [
        paths.model,
        paths.metrics,
        paths.feature_importance,
        paths.validation_predictions,
        paths.metadata,
        paths.learning_curve,
        CATEGORY_MAPPINGS_PATH,
    ]
    missing_artifacts = [
        str(path) for path in expected_artifacts if not path.exists()
    ]
    if missing_artifacts:
        raise OSError(f"Expected artifacts were not created: {missing_artifacts}")

    print_validation_results(metrics, learning_curve_summary)
    print(f"\nModel saved: {paths.model}")
    print(f"Metrics saved: {paths.metrics}")
    print(f"Feature importance saved: {paths.feature_importance}")
    print(f"Validation predictions saved: {paths.validation_predictions}")
    print(f"Metadata saved: {paths.metadata}")
    print(f"Learning curve saved: {paths.learning_curve}")
    print(f"Categorical mappings verified unchanged: {CATEGORY_MAPPINGS_PATH}")
    print("\nTest set evaluated: NO")


if __name__ == "__main__":
    main()
