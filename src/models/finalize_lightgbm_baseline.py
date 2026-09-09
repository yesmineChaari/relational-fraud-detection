from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

try:
    from .train_lightgbm_baseline import (
        FEATURE_IMPORTANCE_PATH,
        METADATA_PATH,
        METRICS_PATH,
        MODEL_PATH,
        REPORT_DIR,
        ROOT_DIR,
        VALIDATION_PREDICTIONS_PATH,
        build_artifact_paths,
        precision_recall_at_fraction,
        repository_relative,
        write_json,
    )
except ImportError:  # Direct script execution.
    from train_lightgbm_baseline import (
        FEATURE_IMPORTANCE_PATH,
        METADATA_PATH,
        METRICS_PATH,
        MODEL_PATH,
        REPORT_DIR,
        ROOT_DIR,
        VALIDATION_PREDICTIONS_PATH,
        build_artifact_paths,
        precision_recall_at_fraction,
        repository_relative,
        write_json,
    )


WEIGHTED_RUN = "weighted_6000"
UNWEIGHTED_RUN = "unweighted_6000"
ORIGINAL_RUN = "original_3000_weighted"

COMPARISON_PATH = REPORT_DIR / "baseline_comparison.csv"
ORIGINAL_REPORT_DIR = REPORT_DIR / "experiments" / ORIGINAL_RUN
ORIGINAL_MODEL_PATH = ROOT_DIR / "models" / f"lightgbm_baseline_{ORIGINAL_RUN}.txt"

ORIGINAL_METRICS = {
    "pr_auc": 0.6319325654367357,
    "roc_auc": 0.9244221422728391,
    "best_iteration": 3_000,
    "scale_pos_weight": 27.434310083918007,
}


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def normalize_original_archive_metadata(path: Path) -> None:
    metadata = read_json(path)
    metadata.update(
        {
            "run_name": ORIGINAL_RUN,
            "archived_original_baseline": True,
            "validation_metrics_path": repository_relative(ORIGINAL_REPORT_DIR / "metrics.json"),
            "validation_predictions_path": repository_relative(
                ORIGINAL_REPORT_DIR / "validation_predictions.parquet"
            ),
            "feature_importance_path": repository_relative(
                ORIGINAL_REPORT_DIR / "feature_importance.csv"
            ),
            "model_path": repository_relative(ORIGINAL_MODEL_PATH),
            "selected_as_b0": False,
            "test_evaluated": False,
        }
    )
    write_json(path, metadata)


def archive_original_baseline() -> None:
    """Preserve the original 3000-tree evidence before canonical replacement."""

    archived_metadata = ORIGINAL_REPORT_DIR / "metadata.json"
    archive_targets = {
        METRICS_PATH: ORIGINAL_REPORT_DIR / "metrics.json",
        METADATA_PATH: archived_metadata,
        VALIDATION_PREDICTIONS_PATH: (ORIGINAL_REPORT_DIR / "validation_predictions.parquet"),
        FEATURE_IMPORTANCE_PATH: ORIGINAL_REPORT_DIR / "feature_importance.csv",
        MODEL_PATH: ORIGINAL_MODEL_PATH,
    }

    if archived_metadata.exists():
        metadata = read_json(archived_metadata)
        if int(metadata.get("best_iteration", -1)) != ORIGINAL_METRICS[
            "best_iteration"
        ] or not np.isclose(
            float(metadata.get("scale_pos_weight", np.nan)),
            ORIGINAL_METRICS["scale_pos_weight"],
        ):
            raise ValueError("Existing original-baseline archive is not authoritative.")
        missing = [str(path) for path in archive_targets.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Original-baseline archive is incomplete: {missing}")
        normalize_original_archive_metadata(archived_metadata)
        return

    if ORIGINAL_REPORT_DIR.exists() and any(ORIGINAL_REPORT_DIR.iterdir()):
        raise FileExistsError(
            f"Refusing to overwrite a partial original-baseline archive: {ORIGINAL_REPORT_DIR}"
        )

    current_metrics = read_json(METRICS_PATH)
    current_metadata = read_json(METADATA_PATH)
    if (
        not np.isclose(current_metrics["pr_auc"], ORIGINAL_METRICS["pr_auc"])
        or not np.isclose(current_metrics["roc_auc"], ORIGINAL_METRICS["roc_auc"])
        or int(current_metadata["best_iteration"]) != ORIGINAL_METRICS["best_iteration"]
        or not np.isclose(
            current_metadata["scale_pos_weight"],
            ORIGINAL_METRICS["scale_pos_weight"],
        )
    ):
        raise ValueError(
            "Canonical artifacts no longer match the original 3000-tree baseline; "
            "cannot create a trustworthy archive."
        )

    ORIGINAL_REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ORIGINAL_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    for source, destination in archive_targets.items():
        if not source.exists():
            raise FileNotFoundError(f"Original baseline artifact missing: {source}")
        shutil.copy2(source, destination)
    normalize_original_archive_metadata(archived_metadata)


def assert_no_test_metrics(payload: dict[str, Any], source_name: str) -> None:
    forbidden = {
        "test_pr_auc",
        "test_roc_auc",
        "test_accuracy",
        "test_confusion_matrix",
        "test_precision",
        "test_recall",
        "test_ranking_metrics",
    }
    present = forbidden & set(payload)
    if present:
        raise AssertionError(f"Test metrics found in {source_name}: {sorted(present)}.")
    if payload.get("test_evaluated") is not False:
        raise AssertionError(f"{source_name} must record test_evaluated=false.")


def validate_candidate(run_name: str, expected_weight: float) -> dict[str, Any]:
    paths = build_artifact_paths(run_name)
    metrics = read_json(paths.metrics)
    metadata = read_json(paths.metadata)
    assert_no_test_metrics(metrics, f"{run_name} metrics")
    assert_no_test_metrics(metadata, f"{run_name} metadata")

    if metrics.get("run_name") != run_name or metadata.get("run_name") != run_name:
        raise ValueError(f"Run-name mismatch in {run_name} artifacts.")
    if not np.isclose(metrics["scale_pos_weight"], expected_weight):
        raise ValueError(f"Unexpected class weight for {run_name}.")
    if not np.isclose(metadata["scale_pos_weight"], expected_weight):
        raise ValueError(f"Unexpected metadata class weight for {run_name}.")
    if int(metadata["n_estimators_cap"]) != 6_000:
        raise ValueError(f"{run_name} did not use the 6000-tree cap.")
    if int(metadata["early_stopping_rounds"]) != 200:
        raise ValueError(f"{run_name} did not use 200-round patience.")
    if metadata["number_of_predictors"] != 435:
        raise ValueError(f"{run_name} predictor count changed.")
    if metadata["number_of_categorical_predictors"] != 31:
        raise ValueError(f"{run_name} categorical predictor count changed.")
    if metadata["number_of_numeric_predictors"] != 404:
        raise ValueError(f"{run_name} numeric predictor count changed.")

    predictions = pd.read_parquet(paths.validation_predictions)
    expected_columns = ["TransactionID", "TransactionDT", "isFraud", "prediction"]
    if predictions.columns.tolist() != expected_columns:
        raise ValueError(f"Unexpected prediction schema for {run_name}.")
    if len(predictions) != 88_581 or not predictions["TransactionID"].is_unique:
        raise ValueError(f"Invalid validation prediction rows for {run_name}.")
    probabilities = predictions["prediction"].to_numpy()
    labels = predictions["isFraud"].to_numpy()
    if not np.isfinite(probabilities).all():
        raise ValueError(f"Non-finite validation predictions for {run_name}.")
    if not ((probabilities >= 0) & (probabilities <= 1)).all():
        raise ValueError(f"Out-of-range validation predictions for {run_name}.")

    if not np.isclose(average_precision_score(labels, probabilities), metrics["pr_auc"]):
        raise ValueError(f"PR-AUC does not reproduce for {run_name}.")
    if not np.isclose(roc_auc_score(labels, probabilities), metrics["roc_auc"]):
        raise ValueError(f"ROC-AUC does not reproduce for {run_name}.")
    for fraction, key in [
        (0.005, "top_0.5_pct"),
        (0.01, "top_1_pct"),
        (0.02, "top_2_pct"),
        (0.05, "top_5_pct"),
    ]:
        recomputed = precision_recall_at_fraction(labels, probabilities, fraction)
        if recomputed != metrics["ranking_metrics"][key]:
            raise ValueError(f"Ranking metrics do not reproduce for {run_name}: {key}.")

    curve = pd.read_csv(paths.learning_curve)
    actual_stopping_iteration = int(metadata["actual_stopping_iteration"])
    if len(curve) != actual_stopping_iteration:
        raise ValueError(f"Learning-curve length mismatch for {run_name}.")
    if int(curve["iteration"].iloc[-1]) != actual_stopping_iteration:
        raise ValueError(f"Learning-curve iteration mismatch for {run_name}.")

    booster = lgb.Booster(model_file=str(paths.model))
    if booster.num_trees() != int(metadata["best_iteration"]):
        raise ValueError(f"Saved model tree count mismatch for {run_name}.")
    return {
        "paths": paths,
        "metrics": metrics,
        "metadata": metadata,
        "prediction_metadata": predictions[["TransactionID", "TransactionDT", "isFraud"]],
    }


def assert_candidates_are_controlled(
    weighted: dict[str, Any],
    unweighted: dict[str, Any],
) -> None:
    weighted_metadata = weighted["metadata"]
    unweighted_metadata = unweighted["metadata"]

    if weighted_metadata["feature_columns"] != unweighted_metadata["feature_columns"]:
        raise AssertionError("Candidate feature columns differ.")
    if (
        weighted_metadata["categorical_mappings_sha256"]
        != unweighted_metadata["categorical_mappings_sha256"]
    ):
        raise AssertionError("Candidate categorical mappings differ.")

    weighted_params = dict(weighted_metadata["lightgbm_parameters"])
    unweighted_params = dict(unweighted_metadata["lightgbm_parameters"])
    weighted_params.pop("scale_pos_weight")
    unweighted_params.pop("scale_pos_weight")
    if weighted_params != unweighted_params:
        raise AssertionError("Candidate model settings differ beyond scale_pos_weight.")

    if not weighted["prediction_metadata"].equals(unweighted["prediction_metadata"]):
        raise AssertionError("Candidate validation observations or order differ.")


def comparison_row(
    run_name: str,
    metrics: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    ranking = metrics["ranking_metrics"]
    return {
        "run_name": run_name,
        "scale_pos_weight": float(metadata["scale_pos_weight"]),
        "max_estimators": int(
            metadata.get("n_estimators_cap", metadata["lightgbm_parameters"]["n_estimators"])
        ),
        "actual_stopping_iteration": int(
            metadata.get("actual_stopping_iteration", metadata["best_iteration"])
        ),
        "best_iteration": int(metadata["best_iteration"]),
        "early_stopping_triggered": bool(metadata.get("early_stopping_triggered", False)),
        "estimator_cap_reached": bool(
            metadata.get(
                "estimator_cap_reached",
                metadata["best_iteration"] == metadata["lightgbm_parameters"]["n_estimators"],
            )
        ),
        "validation_pr_auc": float(metrics["pr_auc"]),
        "validation_roc_auc": float(metrics["roc_auc"]),
        "precision_at_0_5_pct": ranking["top_0.5_pct"]["precision"],
        "recall_at_0_5_pct": ranking["top_0.5_pct"]["recall"],
        "precision_at_1_pct": ranking["top_1_pct"]["precision"],
        "recall_at_1_pct": ranking["top_1_pct"]["recall"],
        "precision_at_2_pct": ranking["top_2_pct"]["precision"],
        "recall_at_2_pct": ranking["top_2_pct"]["recall"],
        "precision_at_5_pct": ranking["top_5_pct"]["precision"],
        "recall_at_5_pct": ranking["top_5_pct"]["recall"],
        "selected_as_b0": False,
    }


def freeze_selected_b0(selected: dict[str, Any], selected_run_name: str) -> None:
    paths = selected["paths"]
    metrics = dict(selected["metrics"])
    metadata = dict(selected["metadata"])

    shutil.copy2(paths.model, MODEL_PATH)
    shutil.copy2(paths.validation_predictions, VALIDATION_PREDICTIONS_PATH)
    shutil.copy2(paths.feature_importance, FEATURE_IMPORTANCE_PATH)

    metrics.update(
        {
            "model": "B0_tabular_lightgbm",
            "selected_run_name": selected_run_name,
            "selected_from_controlled_baseline_comparison": True,
            "selection_metric": "validation_pr_auc",
            "baseline_frozen": True,
        }
    )
    write_json(METRICS_PATH, metrics)

    metadata.update(
        {
            "model_name": "B0_tabular_lightgbm",
            "selected_run_name": selected_run_name,
            "selected_as_b0": True,
            "selected_from_controlled_baseline_comparison": True,
            "selection_metric": "validation_pr_auc",
            "baseline_frozen": True,
            "baseline_frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "baseline_frozen_before_relational_feature_modeling": True,
            "validation_pr_auc": float(metrics["pr_auc"]),
            "validation_roc_auc": float(metrics["roc_auc"]),
            "validation_ranking_metrics": metrics["ranking_metrics"],
            "baseline_comparison_path": repository_relative(COMPARISON_PATH),
            "selected_experiment_metadata_path": repository_relative(paths.metadata),
            "selected_experiment_model_path": repository_relative(paths.model),
            "validation_metrics_path": repository_relative(METRICS_PATH),
            "validation_predictions_path": repository_relative(VALIDATION_PREDICTIONS_PATH),
            "feature_importance_path": repository_relative(FEATURE_IMPORTANCE_PATH),
            "model_path": repository_relative(MODEL_PATH),
            "test_evaluated": False,
            "relational_or_graph_features_used": False,
        }
    )
    write_json(METADATA_PATH, metadata)


def main() -> None:
    archive_original_baseline()

    weighted = validate_candidate(
        WEIGHTED_RUN,
        expected_weight=ORIGINAL_METRICS["scale_pos_weight"],
    )
    unweighted = validate_candidate(UNWEIGHTED_RUN, expected_weight=1.0)
    assert_candidates_are_controlled(weighted, unweighted)

    original_metrics = read_json(ORIGINAL_REPORT_DIR / "metrics.json")
    original_metadata = read_json(ORIGINAL_REPORT_DIR / "metadata.json")
    assert_no_test_metrics(original_metrics, "original metrics")
    assert_no_test_metrics(original_metadata, "original metadata")

    candidates = {
        WEIGHTED_RUN: weighted,
        UNWEIGHTED_RUN: unweighted,
    }
    selected_run_name = max(
        candidates,
        key=lambda name: candidates[name]["metrics"]["pr_auc"],
    )
    selected = candidates[selected_run_name]

    rows = [
        comparison_row(ORIGINAL_RUN, original_metrics, original_metadata),
        comparison_row(WEIGHTED_RUN, weighted["metrics"], weighted["metadata"]),
        comparison_row(
            UNWEIGHTED_RUN,
            unweighted["metrics"],
            unweighted["metadata"],
        ),
    ]
    for row in rows:
        row["selected_as_b0"] = row["run_name"] == selected_run_name
    comparison = pd.DataFrame(rows)
    COMPARISON_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(COMPARISON_PATH, index=False)

    freeze_selected_b0(selected, selected_run_name)

    print("\nControlled baseline comparison:\n")
    print(comparison.to_string(index=False))
    print(f"\nSelected B0: {selected_run_name}")
    print("Selection metric: validation PR-AUC")
    print(f"Canonical model: {MODEL_PATH}")
    print(f"Comparison: {COMPARISON_PATH}")
    print("Final test evaluated: NO")
    print("Relational or graph features added: NO")


if __name__ == "__main__":
    main()
