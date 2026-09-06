"""Seed-variance runs for B0 and B1-card1.

The project's reported deltas come from exactly one training run per
configuration, so the only uncertainty ever quantified is sampling variance:
the paired bootstrap holds the trained model fixed and resamples validation
rows. Run-to-run variance under a different seed is a separate and unmeasured
source of spread, and it matters here because the headline B1-card1 gain over
B0 is +0.0059 PR-AUC. If retraining B0 alone under a different seed moves
validation PR-AUC by a comparable amount, the reported gain is partly seed
lottery.

LightGBM is configured with subsample=0.8 and subsample_freq=1, so the row
subsample is drawn from `random_state` and a seed change produces a genuinely
different fit rather than a bit-identical one.

Everything except `random_state` is frozen:

  * the B0 feature manifest, read from the frozen B0 metadata
  * the train-only categorical mappings, read from the frozen artifact and
    applied without refitting
  * the class weight, the estimator cap, the early-stopping patience and
    every other LightGBM parameter

Nothing here writes to a frozen artifact. Runs land under their own directory
tree, and the frozen B0 and B1 artifacts are hash-pinned before training and
re-checked afterwards.

Usage:
    python -m src.models.train_seed_variants --config b0 --config b1_card1
    python -m src.models.train_seed_variants --seed 42 --config b0
"""

from __future__ import annotations

import argparse
import gc
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

from src.features.build_relational_features import _feature_names
from src.models.train_lightgbm_baseline import (
    CATEGORY_MAPPINGS_PATH,
    EXPECTED_SPLIT_COUNTS,
    METADATA_PATH as B0_METADATA_PATH,
    METRICS_PATH as B0_METRICS_PATH,
    MODEL_PATH as B0_MODEL_PATH,
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
    write_json,
)
from src.models.train_lightgbm_relational import (
    EARLY_STOPPING_ROUNDS,
    FROZEN_PARAMETER_NAMES,
    LOG_EVALUATION_PERIOD,
    MAX_ESTIMATORS,
    apply_frozen_category_mappings,
    assert_protected_artifacts_unchanged,
    build_b1_feature_manifest,
    load_b1_datasets,
    load_frozen_b0_metadata,
    load_frozen_category_mappings,
    read_json,
    snapshot_protected_artifacts,
    validate_model_columns_against_frozen_b0,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT_DIR / "reports" / "seed_variance"
MODEL_DIR = ROOT_DIR / "models" / "seed_variance"

# Five seeds, as required. 42 is the seed every frozen artifact was trained
# under, so its run doubles as a fidelity check: if this harness reproduces the
# frozen PR-AUC to the last digit, the other four runs differ from the frozen
# ones by seed alone and by nothing else.
DEFAULT_SEEDS = [42, 202, 707, 1337, 2024]
CONFIGURATIONS = ["b0", "b1_card1"]
EVALUATION_SPLIT = "validation"

# Reference metrics for the seed-42 fidelity check, one per configuration.
FROZEN_REFERENCE_METRICS = {
    "b0": B0_METRICS_PATH,
    "b1_card1": ROOT_DIR / "reports" / "b1" / "card1" / "metrics.json",
}

# No seed run may modify any artifact another result is compared against.
PROTECTED_PATHS = [
    B0_MODEL_PATH,
    B0_METADATA_PATH,
    B0_METRICS_PATH,
    CATEGORY_MAPPINGS_PATH,
    *(
        path
        for relation in ("card_core_addr1", "card1", "card1_card2")
        for path in (
            ROOT_DIR / "models" / f"lightgbm_b1_{relation}.txt",
            ROOT_DIR / "reports" / "b1" / relation / "metrics.json",
            ROOT_DIR / "reports" / "b1" / relation / "metadata.json",
        )
    ),
]


def resolve_run_paths(config: str, seed: int) -> dict[str, Path]:
    if config not in CONFIGURATIONS:
        raise ValueError(f"Unknown configuration: {config!r}. Supported: {CONFIGURATIONS}.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    run_dir = REPORT_DIR / config / f"seed_{seed}"
    return {
        "run_dir": run_dir,
        "model": MODEL_DIR / f"lightgbm_{config}_seed{seed}.txt",
        "metrics": run_dir / "metrics.json",
        "metadata": run_dir / "metadata.json",
        "feature_importance": run_dir / "feature_importance.csv",
        "validation_predictions": run_dir / "validation_predictions.parquet",
        "learning_curve": run_dir / "learning_curve.csv",
    }


def load_configuration_data(
    config: str,
    b0_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return train/validation partitions and the predictor manifest for `config`.

    B0 uses the frozen 435-column manifest verbatim. B1-card1 appends the four
    card1 relational summaries in the same order the frozen B1 run used, so the
    two configurations differ by those four columns and nothing else.
    """

    b0_features = list(b0_metadata["feature_columns"])
    if config == "b0":
        train_df, validation_df, _ = load_model_dataset()
        validate_model_columns_against_frozen_b0(list(train_df.columns), b0_features, [])
        validate_model_columns_against_frozen_b0(
            list(validation_df.columns), b0_features, []
        )
        return train_df, validation_df, b0_features

    relation = config.removeprefix("b1_")
    feat_names = _feature_names(relation)
    train_df, validation_df, _ = load_b1_datasets(relation)
    validate_model_columns_against_frozen_b0(
        list(train_df.columns), b0_features, feat_names
    )
    validate_model_columns_against_frozen_b0(
        list(validation_df.columns), b0_features, feat_names
    )
    return train_df, validation_df, build_b1_feature_manifest(b0_features, feat_names)


def validate_seed_variant_configuration(
    model: LGBMClassifier,
    b0_metadata: dict[str, Any],
    seed: int,
    n_estimators: int,
) -> None:
    """Assert that `random_state` is the only parameter that moved.

    This is the whole experiment: if anything else differs from the frozen
    configuration, the measured spread stops being seed spread.
    """

    reference = b0_metadata.get("lightgbm_parameters")
    if not isinstance(reference, dict):
        raise ValueError("Frozen B0 LightGBM parameters are missing.")
    actual = model.get_params(deep=False)

    for name in FROZEN_PARAMETER_NAMES:
        if name == "random_state":
            continue
        if name not in reference:
            raise ValueError(f"Frozen B0 parameter is missing: {name}.")
        if actual.get(name) != reference[name]:
            raise ValueError(
                f"Seed variant parameter {name} differs from frozen B0: "
                f"variant={actual.get(name)!r}, B0={reference[name]!r}."
            )
    if actual.get("random_state") != seed:
        raise ValueError(
            f"Seed variant must carry random_state={seed}; got {actual.get('random_state')!r}."
        )
    if not np.isclose(
        actual.get("scale_pos_weight"),
        b0_metadata.get("scale_pos_weight"),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Seed variant class weight differs from frozen B0.")
    if n_estimators == MAX_ESTIMATORS and actual.get("n_estimators") != MAX_ESTIMATORS:
        raise ValueError("Seed variant estimator cap must remain 6000.")


def check_frozen_reproduction(config: str, seed: int, pr_auc: float) -> dict[str, Any]:
    """Compare the seed-42 run against the frozen artifact trained under it."""

    if seed != RANDOM_SEED:
        return {"applicable": False}
    reference_path = FROZEN_REFERENCE_METRICS[config]
    if not reference_path.exists():
        return {"applicable": True, "reference_available": False}
    reference_pr_auc = float(read_json(reference_path)["pr_auc"])
    difference = pr_auc - reference_pr_auc
    return {
        "applicable": True,
        "reference_available": True,
        "reference_metrics_path": repository_relative(reference_path),
        "reference_pr_auc": reference_pr_auc,
        "harness_pr_auc": pr_auc,
        "difference": difference,
        "reproduces_frozen_run": bool(difference == 0.0),
    }


def train_seed_variant(
    config: str,
    seed: int,
    n_estimators: int = MAX_ESTIMATORS,
) -> dict[str, Any]:
    """Train one configuration under one seed and write its artifacts."""

    paths = resolve_run_paths(config, seed)
    protected_before = snapshot_protected_artifacts(PROTECTED_PATHS)
    b0_metadata = load_frozen_b0_metadata()

    label = f"{config}/seed_{seed}"
    print(f"[{label}] Loading data...")
    train_df, validation_df, feature_columns = load_configuration_data(config, b0_metadata)

    categorical_columns = list(b0_metadata["categorical_feature_columns"])
    if identify_categorical_columns(train_df[feature_columns]) != categorical_columns:
        raise TypeError(f"[{label}] Categorical predictors differ from frozen B0.")
    if identify_categorical_columns(validation_df[feature_columns]) != categorical_columns:
        raise TypeError(f"[{label}] Validation categorical predictors differ from frozen B0.")

    validation_metadata = validation_df[
        ["TransactionID", "TransactionDT", "isFraud"]
    ].copy()
    y_train = train_df["isFraud"].astype("int8").copy()
    y_validation = validation_df["isFraud"].astype("int8").copy()

    mappings, mapping_sha256 = load_frozen_category_mappings(categorical_columns)
    print(f"[{label}] Applying frozen categorical mappings (no fitting)...")
    # Encode in place on the source partitions, before the predictor matrices
    # exist. The train partition's 31 object columns hold roughly 0.7 GB of
    # Python strings; mapping them to int32 first keeps that off the peak
    # instead of carrying it through the extraction below.
    apply_frozen_category_mappings(train_df, validation_df, categorical_columns, mappings)

    # Move each predictor out of its source frame rather than copying it. A
    # slice-and-copy would hold a full duplicate of a 1.79 GB partition
    # alongside the original; popping transfers ownership column by column, so
    # only one copy is ever resident. This is a memory change only -- the
    # values, their order and their dtypes are identical either way, which the
    # seed-42 reproduction check at the end of this function verifies.
    X_train = pd.DataFrame(
        {column: train_df.pop(column) for column in feature_columns}, copy=False
    )
    X_validation = pd.DataFrame(
        {column: validation_df.pop(column) for column in feature_columns}, copy=False
    )
    del train_df, validation_df
    gc.collect()

    if list(X_train.columns) != feature_columns:
        raise AssertionError(f"[{label}] Predictor extraction changed column order.")
    if list(X_validation.columns) != feature_columns:
        raise AssertionError(f"[{label}] Validation predictor order differs from train.")
    assert_supported_model_dtypes(X_train, f"{label} training predictors")
    assert_supported_model_dtypes(X_validation, f"{label} validation predictors")

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    model = build_lightgbm_model(scale_pos_weight, n_estimators=n_estimators)
    model.set_params(random_state=seed)
    validate_seed_variant_configuration(model, b0_metadata, seed, n_estimators)

    print(f"[{label}] Predictors: {len(feature_columns):,}  seed: {seed}")
    evaluation_results: dict[str, dict[str, list[float]]] = {}
    callbacks = [
        lightgbm.early_stopping(
            stopping_rounds=EARLY_STOPPING_ROUNDS, first_metric_only=True
        ),
        lightgbm.record_evaluation(evaluation_results),
        lightgbm.log_evaluation(period=LOG_EVALUATION_PERIOD),
    ]
    print(f"[{label}] Training...")
    started_at = datetime.now(timezone.utc)
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
    finished_at = datetime.now(timezone.utc)
    if not model.best_iteration_ or model.best_iteration_ <= 0:
        raise RuntimeError(f"[{label}] LightGBM did not report a valid best iteration.")

    learning_curve = build_learning_curve(evaluation_results)
    learning_curve_summary = summarize_learning_curve(
        learning_curve,
        best_iteration=int(model.best_iteration_),
        maximum_estimators=n_estimators,
    )
    validation_scores = model.predict_proba(
        X_validation, num_iteration=model.best_iteration_
    )[:, 1]
    if len(validation_scores) != EXPECTED_SPLIT_COUNTS[EVALUATION_SPLIT]:
        raise AssertionError(f"[{label}] Validation prediction count is incorrect.")
    if not np.isfinite(validation_scores).all():
        raise AssertionError(f"[{label}] Validation predictions contain non-finite values.")

    metrics = evaluate_validation(y_validation, validation_scores)
    reproduction = check_frozen_reproduction(config, seed, float(metrics["pr_auc"]))
    metrics.update(
        {
            "model": f"seed_variant_{config}",
            "configuration": config,
            "random_seed": seed,
            "weighting": "weighted",
            "scale_pos_weight": float(scale_pos_weight),
            "maximum_estimators": int(n_estimators),
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
            "estimator_cap_reached": bool(learning_curve_summary["estimator_cap_reached"]),
            "frozen_run_reproduction": reproduction,
        }
    )

    feature_importance = build_feature_importance(model)
    if len(feature_importance) != len(feature_columns):
        raise AssertionError(f"[{label}] Feature importance does not match its manifest.")
    validation_predictions = validation_metadata.copy()
    validation_predictions["prediction"] = validation_scores

    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(paths["model"]), num_iteration=model.best_iteration_)
    write_json(paths["metrics"], metrics)
    feature_importance.to_csv(paths["feature_importance"], index=False)
    validation_predictions.to_parquet(
        paths["validation_predictions"],
        index=False,
        engine="pyarrow",
        compression="snappy",
    )
    learning_curve.to_csv(paths["learning_curve"], index=False)

    metadata = {
        "experiment_name": "seed_variance",
        "configuration": config,
        "random_seed": seed,
        "varied_parameter": "random_state",
        "held_frozen": (
            "B0 feature manifest, train-only categorical mappings, class weight, "
            "estimator cap, early-stopping patience and every other LightGBM parameter"
        ),
        "feature_count": len(feature_columns),
        "categorical_feature_count": len(categorical_columns),
        "categorical_mappings_sha256": mapping_sha256,
        "scale_pos_weight": float(scale_pos_weight),
        "lightgbm_parameters": model.get_params(deep=False),
        "max_estimators": int(n_estimators),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "best_iteration": int(model.best_iteration_),
        "learning_curve_summary": learning_curve_summary,
        "evaluation_split": EVALUATION_SPLIT,
        "validation_pr_auc": float(metrics["pr_auc"]),
        "validation_roc_auc": float(metrics["roc_auc"]),
        "frozen_run_reproduction": reproduction,
        "protected_artifact_sha256": protected_before,
        "training_seconds": (finished_at - started_at).total_seconds(),
        "test_evaluated": False,
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
        "generated_at_utc": finished_at.isoformat(),
    }
    write_json(paths["metadata"], metadata)
    assert_protected_artifacts_unchanged(
        protected_before, PROTECTED_PATHS, label="frozen B0/B1"
    )

    print(
        f"[{label}] PR-AUC {metrics['pr_auc']:.12f}  "
        f"ROC-AUC {metrics['roc_auc']:.12f}  "
        f"best_iteration {model.best_iteration_:,}  "
        f"early stopping {'YES' if metrics['early_stopping_triggered'] else 'NO'}"
    )
    if reproduction.get("reference_available"):
        verdict = "EXACT" if reproduction["reproduces_frozen_run"] else "MISMATCH"
        print(
            f"[{label}] Frozen-run reproduction: {verdict} "
            f"(difference {reproduction['difference']:+.3e})"
        )
    print(f"[{label}] Elapsed: {metadata['training_seconds'] / 60:.1f} min")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train B0 and B1-card1 across seeds to measure run-to-run variance."
    )
    parser.add_argument(
        "--config",
        action="append",
        choices=CONFIGURATIONS,
        help="Configuration to run; repeatable. Defaults to both.",
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Seed to run; repeatable. Defaults to the five-seed panel.",
    )
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=MAX_ESTIMATORS,
        help=(
            "Estimator cap. Only change this for a smoke test; the reported panel "
            "must run at the frozen cap to stay comparable with the frozen numbers."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run whose metrics.json already exists, so an interrupted panel resumes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configs = args.config or CONFIGURATIONS
    seeds = args.seed or DEFAULT_SEEDS
    if args.n_estimators != MAX_ESTIMATORS:
        print(
            f"WARNING: estimator cap is {args.n_estimators:,}, not the frozen "
            f"{MAX_ESTIMATORS:,}. These runs are a smoke test, not the reported panel."
        )

    # Seed-major: each seed's B0 and B1 finish together, so an interrupted panel
    # still yields complete paired deltas for the seeds it got through. The
    # paired delta is the quantity that matters, and a half-finished
    # configuration-major run would produce none.
    planned = [(config, seed) for seed in seeds for config in configs]
    print(f"Planned runs: {len(planned)}")
    for index, (config, seed) in enumerate(planned, start=1):
        paths = resolve_run_paths(config, seed)
        if args.skip_existing and paths["metrics"].exists():
            print(f"({index}/{len(planned)}) SKIP {config}/seed_{seed}: already run.")
            continue
        print(f"\n({index}/{len(planned)}) === {config} seed {seed} ===")
        train_seed_variant(config, seed, n_estimators=args.n_estimators)

    print("\nAll planned runs complete.")
    print(f"Reports: {REPORT_DIR}")
    print("Test set evaluated: NO")


if __name__ == "__main__":
    main()
