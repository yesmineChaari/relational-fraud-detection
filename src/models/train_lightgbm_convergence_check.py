"""Extend B0, B1-card1 and G1-card1 past the frozen 6,000-estimator cap.

Every recorded run of these three configurations stops at the frozen cap. The
seed-variance panel (src/models/train_seed_variants.py) showed why that
matters beyond "not fully converged": patience is evaluated on validation
`average_precision`, the same quantity the comparison reports, so whichever
configuration stops early loses that optimisation and the reported delta
partly scores the stopping point rather than the predictors. This module
answers two separate questions in one harness:

  1. Where does each configuration actually stop, if the cap is raised well
     past 6,000 rounds at the same 200-round patience? This is the required
     run every acceptance criterion in this investigation depends on.
  2. Does the confound go away if patience watches a metric other than the
     one being reported? Reordering `eval_metric` so ROC-AUC is first makes
     `first_metric_only=True` patience on ROC-AUC instead of PR-AUC. This is
     evaluated only for B0 and B1-card1, to bound the added training cost.

Cap and stopping-metric are varied one at a time, never together in the same
run: conflating them would leave no way to tell which change produced which
effect. Patience itself (200 rounds) is never changed here either, for the
same reason -- it is not the variable this investigation is about.

Every other setting -- the B0 feature manifest, the train-only categorical
mappings, the class weight, and every other LightGBM parameter -- is read
from the frozen artifacts and asserted unchanged before training, exactly as
train_seed_variants.py does for its own single varied parameter. Nothing here
writes to a frozen artifact: runs land under their own report and model
trees, and every frozen reference this investigation depends on or is
compared against (B0, B1-card1, G1-card1, and the GraphSAGE embeddings G1
depends on) is hash-pinned before training and re-checked afterwards.

Usage:
    python -m src.models.train_lightgbm_convergence_check
    python -m src.models.train_lightgbm_convergence_check --config b0 --config b1_card1 --stop-metric auc
"""

from __future__ import annotations

import argparse
import gc
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import lightgbm
import numpy as np
import pandas as pd
import platform
import sklearn
from lightgbm import LGBMClassifier

from src.models.train_lightgbm_baseline import (
    EXPECTED_SPLIT_COUNTS,
    METRICS_PATH as B0_METRICS_PATH,
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
from src.features.build_relational_features import _feature_names
from src.models.train_lightgbm_relational import (
    B0_PROTECTED_PATHS,
    EARLY_STOPPING_ROUNDS,
    EVALUATION_SPLIT,
    FROZEN_PARAMETER_NAMES,
    LOG_EVALUATION_PERIOD,
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
from src.models.train_lightgbm_g1 import (
    B1_CARD1_PROTECTED_PATHS,
    RELATION as G1_RELATION,
    build_g1_feature_manifest,
    embedding_feature_names,
    load_embedding_metadata,
    load_g1_datasets,
)
from src.models.train_lightgbm_g1_controls import G1_PROTECTED_PATHS

ROOT_DIR = Path(__file__).resolve().parents[2]

REPORT_DIR = ROOT_DIR / "reports" / "convergence_check"
MODEL_DIR = ROOT_DIR / "models" / "convergence_check"

CONFIGURATIONS = ["b0", "b1_card1", "g1_card1"]
STOP_METRICS = ["average_precision", "auc"]
DEFAULT_MAX_ESTIMATORS = 15_000

# eval_metric ordering controls which metric first_metric_only=True patiences
# on. Both metrics are always logged either way, so the learning curve always
# has both columns regardless of which one stopping actually watched.
STOP_METRIC_EVAL_ORDER: dict[str, list[str]] = {
    "average_precision": ["average_precision", "auc"],
    "auc": ["auc", "average_precision"],
}

# The frozen, 6,000-cap reference every extended run is measured against.
FROZEN_REFERENCE_METRICS = {
    "b0": B0_METRICS_PATH,
    "b1_card1": ROOT_DIR / "reports" / "b1" / "card1" / "metrics.json",
    "g1_card1": ROOT_DIR / "reports" / "g1" / "card1" / "metrics.json",
}

# Every artifact this investigation reads from or is compared against, none of
# which it may ever write to. Imported from the modules that already own and
# protect them, never redefined, so this list cannot drift from what upstream
# scripts already guarantee.
ALL_PROTECTED_PATHS = [
    *B0_PROTECTED_PATHS,
    *B1_CARD1_PROTECTED_PATHS,
    *G1_PROTECTED_PATHS,
]


def resolve_run_paths(
    config: str, stop_metric: str, n_estimators: int, seed: int
) -> dict[str, Path]:
    if config not in CONFIGURATIONS:
        raise ValueError(f"Unknown configuration: {config!r}. Supported: {CONFIGURATIONS}.")
    if stop_metric not in STOP_METRICS:
        raise ValueError(f"Unknown stop metric: {stop_metric!r}. Supported: {STOP_METRICS}.")
    if n_estimators <= 0:
        raise ValueError("n_estimators must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    label = f"stop_{stop_metric}"
    run_dir = REPORT_DIR / config / label / f"cap{n_estimators}_seed{seed}"
    return {
        "run_dir": run_dir,
        "model": MODEL_DIR / f"lightgbm_{config}__{label}_cap{n_estimators}_seed{seed}.txt",
        "metrics": run_dir / "metrics.json",
        "metadata": run_dir / "metadata.json",
        "feature_importance": run_dir / "feature_importance.csv",
        "validation_predictions": run_dir / "validation_predictions.parquet",
        "learning_curve": run_dir / "learning_curve.csv",
    }


def run_is_complete(config: str, stop_metric: str, n_estimators: int, seed: int) -> bool:
    """True when every artifact this run publishes is already on disk."""
    paths = resolve_run_paths(config, stop_metric, n_estimators, seed)
    return all(path.exists() for key, path in paths.items() if key != "run_dir")


def _load_b0(b0_metadata: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feature_columns = list(b0_metadata["feature_columns"])
    train_df, validation_df, _ = load_model_dataset()
    validate_model_columns_against_frozen_b0(list(train_df.columns), feature_columns, [])
    validate_model_columns_against_frozen_b0(list(validation_df.columns), feature_columns, [])
    return train_df, validation_df, feature_columns


def _load_b1_card1(b0_metadata: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    b0_features = list(b0_metadata["feature_columns"])
    feat_names = _feature_names("card1")
    train_df, validation_df, _ = load_b1_datasets("card1")
    validate_model_columns_against_frozen_b0(list(train_df.columns), b0_features, feat_names)
    validate_model_columns_against_frozen_b0(list(validation_df.columns), b0_features, feat_names)
    feature_columns = build_b1_feature_manifest(b0_features, feat_names)
    return train_df, validation_df, feature_columns


def _load_g1_card1(b0_metadata: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    b0_features = list(b0_metadata["feature_columns"])
    feat_names = embedding_feature_names()
    # Re-validated here, not just at G1's own training time: no leakage, no
    # split mismatch, same encoder this investigation is meant to speak to.
    load_embedding_metadata(G1_RELATION)
    train_df, validation_df, _ = load_g1_datasets(G1_RELATION)
    validate_model_columns_against_frozen_b0(list(train_df.columns), b0_features, feat_names)
    validate_model_columns_against_frozen_b0(list(validation_df.columns), b0_features, feat_names)
    feature_columns = build_g1_feature_manifest(b0_features, feat_names)
    return train_df, validation_df, feature_columns


LOADERS: dict[str, Callable[[dict[str, Any]], tuple[pd.DataFrame, pd.DataFrame, list[str]]]] = {
    "b0": _load_b0,
    "b1_card1": _load_b1_card1,
    "g1_card1": _load_g1_card1,
}


def validate_convergence_configuration(
    model: LGBMClassifier,
    b0_metadata: dict[str, Any],
    seed: int,
    n_estimators: int,
) -> None:
    """Assert that only `random_state` and `n_estimators` differ from frozen B0.

    This is the whole experiment: if anything else moved, a measured
    convergence point or delta shift stops being about the cap or the seed.
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
                f"Convergence-check parameter {name} differs from frozen B0: "
                f"actual={actual.get(name)!r}, B0={reference[name]!r}."
            )
    if actual.get("random_state") != seed:
        raise ValueError(
            f"Convergence-check run must carry random_state={seed}; "
            f"got {actual.get('random_state')!r}."
        )
    if actual.get("n_estimators") != n_estimators:
        raise ValueError(
            f"Convergence-check run must carry n_estimators={n_estimators}; "
            f"got {actual.get('n_estimators')!r}."
        )
    if not np.isclose(
        actual.get("scale_pos_weight"),
        b0_metadata.get("scale_pos_weight"),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError("Convergence-check class weight differs from frozen B0.")


def summarize_learning_curve_for_stop_metric(
    learning_curve: pd.DataFrame,
    best_iteration: int,
    maximum_estimators: int,
    stop_metric: str,
) -> dict[str, Any]:
    """Like summarize_learning_curve, generalised to whichever metric patience watched.

    summarize_learning_curve (train_lightgbm_baseline.py) hard-asserts that
    `best_iteration` equals the argmax of validation_average_precision --
    correct only when average_precision is the stopping metric. The
    decoupling runs here reorder eval_metric so patience watches AUC instead,
    in which case best_iteration tracks AUC's argmax and PR-AUC must be read
    off at that round rather than at its own maximum -- reading it off at its
    own maximum would just reintroduce the same circularity this
    investigation exists to remove.
    """
    if stop_metric == "average_precision":
        return summarize_learning_curve(learning_curve, best_iteration, maximum_estimators)

    actual_stopping_iteration = int(learning_curve["iteration"].iloc[-1])
    stop_series = learning_curve[f"validation_{stop_metric}"]
    ap_series = learning_curve["validation_average_precision"]
    best_logged_iteration = int(stop_series.idxmax() + 1)
    if best_logged_iteration != best_iteration:
        raise AssertionError(
            "LightGBM best_iteration differs from the recorded validation-"
            f"{stop_metric} maximum: model={best_iteration}, "
            f"history={best_logged_iteration}."
        )
    estimator_cap_reached = actual_stopping_iteration == maximum_estimators
    return {
        "maximum_estimators": int(maximum_estimators),
        "actual_stopping_iteration": actual_stopping_iteration,
        "best_iteration": int(best_iteration),
        "stop_metric": stop_metric,
        f"best_validation_{stop_metric}": float(stop_series.iloc[best_iteration - 1]),
        "best_validation_average_precision": float(ap_series.iloc[best_iteration - 1]),
        "final_logged_validation_average_precision": float(ap_series.iloc[-1]),
        "estimator_cap_reached": estimator_cap_reached,
        "early_stopping_triggered": not estimator_cap_reached,
    }


@dataclass(frozen=True)
class ConvergenceRun:
    config: str
    stop_metric: str
    seed: int
    n_estimators: int

    @property
    def label(self) -> str:
        return f"{self.config}/stop_{self.stop_metric}/cap{self.n_estimators}/seed{self.seed}"


def run_convergence(
    config: str,
    stop_metric: str,
    seed: int,
    n_estimators: int,
    skip_existing: bool = False,
) -> None:
    """Train one (config, stop_metric) cell at an extended cap and write its artifacts."""

    run = ConvergenceRun(config=config, stop_metric=stop_metric, seed=seed, n_estimators=n_estimators)
    label = run.label
    if skip_existing and run_is_complete(config, stop_metric, n_estimators, seed):
        print(f"[{label}] Already complete; skipping.")
        return

    paths = resolve_run_paths(config, stop_metric, n_estimators, seed)
    protected_before = snapshot_protected_artifacts(ALL_PROTECTED_PATHS)
    b0_metadata = load_frozen_b0_metadata()

    print(f"[{label}] Loading data...")
    train_df, validation_df, feature_columns = LOADERS[config](b0_metadata)

    categorical_columns = list(b0_metadata["categorical_feature_columns"])
    if identify_categorical_columns(train_df[feature_columns]) != categorical_columns:
        raise TypeError(f"[{label}] Categorical predictors differ from frozen B0.")
    if identify_categorical_columns(validation_df[feature_columns]) != categorical_columns:
        raise TypeError(f"[{label}] Validation categorical predictors differ from frozen B0.")

    validation_metadata = validation_df[["TransactionID", "TransactionDT", "isFraud"]].copy()
    y_train = train_df["isFraud"].astype("int8").copy()
    y_validation = validation_df["isFraud"].astype("int8").copy()

    mappings, mapping_sha256 = load_frozen_category_mappings(categorical_columns)
    print(f"[{label}] Applying frozen categorical mappings (no fitting)...")
    apply_frozen_category_mappings(train_df, validation_df, categorical_columns, mappings)

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
    validate_convergence_configuration(model, b0_metadata, seed, n_estimators)

    print(
        f"[{label}] Predictors: {len(feature_columns):,}  "
        f"stop metric: {stop_metric}  cap: {n_estimators:,}  seed: {seed}"
    )
    evaluation_results: dict[str, dict[str, list[float]]] = {}
    callbacks = [
        lightgbm.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, first_metric_only=True),
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
        eval_metric=STOP_METRIC_EVAL_ORDER[stop_metric],
        categorical_feature=categorical_columns,
        callbacks=callbacks,
    )
    finished_at = datetime.now(timezone.utc)
    if not model.best_iteration_ or model.best_iteration_ <= 0:
        raise RuntimeError(f"[{label}] LightGBM did not report a valid best iteration.")

    learning_curve = build_learning_curve(evaluation_results)
    learning_curve_summary = summarize_learning_curve_for_stop_metric(
        learning_curve,
        best_iteration=int(model.best_iteration_),
        maximum_estimators=n_estimators,
        stop_metric=stop_metric,
    )
    validation_scores = model.predict_proba(
        X_validation, num_iteration=model.best_iteration_
    )[:, 1]
    if len(validation_scores) != EXPECTED_SPLIT_COUNTS[EVALUATION_SPLIT]:
        raise AssertionError(f"[{label}] Validation prediction count is incorrect.")
    if not np.isfinite(validation_scores).all():
        raise AssertionError(f"[{label}] Validation predictions contain non-finite values.")

    frozen_reference = read_json(FROZEN_REFERENCE_METRICS[config])
    metrics = evaluate_validation(y_validation, validation_scores)
    metrics.update(
        {
            "model": f"convergence_check_{config}",
            "configuration": config,
            "stop_metric": stop_metric,
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
            "frozen_reference_maximum_estimators": int(frozen_reference["maximum_estimators"]),
            "frozen_reference_best_iteration": int(frozen_reference["best_iteration"]),
            "frozen_reference_pr_auc": float(frozen_reference["pr_auc"]),
            "frozen_reference_roc_auc": float(frozen_reference["roc_auc"]),
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
        paths["validation_predictions"], index=False, engine="pyarrow", compression="snappy"
    )
    learning_curve.to_csv(paths["learning_curve"], index=False)

    varied_parameters = ["n_estimators"] if seed == RANDOM_SEED else ["n_estimators", "random_state"]
    metadata = {
        "experiment_name": "estimator_cap_convergence_check",
        "configuration": config,
        "stop_metric": stop_metric,
        "random_seed": seed,
        "varied_parameters": varied_parameters,
        "held_frozen": (
            "B0 feature manifest, train-only categorical mappings, class weight, "
            "early-stopping patience and every other LightGBM parameter"
        ),
        "feature_count": len(feature_columns),
        "categorical_feature_count": len(categorical_columns),
        "categorical_mappings_sha256": mapping_sha256,
        "scale_pos_weight": float(scale_pos_weight),
        "lightgbm_parameters": model.get_params(deep=False),
        "max_estimators": int(n_estimators),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "early_stopping_metric": stop_metric,
        "best_iteration": int(model.best_iteration_),
        "learning_curve_summary": learning_curve_summary,
        "evaluation_split": EVALUATION_SPLIT,
        "validation_pr_auc": float(metrics["pr_auc"]),
        "validation_roc_auc": float(metrics["roc_auc"]),
        "frozen_reference_metrics_path": repository_relative(FROZEN_REFERENCE_METRICS[config]),
        "frozen_reference_maximum_estimators": int(frozen_reference["maximum_estimators"]),
        "frozen_reference_best_iteration": int(frozen_reference["best_iteration"]),
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
        protected_before, ALL_PROTECTED_PATHS, label="frozen B0/B1-card1/G1-card1"
    )

    expected_artifacts = [
        paths["model"], paths["metrics"], paths["metadata"],
        paths["feature_importance"], paths["validation_predictions"], paths["learning_curve"],
    ]
    missing = [str(p) for p in expected_artifacts if not p.exists()]
    if missing:
        raise OSError(f"[{label}] Convergence-check artifacts were not created: {missing}.")

    print(
        f"[{label}] Best iteration: {model.best_iteration_:,} of {n_estimators:,}  "
        f"(frozen cap-6000 best_iteration: {frozen_reference['best_iteration']:,})"
    )
    print(
        f"[{label}] Early stopping triggered: "
        f"{'YES' if metrics['early_stopping_triggered'] else 'NO'}"
    )
    print(
        f"[{label}] Validation PR-AUC:  {metrics['pr_auc']:.12f}  "
        f"(frozen: {frozen_reference['pr_auc']:.12f})"
    )
    print(
        f"[{label}] Validation ROC-AUC: {metrics['roc_auc']:.12f}  "
        f"(frozen: {frozen_reference['roc_auc']:.12f})"
    )
    print(f"[{label}] Elapsed: {(finished_at - started_at).total_seconds() / 60:.1f} min")
    print(f"[{label}] Frozen B0/B1-card1/G1-card1 artifacts unchanged: YES")
    print(f"[{label}] Final test evaluated: NO")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extend B0, B1-card1 and G1-card1 past the frozen 6,000-estimator cap "
            "to find where each genuinely converges, optionally decoupling the "
            "early-stopping metric from the reported one."
        )
    )
    parser.add_argument(
        "--config",
        action="append",
        choices=CONFIGURATIONS,
        help="Configuration to run; repeatable. Defaults to all three.",
    )
    parser.add_argument(
        "--stop-metric",
        action="append",
        choices=STOP_METRICS,
        help=(
            "Metric early-stopping patience watches; repeatable. Defaults to "
            "average_precision only, which reproduces the current stopping rule "
            "at the extended cap. Add auc to test decoupling patience from the "
            "reported metric (only meaningful for b0 and b1_card1)."
        ),
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help="Seed to run; repeatable. Defaults to [42], the frozen seed.",
    )
    parser.add_argument(
        "--max-estimators",
        type=int,
        default=DEFAULT_MAX_ESTIMATORS,
        help=f"Estimator cap for this investigation (default: {DEFAULT_MAX_ESTIMATORS:,}).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run whose artifacts already exist, so an interrupted job resumes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configs = args.config or CONFIGURATIONS
    stop_metrics = args.stop_metric or ["average_precision"]
    seeds = args.seed or [RANDOM_SEED]

    planned = [
        (config, stop_metric, seed)
        for config in configs
        for stop_metric in stop_metrics
        for seed in seeds
    ]
    print(f"Planned runs: {len(planned)}")
    for index, (config, stop_metric, seed) in enumerate(planned, start=1):
        if args.skip_existing and run_is_complete(config, stop_metric, args.max_estimators, seed):
            print(f"({index}/{len(planned)}) SKIP {config}/stop_{stop_metric}/seed{seed}: already run.")
            continue
        print(
            f"\n({index}/{len(planned)}) === {config} stop_metric={stop_metric} "
            f"cap={args.max_estimators} seed={seed} ==="
        )
        run_convergence(config, stop_metric, seed, args.max_estimators, skip_existing=False)

    print("\nAll planned runs complete.")
    print(f"Reports: {REPORT_DIR}")
    print("Test set evaluated: NO")


if __name__ == "__main__":
    main()
