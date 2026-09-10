"""Retrain the card1 ablation panel at a budget fixed before any result was seen.

The published ablation verdict and its equal-budget re-derivation disagree, and
both are biased -- in opposite directions -- by how many boosting rounds each arm
was granted. Reporting the argmax of a validation curve rewards whichever arm ran
longer; re-scoring everything at the panel *minimum* truncates whichever arm
wanted more. The two bracket the answer and neither settles it, and no further
reading of the artifacts on disk can close the gap, because the gap is in how the
runs were trained rather than in how they were read.

This module removes the selection instead of equalising it. Every run trains
exactly `FIXED_BUDGET` rounds with **early stopping disabled**, and is reported
at that round. No argmax over a validation curve enters the comparison anywhere.

Why 10,000. The highest best-iteration anywhere in the existing panel is 9,137,
so at 10,000 no arm is truncated relative to any optimum it reached under the
argmax protocol, and every arm receives identical compute. It is a round number
below the established 15,000 cap, chosen to sit above the whole panel rather than
tuned to any cell. It was fixed and recorded before this file was written.

Ten runs, because the references must move too. The converged B0 and B1-card1
references were themselves early-stopped, so they cannot serve as references for
a no-argmax comparison -- comparing a fixed-budget variant against an
early-stopped reference would reintroduce exactly the asymmetry being removed.
Both are retrained here at the same fixed budget.

Only `n_estimators` and the absence of early stopping differ from frozen B0.
Every other parameter, the feature manifest, the train-only categorical mappings
and the class weight are inherited unchanged, so a measured difference cannot be
attributed to anything else. The shared convergence-check validator enforces that
rather than a private copy of the same rules.

Runs are one process each: roughly 4.5 GB peak on a 16 GB machine means a
parallel sweep is an out-of-memory failure, not a speed-up. `--skip-existing`
resumes an interrupted panel instead of recomputing published cells.

Nothing here writes to a frozen or converged artifact. Outputs land under their
own report and model trees, and the frozen set is hash-checked before and after
every run.
"""

from __future__ import annotations

import argparse
import gc
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import sklearn

from src.config.paths import ROOT_DIR
from src.models.train_lightgbm_ablation import (
    ABLATION_FEATURES,
    FEATURE_KEYS,
    LEAVE_ONE_OUT,
    MODES,
    RELATION,
    SINGLETON,
    feature_subset,
    load_ablation_datasets,
    resolve_feature,
)
from src.models.train_lightgbm_baseline import (
    EXPECTED_SPLIT_COUNTS,
    RANDOM_SEED,
    assert_supported_model_dtypes,
    build_feature_importance,
    build_learning_curve,
    build_lightgbm_model,
    calculate_scale_pos_weight,
    evaluate_validation,
    identify_categorical_columns,
    write_json,
)
from src.models.train_lightgbm_convergence_check import (
    ALL_PROTECTED_PATHS,
    LOADERS,
    validate_convergence_configuration,
)
from src.models.train_lightgbm_relational import (
    EVALUATION_SPLIT,
    LOG_EVALUATION_PERIOD,
    apply_frozen_category_mappings,
    assert_protected_artifacts_unchanged,
    load_frozen_b0_metadata,
    load_frozen_category_mappings,
    snapshot_protected_artifacts,
)

# Fixed before this module existed and before any result was seen. Changing it
# invalidates the pre-registration, so it is a constant rather than a default
# anybody is invited to tune.
FIXED_BUDGET = 10_000

STOP_METRIC = "average_precision"
EVAL_METRIC_ORDER = ["average_precision", "auc"]

REPORT_DIR = ROOT_DIR / "reports" / "fixed_budget" / RELATION
MODEL_DIR = ROOT_DIR / "models" / "fixed_budget"

REFERENCES = ("b0", "b1_card1")

# Which reference each ablation mode is measured against, mirroring the
# published panel so the interpretation rule transfers unchanged.
MODE_REFERENCE = {SINGLETON: "b0", LEAVE_ONE_OUT: "b1_card1"}

PROTOCOL = "fixed_budget_no_early_stopping"


def planned_runs() -> list[str]:
    """The ten runs: two references, then four singletons and four leave-one-out."""
    cells = [f"{mode}_{FEATURE_KEYS[feature]}" for mode in MODES for feature in ABLATION_FEATURES]
    return [*REFERENCES, *cells]


def split_run_name(run_name: str) -> tuple[str, str | None]:
    """('b0', None) for a reference, ('singleton', 'prior_count') for a cell."""
    if run_name in REFERENCES:
        return run_name, None
    valid_keys = set(FEATURE_KEYS.values())
    for mode in MODES:
        prefix = f"{mode}_"
        if run_name.startswith(prefix):
            key = run_name[len(prefix) :]
            if key not in valid_keys:
                raise ValueError(
                    f"Unknown ablation feature {key!r} in run {run_name!r}. "
                    f"Known features: {sorted(valid_keys)}."
                )
            return mode, key
    raise ValueError(f"Unknown run {run_name!r}. Known runs: {planned_runs()}.")


def resolve_run_paths(
    run_name: str, budget: int = FIXED_BUDGET, seed: int = RANDOM_SEED
) -> dict[str, Path]:
    if budget <= 0:
        raise ValueError("budget must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    split_run_name(run_name)
    run_dir = REPORT_DIR / run_name / f"fixed{budget}_seed{seed}"
    return {
        "run_dir": run_dir,
        "model": MODEL_DIR / f"lightgbm_fixed_{RELATION}__{run_name}_fixed{budget}_seed{seed}.txt",
        "metrics": run_dir / "metrics.json",
        "metadata": run_dir / "metadata.json",
        "feature_importance": run_dir / "feature_importance.csv",
        "validation_predictions": run_dir / "validation_predictions.parquet",
        "learning_curve": run_dir / "learning_curve.csv",
    }


def run_is_complete(run_name: str, budget: int = FIXED_BUDGET, seed: int = RANDOM_SEED) -> bool:
    paths = resolve_run_paths(run_name, budget, seed)
    return all(path.exists() for key, path in paths.items() if key != "run_dir")


def load_datasets_for(
    run_name: str, b0_metadata: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str] | None]:
    """Train and validation frames for one run, plus its relational subset."""
    mode, feature_key = split_run_name(run_name)
    if feature_key is None:
        train_df, validation_df, feature_columns = LOADERS[mode](b0_metadata)
        return train_df, validation_df, feature_columns, None
    subset_names = feature_subset(mode, feature_key)
    train_df, validation_df, feature_columns = load_ablation_datasets(subset_names, b0_metadata)
    return train_df, validation_df, feature_columns, subset_names


def train_one(
    run_name: str,
    budget: int = FIXED_BUDGET,
    seed: int = RANDOM_SEED,
    skip_existing: bool = False,
) -> None:
    label = f"fixed{budget}/{run_name}/seed{seed}"
    if skip_existing and run_is_complete(run_name, budget, seed):
        print(f"[{label}] Already complete; skipping.")
        return

    mode, feature_key = split_run_name(run_name)
    paths = resolve_run_paths(run_name, budget, seed)
    protected_before = snapshot_protected_artifacts(ALL_PROTECTED_PATHS)
    b0_metadata = load_frozen_b0_metadata()

    print(f"[{label}] Loading data...")
    train_df, validation_df, feature_columns, subset_names = load_datasets_for(
        run_name, b0_metadata
    )

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

    X_train = pd.DataFrame({column: train_df.pop(column) for column in feature_columns}, copy=False)
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
    model = build_lightgbm_model(scale_pos_weight, n_estimators=budget)
    model.set_params(random_state=seed)
    # Only n_estimators and random_state may differ from frozen B0. Reused
    # rather than reimplemented so this panel cannot drift from the rule the
    # convergence check already enforces.
    validate_convergence_configuration(model, b0_metadata, seed, budget)

    print(
        f"[{label}] Predictors: {len(feature_columns):,}  budget: {budget:,} rounds  "
        f"early stopping: disabled"
    )
    evaluation_results: dict[str, dict[str, list[float]]] = {}
    # No early_stopping callback: the whole point is that no round is selected
    # by watching a validation metric. The curve is still recorded, so the run
    # can be inspected afterwards without that inspection feeding the estimate.
    callbacks = [
        lightgbm.record_evaluation(evaluation_results),
        lightgbm.log_evaluation(period=LOG_EVALUATION_PERIOD),
    ]
    print(f"[{label}] Training {budget:,} rounds...")
    started_at = datetime.now(timezone.utc)
    model.fit(
        X_train,
        y_train,
        eval_X=X_validation,
        eval_y=y_validation,
        eval_names=[EVALUATION_SPLIT],
        eval_metric=EVAL_METRIC_ORDER,
        categorical_feature=categorical_columns,
        callbacks=callbacks,
    )
    finished_at = datetime.now(timezone.utc)

    learning_curve = build_learning_curve(evaluation_results)
    if len(learning_curve) != budget:
        raise AssertionError(
            f"[{label}] Learning curve has {len(learning_curve):,} rows, expected the "
            f"full {budget:,}. Early stopping must not be active in this panel."
        )

    # Everything is read at the fixed budget, never at an argmax.
    validation_scores = model.predict_proba(X_validation, num_iteration=budget)[:, 1]
    if len(validation_scores) != EXPECTED_SPLIT_COUNTS[EVALUATION_SPLIT]:
        raise AssertionError(f"[{label}] Validation prediction count is incorrect.")
    if not np.isfinite(validation_scores).all():
        raise AssertionError(f"[{label}] Validation predictions contain non-finite values.")

    metrics = evaluate_validation(y_validation, validation_scores)
    curve_ap = learning_curve["validation_average_precision"]
    metrics.update(
        {
            "model": f"fixed_budget_{RELATION}",
            "relation": RELATION,
            "protocol": PROTOCOL,
            "run": run_name,
            "ablation_mode": None if subset_names is None else mode,
            "ablated_feature": None if subset_names is None else resolve_feature(feature_key),
            "relational_features_used": subset_names,
            "reference_configuration": (None if subset_names is None else MODE_REFERENCE[mode]),
            "number_of_predictors": len(feature_columns),
            "random_seed": int(seed),
            "weighting": "weighted",
            "scale_pos_weight": float(scale_pos_weight),
            "fixed_budget_trees": int(budget),
            "reported_iteration": int(budget),
            "early_stopping_used": False,
            "stop_metric": None,
            # Recorded for inspection only. This is what the argmax protocol
            # would have reported; it is deliberately not what is reported here.
            "argmax_iteration_not_used": int(curve_ap.idxmax()) + 1,
            "argmax_average_precision_not_used": float(curve_ap.max()),
            "test_evaluated": False,
        }
    )

    feature_importance = build_feature_importance(model)
    if len(feature_importance) != len(feature_columns):
        raise AssertionError(f"[{label}] Feature importance does not match its manifest.")
    validation_predictions = validation_metadata.copy()
    validation_predictions["prediction"] = validation_scores

    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(paths["model"]), num_iteration=budget)
    write_json(paths["metrics"], metrics)
    feature_importance.to_csv(paths["feature_importance"], index=False)
    validation_predictions.to_parquet(
        paths["validation_predictions"], index=False, engine="pyarrow", compression="snappy"
    )
    learning_curve.to_csv(paths["learning_curve"], index=False)

    metadata = {
        "experiment_name": "fixed_budget_relational_ablation",
        "relation": RELATION,
        "protocol": PROTOCOL,
        "run": run_name,
        "relational_features_used": subset_names,
        "random_seed": int(seed),
        "varied_parameter": (
            "estimator budget and the removal of early stopping; for ablation "
            "cells also the relational predictor manifest"
        ),
        "held_frozen": (
            "B0 feature manifest, train-only categorical mappings, class weight "
            "and every other LightGBM parameter"
        ),
        "fixed_budget_trees": int(budget),
        "early_stopping_used": False,
        "why_fixed_budget": (
            "The argmax protocol rewards whichever arm was granted more rounds "
            "and the panel-minimum protocol truncates whichever arm wanted more. "
            "A common budget above every arm's observed optimum removes the "
            "selection rather than equalising it."
        ),
        "feature_count": len(feature_columns),
        "categorical_feature_count": len(categorical_columns),
        "categorical_mappings_sha256": mapping_sha256,
        "scale_pos_weight": float(scale_pos_weight),
        "lightgbm_parameters": model.get_params(deep=False),
        "evaluation_split": EVALUATION_SPLIT,
        "validation_pr_auc": float(metrics["pr_auc"]),
        "validation_roc_auc": float(metrics["roc_auc"]),
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
        protected_before, ALL_PROTECTED_PATHS, label="frozen and converged references"
    )
    print(
        f"[{label}] Done in {(finished_at - started_at).total_seconds() / 60:.1f} min  "
        f"PR-AUC {metrics['pr_auc']:.6f}  ROC-AUC {metrics['roc_auc']:.6f}"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.train_ablation_fixed_budget",
        description="Retrain the ablation panel at a fixed budget, no early stopping.",
    )
    parser.add_argument("--run", action="append", help="Run only these runs.")
    parser.add_argument(
        "--max-estimators",
        type=int,
        default=FIXED_BUDGET,
        help=f"Estimator budget (default {FIXED_BUDGET}, the pre-registered value).",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--list", action="store_true", help="Show the panel and its status.")
    args = parser.parse_args(argv)

    runs = args.run or planned_runs()
    unknown = [run for run in runs if run not in planned_runs()]
    if unknown:
        raise SystemExit(f"Unknown run(s): {unknown}. Known: {planned_runs()}")

    if args.list:
        for run in runs:
            state = (
                "complete" if run_is_complete(run, args.max_estimators, args.seed) else "pending"
            )
            print(f"{run:36s} {state}")
        return

    if args.max_estimators != FIXED_BUDGET:
        print(
            f"WARNING: budget {args.max_estimators:,} is not the pre-registered "
            f"{FIXED_BUDGET:,}. Results are a smoke test, not the panel.\n"
        )

    for index, run in enumerate(runs, start=1):
        print(f"\n=== [{index}/{len(runs)}] {run} ===")
        train_one(run, args.max_estimators, args.seed, args.skip_existing)


if __name__ == "__main__":
    main()
