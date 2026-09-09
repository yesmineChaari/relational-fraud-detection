"""Per-feature ablation of the four card1 relational summaries.

B1-card1 beats B0 by +0.00630 validation PR-AUC under the converged protocol.
That gain has now survived every challenge raised against it: the paired
bootstrap (sampling noise), the seed-variance panel (refit noise), the
estimator-cap extension (early-stopping artifact) and the permuted-entity null
(split-search artifact -- no permuted variant significantly beat B0, so the
gain is attributable to genuine card1 entity history). What none of those
answer is *which* history: B1-card1 adds four columns at once, and the headline
number is silent on whether one of them carries the effect or all four
contribute.

This module localises the gain by training the ablation variants in two
directions, because with the four summaries correlated 0.83-0.95 either
direction alone is misleading:

  * Singleton (B0 + exactly one summary, 436 predictors) -- what a feature
    carries on its own.
  * Leave-one-out (full B1-card1 minus exactly one summary, 438 predictors) --
    what a feature carries given the other three.

Run only leave-one-out and the redundancy between correlated summaries buys
four null results: dropping any single column costs nothing because the other
three reconstruct it, and nothing is localised. Run only singletons and a
strong result cannot distinguish four views of one underlying factor from four
genuinely additive signals. Together the two directions separate "one summary
carries the gain" from "any one of them suffices".

Design choices
--------------

Comparison protocol: converged (cap 15,000, average_precision patience, seed
42), the same corrected protocol the permuted-entity null used. Singletons are
measured against the converged B0 reference; leave-one-out variants against the
converged B1-card1 reference. Comparing against the stale cap-6,000 frozen
artifacts would reintroduce the early-stopping confound the estimator-cap
investigation removed.

What varies across runs: only which of the four relational columns enter the
predictor manifest. LightGBM's random_state stays pinned at the frozen seed
(42) and every other parameter is inherited from frozen B0, so a measured
difference cannot be conflated with ordinary refit noise -- the same
one-thing-at-a-time discipline the seed-variance, convergence and permuted-null
panels use.

`--seed` is the one deliberate exception. Every cell of the eight-run panel is
a single fit, and the seed-variance panel established that refits alone move
these deltas by ~0.0005 on the clean stratum. Re-running one ablation variant
across several seeds measures that noise directly for that variant, and a run
under a non-default seed records `random_state` as a varied parameter in its
own metadata so it can never be mistaken for a panel cell.

Feature artifacts are read, never rebuilt. The ablation varies the predictor
manifest, not the feature computation, so it consumes the frozen
relational_features_card1.parquet exactly as B1-card1 did. Only the subset
selected into the manifest is ever attached to the training frames, so a
dropped column cannot reach the model through any path.

Nothing here writes to any frozen or converged artifact. Ablation runs land
under their own report and model trees, and every artifact this investigation
reads or is compared against is hash-pinned before training and re-checked
afterwards.

Usage:
    python -m src.models.train_lightgbm_ablation
    python -m src.models.train_lightgbm_ablation --mode singleton --feature prior_count
    python -m src.models.train_lightgbm_ablation --skip-existing
    python -m src.models.train_lightgbm_ablation --mode loo --feature prior_count_24h         --seed 202 --seed 707 --skip-existing
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
from lightgbm import LGBMClassifier

from src.features.build_relational_features import (
    EXPECTED_ROWS as RELATIONAL_EXPECTED_ROWS,
    _feature_names,
    _output_path as _rel_output_path,
    _metadata_path as _rel_metadata_path,
)
from src.models.train_lightgbm_baseline import (
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
    EXPECTED_B0_FEATURE_COUNT,
    FROZEN_PARAMETER_NAMES,
    LOG_EVALUATION_PERIOD,
    apply_frozen_category_mappings,
    assert_protected_artifacts_unchanged,
    attach_relational_features,
    load_feature_builder_metadata,
    load_frozen_b0_metadata,
    load_frozen_category_mappings,
    snapshot_protected_artifacts,
    validate_model_columns_against_frozen_b0,
    validate_relational_merge,
)
from src.models.train_lightgbm_g1 import B1_CARD1_PROTECTED_PATHS
from src.models.train_lightgbm_convergence_check import (
    DEFAULT_MAX_ESTIMATORS as CONVERGED_MAX_ESTIMATORS,
    resolve_run_paths as resolve_convergence_run_paths,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

RELATION = "card1"
STOP_METRIC = "average_precision"
EVAL_METRIC_ORDER = ["average_precision", "auc"]

REPORT_DIR = ROOT_DIR / "reports" / "ablation" / RELATION
MODEL_DIR = ROOT_DIR / "models" / "ablation"

# The four summaries under ablation, in the builder's canonical order. Subsets
# are always emitted in this order so a variant's predictor manifest is a
# deterministic function of which features it includes, never of CLI argument
# order.
ABLATION_FEATURES = _feature_names(RELATION)

SINGLETON = "singleton"
LEAVE_ONE_OUT = "loo"
MODES = [SINGLETON, LEAVE_ONE_OUT]

# Short path/CLI keys, derived from the feature names rather than restated, so
# a rename in the feature builder cannot leave a stale key behind here.
FEATURE_KEYS = {name: name.removeprefix(f"{RELATION}_") for name in ABLATION_FEATURES}
KEY_TO_FEATURE = {key: name for name, key in FEATURE_KEYS.items()}

# Converged (cap 15,000, average_precision patience, seed 42) references.
# Singletons are read against B0, leave-one-out variants against B1-card1.
# Reusing the convergence-check module's own path resolver keeps these paths
# from ever drifting apart from what that module actually wrote.
B0_CONVERGED_PATHS = resolve_convergence_run_paths(
    "b0", STOP_METRIC, CONVERGED_MAX_ESTIMATORS, RANDOM_SEED
)
B1_CARD1_CONVERGED_PATHS = resolve_convergence_run_paths(
    "b1_card1", STOP_METRIC, CONVERGED_MAX_ESTIMATORS, RANDOM_SEED
)

# The reference each mode is interpreted against, named once here so the
# trainer's metadata and the comparison module cannot disagree about it.
MODE_REFERENCE = {
    SINGLETON: "b0_converged",
    LEAVE_ONE_OUT: "b1_card1_converged",
}

# Every artifact this investigation reads from or is compared against: the
# frozen cap-6,000 originals, the converged cap-15,000 references, and the
# card1 relational feature artifacts. None may ever be written to by this
# module.
ALL_PROTECTED_PATHS = [
    *B0_PROTECTED_PATHS,
    *B1_CARD1_PROTECTED_PATHS,
    *(p for key, p in B0_CONVERGED_PATHS.items() if key != "run_dir"),
    *(p for key, p in B1_CARD1_CONVERGED_PATHS.items() if key != "run_dir"),
    _rel_output_path(RELATION),
    _rel_metadata_path(RELATION),
]


def resolve_feature(feature: str) -> str:
    """Accept either a short key (`prior_count_24h`) or a full column name."""
    if feature in KEY_TO_FEATURE:
        return KEY_TO_FEATURE[feature]
    if feature in FEATURE_KEYS:
        return feature
    raise ValueError(
        f"Unknown ablation feature: {feature!r}. Supported: {sorted(KEY_TO_FEATURE)}."
    )


def feature_subset(mode: str, feature: str) -> list[str]:
    """The relational columns a variant trains on, in canonical order."""
    if mode not in MODES:
        raise ValueError(f"Unknown ablation mode: {mode!r}. Supported: {MODES}.")
    target = resolve_feature(feature)
    if mode == SINGLETON:
        subset = [target]
    else:
        subset = [name for name in ABLATION_FEATURES if name != target]
    expected_size = 1 if mode == SINGLETON else len(ABLATION_FEATURES) - 1
    if len(subset) != expected_size:
        raise AssertionError(
            f"{mode} subset for {target} must hold {expected_size} feature(s); "
            f"got {len(subset)}."
        )
    return subset


def resolve_run_paths(
    mode: str, feature: str, seed: int = RANDOM_SEED
) -> dict[str, Path]:
    if mode not in MODES:
        raise ValueError(f"Unknown ablation mode: {mode!r}. Supported: {MODES}.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")
    key = FEATURE_KEYS[resolve_feature(feature)]
    run_dir = REPORT_DIR / f"{mode}_{key}" / f"cap{CONVERGED_MAX_ESTIMATORS}_seed{seed}"
    return {
        "run_dir": run_dir,
        "model": MODEL_DIR / (
            f"lightgbm_ablation_{RELATION}__{mode}_{key}"
            f"_cap{CONVERGED_MAX_ESTIMATORS}_seed{seed}.txt"
        ),
        "metrics": run_dir / "metrics.json",
        "metadata": run_dir / "metadata.json",
        "feature_importance": run_dir / "feature_importance.csv",
        "validation_predictions": run_dir / "validation_predictions.parquet",
        "learning_curve": run_dir / "learning_curve.csv",
    }


def run_is_complete(mode: str, feature: str, seed: int = RANDOM_SEED) -> bool:
    """True when every artifact this run publishes is already on disk."""
    paths = resolve_run_paths(mode, feature, seed)
    return all(path.exists() for key, path in paths.items() if key != "run_dir")


def planned_runs() -> list[tuple[str, str]]:
    """The full eight-run panel: four singletons then four leave-one-out."""
    return [(mode, name) for mode in MODES for name in ABLATION_FEATURES]


def build_ablation_feature_manifest(
    b0_features: list[str],
    subset_names: list[str],
) -> list[str]:
    """B0's frozen manifest plus exactly the ablation subset, in canonical order.

    Deliberately not `build_b1_feature_manifest`, which asserts exactly four
    additional predictors -- the whole point here is to add one or three.
    """
    if not b0_features or len(b0_features) != len(set(b0_features)):
        raise ValueError("B0 feature manifest is empty or contains duplicates.")
    if not subset_names:
        raise ValueError("An ablation variant must carry at least one relational feature.")
    if len(subset_names) != len(set(subset_names)):
        raise ValueError("Ablation subset contains duplicates.")
    unknown = sorted(set(subset_names) - set(ABLATION_FEATURES))
    if unknown:
        raise ValueError(f"Ablation subset contains unknown features: {unknown}.")
    if len(subset_names) >= len(ABLATION_FEATURES):
        raise ValueError(
            "An ablation variant must omit at least one relational feature; "
            "the full four-feature manifest is B1-card1 itself."
        )
    overlap = sorted(set(b0_features) & set(subset_names))
    if overlap:
        raise ValueError(f"B0 feature manifest already contains relational features: {overlap}.")

    ordered_subset = [name for name in ABLATION_FEATURES if name in set(subset_names)]
    manifest = [*b0_features, *ordered_subset]
    if len(manifest) != len(b0_features) + len(subset_names):
        raise AssertionError("Ablation manifest lost or duplicated a predictor.")
    return manifest


def load_ablation_datasets(
    subset_names: list[str],
    b0_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Attach only the ablation subset to the frozen B0 train/validation frames.

    The relational parquet is validated in full (all four columns, dtypes,
    one-to-one membership against the model index) before the subset is
    selected, so an ablation run gets exactly the same integrity guarantees
    B1-card1 did -- it simply carries fewer columns forward.
    """
    feat_names = _feature_names(RELATION)
    rel_path = _rel_output_path(RELATION)
    if not rel_path.exists():
        raise FileNotFoundError(
            f"Relational features not found for relation {RELATION!r}. "
            f"Run: python -m src.features.build_relational_features --relation {RELATION}\n"
            f"Expected: {rel_path}"
        )
    load_feature_builder_metadata(RELATION)

    model_index = pd.read_parquet(MODEL_DATASET_PATH, columns=["TransactionID", "split"])
    validate_split_counts(model_index, "model_dataset.parquet ablation index")
    relational_df = pd.read_parquet(rel_path)
    if len(relational_df) != RELATIONAL_EXPECTED_ROWS:
        raise ValueError(
            f"Expected {RELATIONAL_EXPECTED_ROWS:,} relational rows; got {len(relational_df):,}."
        )
    merged_index = validate_relational_merge(model_index, relational_df, feat_names)
    del relational_df
    gc.collect()

    train_df, validation_df, _ = load_model_dataset()
    # Only the subset is attached: a dropped column never enters the frame, so
    # it cannot reach the model through the manifest or any later selection.
    train_with_relations = attach_relational_features(
        train_df, merged_index, "train", subset_names
    )
    validation_with_relations = attach_relational_features(
        validation_df, merged_index, "validation", subset_names
    )
    if len(train_with_relations) != EXPECTED_SPLIT_COUNTS["train"]:
        raise AssertionError("Ablation train row count changed.")
    if len(validation_with_relations) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise AssertionError("Ablation validation row count changed.")

    dropped = [name for name in ABLATION_FEATURES if name not in set(subset_names)]
    for frame_name, frame in (
        ("train", train_with_relations),
        ("validation", validation_with_relations),
    ):
        present = sorted(set(dropped) & set(frame.columns))
        if present:
            raise AssertionError(
                f"Ablated features leaked into the {frame_name} frame: {present}."
            )

    b0_features = list(b0_metadata["feature_columns"])
    validate_model_columns_against_frozen_b0(
        list(train_with_relations.columns), b0_features, feat_names
    )
    validate_model_columns_against_frozen_b0(
        list(validation_with_relations.columns), b0_features, feat_names
    )
    feature_columns = build_ablation_feature_manifest(b0_features, subset_names)
    return train_with_relations, validation_with_relations, feature_columns


def validate_ablation_configuration(
    model: LGBMClassifier,
    b0_metadata: dict[str, Any],
    feature_columns: list[str],
    subset_names: list[str],
    seed: int = RANDOM_SEED,
) -> None:
    """Assert only the predictor manifest (and, on a seed panel, the seed) moved.

    The cap is pinned at the converged reference's value (15,000). At the
    default seed this run differs from that reference in exactly one way --
    which of the four relational columns entered the manifest. On a seed panel
    `random_state` is the second thing allowed to move, and it is checked
    against the requested seed rather than against frozen B0, the same
    exemption train_seed_variants makes for the same reason.
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
                f"Ablation parameter {name} differs from frozen B0: "
                f"actual={actual.get(name)!r}, B0={reference[name]!r}."
            )
    if actual.get("random_state") != seed:
        raise ValueError(
            f"Ablation run must carry random_state={seed}; "
            f"got {actual.get('random_state')!r}."
        )
    if actual.get("n_estimators") != CONVERGED_MAX_ESTIMATORS:
        raise ValueError(
            f"Ablation run must carry n_estimators={CONVERGED_MAX_ESTIMATORS:,}; "
            f"got {actual.get('n_estimators')!r}."
        )
    if not np.isclose(
        actual.get("scale_pos_weight"), b0_metadata.get("scale_pos_weight"), rtol=0.0, atol=0.0
    ):
        raise ValueError("Ablation class weight differs from frozen B0.")

    expected_count = EXPECTED_B0_FEATURE_COUNT + len(subset_names)
    if len(feature_columns) != expected_count:
        raise ValueError(
            f"Ablation predictor count is wrong: expected {expected_count} "
            f"({EXPECTED_B0_FEATURE_COUNT} frozen B0 + {len(subset_names)} relational); "
            f"got {len(feature_columns)}."
        )


def run_ablation(
    mode: str,
    feature: str,
    seed: int = RANDOM_SEED,
    skip_existing: bool = False,
) -> dict[str, Any]:
    target = resolve_feature(feature)
    key = FEATURE_KEYS[target]
    label = f"ablation_{RELATION}/{mode}_{key}/seed_{seed}"
    paths = resolve_run_paths(mode, target, seed)
    if skip_existing and run_is_complete(mode, target, seed):
        print(f"[{label}] Already complete; skipping.")
        return {}

    subset_names = feature_subset(mode, target)
    protected_before = snapshot_protected_artifacts(ALL_PROTECTED_PATHS)
    b0_metadata = load_frozen_b0_metadata()

    print(f"[{label}] Loading frozen B0 data and the card1 relational features...")
    train_df, validation_df, feature_columns = load_ablation_datasets(subset_names, b0_metadata)

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
    model = build_lightgbm_model(scale_pos_weight, n_estimators=CONVERGED_MAX_ESTIMATORS)
    model.set_params(random_state=seed)
    validate_ablation_configuration(
        model, b0_metadata, feature_columns, subset_names, seed
    )

    print(
        f"[{label}] Relational features: {subset_names}  "
        f"predictors: {len(feature_columns):,}  cap: {CONVERGED_MAX_ESTIMATORS:,}  "
        f"training seed: {seed}"
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
        eval_metric=EVAL_METRIC_ORDER,
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
        maximum_estimators=CONVERGED_MAX_ESTIMATORS,
    )

    validation_scores = model.predict_proba(
        X_validation, num_iteration=model.best_iteration_
    )[:, 1]
    if len(validation_scores) != EXPECTED_SPLIT_COUNTS[EVALUATION_SPLIT]:
        raise AssertionError(f"[{label}] Validation prediction count is incorrect.")
    if not np.isfinite(validation_scores).all():
        raise AssertionError(f"[{label}] Validation predictions contain non-finite values.")

    metrics = evaluate_validation(y_validation, validation_scores)
    metrics.update(
        {
            "model": f"ablation_{RELATION}",
            "relation": RELATION,
            "ablation_mode": mode,
            "ablated_feature": target,
            "ablated_feature_key": key,
            "relational_features_used": subset_names,
            "relational_features_dropped": [
                name for name in ABLATION_FEATURES if name not in set(subset_names)
            ],
            "number_of_predictors": len(feature_columns),
            "reference_configuration": MODE_REFERENCE[mode],
            "random_seed": int(seed),
            "weighting": "weighted",
            "scale_pos_weight": float(scale_pos_weight),
            "maximum_estimators": int(CONVERGED_MAX_ESTIMATORS),
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
            "stop_metric": STOP_METRIC,
        }
    )

    feature_importance = build_feature_importance(model)
    if len(feature_importance) != len(feature_columns):
        raise AssertionError(f"[{label}] Feature importance does not match its manifest.")
    validation_predictions = validation_metadata.copy()
    validation_predictions["prediction"] = validation_scores

    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(paths["model"]), num_iteration=model.best_iteration_)
    write_json(paths["metrics"], metrics)
    feature_importance.to_csv(paths["feature_importance"], index=False)
    validation_predictions.to_parquet(
        paths["validation_predictions"], index=False, engine="pyarrow", compression="snappy"
    )
    learning_curve.to_csv(paths["learning_curve"], index=False)

    metadata = {
        "experiment_name": "per_feature_relational_ablation",
        "relation": RELATION,
        "ablation_mode": mode,
        "ablated_feature": target,
        "relational_features_used": subset_names,
        "relational_features_dropped": [
            name for name in ABLATION_FEATURES if name not in set(subset_names)
        ],
        "random_seed": int(seed),
        "varied_parameter": (
            "relational predictor manifest only (LightGBM random_state pinned)"
            if seed == RANDOM_SEED
            else "relational predictor manifest and random_state (seed panel)"
        ),
        "held_frozen": (
            "B0 feature manifest, train-only categorical mappings, class weight, "
            "estimator cap, early-stopping patience, stopping metric and every "
            "other LightGBM parameter -- identical to the converged B1-card1 "
            "reference except which of the four relational columns are present"
        ),
        "reference_configuration": MODE_REFERENCE[mode],
        "feature_count": len(feature_columns),
        "categorical_feature_count": len(categorical_columns),
        "categorical_mappings_sha256": mapping_sha256,
        "relational_features_source": repository_relative(_rel_output_path(RELATION)),
        "relational_features_rebuilt": False,
        "scale_pos_weight": float(scale_pos_weight),
        "lightgbm_parameters": model.get_params(deep=False),
        "max_estimators": int(CONVERGED_MAX_ESTIMATORS),
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "early_stopping_metric": STOP_METRIC,
        "best_iteration": int(model.best_iteration_),
        "learning_curve_summary": learning_curve_summary,
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
        protected_before,
        ALL_PROTECTED_PATHS,
        label="frozen/converged B0, B1-card1 and the card1 relation",
    )

    print(
        f"[{label}] PR-AUC {metrics['pr_auc']:.12f}  ROC-AUC {metrics['roc_auc']:.12f}  "
        f"best_iteration {model.best_iteration_:,} of {CONVERGED_MAX_ESTIMATORS:,}  "
        f"early stopping {'YES' if metrics['early_stopping_triggered'] else 'NO'}"
    )
    print(f"[{label}] Elapsed: {metadata['training_seconds'] / 60:.1f} min")
    print(f"[{label}] Protected artifacts unchanged: YES")
    return metrics


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the per-feature card1 relational ablation panel -- four "
            "singleton and four leave-one-out variants, at the converged "
            "(cap 15,000) estimator cap."
        )
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=MODES,
        help="Ablation direction to run; repeatable. Defaults to both.",
    )
    parser.add_argument(
        "--feature",
        action="append",
        help=(
            "Relational feature to ablate; repeatable. Accepts a short key "
            f"({', '.join(sorted(KEY_TO_FEATURE))}) or the full column name. "
            "Defaults to all four."
        ),
    )
    parser.add_argument(
        "--seed",
        action="append",
        type=int,
        help=(
            "LightGBM random_state to train under; repeatable. Defaults to the "
            f"frozen seed ({RANDOM_SEED}). Use several to measure refit noise "
            "on an ablation variant."
        ),
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run whose artifacts already exist, so an interrupted panel resumes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    modes = args.mode or MODES
    features = [resolve_feature(f) for f in args.feature] if args.feature else ABLATION_FEATURES
    seeds = args.seed or [RANDOM_SEED]
    planned = [(mode, name, seed) for mode in modes for name in features for seed in seeds]

    print(f"Planned runs: {len(planned)}")
    for index, (mode, name, seed) in enumerate(planned, start=1):
        print(
            f"\n({index}/{len(planned)}) === {mode} :: {FEATURE_KEYS[name]} :: seed {seed} ==="
        )
        run_ablation(mode, name, seed=seed, skip_existing=args.skip_existing)

    print("\nAll planned runs complete.")
    print(f"Reports: {REPORT_DIR}")
    print("Test set evaluated: NO")


if __name__ == "__main__":
    main()
