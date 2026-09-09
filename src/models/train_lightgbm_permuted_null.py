"""Permuted-entity null control for the B1-card1 gain.

B1-card1 beats B0 by a validation PR-AUC gain that survives the paired
bootstrap (sampling noise), survives the seed-variance panel (refit noise),
and persists once the estimator cap is lifted -- the cap extension revised
the capped headline figures rather than overturning the gain, moving it from
+0.00590 to +0.00630. None of those checks rule out the remaining
explanation, though: adding four numeric
columns changes LightGBM's split search, and over thousands of boosting
rounds that can move validation PR-AUC on its own, independent of whether the
columns carry real entity history. The G1 dilution finding (a random
32-column block costing -0.01957 against B0) is the project's own evidence
that this failure mode is real at a larger column count.

This module answers the question directly: permute `card1` across
transactions, breaking the correspondence between a transaction and its
entity, then build the same four relational features on that permutation
with the existing feature builder unchanged, and train under the identical
frozen configuration. If the permuted variant lands at or below B0, the gain
is attributable to genuine entity history. If it recovers a meaningful share
of the gain, the effect is substantially structural.

Design choices
--------------

Comparison protocol: converged (cap 15,000, average_precision patience), not
the frozen cap-6,000 artifacts. The estimator-cap investigation
(train_lightgbm_convergence_check.py) found the capped headline numbers do
not survive convergence, and this control was explicitly blocked pending that
fix ("run under several seeds with the corrected protocol"). Comparing a
permuted run against a stale cap-6,000 reference would reintroduce the exact
early-stopping confound that investigation removed, so every reference here
is the converged (cap 15,000, stop_average_precision, seed 42) artifact.

What varies across runs: only the permutation seed. LightGBM's own
random_state stays pinned at the frozen seed (42), identical to the converged
B1-card1 reference run -- the same discipline seed_variance and
convergence_check use of moving exactly one thing at a time. This isolates
the measured effect to the entity permutation and prevents it from being
conflated with ordinary refit noise.

Global permutation, not within-split. G1's shuffled-embedding control
permutes within each split because the GraphSAGE encoder is fit on train rows
and a global shuffle would move train-fitted rows across split boundaries.
card1 has no such fitting step -- it is a raw identifier -- and the relational
feature builder already treats entity history as continuous across train,
validation and test by design. A global permutation of the card1 column
preserves the exact multiset of card1 values (hence the exact per-entity
transaction-count distribution) and leaves TransactionDT untouched for every
row (hence the exact timestamp distribution); only the assignment of a
transaction to an entity is randomised.

Nothing here writes to any frozen or converged artifact. Permuted runs land
under their own report and model trees, and every artifact this investigation
reads or is compared against is hash-pinned before training and re-checked
afterwards.

Usage:
    python -m src.models.train_lightgbm_permuted_null
    python -m src.models.train_lightgbm_permuted_null --permutation-seed 1 --skip-existing
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
    INPUT_PATH as RELATIONAL_INPUT_PATH,
    RELATION_REGISTRY,
    _feature_names,
    _output_path as _rel_output_path,
    _metadata_path as _rel_metadata_path,
    build_relational_features_for,
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
    FROZEN_PARAMETER_NAMES,
    LOG_EVALUATION_PERIOD,
    apply_frozen_category_mappings,
    assert_protected_artifacts_unchanged,
    attach_relational_features,
    build_b1_feature_manifest,
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

REPORT_DIR = ROOT_DIR / "reports" / "permuted_null" / RELATION
MODEL_DIR = ROOT_DIR / "models" / "permuted_null"
PERMUTED_FEATURES_DIR = ROOT_DIR / "data" / "processed" / "permuted_null"

# Three permutation seeds, "several" per the ticket. Numbered plainly rather
# than reusing a training-seed panel, since these seeds control the entity
# permutation, not LightGBM's own randomness -- random_state stays pinned.
PERMUTATION_SEEDS = [1, 2, 3]

# Converged (cap 15,000, average_precision patience, seed 42) references --
# the corrected protocol this control was blocked pending. Reusing the
# convergence-check module's own path resolver keeps these paths from ever
# drifting apart from what that module actually wrote.
B0_CONVERGED_PATHS = resolve_convergence_run_paths(
    "b0", STOP_METRIC, CONVERGED_MAX_ESTIMATORS, RANDOM_SEED
)
B1_CARD1_CONVERGED_PATHS = resolve_convergence_run_paths(
    "b1_card1", STOP_METRIC, CONVERGED_MAX_ESTIMATORS, RANDOM_SEED
)

# Every artifact this investigation reads from or is compared against: the
# frozen cap-6,000 originals, the converged cap-15,000 references, and the
# real (non-permuted) card1 relational feature artifacts. None may ever be
# written to by this module.
ALL_PROTECTED_PATHS = [
    *B0_PROTECTED_PATHS,
    *B1_CARD1_PROTECTED_PATHS,
    *(p for key, p in B0_CONVERGED_PATHS.items() if key != "run_dir"),
    *(p for key, p in B1_CARD1_CONVERGED_PATHS.items() if key != "run_dir"),
    _rel_output_path(RELATION),
    _rel_metadata_path(RELATION),
]


def resolve_run_paths(permutation_seed: int) -> dict[str, Path]:
    if permutation_seed < 0:
        raise ValueError("permutation_seed must be non-negative.")
    run_dir = REPORT_DIR / f"permseed_{permutation_seed}" / f"cap{CONVERGED_MAX_ESTIMATORS}"
    return {
        "run_dir": run_dir,
        "model": MODEL_DIR / (
            f"lightgbm_permuted_{RELATION}__cap{CONVERGED_MAX_ESTIMATORS}"
            f"_permseed{permutation_seed}.txt"
        ),
        "metrics": run_dir / "metrics.json",
        "metadata": run_dir / "metadata.json",
        "feature_importance": run_dir / "feature_importance.csv",
        "validation_predictions": run_dir / "validation_predictions.parquet",
        "learning_curve": run_dir / "learning_curve.csv",
        "permuted_relational_features": (
            PERMUTED_FEATURES_DIR / f"relational_features_{RELATION}_permseed{permutation_seed}.parquet"
        ),
        "permuted_relational_metadata": run_dir / "relational_features_metadata.json",
    }


def run_is_complete(permutation_seed: int) -> bool:
    paths = resolve_run_paths(permutation_seed)
    return all(path.exists() for key, path in paths.items() if key != "run_dir")


def permute_card1(source_df: pd.DataFrame, permutation_seed: int) -> pd.DataFrame:
    """Shuffle card1 across every row, preserving its exact value multiset.

    A permutation of the column (not a resample) guarantees the count of
    transactions per card1 value -- the entity-size marginal -- is identical
    before and after. TransactionDT is untouched, so the timestamp marginal
    is trivially identical too. Only which entity a transaction is assigned
    to changes.
    """
    original = source_df["card1"].to_numpy(copy=True)
    rng = np.random.default_rng(permutation_seed)
    permuted = rng.permutation(original)

    before_counts = pd.Series(original).value_counts(dropna=False).sort_index()
    after_counts = pd.Series(permuted).value_counts(dropna=False).sort_index()
    if not before_counts.equals(after_counts):
        raise AssertionError(
            "card1 permutation changed the entity-size marginal; this must "
            "be an exact permutation."
        )

    permuted_df = source_df.copy()
    permuted_df["card1"] = permuted
    return permuted_df


def build_permuted_card1_features(permutation_seed: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Rebuild the four card1 relational features on a permuted entity assignment."""
    group_columns = RELATION_REGISTRY[RELATION]
    load_columns = ["TransactionID", "TransactionDT", "split", *group_columns]
    source_df = pd.read_parquet(RELATIONAL_INPUT_PATH, columns=load_columns)
    if len(source_df) != RELATIONAL_EXPECTED_ROWS:
        raise ValueError(
            f"Expected {RELATIONAL_EXPECTED_ROWS:,} source rows; got {len(source_df):,}."
        )

    permuted_source = permute_card1(source_df, permutation_seed)
    # Reuses the frozen, relation-agnostic builder unchanged: the null control
    # tests the entity assignment, not the feature computation itself.
    feature_df = build_relational_features_for(permuted_source, RELATION, group_columns)

    permutation_metadata = {
        "relation": RELATION,
        "group_columns": group_columns,
        "permutation_seed": int(permutation_seed),
        "permutation_method": (
            "Global permutation of the card1 column across all rows (train, "
            "validation and test together), independent of TransactionDT. "
            "Preserves the exact per-entity transaction-count distribution "
            "and the exact timestamp distribution; breaks the correspondence "
            "between a transaction and its entity."
        ),
        "feature_builder": "src.features.build_relational_features.build_relational_features_for",
        "feature_builder_unchanged": True,
        "target_labels_used": False,
        "real_relation_artifacts_untouched": [
            repository_relative(_rel_output_path(RELATION)),
            repository_relative(_rel_metadata_path(RELATION)),
        ],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return feature_df, permutation_metadata


def load_permuted_null_datasets(
    permutation_seed: int,
    b0_metadata: dict[str, Any],
    paths: dict[str, Path],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    feat_names = _feature_names(RELATION)
    feature_df, permutation_metadata = build_permuted_card1_features(permutation_seed)

    paths["permuted_relational_features"].parent.mkdir(parents=True, exist_ok=True)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(
        paths["permuted_relational_features"], index=False, engine="pyarrow", compression="snappy"
    )
    write_json(paths["permuted_relational_metadata"], permutation_metadata)

    model_index = pd.read_parquet(MODEL_DATASET_PATH, columns=["TransactionID", "split"])
    validate_split_counts(model_index, "model_dataset.parquet permuted-null index")
    merged_index = validate_relational_merge(model_index, feature_df, feat_names)

    train_df, validation_df, _ = load_model_dataset()
    train_with_relations = attach_relational_features(train_df, merged_index, "train", feat_names)
    validation_with_relations = attach_relational_features(
        validation_df, merged_index, "validation", feat_names
    )
    if len(train_with_relations) != EXPECTED_SPLIT_COUNTS["train"]:
        raise AssertionError("Permuted-null train row count changed.")
    if len(validation_with_relations) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise AssertionError("Permuted-null validation row count changed.")

    b0_features = list(b0_metadata["feature_columns"])
    validate_model_columns_against_frozen_b0(
        list(train_with_relations.columns), b0_features, feat_names
    )
    validate_model_columns_against_frozen_b0(
        list(validation_with_relations.columns), b0_features, feat_names
    )
    feature_columns = build_b1_feature_manifest(b0_features, feat_names)
    return train_with_relations, validation_with_relations, feature_columns


def validate_permuted_null_configuration(model: LGBMClassifier, b0_metadata: dict[str, Any]) -> None:
    """Assert every LightGBM setting matches the converged B1-card1 reference exactly.

    Cap and seed are both pinned at the converged reference's values (15,000,
    42). The only thing that may differ between this run and that reference
    is which entity assignment the four relational columns were built on.
    """
    reference = b0_metadata.get("lightgbm_parameters")
    if not isinstance(reference, dict):
        raise ValueError("Frozen B0 LightGBM parameters are missing.")
    actual = model.get_params(deep=False)

    for name in FROZEN_PARAMETER_NAMES:
        if name not in reference:
            raise ValueError(f"Frozen B0 parameter is missing: {name}.")
        if actual.get(name) != reference[name]:
            raise ValueError(
                f"Permuted-null parameter {name} differs from frozen B0: "
                f"actual={actual.get(name)!r}, B0={reference[name]!r}."
            )
    if actual.get("n_estimators") != CONVERGED_MAX_ESTIMATORS:
        raise ValueError(
            f"Permuted-null run must carry n_estimators={CONVERGED_MAX_ESTIMATORS:,}; "
            f"got {actual.get('n_estimators')!r}."
        )
    if not np.isclose(
        actual.get("scale_pos_weight"), b0_metadata.get("scale_pos_weight"), rtol=0.0, atol=0.0
    ):
        raise ValueError("Permuted-null class weight differs from frozen B0.")


def run_permuted_null(permutation_seed: int, skip_existing: bool = False) -> dict[str, Any]:
    label = f"permuted_{RELATION}/permseed_{permutation_seed}"
    paths = resolve_run_paths(permutation_seed)
    if skip_existing and run_is_complete(permutation_seed):
        print(f"[{label}] Already complete; skipping.")
        return {}

    protected_before = snapshot_protected_artifacts(ALL_PROTECTED_PATHS)
    b0_metadata = load_frozen_b0_metadata()

    print(f"[{label}] Building permuted card1 relation and loading data...")
    train_df, validation_df, feature_columns = load_permuted_null_datasets(
        permutation_seed, b0_metadata, paths
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
    model.set_params(random_state=RANDOM_SEED)
    validate_permuted_null_configuration(model, b0_metadata)

    print(
        f"[{label}] Predictors: {len(feature_columns):,}  "
        f"cap: {CONVERGED_MAX_ESTIMATORS:,}  training seed: {RANDOM_SEED}  "
        f"permutation seed: {permutation_seed}"
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
            "model": "permuted_null_card1",
            "relation": RELATION,
            "permutation_seed": int(permutation_seed),
            "random_seed": RANDOM_SEED,
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
    model.booster_.save_model(str(paths["model"]), num_iteration=model.best_iteration_)
    write_json(paths["metrics"], metrics)
    feature_importance.to_csv(paths["feature_importance"], index=False)
    validation_predictions.to_parquet(
        paths["validation_predictions"], index=False, engine="pyarrow", compression="snappy"
    )
    learning_curve.to_csv(paths["learning_curve"], index=False)

    metadata = {
        "experiment_name": "permuted_entity_null_control",
        "relation": RELATION,
        "permutation_seed": int(permutation_seed),
        "random_seed": RANDOM_SEED,
        "varied_parameter": "permutation_seed only (LightGBM random_state pinned)",
        "held_frozen": (
            "B0 feature manifest, train-only categorical mappings, class weight, "
            "estimator cap, early-stopping patience, stopping metric and every "
            "other LightGBM parameter -- identical to the converged B1-card1 "
            "reference except which entity assignment the four relational "
            "columns were built on"
        ),
        "reference_configuration": "b1_card1 converged (cap 15,000, stop_average_precision, seed 42)",
        "feature_count": len(feature_columns),
        "categorical_feature_count": len(categorical_columns),
        "categorical_mappings_sha256": mapping_sha256,
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
        protected_before, ALL_PROTECTED_PATHS, label="frozen/converged B0, B1-card1 and the real card1 relation"
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
            "Train the card1 permuted-entity null control across permutation "
            "seeds, at the converged (cap 15,000) estimator cap."
        )
    )
    parser.add_argument(
        "--permutation-seed",
        action="append",
        type=int,
        help="Permutation seed to run; repeatable. Defaults to the three-seed panel.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a run whose artifacts already exist, so an interrupted panel resumes.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    seeds = args.permutation_seed or PERMUTATION_SEEDS

    print(f"Planned runs: {len(seeds)}")
    for index, seed in enumerate(seeds, start=1):
        print(f"\n({index}/{len(seeds)}) === permutation seed {seed} ===")
        run_permuted_null(seed, skip_existing=args.skip_existing)

    print("\nAll planned runs complete.")
    print(f"Reports: {REPORT_DIR}")
    print("Test set evaluated: NO")


if __name__ == "__main__":
    main()
