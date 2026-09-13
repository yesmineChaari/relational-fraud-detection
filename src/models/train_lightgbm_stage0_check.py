"""Stage 0 Track B (G1-v2 plan): does addr1 add anything to B1-card1?

`addr1` alone has never been tested -- it was only ever combined into the
rejected `card_core_addr1` key, whose rejection was attributed to a
missingness confound in the *combined* key rather than necessarily to
`addr1`'s own signal (see `reports/relational_screening/candidate_selection.json`'s
`addr1` entry and the G1-v2 plan, section 1.5). This trains B1-card1's 439
predictors plus addr1's four scalar relational features (443 total) and reads
whether the extra four columns move PR-AUC.

Trained at the fixed 10,000-round budget with early stopping disabled, reusing
`train_ablation_fixed_budget.py`'s protocol: any argmax-of-a-validation-curve
comparison rewards whichever arm was granted more rounds (README, "selection-
bias correction"). For the same reason, the reference this candidate is
compared against is the fixed-budget B1-card1 rerun already published under
`reports/fixed_budget/card1/b1_card1/`, not the early-stopped canonical
`reports/b1/card1/` artifact -- comparing a fixed-budget candidate against an
early-stopped reference would reintroduce exactly the asymmetry the
fixed-budget protocol exists to remove. Only the feature manifest differs
between this run and that reference; every other setting matches it exactly.

Nothing here writes to a frozen or fixed-budget reference artifact. Outputs
land under their own report and model trees, and every artifact this
comparison depends on is hash-pinned before and after the run.
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
from src.features.build_relational_features import (
    RELATION_REGISTRY,
    _feature_names,
)
from src.features.build_relational_features import (
    _output_path as _rel_output_path,
)
from src.models.significance import compare_variants
from src.models.train_ablation_fixed_budget import FIXED_BUDGET
from src.models.train_ablation_fixed_budget import (
    resolve_run_paths as resolve_fixed_budget_run_paths,
)
from src.models.train_lightgbm_baseline import (
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
    validate_split_counts,
    write_json,
)
from src.models.train_lightgbm_convergence_check import validate_convergence_configuration
from src.models.train_lightgbm_g1 import B1_CARD1_PROTECTED_PATHS
from src.models.train_lightgbm_relational import (
    B0_PROTECTED_PATHS,
    EVALUATION_SPLIT,
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

RELATION = "addr1"
CARD1_RELATION = "card1"
REFERENCE_RUN_NAME = "b1_card1"

EXPECTED_B1_CARD1_FEATURE_COUNT = 439
EXPECTED_STAGE0_FEATURE_COUNT = 443

REPORT_DIR = ROOT_DIR / "reports" / "stage0_screening" / RELATION
MODEL_DIR = ROOT_DIR / "models" / "stage0_screening"
MODEL_PATH = MODEL_DIR / f"lightgbm_stage0_{RELATION}.txt"
METRICS_PATH = REPORT_DIR / "metrics.json"
METADATA_PATH = REPORT_DIR / "metadata.json"
FEATURE_IMPORTANCE_PATH = REPORT_DIR / "feature_importance.csv"
VALIDATION_PREDICTIONS_PATH = REPORT_DIR / "validation_predictions.parquet"
LEARNING_CURVE_PATH = REPORT_DIR / "learning_curve.csv"
COMPARISON_PATH = REPORT_DIR / "comparison_to_b1_card1_fixed_budget.csv"

PROTOCOL = "fixed_budget_no_early_stopping"

# The pre-registered bar (G1-v2 plan, Stage 0 Track B, step 5): report as an
# independent win only if the 95% CI excludes zero on the positive side.
PRE_REGISTERED_BAR = (
    "Independent win only if the 95% paired-bootstrap CI on "
    "PR-AUC(candidate) - PR-AUC(b1_card1_fixed_budget) excludes zero on the positive side."
)


def _stage0_artifact_paths() -> list[Path]:
    return [
        MODEL_PATH,
        METRICS_PATH,
        METADATA_PATH,
        FEATURE_IMPORTANCE_PATH,
        VALIDATION_PREDICTIONS_PATH,
        LEARNING_CURVE_PATH,
        COMPARISON_PATH,
    ]


def stage0_run_is_complete() -> bool:
    return all(path.exists() for path in _stage0_artifact_paths())


def load_stage0_datasets(
    b0_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Train/validation frames carrying B0 + card1's 4 scalars + addr1's 4 scalars."""
    if RELATION not in RELATION_REGISTRY:
        raise ValueError(f"Unknown relation: {RELATION!r}. Supported: {sorted(RELATION_REGISTRY)}.")

    addr1_path = _rel_output_path(RELATION)
    if not addr1_path.exists():
        raise FileNotFoundError(
            f"Relational features not found for relation {RELATION!r}. "
            f"Run: python -m src.features.build_relational_features --relation {RELATION}\n"
            f"Expected: {addr1_path}"
        )

    b0_features = list(b0_metadata["feature_columns"])
    card1_feat_names = _feature_names(CARD1_RELATION)
    addr1_feat_names = _feature_names(RELATION)

    train_df, validation_df, _ = load_model_dataset()
    model_index = pd.read_parquet(MODEL_DATASET_PATH, columns=["TransactionID", "split"])
    validate_split_counts(model_index, "model_dataset.parquet stage0 index")

    card1_relational = pd.read_parquet(_rel_output_path(CARD1_RELATION))
    if len(card1_relational) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} card1 relational rows.")
    card1_merged_index = validate_relational_merge(model_index, card1_relational, card1_feat_names)
    train_df = attach_relational_features(train_df, card1_merged_index, "train", card1_feat_names)
    validation_df = attach_relational_features(
        validation_df, card1_merged_index, "validation", card1_feat_names
    )

    addr1_relational = pd.read_parquet(addr1_path)
    if len(addr1_relational) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} addr1 relational rows.")
    addr1_merged_index = validate_relational_merge(model_index, addr1_relational, addr1_feat_names)
    train_df = attach_relational_features(train_df, addr1_merged_index, "train", addr1_feat_names)
    validation_df = attach_relational_features(
        validation_df, addr1_merged_index, "validation", addr1_feat_names
    )

    combined_feat_names = [*card1_feat_names, *addr1_feat_names]
    validate_model_columns_against_frozen_b0(
        list(train_df.columns), b0_features, combined_feat_names
    )
    validate_model_columns_against_frozen_b0(
        list(validation_df.columns), b0_features, combined_feat_names
    )

    b1_card1_features = build_b1_feature_manifest(b0_features, card1_feat_names)
    if len(b1_card1_features) != EXPECTED_B1_CARD1_FEATURE_COUNT:
        raise ValueError(
            f"B1-card1 predictor-count sanity check failed: expected "
            f"{EXPECTED_B1_CARD1_FEATURE_COUNT}, got {len(b1_card1_features)}."
        )
    stage0_features = build_b1_feature_manifest(b1_card1_features, addr1_feat_names)
    if len(stage0_features) != EXPECTED_STAGE0_FEATURE_COUNT:
        raise ValueError(
            f"Stage 0 predictor-count sanity check failed: expected "
            f"{EXPECTED_STAGE0_FEATURE_COUNT}, got {len(stage0_features)}."
        )

    return train_df, validation_df, stage0_features


def train_stage0_check(skip_existing: bool = False) -> None:
    label = f"stage0/{RELATION}/fixed{FIXED_BUDGET}_seed{RANDOM_SEED}"
    if skip_existing and stage0_run_is_complete():
        print(f"[{label}] Already complete; skipping.")
        return

    reference_paths = resolve_fixed_budget_run_paths(REFERENCE_RUN_NAME)
    if not reference_paths["validation_predictions"].exists():
        raise FileNotFoundError(
            "Fixed-budget B1-card1 reference not found: "
            f"{reference_paths['validation_predictions']}\n"
            "Run: python -m src.models.train_ablation_fixed_budget --run b1_card1"
        )
    protected_reference_paths = [p for key, p in reference_paths.items() if key != "run_dir"]
    all_protected = [*B0_PROTECTED_PATHS, *B1_CARD1_PROTECTED_PATHS, *protected_reference_paths]
    protected_before = snapshot_protected_artifacts(all_protected)
    b0_metadata = load_frozen_b0_metadata()

    print(f"[{label}] Loading data...")
    train_df, validation_df, feature_columns = load_stage0_datasets(b0_metadata)

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
    model = build_lightgbm_model(scale_pos_weight, n_estimators=FIXED_BUDGET)
    model.set_params(random_state=RANDOM_SEED)
    # Only feature manifest may differ from frozen B0/the fixed-budget
    # reference. Reused rather than reimplemented so this check cannot drift
    # from the rule the convergence check already enforces.
    validate_convergence_configuration(model, b0_metadata, RANDOM_SEED, FIXED_BUDGET)

    print(
        f"[{label}] Predictors: {len(feature_columns):,}  budget: {FIXED_BUDGET:,} rounds  "
        f"early stopping: disabled"
    )
    evaluation_results: dict[str, dict[str, list[float]]] = {}
    callbacks = [
        lightgbm.record_evaluation(evaluation_results),
        lightgbm.log_evaluation(period=LOG_EVALUATION_PERIOD),
    ]
    print(f"[{label}] Training {FIXED_BUDGET:,} rounds...")
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

    learning_curve = build_learning_curve(evaluation_results)
    if len(learning_curve) != FIXED_BUDGET:
        raise AssertionError(
            f"[{label}] Learning curve has {len(learning_curve):,} rows, expected "
            f"{FIXED_BUDGET:,}. Early stopping must not be active in this run."
        )

    validation_scores = model.predict_proba(X_validation, num_iteration=FIXED_BUDGET)[:, 1]
    if len(validation_scores) != EXPECTED_SPLIT_COUNTS[EVALUATION_SPLIT]:
        raise AssertionError(f"[{label}] Validation prediction count is incorrect.")
    if not np.isfinite(validation_scores).all():
        raise AssertionError(f"[{label}] Validation predictions contain non-finite values.")

    metrics = evaluate_validation(y_validation, validation_scores)
    metrics.update(
        {
            "model": f"stage0_screening_{RELATION}",
            "relation": RELATION,
            "protocol": PROTOCOL,
            "number_of_predictors": len(feature_columns),
            "random_seed": RANDOM_SEED,
            "weighting": "weighted",
            "scale_pos_weight": float(scale_pos_weight),
            "fixed_budget_trees": int(FIXED_BUDGET),
            "reported_iteration": int(FIXED_BUDGET),
            "early_stopping_used": False,
            "test_evaluated": False,
        }
    )

    feature_importance = build_feature_importance(model)
    if len(feature_importance) != len(feature_columns):
        raise AssertionError(f"[{label}] Feature importance does not match its manifest.")
    validation_predictions = validation_metadata.copy()
    validation_predictions["prediction"] = validation_scores

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(MODEL_PATH), num_iteration=FIXED_BUDGET)
    write_json(METRICS_PATH, metrics)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    validation_predictions.to_parquet(
        VALIDATION_PREDICTIONS_PATH, index=False, engine="pyarrow", compression="snappy"
    )
    learning_curve.to_csv(LEARNING_CURVE_PATH, index=False)

    print(f"[{label}] Comparing against the fixed-budget B1-card1 reference...")
    comparison = compare_variants(
        candidate_label=f"stage0_{RELATION}",
        candidate_predictions_path=VALIDATION_PREDICTIONS_PATH,
        reference_label="b1_card1_fixed_budget",
        reference_predictions_path=reference_paths["validation_predictions"],
    )
    pd.DataFrame([comparison]).to_csv(COMPARISON_PATH, index=False)
    meets_bar = bool(comparison["ci_lower_95"] > 0.0)

    metadata = {
        "experiment_name": "stage0_screening_addr1",
        "relation": RELATION,
        "protocol": PROTOCOL,
        "reference_configuration": "b1_card1_fixed_budget",
        "reference_rationale": (
            "Compared against the fixed-budget B1-card1 rerun "
            f"({repository_relative(reference_paths['validation_predictions'])}), not the "
            "early-stopped canonical B1-card1 artifact -- comparing a fixed-budget "
            "candidate against an early-stopped reference would reintroduce exactly "
            "the argmax asymmetry the fixed-budget protocol exists to remove."
        ),
        "pre_registered_bar": PRE_REGISTERED_BAR,
        "meets_pre_registered_bar": meets_bar,
        "varied_parameter": "feature manifest only (B1-card1's 439 features + addr1's 4 scalars)",
        "held_frozen": (
            "B0 feature manifest, train-only categorical mappings, class weight, "
            "estimator budget, and every other LightGBM parameter, matching the "
            "fixed-budget b1_card1 reference exactly"
        ),
        "fixed_budget_trees": int(FIXED_BUDGET),
        "feature_count": len(feature_columns),
        "categorical_feature_count": len(categorical_columns),
        "categorical_mappings_sha256": mapping_sha256,
        "scale_pos_weight": float(scale_pos_weight),
        "lightgbm_parameters": model.get_params(deep=False),
        "evaluation_split": EVALUATION_SPLIT,
        "validation_pr_auc": float(metrics["pr_auc"]),
        "validation_roc_auc": float(metrics["roc_auc"]),
        "comparison_to_b1_card1_fixed_budget": comparison,
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
    write_json(METADATA_PATH, metadata)
    assert_protected_artifacts_unchanged(
        protected_before,
        all_protected,
        label="frozen B0/B1-card1 and the fixed-budget b1_card1 reference",
    )

    missing = [str(p) for p in _stage0_artifact_paths() if not p.exists()]
    if missing:
        raise OSError(f"[{label}] Stage 0 artifacts were not created: {missing}.")

    print(f"[{label}] Done in {(finished_at - started_at).total_seconds() / 60:.1f} min")
    print(f"[{label}] PR-AUC {metrics['pr_auc']:.6f}  ROC-AUC {metrics['roc_auc']:.6f}")
    print(
        f"[{label}] Delta vs fixed-budget B1-card1: {comparison['observed_delta']:+.5f} "
        f"95% CI [{comparison['ci_lower_95']:+.5f}, {comparison['ci_upper_95']:+.5f}]"
    )
    print(f"[{label}] Meets pre-registered bar (independent win): {'YES' if meets_bar else 'NO'}")
    print(f"[{label}] Frozen/fixed-budget references unchanged: YES")
    print(f"[{label}] Final test evaluated: NO")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.train_lightgbm_stage0_check",
        description="Stage 0 Track B: addr1's scalar features on top of B1-card1, fixed budget.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    train_stage0_check(skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
