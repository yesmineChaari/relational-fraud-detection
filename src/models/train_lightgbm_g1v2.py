"""G1-v2 downstream LightGBM: B1-card1's 439 predictors plus one G1-v2 embedding block.

The encoder is credited only for what it adds beyond the scalar relational
signal already known to work, so the manifest is B0 (435) + card1's four
relational scalars + the 32-column block (471), and the reference is B1-card1
retrained at the same fixed budget (reports/fixed_budget/card1/b1_card1/). A
fixed-budget candidate read against an early-stopped reference would
reintroduce the argmax asymmetry the fixed-budget protocol exists to remove, so
only the manifest differs from that reference.

Every G1-v2 run -- each encoder seed, the count-blind attribution arm and the
permuted width null -- goes through `train_g1v2_run`, so they differ in their
embedding block and nothing else. Each writes validation_predictions.parquet
for the paired bootstrap and records its leakage gate's verdict, which the
pre-registered verdict checks before crediting anything.
"""

from __future__ import annotations

import argparse
import gc
import platform
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
import sklearn
from pandas.api.types import is_numeric_dtype

from src.config.paths import MODELS_DIR, REPORTS_DIR
from src.features.build_relational_features import _feature_names
from src.features.build_relational_features import _output_path as _rel_output_path
from src.graph.train_graphsage_encoder_v2 import COUNT_BLIND, WITH_COUNTS, V2Run
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
from src.models.train_lightgbm_g1 import (
    B1_CARD1_PROTECTED_PATHS,
    build_g1_feature_manifest,
    embedding_feature_names,
    embedding_gain_ranks,
    load_embedding_metadata,
    load_embeddings,
    validate_embedding_merge,
)
from src.models.train_lightgbm_g1_controls import G1_PROTECTED_PATHS
from src.models.train_lightgbm_relational import (
    B0_PROTECTED_PATHS,
    EVALUATION_SPLIT,
    EXPECTED_B1_FEATURE_COUNT,
    LOG_EVALUATION_PERIOD,
    apply_frozen_category_mappings,
    assert_protected_artifacts_unchanged,
    attach_relational_features,
    build_b1_feature_manifest,
    file_sha256,
    load_frozen_b0_metadata,
    load_frozen_category_mappings,
    read_json,
    snapshot_protected_artifacts,
    validate_model_columns_against_frozen_b0,
    validate_relational_merge,
)

RELATION = "card1"
EXPECTED_G1V2_FEATURE_COUNT = EXPECTED_B1_FEATURE_COUNT + len(embedding_feature_names())

REPORT_ROOT = REPORTS_DIR / "g1_v2" / RELATION
MODEL_DIR = MODELS_DIR / "g1_v2"

REFERENCE_RUN_NAME = "b1_card1"
REFERENCE_LABEL = "b1_card1_fixed_budget"
PROTOCOL = "fixed_budget_no_early_stopping"


@dataclass(frozen=True)
class G1V2Run:
    """One downstream run: an embedding block plus where its results go."""

    name: str
    embeddings_path: Path
    encoder_metadata_path: Path
    leakage_gate_path: Path | None
    description: str

    @property
    def report_dir(self) -> Path:
        return REPORT_ROOT / self.name

    @property
    def model_path(self) -> Path:
        return MODEL_DIR / f"lightgbm_g1v2_{RELATION}_{self.name}.txt"

    def paths(self) -> dict[str, Path]:
        report_dir = self.report_dir
        return {
            "model": self.model_path,
            "report_dir": report_dir,
            "metrics": report_dir / "metrics.json",
            "metadata": report_dir / "metadata.json",
            "feature_importance": report_dir / "feature_importance.csv",
            "validation_predictions": report_dir / "validation_predictions.parquet",
            "learning_curve": report_dir / "learning_curve.csv",
            "significance": report_dir / "significance.json",
        }

    def is_complete(self) -> bool:
        return all(path.exists() for key, path in self.paths().items() if key != "report_dir")


def encoder_backed_run(seed: int, count_mode: str = WITH_COUNTS) -> G1V2Run:
    encoder = V2Run(seed=seed, count_mode=count_mode)
    kind = "count-blind attribution arm" if count_mode == COUNT_BLIND else "cardinality-aware"
    return G1V2Run(
        name=encoder.name,
        embeddings_path=encoder.embeddings_path,
        encoder_metadata_path=encoder.metadata_path,
        leakage_gate_path=encoder.leakage_gate_path,
        description=f"B1-card1 plus the {kind} G1-v2 embedding block, encoder seed {seed}.",
    )


def build_g1v2_feature_manifest(
    b0_features: list[str],
    card1_feat_names: list[str],
    embedding_feat_names: list[str],
) -> list[str]:
    """B1-card1's manifest followed by the embedding block."""
    b1_card1_features = build_b1_feature_manifest(b0_features, card1_feat_names)
    features = build_g1_feature_manifest(b1_card1_features, embedding_feat_names)
    if len(features) != len(b0_features) + len(card1_feat_names) + len(embedding_feat_names):
        raise AssertionError("G1-v2 manifest lost or duplicated a predictor.")
    return features


def load_g1v2_datasets(
    embeddings_path: Path,
    b0_metadata: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Train/validation frames carrying B0, card1's four scalars and one embedding block."""
    card1_path = _rel_output_path(RELATION)
    if not card1_path.exists():
        raise FileNotFoundError(
            f"Relational features not found for relation {RELATION!r}. Run: "
            f"python -m src.features.build_relational_features --relation {RELATION}"
        )
    b0_features = list(b0_metadata["feature_columns"])
    card1_feat_names = _feature_names(RELATION)
    embedding_names = embedding_feature_names()

    train_df, validation_df, _ = load_model_dataset()
    model_index = pd.read_parquet(MODEL_DATASET_PATH, columns=["TransactionID", "split"])
    validate_split_counts(model_index, "model_dataset.parquet G1-v2 index")

    card1_relational = pd.read_parquet(card1_path)
    if len(card1_relational) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} card1 relational rows.")
    card1_index = validate_relational_merge(model_index, card1_relational, card1_feat_names)
    embeddings_df = load_embeddings(RELATION, embedding_names, path=embeddings_path)
    embedding_index = validate_embedding_merge(model_index, embeddings_df, embedding_names)
    del card1_relational, embeddings_df

    for names, merged in ((card1_feat_names, card1_index), (embedding_names, embedding_index)):
        train_df = attach_relational_features(train_df, merged, "train", names)
        validation_df = attach_relational_features(validation_df, merged, "validation", names)

    added = [*card1_feat_names, *embedding_names]
    validate_model_columns_against_frozen_b0(list(train_df.columns), b0_features, added)
    validate_model_columns_against_frozen_b0(list(validation_df.columns), b0_features, added)
    for column in added:
        if not is_numeric_dtype(train_df[column].dtype):
            raise TypeError(f"G1-v2 predictor {column} is not numeric.")
    features = build_g1v2_feature_manifest(b0_features, card1_feat_names, embedding_names)
    return train_df, validation_df, features


def encoder_summary(encoder_metadata: dict[str, Any]) -> dict[str, Any]:
    cross_fitting = encoder_metadata.get("cross_fitting") or {}
    return {
        "random_seed": encoder_metadata["random_seed"],
        "readout": encoder_metadata["architecture"]["readout"],
        "count_mode": encoder_metadata["count_mode"],
        "cross_fit_folds": cross_fitting.get("n_folds"),
        "shared_initialisation": bool(cross_fitting.get("shared_initialisation", False)),
        "standalone_validation_pr_auc": encoder_metadata.get("standalone_validation_pr_auc"),
        "permutation": encoder_metadata.get("permutation"),
    }


def train_g1v2_run(run: G1V2Run, skip_existing: bool = False) -> None:
    label = f"g1v2/{run.name}"
    if skip_existing and run.is_complete():
        print(f"[{label}] Already complete; skipping.")
        return
    for path in (run.embeddings_path, run.encoder_metadata_path):
        if not path.exists():
            raise FileNotFoundError(f"[{label}] Embedding input not found: {path}")
    gate = read_json(run.leakage_gate_path) if run.leakage_gate_path is not None else None

    reference_paths = resolve_fixed_budget_run_paths(REFERENCE_RUN_NAME)
    if not reference_paths["validation_predictions"].exists():
        raise FileNotFoundError(
            f"Fixed-budget B1-card1 reference not found: {reference_paths['validation_predictions']}"
            "\nRun: python -m src.models.train_ablation_fixed_budget --run b1_card1"
        )
    inputs = [run.embeddings_path, run.encoder_metadata_path]
    if run.leakage_gate_path is not None:
        inputs.append(run.leakage_gate_path)
    all_protected = [
        *B0_PROTECTED_PATHS,
        *B1_CARD1_PROTECTED_PATHS,
        *G1_PROTECTED_PATHS,
        *(path for key, path in reference_paths.items() if key != "run_dir"),
        *inputs,
    ]
    protected_before = snapshot_protected_artifacts(all_protected)
    b0_metadata = load_frozen_b0_metadata()
    encoder_metadata = load_embedding_metadata(RELATION, path=run.encoder_metadata_path)

    print(f"[{label}] Loading data...")
    train_df, validation_df, feature_columns = load_g1v2_datasets(run.embeddings_path, b0_metadata)
    if len(feature_columns) != EXPECTED_G1V2_FEATURE_COUNT:
        raise ValueError(
            f"G1-v2 predictor-count sanity check failed: expected "
            f"{EXPECTED_G1V2_FEATURE_COUNT}, got {len(feature_columns)}."
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
    apply_frozen_category_mappings(train_df, validation_df, categorical_columns, mappings)
    X_train = pd.DataFrame({column: train_df.pop(column) for column in feature_columns}, copy=False)
    X_validation = pd.DataFrame(
        {column: validation_df.pop(column) for column in feature_columns}, copy=False
    )
    del train_df, validation_df
    gc.collect()
    if list(X_train.columns) != feature_columns or list(X_validation.columns) != feature_columns:
        raise AssertionError(f"[{label}] Predictor extraction changed column order.")
    assert_supported_model_dtypes(X_train, f"{label} training predictors")
    assert_supported_model_dtypes(X_validation, f"{label} validation predictors")

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    model = build_lightgbm_model(scale_pos_weight, n_estimators=FIXED_BUDGET)
    model.set_params(random_state=RANDOM_SEED)
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
            "model": f"g1_v2_{RELATION}_{run.name}",
            "run": run.name,
            "relation": RELATION,
            "protocol": PROTOCOL,
            "number_of_predictors": len(feature_columns),
            "random_seed": RANDOM_SEED,
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
    ranks = embedding_gain_ranks(feature_importance, embedding_feature_names())
    validation_predictions = validation_metadata.copy()
    validation_predictions["prediction"] = validation_scores

    paths = run.paths()
    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(paths["model"]), num_iteration=FIXED_BUDGET)
    write_json(paths["metrics"], metrics)
    feature_importance.to_csv(paths["feature_importance"], index=False)
    validation_predictions.to_parquet(
        paths["validation_predictions"], index=False, engine="pyarrow", compression="snappy"
    )
    learning_curve.to_csv(paths["learning_curve"], index=False)

    print(f"[{label}] Comparing against the fixed-budget B1-card1 reference...")
    comparison = compare_variants(
        candidate_label=f"g1v2_{run.name}",
        candidate_predictions_path=paths["validation_predictions"],
        reference_label=REFERENCE_LABEL,
        reference_predictions_path=reference_paths["validation_predictions"],
    )
    write_json(paths["significance"], comparison)

    metadata = {
        "experiment_name": f"g1_v2_{RELATION}_{run.name}",
        "run": run.name,
        "description": run.description,
        "relation": RELATION,
        "protocol": PROTOCOL,
        "controlled_experiment_definition": (
            "B1-card1's 439 predictors plus one 32-column embedding block; only the "
            "block differs from the fixed-budget B1-card1 reference."
        ),
        "reference_configuration": REFERENCE_LABEL,
        "reference_validation_predictions_path": repository_relative(
            reference_paths["validation_predictions"]
        ),
        "embedding_source_path": repository_relative(run.embeddings_path),
        "embedding_source_sha256": file_sha256(run.embeddings_path),
        "encoder_metadata_path": repository_relative(run.encoder_metadata_path),
        "encoder": encoder_summary(encoder_metadata),
        "leakage_gate_path": (
            None if run.leakage_gate_path is None else repository_relative(run.leakage_gate_path)
        ),
        "leakage_gate_usable": None if gate is None else bool(gate["usable"]),
        "fixed_budget_trees": int(FIXED_BUDGET),
        "early_stopping_used": False,
        "feature_count": len(feature_columns),
        "categorical_feature_count": len(categorical_columns),
        "categorical_mappings_sha256": mapping_sha256,
        "scale_pos_weight": float(scale_pos_weight),
        "lightgbm_parameters": model.get_params(deep=False),
        "evaluation_split": EVALUATION_SPLIT,
        "validation_pr_auc": float(metrics["pr_auc"]),
        "validation_roc_auc": float(metrics["roc_auc"]),
        "comparison_to_b1_card1_fixed_budget": comparison,
        "embedding_gain_importance": ranks,
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
        protected_before, all_protected, label="frozen references and G1-v2 inputs"
    )

    print(f"[{label}] Done in {(finished_at - started_at).total_seconds() / 60:.1f} min")
    print(f"[{label}] PR-AUC {metrics['pr_auc']:.6f}  ROC-AUC {metrics['roc_auc']:.6f}")
    print(
        f"[{label}] Delta vs fixed-budget B1-card1: {comparison['observed_delta']:+.5f} "
        f"95% CI [{comparison['ci_lower_95']:+.5f}, {comparison['ci_upper_95']:+.5f}]"
    )
    if gate is not None:
        print(f"[{label}] Leakage gate: {'USABLE' if gate['usable'] else 'NOT USABLE'}")
    print(f"[{label}] Final test evaluated: NO")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.train_lightgbm_g1v2",
        description="Fixed-budget LightGBM on B1-card1 plus one G1-v2 embedding block.",
    )
    parser.add_argument("--seed", type=int, required=True, help="Encoder seed of the block.")
    parser.add_argument("--count-blind", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    count_mode = COUNT_BLIND if args.count_blind else WITH_COUNTS
    train_g1v2_run(encoder_backed_run(args.seed, count_mode), skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
