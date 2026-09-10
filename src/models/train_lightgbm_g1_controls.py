"""Downstream LightGBM runs for the G1 attribution controls.

Each control swaps one 32-column embedding block into the frozen G1 recipe --
frozen B0 predictors, frozen categorical mappings, frozen LightGBM
configuration, frozen class weight -- and changes nothing else, so the measured
delta is attributable to the embedding block and not to a retuned model. The
encoder-side blocks come from src/graph/train_graphsage_variants.py; the
shuffled-embedding block is built here, because it needs no encoder.

Controls
----------
* `extended_budget`     -- reference for the other encoder controls: frozen
                           architecture, raised encoder budget, full-validation
                           selection.
* `neighbourhood_only`  -- self-contribution removed from the encoder readout.
* `cross_fitted`        -- K-fold cross-fitted embeddings.
* `shuffled_embedding`  -- the *frozen* G1 embedding block, permuted within each
                           split. A 32-column block with unchanged marginals and
                           destroyed row alignment. Whatever this costs relative
                           to B0 is what adding 32 uninformative columns costs a
                           6,000-round split search; only the excess beyond it is
                           a statement about the graph.

Every control is measured against three references -- B0, B1-card1 and the
frozen G1 -- by the same paired bootstrap G1 already uses, and every one of
those references is hash-pinned before and after the run. No control run can
touch a frozen artifact, and each writes to its own report directory under
reports/g1_controls/.

Why the shuffled control permutes within split
------------------------------------------------
Permuting across the whole table would move validation-partition embedding rows
into train and vice versa. Those blocks have different marginal distributions
(the encoder was fitted on train rows), so a global permutation would confound
"randomly aligned" with "drawn from a different distribution". Permuting inside
each split leaves every split's 32-column marginal distribution exactly as the
real block's and destroys only the row alignment, which is the single property
the control exists to remove.
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

from src.config.paths import ROOT_DIR
from src.graph.train_graphsage_encoder import (
    EMBEDDING_DIM,
)
from src.graph.train_graphsage_encoder import (
    EMBEDDINGS_PATH as FROZEN_EMBEDDINGS_PATH,
)
from src.graph.train_graphsage_encoder import (
    METADATA_PATH as FROZEN_ENCODER_METADATA_PATH,
)
from src.graph.train_graphsage_encoder import (
    METRICS_PATH as FROZEN_ENCODER_METRICS_PATH,
)
from src.graph.train_graphsage_variants import (
    CONTROL_REPORT_ROOT,
    ENCODER_VARIANTS,
    PROCESSED_DIR,
)
from src.models.train_lightgbm_baseline import (
    CATEGORY_MAPPINGS_PATH,
    EXPECTED_ROWS,
    EXPECTED_SPLIT_COUNTS,
    RANDOM_SEED,
    assert_supported_model_dtypes,
    build_feature_importance,
    build_learning_curve,
    build_lightgbm_model,
    calculate_scale_pos_weight,
    evaluate_validation,
    identify_categorical_columns,
    repository_relative,
    summarize_learning_curve,
    write_json,
)
from src.models.train_lightgbm_g1 import (
    B1_CARD1_METRICS_PATH,
    B1_CARD1_PROTECTED_PATHS,
    B1_CARD1_VALIDATION_PREDICTIONS_PATH,
    EXPECTED_G1_FEATURE_COUNT,
    RELATION,
    SIGNIFICANCE_RESAMPLES,
    _resolve_g1_paths,
    build_comparison_table,
    build_g1_feature_manifest,
    embedding_feature_names,
    embedding_gain_ranks,
    load_embedding_metadata,
    load_frozen_b1_card1_metadata,
    load_g1_datasets,
    load_validation_predictions,
    paired_bootstrap_pr_auc_delta,
)
from src.models.train_lightgbm_relational import (
    B0_PROTECTED_PATHS,
    EARLY_STOPPING_ROUNDS,
    EVALUATION_SPLIT,
    LOG_EVALUATION_PERIOD,
    MAX_ESTIMATORS,
    TEST_EVALUATED,
    apply_frozen_category_mappings,
    assert_protected_artifacts_unchanged,
    file_sha256,
    load_frozen_b0_metadata,
    load_frozen_category_mappings,
    read_json,
    snapshot_protected_artifacts,
    validate_frozen_lightgbm_configuration,
    validate_model_columns_against_frozen_b0,
)

B0_METRICS_PATH = ROOT_DIR / "reports" / "baseline" / "lightgbm_metrics.json"
B0_VALIDATION_PREDICTIONS_PATH = (
    ROOT_DIR / "reports" / "baseline" / "validation_predictions.parquet"
)

_G1_PATHS = _resolve_g1_paths(RELATION)
G1_METRICS_PATH = _G1_PATHS["metrics"]
G1_METADATA_PATH = _G1_PATHS["metadata"]
G1_VALIDATION_PREDICTIONS_PATH = _G1_PATHS["validation_predictions"]

# The frozen G1 run is now a reference in its own right, protected exactly the
# way B0 and B1-card1 are: the controls exist to reinterpret it, not to move it.
G1_PROTECTED_PATHS = [
    _G1_PATHS["model"],
    G1_METRICS_PATH,
    G1_METADATA_PATH,
    _G1_PATHS["feature_importance"],
    _G1_PATHS["significance"],
    G1_VALIDATION_PREDICTIONS_PATH,
    FROZEN_ENCODER_METADATA_PATH,
    FROZEN_ENCODER_METRICS_PATH,
    FROZEN_EMBEDDINGS_PATH,
]

SHUFFLED_VARIANT_NAME = "shuffled_embedding"
SHUFFLED_EMBEDDINGS_PATH = (
    PROCESSED_DIR / f"graphsage_card1_embeddings_{SHUFFLED_VARIANT_NAME}.parquet"
)
SHUFFLE_SEED = 42

REFERENCE_LABELS = ("b0", "b1_card1", "g1")


@dataclass(frozen=True)
class ControlRun:
    """One downstream control: an embedding block plus where its results go."""

    name: str
    embeddings_path: Path
    embedding_metadata_path: Path
    isolates: str
    description: str

    @property
    def report_dir(self) -> Path:
        return CONTROL_REPORT_ROOT / self.name

    @property
    def model_path(self) -> Path:
        return ROOT_DIR / "models" / f"lightgbm_g1_control_{self.name}.txt"

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
            "comparison_to_b0": report_dir / "comparison_to_b0.csv",
            "comparison_to_b1_card1": report_dir / "comparison_to_b1_card1.csv",
            "comparison_to_g1": report_dir / "comparison_to_g1.csv",
            "significance": report_dir / "significance.json",
        }


def _shuffled_control() -> ControlRun:
    report_dir = CONTROL_REPORT_ROOT / SHUFFLED_VARIANT_NAME
    return ControlRun(
        name=SHUFFLED_VARIANT_NAME,
        embeddings_path=SHUFFLED_EMBEDDINGS_PATH,
        embedding_metadata_path=report_dir / "encoder_metadata.json",
        isolates="split-search dilution from 32 extra columns",
        description=(
            "The frozen G1 embedding block permuted within each split: identical "
            "per-split marginals, no row alignment. Measures what adding 32 "
            "uninformative columns costs the frozen LightGBM configuration."
        ),
    )


CONTROL_RUNS: dict[str, ControlRun] = {
    **{
        variant.name: ControlRun(
            name=variant.name,
            embeddings_path=variant.embeddings_path,
            embedding_metadata_path=variant.metadata_path,
            isolates=variant.isolates,
            description=variant.description,
        )
        for variant in ENCODER_VARIANTS.values()
    },
    SHUFFLED_VARIANT_NAME: _shuffled_control(),
}

# Order matters for reporting: the reference control first, then the two
# encoder controls read against it, then the cheap null.
CONTROL_ORDER = [
    "extended_budget",
    "neighbourhood_only",
    "cross_fitted",
    SHUFFLED_VARIANT_NAME,
]


def permute_block_within_split(
    frame: pd.DataFrame,
    feat_names: list[str],
    seed: int,
    expected_split_counts: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Permute `feat_names` within each split, leaving every other column put.

    TransactionID and split stay in their original rows; only the embedding
    columns move, and they move as whole rows (one permutation applied to all
    of them at once) so the block keeps its internal correlation structure and
    loses nothing but its alignment to transactions.
    """
    missing = [column for column in ("TransactionID", "split") if column not in frame.columns]
    if missing:
        raise ValueError(f"Frame is missing required columns: {missing}.")
    missing_features = [column for column in feat_names if column not in frame.columns]
    if missing_features:
        raise ValueError(f"Frame is missing embedding columns: {missing_features}.")

    splits = frame["split"].astype("string").to_numpy()
    block = frame[feat_names].to_numpy(copy=True)
    permuted = np.empty_like(block)
    rng = np.random.default_rng(seed)

    for split_name in sorted(pd.unique(splits)):
        positions = np.flatnonzero(splits == split_name)
        if expected_split_counts is not None:
            expected = expected_split_counts.get(str(split_name))
            if expected is None:
                raise ValueError(f"Unexpected split {split_name!r} in the embedding block.")
            if len(positions) != expected:
                raise ValueError(
                    f"Split {split_name!r} has {len(positions):,} rows; expected {expected:,}."
                )
        permuted[positions] = block[positions[rng.permutation(len(positions))]]

    shuffled = frame.copy()
    shuffled[feat_names] = permuted
    if not np.array_equal(np.sort(permuted, axis=0), np.sort(block, axis=0)):
        raise AssertionError("Within-split permutation changed the column marginals.")
    if shuffled.isna().any().any():
        raise AssertionError("Shuffled embedding block contains nulls.")
    return shuffled


def build_shuffled_embeddings(
    source_path: Path = FROZEN_EMBEDDINGS_PATH,
    seed: int = SHUFFLE_SEED,
) -> pd.DataFrame:
    """Load the frozen embedding artifact and permute its block within each split."""
    feat_names = embedding_feature_names()
    frame = pd.read_parquet(source_path)
    expected_cols = ["TransactionID", "split", *feat_names]
    if list(frame.columns) != expected_cols:
        raise ValueError(f"Source embedding columns must be exactly {expected_cols}.")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} source embedding rows.")
    return permute_block_within_split(
        frame, feat_names, seed, expected_split_counts=dict(EXPECTED_SPLIT_COUNTS)
    )


def write_shuffled_embedding_artifacts(
    control: ControlRun, seed: int = SHUFFLE_SEED
) -> dict[str, Any]:
    shuffled = build_shuffled_embeddings(seed=seed)
    control.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    control.report_dir.mkdir(parents=True, exist_ok=True)
    shuffled.to_parquet(
        control.embeddings_path, index=False, engine="pyarrow", compression="snappy"
    )

    source_metadata = read_json(FROZEN_ENCODER_METADATA_PATH)
    feat_names = embedding_feature_names()
    aligned = pd.read_parquet(FROZEN_EMBEDDINGS_PATH, columns=feat_names).to_numpy()
    identical_rows = int(np.all(shuffled[feat_names].to_numpy() == aligned, axis=1).sum())

    metadata = {
        "relation_name": source_metadata["relation_name"],
        "model_name": f"graphsage_card1_embeddings_{control.name}",
        "control_variant": control.name,
        "control_isolates": control.isolates,
        "control_description": control.description,
        "random_seed": seed,
        "architecture": source_metadata["architecture"],
        "training_objective": (
            "none: this block is not trained. It is the frozen G1 embedding "
            "block with its rows permuted within each split."
        ),
        "training_regime": "permutation_null_control",
        "permutation": {
            "scope": "within_split",
            "seed": seed,
            "unit": "whole 32-column row",
            "source_embeddings_path": repository_relative(FROZEN_EMBEDDINGS_PATH),
            "source_embeddings_sha256": file_sha256(FROZEN_EMBEDDINGS_PATH),
            "source_encoder_metadata_path": repository_relative(FROZEN_ENCODER_METADATA_PATH),
            "rows_left_in_original_position": identical_rows,
            "per_split_marginals_preserved": True,
            "rationale": (
                "A global permutation would move train-fitted embedding rows into "
                "the validation partition and confound random alignment with a "
                "distribution shift. Permuting inside each split removes only the "
                "row alignment."
            ),
        },
        "target_labels_used_for_training": False,
        "target_labels_used_in_node_features": False,
        "test_labels_used": False,
        "split_row_counts": dict(EXPECTED_SPLIT_COUNTS),
        "embeddings_path": repository_relative(control.embeddings_path),
        "embedding_row_count": EXPECTED_ROWS,
        "embedding_has_no_nulls": True,
        "embedding_has_no_target_column": True,
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    write_json(control.embedding_metadata_path, metadata)
    print(
        f"[{control.name}] Shuffled embeddings written: {control.embeddings_path} "
        f"({identical_rows:,} of {EXPECTED_ROWS:,} rows unmoved by chance)"
    )
    return metadata


def compute_control_significance(
    control_validation_predictions: pd.DataFrame,
) -> dict[str, Any]:
    """Paired bootstrap PR-AUC deltas against B0, B1-card1 and the frozen G1."""
    references = {
        "b0": load_validation_predictions(B0_VALIDATION_PREDICTIONS_PATH, "b0_prediction"),
        "b1_card1": load_validation_predictions(
            B1_CARD1_VALIDATION_PREDICTIONS_PATH, "b1_card1_prediction"
        ),
        "g1": load_validation_predictions(G1_VALIDATION_PREDICTIONS_PATH, "g1_prediction"),
    }

    merged = control_validation_predictions[["TransactionID", "isFraud", "prediction"]].rename(
        columns={"prediction": "control_prediction"}
    )
    for label, frame in references.items():
        merged = merged.merge(
            frame[["TransactionID", "isFraud", f"{label}_prediction"]].rename(
                columns={"isFraud": f"isFraud_{label}"}
            ),
            on="TransactionID",
            validate="one_to_one",
        )
    if len(merged) != EXPECTED_SPLIT_COUNTS["validation"]:
        raise AssertionError("Significance merge lost or duplicated validation rows.")
    for label in references:
        if not merged["isFraud"].equals(merged[f"isFraud_{label}"]):
            raise AssertionError(f"isFraud labels disagree between the control and {label}.")

    y_true = merged["isFraud"].to_numpy(dtype=np.int8)
    control_scores = merged["control_prediction"].to_numpy(dtype=np.float64)

    results: dict[str, Any] = {
        "method": (
            "paired_bootstrap_pr_auc_delta: resample validation-row indices with "
            "replacement (identical indices applied to both score vectors per "
            "resample), n_resamples draws, 95% CI from the empirical percentiles "
            "of the resampled PR-AUC delta"
        ),
        "validation_row_count": int(len(merged)),
    }
    for label in REFERENCE_LABELS:
        print(
            f"Running paired bootstrap (control vs {label}, {SIGNIFICANCE_RESAMPLES:,} resamples)..."
        )
        reference_scores = merged[f"{label}_prediction"].to_numpy(dtype=np.float64)
        results[f"control_vs_{label}"] = paired_bootstrap_pr_auc_delta(
            y_true, control_scores, reference_scores
        )
    return results


def build_control_metadata(
    *,
    control: ControlRun,
    model,
    b0_metadata: dict[str, Any],
    b1_card1_metadata: dict[str, Any],
    g1_metadata: dict[str, Any],
    embedding_metadata: dict[str, Any],
    control_features: list[str],
    categorical_columns: list[str],
    category_mappings_sha256: str,
    scale_pos_weight: float,
    train_row_count: int,
    validation_row_count: int,
    learning_curve_summary: dict[str, Any],
    validation_metrics: dict[str, Any],
    protected_hashes: dict[str, str],
    embedding_ranks: dict[str, Any],
    significance: dict[str, Any],
    paths: dict[str, Path],
) -> dict[str, Any]:
    feat_names = embedding_feature_names()
    numeric_columns = [col for col in control_features if col not in categorical_columns]
    return {
        "experiment_name": f"G1_control_{control.name}",
        "model_family": "LightGBM",
        "control_variant": control.name,
        "control_isolates": control.isolates,
        "control_description": control.description,
        "controlled_experiment_definition": (
            "Frozen B0 feature set plus one 32-column embedding block, under the "
            "frozen LightGBM configuration. Only the embedding block differs from "
            "the frozen G1 run."
        ),
        "relation_name": RELATION,
        "embedding_source_path": repository_relative(control.embeddings_path),
        "embedding_metadata_path": repository_relative(control.embedding_metadata_path),
        "embedding_dim": EMBEDDING_DIM,
        "embedding_feature_names": feat_names,
        "embedding_encoder_architecture": embedding_metadata["architecture"],
        "embedding_encoder_readout": embedding_metadata["architecture"].get("readout"),
        "embedding_cross_fitting": embedding_metadata.get("cross_fitting"),
        "embedding_permutation": embedding_metadata.get("permutation"),
        "embedding_training_budget": embedding_metadata.get("mini_batch_budget"),
        "embedding_early_stopping": embedding_metadata.get("early_stopping"),
        "b0_reference_metrics_path": repository_relative(B0_METRICS_PATH),
        "b1_card1_reference_metrics_path": repository_relative(B1_CARD1_METRICS_PATH),
        "g1_reference_metrics_path": repository_relative(G1_METRICS_PATH),
        "frozen_reference_artifact_sha256": protected_hashes,
        "b0_feature_count": len(b0_metadata["feature_columns"]),
        "b1_card1_feature_count": b1_card1_metadata["b1_feature_count"],
        "g1_feature_count": g1_metadata["g1_feature_count"],
        "control_feature_count": len(control_features),
        "feature_columns": control_features,
        "categorical_feature_count": len(categorical_columns),
        "numeric_feature_count": len(numeric_columns),
        "categorical_feature_columns": categorical_columns,
        "numeric_feature_columns": numeric_columns,
        "categorical_mapping_reference": repository_relative(CATEGORY_MAPPINGS_PATH),
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
        "actual_stopping_iteration": int(learning_curve_summary["actual_stopping_iteration"]),
        "early_stopping_triggered": bool(learning_curve_summary["early_stopping_triggered"]),
        "train_row_count": int(train_row_count),
        "validation_row_count": int(validation_row_count),
        "evaluation_split": EVALUATION_SPLIT,
        "validation_pr_auc": float(validation_metrics["pr_auc"]),
        "validation_roc_auc": float(validation_metrics["roc_auc"]),
        "validation_ranking_metrics": validation_metrics["ranking_metrics"],
        "test_evaluated": TEST_EVALUATED,
        "embedding_gain_importance": embedding_ranks,
        "statistical_significance": significance,
        "statistical_significance_path": repository_relative(paths["significance"]),
        "model_path": repository_relative(paths["model"]),
        "metrics_path": repository_relative(paths["metrics"]),
        "feature_importance_path": repository_relative(paths["feature_importance"]),
        "validation_predictions_path": repository_relative(paths["validation_predictions"]),
        "learning_curve_path": repository_relative(paths["learning_curve"]),
        "comparison_to_b0_path": repository_relative(paths["comparison_to_b0"]),
        "comparison_to_b1_card1_path": repository_relative(paths["comparison_to_b1_card1"]),
        "comparison_to_g1_path": repository_relative(paths["comparison_to_g1"]),
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


def control_is_complete(control: ControlRun) -> bool:
    """True when every artifact this control publishes is already on disk."""
    return all(path.exists() for key, path in control.paths().items() if key != "report_dir")


def run_control(control_name: str, skip_existing: bool = False) -> None:
    if control_name not in CONTROL_RUNS:
        raise KeyError(f"Unknown control {control_name!r}; known: {sorted(CONTROL_RUNS)}.")
    control = CONTROL_RUNS[control_name]
    if skip_existing and control_is_complete(control):
        print(f"[control-{control.name}] Already complete; skipping.")
        return
    paths = control.paths()
    feat_names = embedding_feature_names()

    all_protected = [*B0_PROTECTED_PATHS, *B1_CARD1_PROTECTED_PATHS, *G1_PROTECTED_PATHS]
    protected_before = snapshot_protected_artifacts(all_protected)

    if control.name == SHUFFLED_VARIANT_NAME and not control.embeddings_path.exists():
        write_shuffled_embedding_artifacts(control)

    if not control.embeddings_path.exists():
        raise FileNotFoundError(
            f"Embedding block for control {control.name!r} not found: "
            f"{control.embeddings_path}\nRun: python -m src.graph.train_graphsage_variants "
            f"--variant {control.name}"
        )

    b0_metadata = load_frozen_b0_metadata()
    b1_card1_metadata = load_frozen_b1_card1_metadata()
    g1_metadata = read_json(G1_METADATA_PATH)
    b0_metrics = read_json(B0_METRICS_PATH)
    b1_card1_metrics = read_json(B1_CARD1_METRICS_PATH)
    g1_metrics = read_json(G1_METRICS_PATH)
    embedding_metadata = load_embedding_metadata(RELATION, path=control.embedding_metadata_path)

    print(f"[control-{control.name}] Isolates: {control.isolates}")
    print(f"[control-{control.name}] Embedding block: {control.embeddings_path}")
    train_df, validation_df, _ = load_g1_datasets(RELATION, embeddings_path=control.embeddings_path)

    b0_features = list(b0_metadata["feature_columns"])
    validate_model_columns_against_frozen_b0(list(train_df.columns), b0_features, feat_names)
    validate_model_columns_against_frozen_b0(list(validation_df.columns), b0_features, feat_names)
    control_features = build_g1_feature_manifest(b0_features, feat_names)
    if len(control_features) != EXPECTED_G1_FEATURE_COUNT:
        raise ValueError(
            f"Control predictor-count sanity check failed: expected "
            f"{EXPECTED_G1_FEATURE_COUNT}, got {len(control_features)}."
        )

    categorical_columns = list(b0_metadata["categorical_feature_columns"])
    if identify_categorical_columns(train_df[control_features]) != categorical_columns:
        raise TypeError("Control categorical features differ from frozen B0.")
    if identify_categorical_columns(validation_df[control_features]) != categorical_columns:
        raise TypeError("Control validation categorical features differ from frozen B0.")
    for col in feat_names:
        if not is_numeric_dtype(train_df[col].dtype):
            raise TypeError(f"Control embedding predictor {col} is not numeric.")

    validation_metadata = validation_df[["TransactionID", "TransactionDT", "isFraud"]].copy()
    y_train = train_df["isFraud"].astype("int8").copy()
    y_validation = validation_df["isFraud"].astype("int8").copy()

    # Release each source partition before materializing the next predictor
    # matrix. Narrowing 470 columns to 467 still copies the whole float64
    # block, so holding both partitions and both copies at once peaks around
    # 3.4GB -- enough to fail on a machine with other work resident. Freeing in
    # this order caps the peak at the train partition's pair instead.
    X_train = train_df[control_features].copy()
    del train_df
    gc.collect()
    X_validation = validation_df[control_features].copy()
    del validation_df
    gc.collect()

    mappings, mapping_sha256 = load_frozen_category_mappings(categorical_columns)
    print(f"[control-{control.name}] Applying frozen B0 categorical mappings (no fitting)...")
    apply_frozen_category_mappings(X_train, X_validation, categorical_columns, mappings)
    assert_supported_model_dtypes(X_train, "control training predictors")
    assert_supported_model_dtypes(X_validation, "control validation predictors")

    scale_pos_weight = calculate_scale_pos_weight(y_train)
    if not np.isclose(scale_pos_weight, float(b0_metadata["scale_pos_weight"]), rtol=0.0, atol=0.0):
        raise ValueError("Control train-only class weight differs from frozen B0.")
    model = build_lightgbm_model(scale_pos_weight, n_estimators=MAX_ESTIMATORS)
    validate_frozen_lightgbm_configuration(model, b0_metadata)

    print(f"[control-{control.name}] Train rows: {len(X_train):,}")
    print(f"[control-{control.name}] Validation rows: {len(X_validation):,}")
    print(f"[control-{control.name}] Predictors: {len(control_features):,}")

    evaluation_results: dict[str, dict[str, list[float]]] = {}
    callbacks = [
        lightgbm.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, first_metric_only=True),
        lightgbm.record_evaluation(evaluation_results),
        lightgbm.log_evaluation(period=LOG_EVALUATION_PERIOD),
    ]
    print(f"[control-{control.name}] Training control LightGBM...")
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
    validation_scores = model.predict_proba(X_validation, num_iteration=model.best_iteration_)[:, 1]
    if len(validation_scores) != EXPECTED_SPLIT_COUNTS[EVALUATION_SPLIT]:
        raise AssertionError("Control validation prediction count is incorrect.")
    if not np.isfinite(validation_scores).all():
        raise AssertionError("Control validation predictions contain non-finite values.")

    metrics = evaluate_validation(y_validation, validation_scores)
    metrics.update(
        {
            "model": f"G1_control_{control.name}",
            "control_variant": control.name,
            "relation": RELATION,
            "weighting": "weighted",
            "scale_pos_weight": float(scale_pos_weight),
            "maximum_estimators": MAX_ESTIMATORS,
            "actual_stopping_iteration": int(learning_curve_summary["actual_stopping_iteration"]),
            "best_iteration": int(model.best_iteration_),
            "best_validation_average_precision": float(
                learning_curve_summary["best_validation_average_precision"]
            ),
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "early_stopping_triggered": bool(learning_curve_summary["early_stopping_triggered"]),
            "estimator_cap_reached": bool(learning_curve_summary["estimator_cap_reached"]),
        }
    )

    feature_importance = build_feature_importance(model)
    if len(feature_importance) != len(control_features):
        raise AssertionError("Control feature importance does not match its manifest.")
    ranks = embedding_gain_ranks(feature_importance, feat_names)

    validation_predictions = validation_metadata.copy()
    validation_predictions["prediction"] = validation_scores

    paths["model"].parent.mkdir(parents=True, exist_ok=True)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)
    model.booster_.save_model(str(paths["model"]), num_iteration=model.best_iteration_)
    write_json(paths["metrics"], metrics)
    feature_importance.to_csv(paths["feature_importance"], index=False)
    validation_predictions.to_parquet(
        paths["validation_predictions"], index=False, engine="pyarrow", compression="snappy"
    )
    learning_curve.to_csv(paths["learning_curve"], index=False)
    build_comparison_table(b0_metrics, metrics, "b0", "control").to_csv(
        paths["comparison_to_b0"], index=False
    )
    build_comparison_table(b1_card1_metrics, metrics, "b1_card1", "control").to_csv(
        paths["comparison_to_b1_card1"], index=False
    )
    build_comparison_table(g1_metrics, metrics, "g1", "control").to_csv(
        paths["comparison_to_g1"], index=False
    )

    significance = compute_control_significance(validation_predictions)
    write_json(paths["significance"], significance)

    write_json(
        paths["metadata"],
        build_control_metadata(
            control=control,
            model=model,
            b0_metadata=b0_metadata,
            b1_card1_metadata=b1_card1_metadata,
            g1_metadata=g1_metadata,
            embedding_metadata=embedding_metadata,
            control_features=control_features,
            categorical_columns=categorical_columns,
            category_mappings_sha256=mapping_sha256,
            scale_pos_weight=scale_pos_weight,
            train_row_count=len(X_train),
            validation_row_count=len(X_validation),
            learning_curve_summary=learning_curve_summary,
            validation_metrics=metrics,
            protected_hashes=protected_before,
            embedding_ranks=ranks,
            significance=significance,
            paths=paths,
        ),
    )
    assert_protected_artifacts_unchanged(
        protected_before, all_protected, label="frozen B0/B1-card1/G1"
    )

    missing = [str(p) for key, p in paths.items() if key != "report_dir" and not p.exists()]
    if missing:
        raise OSError(f"Control artifacts were not created: {missing}.")

    print(f"[control-{control.name}] Best iteration: {model.best_iteration_:,}")
    print(
        f"[control-{control.name}] Early stopping triggered: {'YES' if learning_curve_summary['early_stopping_triggered'] else 'NO'}"
    )
    print(f"[control-{control.name}] Validation PR-AUC:  {metrics['pr_auc']:.12f}")
    print(f"[control-{control.name}] Validation ROC-AUC: {metrics['roc_auc']:.12f}")
    print(
        f"[control-{control.name}] Embedding gain-rank range: {ranks['best_embedding_gain_rank']}-{ranks['worst_embedding_gain_rank']} of {ranks['total_features_in_model']}"
    )
    for label in REFERENCE_LABELS:
        block = significance[f"control_vs_{label}"]
        print(
            f"[control-{control.name}] vs {label}: delta {block['observed_delta']:+.5f} "
            f"95% CI [{block['ci_lower_95']:+.5f}, {block['ci_upper_95']:+.5f}] "
            f"excludes zero: {block['excludes_zero']}"
        )
    print(f"[control-{control.name}] Reports saved: {paths['report_dir']}")
    print(f"[control-{control.name}] Frozen B0/B1-card1/G1 artifacts unchanged: YES")
    print(f"[control-{control.name}] Final test evaluated: NO")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a G1 attribution control through the frozen LightGBM configuration."
    )
    parser.add_argument(
        "--control",
        action="append",
        choices=CONTROL_ORDER,
        help="Control to run; repeatable. Defaults to every control, in reporting order.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a control whose artifacts already exist.",
    )
    args = parser.parse_args()
    for name in args.control or CONTROL_ORDER:
        run_control(name, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
