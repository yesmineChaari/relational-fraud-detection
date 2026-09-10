"""Execute the one-shot final test protocol.

Nothing here is a decision. What is scored, which metrics, the single comparison
and its outcomes all come from `final_test_protocol`, which must be committed and
unmodified before this runs. This module only enforces that protocol.

Two modes.

**Dry run (the default).** Every guard that does not need the test partition:
the protocol is committed, the read has not been claimed, and every named model
re-scores the validation rows through this module's inference path and reproduces
its persisted validation predictions exactly. The test partition is never opened,
so a dry run can be repeated as often as needed.

**Execute (`--execute`).** Irreversible. The read is *claimed* before the test
partition is opened: a marker is written to the output directory first. A crash
midway therefore still counts as the one read, and any rerun is refused rather
than quietly becoming a second look. Only then is the test partition loaded and
scored.

Every input artifact is hashed before and after, so the run provably changes
nothing it reads.

Outputs (execute only):
  reports/final_test/READ_CLAIMED.json
  reports/final_test/<model>/test_metrics.json
  reports/final_test/<model>/test_predictions.parquet
  reports/final_test/validation_to_test_gap.csv
  reports/final_test/final_test_summary.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config.paths import ROOT_DIR
from src.features.build_relational_features import _feature_names
from src.features.build_relational_features import _output_path as relational_output_path
from src.models import final_test_protocol as protocol
from src.models.significance import paired_bootstrap_pr_auc_delta
from src.models.train_lightgbm_baseline import (
    EXPECTED_ROWS,
    EXPECTED_SPLIT_COUNTS,
    MODEL_DATASET_PATH,
    apply_category_mapping,
    fraction_metric_name,
    identify_categorical_columns,
    precision_recall_at_fraction,
    validate_split_counts,
)
from src.models.train_lightgbm_relational import (
    attach_relational_features,
    build_b1_feature_manifest,
    load_frozen_b0_metadata,
    load_frozen_category_mappings,
    validate_relational_merge,
)

PROTOCOL_PATH = Path(protocol.__file__).resolve()
OUTPUT_DIR = ROOT_DIR / protocol.OUTPUT_DIR
CLAIM_MARKER = "READ_CLAIMED.json"
SUMMARY_NAME = "final_test_summary.json"
GAP_NAME = "validation_to_test_gap.csv"
RELATION = "card1"

# Which predictor set each named model was trained on. Must cover exactly the
# protocol's models, so a model cannot be added here without the protocol.
FEATURE_SET: dict[str, str] = {
    "b0": "b0",
    "b1_card1": "b1_card1",
    "b0_frozen": "b0",
    "b1_card1_frozen": "b1_card1",
}


def all_models() -> dict[str, dict[str, str]]:
    """Primary then secondary models, as the protocol names them."""
    overlap = set(protocol.PRIMARY_MODELS) & set(protocol.SECONDARY_MODELS)
    if overlap:
        raise ValueError(f"Models named as both primary and secondary: {sorted(overlap)}.")
    models = {**protocol.PRIMARY_MODELS, **protocol.SECONDARY_MODELS}
    if set(models) != set(FEATURE_SET):
        raise ValueError(
            f"Executor feature sets {sorted(FEATURE_SET)} do not match the protocol's "
            f"models {sorted(models)}."
        )
    not_scored = set(protocol.NOT_SCORED) & set(models)
    if not_scored:
        raise ValueError(f"Models listed as not scored are scheduled: {sorted(not_scored)}.")
    return models


# ---------------------------------------------------------------------------
# Guards that run before the test partition could be opened.
# ---------------------------------------------------------------------------


def protocol_commit_status(path: Path = PROTOCOL_PATH, run: Any = subprocess.run) -> str:
    """The commit that last touched the protocol, or raise if it is not clean."""
    relative = path.resolve().relative_to(ROOT_DIR).as_posix()

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return run(["git", *args], cwd=ROOT_DIR, capture_output=True, text=True, check=False)

    if git("ls-files", "--error-unmatch", relative).returncode != 0:
        raise RuntimeError(f"The protocol {relative} is not committed.")
    if git("status", "--porcelain", "--", relative).stdout.strip():
        raise RuntimeError(f"The protocol {relative} has uncommitted modifications.")
    commit = git("log", "-1", "--format=%H", "--", relative).stdout.strip()
    if not commit:
        raise RuntimeError(f"No commit found for the protocol {relative}.")
    return commit


def assert_read_not_claimed(output_dir: Path) -> None:
    if output_dir.exists():
        raise FileExistsError(
            f"{output_dir} already exists, so the one-shot test read has been claimed. "
            "The protocol forbids a second read. If the previous run crashed, that is "
            "a deviation to record, not a reason to read again."
        )


def claim_read(output_dir: Path, protocol_commit: str) -> Path:
    """Mark the read as spent before a single test row is loaded."""
    output_dir.mkdir(parents=True, exist_ok=False)
    marker = output_dir / CLAIM_MARKER
    write_json(
        marker,
        {
            "claimed_at_utc": datetime.now(timezone.utc).isoformat(),
            "protocol_version": protocol.PROTOCOL_VERSION,
            "protocol_commit": protocol_commit,
            "note": (
                "Written before the test partition was opened. Its presence means "
                "the one-shot read is spent, whether or not the run completed."
            ),
        },
    )
    return marker


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(paths: list[Path]) -> dict[str, str]:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Protocol inputs are missing: {missing}.")
    return {str(path): file_sha256(path) for path in paths}


# ---------------------------------------------------------------------------
# Data and inference.
# ---------------------------------------------------------------------------


def load_partition(split: str) -> pd.DataFrame:
    """One evaluation partition. Train is never needed and never loaded here."""
    if split not in {"validation", protocol.TEST_SPLIT}:
        raise ValueError(f"The final test run loads only validation or test, not {split!r}.")
    frame = pd.read_parquet(MODEL_DATASET_PATH, filters=[("split", "==", split)])
    frame = frame.reset_index(drop=True)
    expected = EXPECTED_SPLIT_COUNTS[split]
    if split == protocol.TEST_SPLIT and expected != protocol.EXPECTED_TEST_ROWS:
        raise AssertionError("Protocol and dataset disagree on the test row count.")
    if len(frame) != expected:
        raise ValueError(f"Expected {expected:,} {split} rows; loaded {len(frame):,}.")
    if not frame["split"].astype("string").eq(split).all():
        raise ValueError(f"The {split} partition contains rows from another split.")
    return frame


def attach_card1_features(partition: pd.DataFrame, split: str) -> pd.DataFrame:
    """The same relational merge B1-card1 was trained and validated through."""
    feat_names = _feature_names(RELATION)
    model_index = pd.read_parquet(MODEL_DATASET_PATH, columns=["TransactionID", "split"])
    validate_split_counts(model_index, "model_dataset.parquet final-test index")
    relational = pd.read_parquet(relational_output_path(RELATION))
    if len(relational) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} relational rows; got {len(relational):,}.")
    merged_index = validate_relational_merge(model_index, relational, feat_names)
    return attach_relational_features(partition, merged_index, split, feat_names)


def build_design_matrix(
    partition: pd.DataFrame,
    feature_set: str,
    b0_metadata: dict[str, Any],
    mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    b0_features = list(b0_metadata["feature_columns"])
    if feature_set == "b0":
        columns = b0_features
    elif feature_set == "b1_card1":
        columns = build_b1_feature_manifest(b0_features, _feature_names(RELATION))
    else:
        raise ValueError(f"Unknown feature set {feature_set!r}.")
    missing = [column for column in columns if column not in partition.columns]
    if missing:
        raise ValueError(f"{feature_set} predictors missing from the partition: {missing}.")

    design = partition[columns].copy()
    categorical = list(b0_metadata["categorical_feature_columns"])
    if identify_categorical_columns(design) != categorical:
        raise TypeError(f"{feature_set} categorical predictors differ from frozen B0.")
    for column in categorical:
        design[column] = apply_category_mapping(design[column], mappings[column])
    if list(design.columns) != columns:
        raise AssertionError(f"{feature_set} predictor order changed.")
    return design


def score(model_path: Path, design: pd.DataFrame) -> np.ndarray:
    """Predict with the booster exactly as saved. Nothing is refit."""
    booster = lightgbm.Booster(model_file=str(model_path))
    if booster.feature_name() != list(design.columns):
        raise AssertionError(f"{model_path.name} was trained on a different predictor order.")
    scores = np.asarray(booster.predict(design), dtype=np.float64)
    if len(scores) != len(design) or not np.isfinite(scores).all():
        raise AssertionError(f"{model_path.name} produced missing or non-finite scores.")
    if (scores < 0).any() or (scores > 1).any():
        raise AssertionError(f"{model_path.name} produced scores outside [0, 1].")
    return scores


def assert_reproduces_validation(
    name: str,
    transaction_ids: np.ndarray,
    rescored: np.ndarray,
    persisted_path: Path,
) -> dict[str, Any]:
    """The inference path must regenerate the persisted validation scores exactly."""
    persisted = pd.read_parquet(persisted_path, columns=["TransactionID", "prediction"])
    aligned = pd.DataFrame({"TransactionID": transaction_ids, "rescored": rescored}).merge(
        persisted, on="TransactionID", how="left", validate="one_to_one"
    )
    if len(aligned) != len(persisted) or aligned["prediction"].isna().any():
        raise AssertionError(f"{name}: validation rows do not match the persisted predictions.")
    difference = np.abs(aligned["rescored"].to_numpy() - aligned["prediction"].to_numpy())
    if not np.array_equal(aligned["rescored"].to_numpy(), aligned["prediction"].to_numpy()):
        raise AssertionError(
            f"{name} does not reproduce its persisted validation predictions: "
            f"{int((difference > 0).sum()):,} rows differ, max |diff| {difference.max():.3e}. "
            "Aborting before any test row is scored."
        )
    return {"rows": int(len(aligned)), "identical": True}


# ---------------------------------------------------------------------------
# Metrics, the gap and the outcome.
# ---------------------------------------------------------------------------


def evaluate(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    """The protocol's metric set, and nothing else."""
    pr_auc = float(average_precision_score(labels, scores))
    roc_auc = float(roc_auc_score(labels, scores))
    return {
        "evaluation_split": protocol.TEST_SPLIT,
        "n_transactions": int(len(labels)),
        "n_fraud": int(labels.sum()),
        "fraud_rate": float(labels.mean()),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "ranking_metrics": {
            fraction_metric_name(fraction): precision_recall_at_fraction(labels, scores, fraction)
            for fraction in protocol.TOP_FRACTIONS
        },
        "test_evaluated": True,
    }


def flatten_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    flat = {name: float(metrics[name]) for name in protocol.METRICS}
    for fraction in protocol.TOP_FRACTIONS:
        key = fraction_metric_name(fraction)
        flat[f"{key}_precision"] = float(metrics["ranking_metrics"][key]["precision"])
        flat[f"{key}_recall"] = float(metrics["ranking_metrics"][key]["recall"])
    return flat


def validation_to_test_gap(
    name: str, validation_metrics: dict[str, Any], test_metrics: dict[str, Any]
) -> list[dict[str, Any]]:
    validation = flatten_metrics(validation_metrics)
    test = flatten_metrics(test_metrics)
    return [
        {
            "model": name,
            "metric": metric,
            "validation": validation[metric],
            "test": test[metric],
            "test_minus_validation": test[metric] - validation[metric],
        }
        for metric in validation
    ]


def classify_outcome(ci_lower: float, ci_upper: float) -> str:
    if ci_lower > 0.0:
        return protocol.OUTCOME_REPLICATES
    if ci_upper < 0.0:
        return protocol.OUTCOME_REVERSES
    return protocol.OUTCOME_NOT_CONFIRMED


# ---------------------------------------------------------------------------
# The run.
# ---------------------------------------------------------------------------


def write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def run(execute: bool, output_dir: Path = OUTPUT_DIR, root: Path = ROOT_DIR) -> dict[str, Any]:
    protocol_commit = protocol_commit_status()
    assert_read_not_claimed(output_dir)
    models = all_models()
    inputs = [root / spec[key] for spec in models.values() for key in sorted(spec) if key != "role"]
    hashes_before = snapshot(inputs)

    b0_metadata = load_frozen_b0_metadata()
    categorical = list(b0_metadata["categorical_feature_columns"])
    mappings, mappings_sha256 = load_frozen_category_mappings(categorical)

    # Stage 1 -- validation only. The test partition is not opened here.
    print("Re-scoring validation to prove the inference path is exact...")
    validation = attach_card1_features(load_partition("validation"), "validation")
    reproduction = {}
    for name, spec in models.items():
        design = build_design_matrix(validation, FEATURE_SET[name], b0_metadata, mappings)
        rescored = score(root / spec["model"], design)
        reproduction[name] = assert_reproduces_validation(
            name,
            validation["TransactionID"].to_numpy(),
            rescored,
            root / spec["validation_predictions"],
        )
        print(f"  {name:18s} reproduces {reproduction[name]['rows']:,} validation rows exactly")
    del validation

    if not execute:
        print("\nDry run passed. The test partition was not opened.")
        return {"mode": "dry_run", "protocol_commit": protocol_commit, "reproduction": reproduction}

    # Stage 2 -- the one read. Claimed before the partition is opened.
    claim_read(output_dir, protocol_commit)
    print(f"\nRead claimed in {output_dir}. Opening the test partition...")
    test = attach_card1_features(load_partition(protocol.TEST_SPLIT), protocol.TEST_SPLIT)
    labels = test["isFraud"].to_numpy(dtype=np.int8)
    row_metadata = test[["TransactionID", "TransactionDT", "isFraud"]].copy()

    scores: dict[str, np.ndarray] = {}
    per_model: dict[str, dict[str, Any]] = {}
    gap_rows: list[dict[str, Any]] = []
    for name, spec in models.items():
        design = build_design_matrix(test, FEATURE_SET[name], b0_metadata, mappings)
        scores[name] = score(root / spec["model"], design)
        test_metrics = evaluate(labels, scores[name])
        test_metrics.update({"model": name, "role": spec["role"], "artifact": spec["model"]})
        validation_metrics = read_json(root / spec["validation_metrics"])
        gap_rows.extend(validation_to_test_gap(name, validation_metrics, test_metrics))

        model_dir = output_dir / name
        model_dir.mkdir()
        write_json(model_dir / "test_metrics.json", test_metrics)
        predictions = row_metadata.copy()
        predictions["prediction"] = scores[name]
        predictions.to_parquet(
            model_dir / "test_predictions.parquet", index=False, engine="pyarrow"
        )
        per_model[name] = {
            "primary": name in protocol.PRIMARY_MODELS,
            "validation_pr_auc": float(validation_metrics["pr_auc"]),
            "test_pr_auc": test_metrics["pr_auc"],
            "validation_roc_auc": float(validation_metrics["roc_auc"]),
            "test_roc_auc": test_metrics["roc_auc"],
        }

    comparison = protocol.PRIMARY_COMPARISON
    primary = paired_bootstrap_pr_auc_delta(
        labels,
        scores[comparison["candidate"]],
        scores[comparison["reference"]],
        n_resamples=int(comparison["n_resamples"]),
        seed=int(comparison["seed"]),
    )
    outcome = classify_outcome(primary["ci_lower_95"], primary["ci_upper_95"])
    secondary_delta = float(
        average_precision_score(labels, scores["b1_card1_frozen"])
        - average_precision_score(labels, scores["b0_frozen"])
    )

    if snapshot(inputs) != hashes_before:
        raise AssertionError("A protocol input changed during the run.")

    gap = pd.DataFrame(gap_rows)
    gap.to_csv(output_dir / GAP_NAME, index=False)
    summary = {
        "report_name": "One-shot final test evaluation",
        "protocol_version": protocol.PROTOCOL_VERSION,
        "protocol_commit": protocol_commit,
        "outcome": outcome,
        "outcome_rule": protocol.OUTCOME_RULE,
        "primary_comparison": {**comparison, **primary},
        "secondary_descriptive_pr_auc_delta_frozen_pair": secondary_delta,
        "models": per_model,
        "not_scored": protocol.NOT_SCORED,
        "validation_reproduction": reproduction,
        "categorical_mappings_sha256": mappings_sha256,
        "treatment": protocol.TREATMENT,
        "forbidden_afterwards": protocol.FORBIDDEN_AFTERWARDS,
        "deviations": [],
        "inputs_unchanged": True,
        "test_evaluated": True,
        "versions": {
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    write_json(output_dir / SUMMARY_NAME, summary)
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Claim and perform the one-shot test read. Irreversible.",
    )
    args = parser.parse_args(argv)
    summary = run(execute=args.execute)
    if args.execute:
        primary = summary["primary_comparison"]
        print(f"\noutcome: {summary['outcome']}")
        print(
            f"B1-card1 - B0 test PR-AUC: {primary['observed_delta']:+.5f} "
            f"[{primary['ci_lower_95']:+.5f}, {primary['ci_upper_95']:+.5f}]"
        )


if __name__ == "__main__":
    main()
