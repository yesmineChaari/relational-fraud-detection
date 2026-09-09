from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT_DIR / "data" / "raw" / "train_transaction.csv"
OUTPUT_DIR = ROOT_DIR / "data" / "processed"
REPORT_DIR = ROOT_DIR / "reports"

SPLIT_PATH = OUTPUT_DIR / "split_assignment.parquet"
SUMMARY_PATH = REPORT_DIR / "split_summary.csv"
METADATA_PATH = REPORT_DIR / "split_metadata.json"

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15


def find_timestamp_safe_boundary(
    transaction_times: np.ndarray,
    nominal_boundary: int,
) -> int:
    """
    Move a nominal row boundary to the end of its TransactionDT tie group.

    Example:
        [..., 100, 100, 100, | 100, 101, ...]
                              nominal boundary

    becomes:
        [..., 100, 100, 100, 100, | 101, ...]
    """

    if nominal_boundary <= 0:
        return 0

    if nominal_boundary >= len(transaction_times):
        return len(transaction_times)

    timestamp_before_boundary = transaction_times[nominal_boundary - 1]

    return int(
        np.searchsorted(
            transaction_times,
            timestamp_before_boundary,
            side="right",
        )
    )


def build_split_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby("split", observed=True)
        .agg(
            n_transactions=("TransactionID", "size"),
            n_fraud=("isFraud", "sum"),
            fraud_rate=("isFraud", "mean"),
            min_transaction_dt=("TransactionDT", "min"),
            max_transaction_dt=("TransactionDT", "max"),
        )
        .reset_index()
    )

    summary["fraction_of_dataset"] = summary["n_transactions"] / len(df)

    return summary


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Transaction file not found: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading {INPUT_PATH} ...")

    df = pd.read_csv(
        INPUT_PATH,
        usecols=[
            "TransactionID",
            "TransactionDT",
            "isFraud",
        ],
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    if df["TransactionID"].isna().any():
        raise ValueError("TransactionID contains missing values.")

    if not df["TransactionID"].is_unique:
        raise ValueError("TransactionID must be unique.")

    if df["TransactionDT"].isna().any():
        raise ValueError("TransactionDT contains missing values.")

    if df["isFraud"].isna().any():
        raise ValueError("isFraud contains missing values.")

    unexpected_labels = set(df["isFraud"].unique()) - {0, 1}

    if unexpected_labels:
        raise ValueError(f"Unexpected isFraud labels: {unexpected_labels}")

    # ---------------------------------------------------------
    # Temporal ordering
    # ---------------------------------------------------------

    # TransactionID is only a deterministic tie-breaker.
    # It is NOT being used to determine chronology.
    df = df.sort_values(
        ["TransactionDT", "TransactionID"],
        kind="mergesort",
    ).reset_index(drop=True)

    n = len(df)

    nominal_train_end = int(n * TRAIN_RATIO)

    nominal_validation_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

    times = df["TransactionDT"].to_numpy()

    # Adjust boundaries so identical TransactionDT values
    # never appear on both sides of a split.
    train_end = find_timestamp_safe_boundary(
        times,
        nominal_train_end,
    )

    validation_end = find_timestamp_safe_boundary(
        times,
        nominal_validation_end,
    )

    if not (0 < train_end < validation_end < n):
        raise ValueError(
            "Invalid temporal split boundaries: "
            f"train_end={train_end}, "
            f"validation_end={validation_end}, "
            f"n={n}"
        )

    # ---------------------------------------------------------
    # Assign splits
    # ---------------------------------------------------------

    df["split"] = "test"

    df.loc[
        : train_end - 1,
        "split",
    ] = "train"

    df.loc[
        train_end : validation_end - 1,
        "split",
    ] = "validation"

    df["split"] = pd.Categorical(
        df["split"],
        categories=["train", "validation", "test"],
        ordered=True,
    )

    # ---------------------------------------------------------
    # Temporal integrity checks
    # ---------------------------------------------------------

    train_max = df.loc[
        df["split"] == "train",
        "TransactionDT",
    ].max()

    validation_min = df.loc[
        df["split"] == "validation",
        "TransactionDT",
    ].min()

    validation_max = df.loc[
        df["split"] == "validation",
        "TransactionDT",
    ].max()

    test_min = df.loc[
        df["split"] == "test",
        "TransactionDT",
    ].min()

    if not train_max < validation_min:
        raise AssertionError("Train and validation overlap in TransactionDT.")

    if not validation_max < test_min:
        raise AssertionError("Validation and test overlap in TransactionDT.")

    # ---------------------------------------------------------
    # Save authoritative split manifest
    # ---------------------------------------------------------

    df.to_parquet(
        SPLIT_PATH,
        index=False,
    )

    summary = build_split_summary(df)

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    metadata = {
        "source_file": str(INPUT_PATH),
        "n_transactions": int(n),
        "requested_ratios": {
            "train": TRAIN_RATIO,
            "validation": VALIDATION_RATIO,
            "test": TEST_RATIO,
        },
        "nominal_boundaries": {
            "train_end_row": nominal_train_end,
            "validation_end_row": nominal_validation_end,
        },
        "timestamp_safe_boundaries": {
            "train_end_row": train_end,
            "validation_end_row": validation_end,
        },
        "boundary_transaction_dt": {
            "train_max": int(train_max),
            "validation_min": int(validation_min),
            "validation_max": int(validation_max),
            "test_min": int(test_min),
        },
        "tie_safe": True,
        "sorting": [
            "TransactionDT",
            "TransactionID",
        ],
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metadata,
            f,
            indent=2,
        )

    print("\nTemporal split created successfully.\n")

    print(summary.to_string(index=False))

    print(f"\nSplit manifest: {SPLIT_PATH}")
    print(f"Summary:        {SUMMARY_PATH}")
    print(f"Metadata:       {METADATA_PATH}")


if __name__ == "__main__":
    main()
