from __future__ import annotations

from pathlib import Path

import pandas as pd
from pandas.api.types import is_numeric_dtype, is_object_dtype, is_string_dtype


ROOT_DIR = Path(__file__).resolve().parents[2]

TRANSACTION_PATH = ROOT_DIR / "data" / "raw" / "train_transaction.csv"
IDENTITY_PATH = ROOT_DIR / "data" / "raw" / "train_identity.csv"
SPLIT_PATH = ROOT_DIR / "data" / "processed" / "split_assignment.parquet"
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "model_dataset.parquet"

EXPECTED_ROWS = 590_540
EXPECTED_SPLIT_COUNTS = {
    "train": 413_378,
    "validation": 88_581,
    "test": 88_581,
}
ALLOWED_SPLITS = set(EXPECTED_SPLIT_COUNTS)
DERIVED_TIME_FEATURES = [
    "elapsed_days",
    "hour_in_day",
    "day_in_week_cycle",
]


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    """Raise a clear error when an input is missing required columns."""

    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {sorted(missing)}."
        )


def load_transaction_data(path: Path = TRANSACTION_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Transaction file not found: {path}")
    return pd.read_csv(path)


def load_identity_data(path: Path = IDENTITY_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Identity file not found: {path}")
    return pd.read_csv(path)


def load_split_manifest(path: Path = SPLIT_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Split manifest not found: {path}")
    return pd.read_parquet(path)


def validate_transaction_data(transaction_df: pd.DataFrame) -> None:
    require_columns(
        transaction_df,
        {"TransactionID", "TransactionDT", "isFraud"},
        "train_transaction.csv",
    )

    if len(transaction_df) != EXPECTED_ROWS:
        raise ValueError(
            "Unexpected transaction row count. "
            f"Expected {EXPECTED_ROWS:,}, got {len(transaction_df):,}."
        )
    if transaction_df["TransactionID"].isna().any():
        raise ValueError("train_transaction.csv contains missing TransactionID values.")
    if not transaction_df["TransactionID"].is_unique:
        raise ValueError("train_transaction.csv contains duplicate TransactionID values.")
    if transaction_df["TransactionDT"].isna().any():
        raise ValueError("train_transaction.csv contains missing TransactionDT values.")
    if transaction_df["isFraud"].isna().any():
        raise ValueError("train_transaction.csv contains missing isFraud values.")

    unexpected_labels = set(transaction_df["isFraud"].unique()) - {0, 1}
    if unexpected_labels:
        raise ValueError(
            "train_transaction.csv contains unexpected isFraud labels: "
            f"{sorted(unexpected_labels)}."
        )


def validate_identity_data(
    identity_df: pd.DataFrame,
    transaction_columns: set[str],
) -> None:
    require_columns(identity_df, {"TransactionID"}, "train_identity.csv")

    if identity_df["TransactionID"].isna().any():
        raise ValueError("train_identity.csv contains missing TransactionID values.")
    if not identity_df["TransactionID"].is_unique:
        raise ValueError("train_identity.csv contains duplicate TransactionID values.")

    overlapping_columns = (set(identity_df.columns) & transaction_columns) - {
        "TransactionID"
    }
    if overlapping_columns:
        raise ValueError(
            "Transaction and identity inputs contain unexpected overlapping columns: "
            f"{sorted(overlapping_columns)}."
        )


def split_counts(split_series: pd.Series) -> dict[str, int]:
    return {
        str(name): int(count)
        for name, count in split_series.astype("string").value_counts().items()
    }


def validate_split_manifest(split_df: pd.DataFrame) -> None:
    require_columns(
        split_df,
        {"TransactionID", "TransactionDT", "isFraud", "split"},
        "split_assignment.parquet",
    )

    if len(split_df) != EXPECTED_ROWS:
        raise ValueError(
            "Unexpected split-manifest row count. "
            f"Expected {EXPECTED_ROWS:,}, got {len(split_df):,}."
        )
    if split_df["TransactionID"].isna().any():
        raise ValueError("Split manifest contains missing TransactionID values.")
    if not split_df["TransactionID"].is_unique:
        raise ValueError("Split manifest contains duplicate TransactionID values.")
    if split_df["TransactionDT"].isna().any():
        raise ValueError("Split manifest contains missing TransactionDT values.")
    if split_df["isFraud"].isna().any():
        raise ValueError("Split manifest contains missing isFraud values.")
    if split_df["split"].isna().any():
        raise ValueError("Split manifest contains missing split assignments.")

    unexpected_labels = set(split_df["isFraud"].unique()) - {0, 1}
    if unexpected_labels:
        raise ValueError(
            "Split manifest contains unexpected isFraud labels: "
            f"{sorted(unexpected_labels)}."
        )

    observed_splits = set(split_df["split"].astype("string").unique())
    unexpected_splits = observed_splits - ALLOWED_SPLITS
    if unexpected_splits:
        raise ValueError(
            f"Split manifest contains unexpected split values: {sorted(unexpected_splits)}."
        )

    actual_counts = split_counts(split_df["split"])
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "Unexpected split counts. "
            f"Expected {EXPECTED_SPLIT_COUNTS}, got {actual_counts}."
        )

    split_as_string = split_df["split"].astype("string")
    train_max = split_df.loc[
        split_as_string == "train", "TransactionDT"
    ].max()
    validation_min = split_df.loc[
        split_as_string == "validation", "TransactionDT"
    ].min()
    validation_max = split_df.loc[
        split_as_string == "validation", "TransactionDT"
    ].max()
    test_min = split_df.loc[split_as_string == "test", "TransactionDT"].min()

    if not train_max < validation_min:
        raise ValueError("Train and validation overlap in TransactionDT.")
    if not validation_max < test_min:
        raise ValueError("Validation and test overlap in TransactionDT.")


def validate_manifest_matches_transactions(
    transaction_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> None:
    """Confirm manifest IDs, elapsed times, and labels match the raw input."""

    comparison = transaction_df[
        ["TransactionID", "TransactionDT", "isFraud"]
    ].merge(
        split_df[["TransactionID", "TransactionDT", "isFraud"]],
        on="TransactionID",
        how="outer",
        suffixes=("_raw", "_manifest"),
        indicator=True,
        validate="one_to_one",
    )

    membership_mismatches = int((comparison["_merge"] != "both").sum())
    if membership_mismatches:
        raise ValueError(
            "Raw transactions and the split manifest contain different TransactionIDs: "
            f"{membership_mismatches:,} mismatches."
        )

    dt_mismatches = int(
        (
            comparison["TransactionDT_raw"]
            != comparison["TransactionDT_manifest"]
        ).sum()
    )
    label_mismatches = int(
        (comparison["isFraud_raw"] != comparison["isFraud_manifest"]).sum()
    )
    if dt_mismatches or label_mismatches:
        raise ValueError(
            "Split manifest does not match the raw transaction data: "
            f"TransactionDT mismatches={dt_mismatches:,}, "
            f"isFraud mismatches={label_mismatches:,}."
        )


def merge_identity_data(
    transaction_df: pd.DataFrame,
    identity_df: pd.DataFrame,
) -> pd.DataFrame:
    identity_ids = pd.Index(identity_df["TransactionID"])
    merged = transaction_df.merge(
        identity_df,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    if len(merged) != len(transaction_df):
        raise ValueError(
            "Identity merge changed the transaction row count: "
            f"before={len(transaction_df):,}, after={len(merged):,}."
        )
    if not merged["TransactionID"].is_unique:
        raise ValueError("Identity merge produced duplicate TransactionID values.")

    # This is reporting metadata only and is explicitly excluded by the trainer.
    merged["has_identity"] = merged["TransactionID"].isin(identity_ids)
    return merged


def attach_split(
    merged_df: pd.DataFrame,
    split_df: pd.DataFrame,
) -> pd.DataFrame:
    result = merged_df.merge(
        split_df[["TransactionID", "split"]],
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )

    if len(result) != EXPECTED_ROWS:
        raise ValueError(
            "Attaching the split changed the model-dataset row count: "
            f"expected={EXPECTED_ROWS:,}, got={len(result):,}."
        )
    if not result["TransactionID"].is_unique:
        raise ValueError("Attaching the split produced duplicate TransactionID values.")
    if result["split"].isna().any():
        raise ValueError("Some transactions have no authoritative split assignment.")

    actual_counts = split_counts(result["split"])
    if actual_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            "Unexpected split counts after attaching the manifest. "
            f"Expected {EXPECTED_SPLIT_COUNTS}, got {actual_counts}."
        )
    return result


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add elapsed-cycle features without inventing a calendar origin."""

    # The production table is wide, so mutate this newly merged frame in place
    # instead of making a second multi-gigabyte copy.
    df["elapsed_days"] = df["TransactionDT"] / 86_400.0
    df["hour_in_day"] = (
        (df["TransactionDT"] // 3_600) % 24
    ).astype("int16")
    df["day_in_week_cycle"] = (
        (df["TransactionDT"] // 86_400) % 7
    ).astype("int8")
    return df


def is_categorical_dtype(dtype: object) -> bool:
    return (
        is_object_dtype(dtype)
        or is_string_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def build_summary(df: pd.DataFrame) -> dict[str, object]:
    predictor_metadata = {"TransactionID", "isFraud", "split", "has_identity"}
    raw_predictors = [
        column
        for column in df.columns
        if column not in predictor_metadata and column not in DERIVED_TIME_FEATURES
    ]
    categorical_columns = [
        column
        for column in raw_predictors
        if is_categorical_dtype(df[column].dtype)
    ]
    numeric_columns = [
        column
        for column in raw_predictors
        if is_numeric_dtype(df[column].dtype)
    ]

    split_as_string = df["split"].astype("string")
    fraud_by_split: dict[str, dict[str, float | int]] = {}
    for split_name in EXPECTED_SPLIT_COUNTS:
        labels = df.loc[split_as_string == split_name, "isFraud"]
        fraud_by_split[split_name] = {
            "n_transactions": int(len(labels)),
            "n_fraud": int(labels.sum()),
            "fraud_rate": float(labels.mean()),
        }

    identity_matched = int(df["has_identity"].sum())
    return {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "split_counts": split_counts(df["split"]),
        "fraud_by_split": fraud_by_split,
        "n_identity_matched": identity_matched,
        "n_identity_missing": int(len(df) - identity_matched),
        "identity_coverage_pct": float(identity_matched / len(df) * 100.0),
        "n_categorical_raw_columns": int(len(categorical_columns)),
        "n_numeric_raw_columns": int(len(numeric_columns)),
    }


def validate_model_dataset(df: pd.DataFrame) -> None:
    require_columns(
        df,
        {
            "TransactionID",
            "TransactionDT",
            "isFraud",
            "split",
            "has_identity",
            *DERIVED_TIME_FEATURES,
        },
        "model dataset",
    )
    if len(df) != EXPECTED_ROWS:
        raise ValueError(
            f"Model dataset must contain {EXPECTED_ROWS:,} rows; got {len(df):,}."
        )
    if not df["TransactionID"].is_unique:
        raise ValueError("Model dataset contains duplicate TransactionID values.")
    if df["split"].isna().any():
        raise ValueError("Model dataset contains missing split assignments.")
    if df["isFraud"].isna().any():
        raise ValueError("Model dataset contains missing isFraud values.")
    if set(df["isFraud"].unique()) - {0, 1}:
        raise ValueError("Model dataset contains non-binary isFraud values.")
    if split_counts(df["split"]) != EXPECTED_SPLIT_COUNTS:
        raise ValueError("Model dataset does not preserve the frozen split counts.")

    accidental_merge_columns = {
        "isFraud_x",
        "isFraud_y",
        "split_x",
        "split_y",
        "TransactionDT_x",
        "TransactionDT_y",
    } & set(df.columns)
    if accidental_merge_columns:
        raise ValueError(
            "Model dataset contains accidental merge columns: "
            f"{sorted(accidental_merge_columns)}."
        )


def save_model_dataset(
    df: pd.DataFrame,
    path: Path = OUTPUT_PATH,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False, engine="pyarrow", compression="snappy")
    if not path.exists() or path.stat().st_size == 0:
        raise OSError(f"Model dataset was not written successfully: {path}")


def print_summary(summary: dict[str, object], output_path: Path) -> None:
    print("\nModel dataset built successfully.\n")
    print(f"Rows: {summary['rows']:,}")
    print(f"Columns: {summary['columns']:,}")
    print("\nSplit counts:")
    for split_name in EXPECTED_SPLIT_COUNTS:
        count = summary["split_counts"][split_name]  # type: ignore[index]
        print(f"  {split_name:<10} {count:>8,}")

    for split_name in EXPECTED_SPLIT_COUNTS:
        split_summary = summary["fraud_by_split"][split_name]  # type: ignore[index]
        print(f"\n{split_name.capitalize()} fraud:")
        print(
            f"  {split_summary['n_fraud']:,} / "
            f"{split_summary['n_transactions']:,}"
        )
        print(f"  rate = {split_summary['fraud_rate']:.8%}")

    print(f"\nIdentity matched: {summary['n_identity_matched']:,}")
    print(f"Identity missing: {summary['n_identity_missing']:,}")
    print(f"Identity coverage: {summary['identity_coverage_pct']:.4f}%")
    print(
        "Categorical raw columns: "
        f"{summary['n_categorical_raw_columns']:,}"
    )
    print(f"Numeric raw columns: {summary['n_numeric_raw_columns']:,}")
    print(f"\nOutput: {output_path}")


def main() -> None:
    print(f"Loading transactions: {TRANSACTION_PATH}")
    transaction_df = load_transaction_data()
    validate_transaction_data(transaction_df)

    print(f"Loading identity data: {IDENTITY_PATH}")
    identity_df = load_identity_data()
    validate_identity_data(identity_df, set(transaction_df.columns))

    print(f"Loading frozen split manifest: {SPLIT_PATH}")
    split_df = load_split_manifest()
    validate_split_manifest(split_df)
    validate_manifest_matches_transactions(transaction_df, split_df)

    print("Left-joining identity attributes...")
    model_df = merge_identity_data(transaction_df, identity_df)
    del transaction_df, identity_df
    model_df = attach_split(model_df, split_df)
    del split_df
    model_df = add_time_features(model_df)
    validate_model_dataset(model_df)

    summary = build_summary(model_df)
    print(f"Writing model dataset: {OUTPUT_PATH}")
    save_model_dataset(model_df)
    print_summary(summary, OUTPUT_PATH)


if __name__ == "__main__":
    main()
