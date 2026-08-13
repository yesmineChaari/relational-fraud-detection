from pathlib import Path

import numpy as np
import pandas as pd


RAW_DIR = Path("data/raw")
REPORT_DIR = Path("reports/data_profile")

TRAIN_TRANSACTION = RAW_DIR / "train_transaction.csv"
TRAIN_IDENTITY = RAW_DIR / "train_identity.csv"


def profile_columns(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for column in df.columns:
        series = df[column]

        missing_count = series.isna().sum()
        non_null_count = series.notna().sum()
        unique_count = series.nunique(dropna=True)

        rows.append(
            {
                "column": column,
                "dtype": str(series.dtype),
                "rows": len(series),
                "non_null_count": non_null_count,
                "missing_count": missing_count,
                "missing_pct": missing_count / len(series),
                "unique_count": unique_count,
                "unique_pct_non_null": (
                    unique_count / non_null_count
                    if non_null_count > 0
                    else np.nan
                ),
                "is_all_missing": non_null_count == 0,
                "is_constant": unique_count <= 1,
            }
        )

    return pd.DataFrame(rows)


def numeric_profile(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=np.number)

    if numeric.empty:
        return pd.DataFrame()

    stats = numeric.describe(
        percentiles=[0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    ).T

    stats.index.name = "column"

    return stats.reset_index()


def categorical_profile(df: pd.DataFrame) -> pd.DataFrame:
    categorical = df.select_dtypes(
        include=["object", "category"]
    )

    rows = []

    for column in categorical.columns:
        series = categorical[column]

        value_counts = series.value_counts(
            dropna=False
        )

        top_values = value_counts.head(10)

        rows.append(
            {
                "column": column,
                "missing_pct": series.isna().mean(),
                "unique_count": series.nunique(dropna=True),
                "top_10_values": " | ".join(
                    f"{repr(value)}:{count}"
                    for value, count in top_values.items()
                ),
            }
        )

    return pd.DataFrame(rows)


def print_feature_families(columns: list[str]) -> None:
    families = {
        "card": [],
        "addr": [],
        "dist": [],
        "email": [],
        "C": [],
        "D": [],
        "M": [],
        "V": [],
        "id": [],
        "device": [],
        "other": [],
    }

    for column in columns:
        if column.startswith("card"):
            families["card"].append(column)

        elif column.startswith("addr"):
            families["addr"].append(column)

        elif column.startswith("dist"):
            families["dist"].append(column)

        elif "emaildomain" in column:
            families["email"].append(column)

        elif column.startswith("C") and column[1:].isdigit():
            families["C"].append(column)

        elif column.startswith("D") and column[1:].isdigit():
            families["D"].append(column)

        elif column.startswith("M") and column[1:].isdigit():
            families["M"].append(column)

        elif column.startswith("V") and column[1:].isdigit():
            families["V"].append(column)

        elif column.startswith("id_"):
            families["id"].append(column)

        elif column.startswith("Device"):
            families["device"].append(column)

        else:
            families["other"].append(column)

    print("\nFEATURE FAMILIES")
    print("=" * 80)

    for family, family_columns in families.items():
        if family_columns:
            print(
                f"{family:10s}: "
                f"{len(family_columns):3d} columns"
            )
            print(f"  {family_columns}")


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading train_transaction.csv...")
    transaction = pd.read_csv(TRAIN_TRANSACTION)

    print("Loading train_identity.csv...")
    identity = pd.read_csv(TRAIN_IDENTITY)

    # ---------------------------------------------------------
    # Overall dataset information
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("DATASET SUMMARY")
    print("=" * 80)

    print(
        f"Transaction table: "
        f"{transaction.shape[0]:,} rows x "
        f"{transaction.shape[1]:,} columns"
    )

    print(
        f"Identity table: "
        f"{identity.shape[0]:,} rows x "
        f"{identity.shape[1]:,} columns"
    )

    print("\nAll transaction columns:")
    print(transaction.columns.tolist())

    print("\nAll identity columns:")
    print(identity.columns.tolist())

    # ---------------------------------------------------------
    # Feature families
    # ---------------------------------------------------------

    print_feature_families(
        transaction.columns.tolist()
        + identity.columns.tolist()
    )

    # ---------------------------------------------------------
    # Target
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("TARGET")
    print("=" * 80)

    target_counts = transaction["isFraud"].value_counts()

    print(target_counts)
    print()

    fraud_rate = transaction["isFraud"].mean()

    print(f"Fraud rate: {fraud_rate:.4%}")

    # ---------------------------------------------------------
    # Time
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("TIME")
    print("=" * 80)

    min_dt = transaction["TransactionDT"].min()
    max_dt = transaction["TransactionDT"].max()

    print(f"TransactionDT min: {min_dt:,}")
    print(f"TransactionDT max: {max_dt:,}")
    print(
        f"Observed duration: "
        f"{(max_dt - min_dt) / 86400:.2f} days"
    )

    # ---------------------------------------------------------
    # Identity relationship
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("IDENTITY RELATIONSHIP")
    print("=" * 80)

    print(
        "TransactionID unique in transaction: ",
        transaction["TransactionID"].is_unique,
    )

    print(
        "TransactionID unique in identity: ",
        identity["TransactionID"].is_unique,
    )

    identity_ids = set(identity["TransactionID"])
    transaction_ids = set(transaction["TransactionID"])

    orphan_identity_ids = identity_ids - transaction_ids

    print(
        f"Identity records without transaction: "
        f"{len(orphan_identity_ids):,}"
    )

    identity_coverage = (
        transaction["TransactionID"]
        .isin(identity["TransactionID"])
        .mean()
    )

    print(
        f"Transactions with identity data: "
        f"{identity_coverage:.2%}"
    )

    # ---------------------------------------------------------
    # Full column profiles
    # ---------------------------------------------------------

    print("\nProfiling every transaction column...")
    transaction_profile = profile_columns(transaction)

    print("Profiling every identity column...")
    identity_profile = profile_columns(identity)

    transaction_profile.to_csv(
        REPORT_DIR / "transaction_columns.csv",
        index=False,
    )

    identity_profile.to_csv(
        REPORT_DIR / "identity_columns.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Numeric distributions
    # ---------------------------------------------------------

    print("Computing numeric statistics...")

    numeric_profile(transaction).to_csv(
        REPORT_DIR / "transaction_numeric_stats.csv",
        index=False,
    )

    numeric_profile(identity).to_csv(
        REPORT_DIR / "identity_numeric_stats.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Categorical distributions
    # ---------------------------------------------------------

    print("Computing categorical statistics...")

    categorical_profile(transaction).to_csv(
        REPORT_DIR / "transaction_categorical_stats.csv",
        index=False,
    )

    categorical_profile(identity).to_csv(
        REPORT_DIR / "identity_categorical_stats.csv",
        index=False,
    )

    # ---------------------------------------------------------
    # Important quality warnings
    # ---------------------------------------------------------

    print("\n" + "=" * 80)
    print("DATA QUALITY FLAGS")
    print("=" * 80)

    combined = pd.concat(
        [
            transaction_profile.assign(table="transaction"),
            identity_profile.assign(table="identity"),
        ],
        ignore_index=True,
    )

    print("\nColumns >90% missing:")
    print(
        combined.loc[
            combined["missing_pct"] > 0.90,
            ["table", "column", "missing_pct"],
        ].to_string(index=False)
    )

    print("\nConstant / all-missing columns:")
    print(
        combined.loc[
            combined["is_constant"],
            [
                "table",
                "column",
                "missing_pct",
                "unique_count",
            ],
        ].to_string(index=False)
    )

    print("\nVery high-cardinality columns (>90% unique):")
    print(
        combined.loc[
            combined["unique_pct_non_null"] > 0.90,
            [
                "table",
                "column",
                "unique_count",
                "unique_pct_non_null",
            ],
        ].to_string(index=False)
    )

    print("\nReports written to:")
    print(REPORT_DIR.resolve())


if __name__ == "__main__":
    main()