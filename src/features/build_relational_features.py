from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype


ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "data" / "processed" / "model_dataset.parquet"
OUTPUT_PATH = (
    ROOT_DIR
    / "data"
    / "processed"
    / "relational_features_card_core_addr1.parquet"
)
METADATA_PATH = (
    ROOT_DIR
    / "reports"
    / "relational_features"
    / "card_core_addr1_metadata.json"
)

EXPECTED_ROWS = 590_540
RELATION_NAME = "card_core_addr1"
GROUP_COLUMNS = ["card1", "card2", "card3", "card5", "addr1"]
INPUT_COLUMNS = ["TransactionID", "TransactionDT", "split", *GROUP_COLUMNS]
RELATIONAL_FEATURES = [
    "card_core_addr1_prior_count",
    "card_core_addr1_prior_count_24h",
    "card_core_addr1_prior_count_7d",
    "card_core_addr1_time_since_previous_hours",
]
OUTPUT_COLUMNS = ["TransactionID", *RELATIONAL_FEATURES]
WINDOW_24H_SECONDS = 86_400
WINDOW_7D_SECONDS = 604_800


def require_columns(
    df: pd.DataFrame,
    required: set[str],
    source_name: str,
) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"{source_name} is missing required columns: {sorted(missing)}."
        )


def validate_source_dataframe(df: pd.DataFrame) -> None:
    require_columns(df, set(INPUT_COLUMNS), "relational feature source")
    if df["TransactionID"].isna().any():
        raise ValueError("Relational feature source contains missing TransactionID values.")
    if not df["TransactionID"].is_unique:
        raise ValueError("Relational feature source contains duplicate TransactionID values.")
    if df["TransactionDT"].isna().any():
        raise ValueError("Relational feature source contains missing TransactionDT values.")
    if not is_numeric_dtype(df["TransactionDT"].dtype):
        raise TypeError("TransactionDT must be numeric.")
    transaction_times = df["TransactionDT"].to_numpy(dtype=np.float64, copy=False)
    if not np.isfinite(transaction_times).all():
        raise ValueError("TransactionDT contains non-finite values.")
    if df["split"].isna().any():
        raise ValueError("Relational feature source contains missing split assignments.")


def _scan_sorted_valid_groups(
    valid_rows: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Scan sorted groups using strict timestamp blocks and moving windows."""

    n_valid = len(valid_rows)
    prior_count = np.zeros(n_valid, dtype=np.int64)
    prior_count_24h = np.zeros(n_valid, dtype=np.int64)
    prior_count_7d = np.zeros(n_valid, dtype=np.int64)
    recency_hours = np.full(n_valid, np.nan, dtype=np.float64)
    if n_valid == 0:
        return prior_count, prior_count_24h, prior_count_7d, recency_hours

    times = valid_rows["TransactionDT"].to_numpy(copy=False)
    group_sizes = (
        valid_rows.groupby(
            GROUP_COLUMNS,
            sort=False,
            observed=True,
        )
        .size()
        .to_numpy(dtype=np.int64, copy=False)
    )

    group_start = 0
    for group_size_value in group_sizes:
        group_size = int(group_size_value)
        group_end = group_start + group_size
        group_times = times[group_start:group_end]
        left_24h = 0
        left_7d = 0
        block_start = 0

        while block_start < group_size:
            current_time = group_times[block_start]
            block_end = block_start + 1
            while (
                block_end < group_size
                and group_times[block_end] == current_time
            ):
                block_end += 1

            lower_24h = current_time - WINDOW_24H_SECONDS
            while (
                left_24h < block_start
                and group_times[left_24h] < lower_24h
            ):
                left_24h += 1

            lower_7d = current_time - WINDOW_7D_SECONDS
            while (
                left_7d < block_start
                and group_times[left_7d] < lower_7d
            ):
                left_7d += 1

            output_slice = slice(
                group_start + block_start,
                group_start + block_end,
            )
            prior_count[output_slice] = block_start
            prior_count_24h[output_slice] = block_start - left_24h
            prior_count_7d[output_slice] = block_start - left_7d
            if block_start > 0:
                recency_hours[output_slice] = (
                    current_time - group_times[block_start - 1]
                ) / 3_600.0

            block_start = block_end

        group_start = group_end

    if group_start != n_valid:
        raise AssertionError("Grouped relational scan did not consume every valid row.")
    return prior_count, prior_count_24h, prior_count_7d, recency_hours


def build_relational_features(source_df: pd.DataFrame) -> pd.DataFrame:
    """Build four label-free, strictly historical card_core_addr1 features."""

    validate_source_dataframe(source_df)
    row_count = len(source_df)
    valid_mask = source_df[GROUP_COLUMNS].notna().all(axis=1).to_numpy()
    valid_positions = np.flatnonzero(valid_mask)

    output_prior = np.zeros(row_count, dtype=np.int64)
    output_24h = np.zeros(row_count, dtype=np.int64)
    output_7d = np.zeros(row_count, dtype=np.int64)
    output_recency = np.full(row_count, np.nan, dtype=np.float64)

    if len(valid_positions):
        valid_rows = source_df.iloc[valid_positions][
            [*GROUP_COLUMNS, "TransactionDT"]
        ].copy()
        valid_rows["_original_position"] = valid_positions
        valid_rows = valid_rows.sort_values(
            [*GROUP_COLUMNS, "TransactionDT", "_original_position"],
            kind="mergesort",
        ).reset_index(drop=True)

        (
            sorted_prior,
            sorted_24h,
            sorted_7d,
            sorted_recency,
        ) = _scan_sorted_valid_groups(valid_rows)
        original_positions = valid_rows["_original_position"].to_numpy(
            dtype=np.int64,
            copy=False,
        )
        output_prior[original_positions] = sorted_prior
        output_24h[original_positions] = sorted_24h
        output_7d[original_positions] = sorted_7d
        output_recency[original_positions] = sorted_recency

    output = pd.DataFrame(
        {
            "TransactionID": source_df["TransactionID"].to_numpy(copy=True),
            RELATIONAL_FEATURES[0]: output_prior,
            RELATIONAL_FEATURES[1]: output_24h,
            RELATIONAL_FEATURES[2]: output_7d,
            RELATIONAL_FEATURES[3]: output_recency,
        }
    )
    validate_feature_output(source_df, output)
    return output


def validate_feature_output(
    source_df: pd.DataFrame,
    feature_df: pd.DataFrame,
) -> None:
    if list(feature_df.columns) != OUTPUT_COLUMNS:
        raise ValueError(
            "Relational output columns must be exactly "
            f"{OUTPUT_COLUMNS}; got {list(feature_df.columns)}."
        )
    if len(feature_df) != len(source_df):
        raise ValueError("Relational feature generation changed the row count.")
    if feature_df["TransactionID"].isna().any():
        raise ValueError("Relational output contains missing TransactionID values.")
    if not feature_df["TransactionID"].is_unique:
        raise ValueError("Relational output contains duplicate TransactionID values.")
    if not np.array_equal(
        feature_df["TransactionID"].to_numpy(),
        source_df["TransactionID"].to_numpy(),
    ):
        raise ValueError("Relational output did not preserve TransactionID order.")
    for column in RELATIONAL_FEATURES[:3]:
        if not is_integer_dtype(feature_df[column].dtype):
            raise TypeError(f"{column} must use an integer dtype.")
        if feature_df[column].lt(0).any():
            raise ValueError(f"{column} contains negative counts.")
    recency_column = RELATIONAL_FEATURES[3]
    if not is_float_dtype(feature_df[recency_column].dtype):
        raise TypeError(f"{recency_column} must use a floating dtype.")
    finite_recencies = feature_df[recency_column].dropna()
    if finite_recencies.lt(0).any():
        raise ValueError(f"{recency_column} contains negative values.")


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(resolved)


def build_metadata(
    source_df: pd.DataFrame,
    input_path: Path = INPUT_PATH,
    output_path: Path = OUTPUT_PATH,
) -> dict[str, Any]:
    valid_group_mask = source_df[GROUP_COLUMNS].notna().all(axis=1)
    valid_group_rows = int(valid_group_mask.sum())
    return {
        "relation_name": RELATION_NAME,
        "group_columns": GROUP_COLUMNS,
        "feature_names": RELATIONAL_FEATURES,
        "strict_temporal_rule": (
            "TransactionDT_previous < TransactionDT_current"
        ),
        "window_24h_seconds": WINDOW_24H_SECONDS,
        "window_7d_seconds": WINDOW_7D_SECONDS,
        "missing_group_policy": (
            "If any group component is missing, counts are 0 and recency is NaN."
        ),
        "equal_timestamp_policy": (
            "Rows at the same timestamp do not observe one another; each sees only "
            "strictly earlier timestamp blocks."
        ),
        "history_across_splits": (
            "History is continuous across train, validation, and test; it is never "
            "reset at split boundaries."
        ),
        "input_path": _display_path(input_path),
        "output_path": _display_path(output_path),
        "row_count": int(len(source_df)),
        "unique_transaction_count": int(source_df["TransactionID"].nunique()),
        "valid_group_row_count": valid_group_rows,
        "invalid_group_row_count": int(len(source_df) - valid_group_rows),
        "target_labels_used": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def main() -> None:
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Model dataset not found: {INPUT_PATH}. Build it before B1 features."
        )
    print(f"Loading required relational columns: {INPUT_PATH}")
    source_df = pd.read_parquet(INPUT_PATH, columns=INPUT_COLUMNS)
    if len(source_df) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS:,} source rows; got {len(source_df):,}."
        )

    print("Building strict historical card_core_addr1 features...")
    feature_df = build_relational_features(source_df)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(
        OUTPUT_PATH,
        index=False,
        engine="pyarrow",
        compression="snappy",
    )
    metadata = build_metadata(source_df)
    write_json(METADATA_PATH, metadata)

    saved = pd.read_parquet(OUTPUT_PATH)
    validate_feature_output(source_df, saved)
    if len(saved) != EXPECTED_ROWS:
        raise AssertionError("Saved relational feature row count changed.")

    print(f"Rows: {len(feature_df):,}")
    print(f"Valid relationship rows: {metadata['valid_group_row_count']:,}")
    print(f"Invalid relationship rows: {metadata['invalid_group_row_count']:,}")
    print(f"Features saved: {OUTPUT_PATH}")
    print(f"Metadata saved: {METADATA_PATH}")
    print("Target labels used: NO")


if __name__ == "__main__":
    main()
