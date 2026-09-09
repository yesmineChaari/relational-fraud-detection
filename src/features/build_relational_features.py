from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype, is_numeric_dtype

ROOT_DIR = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT_DIR / "data" / "processed" / "model_dataset.parquet"

# ---------------------------------------------------------------------------
# Canonical relation registry
# ---------------------------------------------------------------------------

RELATION_REGISTRY: dict[str, list[str]] = {
    "card_core_addr1": ["card1", "card2", "card3", "card5", "addr1"],
    "card1": ["card1"],
    "card1_card2": ["card1", "card2"],
}

# ---------------------------------------------------------------------------
# Default (backward-compatible) constants — kept for existing imports
# ---------------------------------------------------------------------------

RELATION_NAME = "card_core_addr1"
GROUP_COLUMNS = RELATION_REGISTRY[RELATION_NAME]
RELATIONAL_FEATURES = [
    "card_core_addr1_prior_count",
    "card_core_addr1_prior_count_24h",
    "card_core_addr1_prior_count_7d",
    "card_core_addr1_time_since_previous_hours",
]
OUTPUT_PATH = ROOT_DIR / "data" / "processed" / "relational_features_card_core_addr1.parquet"
METADATA_PATH = ROOT_DIR / "reports" / "relational_features" / "card_core_addr1_metadata.json"
OUTPUT_COLUMNS = ["TransactionID", *RELATIONAL_FEATURES]
INPUT_COLUMNS = ["TransactionID", "TransactionDT", "split", *GROUP_COLUMNS]

EXPECTED_ROWS = 590_540
WINDOW_24H_SECONDS = 86_400
WINDOW_7D_SECONDS = 604_800


# ---------------------------------------------------------------------------
# Helpers: relation-specific path / name derivation
# ---------------------------------------------------------------------------


def _feature_names(relation: str) -> list[str]:
    return [
        f"{relation}_prior_count",
        f"{relation}_prior_count_24h",
        f"{relation}_prior_count_7d",
        f"{relation}_time_since_previous_hours",
    ]


def _output_path(relation: str) -> Path:
    return ROOT_DIR / "data" / "processed" / f"relational_features_{relation}.parquet"


def _metadata_path(relation: str) -> Path:
    return ROOT_DIR / "reports" / "relational_features" / f"{relation}_metadata.json"


def _input_columns(group_columns: list[str]) -> list[str]:
    return ["TransactionID", "TransactionDT", "split", *group_columns]


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def require_columns(df: pd.DataFrame, required: set[str], source_name: str) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{source_name} is missing required columns: {sorted(missing)}.")


def validate_source_dataframe(df: pd.DataFrame, group_columns: list[str]) -> None:
    input_cols = set(_input_columns(group_columns))
    require_columns(df, input_cols, "relational feature source")
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


def validate_feature_output(
    source_df: pd.DataFrame,
    feature_df: pd.DataFrame,
    relation: str,
) -> None:
    feat_names = _feature_names(relation)
    expected_columns = ["TransactionID", *feat_names]
    if list(feature_df.columns) != expected_columns:
        raise ValueError(
            f"Relational output columns must be exactly {expected_columns}; "
            f"got {list(feature_df.columns)}."
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
    for column in feat_names[:3]:
        if not is_integer_dtype(feature_df[column].dtype):
            raise TypeError(f"{column} must use an integer dtype.")
        if feature_df[column].lt(0).any():
            raise ValueError(f"{column} contains negative counts.")
    recency_column = feat_names[3]
    if not is_float_dtype(feature_df[recency_column].dtype):
        raise TypeError(f"{recency_column} must use a floating dtype.")
    finite_recencies = feature_df[recency_column].dropna()
    if finite_recencies.lt(0).any():
        raise ValueError(f"{recency_column} contains negative values.")


# ---------------------------------------------------------------------------
# Core scan (relation-agnostic)
# ---------------------------------------------------------------------------


def _scan_sorted_valid_groups(
    valid_rows: pd.DataFrame,
    group_columns: list[str],
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
        valid_rows.groupby(group_columns, sort=False, observed=True)
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
            while block_end < group_size and group_times[block_end] == current_time:
                block_end += 1

            lower_24h = current_time - WINDOW_24H_SECONDS
            while left_24h < block_start and group_times[left_24h] < lower_24h:
                left_24h += 1

            lower_7d = current_time - WINDOW_7D_SECONDS
            while left_7d < block_start and group_times[left_7d] < lower_7d:
                left_7d += 1

            output_slice = slice(group_start + block_start, group_start + block_end)
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


# ---------------------------------------------------------------------------
# Main feature-building function (relation-parameterised)
# ---------------------------------------------------------------------------


def compute_relational_features(
    source_df: pd.DataFrame,
    relation: str,
    group_columns: list[str],
) -> pd.DataFrame:
    """Compute the four historical summaries without validating the source.

    This is the single implementation of the relational temporal semantics --
    strictly-before timestamp blocks, inclusive-lower-bound 24h/7d windows, and
    a NaN recency for a first occurrence. The screening module reuses it so the
    diagnostic features can never drift from the ones B1 is trained on.

    Rows whose grouping components are missing (or whose grouping columns are
    absent entirely) get counts of 0 and a NaN recency.
    """
    feat_names = _feature_names(relation)
    row_count = len(source_df)

    available = [c for c in group_columns if c in source_df.columns]
    if len(available) < len(group_columns):
        valid_mask = np.zeros(row_count, dtype=bool)
    else:
        valid_mask = source_df[group_columns].notna().all(axis=1).to_numpy()
    valid_positions = np.flatnonzero(valid_mask)

    output_prior = np.zeros(row_count, dtype=np.int64)
    output_24h = np.zeros(row_count, dtype=np.int64)
    output_7d = np.zeros(row_count, dtype=np.int64)
    output_recency = np.full(row_count, np.nan, dtype=np.float64)

    if len(valid_positions):
        valid_rows = source_df.iloc[valid_positions][[*group_columns, "TransactionDT"]].copy()
        valid_rows["_original_position"] = valid_positions
        valid_rows = valid_rows.sort_values(
            [*group_columns, "TransactionDT", "_original_position"],
            kind="mergesort",
        ).reset_index(drop=True)

        sorted_prior, sorted_24h, sorted_7d, sorted_recency = _scan_sorted_valid_groups(
            valid_rows, group_columns
        )
        original_positions = valid_rows["_original_position"].to_numpy(dtype=np.int64, copy=False)
        output_prior[original_positions] = sorted_prior
        output_24h[original_positions] = sorted_24h
        output_7d[original_positions] = sorted_7d
        output_recency[original_positions] = sorted_recency

    return pd.DataFrame(
        {
            "TransactionID": source_df["TransactionID"].to_numpy(copy=True),
            feat_names[0]: output_prior,
            feat_names[1]: output_24h,
            feat_names[2]: output_7d,
            feat_names[3]: output_recency,
        }
    )


def build_relational_features_for(
    source_df: pd.DataFrame,
    relation: str,
    group_columns: list[str],
) -> pd.DataFrame:
    """Build four label-free, strictly historical features for any relation."""
    validate_source_dataframe(source_df, group_columns)
    output = compute_relational_features(source_df, relation, group_columns)
    validate_feature_output(source_df, output, relation)
    return output


def build_relational_features(source_df: pd.DataFrame) -> pd.DataFrame:
    """Backward-compatible wrapper — always uses card_core_addr1."""
    return build_relational_features_for(source_df, RELATION_NAME, GROUP_COLUMNS)


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _display_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT_DIR).as_posix()
    except ValueError:
        return str(resolved)


def build_metadata(
    source_df: pd.DataFrame,
    relation: str,
    group_columns: list[str],
    input_path: Path = INPUT_PATH,
    output_path: Path | None = None,
) -> dict[str, Any]:
    if output_path is None:
        output_path = _output_path(relation)
    feat_names = _feature_names(relation)
    valid_group_mask = source_df[group_columns].notna().all(axis=1)
    valid_group_rows = int(valid_group_mask.sum())
    return {
        "relation_name": relation,
        "group_columns": group_columns,
        "feature_names": feat_names,
        "strict_temporal_rule": "TransactionDT_previous < TransactionDT_current",
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


# ---------------------------------------------------------------------------
# Public API used by AGENTS.md Step 1
# ---------------------------------------------------------------------------


def build_and_save_relational_features(relation_name: str) -> None:
    """Generate, validate, and persist features for the given relation."""
    if relation_name not in RELATION_REGISTRY:
        raise ValueError(
            f"Unknown relation: {relation_name!r}. Supported: {sorted(RELATION_REGISTRY)}."
        )
    group_columns = RELATION_REGISTRY[relation_name]
    output_path = _output_path(relation_name)
    metadata_out = _metadata_path(relation_name)
    load_cols = _input_columns(group_columns)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Model dataset not found: {INPUT_PATH}. Build it before relational features."
        )
    print(f"[{relation_name}] Loading columns from: {INPUT_PATH}")
    source_df = pd.read_parquet(INPUT_PATH, columns=load_cols)
    if len(source_df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} source rows; got {len(source_df):,}.")

    print(f"[{relation_name}] Building features (group_columns={group_columns})...")
    feature_df = build_relational_features_for(source_df, relation_name, group_columns)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_parquet(output_path, index=False, engine="pyarrow", compression="snappy")

    metadata = build_metadata(source_df, relation_name, group_columns, output_path=output_path)
    write_json(metadata_out, metadata)

    # Round-trip validation
    saved = pd.read_parquet(output_path)
    validate_feature_output(source_df, saved, relation_name)
    if len(saved) != EXPECTED_ROWS:
        raise AssertionError("Saved relational feature row count changed.")

    print(f"[{relation_name}] Rows: {len(feature_df):,}")
    print(f"[{relation_name}] Valid group rows: {metadata['valid_group_row_count']:,}")
    print(f"[{relation_name}] Invalid group rows: {metadata['invalid_group_row_count']:,}")
    print(f"[{relation_name}] Features saved: {output_path}")
    print(f"[{relation_name}] Metadata saved: {metadata_out}")
    print(f"[{relation_name}] Target labels used: NO")


# ---------------------------------------------------------------------------
# CLI entry point (backward-compatible default: card_core_addr1)
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build relational historical features for a given entity relation."
    )
    parser.add_argument(
        "--relation",
        default="card_core_addr1",
        choices=sorted(RELATION_REGISTRY),
        help="Named relation to generate features for (default: card_core_addr1).",
    )
    args = parser.parse_args()
    build_and_save_relational_features(args.relation)


if __name__ == "__main__":
    main()
