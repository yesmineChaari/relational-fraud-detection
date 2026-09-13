"""Derived grouping-key columns, defined once for every relation consumer.

Every other relation groups on raw columns. The user-ID key needs one column
the raw data does not hold: the day the card's history began. If each consumer
derived it, the audit, the screening, the feature builder and the graph builder
could disagree about what a user is, so the derivation is defined here only.

``account_start_day = floor(TransactionDT / 86400) - D1``. ``TransactionDT`` is
seconds from a fixed reference, so its floor division is the transaction's day
index; ``D1`` is the number of days since the card was first used. Their
difference stays the same for every transaction of one card history. Both
inputs are known at transaction time and ``isFraud`` is never read, so the
column is label-free and uses no future information.

``uid = card1 | addr1 | account_start_day``. A row missing any component has no
uid, as for every other relation.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import numpy as np
import pandas as pd

ACCOUNT_START_DAY = "account_start_day"
SECONDS_PER_DAY = 86_400

# The raw inputs each derived column is computed from. A consumer that reads
# raw columns expands its group definition through this map before loading.
DERIVED_COLUMNS: dict[str, tuple[str, ...]] = {
    ACCOUNT_START_DAY: ("TransactionDT", "D1"),
}
DERIVED_COLUMN_DEFINITIONS: dict[str, str] = {
    ACCOUNT_START_DAY: "floor(TransactionDT / 86400) - D1",
}

UID_RELATION = "uid"
UID_COLUMNS: tuple[str, ...] = ("card1", "addr1", ACCOUNT_START_DAY)

LABEL_COLUMN = "isFraud"
if any(LABEL_COLUMN in inputs for inputs in DERIVED_COLUMNS.values()):
    raise AssertionError("A derived key column must never be computed from the label.")


def add_account_start_day(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with ``account_start_day`` added.

    Only the declared raw inputs are read, so the label cannot influence the
    result even when ``df`` carries it. The value is NaN where ``D1`` is missing.
    """
    inputs = DERIVED_COLUMNS[ACCOUNT_START_DAY]
    missing = [column for column in inputs if column not in df.columns]
    if missing:
        raise ValueError(f"{ACCOUNT_START_DAY} needs columns {missing}, which are absent.")

    transaction_dt = df["TransactionDT"].to_numpy(dtype=np.float64, na_value=np.nan)
    days_since_first_use = df["D1"].to_numpy(dtype=np.float64, na_value=np.nan)
    if np.isnan(transaction_dt).any():
        raise ValueError(f"{ACCOUNT_START_DAY} cannot be derived where TransactionDT is missing.")
    observed = days_since_first_use[~np.isnan(days_since_first_use)]
    if not np.array_equal(observed, np.floor(observed)):
        raise ValueError(
            "D1 must be a whole number of days; a fractional value would split one "
            "card history across several keys."
        )

    start_day = np.floor(transaction_dt / SECONDS_PER_DAY) - days_since_first_use
    return df.assign(**{ACCOUNT_START_DAY: start_day})


_BUILDERS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    ACCOUNT_START_DAY: add_account_start_day,
}
if set(_BUILDERS) != set(DERIVED_COLUMNS):
    raise AssertionError("Every derived column needs exactly one builder.")


def source_columns(columns: Iterable[str]) -> list[str]:
    """``columns`` with each derived column replaced by its raw inputs.

    Order is preserved and duplicates are dropped, so the result can be passed
    straight to ``pd.read_parquet(columns=...)``.
    """
    expanded: list[str] = []
    for column in columns:
        expanded.extend(DERIVED_COLUMNS.get(column, (column,)))
    return list(dict.fromkeys(expanded))


def add_derived_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Compute every derived column named in ``columns``.

    A derived column already present in ``df`` is recomputed, so a stale copy
    can never disagree with the definition. Raw columns are left untouched.
    """
    for column in dict.fromkeys(columns):
        if column in _BUILDERS:
            df = _BUILDERS[column](df)
    return df
