"""Tests for the shared user-ID key definition in src/features/derived_keys.py."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import src.features.build_relational_features as brf
from src.features import screen_relations
from src.features.build_relational_features import (
    RELATION_REGISTRY,
    _feature_names,
    build_relational_features_for,
)
from src.features.derived_keys import (
    ACCOUNT_START_DAY,
    DERIVED_COLUMN_DEFINITIONS,
    DERIVED_COLUMNS,
    LABEL_COLUMN,
    SECONDS_PER_DAY,
    UID_COLUMNS,
    UID_RELATION,
    add_account_start_day,
    add_derived_columns,
    source_columns,
)
from src.graph.analyze_relations import CANDIDATES, build_strict_proxy

DAY = SECONDS_PER_DAY
PRIOR, PRIOR_24H, PRIOR_7D, RECENCY = _feature_names(UID_RELATION)


def _raw(
    times: list[int],
    d1: list[float],
    card1: list[int] | None = None,
    addr1: list[float] | None = None,
    splits: list[str] | None = None,
) -> pd.DataFrame:
    n = len(times)
    return pd.DataFrame(
        {
            "TransactionID": np.arange(1, n + 1),
            "TransactionDT": times,
            "split": splits or ["train"] * n,
            "card1": card1 if card1 is not None else [1_000] * n,
            "addr1": addr1 if addr1 is not None else [300.0] * n,
            "D1": np.asarray(d1, dtype=np.float64),
        }
    )


def _hand_fixture() -> pd.DataFrame:
    """Six rows sharing card1 and (mostly) addr1, split by account start day.

    Row 0: day 5, D1 0 -> start 5 (user A)
    Row 1: day 6, D1 1 -> start 5 (user A), exactly 24h after row 0
    Row 2: day 6, D1 0 -> start 6 (user B: same card1|addr1, another history)
    Row 3: day 12, D1 7 -> start 5 (user A), exactly 7 days after row 0
    Row 4: addr1 missing -> no uid
    Row 5: D1 missing -> no uid
    """
    return _raw(
        times=[
            5 * DAY + 100,
            6 * DAY + 100,
            6 * DAY + 200,
            12 * DAY + 100,
            12 * DAY + 200,
            12 * DAY + 300,
        ],
        d1=[0, 1, 0, 7, 7, np.nan],
        addr1=[300.0, 300.0, 300.0, 300.0, np.nan, 300.0],
    )


# ===========================================================================
# The derivation
# ===========================================================================


class TestAccountStartDay:
    def test_start_day_is_the_transaction_day_minus_d1(self):
        df = _raw(times=[10 * DAY, 10 * DAY + DAY - 1, 11 * DAY], d1=[3, 3, 4])
        result = add_account_start_day(df)
        assert result[ACCOUNT_START_DAY].tolist() == [7.0, 7.0, 7.0]

    def test_day_index_floors_at_the_day_boundary(self):
        df = _raw(times=[10 * DAY - 1, 10 * DAY], d1=[0, 0])
        result = add_account_start_day(df)
        assert result[ACCOUNT_START_DAY].tolist() == [9.0, 10.0]

    def test_missing_d1_gives_a_missing_start_day(self):
        df = _raw(times=[10 * DAY, 10 * DAY], d1=[np.nan, 2])
        result = add_account_start_day(df)
        assert np.isnan(result.loc[0, ACCOUNT_START_DAY])
        assert result.loc[1, ACCOUNT_START_DAY] == 8.0

    def test_nullable_d1_is_read_as_missing(self):
        df = _raw(times=[10 * DAY, 10 * DAY], d1=[0, 0])
        df["D1"] = pd.array([1, None], dtype="Float64")
        result = add_account_start_day(df)
        assert result.loc[0, ACCOUNT_START_DAY] == 9.0
        assert np.isnan(result.loc[1, ACCOUNT_START_DAY])

    def test_values_are_whole_days(self):
        rng = np.random.default_rng(7)
        d1 = rng.integers(0, 641, 500).astype(np.float64)
        d1[rng.random(500) < 0.1] = np.nan
        df = _raw(times=rng.integers(DAY, 183 * DAY, 500).tolist(), d1=d1.tolist())
        values = add_account_start_day(df)[ACCOUNT_START_DAY].to_numpy()
        observed = values[~np.isnan(values)]
        assert np.array_equal(observed, np.floor(observed))
        assert np.array_equal(np.isnan(values), np.isnan(d1))

    def test_fractional_d1_is_refused(self):
        with pytest.raises(ValueError, match="whole number"):
            add_account_start_day(_raw(times=[10 * DAY], d1=[1.5]))

    def test_missing_input_column_is_refused(self):
        with pytest.raises(ValueError, match="D1"):
            add_account_start_day(_raw(times=[10 * DAY], d1=[1]).drop(columns="D1"))

    def test_the_label_is_never_read(self):
        base = _raw(times=[10 * DAY, 11 * DAY, 12 * DAY], d1=[0, 1, np.nan])
        with_label = base.assign(**{LABEL_COLUMN: [0, 1, 0]})
        flipped = base.assign(**{LABEL_COLUMN: [1, 0, 1]})

        expected = add_account_start_day(base)[ACCOUNT_START_DAY]
        for frame in (with_label, flipped):
            pd.testing.assert_series_equal(
                add_account_start_day(frame)[ACCOUNT_START_DAY], expected
            )
        for inputs in DERIVED_COLUMNS.values():
            assert LABEL_COLUMN not in inputs

    def test_the_input_frame_is_not_mutated(self):
        df = _raw(times=[10 * DAY], d1=[1])
        add_account_start_day(df)
        assert ACCOUNT_START_DAY not in df.columns


class TestDerivedColumnHelpers:
    def test_source_columns_expands_the_uid_to_raw_inputs(self):
        columns = ["TransactionID", "TransactionDT", "split", *UID_COLUMNS]
        assert source_columns(columns) == [
            "TransactionID",
            "TransactionDT",
            "split",
            "card1",
            "addr1",
            "D1",
        ]

    def test_source_columns_leaves_raw_only_definitions_unchanged(self):
        for columns in RELATION_REGISTRY.values():
            if not set(columns) & set(DERIVED_COLUMNS):
                assert source_columns(columns) == list(columns)

    def test_derived_inputs_are_raw_columns(self):
        for inputs in DERIVED_COLUMNS.values():
            assert not set(inputs) & set(DERIVED_COLUMNS)
        assert set(DERIVED_COLUMN_DEFINITIONS) == set(DERIVED_COLUMNS)

    def test_add_derived_columns_ignores_raw_only_definitions(self):
        df = _raw(times=[10 * DAY], d1=[1])
        result = add_derived_columns(df, ["card1", "addr1"])
        assert list(result.columns) == list(df.columns)

    def test_add_derived_columns_recomputes_a_stale_copy(self):
        df = _raw(times=[10 * DAY], d1=[1]).assign(**{ACCOUNT_START_DAY: [999.0]})
        result = add_derived_columns(df, UID_COLUMNS)
        assert result[ACCOUNT_START_DAY].tolist() == [9.0]


# ===========================================================================
# One definition across every registry
# ===========================================================================


class TestUidRegistries:
    def test_every_registry_defines_the_uid_identically(self):
        expected = list(UID_COLUMNS)
        assert RELATION_REGISTRY[UID_RELATION] == expected
        assert CANDIDATES[UID_RELATION] == expected
        assert screen_relations.CANDIDATES[UID_RELATION] == expected

    def test_the_audit_proxy_groups_rows_as_the_feature_builder_does(self):
        derived = add_account_start_day(_hand_fixture())
        proxy = build_strict_proxy(derived, CANDIDATES[UID_RELATION])

        assert proxy.isna().tolist() == [False, False, False, False, True, True]
        assert proxy[0] == proxy[1] == proxy[3]
        assert proxy[2] != proxy[0]


# ===========================================================================
# uid features through the unchanged relational kernel
# ===========================================================================


class TestUidFeatures:
    def _build(self, raw: pd.DataFrame) -> pd.DataFrame:
        derived = add_derived_columns(raw, UID_COLUMNS)
        return build_relational_features_for(derived, UID_RELATION, RELATION_REGISTRY[UID_RELATION])

    def test_features_match_a_hand_computation(self):
        result = self._build(_hand_fixture())

        assert result[PRIOR].tolist() == [0, 1, 0, 2, 0, 0]
        # Row 1 is exactly 24h after row 0 and row 3 exactly 7 days after it:
        # both lower bounds are inclusive, as for every other relation.
        assert result[PRIOR_24H].tolist() == [0, 1, 0, 0, 0, 0]
        assert result[PRIOR_7D].tolist() == [0, 1, 0, 2, 0, 0]
        recency = result[RECENCY].tolist()
        assert recency[1] == pytest.approx(24.0)
        assert recency[3] == pytest.approx(144.0)
        assert all(np.isnan(recency[i]) for i in (0, 2, 4, 5))

    def test_a_new_start_day_is_a_new_user(self):
        """Row 2 shares card1 and addr1 with rows 0-1 but not their history."""
        raw = _hand_fixture()
        uid = self._build(raw)
        card_addr = build_relational_features_for(raw, "card1_addr1", ["card1", "addr1"])

        assert uid.loc[2, PRIOR] == 0
        assert card_addr.loc[2, "card1_addr1_prior_count"] == 2

    def test_equal_timestamps_do_not_see_each_other(self):
        raw = _raw(times=[10 * DAY, 10 * DAY, 11 * DAY], d1=[2, 2, 3])
        assert self._build(raw)[PRIOR].tolist() == [0, 0, 2]

    def test_scan_matches_a_strict_brute_force(self):
        rng = np.random.default_rng(42)
        n = 120
        d1 = rng.integers(0, 3, n).astype(np.float64)
        d1[rng.random(n) < 0.1] = np.nan
        raw = _raw(
            times=rng.choice([DAY, DAY + 100, 2 * DAY, 2 * DAY + 100, 8 * DAY], n).tolist(),
            d1=d1.tolist(),
            card1=rng.integers(1, 3, n).tolist(),
            addr1=rng.choice([300.0, 301.0, np.nan], n).tolist(),
            splits=rng.choice(["train", "validation", "test"], n).tolist(),
        )
        derived = add_derived_columns(raw, UID_COLUMNS)
        actual = self._build(raw)
        keys = derived[list(UID_COLUMNS)]

        for i, row in derived.iterrows():
            if keys.loc[i].isna().any():
                assert actual.loc[i, PRIOR] == 0
                assert np.isnan(actual.loc[i, RECENCY])
                continue
            same_user = keys.eq(keys.loc[i]).all(axis=1)
            earlier = derived.loc[same_user & derived["TransactionDT"].lt(row["TransactionDT"])]
            now = int(row["TransactionDT"])
            assert actual.loc[i, PRIOR] == len(earlier)
            assert actual.loc[i, PRIOR_24H] == int(earlier["TransactionDT"].ge(now - DAY).sum())
            assert actual.loc[i, PRIOR_7D] == int(earlier["TransactionDT"].ge(now - 7 * DAY).sum())
            if earlier.empty:
                assert np.isnan(actual.loc[i, RECENCY])
            else:
                expected = (now - int(earlier["TransactionDT"].max())) / 3_600
                assert actual.loc[i, RECENCY] == pytest.approx(expected)


class TestFeatureBuilderEntryPoint:
    def test_the_builder_derives_the_key_from_raw_inputs(self, tmp_path, monkeypatch):
        raw = _hand_fixture()
        assert ACCOUNT_START_DAY not in raw.columns
        dataset = tmp_path / "model_dataset.parquet"
        raw.to_parquet(dataset, index=False)

        monkeypatch.setattr(brf, "INPUT_PATH", dataset)
        monkeypatch.setattr(brf, "EXPECTED_ROWS", len(raw))
        monkeypatch.setattr(
            brf,
            "_output_path",
            lambda relation: tmp_path / f"relational_features_{relation}.parquet",
        )
        monkeypatch.setattr(
            brf, "_metadata_path", lambda relation: tmp_path / f"{relation}_metadata.json"
        )

        brf.build_and_save_relational_features(UID_RELATION)

        saved = pd.read_parquet(tmp_path / "relational_features_uid.parquet")
        expected = build_relational_features_for(
            add_derived_columns(raw, UID_COLUMNS), UID_RELATION, list(UID_COLUMNS)
        )
        pd.testing.assert_frame_equal(saved, expected)

        metadata = json.loads((tmp_path / "uid_metadata.json").read_text(encoding="utf-8"))
        assert metadata["group_columns"] == list(UID_COLUMNS)
        assert metadata["derived_group_columns"] == {
            ACCOUNT_START_DAY: {
                "inputs": ["TransactionDT", "D1"],
                "definition": "floor(TransactionDT / 86400) - D1",
            }
        }
        assert metadata["valid_group_row_count"] == 4
        assert metadata["target_labels_used"] is False
