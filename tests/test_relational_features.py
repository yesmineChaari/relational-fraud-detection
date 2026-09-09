from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
from pandas.api.types import is_float_dtype, is_integer_dtype

from src.features.build_relational_features import (
    GROUP_COLUMNS,
    OUTPUT_COLUMNS,
    RELATIONAL_FEATURES,
    build_relational_features,
)

PRIOR = RELATIONAL_FEATURES[0]
PRIOR_24H = RELATIONAL_FEATURES[1]
PRIOR_7D = RELATIONAL_FEATURES[2]
RECENCY = RELATIONAL_FEATURES[3]


def make_source(
    times: list[int],
    groups: list[int] | None = None,
    splits: list[str] | None = None,
) -> pd.DataFrame:
    row_count = len(times)
    group_values = groups if groups is not None else [1] * row_count
    split_values = splits if splits is not None else ["train"] * row_count
    if len(group_values) != row_count or len(split_values) != row_count:
        raise ValueError("Synthetic fixture lengths do not match.")
    return pd.DataFrame(
        {
            "TransactionID": np.arange(10_000, 10_000 + row_count),
            "TransactionDT": times,
            "split": split_values,
            "card1": [1_000 + group for group in group_values],
            "card2": [200 + group for group in group_values],
            "card3": [300 + group for group in group_values],
            "card5": [500 + group for group in group_values],
            "addr1": [700 + group for group in group_values],
        }
    )


class RelationalFeatureSemanticsTests(unittest.TestCase):
    def test_first_transaction_has_empty_history(self) -> None:
        result = build_relational_features(make_source([100]))

        self.assertEqual(int(result.loc[0, PRIOR]), 0)
        self.assertEqual(int(result.loc[0, PRIOR_24H]), 0)
        self.assertEqual(int(result.loc[0, PRIOR_7D]), 0)
        self.assertTrue(np.isnan(result.loc[0, RECENCY]))

    def test_normal_strict_history(self) -> None:
        result = build_relational_features(make_source([100, 200, 300]))

        self.assertEqual(result[PRIOR].tolist(), [0, 1, 2])

    def test_equal_timestamps_see_only_strictly_earlier_rows(self) -> None:
        result = build_relational_features(make_source([100, 200, 200, 200]))

        self.assertEqual(result[PRIOR].tolist(), [0, 1, 1, 1])
        self.assertEqual(result[PRIOR_24H].tolist(), [0, 1, 1, 1])
        self.assertEqual(result[PRIOR_7D].tolist(), [0, 1, 1, 1])

    def test_deep_history_is_not_hidden_by_a_tie_block(self) -> None:
        times = [100, *([200] * 10), 300]
        result = build_relational_features(make_source(times))

        self.assertEqual(result.loc[1:10, PRIOR].tolist(), [1] * 10)
        self.assertEqual(int(result.loc[11, PRIOR]), 11)
        self.assertAlmostEqual(float(result.loc[11, RECENCY]), 100 / 3_600)

    def test_24_hour_lower_boundary_is_inclusive(self) -> None:
        source = make_source(
            [0, 86_400, 0, 86_401],
            groups=[1, 1, 2, 2],
        )
        result = build_relational_features(source)

        self.assertEqual(int(result.loc[1, PRIOR_24H]), 1)
        self.assertEqual(int(result.loc[3, PRIOR_24H]), 0)

    def test_7_day_lower_boundary_is_inclusive(self) -> None:
        source = make_source(
            [0, 604_800, 0, 604_801],
            groups=[1, 1, 2, 2],
        )
        result = build_relational_features(source)

        self.assertEqual(int(result.loc[1, PRIOR_7D]), 1)
        self.assertEqual(int(result.loc[3, PRIOR_7D]), 0)

    def test_recency_uses_latest_strictly_earlier_timestamp(self) -> None:
        result = build_relational_features(make_source([100, 200, 200, 500]))

        self.assertAlmostEqual(float(result.loc[1, RECENCY]), 100 / 3_600)
        self.assertAlmostEqual(float(result.loc[2, RECENCY]), 100 / 3_600)
        self.assertAlmostEqual(float(result.loc[3, RECENCY]), 300 / 3_600)

    def test_groups_are_isolated(self) -> None:
        source = make_source([100, 150, 200, 250], groups=[1, 2, 1, 2])
        result = build_relational_features(source)

        self.assertEqual(result[PRIOR].tolist(), [0, 0, 1, 1])

    def test_any_missing_component_produces_empty_history(self) -> None:
        source = make_source([100, 200, 300, 400, 500, 600])
        for row_index, column in enumerate(GROUP_COLUMNS):
            source.loc[row_index + 1, column] = np.nan
        result = build_relational_features(source)

        self.assertEqual(result.loc[1:, PRIOR].tolist(), [0] * 5)
        self.assertEqual(result.loc[1:, PRIOR_24H].tolist(), [0] * 5)
        self.assertEqual(result.loc[1:, PRIOR_7D].tolist(), [0] * 5)
        self.assertTrue(result.loc[1:, RECENCY].isna().all())

    def test_history_continues_across_split_boundary(self) -> None:
        source = make_source(
            [100, 200],
            splits=["train", "validation"],
        )
        result = build_relational_features(source)

        self.assertEqual(int(result.loc[1, PRIOR]), 1)
        self.assertAlmostEqual(float(result.loc[1, RECENCY]), 100 / 3_600)

    def test_adding_future_rows_does_not_change_past_features(self) -> None:
        past = make_source([100, 200, 200, 300])
        extended = make_source([100, 200, 200, 300, 400, 500])

        past_result = build_relational_features(past)
        extended_result = build_relational_features(extended).iloc[: len(past)]

        pd.testing.assert_frame_equal(
            past_result.reset_index(drop=True),
            extended_result.reset_index(drop=True),
        )

    def test_feature_construction_does_not_require_labels(self) -> None:
        source = make_source([100, 200])
        self.assertNotIn("isFraud", source.columns)

        result = build_relational_features(source)

        self.assertEqual(result[PRIOR].tolist(), [0, 1])

    def test_output_integrity_and_dtypes(self) -> None:
        source = make_source([300, 100, 200], groups=[1, 2, 1])
        result = build_relational_features(source)

        self.assertEqual(list(result.columns), OUTPUT_COLUMNS)
        self.assertEqual(len(result), len(source))
        self.assertTrue(result["TransactionID"].is_unique)
        self.assertEqual(
            result["TransactionID"].tolist(),
            source["TransactionID"].tolist(),
        )
        for column in (PRIOR, PRIOR_24H, PRIOR_7D):
            self.assertTrue(is_integer_dtype(result[column].dtype))
        self.assertTrue(is_float_dtype(result[RECENCY].dtype))

    def test_vectorized_scan_matches_strict_brute_force(self) -> None:
        rng = np.random.default_rng(42)
        row_count = 80
        source = make_source(
            rng.choice([0, 100, 86_400, 86_500, 604_800, 700_000], row_count).tolist(),
            groups=rng.integers(1, 6, row_count).tolist(),
            splits=rng.choice(["train", "validation", "test"], row_count).tolist(),
        )
        actual = build_relational_features(source)

        for row_index, row in source.iterrows():
            same_group = source[GROUP_COLUMNS].eq(row[GROUP_COLUMNS]).all(axis=1)
            earlier = source.loc[
                same_group & source["TransactionDT"].lt(row["TransactionDT"]),
                "TransactionDT",
            ]
            current_time = int(row["TransactionDT"])
            self.assertEqual(int(actual.loc[row_index, PRIOR]), len(earlier))
            self.assertEqual(
                int(actual.loc[row_index, PRIOR_24H]),
                int(earlier.ge(current_time - 86_400).sum()),
            )
            self.assertEqual(
                int(actual.loc[row_index, PRIOR_7D]),
                int(earlier.ge(current_time - 604_800).sum()),
            )
            if earlier.empty:
                self.assertTrue(np.isnan(actual.loc[row_index, RECENCY]))
            else:
                self.assertAlmostEqual(
                    float(actual.loc[row_index, RECENCY]),
                    (current_time - int(earlier.max())) / 3_600,
                )


if __name__ == "__main__":
    unittest.main()
