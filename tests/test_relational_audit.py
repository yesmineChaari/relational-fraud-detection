from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.graph.analyze_relations import build_temporal_edges


def decode_edges(edge_ids: np.ndarray, n_nodes: int) -> set[tuple[int, int]]:
    return {(int(edge_id // n_nodes), int(edge_id % n_nodes)) for edge_id in edge_ids}


class TemporalEdgeTests(unittest.TestCase):
    def test_ties_do_not_hide_a_strictly_earlier_neighbor(self) -> None:
        # T5's previous three rows are tied at time 200, but T1 at time 100
        # must still be selected because strict-time filtering comes first.
        df = pd.DataFrame(
            {
                "TransactionDT": [100, 200, 200, 200, 200],
                "node_id": [0, 1, 2, 3, 4],
            }
        )
        proxy = pd.Series(["entity"] * len(df), dtype="string")

        edges = decode_edges(build_temporal_edges(df, proxy, 3), len(df))

        self.assertEqual(edges, {(1, 0), (2, 0), (3, 0), (4, 0)})

    def test_last_k_is_applied_after_strict_time_filter(self) -> None:
        df = pd.DataFrame(
            {
                "TransactionDT": [100, 100, 150, 150, 200],
                "node_id": [0, 1, 2, 3, 4],
            }
        )
        proxy = pd.Series(["entity"] * len(df), dtype="string")

        edges = decode_edges(build_temporal_edges(df, proxy, 3), len(df))
        destinations_for_last = {destination for source, destination in edges if source == 4}

        self.assertEqual(destinations_for_last, {1, 2, 3})
        self.assertNotIn((4, 0), edges)

    def test_equal_timestamp_transactions_never_connect(self) -> None:
        df = pd.DataFrame(
            {
                "TransactionDT": [100, 100, 100, 100],
                "node_id": [0, 1, 2, 3],
            }
        )
        proxy = pd.Series(["entity"] * len(df), dtype="string")

        edges = build_temporal_edges(df, proxy, 3)

        self.assertEqual(len(edges), 0)

    def test_edges_never_cross_entity_groups(self) -> None:
        df = pd.DataFrame(
            {
                "TransactionDT": [100, 200, 100, 200],
                "node_id": [0, 1, 2, 3],
            }
        )
        proxy = pd.Series(["a", "a", "b", "b"], dtype="string")

        edges = decode_edges(build_temporal_edges(df, proxy, 3), len(df))

        self.assertEqual(edges, {(1, 0), (3, 2)})

    def test_k_previous_must_be_positive(self) -> None:
        df = pd.DataFrame({"TransactionDT": [100], "node_id": [0]})
        proxy = pd.Series(["entity"], dtype="string")

        with self.assertRaises(ValueError):
            build_temporal_edges(df, proxy, 0)

    def test_vectorized_edges_match_strict_time_brute_force(self) -> None:
        rng = np.random.default_rng(42)
        n_nodes = 40
        df = pd.DataFrame(
            {
                "TransactionDT": rng.choice([100, 200, 300, 400], n_nodes),
                "node_id": rng.permutation(n_nodes),
            }
        )
        proxy = pd.Series(
            rng.choice(["a", "b", "c", "d"], n_nodes),
            dtype="string",
        )
        k_previous = 3

        actual = decode_edges(
            build_temporal_edges(df, proxy, k_previous),
            n_nodes,
        )
        expected: set[tuple[int, int]] = set()
        for row_index, row in df.iterrows():
            strictly_earlier = df.loc[
                proxy.eq(proxy.iloc[row_index]) & df["TransactionDT"].lt(row["TransactionDT"])
            ].sort_values(["TransactionDT", "node_id"])
            for destination in strictly_earlier["node_id"].tail(k_previous):
                expected.add((int(row["node_id"]), int(destination)))

        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
