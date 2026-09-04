from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.graph.build_transaction_graph import (
    ENTITY_EDGES_PATH,
    METADATA_PATH,
    NODES_PATH,
    assign_node_ids,
    build_entity_edges,
    compute_entity_diagnostics,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = 590_540


def make_frame(transaction_ids, times, card1_values) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": transaction_ids,
            "TransactionDT": times,
            "split": ["train"] * len(transaction_ids),
            "card1": card1_values,
        }
    )


class AssignNodeIdsTests(unittest.TestCase):
    def test_node_ids_follow_ascending_transaction_id_order(self) -> None:
        df = make_frame([300, 100, 200], [3, 1, 2], [1, 1, 1])

        result = assign_node_ids(df)

        self.assertEqual(result["TransactionID"].tolist(), [100, 200, 300])
        self.assertEqual(result["node_id"].tolist(), [0, 1, 2])

    def test_node_id_assignment_is_deterministic_regardless_of_input_order(self) -> None:
        df_a = make_frame([100, 200, 300], [1, 2, 3], [1, 1, 1])
        df_b = make_frame([300, 200, 100], [3, 2, 1], [1, 1, 1])

        result_a = assign_node_ids(df_a)
        result_b = assign_node_ids(df_b)

        pd.testing.assert_frame_equal(
            result_a.reset_index(drop=True), result_b.reset_index(drop=True)
        )


class BuildEntityEdgesTests(unittest.TestCase):
    def test_edges_are_sorted_by_entity_then_time(self) -> None:
        df = assign_node_ids(
            make_frame(
                [10, 11, 12, 13],
                [500, 100, 300, 200],
                [7, 5, 5, 7],
            )
        )

        edges = build_entity_edges(df)

        self.assertEqual(edges["card1"].tolist(), [5, 5, 7, 7])
        self.assertEqual(edges["TransactionDT"].tolist(), [100, 300, 200, 500])

    def test_entity_ids_are_a_deterministic_dense_ranking_of_card1_values(self) -> None:
        df = assign_node_ids(make_frame([1, 2, 3], [10, 20, 30], [50, 20, 20]))

        edges = build_entity_edges(df)

        entity_by_card1 = dict(zip(edges["card1"], edges["entity_id"]))
        self.assertEqual(entity_by_card1[20], 0)
        self.assertEqual(entity_by_card1[50], 1)

    def test_edge_row_count_equals_transaction_count(self) -> None:
        df = assign_node_ids(make_frame([1, 2, 3, 4], [1, 2, 3, 4], [1, 1, 2, 2]))

        edges = build_entity_edges(df)

        self.assertEqual(len(edges), len(df))


class ComputeEntityDiagnosticsTests(unittest.TestCase):
    def test_singleton_entities_are_isolated(self) -> None:
        df = assign_node_ids(make_frame([1, 2, 3], [1, 2, 3], [1, 2, 3]))
        edges = build_entity_edges(df)

        diagnostics = compute_entity_diagnostics(edges)

        self.assertEqual(diagnostics["entity_count"], 3)
        self.assertEqual(diagnostics["isolated_nodes"], 3)
        self.assertEqual(diagnostics["largest_component_size"], 1)

    def test_largest_component_matches_the_biggest_entity(self) -> None:
        df = assign_node_ids(
            make_frame(
                [1, 2, 3, 4, 5],
                [1, 2, 3, 4, 5],
                [9, 9, 9, 9, 1],
            )
        )
        edges = build_entity_edges(df)

        diagnostics = compute_entity_diagnostics(edges)

        self.assertEqual(diagnostics["entity_count"], 2)
        self.assertEqual(diagnostics["largest_component_size"], 4)
        self.assertAlmostEqual(diagnostics["largest_component_pct"], 80.0)


@unittest.skipUnless(
    NODES_PATH.exists() and ENTITY_EDGES_PATH.exists() and METADATA_PATH.exists(),
    "Graph artifacts have not been generated; run "
    "`python -m src.graph.build_transaction_graph` first.",
)
class GeneratedGraphArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.nodes = pd.read_parquet(NODES_PATH)
        cls.edges = pd.read_parquet(ENTITY_EDGES_PATH)
        with METADATA_PATH.open(encoding="utf-8") as handle:
            cls.metadata = json.load(handle)

    def test_node_row_count(self) -> None:
        self.assertEqual(len(self.nodes), EXPECTED_ROWS)

    def test_node_ids_are_unique_and_contiguous(self) -> None:
        node_ids = self.nodes["node_id"].to_numpy()
        self.assertTrue(np.array_equal(np.sort(node_ids), np.arange(EXPECTED_ROWS)))

    def test_isFraud_is_absent_from_nodes(self) -> None:
        self.assertNotIn("isFraud", self.nodes.columns)

    def test_edge_row_count_matches_node_count(self) -> None:
        self.assertEqual(len(self.edges), EXPECTED_ROWS)

    def test_every_node_has_exactly_one_entity_edge(self) -> None:
        counts = self.edges["node_id"].value_counts()
        self.assertEqual(len(counts), EXPECTED_ROWS)
        self.assertTrue((counts == 1).all())

    def test_entity_edges_are_sorted_by_entity_then_time(self) -> None:
        sort_keys = self.edges[["entity_id", "TransactionDT", "node_id"]]
        expected = sort_keys.sort_values(
            ["entity_id", "TransactionDT", "node_id"], kind="mergesort"
        ).reset_index(drop=True)
        pd.testing.assert_frame_equal(sort_keys.reset_index(drop=True), expected)

    def test_metadata_declares_no_target_labels_used(self) -> None:
        self.assertFalse(self.metadata["target_labels_used"])
        self.assertFalse(self.metadata["isFraud_present_in_node_features"])

    def test_metadata_reconciled_against_relational_audit(self) -> None:
        self.assertTrue(self.metadata["reconciled_against_relational_audit"])

    def test_metadata_declares_frozen_split_consumed(self) -> None:
        self.assertTrue(self.metadata["frozen_split_assignment_consumed"])

    def test_metadata_feature_count_matches_frozen_b0(self) -> None:
        self.assertEqual(self.metadata["feature_column_count"], 435)


if __name__ == "__main__":
    unittest.main()
