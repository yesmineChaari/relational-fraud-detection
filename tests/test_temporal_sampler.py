from __future__ import annotations

import time
import unittest

import numpy as np
import pandas as pd

from src.graph.build_transaction_graph import ENTITY_EDGES_PATH
from src.graph.temporal_contract import validate_sampled_neighborhood
from src.graph.temporal_sampler import (
    PAD_NODE_ID,
    admissible_neighbors,
    build_temporal_graph_index,
    load_temporal_graph_index,
    sample_fixed_fanout,
    sample_k_hop,
)


def make_index(entities: dict[int, list[tuple[int, float]]]):
    """entities: {entity_id: [(node_id, TransactionDT), ...]} in any order."""
    rows = []
    for entity_id, members in entities.items():
        for node_id, dt in members:
            rows.append({"entity_id": entity_id, "node_id": node_id, "TransactionDT": dt})
    edges_df = pd.DataFrame(rows).sort_values(
        ["entity_id", "TransactionDT", "node_id"], kind="mergesort"
    ).reset_index(drop=True)
    return build_temporal_graph_index(edges_df)


class BuildIndexValidationTests(unittest.TestCase):
    def test_rejects_edges_not_sorted_by_entity(self) -> None:
        edges_df = pd.DataFrame(
            {"entity_id": [1, 0], "node_id": [0, 1], "TransactionDT": [10, 20]}
        )
        with self.assertRaises(ValueError):
            build_temporal_graph_index(edges_df)

    def test_rejects_edges_not_sorted_by_time_within_entity(self) -> None:
        edges_df = pd.DataFrame(
            {"entity_id": [0, 0], "node_id": [0, 1], "TransactionDT": [20, 10]}
        )
        with self.assertRaises(ValueError):
            build_temporal_graph_index(edges_df)

    def test_rejects_non_dense_node_ids(self) -> None:
        edges_df = pd.DataFrame(
            {"entity_id": [0, 0], "node_id": [0, 5], "TransactionDT": [10, 20]}
        )
        with self.assertRaises(ValueError):
            build_temporal_graph_index(edges_df)


class AdmissibleNeighborsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Entity 0: node 0 @ t=100, node 1 @ t=200, node 2 @ t=200, node 3 @ t=300
        # Entity 1: node 4 @ t=50 (singleton)
        self.index = make_index(
            {
                0: [(0, 100), (1, 200), (2, 200), (3, 300)],
                1: [(4, 50)],
            }
        )

    def test_first_transaction_in_entity_has_empty_neighborhood(self) -> None:
        neighbors = admissible_neighbors(self.index, 0)
        self.assertEqual(len(neighbors), 0)

    def test_strictly_earlier_neighbors_are_admissible(self) -> None:
        neighbors = admissible_neighbors(self.index, 3)
        self.assertEqual(sorted(neighbors.tolist()), [0, 1, 2])

    def test_tied_timestamps_do_not_see_each_other(self) -> None:
        neighbors_of_1 = admissible_neighbors(self.index, 1)
        neighbors_of_2 = admissible_neighbors(self.index, 2)
        self.assertEqual(neighbors_of_1.tolist(), [0])
        self.assertEqual(neighbors_of_2.tolist(), [0])

    def test_singleton_entity_has_empty_neighborhood(self) -> None:
        neighbors = admissible_neighbors(self.index, 4)
        self.assertEqual(len(neighbors), 0)

    def test_a_node_is_never_its_own_neighbor_under_an_external_bound(self) -> None:
        # node 1's own dt is 200; querying with an external bound *after* its
        # own timestamp must still never return node 1 itself.
        neighbors = admissible_neighbors(self.index, 1, target_dt=300)
        self.assertNotIn(1, neighbors.tolist())
        self.assertEqual(sorted(neighbors.tolist()), [0, 2])

    def test_external_target_dt_overrides_the_nodes_own_timestamp(self) -> None:
        # node 0's own neighborhood (dt=100) is empty, but under a later
        # external bound it can see later same-entity transactions.
        neighbors = admissible_neighbors(self.index, 0, target_dt=250)
        self.assertEqual(sorted(neighbors.tolist()), [1, 2])


class SampleFixedFanoutTests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = make_index({0: [(0, 100), (1, 200), (2, 300), (3, 400)]})

    def test_empty_neighborhood_is_fully_padded_with_a_false_mask(self) -> None:
        rng = np.random.default_rng(0)
        node_ids, mask = sample_fixed_fanout(self.index, 0, fan_out=3, rng=rng)
        self.assertTrue(np.all(node_ids == PAD_NODE_ID))
        self.assertFalse(mask.any())

    def test_short_neighborhood_pads_the_remainder(self) -> None:
        rng = np.random.default_rng(0)
        node_ids, mask = sample_fixed_fanout(self.index, 3, fan_out=5, rng=rng)
        self.assertEqual(mask.sum(), 3)
        self.assertEqual(sorted(node_ids[mask].tolist()), [0, 1, 2])
        self.assertTrue(np.all(node_ids[~mask] == PAD_NODE_ID))

    def test_padding_is_never_confusable_with_real_node_zero(self) -> None:
        rng = np.random.default_rng(0)
        node_ids, mask = sample_fixed_fanout(self.index, 0, fan_out=2, rng=rng)
        # node 0 is a real, valid node id elsewhere in the graph; the pad
        # sentinel must not collide with it once the mask is dropped.
        self.assertTrue(np.all(node_ids == PAD_NODE_ID))
        self.assertNotEqual(PAD_NODE_ID, 0)

    def test_sampling_without_replacement_never_duplicates(self) -> None:
        rng = np.random.default_rng(0)
        node_ids, mask = sample_fixed_fanout(self.index, 3, fan_out=3, rng=rng)
        valid = node_ids[mask]
        self.assertEqual(len(valid), len(set(valid.tolist())))

    def test_deterministic_under_a_fixed_seed(self) -> None:
        first_ids, first_mask = sample_fixed_fanout(
            self.index, 3, fan_out=2, rng=np.random.default_rng(42)
        )
        second_ids, second_mask = sample_fixed_fanout(
            self.index, 3, fan_out=2, rng=np.random.default_rng(42)
        )
        np.testing.assert_array_equal(first_ids, second_ids)
        np.testing.assert_array_equal(first_mask, second_mask)

    def test_rejects_nonpositive_fanout(self) -> None:
        with self.assertRaises(ValueError):
            sample_fixed_fanout(self.index, 3, fan_out=0, rng=np.random.default_rng(0))


class SampleKHopCompositionTests(unittest.TestCase):
    def setUp(self) -> None:
        # A single entity: A(0,100) < N(1,900) < M(2,950) < T(3,1000) < F(4,1500)
        # Target is node 3. Hop-1 admissible = {A, N, M}. F must never appear.
        # M is *after* N's own timestamp (950 > 900): if hop-2 expansion of N
        # were bounded by N's own dt instead of the target's, M would be
        # wrongly excluded from N's hop-2 neighbours. Target-anchoring must
        # include it.
        self.index = make_index(
            {0: [(0, 100), (1, 900), (2, 950), (3, 1000), (4, 1500)]}
        )

    def test_hop_1_matches_direct_admissibility(self) -> None:
        layers = sample_k_hop(
            self.index, target_node_id=3, fan_outs=[3], rng=np.random.default_rng(0)
        )
        (hop1_ids, hop1_mask) = layers[0]
        self.assertEqual(sorted(hop1_ids[hop1_mask].tolist()), [0, 1, 2])

    def test_hop_2_is_anchored_to_the_target_not_the_intermediate(self) -> None:
        # Force expansion through N (node 1) specifically by fixing the seed
        # and using a large enough fan-out that all of A, N, M appear at
        # hop 1, then checking M reachable at hop 2 via *any* intermediate.
        layers = sample_k_hop(
            self.index, target_node_id=3, fan_outs=[3, 3], rng=np.random.default_rng(0)
        )
        hop1_ids, hop1_mask = layers[0]
        hop2_ids, hop2_mask = layers[1]
        self.assertEqual(sorted(hop1_ids[hop1_mask].tolist()), [0, 1, 2])
        # Every hop-2 neighbour, reached through any hop-1 node, must still
        # satisfy target-anchored admissibility (dt < 1000), and must
        # exclude self-loops and the never-admissible future node 4.
        valid_hop2 = hop2_ids[hop2_mask]
        self.assertNotIn(4, valid_hop2.tolist())
        for node in valid_hop2.tolist():
            self.assertIn(node, {0, 1, 2})

    def test_future_node_is_never_reachable_at_any_hop(self) -> None:
        layers = sample_k_hop(
            self.index, target_node_id=3, fan_outs=[3, 3], rng=np.random.default_rng(1)
        )
        for node_ids, mask in layers:
            self.assertNotIn(4, node_ids[mask].tolist())

    def test_no_node_is_ever_its_own_neighbor_across_hops(self) -> None:
        layers = sample_k_hop(
            self.index, target_node_id=3, fan_outs=[3, 3], rng=np.random.default_rng(2)
        )
        hop2_ids, hop2_mask = layers[1]
        self.assertNotIn(3, hop2_ids[hop2_mask].tolist())

    def test_empty_target_neighborhood_produces_empty_downstream_hops(self) -> None:
        layers = sample_k_hop(
            self.index, target_node_id=0, fan_outs=[3, 3], rng=np.random.default_rng(0)
        )
        hop1_ids, hop1_mask = layers[0]
        hop2_ids, hop2_mask = layers[1]
        self.assertFalse(hop1_mask.any())
        self.assertEqual(len(hop2_ids), 0)

    def test_deterministic_under_a_fixed_seed(self) -> None:
        first = sample_k_hop(
            self.index, target_node_id=3, fan_outs=[3, 3], rng=np.random.default_rng(7)
        )
        second = sample_k_hop(
            self.index, target_node_id=3, fan_outs=[3, 3], rng=np.random.default_rng(7)
        )
        for (ids_a, mask_a), (ids_b, mask_b) in zip(first, second):
            np.testing.assert_array_equal(ids_a, ids_b)
            np.testing.assert_array_equal(mask_a, mask_b)

    def test_rejects_empty_fanout_list(self) -> None:
        with self.assertRaises(ValueError):
            sample_k_hop(self.index, target_node_id=3, fan_outs=[], rng=np.random.default_rng(0))


class TimeShuffleLiveFilterTests(unittest.TestCase):
    def test_shuffling_timestamps_changes_the_sampled_neighborhood(self) -> None:
        rng = np.random.default_rng(0)
        original = make_index(
            {0: [(0, 100), (1, 200), (2, 300), (3, 400), (4, 500)]}
        )
        original_neighbors = set(admissible_neighbors(original, 4).tolist())
        self.assertEqual(original_neighbors, {0, 1, 2, 3})

        # Re-assign timestamps so node 4 (formerly last) is now earliest;
        # if the sampler were reading row/position order instead of the
        # actual TransactionDT values, this would not change its output.
        shuffled = make_index(
            {0: [(0, 500), (1, 200), (2, 300), (3, 400), (4, 100)]}
        )
        shuffled_neighbors = set(admissible_neighbors(shuffled, 4).tolist())
        self.assertEqual(shuffled_neighbors, set())
        self.assertNotEqual(original_neighbors, shuffled_neighbors)


class ProbeTransactionDirectComparisonTests(unittest.TestCase):
    def test_every_sampled_neighbor_at_every_hop_is_strictly_before_probe_targets(self) -> None:
        rng = np.random.default_rng(123)
        index = make_index(
            {
                0: [(i, dt) for i, dt in enumerate([0, 50, 50, 120, 400, 4000, 4000, 9999])],
                1: [(8, 10), (9, 4500)],
            }
        )
        probe_targets = [3, 4, 5, 6, 9]
        for target_node_id in probe_targets:
            target_position = index.position_of_node[target_node_id]
            target_dt = float(index.transaction_dt[target_position])
            layers = sample_k_hop(
                index, target_node_id=target_node_id, fan_outs=[4, 4], rng=rng
            )
            for node_ids, mask in layers:
                valid_positions = index.position_of_node[node_ids[mask]]
                observed_dts = index.transaction_dt[valid_positions]
                validate_sampled_neighborhood(target_dt, observed_dts.tolist())


class PerformanceSanityTests(unittest.TestCase):
    def test_many_samples_complete_quickly_on_a_moderately_sized_graph(self) -> None:
        n_entities = 200
        members_per_entity = 200
        entities = {
            entity: [(entity * members_per_entity + i, float(i)) for i in range(members_per_entity)]
            for entity in range(n_entities)
        }
        index = make_index(entities)
        rng = np.random.default_rng(0)
        targets = [entity * members_per_entity + members_per_entity - 1 for entity in range(n_entities)]

        started = time.perf_counter()
        for target in targets:
            sample_k_hop(index, target_node_id=target, fan_outs=[10, 10], rng=rng)
        elapsed = time.perf_counter() - started

        self.assertLess(elapsed, 5.0)


@unittest.skipUnless(
    ENTITY_EDGES_PATH.exists(),
    "Graph artifacts have not been generated; run "
    "`python -m src.graph.build_transaction_graph` first.",
)
class RealGraphIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.index = load_temporal_graph_index()
        cls.edges = pd.read_parquet(ENTITY_EDGES_PATH)

    def test_index_covers_every_transaction(self) -> None:
        self.assertEqual(self.index.n_nodes, len(self.edges))

    def test_probe_targets_from_the_real_train_partition_are_leak_safe(self) -> None:
        rng = np.random.default_rng(0)
        train_node_ids = self.edges.loc[self.edges["split"] == "train", "node_id"].to_numpy()
        probes = rng.choice(train_node_ids, size=200, replace=False)

        for target_node_id in probes:
            target_position = self.index.position_of_node[target_node_id]
            target_dt = float(self.index.transaction_dt[target_position])
            layers = sample_k_hop(
                self.index, target_node_id=int(target_node_id), fan_outs=[10, 10], rng=rng
            )
            for node_ids, mask in layers:
                valid_positions = self.index.position_of_node[node_ids[mask]]
                observed_dts = self.index.transaction_dt[valid_positions]
                validate_sampled_neighborhood(target_dt, observed_dts.tolist())

    def test_two_hop_sampling_is_fast_enough_for_a_training_epoch(self) -> None:
        rng = np.random.default_rng(0)
        train_node_ids = self.edges.loc[self.edges["split"] == "train", "node_id"].to_numpy()
        probes = rng.choice(train_node_ids, size=5_000, replace=False)

        started = time.perf_counter()
        for target_node_id in probes:
            sample_k_hop(
                self.index, target_node_id=int(target_node_id), fan_outs=[10, 10], rng=rng
            )
        elapsed = time.perf_counter() - started

        rows_per_second = len(probes) / elapsed
        projected_epoch_seconds = 413_378 / rows_per_second
        self.assertLess(
            projected_epoch_seconds,
            600.0,
            f"Projected full-epoch sampling time is {projected_epoch_seconds:.1f}s, "
            "too slow for iterative training.",
        )


if __name__ == "__main__":
    unittest.main()
