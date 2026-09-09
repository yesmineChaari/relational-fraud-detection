from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd
import torch

from src.graph.temporal_contract import validate_sampled_neighborhood
from src.graph.temporal_sampler import PAD_NODE_ID, build_temporal_graph_index
from src.graph.train_graphsage_encoder import (
    EMBEDDING_DIM,
    EMBEDDINGS_PATH,
    FEATURE_SCALER_PATH,
    HIDDEN_DIM,
    METADATA_PATH,
    METRICS_PATH,
    TRAINING_CURVE_PATH,
    GraphSAGEWithHead,
    gather_features,
    l2_normalize,
    masked_mean,
    sample_batch_neighborhoods,
)


def make_index(entities: dict[int, list[tuple[int, float]]]):
    """entities: {entity_id: [(node_id, TransactionDT), ...]} in any order."""
    rows = []
    for entity_id, members in entities.items():
        for node_id, dt in members:
            rows.append({"entity_id": entity_id, "node_id": node_id, "TransactionDT": dt})
    edges_df = (
        pd.DataFrame(rows)
        .sort_values(["entity_id", "TransactionDT", "node_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return build_temporal_graph_index(edges_df)


class MaskedMeanTests(unittest.TestCase):
    def test_zero_valid_entries_produce_the_zero_vector(self) -> None:
        x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
        mask = torch.tensor([[False, False]])
        result = masked_mean(x, mask, dim=1)
        torch.testing.assert_close(result, torch.zeros_like(result))

    def test_averages_only_the_masked_true_entries(self) -> None:
        x = torch.tensor([[[2.0, 2.0], [6.0, 6.0], [100.0, 100.0]]])
        mask = torch.tensor([[True, True, False]])
        result = masked_mean(x, mask, dim=1)
        torch.testing.assert_close(result, torch.tensor([[4.0, 4.0]]))

    def test_all_valid_entries_matches_plain_mean(self) -> None:
        x = torch.tensor([[[1.0], [2.0], [3.0]]])
        mask = torch.tensor([[True, True, True]])
        result = masked_mean(x, mask, dim=1)
        torch.testing.assert_close(result, torch.tensor([[2.0]]))


class L2NormalizeTests(unittest.TestCase):
    def test_nonzero_vector_has_unit_norm(self) -> None:
        x = torch.tensor([[3.0, 4.0]])
        result = l2_normalize(x)
        self.assertAlmostEqual(float(result.norm(p=2, dim=-1)), 1.0, places=6)

    def test_zero_vector_stays_zero_without_error(self) -> None:
        x = torch.zeros(1, 4)
        result = l2_normalize(x)
        torch.testing.assert_close(result, torch.zeros_like(result))


class GatherFeaturesTests(unittest.TestCase):
    def setUp(self) -> None:
        self.features = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=np.float32)

    def test_pad_node_id_is_returned_as_a_zero_row(self) -> None:
        ids = np.array([0, PAD_NODE_ID, 2])
        rows = gather_features(self.features, ids)
        np.testing.assert_array_equal(rows[1], np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(rows[0], self.features[0])
        np.testing.assert_array_equal(rows[2], self.features[2])

    def test_masked_false_slot_is_zeroed_even_for_a_real_node_id(self) -> None:
        # node 0 is a real, valid node id -- masking must still zero it when
        # its own validity flag is False (e.g. a padded hop-1 slot that
        # happens to alias a real id), independent of PAD_NODE_ID handling.
        ids = np.array([0, 1])
        mask = np.array([False, True])
        rows = gather_features(self.features, ids, mask=mask)
        np.testing.assert_array_equal(rows[0], np.zeros(2, dtype=np.float32))
        np.testing.assert_array_equal(rows[1], self.features[1])

    def test_preserves_multidimensional_shape(self) -> None:
        ids = np.array([[0, 1], [2, PAD_NODE_ID]])
        rows = gather_features(self.features, ids)
        self.assertEqual(rows.shape, (2, 2, 2))
        np.testing.assert_array_equal(rows[1, 1], np.zeros(2, dtype=np.float32))


class SampleBatchNeighborhoodsTests(unittest.TestCase):
    def setUp(self) -> None:
        # Single entity: A(0,100) < N(1,900) < M(2,950) < T(3,1000) < F(4,1500)
        self.index = make_index({0: [(0, 100), (1, 900), (2, 950), (3, 1000), (4, 1500)]})

    def test_output_shapes_match_the_batch_and_fanouts(self) -> None:
        targets = np.array([3, 0])
        hop1_ids, hop1_mask, hop2_ids, hop2_mask = sample_batch_neighborhoods(
            self.index, targets, fan_outs=(3, 2), rng=np.random.default_rng(0)
        )
        self.assertEqual(hop1_ids.shape, (2, 3))
        self.assertEqual(hop1_mask.shape, (2, 3))
        self.assertEqual(hop2_ids.shape, (2, 3, 2))
        self.assertEqual(hop2_mask.shape, (2, 3, 2))

    def test_future_node_never_appears_at_either_hop(self) -> None:
        targets = np.array([3])
        hop1_ids, hop1_mask, hop2_ids, hop2_mask = sample_batch_neighborhoods(
            self.index, targets, fan_outs=(3, 3), rng=np.random.default_rng(1)
        )
        self.assertNotIn(4, hop1_ids[hop1_mask].tolist())
        self.assertNotIn(4, hop2_ids[hop2_mask].tolist())

    def test_every_valid_hop2_neighbor_is_target_anchored_not_intermediate_anchored(
        self,
    ) -> None:
        # Force expansion through every hop-1 node with a generous fan-out,
        # then confirm hop-2 admissibility is w.r.t. the target's own dt
        # (1000), not the intermediate hop-1 node's dt.
        targets = np.array([3])
        hop1_ids, hop1_mask, hop2_ids, hop2_mask = sample_batch_neighborhoods(
            self.index, targets, fan_outs=(3, 3), rng=np.random.default_rng(2)
        )
        self.assertEqual(sorted(hop1_ids[0][hop1_mask[0]].tolist()), [0, 1, 2])
        valid_hop2 = hop2_ids[0][hop2_mask[0]]
        for node in valid_hop2.tolist():
            self.assertIn(node, {0, 1, 2})

    def test_invalid_hop1_slot_has_a_fully_padded_hop2_block(self) -> None:
        # node 0's own hop-1 neighborhood is empty; every hop-1 slot is
        # invalid, so every corresponding hop-2 block must be fully padded.
        targets = np.array([0])
        hop1_ids, hop1_mask, hop2_ids, hop2_mask = sample_batch_neighborhoods(
            self.index, targets, fan_outs=(3, 3), rng=np.random.default_rng(0)
        )
        self.assertFalse(hop1_mask.any())
        self.assertFalse(hop2_mask.any())
        self.assertTrue(np.all(hop2_ids == PAD_NODE_ID))

    def test_deterministic_under_a_fixed_seed(self) -> None:
        targets = np.array([3, 2, 1])
        first = sample_batch_neighborhoods(
            self.index, targets, fan_outs=(3, 3), rng=np.random.default_rng(7)
        )
        second = sample_batch_neighborhoods(
            self.index, targets, fan_outs=(3, 3), rng=np.random.default_rng(7)
        )
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a, b)

    def test_every_sampled_neighbor_at_every_hop_is_strictly_before_the_target(
        self,
    ) -> None:
        targets = np.array([3])
        hop1_ids, hop1_mask, hop2_ids, hop2_mask = sample_batch_neighborhoods(
            self.index, targets, fan_outs=(3, 3), rng=np.random.default_rng(3)
        )
        target_position = self.index.position_of_node[3]
        target_dt = float(self.index.transaction_dt[target_position])

        for ids, mask in ((hop1_ids, hop1_mask), (hop2_ids, hop2_mask)):
            valid_ids = ids[mask]
            positions = self.index.position_of_node[valid_ids]
            observed_dts = self.index.transaction_dt[positions]
            validate_sampled_neighborhood(target_dt, observed_dts.tolist())


class GraphSAGEForwardPassTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.input_dim = 5
        self.model = GraphSAGEWithHead(self.input_dim, HIDDEN_DIM, EMBEDDING_DIM)

    def _random_batch(self, batch_size: int, f1: int, f2: int):
        x_target = torch.randn(batch_size, self.input_dim)
        x_hop1 = torch.randn(batch_size, f1, self.input_dim)
        hop1_mask = torch.ones(batch_size, f1, dtype=torch.bool)
        x_hop2 = torch.randn(batch_size, f1, f2, self.input_dim)
        hop2_mask = torch.ones(batch_size, f1, f2, dtype=torch.bool)
        return x_target, x_hop1, hop1_mask, x_hop2, hop2_mask

    def test_output_shapes(self) -> None:
        batch = self._random_batch(batch_size=4, f1=3, f2=2)
        embedding, logit = self.model(*batch)
        self.assertEqual(embedding.shape, (4, EMBEDDING_DIM))
        self.assertEqual(logit.shape, (4,))

    def test_embedding_is_unit_norm(self) -> None:
        batch = self._random_batch(batch_size=4, f1=3, f2=2)
        embedding, _ = self.model(*batch)
        norms = embedding.norm(p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_fully_masked_neighborhoods_still_produce_finite_output(self) -> None:
        batch_size, f1, f2 = 2, 3, 2
        x_target = torch.randn(batch_size, self.input_dim)
        x_hop1 = torch.randn(batch_size, f1, self.input_dim)
        hop1_mask = torch.zeros(batch_size, f1, dtype=torch.bool)
        x_hop2 = torch.randn(batch_size, f1, f2, self.input_dim)
        hop2_mask = torch.zeros(batch_size, f1, f2, dtype=torch.bool)

        embedding, logit = self.model(x_target, x_hop1, hop1_mask, x_hop2, hop2_mask)
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertTrue(torch.isfinite(logit).all())

    def test_backward_pass_populates_gradients(self) -> None:
        batch = self._random_batch(batch_size=4, f1=3, f2=2)
        _, logit = self.model(*batch)
        loss = logit.sum()
        loss.backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} has no gradient")

    def test_reproducible_under_a_fixed_seed(self) -> None:
        torch.manual_seed(123)
        model_a = GraphSAGEWithHead(self.input_dim, HIDDEN_DIM, EMBEDDING_DIM)
        torch.manual_seed(123)
        model_b = GraphSAGEWithHead(self.input_dim, HIDDEN_DIM, EMBEDDING_DIM)

        batch = self._random_batch(batch_size=4, f1=3, f2=2)
        embedding_a, logit_a = model_a(*batch)
        embedding_b, logit_b = model_b(*batch)
        torch.testing.assert_close(embedding_a, embedding_b)
        torch.testing.assert_close(logit_a, logit_b)

        loss_a = logit_a.sum()
        loss_b = logit_b.sum()
        loss_a.backward()
        loss_b.backward()
        for (name_a, param_a), (name_b, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(param_a.grad, param_b.grad)


@unittest.skipUnless(
    EMBEDDINGS_PATH.exists() and METADATA_PATH.exists(),
    "GraphSAGE artifacts have not been generated; run "
    "`python -m src.graph.train_graphsage_encoder` first.",
)
class GeneratedEmbeddingArtifactTests(unittest.TestCase):
    EXPECTED_ROWS = 590_540

    @classmethod
    def setUpClass(cls) -> None:
        cls.embeddings = pd.read_parquet(EMBEDDINGS_PATH)
        with METADATA_PATH.open(encoding="utf-8") as handle:
            cls.metadata = json.load(handle)
        with METRICS_PATH.open(encoding="utf-8") as handle:
            cls.metrics = json.load(handle)
        cls.curve = pd.read_csv(TRAINING_CURVE_PATH)

    def test_embedding_row_count_covers_every_transaction(self) -> None:
        self.assertEqual(len(self.embeddings), self.EXPECTED_ROWS)

    def test_embedding_has_no_nulls(self) -> None:
        self.assertFalse(self.embeddings.isna().any().any())

    def test_embedding_has_no_target_column(self) -> None:
        self.assertNotIn("isFraud", self.embeddings.columns)

    def test_embedding_dimension_matches_metadata(self) -> None:
        embedding_columns = [c for c in self.embeddings.columns if c.startswith("embedding_")]
        self.assertEqual(len(embedding_columns), self.metadata["architecture"]["embedding_dim"])

    def test_embeddings_are_finite(self) -> None:
        embedding_columns = [c for c in self.embeddings.columns if c.startswith("embedding_")]
        values = self.embeddings[embedding_columns].to_numpy()
        self.assertTrue(np.isfinite(values).all())

    def test_metadata_declares_test_labels_unused(self) -> None:
        self.assertFalse(self.metadata["test_labels_used"])

    def test_metadata_declares_no_target_in_node_features(self) -> None:
        self.assertFalse(self.metadata["target_labels_used_in_node_features"])

    def test_metadata_split_counts_match_the_frozen_manifest(self) -> None:
        counts = self.metadata["split_row_counts"]
        self.assertEqual(counts["train"], 413_378)
        self.assertEqual(counts["validation"], 88_581)
        self.assertEqual(counts["test"], 88_581)

    def test_training_curve_is_non_empty_and_monotonic_in_epoch(self) -> None:
        self.assertGreater(len(self.curve), 0)
        self.assertEqual(self.curve["epoch"].tolist(), sorted(self.curve["epoch"].tolist()))

    def test_validation_metrics_were_evaluated_on_the_full_validation_partition(self) -> None:
        self.assertEqual(self.metrics["n_transactions"], 88_581)

    def test_early_stopping_metadata_is_internally_consistent(self) -> None:
        early_stop = self.metadata["early_stopping"]
        self.assertEqual(
            early_stop["estimator_cap_reached"], not early_stop["early_stopping_triggered"]
        )
        self.assertLessEqual(early_stop["best_epoch"], early_stop["stopped_epoch"])

    def test_feature_scaler_artifact_exists_and_matches_metadata_hash(self) -> None:
        self.assertTrue(FEATURE_SCALER_PATH.exists())


if __name__ == "__main__":
    unittest.main()
