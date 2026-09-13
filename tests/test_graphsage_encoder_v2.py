from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import torch

from src.graph.leakage_checks import cross_fit_provenance
from src.graph.temporal_sampler import build_temporal_graph_index, count_admissible_neighbors
from src.graph.train_graphsage_encoder import (
    EMBEDDING_DIM,
    EMBEDDINGS_PATH,
    HIDDEN_DIM,
    READOUT_NEIGHBOURHOOD_ONLY,
    READOUT_SELF_AND_NEIGHBOURHOOD,
    sample_batch_neighborhoods,
)
from src.graph.train_graphsage_encoder_v2 import (
    COUNT_BLIND,
    READOUT,
    WITH_COUNTS,
    CardinalityAwareGraphSAGEEncoder,
    CardinalityAwareGraphSAGEWithHead,
    V2Run,
    build_batch_tensors_v2,
    fit_cross_fitted_encoders,
    initial_encoder_state,
    sample_batch_neighborhoods_v2,
    state_fingerprint,
)
from src.graph.train_graphsage_variants import ENCODER_VARIANTS, EncoderBudget, EncoderContext


def make_index(entities: dict[int, list[tuple[int, float]]]):
    """entities: {entity_id: [(node_id, TransactionDT), ...]} in any order."""
    rows = [
        {"entity_id": entity_id, "node_id": node_id, "TransactionDT": dt}
        for entity_id, members in entities.items()
        for node_id, dt in members
    ]
    edges_df = (
        pd.DataFrame(rows)
        .sort_values(["entity_id", "TransactionDT", "node_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return build_temporal_graph_index(edges_df)


def random_batch(batch_size: int, f1: int, f2: int, input_dim: int, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return {
        "x_target": torch.randn(batch_size, input_dim, generator=generator),
        "x_hop1": torch.randn(batch_size, f1, input_dim, generator=generator),
        "hop1_mask": torch.ones(batch_size, f1, dtype=torch.bool),
        "x_hop2": torch.randn(batch_size, f1, f2, input_dim, generator=generator),
        "hop2_mask": torch.ones(batch_size, f1, f2, dtype=torch.bool),
        "hop1_count": torch.full((batch_size, f1), 5.0),
        "target_count": torch.full((batch_size,), 20.0),
    }


class SampleBatchNeighborhoodsV2Tests(unittest.TestCase):
    def test_neighbourhood_is_exactly_the_frozen_samplers(self) -> None:
        index = make_index({0: [(i, float(i)) for i in range(30)]})
        targets = np.array([29, 15, 3])
        frozen = sample_batch_neighborhoods(index, targets, (4, 3), np.random.default_rng(7))
        v2 = sample_batch_neighborhoods_v2(index, targets, (4, 3), np.random.default_rng(7))
        for a, b in zip(frozen, v2[:4]):
            np.testing.assert_array_equal(a, b)

    def test_counts_see_past_the_fan_out_cap(self) -> None:
        # The blind spot this encoder closes. Both targets saturate the 10-slot
        # draw, so the frozen encoder's inputs cannot tell 11 prior transactions
        # from 499; the true counts can.
        index = make_index(
            {
                0: [(i, float(i)) for i in range(12)],
                1: [(12 + i, float(i)) for i in range(500)],
            }
        )
        _, hop1_mask, _, _, _, target_count = sample_batch_neighborhoods_v2(
            index, np.array([11, 511]), (10, 10), np.random.default_rng(0)
        )
        np.testing.assert_array_equal(hop1_mask.sum(axis=1), [10, 10])
        np.testing.assert_array_equal(target_count, [11.0, 499.0])

    def test_hop1_count_is_the_neighbours_own_history_depth(self) -> None:
        # Node i sits at t=i, so exactly i transactions precede it.
        index = make_index({0: [(i, float(i)) for i in range(30)]})
        hop1_ids, hop1_mask, _, _, hop1_count, _ = sample_batch_neighborhoods_v2(
            index, np.array([29]), (10, 3), np.random.default_rng(1)
        )
        valid = hop1_mask[0]
        np.testing.assert_array_equal(hop1_count[0][valid], hop1_ids[0][valid].astype(np.float32))

    def test_a_target_anchored_hop1_count_would_carry_nothing(self) -> None:
        # Why the hop-1 count is bounded by the neighbour's own timestamp: on a
        # single flat relation every neighbour shares the target's entity, so
        # the target-anchored count is the target's count minus one for all.
        index = make_index({0: [(i, float(i)) for i in range(30)]})
        hop1_ids, hop1_mask, _, _, _, target_count = sample_batch_neighborhoods_v2(
            index, np.array([29]), (10, 3), np.random.default_rng(2)
        )
        anchored = {
            count_admissible_neighbors(index, int(node), target_dt=29.0)
            for node in hop1_ids[0][hop1_mask[0]]
        }
        self.assertEqual(anchored, {int(target_count[0]) - 1})

    def test_padded_hop1_slots_carry_a_zero_count(self) -> None:
        index = make_index({0: [(i, float(i)) for i in range(4)]})
        _, hop1_mask, _, _, hop1_count, target_count = sample_batch_neighborhoods_v2(
            index, np.array([3]), (10, 3), np.random.default_rng(0)
        )
        self.assertEqual(int(hop1_mask.sum()), 3)
        self.assertTrue(np.all(hop1_count[~hop1_mask] == 0.0))
        self.assertEqual(float(target_count[0]), 3.0)

    def test_first_transaction_has_a_zero_target_count(self) -> None:
        index = make_index({0: [(i, float(i)) for i in range(4)]})
        *_, target_count = sample_batch_neighborhoods_v2(
            index, np.array([0]), (10, 3), np.random.default_rng(0)
        )
        self.assertEqual(float(target_count[0]), 0.0)


class BuildBatchTensorsV2Tests(unittest.TestCase):
    def setUp(self) -> None:
        self.index = make_index({0: [(i, float(i)) for i in range(40)]})
        self.features = np.arange(40 * 3, dtype=np.float32).reshape(40, 3)
        self.targets = np.array([39, 20, 10])

    def _tensors(self, count_mode: str):
        return build_batch_tensors_v2(
            self.index, self.features, self.targets, (5, 4), np.random.default_rng(3), count_mode
        )

    def test_count_blind_zeroes_every_count_and_changes_nothing_else(self) -> None:
        with_counts = self._tensors(WITH_COUNTS)
        blind = self._tensors(COUNT_BLIND)
        self.assertTrue(torch.all(blind["hop1_count"] == 0))
        self.assertTrue(torch.all(blind["target_count"] == 0))
        self.assertTrue(torch.any(with_counts["target_count"] > 0))
        for key in ("x_target", "x_hop1", "hop1_mask", "x_hop2", "hop2_mask"):
            torch.testing.assert_close(with_counts[key], blind[key])

    def test_rejects_an_unknown_count_mode(self) -> None:
        with self.assertRaises(ValueError):
            self._tensors("sometimes")

    def test_keys_match_the_encoder_signature(self) -> None:
        model = CardinalityAwareGraphSAGEWithHead(3, HIDDEN_DIM, EMBEDDING_DIM)
        embedding, logit = model(**self._tensors(WITH_COUNTS))
        self.assertEqual(embedding.shape, (3, EMBEDDING_DIM))
        self.assertEqual(logit.shape, (3,))


class CardinalityAwareEncoderTests(unittest.TestCase):
    INPUT_DIM = 5

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.model = CardinalityAwareGraphSAGEWithHead(self.INPUT_DIM, HIDDEN_DIM, EMBEDDING_DIM)

    def test_layer_widths_gain_one_count_column_per_layer(self) -> None:
        nbhd = CardinalityAwareGraphSAGEEncoder(
            self.INPUT_DIM, HIDDEN_DIM, EMBEDDING_DIM, READOUT_NEIGHBOURHOOD_ONLY
        )
        full = CardinalityAwareGraphSAGEEncoder(
            self.INPUT_DIM, HIDDEN_DIM, EMBEDDING_DIM, READOUT_SELF_AND_NEIGHBOURHOOD
        )
        self.assertEqual(nbhd.layer1.in_features, 2 * self.INPUT_DIM + 1)
        self.assertEqual(nbhd.layer2.in_features, HIDDEN_DIM + 1)
        self.assertEqual(full.layer2.in_features, self.INPUT_DIM + HIDDEN_DIM + 1)

    def test_default_readout_is_the_registered_one(self) -> None:
        self.assertEqual(self.model.encoder.readout, READOUT)
        self.assertEqual(READOUT, READOUT_NEIGHBOURHOOD_ONLY)

    def test_rejects_an_unknown_readout(self) -> None:
        with self.assertRaises(ValueError):
            CardinalityAwareGraphSAGEEncoder(self.INPUT_DIM, HIDDEN_DIM, EMBEDDING_DIM, "self")

    def test_output_shapes_and_unit_norm(self) -> None:
        embedding, logit = self.model(**random_batch(4, 3, 2, self.INPUT_DIM))
        self.assertEqual(embedding.shape, (4, EMBEDDING_DIM))
        self.assertEqual(logit.shape, (4,))
        torch.testing.assert_close(embedding.norm(p=2, dim=-1), torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_fully_masked_neighbourhoods_still_produce_finite_output(self) -> None:
        batch = random_batch(2, 3, 2, self.INPUT_DIM)
        batch["hop1_mask"] = torch.zeros(2, 3, dtype=torch.bool)
        batch["hop2_mask"] = torch.zeros(2, 3, 2, dtype=torch.bool)
        batch["hop1_count"] = torch.zeros(2, 3)
        batch["target_count"] = torch.zeros(2)
        embedding, logit = self.model(**batch)
        self.assertTrue(torch.isfinite(embedding).all())
        self.assertTrue(torch.isfinite(logit).all())

    def test_backward_pass_populates_every_gradient(self) -> None:
        _, logit = self.model(**random_batch(4, 3, 2, self.INPUT_DIM))
        logit.sum().backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, f"{name} has no gradient")

    def test_the_embedding_responds_to_the_target_count(self) -> None:
        batch = random_batch(8, 3, 2, self.INPUT_DIM)
        shallow, _ = self.model(**{**batch, "target_count": torch.full((8,), 10.0)})
        deep, _ = self.model(**{**batch, "target_count": torch.full((8,), 500.0)})
        self.assertFalse(torch.allclose(shallow, deep))

    def test_the_embedding_responds_to_the_hop1_counts(self) -> None:
        batch = random_batch(8, 3, 2, self.INPUT_DIM)
        shallow, _ = self.model(**{**batch, "hop1_count": torch.full((8, 3), 1.0)})
        deep, _ = self.model(**{**batch, "hop1_count": torch.full((8, 3), 300.0)})
        self.assertFalse(torch.allclose(shallow, deep))

    def test_reproducible_under_a_fixed_seed(self) -> None:
        torch.manual_seed(123)
        model_a = CardinalityAwareGraphSAGEWithHead(self.INPUT_DIM, HIDDEN_DIM, EMBEDDING_DIM)
        torch.manual_seed(123)
        model_b = CardinalityAwareGraphSAGEWithHead(self.INPUT_DIM, HIDDEN_DIM, EMBEDDING_DIM)
        batch = random_batch(4, 3, 2, self.INPUT_DIM)
        embedding_a, logit_a = model_a(**batch)
        embedding_b, logit_b = model_b(**batch)
        torch.testing.assert_close(embedding_a, embedding_b)
        logit_a.sum().backward()
        logit_b.sum().backward()
        for (_, param_a), (_, param_b) in zip(
            model_a.named_parameters(), model_b.named_parameters()
        ):
            torch.testing.assert_close(param_a.grad, param_b.grad)


class SharedInitialisationTests(unittest.TestCase):
    def test_the_same_seed_gives_the_same_state(self) -> None:
        first = initial_encoder_state(6, READOUT, seed=42)
        second = initial_encoder_state(6, READOUT, seed=42)
        self.assertEqual(state_fingerprint(first), state_fingerprint(second))

    def test_different_seeds_give_different_states(self) -> None:
        self.assertNotEqual(
            state_fingerprint(initial_encoder_state(6, READOUT, seed=42)),
            state_fingerprint(initial_encoder_state(6, READOUT, seed=43)),
        )

    def test_models_loaded_from_the_snapshot_start_identical(self) -> None:
        state = initial_encoder_state(6, READOUT, seed=7)
        models = []
        for other_seed in (0, 1):
            torch.manual_seed(other_seed)
            model = CardinalityAwareGraphSAGEWithHead(6, HIDDEN_DIM, EMBEDDING_DIM)
            model.load_state_dict(state)
            models.append(model)
        fingerprints = {state_fingerprint(model.state_dict()) for model in models}
        self.assertEqual(fingerprints, {state_fingerprint(state)})


TINY_BUDGET = EncoderBudget(
    steps_per_epoch=2,
    batch_size=8,
    max_epochs=2,
    patience=1,
    min_delta=0.0,
    monitor="full_validation_partition",
)


def make_tiny_context(n_entities: int = 6, per_entity: int = 24, dim: int = 4) -> EncoderContext:
    entities = {
        e: [(e * per_entity + i, float(i)) for i in range(per_entity)] for e in range(n_entities)
    }
    n = n_entities * per_entity
    node_ids = np.arange(n)
    position = node_ids % per_entity
    split = np.where(position < 16, "train", np.where(position < 20, "validation", "test"))
    labels = (node_ids % 3 == 0).astype(np.float64)
    train = node_ids[split == "train"]
    positives = int(labels[train].sum())
    return EncoderContext(
        meta=pd.DataFrame(
            {"node_id": node_ids, "TransactionID": node_ids + 10_000, "split": split}
        ),
        feature_columns=[f"f{i}" for i in range(dim)],
        index=make_index(entities),
        feature_cache=np.random.default_rng(0).standard_normal((n, dim)).astype(np.float32),
        labels_by_node_id=labels,
        train_node_ids=train,
        validation_node_ids=node_ids[split == "validation"],
        test_node_ids=node_ids[split == "test"],
        all_node_ids=node_ids,
        pos_weight=(len(train) - positives) / positives,
        train_fraud_count=positives,
    )


class CrossFittedEncodersTests(unittest.TestCase):
    SEED = 5

    @classmethod
    def setUpClass(cls) -> None:
        cls.ctx = make_tiny_context()
        cls.outcome = fit_cross_fitted_encoders(
            cls.ctx,
            seed=cls.SEED,
            count_mode=WITH_COUNTS,
            label="tiny",
            budget=TINY_BUDGET,
            n_folds=3,
            inference_batch_size=16,
        )

    def test_every_encoder_started_from_the_shared_initialisation(self) -> None:
        # The fix for the earlier cross-fitted control's unaligned latent bases.
        shared = self.outcome["cross_fitting"]["initial_state_sha256"]
        started = [self.outcome["full"]["initial_state_sha256"]]
        started += [result["initial_state_sha256"] for result in self.outcome["folds"]]
        self.assertEqual(set(started), {shared})
        self.assertEqual(len(started), 4)

    def test_training_never_mutates_the_shared_snapshot(self) -> None:
        recomputed = initial_encoder_state(len(self.ctx.feature_columns), READOUT, self.SEED)
        self.assertEqual(
            state_fingerprint(recomputed), self.outcome["cross_fitting"]["initial_state_sha256"]
        )

    def test_provenance_is_sound(self) -> None:
        provenance = cross_fit_provenance(
            self.outcome["cross_fitting"],
            len(self.ctx.train_node_ids),
            len(self.ctx.validation_node_ids) + len(self.ctx.test_node_ids),
        )
        self.assertTrue(provenance["provenance_sound"], provenance["failures"])

    def test_folds_train_for_the_full_encoders_selected_epochs(self) -> None:
        epochs = self.outcome["full"]["best_epoch"]
        for result in self.outcome["folds"]:
            self.assertFalse(result["monitored"])
            self.assertEqual(result["stopped_epoch"], epochs)

    def test_every_node_gets_a_finite_embedding(self) -> None:
        embeddings = self.outcome["embeddings"]
        self.assertEqual(embeddings.shape, (len(self.ctx.meta), EMBEDDING_DIM))
        self.assertTrue(np.isfinite(embeddings).all())


class V2RunTests(unittest.TestCase):
    def test_seeds_and_count_modes_never_share_artifacts(self) -> None:
        runs = [V2Run(42), V2Run(43), V2Run(42, COUNT_BLIND)]
        for attribute in ("embeddings_path", "model_path", "report_dir", "leakage_gate_path"):
            paths = {getattr(run, attribute) for run in runs}
            self.assertEqual(len(paths), len(runs), attribute)

    def test_never_writes_over_the_frozen_or_control_embeddings(self) -> None:
        protected = {EMBEDDINGS_PATH, *(v.embeddings_path for v in ENCODER_VARIANTS.values())}
        for run in (V2Run(42), V2Run(42, COUNT_BLIND)):
            self.assertNotIn(run.embeddings_path, protected)

    def test_rejects_an_unknown_count_mode_and_a_negative_seed(self) -> None:
        with self.assertRaises(ValueError):
            V2Run(42, "half")
        with self.assertRaises(ValueError):
            V2Run(-1)


if __name__ == "__main__":
    unittest.main()
