from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.graph.temporal_sampler import build_temporal_graph_index
from src.graph.train_graphsage_encoder import (
    EMBEDDING_DIM,
    EMBEDDINGS_PATH as FROZEN_EMBEDDINGS_PATH,
    HIDDEN_DIM,
    READOUT_NEIGHBOURHOOD_ONLY,
    READOUT_SELF_AND_NEIGHBOURHOOD,
    GraphSAGEWithHead,
    TemporalGraphSAGEEncoder,
    run_inference,
    sample_batch_neighborhoods,
)
from src.graph.train_graphsage_variants import (
    CONTROL_BUDGET,
    ENCODER_VARIANTS,
    EncoderBudget,
    EncoderVariant,
    assign_cross_fit_folds,
    embedding_frame,
)
from src.models.compare_g1_controls import (
    VERDICT_CONFOUND,
    build_known_limitations,
    embedding_partition_alignment,
    VERDICT_CONTROLS,
    VERDICT_NEGATIVE_STANDS,
    build_verdict,
    reaches_parity,
)
from src.models.train_lightgbm_g1 import embedding_feature_names
from src.models.train_lightgbm_g1_controls import (
    CONTROL_ORDER,
    CONTROL_RUNS,
    G1_PROTECTED_PATHS,
    SHUFFLED_VARIANT_NAME,
    permute_block_within_split,
)
from src.models.train_lightgbm_relational import B0_PROTECTED_PATHS
from src.models.train_lightgbm_g1 import B1_CARD1_PROTECTED_PATHS


def make_budget(**overrides) -> EncoderBudget:
    kwargs = {
        "steps_per_epoch": 4,
        "batch_size": 8,
        "max_epochs": 3,
        "patience": 2,
        "min_delta": 1e-4,
        "monitor": "full_validation_partition",
    }
    kwargs.update(overrides)
    return EncoderBudget(**kwargs)


class EncoderBudgetTests(unittest.TestCase):
    def test_targets_are_reported_in_rows_not_epochs(self) -> None:
        budget = make_budget(steps_per_epoch=10, batch_size=32, max_epochs=5)
        self.assertEqual(budget.targets_per_epoch, 320)
        described = budget.describe(train_rows=1_600)
        self.assertEqual(described["max_targets"], 1_600)
        self.assertAlmostEqual(described["passes_per_epoch"], 0.2)
        self.assertAlmostEqual(described["max_passes"], 1.0)

    def test_non_positive_sizes_are_rejected(self) -> None:
        for field in ("steps_per_epoch", "batch_size", "max_epochs", "patience"):
            with self.subTest(field=field):
                with self.assertRaises(ValueError):
                    make_budget(**{field: 0})

    def test_negative_min_delta_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            make_budget(min_delta=-1e-6)

    def test_a_subsampled_monitor_is_rejected(self) -> None:
        """The whole point of the budget control is full-partition selection."""
        with self.assertRaises(ValueError):
            make_budget(monitor="fixed_seed_subsample_of_validation_partition")

    def test_the_control_budget_raises_the_frozen_run_substantially(self) -> None:
        frozen_targets_per_epoch = 160 * 256
        self.assertGreater(CONTROL_BUDGET.targets_per_epoch, frozen_targets_per_epoch)
        self.assertGreater(
            CONTROL_BUDGET.targets_per_epoch * CONTROL_BUDGET.max_epochs,
            frozen_targets_per_epoch * 8,
        )
        self.assertEqual(CONTROL_BUDGET.monitor, "full_validation_partition")


class EncoderVariantTests(unittest.TestCase):
    def make_variant(self, **overrides) -> EncoderVariant:
        kwargs = {
            "name": "unit_variant",
            "readout": READOUT_SELF_AND_NEIGHBOURHOOD,
            "cross_fit_folds": None,
            "budget": make_budget(),
            "seed": 7,
            "isolates": "nothing",
            "description": "unit test variant",
        }
        kwargs.update(overrides)
        return EncoderVariant(**kwargs)

    def test_unknown_readout_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.make_variant(readout="self_only")

    def test_single_fold_cross_fitting_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.make_variant(cross_fit_folds=1)

    def test_every_variant_writes_to_its_own_paths(self) -> None:
        embedding_paths = {v.embeddings_path for v in ENCODER_VARIANTS.values()}
        report_dirs = {v.report_dir for v in ENCODER_VARIANTS.values()}
        model_paths = {v.model_path for v in ENCODER_VARIANTS.values()}
        self.assertEqual(len(embedding_paths), len(ENCODER_VARIANTS))
        self.assertEqual(len(report_dirs), len(ENCODER_VARIANTS))
        self.assertEqual(len(model_paths), len(ENCODER_VARIANTS))
        self.assertNotIn(FROZEN_EMBEDDINGS_PATH, embedding_paths)

    def test_the_three_encoder_controls_change_one_thing_each(self) -> None:
        reference = ENCODER_VARIANTS["extended_budget"]
        self.assertIsNone(reference.cross_fit_folds)
        self.assertEqual(reference.readout, READOUT_SELF_AND_NEIGHBOURHOOD)

        neighbourhood = ENCODER_VARIANTS["neighbourhood_only"]
        self.assertEqual(neighbourhood.readout, READOUT_NEIGHBOURHOOD_ONLY)
        self.assertIsNone(neighbourhood.cross_fit_folds)

        cross_fitted = ENCODER_VARIANTS["cross_fitted"]
        self.assertEqual(cross_fitted.readout, READOUT_SELF_AND_NEIGHBOURHOOD)
        self.assertGreaterEqual(cross_fitted.cross_fit_folds, 2)

        for variant in ENCODER_VARIANTS.values():
            self.assertIs(variant.budget, CONTROL_BUDGET)


class CrossFitFoldTests(unittest.TestCase):
    def test_folds_are_a_disjoint_cover_of_the_train_rows(self) -> None:
        train_node_ids = np.arange(1_000, 1_101)
        folds = assign_cross_fit_folds(train_node_ids, n_folds=3, seed=42)
        self.assertEqual(len(folds), len(train_node_ids))
        self.assertEqual(sorted(np.unique(folds)), [0, 1, 2])
        covered = np.concatenate([train_node_ids[folds == k] for k in range(3)])
        self.assertEqual(sorted(covered), sorted(train_node_ids))

    def test_fold_sizes_differ_by_at_most_one(self) -> None:
        folds = assign_cross_fit_folds(np.arange(101), n_folds=3, seed=1)
        sizes = [int((folds == k).sum()) for k in range(3)]
        self.assertLessEqual(max(sizes) - min(sizes), 1)

    def test_assignment_is_deterministic_for_a_seed(self) -> None:
        ids = np.arange(500)
        first = assign_cross_fit_folds(ids, n_folds=4, seed=42)
        second = assign_cross_fit_folds(ids, n_folds=4, seed=42)
        other = assign_cross_fit_folds(ids, n_folds=4, seed=43)
        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))

    def test_degenerate_fold_counts_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            assign_cross_fit_folds(np.arange(10), n_folds=1, seed=0)
        with self.assertRaises(ValueError):
            assign_cross_fit_folds(np.arange(3), n_folds=4, seed=0)


class ReadoutTests(unittest.TestCase):
    """The neighbourhood-only readout must genuinely drop the self-contribution."""

    def setUp(self) -> None:
        torch.manual_seed(0)
        self.input_dim = 6
        self.batch = 4
        self.fan_out = 3
        self.x_target = torch.randn(self.batch, self.input_dim)
        self.x_hop1 = torch.randn(self.batch, self.fan_out, self.input_dim)
        self.hop1_mask = torch.ones(self.batch, self.fan_out, dtype=torch.bool)
        self.x_hop2 = torch.randn(self.batch, self.fan_out, self.fan_out, self.input_dim)
        self.hop2_mask = torch.ones(self.batch, self.fan_out, self.fan_out, dtype=torch.bool)

    def encode(self, encoder: TemporalGraphSAGEEncoder, x_target: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return encoder(x_target, self.x_hop1, self.hop1_mask, self.x_hop2, self.hop2_mask)

    def test_neighbourhood_only_output_ignores_the_target_features(self) -> None:
        encoder = TemporalGraphSAGEEncoder(
            self.input_dim, HIDDEN_DIM, EMBEDDING_DIM, READOUT_NEIGHBOURHOOD_ONLY
        )
        baseline = self.encode(encoder, self.x_target)
        perturbed = self.encode(encoder, self.x_target + 100.0)
        torch.testing.assert_close(baseline, perturbed)

    def test_the_default_readout_does_depend_on_the_target_features(self) -> None:
        encoder = TemporalGraphSAGEEncoder(
            self.input_dim, HIDDEN_DIM, EMBEDDING_DIM, READOUT_SELF_AND_NEIGHBOURHOOD
        )
        baseline = self.encode(encoder, self.x_target)
        perturbed = self.encode(encoder, self.x_target + 100.0)
        self.assertFalse(torch.allclose(baseline, perturbed))

    def test_readout_mode_sets_the_final_layer_width(self) -> None:
        default = TemporalGraphSAGEEncoder(self.input_dim, HIDDEN_DIM, EMBEDDING_DIM)
        neighbourhood = TemporalGraphSAGEEncoder(
            self.input_dim, HIDDEN_DIM, EMBEDDING_DIM, READOUT_NEIGHBOURHOOD_ONLY
        )
        self.assertEqual(default.layer2.in_features, self.input_dim + HIDDEN_DIM)
        self.assertEqual(neighbourhood.layer2.in_features, HIDDEN_DIM)

    def test_an_unknown_readout_is_rejected_at_construction(self) -> None:
        with self.assertRaises(ValueError):
            TemporalGraphSAGEEncoder(self.input_dim, HIDDEN_DIM, EMBEDDING_DIM, "self_only")

    def test_the_head_wrapper_forwards_the_readout_mode(self) -> None:
        model = GraphSAGEWithHead(
            self.input_dim, HIDDEN_DIM, EMBEDDING_DIM, READOUT_NEIGHBOURHOOD_ONLY
        )
        self.assertEqual(model.encoder.readout, READOUT_NEIGHBOURHOOD_ONLY)
        self.assertEqual(model.encoder.layer2.in_features, HIDDEN_DIM)

    def test_the_default_readout_is_what_the_frozen_encoder_used(self) -> None:
        model = GraphSAGEWithHead(self.input_dim, HIDDEN_DIM, EMBEDDING_DIM)
        self.assertEqual(model.encoder.readout, READOUT_SELF_AND_NEIGHBOURHOOD)


class InferenceBatchSizeTests(unittest.TestCase):
    """Batch size must be a memory knob only, never a result-changing one.

    The controls lowered the inference batch size to keep peak allocation off
    this machine's memory ceiling, and `extended_budget` was extracted before
    that change. Comparing it against the later variants is only legitimate if
    batching cannot move which neighbours a target aggregates -- so that is
    checked here rather than argued.

    The sampled neighbourhood is bit-identical, because targets are visited in
    the same global order with the same generator and each target consumes the
    same draws regardless of which batch it lands in. The resulting embeddings
    agree only to float32 tolerance: torch's batched Linear reduces in an order
    that depends on tensor shape, which is the same residual nondeterminism the
    frozen encoder's metadata already records. The distinction matters -- a
    changed neighbourhood would be a different experiment, a 3e-07 rounding
    difference is not.
    """

    fan_outs = (4, 4)

    def build_index(self):
        rows = [
            {"entity_id": entity_id, "node_id": entity_id * 12 + k, "TransactionDT": float(k * 100)}
            for entity_id in range(6)
            for k in range(12)
        ]
        edges = pd.DataFrame(rows).sort_values(
            ["entity_id", "TransactionDT", "node_id"], kind="mergesort"
        ).reset_index(drop=True)
        return build_temporal_graph_index(edges)

    def sample_in_batches(self, index, node_ids, batch_size, seed):
        rng = np.random.default_rng(seed)
        hop1, hop2 = [], []
        for start in range(0, len(node_ids), batch_size):
            ids1, _, ids2, _ = sample_batch_neighborhoods(
                index, node_ids[start : start + batch_size], self.fan_outs, rng
            )
            hop1.append(ids1)
            hop2.append(ids2)
        return np.concatenate(hop1), np.concatenate(hop2)

    def test_sampled_neighbourhoods_are_bit_identical_across_batch_sizes(self) -> None:
        index = self.build_index()
        node_ids = np.arange(72)
        big_hop1, big_hop2 = self.sample_in_batches(index, node_ids, 64, seed=11)
        small_hop1, small_hop2 = self.sample_in_batches(index, node_ids, 8, seed=11)
        np.testing.assert_array_equal(big_hop1, small_hop1)
        np.testing.assert_array_equal(big_hop2, small_hop2)

    def test_a_different_seed_does_move_the_sampled_neighbourhood(self) -> None:
        """Guards the test above against passing for a trivial reason."""
        index = self.build_index()
        node_ids = np.arange(72)
        first, _ = self.sample_in_batches(index, node_ids, 16, seed=11)
        second, _ = self.sample_in_batches(index, node_ids, 16, seed=12)
        self.assertFalse(np.array_equal(first, second))

    def test_embeddings_agree_to_float32_tolerance_across_batch_sizes(self) -> None:
        index = self.build_index()
        input_dim = 5
        feature_cache = np.random.default_rng(0).normal(size=(72, input_dim)).astype(np.float32)
        node_ids = np.arange(72)

        torch.manual_seed(3)
        model = GraphSAGEWithHead(input_dim, HIDDEN_DIM, EMBEDDING_DIM)
        large = run_inference(
            model, index, feature_cache, node_ids, self.fan_outs, np.random.default_rng(11), 64
        )
        small = run_inference(
            model, index, feature_cache, node_ids, self.fan_outs, np.random.default_rng(11), 8
        )
        np.testing.assert_allclose(large, small, rtol=0.0, atol=1e-6)


class EmbeddingFrameTests(unittest.TestCase):
    def make_meta(self, n: int = 5) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "node_id": np.arange(n),
                "TransactionID": np.arange(100, 100 + n),
                "split": pd.array(["train"] * n, dtype="string"),
            }
        )

    def test_shape_mismatch_is_rejected(self) -> None:
        meta = self.make_meta()
        with self.assertRaises(AssertionError):
            embedding_frame(meta, np.zeros((len(meta) - 1, EMBEDDING_DIM), dtype=np.float32))

    def test_non_finite_embeddings_are_rejected(self) -> None:
        meta = self.make_meta()
        embeddings = np.zeros((len(meta), EMBEDDING_DIM), dtype=np.float32)
        embeddings[2, 0] = np.nan
        with self.assertRaises(AssertionError):
            embedding_frame(meta, embeddings)


class ShuffledEmbeddingTests(unittest.TestCase):
    def make_frame(self, per_split: int = 40) -> tuple[pd.DataFrame, list[str]]:
        feat_names = [f"embedding_{i:02d}" for i in range(4)]
        splits = np.repeat(["train", "validation", "test"], per_split)
        rng = np.random.default_rng(0)
        block = rng.normal(size=(len(splits), len(feat_names)))
        # Give each split a distinct location so a cross-split leak is visible.
        offsets = {"train": 0.0, "validation": 10.0, "test": 20.0}
        block = block + np.array([offsets[s] for s in splits])[:, None]
        frame = pd.DataFrame(
            {
                "TransactionID": np.arange(len(splits)),
                "split": pd.array(splits, dtype="string"),
                **{name: block[:, i] for i, name in enumerate(feat_names)},
            }
        )
        return frame, feat_names

    def test_identifiers_and_splits_stay_in_place(self) -> None:
        frame, feat_names = self.make_frame()
        shuffled = permute_block_within_split(frame, feat_names, seed=1)
        pd.testing.assert_series_equal(shuffled["TransactionID"], frame["TransactionID"])
        pd.testing.assert_series_equal(shuffled["split"], frame["split"])

    def test_per_split_marginals_are_preserved_exactly(self) -> None:
        frame, feat_names = self.make_frame()
        shuffled = permute_block_within_split(frame, feat_names, seed=1)
        for split_name in ("train", "validation", "test"):
            mask = (frame["split"] == split_name).to_numpy()
            np.testing.assert_array_equal(
                np.sort(frame.loc[mask, feat_names].to_numpy(), axis=0),
                np.sort(shuffled.loc[mask, feat_names].to_numpy(), axis=0),
            )

    def test_no_row_crosses_a_split_boundary(self) -> None:
        """A global permutation would mix train-fitted rows into validation."""
        frame, feat_names = self.make_frame()
        shuffled = permute_block_within_split(frame, feat_names, seed=1)
        for split_name, low, high in (
            ("train", -5.0, 5.0),
            ("validation", 5.0, 15.0),
            ("test", 15.0, 25.0),
        ):
            mask = (frame["split"] == split_name).to_numpy()
            values = shuffled.loc[mask, feat_names].to_numpy()
            self.assertTrue(((values > low) & (values < high)).all())

    def test_alignment_is_actually_destroyed(self) -> None:
        frame, feat_names = self.make_frame()
        shuffled = permute_block_within_split(frame, feat_names, seed=1)
        moved = (shuffled[feat_names].to_numpy() != frame[feat_names].to_numpy()).any(axis=1)
        self.assertGreater(moved.mean(), 0.5)

    def test_whole_rows_move_together(self) -> None:
        """The block keeps its internal correlation structure; only alignment goes."""
        frame, feat_names = self.make_frame()
        shuffled = permute_block_within_split(frame, feat_names, seed=1)
        original_rows = {tuple(row) for row in frame[feat_names].to_numpy()}
        for row in shuffled[feat_names].to_numpy():
            self.assertIn(tuple(row), original_rows)

    def test_permutation_is_deterministic_for_a_seed(self) -> None:
        frame, feat_names = self.make_frame()
        first = permute_block_within_split(frame, feat_names, seed=5)
        second = permute_block_within_split(frame, feat_names, seed=5)
        other = permute_block_within_split(frame, feat_names, seed=6)
        pd.testing.assert_frame_equal(first, second)
        self.assertFalse(first[feat_names].equals(other[feat_names]))

    def test_unexpected_split_counts_are_rejected(self) -> None:
        frame, feat_names = self.make_frame(per_split=40)
        with self.assertRaises(ValueError):
            permute_block_within_split(
                frame,
                feat_names,
                seed=1,
                expected_split_counts={"train": 40, "validation": 39, "test": 40},
            )

    def test_missing_columns_are_rejected(self) -> None:
        frame, feat_names = self.make_frame()
        with self.assertRaises(ValueError):
            permute_block_within_split(frame.drop(columns="split"), feat_names, seed=1)
        with self.assertRaises(ValueError):
            permute_block_within_split(frame, [*feat_names, "embedding_99"], seed=1)


class ParityRuleTests(unittest.TestCase):
    @staticmethod
    def block(lower: float, upper: float) -> dict[str, float | bool]:
        return {
            "ci_lower_95": lower,
            "ci_upper_95": upper,
            "excludes_zero": lower > 0.0 or upper < 0.0,
            "observed_delta": (lower + upper) / 2.0,
        }

    def test_an_interval_entirely_below_zero_is_not_parity(self) -> None:
        self.assertFalse(reaches_parity(self.block(-0.030, -0.010)))

    def test_an_interval_spanning_zero_counts_as_parity(self) -> None:
        self.assertTrue(reaches_parity(self.block(-0.010, 0.004)))

    def test_an_interval_entirely_above_zero_counts_as_parity(self) -> None:
        self.assertTrue(reaches_parity(self.block(0.002, 0.020)))


class VerdictTests(unittest.TestCase):
    @staticmethod
    def result(lower: float, upper: float, pr_auc: float = 0.64) -> dict:
        return {
            "metrics": {"pr_auc": pr_auc},
            "significance": {
                "control_vs_b1_card1": ParityRuleTests.block(lower, upper),
            },
            "metadata": {},
        }

    def test_both_controls_at_parity_blames_the_encoder_setup(self) -> None:
        results = {name: self.result(-0.004, 0.006) for name in VERDICT_CONTROLS}
        verdict = build_verdict(results)
        self.assertTrue(verdict["decidable"])
        self.assertEqual(verdict["verdict"], VERDICT_CONFOUND)

    def test_one_control_short_of_parity_leaves_the_negative_result_standing(self) -> None:
        results = {name: self.result(-0.004, 0.006) for name in VERDICT_CONTROLS}
        results[VERDICT_CONTROLS[0]] = self.result(-0.035, -0.012)
        verdict = build_verdict(results)
        self.assertTrue(verdict["decidable"])
        self.assertEqual(verdict["verdict"], VERDICT_NEGATIVE_STANDS)

    def test_a_missing_verdict_control_is_not_decidable(self) -> None:
        results = {VERDICT_CONTROLS[0]: self.result(-0.004, 0.006)}
        verdict = build_verdict(results)
        self.assertFalse(verdict["decidable"])
        self.assertIsNone(verdict["verdict"])

    def test_the_shuffled_null_does_not_enter_the_verdict(self) -> None:
        self.assertNotIn(SHUFFLED_VARIANT_NAME, VERDICT_CONTROLS)


class PartitionAlignmentTests(unittest.TestCase):
    """The diagnostic that catches a feature block whose columns mean two things."""

    def write_block(self, directory: str, shift: float) -> Path:
        feat_names = embedding_feature_names()
        rng = np.random.default_rng(0)
        per_split = 200
        splits = np.repeat(["train", "validation", "test"], per_split)
        values = rng.normal(size=(len(splits), len(feat_names)))
        # Displace only the train partition, the way unaligned fold encoders do.
        values[splits == "train"] += shift
        frame = pd.DataFrame(
            {
                "TransactionID": np.arange(len(splits)),
                "split": pd.array(splits, dtype="string"),
                **{name: values[:, i] for i, name in enumerate(feat_names)},
            }
        )
        path = Path(directory) / "block.parquet"
        frame.to_parquet(path, index=False, engine="pyarrow")
        return path

    def test_an_aligned_block_reports_a_small_gap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = embedding_partition_alignment(self.write_block(directory, shift=0.0))
        self.assertLess(result["mean_gap"], 0.3)
        self.assertEqual(result["columns_above_threshold"], 0)
        self.assertEqual(result["column_count"], EMBEDDING_DIM)

    def test_a_displaced_train_partition_is_flagged(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = embedding_partition_alignment(self.write_block(directory, shift=1.0))
        self.assertGreater(result["mean_gap"], 0.5)
        self.assertEqual(result["columns_above_threshold"], EMBEDDING_DIM)


class KnownLimitationTests(unittest.TestCase):
    @staticmethod
    def cross_fitted_result(mean_gap: float, above: int) -> dict:
        return {
            "alignment": {
                "mean_gap": mean_gap,
                "max_gap": mean_gap + 0.2,
                "threshold": 0.5,
                "columns_above_threshold": above,
                "column_count": EMBEDDING_DIM,
            }
        }

    def test_the_cross_fit_basis_defect_is_published(self) -> None:
        limitations = build_known_limitations(
            {"cross_fitted": self.cross_fitted_result(0.390, 11)}
        )
        self.assertEqual(len(limitations), 1)
        entry = limitations[0]
        self.assertEqual(entry["control"], "cross_fitted")
        self.assertIn("0.390", entry["evidence"])
        self.assertIn("11 of 32", entry["evidence"])
        self.assertTrue(entry["does_not_affect_verdict"])
        self.assertTrue(entry["remedy"])

    def test_no_limitation_is_invented_when_the_control_is_absent(self) -> None:
        self.assertEqual(build_known_limitations({}), [])


class ControlRegistryTests(unittest.TestCase):
    def test_every_encoder_variant_has_a_downstream_control(self) -> None:
        for name in ENCODER_VARIANTS:
            self.assertIn(name, CONTROL_RUNS)
            self.assertEqual(
                CONTROL_RUNS[name].embeddings_path, ENCODER_VARIANTS[name].embeddings_path
            )

    def test_the_reporting_order_covers_every_control_exactly_once(self) -> None:
        self.assertEqual(sorted(CONTROL_ORDER), sorted(CONTROL_RUNS))
        self.assertEqual(len(CONTROL_ORDER), len(set(CONTROL_ORDER)))

    def test_controls_write_to_distinct_directories(self) -> None:
        report_dirs = {control.report_dir for control in CONTROL_RUNS.values()}
        model_paths = {control.model_path for control in CONTROL_RUNS.values()}
        self.assertEqual(len(report_dirs), len(CONTROL_RUNS))
        self.assertEqual(len(model_paths), len(CONTROL_RUNS))

    def test_no_control_output_path_is_a_protected_artifact(self) -> None:
        protected = {
            path.resolve()
            for path in (*B0_PROTECTED_PATHS, *B1_CARD1_PROTECTED_PATHS, *G1_PROTECTED_PATHS)
        }
        for control in CONTROL_RUNS.values():
            outputs = {
                path.resolve()
                for key, path in control.paths().items()
                if key != "report_dir"
            }
            outputs.add(control.embeddings_path.resolve())
            self.assertEqual(outputs & protected, set())

    def test_the_frozen_g1_run_is_protected_from_the_controls(self) -> None:
        protected_names = {path.name for path in G1_PROTECTED_PATHS}
        self.assertIn("lightgbm_g1_card1.txt", protected_names)
        self.assertIn("metrics.json", protected_names)
        self.assertIn("validation_predictions.parquet", protected_names)
        self.assertIn(FROZEN_EMBEDDINGS_PATH, G1_PROTECTED_PATHS)

    def test_controls_keep_the_frozen_embedding_column_manifest(self) -> None:
        self.assertEqual(len(embedding_feature_names()), EMBEDDING_DIM)


if __name__ == "__main__":
    unittest.main()
