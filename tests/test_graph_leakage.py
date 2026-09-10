from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.graph.leakage_checks import (
    INFORMATIVENESS_GAP_TOLERANCE,
    MIN_POSITIVES_PER_SPLIT,
    PARITY_HOLDS,
    PARITY_VIOLATED,
    cross_fit_provenance,
    informativeness_gap,
    probe_auc,
)
from src.graph.temporal_sampler import (
    admissible_neighbors,
    build_temporal_graph_index,
    sample_k_hop,
)

ROOT_DIR = Path(__file__).resolve().parents[1]
CROSS_FIT_METADATA = ROOT_DIR / "reports" / "g1_controls" / "cross_fitted" / "encoder_metadata.json"

TRAIN_ROWS = 413_378
INFERENCE_ROWS = 177_162


def make_index(entities: dict[int, list[tuple[int, float]]]):
    rows = []
    for entity_id, members in entities.items():
        for node_id, dt in members:
            rows.append({"entity_id": entity_id, "node_id": node_id, "TransactionDT": dt})
    edges = (
        pd.DataFrame(rows)
        .sort_values(["entity_id", "TransactionDT", "node_id"], kind="mergesort")
        .reset_index(drop=True)
    )
    return build_temporal_graph_index(edges)


def synthetic_split(n=60_000, dim=8, prevalence=0.034, contamination=0.0, seed=0):
    """Embeddings for one split, optionally carrying each row's own label.

    `contamination` is how strongly a row's own label is written into its
    embedding — the thing a supervised encoder fitted through those rows could
    in principle leak, and the thing the parity check exists to detect.
    """
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < prevalence).astype(np.int8)
    embeddings = rng.normal(size=(n, dim))
    if contamination:
        embeddings[:, 0] += contamination * labels
    return embeddings, labels


class ProbeTests(unittest.TestCase):
    def test_probe_recovers_a_planted_signal(self):
        embeddings, labels = synthetic_split(contamination=3.0)
        self.assertGreater(probe_auc(embeddings, labels), 0.95)

    def test_probe_on_noise_is_near_chance(self):
        embeddings, labels = synthetic_split()
        self.assertLess(abs(probe_auc(embeddings, labels) - 0.5), 0.03)

    def test_too_few_positives_is_refused_rather_than_reported(self):
        # Below the floor, ordinary sampling scatter alone exceeds the tolerance.
        # Refusing is the only honest answer; a number here would be a false alarm.
        embeddings, labels = synthetic_split(n=4_000)
        with self.assertRaises(ValueError) as caught:
            probe_auc(embeddings, labels)
        self.assertIn("stable enough", str(caught.exception))

    def test_a_single_class_is_rejected(self):
        with self.assertRaises(ValueError):
            probe_auc(np.zeros((100, 4)), np.ones(100, dtype=np.int8))

    def test_mismatched_lengths_are_rejected(self):
        with self.assertRaises(ValueError):
            probe_auc(np.zeros((10, 4)), np.zeros(9, dtype=np.int8))

    def test_one_dimensional_embeddings_are_rejected(self):
        with self.assertRaises(ValueError):
            probe_auc(np.zeros(10), np.zeros(10, dtype=np.int8))


class ContaminationDetectionTests(unittest.TestCase):
    """The check must fail against a pipeline that actually leaks.

    Ground truth is controlled here rather than assumed: contamination is
    written into the train embeddings deliberately, so a check that failed to
    flag it would be worthless, and one that flags the clean case would be worse
    than nothing.
    """

    def test_contaminated_train_embeddings_are_flagged(self):
        train = synthetic_split(contamination=3.0, seed=1)
        validation = synthetic_split(seed=2)
        result = informativeness_gap(*train, *validation)
        self.assertFalse(result["parity_holds"])
        self.assertEqual(result["outcome_code"], PARITY_VIOLATED)
        self.assertGreater(result["gap"], 0.3)

    def test_clean_embeddings_pass(self):
        train = synthetic_split(seed=1)
        validation = synthetic_split(seed=2)
        result = informativeness_gap(*train, *validation)
        self.assertTrue(result["parity_holds"])
        self.assertEqual(result["outcome_code"], PARITY_HOLDS)
        self.assertLess(abs(result["gap"]), INFORMATIVENESS_GAP_TOLERANCE)

    def test_equally_informative_splits_pass(self):
        # Both splits carry the same real signal. That is a good representation,
        # not a leak, and must not be flagged.
        train = synthetic_split(contamination=3.0, seed=1)
        validation = synthetic_split(contamination=3.0, seed=2)
        result = informativeness_gap(*train, *validation)
        self.assertTrue(result["parity_holds"])

    def test_detection_strengthens_with_contamination(self):
        validation = synthetic_split(seed=2)
        gaps = [
            informativeness_gap(*synthetic_split(contamination=c, seed=1), *validation)["gap"]
            for c in (0.0, 0.5, 1.5, 3.0)
        ]
        self.assertEqual(gaps, sorted(gaps))
        self.assertGreater(gaps[-1], gaps[0])

    def test_the_tolerance_is_load_bearing(self):
        # Remove the guard by widening the tolerance and the contaminated case
        # passes. This is what the check would be worth without its threshold.
        train = synthetic_split(contamination=3.0, seed=1)
        validation = synthetic_split(seed=2)
        self.assertFalse(informativeness_gap(*train, *validation)["parity_holds"])
        self.assertTrue(informativeness_gap(*train, *validation, tolerance=0.9)["parity_holds"])

    def test_a_non_positive_tolerance_is_rejected(self):
        train = synthetic_split(seed=1)
        validation = synthetic_split(seed=2)
        with self.assertRaises(ValueError):
            informativeness_gap(*train, *validation, tolerance=0.0)

    def test_the_gap_is_directional(self):
        # Validation more informative than train is not the leak signature and
        # must not be reported as one.
        train = synthetic_split(seed=1)
        validation = synthetic_split(contamination=3.0, seed=2)
        result = informativeness_gap(*train, *validation)
        self.assertLess(result["gap"], 0)
        self.assertTrue(result["parity_holds"])


class CrossFitProvenanceTests(unittest.TestCase):
    def sound(self):
        return {
            "n_folds": 3,
            "fold_sizes": [40, 30, 30],
            "train_rows_embedded_by_held_out_encoder": 100,
            "inference_rows_embedded_by_full_train_encoder": 50,
        }

    def test_a_sound_record_passes(self):
        result = cross_fit_provenance(self.sound(), 100, 50)
        self.assertTrue(result["provenance_sound"])
        self.assertEqual(result["failures"], [])

    def test_absent_cross_fitting_is_not_sound(self):
        result = cross_fit_provenance(None, 100, 50)
        self.assertFalse(result["cross_fitted"])
        self.assertFalse(result["provenance_sound"])
        self.assertIn("own label", result["reason"])

    def test_folds_that_do_not_partition_the_train_rows_fail(self):
        record = self.sound()
        record["fold_sizes"] = [40, 30, 20]
        result = cross_fit_provenance(record, 100, 50)
        self.assertFalse(result["provenance_sound"])
        self.assertTrue(any("Fold sizes sum" in f for f in result["failures"]))

    def test_train_rows_not_all_embedded_by_a_held_out_encoder_fail(self):
        record = self.sound()
        record["train_rows_embedded_by_held_out_encoder"] = 99
        result = cross_fit_provenance(record, 100, 50)
        self.assertFalse(result["provenance_sound"])
        self.assertTrue(any("held-out encoder" in f for f in result["failures"]))

    def test_inference_rows_miscounted_fail(self):
        record = self.sound()
        record["inference_rows_embedded_by_full_train_encoder"] = 49
        self.assertFalse(cross_fit_provenance(record, 100, 50)["provenance_sound"])

    def test_a_single_fold_is_not_cross_fitting(self):
        record = self.sound()
        record["n_folds"] = 1
        record["fold_sizes"] = [100]
        self.assertFalse(cross_fit_provenance(record, 100, 50)["provenance_sound"])

    def test_fold_count_and_fold_sizes_must_agree(self):
        record = self.sound()
        record["fold_sizes"] = [50, 50]
        result = cross_fit_provenance(record, 100, 50)
        self.assertFalse(result["provenance_sound"])

    def test_the_published_cross_fitted_run_has_sound_provenance(self):
        if not CROSS_FIT_METADATA.exists():
            self.skipTest(f"Cross-fitted control metadata not present: {CROSS_FIT_METADATA}")
        metadata = json.loads(CROSS_FIT_METADATA.read_text(encoding="utf-8"))
        result = cross_fit_provenance(metadata.get("cross_fitting"), TRAIN_ROWS, INFERENCE_ROWS)
        self.assertTrue(result["provenance_sound"], result.get("failures"))
        self.assertGreaterEqual(result["n_folds"], 2)


class SingleTimestampEntityTests(unittest.TestCase):
    """An entity whose transactions all share one timestamp.

    The strictly-before rule makes every such neighbourhood empty. Worth its own
    fixture because it is the degenerate case where a rule written as
    less-than-or-equal would still look correct on ordinary data.
    """

    def setUp(self):
        self.index = make_index({0: [(node, 500.0) for node in range(6)]})

    def test_every_node_has_an_empty_admissible_neighbourhood(self):
        for node in range(6):
            self.assertEqual(len(admissible_neighbors(self.index, node)), 0)

    def test_sampling_yields_no_valid_neighbour_at_any_hop(self):
        rng = np.random.default_rng(0)
        for node in range(6):
            for _, mask in sample_k_hop(self.index, target_node_id=node, fan_outs=[4, 4], rng=rng):
                self.assertFalse(mask.any())

    def test_one_later_transaction_sees_all_the_tied_ones(self):
        # The complement: the guard must not be so strict that nothing is ever
        # admissible. A strictly later node sees every tied earlier node.
        index = make_index({0: [(node, 500.0) for node in range(6)] + [(6, 501.0)]})
        self.assertEqual(sorted(admissible_neighbors(index, 6).tolist()), [0, 1, 2, 3, 4, 5])
        self.assertEqual(len(admissible_neighbors(index, 0)), 0)


class GraphStageArtifactProtectionTests(unittest.TestCase):
    """Running the graph stage must leave the earlier stages byte-identical."""

    def test_the_graph_stage_protects_the_baseline_and_relational_artifacts(self):
        from src.models.train_lightgbm_g1 import (
            B0_PROTECTED_PATHS,
            B1_CARD1_PROTECTED_PATHS,
        )

        protected = {path.name for path in (*B0_PROTECTED_PATHS, *B1_CARD1_PROTECTED_PATHS)}
        self.assertIn("lightgbm_baseline.txt", protected)
        self.assertIn("lightgbm_b1_card1.txt", protected)
        self.assertIn("categorical_mappings.json", protected)

    def test_no_graph_stage_output_is_a_protected_path(self):
        from src.graph.train_graphsage_encoder import EMBEDDINGS_PATH, MODEL_PATH
        from src.models.train_lightgbm_g1 import (
            B0_PROTECTED_PATHS,
            B1_CARD1_PROTECTED_PATHS,
        )

        protected = {path.resolve() for path in (*B0_PROTECTED_PATHS, *B1_CARD1_PROTECTED_PATHS)}
        self.assertNotIn(MODEL_PATH.resolve(), protected)
        self.assertNotIn(EMBEDDINGS_PATH.resolve(), protected)

    def test_the_hash_snapshot_detects_a_changed_artifact(self):
        # The guard the stage relies on, exercised rather than assumed. The
        # scratch file lives inside the repository because the snapshot records
        # repository-relative paths and cannot describe anything outside it.
        import shutil
        import uuid

        from src.models.train_lightgbm_relational import (
            assert_protected_artifacts_unchanged,
            snapshot_protected_artifacts,
        )

        scratch = ROOT_DIR / f".protection_probe_{uuid.uuid4().hex}"
        scratch.mkdir()
        try:
            artifact = scratch / "frozen.txt"
            artifact.write_text("original", encoding="utf-8")
            before = snapshot_protected_artifacts([artifact])
            self.assertEqual(len(before), 1)
            assert_protected_artifacts_unchanged(before, [artifact], label="test")

            artifact.write_text("tampered", encoding="utf-8")
            with self.assertRaises(AssertionError):
                assert_protected_artifacts_unchanged(before, [artifact], label="test")

            artifact.unlink()
            with self.assertRaises(AssertionError):
                assert_protected_artifacts_unchanged(before, [artifact], label="test")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)


class RealEmbeddingParityTests(unittest.TestCase):
    """The parity check applied to the artifacts actually on disk.

    These record a measured result rather than asserting a hoped-for one. Both
    pipelines pass: no train/inference informativeness gap is detectable in
    either, which is a finding about the encoder's training budget rather than a
    weakness of the check — the contamination tests above show it detects a real
    leak when one exists.
    """

    EMBEDDINGS = {
        "frozen": ROOT_DIR / "data" / "processed" / "graphsage_card1_embeddings.parquet",
        "cross_fitted": (
            ROOT_DIR / "data" / "processed" / "graphsage_card1_embeddings_cross_fitted.parquet"
        ),
    }
    DATASET = ROOT_DIR / "data" / "processed" / "model_dataset.parquet"
    SUBSAMPLE = 60_000

    def gap_for(self, name):
        path = self.EMBEDDINGS[name]
        if not path.exists() or not self.DATASET.exists():
            self.skipTest("Embedding or dataset artifacts not present.")
        labels = pd.read_parquet(self.DATASET, columns=["TransactionID", "isFraud"])
        frame = pd.read_parquet(path).merge(labels, on="TransactionID", validate="one_to_one")
        columns = [c for c in frame.columns if c.startswith("embedding_")]
        rng = np.random.default_rng(0)
        parts = {}
        for split in ("train", "validation"):
            part = frame[frame["split"] == split]
            index = rng.choice(len(part), size=min(self.SUBSAMPLE, len(part)), replace=False)
            part = part.iloc[index]
            parts[split] = (
                part[columns].to_numpy(),
                part["isFraud"].to_numpy().astype("int8"),
            )
        return informativeness_gap(*parts["train"], *parts["validation"])

    def test_the_frozen_pipeline_shows_no_informativeness_gap(self):
        result = self.gap_for("frozen")
        self.assertTrue(
            result["parity_holds"],
            f"Frozen embeddings now show a gap of {result['gap']:+.4f}; the recorded "
            f"measurement was well inside tolerance.",
        )

    def test_the_cross_fitted_pipeline_shows_no_informativeness_gap(self):
        self.assertTrue(self.gap_for("cross_fitted")["parity_holds"])

    def test_both_splits_clear_the_positives_floor(self):
        if not self.DATASET.exists():
            self.skipTest("Dataset not present.")
        labels = pd.read_parquet(self.DATASET, columns=["isFraud", "split"])
        for split in ("train", "validation"):
            part = labels[labels["split"] == split]
            expected = int(part["isFraud"].sum() * self.SUBSAMPLE / len(part))
            self.assertGreater(expected, MIN_POSITIVES_PER_SPLIT)


if __name__ == "__main__":
    unittest.main()
