from __future__ import annotations

import re
import unittest

import numpy as np

from src.models.build_experiment_ledger import FAMILIES
from src.models.diagnose_g1v2_alignment import (
    CONTROL_SUMMARY_PATH,
    aligned_run,
    apply_alignment,
    fit_affine_procrustes,
    single_encoder_alignment_ceiling,
)
from src.models.train_lightgbm_g1v2 import encoder_backed_run
from src.models.train_lightgbm_g1v2_controls import shuffled_run


def random_orthogonal(dim: int, seed: int) -> np.ndarray:
    q, r = np.linalg.qr(np.random.default_rng(seed).standard_normal((dim, dim)))
    return q * np.sign(np.diag(r))


class ProcrustesTests(unittest.TestCase):
    def test_recovers_a_known_rotation_and_shift_exactly(self) -> None:
        rng = np.random.default_rng(0)
        source = rng.standard_normal((500, 8))
        rotation = random_orthogonal(8, seed=1)
        shift = rng.standard_normal(8)
        target = source @ rotation + shift
        alignment = fit_affine_procrustes(source, target)
        np.testing.assert_allclose(apply_alignment(source, alignment), target, atol=1e-10)
        np.testing.assert_allclose(alignment["rotation"], rotation, atol=1e-10)

    def test_the_map_is_orthogonal_under_noise(self) -> None:
        rng = np.random.default_rng(2)
        source = rng.standard_normal((400, 6))
        target = source @ random_orthogonal(6, seed=3) + 0.3 * rng.standard_normal((400, 6))
        rotation = fit_affine_procrustes(source, target)["rotation"]
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(6), atol=1e-10)

    def test_alignment_never_increases_the_anchor_error(self) -> None:
        rng = np.random.default_rng(4)
        source = rng.standard_normal((300, 5))
        target = source @ random_orthogonal(5, seed=5) + 1.0 + 0.2 * rng.standard_normal((300, 5))
        alignment = fit_affine_procrustes(source, target)
        before = np.mean((source - target) ** 2)
        after = np.mean((apply_alignment(source, alignment) - target) ** 2)
        self.assertLess(after, before)

    def test_rejects_mismatched_shapes(self) -> None:
        with self.assertRaises(ValueError):
            fit_affine_procrustes(np.zeros((10, 4)), np.zeros((10, 5)))


class LayoutTests(unittest.TestCase):
    def test_the_aligned_run_never_shares_artifacts_with_the_published_runs(self) -> None:
        runs = [aligned_run(42), encoder_backed_run(42), shuffled_run(42)]
        for attribute in ("report_dir", "model_path", "embeddings_path"):
            self.assertEqual(len({getattr(run, attribute) for run in runs}), 3, attribute)

    def test_the_aligned_model_matches_exactly_one_ledger_family(self) -> None:
        path = aligned_run(42).model_path.relative_to(aligned_run(42).model_path.parents[1])
        matched = [f["family"] for f in FAMILIES if re.match(f["pattern"], path.as_posix())]
        self.assertEqual(matched, ["g1v2_graph_model"])

    @unittest.skipUnless(CONTROL_SUMMARY_PATH.exists(), "G1 control summary not generated.")
    def test_the_ceiling_is_read_from_single_encoder_blocks_only(self) -> None:
        ceiling = single_encoder_alignment_ceiling()
        # The published single-encoder blocks sit at 0.12-0.23; the unaligned
        # cross-fitted control (0.39) must not raise the ceiling.
        self.assertLess(ceiling, 0.3)
        self.assertGreater(ceiling, 0.1)


if __name__ == "__main__":
    unittest.main()
