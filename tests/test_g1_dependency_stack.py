from __future__ import annotations

import unittest

import torch

from src.graph.environment_check import (
    FRAMEWORK_DECISION,
    run_minimal_forward_backward_pass,
)


class TorchInstallationTests(unittest.TestCase):
    def test_torch_version_matches_the_requirements_pin(self) -> None:
        self.assertTrue(torch.__version__.startswith("2.14.0"))

    def test_framework_decision_is_hand_rolled_pytorch(self) -> None:
        self.assertEqual(FRAMEWORK_DECISION, "hand_rolled_pytorch")


class ForwardBackwardPassTests(unittest.TestCase):
    def test_pass_produces_a_finite_scalar_loss(self) -> None:
        result = run_minimal_forward_backward_pass()

        self.assertEqual(result["forward_output_shape"], [8, 1])
        self.assertTrue(
            result["loss"] == result["loss"]  # not NaN
        )

    def test_pass_populates_a_nonzero_gradient(self) -> None:
        result = run_minimal_forward_backward_pass()

        self.assertGreater(result["w2_grad_norm"], 0.0)

    def test_pass_is_reproducible_under_the_fixed_seed(self) -> None:
        first = run_minimal_forward_backward_pass()
        second = run_minimal_forward_backward_pass()

        self.assertAlmostEqual(first["loss"], second["loss"], places=10)
        self.assertAlmostEqual(first["w2_grad_norm"], second["w2_grad_norm"], places=10)


if __name__ == "__main__":
    unittest.main()
