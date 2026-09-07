"""Unit tests for the B1 paired-bootstrap significance report."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.models import significance
from src.models.compare_b1_significance import build_comparison_table, build_significance


def _write_predictions(path: Path, n: int, seed: int, signal: float) -> None:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.1).astype(int)
    scores = labels.astype(np.float64) * signal + rng.random(n)
    df = pd.DataFrame(
        {"TransactionID": range(n), "isFraud": labels, "prediction": scores}
    )
    df.to_parquet(path, index=False)


class BuildSignificanceTests(unittest.TestCase):
    def test_three_required_comparisons_are_produced(self) -> None:
        n = 300
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            b0_path = root / "b0.parquet"
            card1_path = root / "card1.parquet"
            card1_card2_path = root / "card1_card2.parquet"
            _write_predictions(b0_path, n, seed=1, signal=0.5)
            _write_predictions(card1_path, n, seed=1, signal=0.8)
            _write_predictions(card1_card2_path, n, seed=1, signal=0.6)

            with patch.dict(significance.EXPECTED_SPLIT_COUNTS, {"validation": n}), patch(
                "src.models.compare_b1_significance.COMPARISONS",
                [
                    ("b1_card1_vs_b0", "b1_card1", card1_path, "b0", b0_path),
                    (
                        "b1_card1_card2_vs_b0",
                        "b1_card1_card2",
                        card1_card2_path,
                        "b0",
                        b0_path,
                    ),
                    (
                        "b1_card1_vs_b1_card1_card2",
                        "b1_card1",
                        card1_path,
                        "b1_card1_card2",
                        card1_card2_path,
                    ),
                ],
            ):
                result = build_significance(n_resamples=100)

        self.assertEqual(
            set(result["comparisons"]),
            {"b1_card1_vs_b0", "b1_card1_card2_vs_b0", "b1_card1_vs_b1_card1_card2"},
        )
        table = build_comparison_table(result)
        self.assertEqual(len(table), 3)
        self.assertIn("ci_lower_95", table.columns)
        self.assertIn("ci_upper_95", table.columns)
        self.assertIn("excludes_zero", table.columns)


if __name__ == "__main__":
    unittest.main()
