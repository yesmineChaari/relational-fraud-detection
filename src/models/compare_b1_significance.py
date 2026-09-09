"""Paired-bootstrap significance for the B1 relational-feature comparisons.

Regenerates the three B1 comparisons quoted in the project narrative directly
from the persisted validation_predictions.parquet files already written by
each frozen trainer -- no retraining. Uses the shared paired-bootstrap module
(src/models/significance.py), the same algorithm and seed the G1 significance
report uses, so intervals across stages are computed identically.

Comparisons:
  B1-card1        vs B0              (the winning relation vs the tabular baseline)
  B1-card1_card2  vs B0              (the second screened relation vs the same baseline)
  B1-card1        vs B1-card1_card2  (head-to-head between the two screened relations)

Outputs:
  reports/b1/b1_significance.csv
  reports/b1/b1_significance.json
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.significance import DEFAULT_N_RESAMPLES, compare_variants
from src.models.train_lightgbm_baseline import RANDOM_SEED, write_json

ROOT_DIR = Path(__file__).resolve().parents[2]

B0_VALIDATION_PREDICTIONS_PATH = (
    ROOT_DIR / "reports" / "baseline" / "validation_predictions.parquet"
)
B1_CARD1_VALIDATION_PREDICTIONS_PATH = (
    ROOT_DIR / "reports" / "b1" / "card1" / "validation_predictions.parquet"
)
B1_CARD1_CARD2_VALIDATION_PREDICTIONS_PATH = (
    ROOT_DIR / "reports" / "b1" / "card1_card2" / "validation_predictions.parquet"
)

REPORT_DIR = ROOT_DIR / "reports" / "b1"
COMPARISON_CSV = REPORT_DIR / "b1_significance.csv"
SUMMARY_JSON = REPORT_DIR / "b1_significance.json"

# (result key, candidate label, candidate path, reference label, reference path)
COMPARISONS: list[tuple[str, str, Path, str, Path]] = [
    (
        "b1_card1_vs_b0",
        "b1_card1",
        B1_CARD1_VALIDATION_PREDICTIONS_PATH,
        "b0",
        B0_VALIDATION_PREDICTIONS_PATH,
    ),
    (
        "b1_card1_card2_vs_b0",
        "b1_card1_card2",
        B1_CARD1_CARD2_VALIDATION_PREDICTIONS_PATH,
        "b0",
        B0_VALIDATION_PREDICTIONS_PATH,
    ),
    (
        "b1_card1_vs_b1_card1_card2",
        "b1_card1",
        B1_CARD1_VALIDATION_PREDICTIONS_PATH,
        "b1_card1_card2",
        B1_CARD1_CARD2_VALIDATION_PREDICTIONS_PATH,
    ),
]


def build_significance(
    n_resamples: int = DEFAULT_N_RESAMPLES, seed: int = RANDOM_SEED
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for key, candidate_label, candidate_path, reference_label, reference_path in COMPARISONS:
        print(
            f"Running paired bootstrap ({candidate_label} vs {reference_label}, "
            f"{n_resamples:,} resamples)..."
        )
        results[key] = compare_variants(
            candidate_label,
            candidate_path,
            reference_label,
            reference_path,
            n_resamples=n_resamples,
            seed=seed,
        )

    return {
        "method": (
            "paired_bootstrap_pr_auc_delta: resample validation-row indices with "
            "replacement (identical indices applied to both score vectors per "
            "resample), n_resamples draws, 95% CI from the empirical percentiles "
            "of the resampled PR-AUC delta"
        ),
        "source": (
            "Regenerated from persisted validation_predictions.parquet files; no retraining."
        ),
        "comparisons": results,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_comparison_table(summary: dict[str, Any]) -> pd.DataFrame:
    rows = []
    for key, result in summary["comparisons"].items():
        rows.append(
            {
                "comparison": key,
                "candidate": result["candidate"],
                "reference": result["reference"],
                "observed_delta_pr_auc": result["observed_delta"],
                "ci_lower_95": result["ci_lower_95"],
                "ci_upper_95": result["ci_upper_95"],
                "excludes_zero": result["excludes_zero"],
                "bootstrap_std_delta": result["bootstrap_std_delta"],
                "n_resamples": result["n_resamples"],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    summary = build_significance()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    build_comparison_table(summary).to_csv(COMPARISON_CSV, index=False)
    write_json(SUMMARY_JSON, summary)

    print(f"Comparison saved: {COMPARISON_CSV}")
    print(f"Summary saved: {SUMMARY_JSON}")
    for key, result in summary["comparisons"].items():
        print(
            f"  {key}: delta={result['observed_delta']:+.5f}  "
            f"95% CI=[{result['ci_lower_95']:+.5f}, {result['ci_upper_95']:+.5f}]  "
            f"excludes_zero={result['excludes_zero']}"
        )


if __name__ == "__main__":
    main()
