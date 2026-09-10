"""Paired-bootstrap intervals for the converged protocol.

The frozen artifacts have significance intervals; the converged ones did not.
Without them the cross-stage report can rank stages under the frozen protocol
but must report the converged protocol as undecided, which is the weaker half
of the evidence -- frozen G1 stopped at its cap without ever triggering early
stopping, so the converged figures are the ones worth quoting.

Nothing needs refitting. Every converged run persisted its validation
predictions, and the shared paired bootstrap compares two such files directly,
so this is a few seconds of resampling over predictions already on disk.

Deliberately the same estimator, resample count and seed as every other
significance artifact in the project, so a converged interval and a frozen one
mean the same thing and can be read side by side.

Outputs:
  reports/convergence_check/convergence_significance.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config.paths import ROOT_DIR
from src.models.significance import DEFAULT_N_RESAMPLES, compare_variants

CONVERGED_DIR = ROOT_DIR / "reports" / "convergence_check"
CONVERGED_RUN = "cap15000_seed42"
STOP_METRIC_DIR = "stop_average_precision"

OUTPUT_JSON = CONVERGED_DIR / "convergence_significance.json"


def predictions_path(model: str) -> Path:
    return (
        CONVERGED_DIR / model / STOP_METRIC_DIR / CONVERGED_RUN / "validation_predictions.parquet"
    )


# Every comparison the cross-stage report needs at the converged protocol.
COMPARISONS: list[tuple[str, str, str]] = [
    ("b1_card1_vs_b0", "b1_card1", "b0"),
    ("g1_card1_vs_b0", "g1_card1", "b0"),
    ("g1_card1_vs_b1_card1", "g1_card1", "b1_card1"),
]


def build() -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for key, candidate, reference in COMPARISONS:
        candidate_path = predictions_path(candidate)
        reference_path = predictions_path(reference)
        for path in (candidate_path, reference_path):
            if not path.exists():
                raise FileNotFoundError(
                    f"Converged validation predictions not found: {path}. Run "
                    f"python -m src.models.train_lightgbm_convergence_check first."
                )
        print(f"  bootstrapping {key}...", flush=True)
        comparisons[key] = compare_variants(
            candidate_label=candidate,
            candidate_predictions_path=candidate_path,
            reference_label=reference,
            reference_predictions_path=reference_path,
        )
    return {
        "report_name": "Paired-bootstrap intervals at the converged protocol",
        "protocol": "converged (cap 15,000, average_precision patience, seed 42)",
        "method": (
            "Identical estimator, resample count and seed to every other "
            "significance artifact in this project, so a converged interval and "
            "a frozen one mean the same thing."
        ),
        "n_resamples": DEFAULT_N_RESAMPLES,
        "comparisons": comparisons,
        "test_evaluated": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    payload = build()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    for key, result in payload["comparisons"].items():
        print(
            f"{key:28s} {result['observed_delta']:+.6f} "
            f"[{result['ci_lower_95']:+.6f}, {result['ci_upper_95']:+.6f}] "
            f"excludes_zero={result['excludes_zero']}"
        )
    print(f"\nWrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
