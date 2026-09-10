"""Screen every published comparison for validation-argmax selection bias.

The reported PR-AUC of a run is the maximum of its validation curve. A run
continues because that curve keeps setting new maxima, so an arm that plateaus
slowly is granted more rounds and draws its maximum from more samples of a noisy
statistic. Where two arms of a comparison stopped far apart, the reported delta
is partly a reward for extra draws rather than for predictors.

This module applies the three estimators in `selection_bias` to every comparison
the project has published, and classifies each by a rule fixed before the panel
was read.

Classification rule
-------------------

1. **Unexposed** -- the argmax and matched deltas agree within the clean-stratum
   noise floor. Unequal selection cannot have moved this comparison, whatever
   the size of its delta, and the reported figure needs no correction.
2. **Exposed, direction holds** -- the two estimators differ by more than the
   noise floor, but the delta keeps its sign under all three. The magnitude is
   revised; the conclusion is not.
3. **Exposed, direction in doubt** -- the delta changes sign between estimators.
   The published reading rests on the argmax convention and cannot be confirmed
   from these artifacts alone.

The noise floor is read from the seed-variance panel's own summary rather than
restated here, so this module cannot quietly disagree with the quantity every
other comparison in the project is judged against.

Why the off-metric column matters. ROC-AUC is not the stopping criterion, so it
is not inflated by the argmax. Read at the *same* selected iteration it answers
the question the three estimators cannot: whether the extra rounds bought real
learning. A positive PR-AUC delta beside a negative ROC-AUC delta is the
signature of selection rather than signal.

Nothing here refits and nothing rewrites a published report. Outputs land under
reports/selection_bias/.

Outputs:
  reports/selection_bias/selection_bias_comparison.csv
  reports/selection_bias/selection_bias_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import ROOT_DIR
from src.models.selection_bias import (
    AVERAGE_PRECISION_COLUMN,
    DEFAULT_PLATEAU_WINDOW,
    compare_estimators,
)

REPORTS_DIR = ROOT_DIR / "reports"
OUTPUT_DIR = REPORTS_DIR / "selection_bias"
COMPARISON_CSV = OUTPUT_DIR / "selection_bias_comparison.csv"
SUMMARY_JSON = OUTPUT_DIR / "selection_bias_summary.json"

SEED_VARIANCE_SUMMARY_PATH = REPORTS_DIR / "seed_variance" / "seed_variance_summary.json"

CONVERGED = "cap15000_seed42"
CONVERGENCE_DIR = REPORTS_DIR / "convergence_check"
ABLATION_DIR = REPORTS_DIR / "ablation" / "card1"

B0_CONVERGED = CONVERGENCE_DIR / "b0" / "stop_average_precision" / CONVERGED
B1_CONVERGED = CONVERGENCE_DIR / "b1_card1" / "stop_average_precision" / CONVERGED
G1_CONVERGED = CONVERGENCE_DIR / "g1_card1" / "stop_average_precision" / CONVERGED

ABLATION_FEATURE_KEYS = [
    "prior_count",
    "prior_count_24h",
    "prior_count_7d",
    "time_since_previous_hours",
]
SEED_PANEL_SEEDS = [42, 202, 707, 1337, 2024]
SEED_PANEL_VARIANT = "loo_prior_count_24h"

UNEXPOSED = "UNEXPOSED"
EXPOSED_DIRECTION_HOLDS = "EXPOSED_DIRECTION_HOLDS"
EXPOSED_DIRECTION_IN_DOUBT = "EXPOSED_DIRECTION_IN_DOUBT"

CLASSIFICATION_RULE = (
    "Fixed before the panel was read. A comparison is 'unexposed' when the "
    "argmax and matched-budget deltas agree within the seed-variance panel's "
    "clean-stratum noise floor -- unequal selection cannot have moved it. "
    "Otherwise it is 'exposed', and the direction 'holds' when the delta keeps "
    "its sign under all three estimators, or is 'in doubt' when the sign "
    "changes between them."
)

ESTIMATOR_DECISION = (
    "Report the matched-budget delta alongside the argmax, and treat the pair "
    "as bounds rather than replacing one with the other. Truncating both arms "
    "to the shorter curve penalises the arm that legitimately wanted more "
    "rounds, so matched is a lower bound and argmax an upper bound on the same "
    "quantity. A comparison is only read as settled where the two agree within "
    "the noise floor, or where the plateau level and the off-metric both point "
    "the same way. This is deliberately weaker than picking a single estimator: "
    "these artifacts cannot separate 'trained longer because it was learning' "
    "from 'trained longer because its curve was noisier' without refits at a "
    "budget fixed in advance."
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_curve(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "learning_curve.csv"
    if not path.exists():
        raise FileNotFoundError(f"Learning curve not found: {path}")
    return pd.read_csv(path)


NOISE_FLOOR_KEY_PATH = ("early_stopping_stratification", "clean_delta_pr_auc", "std")


def load_noise_floor() -> tuple[float, str]:
    """The clean-stratum paired-delta standard deviation, from its own panel.

    Read at one fixed key path rather than searched for. The summary carries
    several standard deviations -- the contaminated stratum's is sixteen times
    larger -- so a permissive lookup could silently pick the wrong floor and
    make every comparison here look unexposed.
    """
    payload: Any = read_json(SEED_VARIANCE_SUMMARY_PATH)
    for key in NOISE_FLOOR_KEY_PATH:
        if not isinstance(payload, dict) or key not in payload:
            raise KeyError(
                f"{SEED_VARIANCE_SUMMARY_PATH} has no "
                f"{'.'.join(NOISE_FLOOR_KEY_PATH)}; the seed-variance summary "
                f"schema changed and this screen's noise floor must be re-sourced."
            )
        payload = payload[key]
    if not isinstance(payload, (int, float)) or float(payload) <= 0:
        raise ValueError(f"Noise floor must be a positive number; got {payload!r}.")
    return float(payload), str(SEED_VARIANCE_SUMMARY_PATH.relative_to(ROOT_DIR).as_posix())


def build_registry() -> list[dict[str, Any]]:
    """Every published comparison, as (label, reference, variant) run directories.

    Data-driven so a future stage is a registration rather than an edit to the
    classifier -- the same property the cross-stage comparison layer needs.
    """
    registry: list[dict[str, Any]] = [
        {
            "family": "headline",
            "comparison": "b1_card1_vs_b0",
            "published_delta": 0.006304055057609892,
            "reference_dir": B0_CONVERGED,
            "variant_dir": B1_CONVERGED,
        },
        {
            "family": "headline",
            "comparison": "g1_card1_vs_b0",
            "published_delta": -0.020925473252758398,
            "reference_dir": B0_CONVERGED,
            "variant_dir": G1_CONVERGED,
        },
    ]
    for key in ABLATION_FEATURE_KEYS:
        registry.append(
            {
                "family": "ablation_singleton",
                "comparison": f"singleton_{key}_vs_b0",
                "published_delta": None,
                "reference_dir": B0_CONVERGED,
                "variant_dir": ABLATION_DIR / f"singleton_{key}" / CONVERGED,
            }
        )
        registry.append(
            {
                "family": "ablation_loo",
                "comparison": f"loo_{key}_vs_b1_card1",
                "published_delta": None,
                "reference_dir": B1_CONVERGED,
                "variant_dir": ABLATION_DIR / f"loo_{key}" / CONVERGED,
            }
        )
    for seed in SEED_PANEL_SEEDS:
        registry.append(
            {
                "family": "seed_panel",
                "comparison": f"{SEED_PANEL_VARIANT}_vs_b1_card1_seed{seed}",
                "published_delta": None,
                "reference_dir": (
                    CONVERGENCE_DIR / "b1_card1" / "stop_average_precision" / f"cap15000_seed{seed}"
                ),
                "variant_dir": ABLATION_DIR / SEED_PANEL_VARIANT / f"cap15000_seed{seed}",
            }
        )
    return registry


def classify(result: dict[str, Any]) -> str:
    if result["estimators_agree"]:
        return UNEXPOSED
    signs = {
        np.sign(result["delta_argmax"]),
        np.sign(result["delta_matched"]),
        np.sign(result["delta_plateau"]),
    }
    signs.discard(0.0)
    return EXPOSED_DIRECTION_HOLDS if len(signs) <= 1 else EXPOSED_DIRECTION_IN_DOUBT


def evaluate(
    noise_floor: float,
    plateau_window: int = DEFAULT_PLATEAU_WINDOW,
) -> pd.DataFrame:
    rows = []
    for entry in build_registry():
        reference = load_curve(entry["reference_dir"])
        variant = load_curve(entry["variant_dir"])
        result = compare_estimators(
            reference,
            variant,
            metric_column=AVERAGE_PRECISION_COLUMN,
            plateau_window=plateau_window,
            noise_floor=noise_floor,
        )
        rows.append(
            {
                "family": entry["family"],
                "comparison": entry["comparison"],
                "published_delta": entry["published_delta"],
                "classification": classify(result),
                **result,
            }
        )
    return pd.DataFrame(rows)


def summarize(table: pd.DataFrame, noise_floor: float, floor_source: str) -> dict[str, Any]:
    counts = table["classification"].value_counts().to_dict()
    exposed = table[table["classification"] != UNEXPOSED]
    off_metric_disagrees = table[
        (np.sign(table["delta_argmax"]) > 0) & (np.sign(table["off_metric_delta_at_selection"]) < 0)
    ]
    return {
        "report_name": "Validation-argmax selection-bias screen",
        "question": (
            "Which published comparisons are distorted by the two arms being "
            "granted unequal numbers of boosting rounds before their reported "
            "maximum is taken?"
        ),
        "clean_stratum_noise_floor": noise_floor,
        "noise_floor_source": floor_source,
        "plateau_window": DEFAULT_PLATEAU_WINDOW,
        "classification_rule": CLASSIFICATION_RULE,
        "estimator_decision": ESTIMATOR_DECISION,
        "n_comparisons": int(len(table)),
        "classification_counts": {str(k): int(v) for k, v in counts.items()},
        "unexposed_comparisons": sorted(
            table.loc[table["classification"] == UNEXPOSED, "comparison"]
        ),
        "exposed_comparisons": sorted(exposed["comparison"]),
        "direction_in_doubt_comparisons": sorted(
            table.loc[table["classification"] == EXPOSED_DIRECTION_IN_DOUBT, "comparison"]
        ),
        "off_metric_contradicts_stopping_metric": sorted(off_metric_disagrees["comparison"]),
        "round_gap_spread": {
            "min": int(table["absolute_round_gap"].min()),
            "max": int(table["absolute_round_gap"].max()),
            "median": float(table["absolute_round_gap"].median()),
        },
        "correlation_round_gap_vs_argmax_minus_matched": float(
            np.corrcoef(table["round_gap"], table["argmax_minus_matched"])[0, 1]
        ),
        "comparison_table_path": str(COMPARISON_CSV.relative_to(ROOT_DIR).as_posix()),
        "test_evaluated": False,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    noise_floor, floor_source = load_noise_floor()
    table = evaluate(noise_floor)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(COMPARISON_CSV, index=False)
    summary = summarize(table, noise_floor, floor_source)
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    pd.set_option("display.width", 220)
    print(f"Noise floor {noise_floor:.5f} (from {floor_source})\n")
    print(
        table[
            [
                "comparison",
                "round_gap",
                "delta_argmax",
                "delta_matched",
                "delta_plateau",
                "off_metric_delta_at_selection",
                "classification",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x: .5f}")
    )
    print(f"\nWrote {COMPARISON_CSV}")
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
