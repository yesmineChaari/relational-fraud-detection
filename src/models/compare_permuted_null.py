"""Compare the card1 permuted-entity null control against B0 and B1-card1.

Applies the interpretation the ticket for this control commits to in advance:

  * Permuted variants land at or below B0 -> the B1-card1 gain is attributable
    to genuine entity history; the correlational result becomes causal.
  * Permuted variants recover a meaningful share of the gain -> the effect is
    substantially structural, and the documented conclusion and the G1
    rationale need revisiting.

Both references are the converged (cap 15,000, average_precision patience,
seed 42) artifacts, not the stale cap-6,000 frozen ones -- comparing a
permuted run against the capped B0 would reintroduce the early-stopping
confound the estimator-cap investigation removed.

"Meaningful share of the gain" is operationalised the same way every other
significance question in this project is answered: the paired-bootstrap 95%
CI on the permuted-minus-B0 delta. If that CI excludes zero on the positive
side for any permutation seed, the permuted variant significantly beats B0,
which is direct evidence the effect is not (only) genuine entity history.

Outputs:
  reports/permuted_null/permuted_null_comparison.csv
  reports/permuted_null/permuted_null_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.significance import compare_variants
from src.models.train_lightgbm_permuted_null import (
    B0_CONVERGED_PATHS,
    B1_CARD1_CONVERGED_PATHS,
    REPORT_DIR,
    resolve_run_paths,
)

COMPARISON_CSV = REPORT_DIR.parent / "permuted_null_comparison.csv"
SUMMARY_JSON = REPORT_DIR.parent / "permuted_null_summary.json"

OUTCOME_AT_OR_BELOW_B0 = "PERMUTED_AT_OR_BELOW_B0"
OUTCOME_RECOVERS_GAIN = "PERMUTED_RECOVERS_GAIN"

VERDICT_RULE = (
    "Fixed before any permuted result was seen. 'Recovers a meaningful "
    "share of the gain' means a permuted variant significantly beats B0: "
    "its paired-bootstrap 95% CI on the permuted-minus-B0 delta excludes "
    "zero on the positive side. This is the same significance test every "
    "other comparison in this project is settled by; no recovered-fraction "
    "threshold is applied, so the fraction is reported for magnitude only."
)

CONCLUSION_AT_OR_BELOW_B0 = (
    "Entity history confirmed real: no permuted variant significantly "
    "beats B0. The B1-card1 gain is attributable to genuine card1 entity "
    "history, not to the split-search artifact this control was designed "
    "to rule out. The correlational result becomes causal."
)

CONCLUSION_RECOVERS_GAIN = (
    "Structural effect detected: at least one permuted variant "
    "significantly beats B0 (95% CI on the delta excludes zero on the "
    "positive side). Part of the B1-card1 gain is attributable to adding "
    "four columns to the split search rather than to entity history. The "
    "documented B1-card1 conclusion and the G1 rationale need revisiting."
)


def classify_permuted_outcome(comparison_df: pd.DataFrame) -> dict[str, Any]:
    """Apply the pre-registered verdict rule to a completed comparison table.

    The gate is the confidence interval, not the sign of the point estimate.
    A permuted run sitting a few 1e-05 above B0 is still "at or below B0" in
    the sense the interpretation means -- indistinguishable from it -- and
    gating on delta <= 0 would let noise in the last decimal decide the
    verdict on a quantity whose real gain is two orders of magnitude larger.
    A CI excluding zero on the *negative* side is a permuted variant losing
    to B0, which is evidence for the null, not against it.
    """
    if comparison_df.empty:
        raise ValueError("No permuted runs to classify.")

    delta_vs_b0 = comparison_df["delta_pr_auc_vs_b0"].to_numpy()
    any_significantly_beats_b0 = bool(
        (comparison_df["delta_pr_auc_vs_b0_excludes_zero"] & (delta_vs_b0 > 0)).any()
    )
    all_at_or_below_b0 = bool((delta_vs_b0 <= 0).all())

    if any_significantly_beats_b0:
        outcome_code = OUTCOME_RECOVERS_GAIN
        conclusion = CONCLUSION_RECOVERS_GAIN
    else:
        outcome_code = OUTCOME_AT_OR_BELOW_B0
        conclusion = CONCLUSION_AT_OR_BELOW_B0

    return {
        "any_permuted_variant_significantly_beats_b0": any_significantly_beats_b0,
        "all_permuted_variants_at_or_below_b0": all_at_or_below_b0,
        "outcome_code": outcome_code,
        "conclusion": conclusion,
    }


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def spread(values: np.ndarray) -> dict[str, float]:
    if len(values) < 2:
        return {
            "n": int(len(values)),
            "mean": float(values.mean()) if len(values) else float("nan"),
            "std": None,
            "min": float(values.min()) if len(values) else float("nan"),
            "max": float(values.max()) if len(values) else float("nan"),
        }
    return {
        "n": int(len(values)),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def discover_runs() -> list[int]:
    """Collect every completed permutation run from the report tree.

    Globs rather than iterating the default seed panel, so a run started with
    a --permutation-seed outside that panel is reported rather than silently
    omitted from a summary that still claims to cover the panel.
    """
    completed = []
    for run_dir in sorted(REPORT_DIR.glob("permseed_*")):
        try:
            seed = int(run_dir.name.removeprefix("permseed_"))
        except ValueError:
            continue
        paths = resolve_run_paths(seed)
        if paths["metrics"].exists() and paths["validation_predictions"].exists():
            completed.append(seed)
    if not completed:
        raise FileNotFoundError(
            f"No completed permuted-null runs found under {REPORT_DIR}. "
            "Run: python -m src.models.train_lightgbm_permuted_null"
        )
    return completed


def main() -> None:
    completed_seeds = discover_runs()
    print(f"Completed permutation seeds: {completed_seeds}")

    b0_metrics = load_metrics(B0_CONVERGED_PATHS["metrics"])
    b1_card1_metrics = load_metrics(B1_CARD1_CONVERGED_PATHS["metrics"])
    b0_pr_auc = float(b0_metrics["pr_auc"])
    b1_card1_pr_auc = float(b1_card1_metrics["pr_auc"])
    real_gain = b1_card1_pr_auc - b0_pr_auc
    if real_gain == 0.0:
        raise ValueError(
            "The converged B0 and B1-card1 references report identical PR-AUC; "
            "there is no gain for this null control to be measured against."
        )
    print(f"B0 converged PR-AUC:        {b0_pr_auc:.12f}")
    print(f"B1-card1 converged PR-AUC:  {b1_card1_pr_auc:.12f}")
    print(f"Real relational gain:       {real_gain:+.12f}")

    rows: list[dict[str, Any]] = []
    for seed in completed_seeds:
        paths = resolve_run_paths(seed)
        metrics = load_metrics(paths["metrics"])
        pr_auc = float(metrics["pr_auc"])

        vs_b0 = compare_variants(
            candidate_label=f"permuted_card1_seed{seed}",
            candidate_predictions_path=paths["validation_predictions"],
            reference_label="b0_converged",
            reference_predictions_path=B0_CONVERGED_PATHS["validation_predictions"],
        )
        vs_b1_card1 = compare_variants(
            candidate_label=f"permuted_card1_seed{seed}",
            candidate_predictions_path=paths["validation_predictions"],
            reference_label="b1_card1_converged",
            reference_predictions_path=B1_CARD1_CONVERGED_PATHS["validation_predictions"],
        )

        recovered_fraction = (pr_auc - b0_pr_auc) / real_gain
        row = {
            "permutation_seed": seed,
            "pr_auc": pr_auc,
            "roc_auc": float(metrics["roc_auc"]),
            "best_iteration": int(metrics["best_iteration"]),
            "estimator_cap_reached": bool(metrics["estimator_cap_reached"]),
            "delta_pr_auc_vs_b0": pr_auc - b0_pr_auc,
            "delta_pr_auc_vs_b0_ci_lower_95": vs_b0["ci_lower_95"],
            "delta_pr_auc_vs_b0_ci_upper_95": vs_b0["ci_upper_95"],
            "delta_pr_auc_vs_b0_excludes_zero": vs_b0["excludes_zero"],
            "delta_pr_auc_vs_b1_card1": pr_auc - b1_card1_pr_auc,
            "delta_pr_auc_vs_b1_card1_ci_lower_95": vs_b1_card1["ci_lower_95"],
            "delta_pr_auc_vs_b1_card1_ci_upper_95": vs_b1_card1["ci_upper_95"],
            "delta_pr_auc_vs_b1_card1_excludes_zero": vs_b1_card1["excludes_zero"],
            "recovered_fraction_of_real_gain": recovered_fraction,
        }
        rows.append(row)
        print(
            f"  seed {seed}: PR-AUC={pr_auc:.6f}  delta_vs_b0={row['delta_pr_auc_vs_b0']:+.6f}  "
            f"CI=[{vs_b0['ci_lower_95']:+.5f}, {vs_b0['ci_upper_95']:+.5f}]  "
            f"excludes_zero={vs_b0['excludes_zero']}  "
            f"recovered_fraction={recovered_fraction:+.3f}"
        )

    comparison_df = pd.DataFrame(rows)
    REPORT_DIR.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_CSV, index=False)
    print(f"\nComparison table saved: {COMPARISON_CSV}")

    delta_vs_b0 = comparison_df["delta_pr_auc_vs_b0"].to_numpy()
    outcome = classify_permuted_outcome(comparison_df)

    summary = {
        "report_name": "Permuted-entity null control for the B1-card1 gain",
        "relation": "card1",
        "reference_configuration": "converged (cap 15,000, stop_average_precision, seed 42)",
        "b0_converged_pr_auc": b0_pr_auc,
        "b1_card1_converged_pr_auc": b1_card1_pr_auc,
        "real_relational_gain_pr_auc": real_gain,
        "permutation_seeds": completed_seeds,
        "delta_pr_auc_vs_b0_spread": spread(delta_vs_b0),
        "recovered_fraction_spread": spread(
            comparison_df["recovered_fraction_of_real_gain"].to_numpy()
        ),
        "any_permuted_variant_significantly_beats_b0": outcome[
            "any_permuted_variant_significantly_beats_b0"
        ],
        "all_permuted_variants_at_or_below_b0": outcome["all_permuted_variants_at_or_below_b0"],
        "verdict_rule": VERDICT_RULE,
        "outcome": {
            "outcome_code": outcome["outcome_code"],
            "conclusion": outcome["conclusion"],
        },
        "runs": rows,
        "comparison_table_path": "reports/permuted_null/permuted_null_comparison.csv",
        "b0_reference_metrics_path": "reports/convergence_check/b0/stop_average_precision/cap15000_seed42/metrics.json",
        "b1_card1_reference_metrics_path": (
            "reports/convergence_check/b1_card1/stop_average_precision/cap15000_seed42/metrics.json"
        ),
        "test_evaluated": False,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(f"Summary saved: {SUMMARY_JSON}")
    print(f"\nOutcome: [{outcome['outcome_code']}] {outcome['conclusion']}")


if __name__ == "__main__":
    main()
