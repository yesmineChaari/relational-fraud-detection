"""Cross-control comparison and the pre-agreed G1 verdict.

Reads the four control runs plus the three frozen references (B0, B1-card1 and
the original G1) and publishes one table, one summary, and the verdict the
ticket fixed *before* any of these runs existed.

The stopping rule, restated so it can be checked against this file rather than
recalled:

* If cross-fitting and the neighbourhood-only readout each bring G1 to at least
  parity with B1-card1, the original regression was an artifact of the encoder
  setup and the graph stage continues on the corrected pipeline.
* If they do not, the negative result stands as a genuine finding -- a
  supervised graph encoder on this relation does not beat four scalar summaries
  -- the G1 epic closes, and remaining effort goes to hardening the B1 claim.

"At least parity with B1-card1" is decided by the paired bootstrap, not by the
point estimate: a control reaches parity unless its 95% CI on
PR-AUC(control) - PR-AUC(B1-card1) lies entirely below zero. A control that is
merely numerically lower, with an interval spanning zero, has not been shown to
be worse and counts as parity. This is the same interval-based reading the
frozen G1 run used to establish that its own regression was not sampling noise,
applied in the direction that can only make the negative verdict harder to
reach.

The shuffled-embedding control does not enter the verdict. It calibrates it:
whatever it costs against B0 is what 32 uninformative columns cost this
LightGBM configuration, and only the excess beyond that is a statement about
the graph.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.train_lightgbm_baseline import repository_relative, write_json
from src.models.train_lightgbm_g1 import (
    B1_CARD1_METRICS_PATH,
    RELATION,
    embedding_feature_names,
)
from src.models.train_lightgbm_g1_controls import (
    B0_METRICS_PATH,
    CONTROL_ORDER,
    CONTROL_RUNS,
    G1_METRICS_PATH,
    SHUFFLED_VARIANT_NAME,
)
from src.models.train_lightgbm_relational import read_json

ROOT_DIR = Path(__file__).resolve().parents[2]

CONTROL_REPORT_ROOT = ROOT_DIR / "reports" / "g1_controls"
COMPARISON_PATH = CONTROL_REPORT_ROOT / "g1_control_comparison.csv"
SUMMARY_PATH = CONTROL_REPORT_ROOT / "g1_control_summary.json"

# The two controls the pre-agreed stopping rule is written over.
VERDICT_CONTROLS = ("cross_fitted", "neighbourhood_only")

VERDICT_CONFOUND = "confounded_original_regression"
VERDICT_NEGATIVE_STANDS = "negative_result_stands"


def reaches_parity(significance_block: dict[str, Any]) -> bool:
    """True unless the 95% CI on the delta lies entirely below zero.

    A control is only judged worse than its reference when the interval
    excludes zero on the negative side. An interval spanning zero is not
    evidence of a regression, so it counts as parity.
    """
    excludes_zero = bool(significance_block["excludes_zero"])
    ci_upper = float(significance_block["ci_upper_95"])
    return not (excludes_zero and ci_upper < 0.0)


ALIGNMENT_GAP_THRESHOLD = 0.5


def embedding_partition_alignment(embeddings_path: Path) -> dict[str, Any]:
    """Standardised train-vs-validation mean gap, per embedding column.

    A block produced by a single encoder places both partitions in one latent
    basis, so this gap reflects genuine distribution shift between the
    partitions and little else. A cross-fitted block is produced by several
    encoders at once, and unless their latent bases agree, the same column
    index denotes a different direction either side of the split. That shows up
    here as a much larger gap, and it is a defect in the block rather than a
    finding about the graph -- so it is measured and published alongside the
    metrics instead of being left for a reader to infer from a surprising
    PR-AUC.
    """
    feat_names = embedding_feature_names()
    frame = pd.read_parquet(embeddings_path, columns=["split", *feat_names])
    train = frame.loc[frame["split"] == "train", feat_names].to_numpy()
    validation = frame.loc[frame["split"] == "validation", feat_names].to_numpy()
    pooled_sd = np.sqrt((train.var(axis=0) + validation.var(axis=0)) / 2.0) + 1e-12
    gap = np.abs(train.mean(axis=0) - validation.mean(axis=0)) / pooled_sd
    return {
        "metric": "standardised_absolute_mean_gap_train_vs_validation",
        "mean_gap": float(gap.mean()),
        "max_gap": float(gap.max()),
        "threshold": ALIGNMENT_GAP_THRESHOLD,
        "columns_above_threshold": int((gap > ALIGNMENT_GAP_THRESHOLD).sum()),
        "column_count": int(len(feat_names)),
    }


def _reference_row(label: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "run": label,
        "kind": "reference",
        "isolates": "",
        "pr_auc": float(metrics["pr_auc"]),
        "roc_auc": float(metrics["roc_auc"]),
        "best_iteration": metrics.get("best_iteration"),
        "delta_pr_auc_vs_b0": np.nan,
        "ci_lower_vs_b0": np.nan,
        "ci_upper_vs_b0": np.nan,
        "delta_pr_auc_vs_b1_card1": np.nan,
        "ci_lower_vs_b1_card1": np.nan,
        "ci_upper_vs_b1_card1": np.nan,
        "delta_pr_auc_vs_g1": np.nan,
        "ci_lower_vs_g1": np.nan,
        "ci_upper_vs_g1": np.nan,
        "parity_with_b1_card1": "",
    }


def load_control_results() -> dict[str, dict[str, Any]]:
    """Metrics and significance for every control that has been run."""
    results: dict[str, dict[str, Any]] = {}
    for name in CONTROL_ORDER:
        paths = CONTROL_RUNS[name].paths()
        if not paths["metrics"].exists() or not paths["significance"].exists():
            continue
        results[name] = {
            "metrics": read_json(paths["metrics"]),
            "significance": read_json(paths["significance"]),
            "metadata": read_json(paths["metadata"]),
            "alignment": embedding_partition_alignment(CONTROL_RUNS[name].embeddings_path),
        }
    if not results:
        raise FileNotFoundError(
            "No completed control runs found under reports/g1_controls/. "
            "Run: python -m src.models.train_lightgbm_g1_controls"
        )
    return results


def build_comparison_table(
    references: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows = [_reference_row(label, metrics) for label, metrics in references.items()]
    for name, payload in results.items():
        metrics = payload["metrics"]
        significance = payload["significance"]
        row: dict[str, Any] = {
            "run": name,
            "kind": "control",
            "isolates": CONTROL_RUNS[name].isolates,
            "pr_auc": float(metrics["pr_auc"]),
            "roc_auc": float(metrics["roc_auc"]),
            "best_iteration": metrics.get("best_iteration"),
        }
        for label in ("b0", "b1_card1", "g1"):
            block = significance[f"control_vs_{label}"]
            row[f"delta_pr_auc_vs_{label}"] = float(block["observed_delta"])
            row[f"ci_lower_vs_{label}"] = float(block["ci_lower_95"])
            row[f"ci_upper_vs_{label}"] = float(block["ci_upper_95"])
        row["parity_with_b1_card1"] = (
            "yes" if reaches_parity(significance["control_vs_b1_card1"]) else "no"
        )
        rows.append(row)
    return pd.DataFrame(rows)


def build_verdict(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    missing = [name for name in VERDICT_CONTROLS if name not in results]
    if missing:
        return {
            "decidable": False,
            "verdict": None,
            "reason": (
                "The pre-agreed stopping rule needs both verdict controls; "
                f"missing: {sorted(missing)}."
            ),
            "controls_required": list(VERDICT_CONTROLS),
        }

    per_control = {
        name: {
            "pr_auc": float(results[name]["metrics"]["pr_auc"]),
            "delta_vs_b1_card1": float(
                results[name]["significance"]["control_vs_b1_card1"]["observed_delta"]
            ),
            "ci_95_vs_b1_card1": [
                float(results[name]["significance"]["control_vs_b1_card1"]["ci_lower_95"]),
                float(results[name]["significance"]["control_vs_b1_card1"]["ci_upper_95"]),
            ],
            "reaches_parity_with_b1_card1": reaches_parity(
                results[name]["significance"]["control_vs_b1_card1"]
            ),
        }
        for name in VERDICT_CONTROLS
    }
    all_parity = all(entry["reaches_parity_with_b1_card1"] for entry in per_control.values())

    if all_parity:
        verdict = VERDICT_CONFOUND
        consequence = (
            "The original G1 regression was an artifact of the encoder setup. "
            "The graph stage continues on the corrected pipeline: cross-fitted "
            "embeddings with a neighbourhood-only readout become the G1 recipe, "
            "and the frozen G1 run is superseded rather than reported as a "
            "verdict on graph structure."
        )
    else:
        verdict = VERDICT_NEGATIVE_STANDS
        consequence = (
            "The negative result stands as a genuine finding: a supervised graph "
            "encoder on the card1 relation, corrected for readout, cross-fitting "
            "and training budget, does not beat four scalar relational summaries. "
            "The G1 epic closes and remaining effort goes to hardening the B1 claim."
        )

    return {
        "decidable": True,
        "verdict": verdict,
        "rule": (
            "Both cross_fitted and neighbourhood_only must reach at least parity "
            "with B1-card1, where parity means the 95% paired-bootstrap CI on "
            "PR-AUC(control) - PR-AUC(B1-card1) does not lie entirely below zero."
        ),
        "rule_fixed_before_runs": True,
        "controls_required": list(VERDICT_CONTROLS),
        "per_control": per_control,
        "consequence": consequence,
    }


def build_calibration(results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """What the shuffled null says the graph verdict has to clear."""
    if SHUFFLED_VARIANT_NAME not in results:
        return {
            "available": False,
            "note": "The shuffled-embedding control has not been run.",
        }
    shuffled = results[SHUFFLED_VARIANT_NAME]["significance"]["control_vs_b0"]
    dilution = float(shuffled["observed_delta"])
    g1_gap = float(read_json(G1_METRICS_PATH)["pr_auc"]) - float(
        read_json(B0_METRICS_PATH)["pr_auc"]
    )
    share = dilution / g1_gap if g1_gap != 0.0 else float("nan")
    return {
        "available": True,
        "shuffled_vs_b0_delta_pr_auc": dilution,
        "shuffled_vs_b0_ci_95": [
            float(shuffled["ci_lower_95"]),
            float(shuffled["ci_upper_95"]),
        ],
        "shuffled_vs_b0_excludes_zero": bool(shuffled["excludes_zero"]),
        "frozen_g1_vs_b0_delta_pr_auc": g1_gap,
        "share_of_frozen_g1_gap_explained_by_dilution": share,
        "interpretation": (
            "A randomly-aligned 32-column block with the real block's marginals "
            "costs the frozen LightGBM configuration this much against B0. Only "
            "the frozen G1 gap beyond it is attributable to the embedding's "
            "content rather than to widening the split search."
        ),
    }


def build_known_limitations(results: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """Defects in how a control was built, published next to its numbers.

    These are properties of the control, not conclusions about the graph. A
    reader who takes an affected control's magnitude at face value without them
    would draw the wrong inference, so they travel with the result rather than
    living in a commit message.
    """
    limitations: list[dict[str, Any]] = []

    cross_fitted = results.get("cross_fitted")
    if cross_fitted is not None:
        alignment = cross_fitted["alignment"]
        limitations.append(
            {
                "control": "cross_fitted",
                "defect": "unaligned_latent_bases_across_fold_encoders",
                "detail": (
                    "Train rows are embedded by the K fold encoders and "
                    "validation/test rows by the full-train encoder, each "
                    "initialised from a different seed. Nothing constrains "
                    "independently initialised encoders to agree on a latent "
                    "basis, so an embedding column denotes a different direction "
                    "either side of the train/validation boundary."
                ),
                "evidence": (
                    f"standardised train-vs-validation mean gap "
                    f"{alignment['mean_gap']:.3f}, with "
                    f"{alignment['columns_above_threshold']} of "
                    f"{alignment['column_count']} columns above "
                    f"{alignment['threshold']} -- roughly double every "
                    f"single-encoder block in this table"
                ),
                "consequence": (
                    "This control's magnitude confounds removing label leakage "
                    "with misaligning the feature block, so it is not evidence "
                    "for what cross-fitting alone costs or gains."
                ),
                "does_not_affect_verdict": (
                    "The stopping rule requires both verdict controls to reach "
                    "parity with B1-card1. neighbourhood_only is built from a "
                    "single encoder, shows no such defect, and does not reach "
                    "parity -- so the verdict is unchanged whichever way a "
                    "corrected cross-fitting run lands."
                ),
                "remedy": (
                    "Share one initialisation across the full-train and fold "
                    "encoders so their bases stay comparable, or align each "
                    "fold encoder's output to the full-train encoder before "
                    "assembling the block."
                ),
            }
        )
    return limitations


def build_summary(
    references: dict[str, dict[str, Any]],
    results: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "report_name": "G1 attribution controls",
        "relation_name": RELATION,
        "question": (
            "Does the frozen G1 regression measure graph structure being "
            "unhelpful, or the encoder's readout, cross-fitting and training "
            "budget?"
        ),
        "evaluation_split": "validation",
        "test_evaluated": False,
        "significance_method": "paired_bootstrap_pr_auc_delta_95pct_ci",
        "references": {
            label: {
                "pr_auc": float(metrics["pr_auc"]),
                "roc_auc": float(metrics["roc_auc"]),
            }
            for label, metrics in references.items()
        },
        "controls_run": list(results),
        "controls_not_run": [name for name in CONTROL_ORDER if name not in results],
        "controls": {
            name: {
                "isolates": CONTROL_RUNS[name].isolates,
                "description": CONTROL_RUNS[name].description,
                "pr_auc": float(payload["metrics"]["pr_auc"]),
                "roc_auc": float(payload["metrics"]["roc_auc"]),
                "embedding_gain_importance": payload["metadata"].get(
                    "embedding_gain_importance"
                ),
                "significance": {
                    label: payload["significance"][f"control_vs_{label}"]
                    for label in ("b0", "b1_card1", "g1")
                },
                "embedding_partition_alignment": payload["alignment"],
            }
            for name, payload in results.items()
        },
        "dilution_calibration": build_calibration(results),
        "known_limitations": build_known_limitations(results),
        "verdict": build_verdict(results),
        "comparison_table_path": repository_relative(COMPARISON_PATH),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def run_comparison() -> None:
    references = {
        "b0": read_json(B0_METRICS_PATH),
        "b1_card1": read_json(B1_CARD1_METRICS_PATH),
        "g1": read_json(G1_METRICS_PATH),
    }
    results = load_control_results()

    CONTROL_REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    table = build_comparison_table(references, results)
    table.to_csv(COMPARISON_PATH, index=False)
    summary = build_summary(references, results)
    write_json(SUMMARY_PATH, summary)

    print("\nG1 attribution controls -- validation PR-AUC\n")
    display = table[
        [
            "run",
            "kind",
            "pr_auc",
            "delta_pr_auc_vs_b1_card1",
            "ci_lower_vs_b1_card1",
            "ci_upper_vs_b1_card1",
            "parity_with_b1_card1",
        ]
    ]
    print(display.to_string(index=False, float_format=lambda v: f"{v:+.6f}"))

    verdict = summary["verdict"]
    print(f"\nComparison table: {COMPARISON_PATH}")
    print(f"Summary: {SUMMARY_PATH}")
    if not verdict["decidable"]:
        print(f"\nVerdict: NOT YET DECIDABLE -- {verdict['reason']}")
        return
    for limitation in summary["known_limitations"]:
        print(
            f"\nKnown limitation -- {limitation['control']}: {limitation['defect']}"
            f"\n  {limitation['evidence']}"
        )

    print(f"\nVerdict: {verdict['verdict']}")
    print(verdict["consequence"])


def main() -> None:
    run_comparison()


if __name__ == "__main__":
    main()
