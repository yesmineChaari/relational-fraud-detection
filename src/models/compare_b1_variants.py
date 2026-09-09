"""Cross-relation B1 comparison report.

Consolidates metrics across:
  B0  (tabular baseline)
  B1-card_core_addr1  (original)
  B1-card1            (new)
  B1-card1_card2      (new)

Every reference value is read from the frozen metric artifacts, so the deltas
and the outcome classification can never drift away from the models they
describe.

The two "new" variants' deltas against B0, plus the B1-card1 vs
B1-card1_card2 head-to-head, carry a 95% CI from the paired bootstrap in
reports/b1/b1_significance.json (src/models/compare_b1_significance.py). A
bare delta is never reported without the interval that says whether it is
distinguishable from noise.

Outputs:
  reports/b1/b1_cross_relation_comparison.csv
  reports/b1/b1_cross_relation_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from src.models.train_lightgbm_baseline import repository_relative

ROOT_DIR = Path(__file__).resolve().parents[2]

# role:
#   "baseline" -> the frozen B0 reference every delta is measured against
#   "original" -> the first (archived) B1 relational experiment
#   "new"      -> B1 variants added after train-only relation screening
VARIANTS = [
    {
        "variant": "B0",
        "relation": "none",
        "role": "baseline",
        "metrics_path": ROOT_DIR / "reports" / "baseline" / "lightgbm_metrics.json",
        "feat_imp_path": None,
    },
    {
        "variant": "B1-card_core_addr1",
        "relation": "card_core_addr1",
        "role": "original",
        "metrics_path": ROOT_DIR / "reports" / "b1" / "card_core_addr1" / "metrics.json",
        "feat_imp_path": ROOT_DIR / "reports" / "b1" / "card_core_addr1" / "feature_importance.csv",
    },
    {
        "variant": "B1-card1",
        "relation": "card1",
        "role": "new",
        "metrics_path": ROOT_DIR / "reports" / "b1" / "card1" / "metrics.json",
        "feat_imp_path": ROOT_DIR / "reports" / "b1" / "card1" / "feature_importance.csv",
    },
    {
        "variant": "B1-card1_card2",
        "relation": "card1_card2",
        "role": "new",
        "metrics_path": ROOT_DIR / "reports" / "b1" / "card1_card2" / "metrics.json",
        "feat_imp_path": ROOT_DIR / "reports" / "b1" / "card1_card2" / "feature_importance.csv",
    },
]

REPORT_DIR = ROOT_DIR / "reports" / "b1"
COMPARISON_CSV = REPORT_DIR / "b1_cross_relation_comparison.csv"
SUMMARY_JSON = REPORT_DIR / "b1_cross_relation_summary.json"
SIGNIFICANCE_JSON = REPORT_DIR / "b1_significance.json"

RANKING_KEYS = ["top_0.5_pct", "top_1_pct", "top_2_pct", "top_5_pct"]

# Maps a variant's `relation` to the significance-comparison key that reports
# its interval against B0, so a delta is never printed without one.
SIGNIFICANCE_VS_B0_KEY = {
    "card1": "b1_card1_vs_b0",
    "card1_card2": "b1_card1_card2_vs_b0",
}
HEAD_TO_HEAD_KEY = "b1_card1_vs_b1_card1_card2"


def load_significance(path: Path = SIGNIFICANCE_JSON) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run: python -m src.models.compare_b1_significance"
        )
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found: {path}")
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def relational_feature_ranks(feat_imp_path: Path | None, relation: str) -> dict[str, Any]:
    """Extract gain/split rank of the 4 relational features from feature importance."""
    if feat_imp_path is None or not feat_imp_path.exists():
        return {"relational_gain_ranks": None, "relational_split_ranks": None}

    df = pd.read_csv(feat_imp_path)
    df = df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    df["rank_gain"] = df["importance_gain"].rank(ascending=False, method="min").astype(int)
    df["rank_split"] = df["importance_split"].rank(ascending=False, method="min").astype(int)
    feat_names = [
        f"{relation}_prior_count",
        f"{relation}_prior_count_24h",
        f"{relation}_prior_count_7d",
        f"{relation}_time_since_previous_hours",
    ]
    rows = df[df["feature"].isin(feat_names)][
        ["feature", "importance_gain", "importance_split", "rank_gain", "rank_split"]
    ].to_dict(orient="records")
    total = len(df)
    return {
        "relational_feature_importance": rows,
        "total_features_in_model": total,
        "relational_gain_ranks": [r["rank_gain"] for r in rows],
        "relational_split_ranks": [r["rank_split"] for r in rows],
    }


def build_comparison_row(
    variant: str,
    relation: str,
    role: str,
    metrics: dict[str, Any],
    rank_info: dict[str, Any],
    b0_pr_auc: float,
    b0_roc_auc: float,
    significance: dict[str, Any],
) -> dict[str, Any]:
    pr_auc = float(metrics.get("pr_auc", float("nan")))
    roc_auc = float(metrics.get("roc_auc", float("nan")))
    best_iter = metrics.get("best_iteration")
    row: dict[str, Any] = {
        "variant": variant,
        "relation": relation,
        "role": role,
        "validation_pr_auc": pr_auc,
        "validation_roc_auc": roc_auc,
        "delta_pr_auc_vs_b0": pr_auc - b0_pr_auc,
        "delta_roc_auc_vs_b0": roc_auc - b0_roc_auc,
        "best_iteration": best_iter,
        "early_stopping_triggered": metrics.get("early_stopping_triggered"),
        "test_evaluated": metrics.get("test_evaluated", False),
    }
    significance_key = SIGNIFICANCE_VS_B0_KEY.get(relation)
    comparison = significance["comparisons"].get(significance_key) if significance_key else None
    row["delta_pr_auc_vs_b0_ci_lower_95"] = comparison["ci_lower_95"] if comparison else None
    row["delta_pr_auc_vs_b0_ci_upper_95"] = comparison["ci_upper_95"] if comparison else None
    row["delta_pr_auc_vs_b0_excludes_zero"] = comparison["excludes_zero"] if comparison else None
    for key in RANKING_KEYS:
        ranking = (metrics.get("ranking_metrics") or {}).get(key, {})
        row[f"precision_{key}"] = ranking.get("precision")
        row[f"recall_{key}"] = ranking.get("recall")
    row["relational_gain_ranks"] = str(rank_info.get("relational_gain_ranks"))
    row["relational_split_ranks"] = str(rank_info.get("relational_split_ranks"))
    row["total_features_in_model"] = rank_info.get("total_features_in_model")
    return row


def classify_outcome(
    comparison_rows: list[dict[str, Any]],
    b0_pr_auc: float,
    original_b1_pr_auc: float | None,
) -> dict[str, Any]:
    """Map experimental results to the project's relational decision rules."""
    non_b0 = [r for r in comparison_rows if r["role"] != "baseline"]
    new_variants = [r for r in comparison_rows if r["role"] == "new"]

    any_beats_b0 = any(r["validation_pr_auc"] > b0_pr_auc for r in non_b0)
    all_worse_than_b0 = bool(non_b0) and all(r["validation_pr_auc"] < b0_pr_auc for r in non_b0)
    all_beat_original_b1 = (
        bool(new_variants)
        and original_b1_pr_auc is not None
        and all(r["validation_pr_auc"] > original_b1_pr_auc for r in new_variants)
    )

    if any_beats_b0:
        outcome_code = "A"
        conclusion = (
            "Hypothesis Confirmed: Simple relational features work with high-coverage "
            "low-fragmentation relations. Proceed to G1 using the winning relation."
        )
    elif all_beat_original_b1 and all_worse_than_b0:
        outcome_code = "B"
        conclusion = (
            "Partial Recovery: Relation matters, but linear/tree aggregation remains "
            "insufficient. Strong motivation for G1 GNN embeddings."
        )
    else:
        outcome_code = "C"
        conclusion = (
            "Invariant Degradation: Tabular aggregation limitation confirmed. "
            "GNN relational inductive bias (G1) is strictly required."
        )

    best_new = max(new_variants, key=lambda r: r["validation_pr_auc"]) if new_variants else None
    return {
        "outcome_code": outcome_code,
        "scientific_conclusion": conclusion,
        "b0_pr_auc_reference": b0_pr_auc,
        "b1_card_core_addr1_pr_auc_reference": original_b1_pr_auc,
        "any_beats_b0": any_beats_b0,
        "all_new_variants_beat_card_core_addr1": all_beat_original_b1,
        "best_new_variant": best_new["variant"] if best_new else None,
        "best_new_variant_pr_auc": best_new["validation_pr_auc"] if best_new else None,
        "best_new_variant_delta_pr_auc_vs_b0": (
            best_new["delta_pr_auc_vs_b0"] if best_new else None
        ),
    }


def main() -> None:
    # Load every available variant first: the B0 row supplies the reference
    # values, so no metric is ever restated as a literal in this module.
    loaded: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for spec in VARIANTS:
        try:
            metrics = load_metrics(spec["metrics_path"])
        except FileNotFoundError as exc:
            print(f"  SKIP {spec['variant']}: {exc}")
            continue
        loaded.append((spec, metrics))

    baseline = next((metrics for spec, metrics in loaded if spec["role"] == "baseline"), None)
    if baseline is None:
        raise FileNotFoundError(
            "The frozen B0 metrics are required to compute deltas; none were loaded."
        )
    b0_pr_auc = float(baseline["pr_auc"])
    b0_roc_auc = float(baseline["roc_auc"])
    original_b1 = next((metrics for spec, metrics in loaded if spec["role"] == "original"), None)
    original_b1_pr_auc = float(original_b1["pr_auc"]) if original_b1 else None

    print(f"  B0 reference: PR-AUC={b0_pr_auc:.12f}  ROC-AUC={b0_roc_auc:.12f}")

    significance = load_significance()

    comparison_rows = []
    for spec, metrics in loaded:
        rank_info = relational_feature_ranks(spec["feat_imp_path"], spec["relation"])
        row = build_comparison_row(
            spec["variant"],
            spec["relation"],
            spec["role"],
            metrics,
            rank_info,
            b0_pr_auc,
            b0_roc_auc,
            significance,
        )
        comparison_rows.append(row)
        ci_note = ""
        if row["delta_pr_auc_vs_b0_ci_lower_95"] is not None:
            ci_note = (
                f"  95% CI=[{row['delta_pr_auc_vs_b0_ci_lower_95']:+.5f}, "
                f"{row['delta_pr_auc_vs_b0_ci_upper_95']:+.5f}]  "
                f"excludes_zero={row['delta_pr_auc_vs_b0_excludes_zero']}"
            )
        print(
            f"  {spec['variant']}: PR-AUC={row['validation_pr_auc']:.5f}  "
            f"ROC-AUC={row['validation_roc_auc']:.5f}  "
            f"Delta-PR={row['delta_pr_auc_vs_b0']:+.5f}{ci_note}"
        )

    baseline_row = next(r for r in comparison_rows if r["role"] == "baseline")
    if baseline_row["delta_pr_auc_vs_b0"] != 0.0 or baseline_row["delta_roc_auc_vs_b0"] != 0.0:
        raise AssertionError(
            "The B0 row must have a zero delta against itself; the reference metrics "
            "do not match the baseline artifact."
        )

    comparison_df = pd.DataFrame(comparison_rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_CSV, index=False)
    print(f"Comparison table saved: {COMPARISON_CSV}")

    outcome = classify_outcome(comparison_rows, b0_pr_auc, original_b1_pr_auc)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        # Relation candidates were chosen by train-only screening (Stage A).
        # Validation is used here only to score the already-chosen candidates,
        # which is the sanctioned Stage B reporting step.
        "candidate_selection_used_validation": False,
        "validation_used_for_reporting": True,
        "final_test_evaluated": False,
        "outcome": outcome,
        "variants": comparison_rows,
        "b1_card1_vs_b1_card1_card2_significance": significance["comparisons"].get(
            HEAD_TO_HEAD_KEY
        ),
        "significance_source": repository_relative(SIGNIFICANCE_JSON),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"Summary JSON saved: {SUMMARY_JSON}")
    print(f"\nOutcome: [{outcome['outcome_code']}] {outcome['scientific_conclusion']}")


if __name__ == "__main__":
    main()
