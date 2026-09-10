"""Cross-stage comparison spanning B0, the B1 relational variants and G1.

Supersedes the B1-only cross-relation report, which read a hardcoded list of
four variants and classified outcomes against B1-specific questions ("did any
variant beat B0", "did all new variants beat the original relational variant").
Appending G1 to that list would have left the classifier answering the wrong
question, so both the registry and the classifier are generalised here.

Two things this module is careful about.

**Stages are data.** A stage is a row in `STAGES`, carrying its own order,
role, artifact paths and significance keys. Adding a future stage is a
registration; the classifier reads `stage_order` and never a stage name, so it
does not need editing when one appears.

**Protocol is explicit.** Frozen artifacts were trained at a 6,000-estimator
cap, and the convergence check showed that cap was binding -- frozen G1 never
triggered early stopping and stopped at the cap with `estimator_cap_reached`
true. Its converged figure is a different, larger number. Silently mixing the
two protocols in one table would compare a converged model against a truncated
one, which is the confound the convergence work removed. Every row therefore
declares its protocol, rows are only ever compared within a protocol, and the
converged figures are carried alongside rather than substituted in.

The frozen rows reproduce the previous report's values exactly, by design and
by test: this module changes what is reported around them, never what they say.

Deltas carry their paired-bootstrap interval from the significance artifacts.
Where a comparison has no interval the fields are null and the classifier
treats it as undecided rather than guessing from the point estimate.

Outputs:
  reports/stages/stage_comparison.csv
  reports/stages/stage_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUT_DIR = REPORTS_DIR / "stages"
COMPARISON_CSV = OUTPUT_DIR / "stage_comparison.csv"
SUMMARY_JSON = OUTPUT_DIR / "stage_summary.json"

FROZEN = "frozen_cap6000"
CONVERGED = "converged_cap15000"

BASELINE_ROLE = "baseline"

RANKING_KEYS = ["top_0.5_pct", "top_1_pct", "top_2_pct", "top_5_pct"]

B1_SIGNIFICANCE = REPORTS_DIR / "b1" / "b1_significance.json"
G1_SIGNIFICANCE = REPORTS_DIR / "g1" / "card1" / "significance.json"
CONVERGED_SIGNIFICANCE = REPORTS_DIR / "convergence_check" / "convergence_significance.json"

# Every stage the project has produced, in the order it was produced. `role`
# distinguishes the reference every delta is measured against from the variants
# measured against it; `stage_order` is what the classifier reads.
STAGES: list[dict[str, Any]] = [
    {
        "variant": "B0",
        "stage": "B0",
        "stage_order": 0,
        "relation": "none",
        "role": BASELINE_ROLE,
        "protocol": FROZEN,
        "metrics_path": REPORTS_DIR / "baseline" / "lightgbm_metrics.json",
        "feature_importance_path": None,
        "significance": None,
    },
    {
        "variant": "B1-card_core_addr1",
        "stage": "B1",
        "stage_order": 1,
        "relation": "card_core_addr1",
        "role": "original",
        "protocol": FROZEN,
        "metrics_path": REPORTS_DIR / "b1" / "card_core_addr1" / "metrics.json",
        "feature_importance_path": REPORTS_DIR
        / "b1"
        / "card_core_addr1"
        / "feature_importance.csv",
        "significance": None,
    },
    {
        "variant": "B1-card1",
        "stage": "B1",
        "stage_order": 1,
        "relation": "card1",
        "role": "new",
        "protocol": FROZEN,
        "metrics_path": REPORTS_DIR / "b1" / "card1" / "metrics.json",
        "feature_importance_path": REPORTS_DIR / "b1" / "card1" / "feature_importance.csv",
        "significance": (B1_SIGNIFICANCE, ["comparisons", "b1_card1_vs_b0"]),
    },
    {
        "variant": "B1-card1_card2",
        "stage": "B1",
        "stage_order": 1,
        "relation": "card1_card2",
        "role": "new",
        "protocol": FROZEN,
        "metrics_path": REPORTS_DIR / "b1" / "card1_card2" / "metrics.json",
        "feature_importance_path": REPORTS_DIR / "b1" / "card1_card2" / "feature_importance.csv",
        "significance": (B1_SIGNIFICANCE, ["comparisons", "b1_card1_card2_vs_b0"]),
    },
    {
        "variant": "G1-card1",
        "stage": "G1",
        "stage_order": 2,
        "relation": "card1",
        "role": "new",
        "protocol": FROZEN,
        "metrics_path": REPORTS_DIR / "g1" / "card1" / "metrics.json",
        "feature_importance_path": REPORTS_DIR / "g1" / "card1" / "feature_importance.csv",
        "significance": (G1_SIGNIFICANCE, ["g1_vs_b0"]),
    },
]

CONVERGED_DIR = REPORTS_DIR / "convergence_check"
CONVERGED_RUN = "cap15000_seed42"

# The same three models re-measured at a cap none of them reaches. Carried
# alongside the frozen rows, never blended into them.
CONVERGED_STAGES: list[dict[str, Any]] = [
    {
        "variant": "B0",
        "stage": "B0",
        "stage_order": 0,
        "relation": "none",
        "role": BASELINE_ROLE,
        "protocol": CONVERGED,
        "metrics_path": CONVERGED_DIR
        / "b0"
        / "stop_average_precision"
        / CONVERGED_RUN
        / "metrics.json",
        "feature_importance_path": None,
        "significance": None,
    },
    {
        "variant": "B1-card1",
        "stage": "B1",
        "stage_order": 1,
        "relation": "card1",
        "role": "new",
        "protocol": CONVERGED,
        "metrics_path": CONVERGED_DIR
        / "b1_card1"
        / "stop_average_precision"
        / CONVERGED_RUN
        / "metrics.json",
        "feature_importance_path": CONVERGED_DIR
        / "b1_card1"
        / "stop_average_precision"
        / CONVERGED_RUN
        / "feature_importance.csv",
        "significance": (CONVERGED_SIGNIFICANCE, ["comparisons", "b1_card1_vs_b0"]),
    },
    {
        "variant": "G1-card1",
        "stage": "G1",
        "stage_order": 2,
        "relation": "card1",
        "role": "new",
        "protocol": CONVERGED,
        "metrics_path": CONVERGED_DIR
        / "g1_card1"
        / "stop_average_precision"
        / CONVERGED_RUN
        / "metrics.json",
        "feature_importance_path": CONVERGED_DIR
        / "g1_card1"
        / "stop_average_precision"
        / CONVERGED_RUN
        / "feature_importance.csv",
        "significance": (CONVERGED_SIGNIFICANCE, ["comparisons", "g1_card1_vs_b0"]),
    },
]

OUTCOME_BEATS_INCUMBENT = "LATEST_STAGE_BEATS_INCUMBENT"
OUTCOME_MATCHES_INCUMBENT = "LATEST_STAGE_MATCHES_INCUMBENT"
OUTCOME_UNDERPERFORMS_INCUMBENT = "LATEST_STAGE_UNDERPERFORMS_INCUMBENT"
OUTCOME_FAILS_BASELINE = "LATEST_STAGE_FAILS_TO_BEAT_BASELINE"
OUTCOME_UNDECIDED = "UNDECIDED_NO_INTERVAL"

CONCLUSIONS = {
    OUTCOME_BEATS_INCUMBENT: (
        "The newest stage improves on the best stage before it by more than "
        "sampling noise. The added representation earns its complexity."
    ),
    OUTCOME_MATCHES_INCUMBENT: (
        "The newest stage is indistinguishable from the best stage before it. "
        "It buys no measured accuracy, so the simpler incumbent is preferred "
        "on cost and interpretability alone."
    ),
    OUTCOME_UNDERPERFORMS_INCUMBENT: (
        "The newest stage is beaten by the best stage before it while still "
        "clearing the baseline. The representation carries signal but less "
        "than the hand-written features it was meant to improve on."
    ),
    OUTCOME_FAILS_BASELINE: (
        "The newest stage does not beat the tabular baseline at all. The "
        "representation is not merely weaker than the incumbent; it costs "
        "accuracy against using no relational information whatsoever."
    ),
    OUTCOME_UNDECIDED: (
        "No paired-bootstrap interval is available for the newest stage, so "
        "its standing is not decided here. A point estimate alone cannot "
        "separate a real difference from sampling noise."
    ),
}

CLASSIFICATION_RULE = (
    "Read from stage order, never from stage names, so registering a future "
    "stage needs no change here. The newest stage is compared against the "
    "baseline and against the best-performing earlier stage, both by the "
    "paired-bootstrap interval on the paired delta. Failing to beat the "
    "baseline takes precedence over any comparison with the incumbent, since a "
    "stage that loses to using no relational information at all is not "
    "meaningfully ranked against one that does. Where an interval is missing "
    "the outcome is undecided rather than inferred from the point estimate."
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def resolve_significance(entry: tuple[Path, list[str]] | None) -> dict[str, Any] | None:
    """Follow a (file, key path) reference into a significance artifact."""
    if entry is None:
        return None
    path, key_path = entry
    payload: Any = read_json(path)
    for key in key_path:
        if not isinstance(payload, dict) or key not in payload:
            raise KeyError(f"{path} has no {'.'.join(key_path)}; significance schema changed.")
        payload = payload[key]
    return payload


def relational_feature_ranks(path: Path | None, relation: str) -> dict[str, Any]:
    """Where the relational columns rank among all predictors, by gain and split."""
    if path is None or not path.exists():
        return {
            "relational_feature_importance": None,
            "total_features_in_model": None,
            "relational_gain_ranks": None,
            "relational_split_ranks": None,
        }
    frame = pd.read_csv(path).sort_values("importance_gain", ascending=False)
    frame = frame.reset_index(drop=True)
    frame["rank_gain"] = frame["importance_gain"].rank(ascending=False, method="min").astype(int)
    frame["rank_split"] = frame["importance_split"].rank(ascending=False, method="min").astype(int)
    names = [
        f"{relation}_prior_count",
        f"{relation}_prior_count_24h",
        f"{relation}_prior_count_7d",
        f"{relation}_time_since_previous_hours",
    ]
    rows = frame[frame["feature"].isin(names)][
        ["feature", "importance_gain", "importance_split", "rank_gain", "rank_split"]
    ].to_dict(orient="records")
    return {
        "relational_feature_importance": rows,
        "total_features_in_model": len(frame),
        "relational_gain_ranks": [row["rank_gain"] for row in rows] or None,
        "relational_split_ranks": [row["rank_split"] for row in rows] or None,
    }


def build_rows(stages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One row per registered stage, with deltas against its own protocol's baseline."""
    baselines = {
        stage["protocol"]: read_json(stage["metrics_path"])
        for stage in stages
        if stage["role"] == BASELINE_ROLE
    }
    missing = {stage["protocol"] for stage in stages} - set(baselines)
    if missing:
        raise ValueError(f"No baseline registered for protocol(s): {sorted(missing)}.")

    rows = []
    for stage in stages:
        metrics = read_json(stage["metrics_path"])
        baseline = baselines[stage["protocol"]]
        significance = resolve_significance(stage["significance"])
        ranks = relational_feature_ranks(stage["feature_importance_path"], stage["relation"])

        pr_auc = float(metrics["pr_auc"])
        roc_auc = float(metrics["roc_auc"])
        row: dict[str, Any] = {
            "variant": stage["variant"],
            "stage": stage["stage"],
            "stage_order": stage["stage_order"],
            "relation": stage["relation"],
            "role": stage["role"],
            "protocol": stage["protocol"],
            "validation_pr_auc": pr_auc,
            "validation_roc_auc": roc_auc,
            "delta_pr_auc_vs_b0": pr_auc - float(baseline["pr_auc"]),
            "delta_roc_auc_vs_b0": roc_auc - float(baseline["roc_auc"]),
            "best_iteration": metrics.get("best_iteration"),
            "early_stopping_triggered": metrics.get("early_stopping_triggered"),
            "estimator_cap_reached": metrics.get("estimator_cap_reached"),
            # Carried through unchanged: selection happened on validation, and
            # no stage may quietly acquire a test number.
            "candidate_selection_used_validation": True,
            "final_test_evaluated": bool(metrics.get("test_evaluated", False)),
            "delta_pr_auc_vs_b0_ci_lower_95": (
                significance["ci_lower_95"] if significance else None
            ),
            "delta_pr_auc_vs_b0_ci_upper_95": (
                significance["ci_upper_95"] if significance else None
            ),
            "delta_pr_auc_vs_b0_excludes_zero": (
                significance["excludes_zero"] if significance else None
            ),
        }
        for key in RANKING_KEYS:
            ranking = (metrics.get("ranking_metrics") or {}).get(key, {})
            row[f"precision_{key}"] = ranking.get("precision")
            row[f"recall_{key}"] = ranking.get("recall")
        row["relational_gain_ranks"] = str(ranks["relational_gain_ranks"])
        row["relational_split_ranks"] = str(ranks["relational_split_ranks"])
        row["total_features_in_model"] = ranks["total_features_in_model"]
        rows.append(row)
    return rows


def classify_stages(
    rows: list[dict[str, Any]],
    incumbent_significance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Rank the newest stage against the baseline and the best earlier stage.

    Driven entirely by `stage_order`, so a stage registered later is picked up
    without touching this function.
    """
    if not rows:
        raise ValueError("No rows to classify.")
    baseline_rows = [row for row in rows if row["role"] == BASELINE_ROLE]
    if not baseline_rows:
        raise ValueError("No baseline row present.")
    baseline = baseline_rows[0]

    latest_order = max(row["stage_order"] for row in rows)
    latest_candidates = [
        row for row in rows if row["stage_order"] == latest_order and row["role"] != BASELINE_ROLE
    ]
    if not latest_candidates:
        raise ValueError("Newest stage has no non-baseline row.")
    latest = max(latest_candidates, key=lambda row: row["validation_pr_auc"])

    earlier = [
        row for row in rows if row["stage_order"] < latest_order and row["role"] != BASELINE_ROLE
    ]
    incumbent = max(earlier, key=lambda row: row["validation_pr_auc"]) if earlier else None

    beats_baseline = latest["delta_pr_auc_vs_b0_excludes_zero"] and (
        latest["delta_pr_auc_vs_b0"] > 0
    )
    fails_baseline = latest["delta_pr_auc_vs_b0_excludes_zero"] and (
        latest["delta_pr_auc_vs_b0"] < 0
    )

    if latest["delta_pr_auc_vs_b0_excludes_zero"] is None:
        outcome = OUTCOME_UNDECIDED
    elif fails_baseline:
        outcome = OUTCOME_FAILS_BASELINE
    elif incumbent is None:
        outcome = OUTCOME_BEATS_INCUMBENT if beats_baseline else OUTCOME_UNDECIDED
    elif incumbent_significance is None:
        outcome = OUTCOME_UNDECIDED
    elif not incumbent_significance["excludes_zero"]:
        outcome = OUTCOME_MATCHES_INCUMBENT
    elif incumbent_significance["observed_delta"] > 0:
        outcome = OUTCOME_BEATS_INCUMBENT
    else:
        outcome = OUTCOME_UNDERPERFORMS_INCUMBENT

    return {
        "baseline_variant": baseline["variant"],
        "latest_stage": latest["stage"],
        "latest_variant": latest["variant"],
        "incumbent_variant": incumbent["variant"] if incumbent else None,
        "latest_delta_vs_baseline": latest["delta_pr_auc_vs_b0"],
        "latest_delta_vs_baseline_excludes_zero": latest["delta_pr_auc_vs_b0_excludes_zero"],
        "latest_delta_vs_incumbent": (
            incumbent_significance["observed_delta"] if incumbent_significance else None
        ),
        "latest_delta_vs_incumbent_excludes_zero": (
            incumbent_significance["excludes_zero"] if incumbent_significance else None
        ),
        "outcome_code": outcome,
        "conclusion": CONCLUSIONS[outcome],
    }


def main() -> None:
    frozen_rows = build_rows(STAGES)
    converged_rows = build_rows(CONVERGED_STAGES)
    table = pd.DataFrame(frozen_rows + converged_rows)

    frozen_outcome = classify_stages(
        frozen_rows,
        incumbent_significance=resolve_significance((G1_SIGNIFICANCE, ["g1_vs_b1_card1"])),
    )
    converged_outcome = classify_stages(
        converged_rows,
        incumbent_significance=resolve_significance(
            (CONVERGED_SIGNIFICANCE, ["comparisons", "g1_card1_vs_b1_card1"])
        ),
    )

    summary: dict[str, Any] = {
        "report_name": "Cross-stage comparison: B0, B1 relational variants and G1",
        "question": (
            "Does each successive stage improve on the best stage before it, "
            "and does the newest stage beat the tabular baseline at all?"
        ),
        "classification_rule": CLASSIFICATION_RULE,
        "protocols": {
            FROZEN: (
                "6,000-estimator cap. The convergence check found this cap "
                "binding: frozen G1 never triggered early stopping and stopped "
                "at the cap, so its figure understates the model."
            ),
            CONVERGED: (
                "15,000-estimator cap, average_precision patience, seed 42. No "
                "run reaches the cap and every one triggers early stopping."
            ),
        },
        "protocols_are_not_comparable_across": (
            "Rows are compared only within a protocol. A converged model scored "
            "against a cap-truncated one would reintroduce the confound the "
            "convergence work removed."
        ),
        "frozen_outcome": frozen_outcome,
        "converged_outcome": converged_outcome,
        "quote_the_converged_g1_deficit": True,
        "n_rows": int(len(table)),
        "comparison_table_path": str(COMPARISON_CSV.relative_to(ROOT_DIR).as_posix()),
        "final_test_evaluated": bool(table["final_test_evaluated"].any()),
        "candidate_selection_used_validation": True,
        "versions": {"pandas": pd.__version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(COMPARISON_CSV, index=False)
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    pd.set_option("display.width", 220)
    print(
        table[
            [
                "variant",
                "protocol",
                "validation_pr_auc",
                "delta_pr_auc_vs_b0",
                "delta_pr_auc_vs_b0_excludes_zero",
                "estimator_cap_reached",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x: .6f}")
    )
    print(f"\nfrozen:    {frozen_outcome['outcome_code']}")
    print(f"converged: {converged_outcome['outcome_code']}")
    print(f"\nWrote {COMPARISON_CSV}\nWrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
