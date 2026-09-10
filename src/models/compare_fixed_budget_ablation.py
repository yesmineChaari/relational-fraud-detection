"""Settle the card1 localisation from the fixed-budget panel, and show the bracket.

Two protocols disagreed about which of the four card1 summaries carries the
relational gain, and both were biased. Reporting each run at its own validation
argmax rewards whichever arm was granted more rounds. Re-scoring every run at the
panel minimum truncates whichever arm wanted more. They disagreed in opposite
directions, which is why the answer was bracketed rather than known.

The fixed-budget panel removes the selection instead of equalising it: every run
trains the same 10,000 rounds with no early stopping and is read at that round.
This module applies the published interpretation rule to those runs.

Three things it is careful about.

**The rule is imported, not restated.** `classify_panel` comes from the
equal-budget module, which took it verbatim from the published ablation. Three
protocols therefore differ in exactly one place -- how the runs were trained and
read -- so any change of outcome is attributable to the protocol alone and not to
a rule that drifted while being retyped.

**All three protocols are reported side by side.** A reader should see the
bracket and how it closed, not only the endpoint this analysis happened to reach.
A result that agrees with one prior protocol and not the other means something
different from one that agrees with both.

**The off-metric is carried through.** ROC-AUC at the same fixed budget is not
the quantity any protocol selected on, so if it contradicts the PR-AUC reading
that is evidence the conclusion is about the stopping rule rather than the
predictors -- the same check that exposed the original artifact.

Outputs:
  reports/fixed_budget/fixed_budget_comparison.csv
  reports/fixed_budget/fixed_budget_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import ROOT_DIR
from src.models.rederive_ablation_at_equal_budget import (
    LEAVE_ONE_OUT,
    OUTCOME_ADDITIVE,
    OUTCOME_REDUNDANT,
    SINGLETON,
    classify_panel,
)
from src.models.significance import compare_variants
from src.models.train_ablation_fixed_budget import (
    FEATURE_KEYS,
    FIXED_BUDGET,
    MODE_REFERENCE,
    PROTOCOL,
    planned_runs,
    resolve_run_paths,
    run_is_complete,
)

OUTPUT_DIR = ROOT_DIR / "reports" / "fixed_budget"
COMPARISON_CSV = OUTPUT_DIR / "fixed_budget_comparison.csv"
SUMMARY_JSON = OUTPUT_DIR / "fixed_budget_summary.json"

ARGMAX_COMPARISON = ROOT_DIR / "reports" / "ablation" / "ablation_comparison.csv"
EQUAL_BUDGET_COMPARISON = ROOT_DIR / "reports" / "selection_bias" / "equal_budget_ablation.csv"

ARGMAX_OUTCOME = OUTCOME_REDUNDANT
EQUAL_BUDGET_OUTCOME = OUTCOME_ADDITIVE

VERDICT_MATCHES_ARGMAX = "FIXED_BUDGET_CONFIRMS_PUBLISHED_VERDICT"
VERDICT_MATCHES_EQUAL_BUDGET = "FIXED_BUDGET_SUPERSEDES_PUBLISHED_VERDICT"
VERDICT_MATCHES_NEITHER = "LOCALISATION_UNDECIDABLE_AT_THIS_PANEL_SIZE"

INTERPRETATION_RULE = (
    "Fixed before the panel was trained. The published ablation rule is applied "
    "verbatim: a feature carries the gain if its singleton interval excludes zero "
    "on the positive side, and is non-redundant if its leave-one-out interval "
    "excludes zero on the negative side. If the fixed-budget outcome matches the "
    "argmax verdict the published result stands; if it matches the equal-budget "
    "verdict the published result is superseded; if it matches neither, the gain "
    "is real but not localisable at this panel size, and that is the finding."
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def assert_panel_complete() -> None:
    missing = [run for run in planned_runs() if not run_is_complete(run)]
    if missing:
        raise SystemExit(
            "The fixed-budget panel is not finished; these runs are missing:\n"
            + "\n".join(f"  - {run}" for run in missing)
            + "\n\nRun: python -m src.models.train_ablation_fixed_budget --skip-existing"
        )


def build_table() -> pd.DataFrame:
    """One row per ablation cell, bootstrapped against its fixed-budget reference."""
    assert_panel_complete()
    rows = []
    for mode in (SINGLETON, LEAVE_ONE_OUT):
        reference = MODE_REFERENCE[mode]
        reference_paths = resolve_run_paths(reference)
        reference_metrics = read_json(reference_paths["metrics"])
        for key in FEATURE_KEYS.values():
            run = f"{mode}_{key}"
            paths = resolve_run_paths(run)
            metrics = read_json(paths["metrics"])
            bootstrap = compare_variants(
                candidate_label=run,
                candidate_predictions_path=paths["validation_predictions"],
                reference_label=reference,
                reference_predictions_path=reference_paths["validation_predictions"],
            )
            rows.append(
                {
                    "ablation_mode": mode,
                    "feature": key,
                    "reference": reference,
                    "fixed_budget_trees": FIXED_BUDGET,
                    "pr_auc": float(metrics["pr_auc"]),
                    "reference_pr_auc": float(reference_metrics["pr_auc"]),
                    "delta_pr_auc": bootstrap["observed_delta"],
                    "ci_lower_95": bootstrap["ci_lower_95"],
                    "ci_upper_95": bootstrap["ci_upper_95"],
                    "excludes_zero": bootstrap["excludes_zero"],
                    "delta_roc_auc": float(metrics["roc_auc"])
                    - float(reference_metrics["roc_auc"]),
                    # What the discarded convention would have reported, kept so
                    # the discarded reading stays inspectable rather than hidden.
                    "argmax_iteration_not_used": metrics.get("argmax_iteration_not_used"),
                }
            )
    return pd.DataFrame(rows)


def load_prior_protocol(path: Path, delta_column: str, excludes_column: str) -> pd.DataFrame | None:
    """A previous protocol's per-cell deltas, or None when it is not on disk."""
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    mode_column = "ablation_mode"
    feature_column = "feature"
    if feature_column not in frame.columns or mode_column not in frame.columns:
        return None
    trimmed = frame[[mode_column, feature_column, delta_column, excludes_column]].copy()
    trimmed[feature_column] = trimmed[feature_column].str.replace("card1_", "", regex=False)
    return trimmed


def build_bracket(table: pd.DataFrame) -> pd.DataFrame:
    """The three protocols side by side, per cell."""
    bracket = table[["ablation_mode", "feature", "delta_pr_auc", "excludes_zero"]].rename(
        columns={
            "delta_pr_auc": "fixed_budget_delta",
            "excludes_zero": "fixed_budget_excludes_zero",
        }
    )

    argmax = load_prior_protocol(
        ARGMAX_COMPARISON, "delta_pr_auc_vs_reference", "delta_pr_auc_vs_reference_excludes_zero"
    )
    if argmax is not None:
        argmax = argmax.rename(
            columns={
                "delta_pr_auc_vs_reference": "argmax_delta",
                "delta_pr_auc_vs_reference_excludes_zero": "argmax_excludes_zero",
            }
        )
        bracket = bracket.merge(argmax, on=["ablation_mode", "feature"], how="left")

    equal = load_prior_protocol(EQUAL_BUDGET_COMPARISON, "delta_pr_auc", "excludes_zero")
    if equal is not None:
        equal = equal.rename(
            columns={
                "delta_pr_auc": "equal_budget_delta",
                "excludes_zero": "equal_budget_excludes_zero",
            }
        )
        bracket = bracket.merge(equal, on=["ablation_mode", "feature"], how="left")
    return bracket


def summarize(table: pd.DataFrame) -> dict[str, Any]:
    carriers, non_redundant, outcome = classify_panel(table)

    if outcome == ARGMAX_OUTCOME:
        verdict = VERDICT_MATCHES_ARGMAX
        conclusion = (
            "The fixed-budget panel reproduces the published outcome. The "
            "disagreement raised by the equal-budget re-derivation was an "
            "artifact of truncating the arms that wanted more rounds, and the "
            "published localisation stands."
        )
    elif outcome == EQUAL_BUDGET_OUTCOME:
        verdict = VERDICT_MATCHES_EQUAL_BUDGET
        conclusion = (
            "The fixed-budget panel reproduces the equal-budget outcome. The "
            "published localisation rested on the argmax convention and is "
            "superseded: the summaries contribute additively rather than being "
            "redundant views any one of which suffices."
        )
    else:
        verdict = VERDICT_MATCHES_NEITHER
        conclusion = (
            "The fixed-budget panel agrees with neither prior protocol. The gain "
            "is real but is not localisable to individual summaries at this panel "
            "size, and that is the finding rather than something to resolve by "
            "preferring whichever estimator reads better."
        )

    off_metric_disagrees = [
        f"{row.ablation_mode}_{row.feature}"
        for row in table.itertuples()
        if row.excludes_zero and np.sign(row.delta_pr_auc) != np.sign(row.delta_roc_auc)
    ]

    return {
        "report_name": "Card1 ablation at a budget fixed in advance",
        "question": (
            "Which of the four card1 summaries carries the relational gain, when "
            "no arm is selected by watching a validation curve?"
        ),
        "protocol": PROTOCOL,
        "fixed_budget_trees": FIXED_BUDGET,
        "early_stopping_used": False,
        "interpretation_rule": INTERPRETATION_RULE,
        "rule_source": "imported verbatim from the equal-budget module, not restated",
        "features_carrying_the_gain": carriers,
        "non_redundant_features": non_redundant,
        "outcome_code": outcome,
        "verdict": verdict,
        "conclusion": conclusion,
        "prior_protocols": {
            "argmax": ARGMAX_OUTCOME,
            "equal_budget_6011": EQUAL_BUDGET_OUTCOME,
        },
        "significant_cells_where_roc_auc_disagrees": off_metric_disagrees,
        "off_metric_note": (
            "ROC-AUC is not the quantity any protocol selected on. A significant "
            "PR-AUC delta whose ROC-AUC delta has the opposite sign would be "
            "evidence about the stopping rule rather than the predictors."
        ),
        "comparison_table_path": str(COMPARISON_CSV.relative_to(ROOT_DIR).as_posix()),
        "n_cells": int(len(table)),
        "test_evaluated": False,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    table = build_table()
    bracket = build_bracket(table)
    summary = summarize(table)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(COMPARISON_CSV, index=False)
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    pd.set_option("display.width", 220)
    print(f"=== Fixed budget: {FIXED_BUDGET:,} trees, no early stopping ===")
    print(
        table[
            [
                "ablation_mode",
                "feature",
                "delta_pr_auc",
                "ci_lower_95",
                "ci_upper_95",
                "excludes_zero",
                "delta_roc_auc",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x: .5f}")
    )
    print("\n=== The bracket, per cell ===")
    print(bracket.to_string(index=False, float_format=lambda x: f"{x: .5f}"))
    print(f"\ncarriers:      {summary['features_carrying_the_gain']}")
    print(f"non-redundant: {summary['non_redundant_features']}")
    print(f"outcome:       {summary['outcome_code']}")
    print(f"verdict:       {summary['verdict']}")
    print(f"\nWrote {COMPARISON_CSV}\nWrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
