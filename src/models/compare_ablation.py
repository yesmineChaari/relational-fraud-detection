"""Localise the B1-card1 gain across the four card1 relational summaries.

Reads the eight ablation runs (four singleton, four leave-one-out) and applies
the interpretation rule fixed before any of them was trained. Singletons are
compared against the converged B0 reference, leave-one-out variants against the
converged B1-card1 reference; both by the paired bootstrap every other
significance question in this project is settled by.

Pre-registered rule
-------------------

1. A feature **carries the gain** if its singleton 95% CI on the
   singleton-minus-B0 delta excludes zero on the positive side.
2. A feature is **non-redundant** if its leave-one-out 95% CI on the
   leave-one-out-minus-B1-card1 delta excludes zero on the negative side --
   dropping it costs something the other three cannot replace.
3. Differences *between* ablation variants smaller than the clean-stratum
   noise floor are not interpretable as differences. The floor is the
   seed-variance panel's standard deviation of the paired delta on the seeds
   where both configurations ran to the estimator cap (0.00051), read from that
   panel's own summary rather than restated here.

Why both directions. The four summaries are correlated 0.83-0.95. Leave-one-out
alone would likely return four nulls -- each column is reconstructible from the
others, so dropping any single one costs nothing, and nothing is localised.
Singletons alone cannot distinguish four views of one underlying factor from
four genuinely additive signals. Read together they separate the cases:

  * exactly one carrier, and it is also non-redundant -> one summary carries
    the gain;
  * several carriers, none non-redundant -> the summaries are redundant views
    of one factor and any one of them suffices;
  * one or more non-redundant features -> the summaries contribute additively;
  * neither -> the gain is real but not localised to any single summary.

Outputs:
  reports/ablation/ablation_comparison.csv
  reports/ablation/ablation_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.significance import compare_variants
from src.models.train_lightgbm_ablation import (
    ABLATION_FEATURES,
    B0_CONVERGED_PATHS,
    B1_CARD1_CONVERGED_PATHS,
    FEATURE_KEYS,
    LEAVE_ONE_OUT,
    MODES,
    RELATION,
    REPORT_DIR,
    SINGLETON,
    resolve_run_paths,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

COMPARISON_CSV = REPORT_DIR.parent / "ablation_comparison.csv"
SUMMARY_JSON = REPORT_DIR.parent / "ablation_summary.json"

SEED_VARIANCE_SUMMARY_PATH = ROOT_DIR / "reports" / "seed_variance" / "seed_variance_summary.json"

OUTCOME_SINGLE_FEATURE = "SINGLE_FEATURE_CARRIES_GAIN"
OUTCOME_REDUNDANT = "REDUNDANT_SUMMARIES_ANY_ONE_SUFFICES"
OUTCOME_ADDITIVE = "ADDITIVE_CONTRIBUTIONS"
OUTCOME_NOT_LOCALISED = "GAIN_NOT_LOCALISED"
OUTCOME_PANEL_INCOMPLETE = "PANEL_INCOMPLETE"

VERDICT_RULE = (
    "Fixed before any ablation result was seen. A feature 'carries the gain' "
    "if its singleton (B0 + that feature) paired-bootstrap 95% CI on the "
    "delta against converged B0 excludes zero on the positive side. A feature "
    "is 'non-redundant' if its leave-one-out (B1-card1 minus that feature) "
    "95% CI on the delta against converged B1-card1 excludes zero on the "
    "negative side. Differences between ablation variants smaller than the "
    "clean-stratum noise floor from the seed-variance panel are not "
    "interpretable as differences, so a singleton ranking is only read as an "
    "ordering where the margin clears that floor."
)

CONCLUSIONS = {
    OUTCOME_SINGLE_FEATURE: (
        "Gain localised to a single summary: exactly one feature both carries "
        "the gain on its own and is non-redundant given the other three. The "
        "B1-card1 effect is that feature's; the remaining three are "
        "redundant with it and could be dropped without cost."
    ),
    OUTCOME_REDUNDANT: (
        "Redundant summaries: one or more features carry the gain on their "
        "own, but no feature is non-redundant -- dropping any single one "
        "costs nothing measurable because the others reconstruct it. The four "
        "summaries are correlated views of one underlying factor (card1 "
        "transaction history), and any one of them recovers the effect."
    ),
    OUTCOME_ADDITIVE: (
        "Additive contributions: at least one feature is non-redundant, "
        "meaning the other three cannot replace what it carries. The "
        "B1-card1 gain is not a single-column story; the summaries "
        "contribute distinct signal and the manifest should keep them."
    ),
    OUTCOME_NOT_LOCALISED: (
        "Gain not localised: no singleton significantly beats B0 and no "
        "leave-one-out variant significantly loses to B1-card1. The gain "
        "survives at the four-feature level but is too diffuse for this "
        "panel to attribute to any individual summary."
    ),
    OUTCOME_PANEL_INCOMPLETE: (
        "No verdict: the eight-run panel is incomplete, so the two "
        "directions cannot be read against each other. The carrier and "
        "non-redundant lists below are provisional and cover only the runs "
        "that exist -- in particular an empty list means 'not yet measured', "
        "not 'measured and null'. Complete the panel before quoting an "
        "outcome."
    ),
}


def load_metrics(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Metrics not found: {path}")
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_noise_floor(path: Path = SEED_VARIANCE_SUMMARY_PATH) -> float:
    """The clean-stratum paired-delta standard deviation from the seed-variance panel.

    Read rather than restated: the interpretability threshold and the panel it
    comes from must not be able to drift apart.
    """
    summary = load_metrics(path)
    try:
        floor = summary["early_stopping_stratification"]["clean_delta_pr_auc"]["std"]
    except (KeyError, TypeError) as error:
        raise KeyError(
            "Seed-variance summary does not carry the clean-stratum paired-delta "
            f"standard deviation this rule depends on: {path}"
        ) from error
    floor = float(floor)
    if not np.isfinite(floor) or floor <= 0.0:
        raise ValueError(f"Clean-stratum noise floor must be positive and finite; got {floor!r}.")
    return floor


def spread(values: np.ndarray) -> dict[str, Any]:
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


def discover_runs() -> list[tuple[str, str]]:
    """Every completed (mode, feature) run in the ablation report tree.

    Enumerates the panel rather than globbing directory names, so a run whose
    directory exists but whose artifacts are incomplete is treated as missing
    instead of being half-read into a summary that claims to cover it.
    """
    completed = []
    for mode in MODES:
        for feature in ABLATION_FEATURES:
            paths = resolve_run_paths(mode, feature)
            if paths["metrics"].exists() and paths["validation_predictions"].exists():
                completed.append((mode, feature))
    if not completed:
        raise FileNotFoundError(
            f"No completed ablation runs found under {REPORT_DIR}. "
            "Run: python -m src.models.train_lightgbm_ablation"
        )
    return completed


def rank_singletons(comparison_df: pd.DataFrame, noise_floor: float) -> list[dict[str, Any]]:
    """Order singletons by PR-AUC, flagging which gaps clear the noise floor.

    A margin below the floor means the two singletons are indistinguishable at
    this panel's resolution; the ordering is still reported, but the rule
    forbids reading it as a difference.
    """
    singletons = comparison_df[comparison_df["ablation_mode"] == SINGLETON]
    ordered = singletons.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    ranking = []
    for position, row in ordered.iterrows():
        if position + 1 < len(ordered):
            margin = float(row["pr_auc"] - ordered.loc[position + 1, "pr_auc"])
            next_feature = str(ordered.loc[position + 1, "feature"])
        else:
            margin = None
            next_feature = None
        ranking.append(
            {
                "rank": int(position) + 1,
                "feature": str(row["feature"]),
                "pr_auc": float(row["pr_auc"]),
                "delta_pr_auc_vs_reference": float(row["delta_pr_auc_vs_reference"]),
                "next_feature": next_feature,
                "margin_over_next": margin,
                "margin_exceeds_noise_floor": (
                    None if margin is None else bool(abs(margin) > noise_floor)
                ),
            }
        )
    return ranking


def classify_ablation_outcome(
    comparison_df: pd.DataFrame,
    noise_floor: float,
) -> dict[str, Any]:
    """Apply the pre-registered rule to a completed ablation table.

    Both gates are confidence intervals, not point-estimate signs. A singleton
    a few 1e-05 above B0 has not "carried the gain" in the sense the rule
    means -- it is indistinguishable from B0 -- and a leave-one-out variant a
    few 1e-05 below B1-card1 has not shown the dropped feature was needed.
    """
    if comparison_df.empty:
        raise ValueError("No ablation runs to classify.")

    singletons = comparison_df[comparison_df["ablation_mode"] == SINGLETON]
    leave_one_out = comparison_df[comparison_df["ablation_mode"] == LEAVE_ONE_OUT]

    carriers = sorted(
        singletons.loc[
            singletons["delta_pr_auc_vs_reference_excludes_zero"]
            & (singletons["delta_pr_auc_vs_reference"] > 0),
            "feature",
        ].tolist()
    )
    non_redundant = sorted(
        leave_one_out.loc[
            leave_one_out["delta_pr_auc_vs_reference_excludes_zero"]
            & (leave_one_out["delta_pr_auc_vs_reference"] < 0),
            "feature",
        ].tolist()
    )

    panel_complete = bool(
        len(singletons) == len(ABLATION_FEATURES) and len(leave_one_out) == len(ABLATION_FEATURES)
    )

    # An outcome code is a claim about both directions at once, so it may only
    # be emitted once both directions are fully measured. On a partial panel
    # every substantive code would be an artifact of what has not run yet: an
    # empty non_redundant list, for instance, reads identically whether the
    # leave-one-out runs came back null or were never trained, and the code
    # would silently assert the first. The carrier and non-redundant lists are
    # still returned -- they are accurate over the runs that exist.
    if not panel_complete:
        outcome_code = OUTCOME_PANEL_INCOMPLETE
    elif not carriers and not non_redundant:
        outcome_code = OUTCOME_NOT_LOCALISED
    elif len(non_redundant) == 1 and carriers == non_redundant:
        outcome_code = OUTCOME_SINGLE_FEATURE
    elif non_redundant:
        outcome_code = OUTCOME_ADDITIVE
    else:
        outcome_code = OUTCOME_REDUNDANT

    return {
        "features_carrying_the_gain": carriers,
        "non_redundant_features": non_redundant,
        "panel_complete": panel_complete,
        "singleton_ranking": rank_singletons(comparison_df, noise_floor),
        "outcome_code": outcome_code,
        "conclusion": CONCLUSIONS[outcome_code],
    }


def main() -> None:
    completed = discover_runs()
    print(f"Completed ablation runs: {len(completed)} of {len(MODES) * len(ABLATION_FEATURES)}")

    noise_floor = load_noise_floor()
    b0_metrics = load_metrics(B0_CONVERGED_PATHS["metrics"])
    b1_card1_metrics = load_metrics(B1_CARD1_CONVERGED_PATHS["metrics"])
    b0_pr_auc = float(b0_metrics["pr_auc"])
    b1_card1_pr_auc = float(b1_card1_metrics["pr_auc"])
    real_gain = b1_card1_pr_auc - b0_pr_auc
    if real_gain == 0.0:
        raise ValueError(
            "The converged B0 and B1-card1 references report identical PR-AUC; "
            "there is no gain for this ablation to localise."
        )

    reference_paths = {
        SINGLETON: B0_CONVERGED_PATHS,
        LEAVE_ONE_OUT: B1_CARD1_CONVERGED_PATHS,
    }
    reference_pr_auc = {SINGLETON: b0_pr_auc, LEAVE_ONE_OUT: b1_card1_pr_auc}
    reference_label = {SINGLETON: "b0_converged", LEAVE_ONE_OUT: "b1_card1_converged"}

    print(f"B0 converged PR-AUC:        {b0_pr_auc:.12f}")
    print(f"B1-card1 converged PR-AUC:  {b1_card1_pr_auc:.12f}")
    print(f"Real relational gain:       {real_gain:+.12f}")
    print(f"Clean-stratum noise floor:  {noise_floor:.12f}")

    rows: list[dict[str, Any]] = []
    for mode, feature in completed:
        paths = resolve_run_paths(mode, feature)
        metrics = load_metrics(paths["metrics"])
        pr_auc = float(metrics["pr_auc"])
        key = FEATURE_KEYS[feature]

        comparison = compare_variants(
            candidate_label=f"ablation_{mode}_{key}",
            candidate_predictions_path=paths["validation_predictions"],
            reference_label=reference_label[mode],
            reference_predictions_path=reference_paths[mode]["validation_predictions"],
        )
        delta = pr_auc - reference_pr_auc[mode]
        rows.append(
            {
                "ablation_mode": mode,
                "feature": feature,
                "feature_key": key,
                "relational_features_used": "|".join(metrics["relational_features_used"]),
                "number_of_predictors": int(metrics["number_of_predictors"]),
                "pr_auc": pr_auc,
                "roc_auc": float(metrics["roc_auc"]),
                "best_iteration": int(metrics["best_iteration"]),
                "estimator_cap_reached": bool(metrics["estimator_cap_reached"]),
                "reference": reference_label[mode],
                "reference_pr_auc": reference_pr_auc[mode],
                "delta_pr_auc_vs_reference": delta,
                "delta_pr_auc_vs_reference_ci_lower_95": comparison["ci_lower_95"],
                "delta_pr_auc_vs_reference_ci_upper_95": comparison["ci_upper_95"],
                "delta_pr_auc_vs_reference_excludes_zero": comparison["excludes_zero"],
                "delta_exceeds_noise_floor": bool(abs(delta) > noise_floor),
                "fraction_of_real_gain": delta / real_gain,
            }
        )
        print(
            f"  {mode:>9} {key:<26} PR-AUC={pr_auc:.6f}  "
            f"delta_vs_{reference_label[mode]}={delta:+.6f}  "
            f"CI=[{comparison['ci_lower_95']:+.5f}, {comparison['ci_upper_95']:+.5f}]  "
            f"excludes_zero={comparison['excludes_zero']}"
        )

    comparison_df = pd.DataFrame(rows)
    REPORT_DIR.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(COMPARISON_CSV, index=False)
    print(f"\nComparison table saved: {COMPARISON_CSV}")

    outcome = classify_ablation_outcome(comparison_df, noise_floor)
    singleton_deltas = comparison_df.loc[
        comparison_df["ablation_mode"] == SINGLETON, "delta_pr_auc_vs_reference"
    ].to_numpy()
    loo_deltas = comparison_df.loc[
        comparison_df["ablation_mode"] == LEAVE_ONE_OUT, "delta_pr_auc_vs_reference"
    ].to_numpy()

    summary = {
        "report_name": "Per-feature ablation of the card1 relational summaries",
        "relation": RELATION,
        "question": (
            "Which of the four card1 relational summaries carries the +0.00630 B1-card1 gain?"
        ),
        "reference_configuration": "converged (cap 15,000, stop_average_precision, seed 42)",
        "b0_converged_pr_auc": b0_pr_auc,
        "b1_card1_converged_pr_auc": b1_card1_pr_auc,
        "real_relational_gain_pr_auc": real_gain,
        "ablation_features": ABLATION_FEATURES,
        "modes": MODES,
        "panel_complete": outcome["panel_complete"],
        "completed_runs": [
            {"ablation_mode": mode, "feature": feature} for mode, feature in completed
        ],
        "clean_stratum_noise_floor": noise_floor,
        "noise_floor_source": "reports/seed_variance/seed_variance_summary.json",
        "singleton_delta_vs_b0_spread": spread(singleton_deltas),
        "leave_one_out_delta_vs_b1_card1_spread": spread(loo_deltas),
        "features_carrying_the_gain": outcome["features_carrying_the_gain"],
        "non_redundant_features": outcome["non_redundant_features"],
        "singleton_ranking": outcome["singleton_ranking"],
        "verdict_rule": VERDICT_RULE,
        "outcome": {
            "outcome_code": outcome["outcome_code"],
            "conclusion": outcome["conclusion"],
        },
        "runs": rows,
        "comparison_table_path": "reports/ablation/ablation_comparison.csv",
        "b0_reference_metrics_path": (
            "reports/convergence_check/b0/stop_average_precision/cap15000_seed42/metrics.json"
        ),
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

    if not outcome["panel_complete"]:
        print(
            "\nWARNING: the panel is incomplete; the outcome below is provisional "
            "until all eight runs exist."
        )
    print(f"\nCarries the gain:     {outcome['features_carrying_the_gain'] or 'none'}")
    print(f"Non-redundant:        {outcome['non_redundant_features'] or 'none'}")
    print(f"\nOutcome: [{outcome['outcome_code']}] {outcome['conclusion']}")


if __name__ == "__main__":
    main()
