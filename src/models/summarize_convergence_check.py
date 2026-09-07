"""Convergence-check comparison and the estimator-cap decision.

Reads the extended-cap runs produced by train_lightgbm_convergence_check.py
plus the three frozen, 6,000-cap references (B0, B1-card1, G1-card1) and
publishes:

  * where each configuration actually converges once the cap is raised,
  * the capped-versus-converged PR-AUC delta for the B1 gain (B1-card1 minus
    B0) and the G1 deficit (G1-card1 minus B0), using the same comparison
    function the frozen pipeline itself uses to compute those deltas,
  * whether decoupling the stopping metric from the reported one (patience on
    AUC instead of average_precision) changes the B1 gain, when that run was
    also produced, and
  * a decision on the cap, decided by a fixed rule rather than free-text
    prose: if any configuration still fails to converge before the extended
    cap, the cap is still binding. Otherwise, if every converged delta agrees
    with its capped counterpart within twice the clean-stratum seed-noise
    floor already published by the seed-variance panel, the existing headline
    conclusions survive and the cap is documented as a known limitation. If a
    converged delta moves by more than that, the headline numbers require
    revision.

Outputs:
  reports/convergence_check/convergence_comparison.csv
  reports/convergence_check/convergence_summary.json
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.train_lightgbm_baseline import RANDOM_SEED, repository_relative, write_json
from src.models.train_lightgbm_convergence_check import (
    CONFIGURATIONS,
    DEFAULT_MAX_ESTIMATORS,
    FROZEN_REFERENCE_METRICS,
    REPORT_DIR,
    STOP_METRICS,
    resolve_run_paths,
)
from src.models.train_lightgbm_g1 import build_comparison_table
from src.models.train_lightgbm_relational import read_json

ROOT_DIR = Path(__file__).resolve().parents[2]

COMPARISON_PATH = REPORT_DIR / "convergence_comparison.csv"
SUMMARY_PATH = REPORT_DIR / "convergence_summary.json"

SEED_VARIANCE_SUMMARY_PATH = ROOT_DIR / "reports" / "seed_variance" / "seed_variance_summary.json"

# The stop metric every acceptance criterion in this investigation depends on;
# reproduces the current stopping rule at the extended cap.
REQUIRED_STOP_METRIC = "average_precision"
# The optional decoupling test, run only for B0 and B1-card1.
DECOUPLING_STOP_METRIC = "auc"

VERDICT_CONVERGED_CONCLUSIONS_SURVIVE = "converged_conclusions_survive_document_cap"
VERDICT_CAP_STILL_BINDING = "cap_still_binding_raise_or_decouple"
VERDICT_HEADLINE_REVISION_REQUIRED = "headline_numbers_require_revision"

# How many clean-stratum seed standard deviations a converged delta may move
# from its capped counterpart before the headline numbers are judged to have
# actually changed, rather than moved within already-quantified seed noise.
NOISE_FLOOR_MULTIPLE = 2.0


def discover_runs(n_estimators: int, seed: int) -> dict[tuple[str, str], dict[str, Any]]:
    """Metrics for every completed (config, stop_metric) cell at (n_estimators, seed)."""
    found: dict[tuple[str, str], dict[str, Any]] = {}
    for config in CONFIGURATIONS:
        for stop_metric in STOP_METRICS:
            paths = resolve_run_paths(config, stop_metric, n_estimators, seed)
            if paths["metrics"].exists():
                found[(config, stop_metric)] = read_json(paths["metrics"])
    if ("b0", REQUIRED_STOP_METRIC) not in found:
        raise FileNotFoundError(
            f"No completed convergence-check runs found under {REPORT_DIR}. "
            "Run: python -m src.models.train_lightgbm_convergence_check"
        )
    return found


def convergence_points(
    runs: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Where each configuration stops under the required (average_precision) rule."""
    points: dict[str, dict[str, Any]] = {}
    for config in CONFIGURATIONS:
        run = runs.get((config, REQUIRED_STOP_METRIC))
        if run is None:
            continue
        frozen = read_json(FROZEN_REFERENCE_METRICS[config])
        points[config] = {
            "extended_maximum_estimators": int(run["maximum_estimators"]),
            "extended_best_iteration": int(run["best_iteration"]),
            "extended_early_stopping_triggered": bool(run["early_stopping_triggered"]),
            "extended_pr_auc": float(run["pr_auc"]),
            "extended_roc_auc": float(run["roc_auc"]),
            "frozen_maximum_estimators": int(frozen["maximum_estimators"]),
            "frozen_best_iteration": int(frozen["best_iteration"]),
            "frozen_pr_auc": float(frozen["pr_auc"]),
            "frozen_roc_auc": float(frozen["roc_auc"]),
            "best_iteration_gap": int(run["best_iteration"]) - int(frozen["best_iteration"]),
        }
    return points


def _delta_pr_auc(table: pd.DataFrame, candidate_label: str, reference_label: str) -> float:
    column = f"delta_{candidate_label}_minus_{reference_label}"
    row = table.loc[table["metric"] == "PR-AUC"]
    if len(row) != 1:
        raise AssertionError("Comparison table does not contain exactly one PR-AUC row.")
    return float(row[column].iloc[0])


def capped_vs_converged(
    runs: dict[tuple[str, str], dict[str, Any]],
    reference_config: str,
    candidate_config: str,
) -> dict[str, Any] | None:
    """Capped (6,000) vs. converged (extended-cap) PR-AUC delta, same comparison function."""
    reference_run = runs.get((reference_config, REQUIRED_STOP_METRIC))
    candidate_run = runs.get((candidate_config, REQUIRED_STOP_METRIC))
    if reference_run is None or candidate_run is None:
        return None

    frozen_reference = read_json(FROZEN_REFERENCE_METRICS[reference_config])
    frozen_candidate = read_json(FROZEN_REFERENCE_METRICS[candidate_config])

    capped_table = build_comparison_table(
        frozen_reference, frozen_candidate, reference_config, candidate_config
    )
    converged_table = build_comparison_table(
        reference_run, candidate_run, reference_config, candidate_config
    )
    capped_delta = _delta_pr_auc(capped_table, candidate_config, reference_config)
    converged_delta = _delta_pr_auc(converged_table, candidate_config, reference_config)

    return {
        "reference": reference_config,
        "candidate": candidate_config,
        "capped_delta_pr_auc": capped_delta,
        "converged_delta_pr_auc": converged_delta,
        "delta_shift": converged_delta - capped_delta,
    }


def decoupled_stopping_effect(
    runs: dict[tuple[str, str], dict[str, Any]]
) -> dict[str, Any] | None:
    """Capped/AP-patience vs. extended/AP-patience vs. extended/AUC-patience, B0 vs B1-card1."""
    b0_auc = runs.get(("b0", DECOUPLING_STOP_METRIC))
    b1_auc = runs.get(("b1_card1", DECOUPLING_STOP_METRIC))
    if b0_auc is None or b1_auc is None:
        return None

    frozen_b0 = read_json(FROZEN_REFERENCE_METRICS["b0"])
    frozen_b1 = read_json(FROZEN_REFERENCE_METRICS["b1_card1"])
    capped_ap_delta = _delta_pr_auc(
        build_comparison_table(frozen_b0, frozen_b1, "b0", "b1_card1"), "b1_card1", "b0"
    )
    extended_auc_delta = _delta_pr_auc(
        build_comparison_table(b0_auc, b1_auc, "b0", "b1_card1"), "b1_card1", "b0"
    )

    result: dict[str, Any] = {
        "capped_ap_patience_delta_pr_auc": capped_ap_delta,
        "extended_auc_patience_delta_pr_auc": extended_auc_delta,
        "b0_extended_auc_patience_early_stopping_triggered": bool(
            b0_auc["early_stopping_triggered"]
        ),
        "b1_card1_extended_auc_patience_early_stopping_triggered": bool(
            b1_auc["early_stopping_triggered"]
        ),
        "b0_extended_auc_patience_best_iteration": int(b0_auc["best_iteration"]),
        "b1_card1_extended_auc_patience_best_iteration": int(b1_auc["best_iteration"]),
    }

    b0_ap = runs.get(("b0", REQUIRED_STOP_METRIC))
    b1_ap = runs.get(("b1_card1", REQUIRED_STOP_METRIC))
    if b0_ap is not None and b1_ap is not None:
        result["extended_ap_patience_delta_pr_auc"] = _delta_pr_auc(
            build_comparison_table(b0_ap, b1_ap, "b0", "b1_card1"), "b1_card1", "b0"
        )
    return result


def load_noise_floor(path: Path = SEED_VARIANCE_SUMMARY_PATH) -> float:
    """The seed-variance panel's clean-stratum paired-delta standard deviation.

    Read from its own published artifact rather than hardcoded, so the
    tolerance used here stays traceable to its source and updates
    automatically if that panel is ever rerun.
    """
    summary = read_json(path)
    return float(summary["early_stopping_stratification"]["clean_delta_pr_auc"]["std"])


def build_cap_decision(
    convergence: dict[str, dict[str, Any]],
    b1_gain: dict[str, Any] | None,
    g1_deficit: dict[str, Any] | None,
    noise_floor: float,
    required_configs: tuple[str, ...] = tuple(CONFIGURATIONS),
) -> dict[str, Any]:
    missing = [config for config in required_configs if config not in convergence]
    if missing:
        return {
            "decidable": False,
            "verdict": None,
            "reasoning": (
                "Convergence points are missing for: "
                f"{sorted(missing)}. All three configurations must have a completed "
                f"{REQUIRED_STOP_METRIC}-patience run before a cap decision can be made."
            ),
            "configs_required": list(required_configs),
        }

    all_converged = all(
        convergence[config]["extended_early_stopping_triggered"] for config in required_configs
    )
    comparisons = [c for c in (b1_gain, g1_deficit) if c is not None]
    deltas_agree = all(
        abs(comparison["delta_shift"]) <= NOISE_FLOOR_MULTIPLE * noise_floor
        for comparison in comparisons
    )

    if not all_converged:
        still_capped = [
            config
            for config in required_configs
            if not convergence[config]["extended_early_stopping_triggered"]
        ]
        verdict = VERDICT_CAP_STILL_BINDING
        reasoning = (
            f"{sorted(still_capped)} still run to the extended cap without early "
            "stopping firing. The cap has not been shown non-binding for every "
            "configuration, so it should be raised further, or the stopping metric "
            "decoupled from the reported one, before any capped-vs-converged "
            "comparison here can be trusted."
        )
    elif deltas_agree:
        verdict = VERDICT_CONVERGED_CONCLUSIONS_SURVIVE
        reasoning = (
            "Every configuration now stops early before the extended cap, and every "
            "converged delta agrees with its capped counterpart within "
            f"{NOISE_FLOOR_MULTIPLE:g}x the clean-stratum seed noise floor "
            f"({noise_floor:.5f}) established by the seed-variance panel. The "
            "existing headline conclusions, measured at the frozen 6,000-estimator "
            "cap, survive; the cap is documented as a known, quantified limitation "
            "rather than raised for the frozen configuration."
        )
    else:
        verdict = VERDICT_HEADLINE_REVISION_REQUIRED
        reasoning = (
            "Every configuration now stops early before the extended cap, but at "
            "least one converged delta moves by more than "
            f"{NOISE_FLOOR_MULTIPLE:g}x the clean-stratum seed noise floor "
            f"({noise_floor:.5f}) relative to its capped value. The capped headline "
            "numbers do not survive convergence and must be revised to the "
            "converged figures."
        )

    return {
        "decidable": True,
        "verdict": verdict,
        "rule": (
            "If any required configuration fails to trigger early stopping before "
            "the extended cap: cap still binding. Else if every converged-vs-capped "
            f"PR-AUC delta shift is within {NOISE_FLOOR_MULTIPLE:g}x the seed-"
            "variance panel's clean-stratum noise floor: converged conclusions "
            "survive. Else: headline numbers require revision."
        ),
        "noise_floor_source": repository_relative(SEED_VARIANCE_SUMMARY_PATH),
        "noise_floor_clean_delta_pr_auc_std": noise_floor,
        "all_required_configs_converged": all_converged,
        "reasoning": reasoning,
        "plain_language": reasoning,
    }


def build_summary(n_estimators: int, seed: int) -> dict[str, Any]:
    runs = discover_runs(n_estimators, seed)
    convergence = convergence_points(runs)
    b1_gain = capped_vs_converged(runs, "b0", "b1_card1")
    g1_deficit = capped_vs_converged(runs, "b0", "g1_card1")
    decoupling = decoupled_stopping_effect(runs)
    noise_floor = load_noise_floor()
    decision = build_cap_decision(convergence, b1_gain, g1_deficit, noise_floor)

    return {
        "report_name": "Estimator-cap convergence check",
        "question": (
            "Where do B0, B1-card1 and G1-card1 actually converge past the frozen "
            "6,000-estimator cap, and do the B1 gain and the G1 deficit survive "
            "convergence?"
        ),
        "extended_maximum_estimators": n_estimators,
        "early_stopping_rounds_unchanged_at": 200,
        "seed": seed,
        "convergence_points": convergence,
        "b1_gain_capped_vs_converged": b1_gain,
        "g1_deficit_capped_vs_converged": g1_deficit,
        "decoupled_stopping_metric_effect": decoupling,
        "decision": decision,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the estimator-cap convergence check and record the cap decision."
    )
    parser.add_argument("--max-estimators", type=int, default=DEFAULT_MAX_ESTIMATORS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    summary = build_summary(args.max_estimators, args.seed)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_rows = [
        {"configuration": config, **points}
        for config, points in summary["convergence_points"].items()
    ]
    pd.DataFrame(comparison_rows).to_csv(COMPARISON_PATH, index=False)
    write_json(SUMMARY_PATH, summary)

    print(f"Comparison saved: {COMPARISON_PATH}")
    print(f"Summary saved: {SUMMARY_PATH}")
    print(f"\nDecision: {summary['decision']['verdict']}")
    print(summary["decision"]["reasoning"])


if __name__ == "__main__":
    main()
