"""Seed panel for a single relational ablation variant against its reference.

The eight-run ablation panel is one fit per cell, all under the frozen seed
(42). That is enough to answer the question the panel was designed for -- which
summaries carry the gain -- because those comparisons are settled by paired
bootstrap intervals over a fixed fit. It is not enough for a claim that one
variant *beats* the four-feature reference, because refit noise moves these
numbers independently of any predictor change, and the seed-variance panel
already showed the paired delta is far less stable than either configuration's
absolute score.

This module measures that directly for one variant: refit both the variant and
its reference under several seeds, pair them seed by seed, and report the
spread of the paired delta. A delta that keeps its sign and its rough magnitude
across seeds is a property of the predictors. A delta whose spread is
comparable to its own size is a seed lottery, and the single-seed number that
motivated the investigation was a draw rather than a result.

Stratification
--------------

`summarize_seed_variance` splits its seeds on whether either configuration
stopped before the estimator cap. That criterion does not transfer here. At the
converged cap (15,000) nothing reaches the cap -- every run early-stops well
short of it -- so the inherited test would label all five seeds identically and
distinguish nothing.

The mechanism it was written to catch is still present, though: early stopping
selects the boosting round that maximises validation average precision, the
very metric being reported, so a model granted far more rounds than its
opponent wins partly on the stopping point rather than on its predictors. This
module therefore strata on the *ratio* between the two best iterations, flagging
a seed as contaminated when one model trained more than `MAX_BEST_ITERATION_RATIO`
times as long as the other.

That threshold was fixed while three of the five seeds had been trained and
before the remaining two existed. It is recorded here as a stated rule rather
than a pre-registered one, and both strata are always reported, so a reader can
apply their own threshold to the per-seed table instead of inheriting this one.

Outputs:
  reports/ablation/ablation_seed_panel_runs.csv
  reports/ablation/ablation_seed_panel_summary.json

Usage:
    python -m src.models.summarize_ablation_seed_panel
    python -m src.models.summarize_ablation_seed_panel --mode loo --feature prior_count_24h
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.compare_ablation import load_metrics, load_noise_floor, spread
from src.models.train_lightgbm_ablation import (
    FEATURE_KEYS,
    LEAVE_ONE_OUT,
    MODES,
    MODE_REFERENCE,
    REPORT_DIR,
    RELATION,
    SINGLETON,
    STOP_METRIC,
    resolve_feature,
    resolve_run_paths,
)
from src.models.train_lightgbm_baseline import RANDOM_SEED
from src.models.train_lightgbm_convergence_check import (
    DEFAULT_MAX_ESTIMATORS as CONVERGED_MAX_ESTIMATORS,
    resolve_run_paths as resolve_convergence_run_paths,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

RUNS_CSV = REPORT_DIR.parent / "ablation_seed_panel_runs.csv"
SUMMARY_JSON = REPORT_DIR.parent / "ablation_seed_panel_summary.json"

# The seed panel this project uses everywhere refit noise is measured.
PANEL_SEEDS = [42, 202, 707, 1337, 2024]

# A seed is contaminated when one model trained more than this many times as
# long as its opponent. See the module docstring for why this replaces the
# inherited cap-reached criterion, and when the threshold was fixed.
MAX_BEST_ITERATION_RATIO = 2.0

# Which convergence-check configuration backs each ablation mode's reference.
REFERENCE_CONFIG = {SINGLETON: "b0", LEAVE_ONE_OUT: "b1_card1"}

OUTCOME_SIGN_STABLE = "DELTA_SIGN_STABLE_ACROSS_SEEDS"
OUTCOME_SIGN_UNSTABLE = "DELTA_SIGN_UNSTABLE_ACROSS_SEEDS"
OUTCOME_PANEL_TOO_SMALL = "SEED_PANEL_TOO_SMALL"


def resolve_reference_paths(mode: str, seed: int) -> dict[str, Path]:
    """The converged reference this mode is read against, at a given seed."""
    if mode not in MODES:
        raise ValueError(f"Unknown ablation mode: {mode!r}. Supported: {MODES}.")
    return resolve_convergence_run_paths(
        REFERENCE_CONFIG[mode], STOP_METRIC, CONVERGED_MAX_ESTIMATORS, seed
    )


def discover_paired_seeds(mode: str, feature: str) -> list[int]:
    """Seeds where both the variant and its reference have completed.

    A seed with only one half trained is skipped rather than paired against
    another seed's reference -- pairing across seeds would mix refit noise in
    the variant with refit noise in the reference, which is the exact error
    this panel exists to measure.
    """
    paired = []
    for seed in PANEL_SEEDS:
        variant = resolve_run_paths(mode, feature, seed)
        reference = resolve_reference_paths(mode, seed)
        if variant["metrics"].exists() and reference["metrics"].exists():
            paired.append(seed)
    if not paired:
        raise FileNotFoundError(
            f"No seed has both a {mode}/{FEATURE_KEYS[resolve_feature(feature)]} run "
            f"and a converged {REFERENCE_CONFIG[mode]} reference. Train both halves "
            "before summarising."
        )
    return paired


def build_paired_table(mode: str, feature: str, seeds: list[int]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in seeds:
        variant = load_metrics(resolve_run_paths(mode, feature, seed)["metrics"])
        reference = load_metrics(resolve_reference_paths(mode, seed)["metrics"])

        variant_iterations = int(variant["best_iteration"])
        reference_iterations = int(reference["best_iteration"])
        ratio = max(variant_iterations, reference_iterations) / min(
            variant_iterations, reference_iterations
        )
        rows.append(
            {
                "seed": seed,
                "variant_pr_auc": float(variant["pr_auc"]),
                "variant_roc_auc": float(variant["roc_auc"]),
                "variant_best_iteration": variant_iterations,
                "reference_pr_auc": float(reference["pr_auc"]),
                "reference_roc_auc": float(reference["roc_auc"]),
                "reference_best_iteration": reference_iterations,
                "delta_pr_auc": float(variant["pr_auc"]) - float(reference["pr_auc"]),
                "delta_roc_auc": float(variant["roc_auc"]) - float(reference["roc_auc"]),
                "best_iteration_ratio": ratio,
                "early_stopping_clean": bool(ratio <= MAX_BEST_ITERATION_RATIO),
                "longer_trained": (
                    "variant" if variant_iterations > reference_iterations else "reference"
                ),
            }
        )
    return pd.DataFrame(rows)


def stratify(table: pd.DataFrame) -> dict[str, Any]:
    """Split the paired deltas on the best-iteration ratio and report both strata."""
    clean = table[table["early_stopping_clean"]]
    contaminated = table[~table["early_stopping_clean"]]

    result: dict[str, Any] = {
        "rule": (
            "A seed is contaminated when one model's best iteration exceeds the "
            f"other's by more than {MAX_BEST_ITERATION_RATIO:g}x. Early stopping "
            "selects the round maximising validation average precision, the "
            "reported metric, so a model granted far more rounds than its "
            "opponent wins partly on the stopping point rather than on its "
            "predictors."
        ),
        "why_not_the_inherited_rule": (
            "summarize_seed_variance strata on whether either model reached the "
            "estimator cap. At the converged cap (15,000) no run reaches it, so "
            "that criterion labels every seed identically and separates nothing."
        ),
        "threshold_fixed_when": (
            "Stated while three of the five seeds had been trained, before the "
            "remaining two existed. Both strata are always reported so a reader "
            "can apply a different threshold to the per-seed table."
        ),
        "max_best_iteration_ratio": MAX_BEST_ITERATION_RATIO,
        "n_clean": int(len(clean)),
        "n_contaminated": int(len(contaminated)),
        "clean_seeds": [int(s) for s in clean["seed"]],
        "contaminated_seeds": [int(s) for s in contaminated["seed"]],
    }
    if len(clean) >= 2:
        clean_deltas = clean["delta_pr_auc"].to_numpy()
        result["clean_delta_pr_auc"] = spread(clean_deltas)
        result["clean_delta_sign_stable"] = bool(
            (clean_deltas > 0).all() or (clean_deltas < 0).all()
        )
    if len(contaminated) >= 2:
        result["contaminated_delta_pr_auc"] = spread(
            contaminated["delta_pr_auc"].to_numpy()
        )

    # ROC-AUC is not the early-stopping criterion, so it is undistorted by the
    # stopping point and can be read across every seed -- the same reasoning
    # summarize_seed_variance applies.
    all_roc_deltas = table["delta_roc_auc"].to_numpy()
    result["roc_auc_delta_all_seeds"] = spread(all_roc_deltas)
    result["roc_auc_delta_sign_stable"] = bool(
        (all_roc_deltas > 0).all() or (all_roc_deltas < 0).all()
    )
    result["roc_auc_is_not_the_stopping_metric"] = True
    return result


def classify(table: pd.DataFrame, stratification: dict[str, Any]) -> dict[str, Any]:
    """Whether the paired delta holds its sign, and how it compares to its own spread."""
    clean = table[table["early_stopping_clean"]]
    if len(clean) < 2:
        return {
            "outcome_code": OUTCOME_PANEL_TOO_SMALL,
            "conclusion": (
                "Fewer than two clean seeds; the paired delta's spread cannot be "
                "estimated. Train more seeds before reading this variant's "
                "advantage as anything but a single draw."
            ),
        }

    deltas = clean["delta_pr_auc"].to_numpy()
    sign_stable = bool((deltas > 0).all() or (deltas < 0).all())
    mean_delta = float(deltas.mean())
    std_delta = float(deltas.std(ddof=1))
    magnitude_ratio = abs(mean_delta) / std_delta if std_delta > 0 else float("inf")

    if sign_stable:
        outcome_code = OUTCOME_SIGN_STABLE
        conclusion = (
            f"The paired delta keeps its sign across all {len(clean)} clean seeds "
            f"(mean {mean_delta:+.5f}, sd {std_delta:.5f}, {magnitude_ratio:.1f} sd "
            "from zero). The direction is a property of the predictors rather than "
            "of the seed. The magnitude, however, is what the seed panel revises: "
            "quote the clean-stratum mean, not the single-seed figure that "
            "motivated the check."
        )
    else:
        outcome_code = OUTCOME_SIGN_UNSTABLE
        conclusion = (
            f"The paired delta changes sign across the {len(clean)} clean seeds "
            f"(mean {mean_delta:+.5f}, sd {std_delta:.5f}). The single-seed result "
            "was a draw, not an effect, and no advantage should be claimed for "
            "this variant."
        )
    return {
        "outcome_code": outcome_code,
        "conclusion": conclusion,
        "clean_delta_sign_stable": sign_stable,
        "clean_delta_mean": mean_delta,
        "clean_delta_std": std_delta,
        "clean_delta_standard_deviations_from_zero": magnitude_ratio,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarise the seed panel for one ablation variant against its "
            "converged reference, paired seed by seed."
        )
    )
    parser.add_argument("--mode", choices=MODES, default=LEAVE_ONE_OUT)
    parser.add_argument("--feature", default="prior_count_24h")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    mode = args.mode
    feature = resolve_feature(args.feature)
    key = FEATURE_KEYS[feature]

    seeds = discover_paired_seeds(mode, feature)
    print(f"Variant:   {mode}_{key} ({RELATION})")
    print(f"Reference: {MODE_REFERENCE[mode]}")
    print(f"Paired seeds: {seeds}")

    table = build_paired_table(mode, feature, seeds)
    RUNS_CSV.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(RUNS_CSV, index=False)

    print(
        f"\n{'seed':>6} {'variant':>10} {'iter':>7} {'reference':>10} {'iter':>7} "
        f"{'delta':>11} {'ratio':>6}  stratum"
    )
    for row in table.itertuples(index=False):
        print(
            f"{row.seed:>6} {row.variant_pr_auc:>10.6f} {row.variant_best_iteration:>7,} "
            f"{row.reference_pr_auc:>10.6f} {row.reference_best_iteration:>7,} "
            f"{row.delta_pr_auc:>+11.6f} {row.best_iteration_ratio:>6.2f}  "
            f"{'clean' if row.early_stopping_clean else 'contaminated'}"
        )

    stratification = stratify(table)
    outcome = classify(table, stratification)
    noise_floor = load_noise_floor()

    summary = {
        "report_name": "Seed panel for a relational ablation variant",
        "relation": RELATION,
        "ablation_mode": mode,
        "ablated_feature": feature,
        "variant": f"{mode}_{key}",
        "reference_configuration": MODE_REFERENCE[mode],
        "protocol": "converged (cap 15,000, stop_average_precision)",
        "question": (
            "Does this variant's advantage over its reference survive refitting "
            "under several seeds, or was the single-seed result a draw?"
        ),
        "panel_seeds": PANEL_SEEDS,
        "paired_seeds": seeds,
        "panel_complete": bool(len(seeds) == len(PANEL_SEEDS)),
        "frozen_seed": RANDOM_SEED,
        "clean_stratum_noise_floor": noise_floor,
        "noise_floor_source": "reports/seed_variance/seed_variance_summary.json",
        "delta_pr_auc_all_seeds": spread(table["delta_pr_auc"].to_numpy()),
        "early_stopping_stratification": stratification,
        "outcome": outcome,
        "runs": table.to_dict(orient="records"),
        "runs_table_path": "reports/ablation/ablation_seed_panel_runs.csv",
        "test_evaluated": False,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"\nRuns table saved: {RUNS_CSV}")
    print(f"Summary saved: {SUMMARY_JSON}")
    if not summary["panel_complete"]:
        missing = sorted(set(PANEL_SEEDS) - set(seeds))
        print(f"\nWARNING: seeds without both halves trained: {missing}")
    print(f"\nOutcome: [{outcome['outcome_code']}] {outcome['conclusion']}")


if __name__ == "__main__":
    main()
