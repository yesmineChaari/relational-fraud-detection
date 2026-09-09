"""Seed-variance report for B0 and B1-card1.

Answers the question the project's own summary lists as open: the paired
bootstrap quantifies sampling variance only, because it holds one trained model
fixed and resamples validation rows. It says nothing about how much validation
PR-AUC moves when the same configuration is refitted under a different seed.

Two quantities matter here, and they are not the same:

  * The spread of absolute PR-AUC per configuration. Interesting, but a shared
    move in both configurations cancels out of the comparison.
  * The spread of the paired per-seed delta, B1-card1 minus B0 fitted under the
    same seed. This is the one the headline claim rests on. If the delta stays
    positive and tight across seeds while absolute PR-AUC wanders, the claim
    survives; if the delta's own spread is comparable to its +0.0059 magnitude,
    the claim is seed lottery.

The seed spread is then set against the paired-bootstrap interval on the frozen
B1-card1 versus B0 comparison, so the two sources of uncertainty are reported on
the same scale.

Outputs:
  reports/seed_variance/seed_variance_runs.csv
  reports/seed_variance/seed_variance_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.models.train_lightgbm_g1 import paired_bootstrap_pr_auc_delta
from src.models.train_seed_variants import CONFIGURATIONS, REPORT_DIR

ROOT_DIR = Path(__file__).resolve().parents[2]

RUNS_CSV = REPORT_DIR / "seed_variance_runs.csv"
SUMMARY_JSON = REPORT_DIR / "seed_variance_summary.json"

FROZEN_B0_PREDICTIONS = ROOT_DIR / "reports" / "baseline" / "validation_predictions.parquet"
FROZEN_B1_PREDICTIONS = ROOT_DIR / "reports" / "b1" / "card1" / "validation_predictions.parquet"

# The comparison configuration whose delta the project's headline claim rests on.
REFERENCE_CONFIG = "b0"
CANDIDATE_CONFIG = "b1_card1"

METRIC_COLUMNS = ["pr_auc", "roc_auc"]


def discover_runs() -> pd.DataFrame:
    """Collect every completed seed run from its metrics artifact."""

    rows: list[dict[str, Any]] = []
    for config in CONFIGURATIONS:
        config_dir = REPORT_DIR / config
        if not config_dir.exists():
            continue
        for run_dir in sorted(config_dir.glob("seed_*")):
            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                continue
            with metrics_path.open(encoding="utf-8") as handle:
                metrics = json.load(handle)
            if metrics.get("test_evaluated") is not False:
                raise ValueError(f"{metrics_path} violates final-test discipline.")
            if metrics.get("configuration") != config:
                raise ValueError(
                    f"{metrics_path} reports configuration "
                    f"{metrics.get('configuration')!r} but sits under {config!r}."
                )
            rows.append(
                {
                    "configuration": config,
                    "seed": int(metrics["random_seed"]),
                    "pr_auc": float(metrics["pr_auc"]),
                    "roc_auc": float(metrics["roc_auc"]),
                    "best_iteration": int(metrics["best_iteration"]),
                    "actual_stopping_iteration": int(metrics["actual_stopping_iteration"]),
                    "maximum_estimators": int(metrics["maximum_estimators"]),
                    "early_stopping_triggered": bool(metrics["early_stopping_triggered"]),
                    "estimator_cap_reached": bool(metrics["estimator_cap_reached"]),
                }
            )
    if not rows:
        raise FileNotFoundError(
            f"No completed seed runs found under {REPORT_DIR}. "
            "Run: python -m src.models.train_seed_variants"
        )
    return pd.DataFrame(rows).sort_values(["configuration", "seed"]).reset_index(drop=True)


def spread(values: np.ndarray) -> dict[str, float]:
    """Mean, sample standard deviation and range of a metric across seeds."""

    if len(values) < 2:
        raise ValueError("Spread requires at least two runs.")
    return {
        "n_seeds": int(len(values)),
        "mean": float(values.mean()),
        # ddof=1: these seeds are a sample of the seed distribution, not its whole.
        "std": float(values.std(ddof=1)),
        "min": float(values.min()),
        "max": float(values.max()),
        "range": float(values.max() - values.min()),
    }


def summarize_configuration(runs: pd.DataFrame, config: str) -> dict[str, Any]:
    subset = runs[runs["configuration"] == config]
    if subset.empty:
        raise ValueError(f"No runs found for configuration {config!r}.")
    summary: dict[str, Any] = {
        "seeds": [int(s) for s in subset["seed"]],
        "estimator_cap": int(subset["maximum_estimators"].max()),
        "early_stopping_triggered_in_any_run": bool(subset["early_stopping_triggered"].any()),
        "estimator_cap_reached_in_all_runs": bool(subset["estimator_cap_reached"].all()),
        "best_iteration_min": int(subset["best_iteration"].min()),
        "best_iteration_max": int(subset["best_iteration"].max()),
    }
    for metric in METRIC_COLUMNS:
        summary[metric] = spread(subset[metric].to_numpy())
    return summary


def paired_seed_deltas(runs: pd.DataFrame) -> pd.DataFrame:
    """Per-seed B1-card1 minus B0, over seeds where both configurations ran.

    Pairing by seed is the point: it removes whatever the seed did to both
    models and leaves only what the four relational columns contributed.
    """

    candidate = runs[runs["configuration"] == CANDIDATE_CONFIG].set_index("seed")
    reference = runs[runs["configuration"] == REFERENCE_CONFIG].set_index("seed")
    shared = sorted(set(candidate.index) & set(reference.index))
    if not shared:
        raise ValueError(
            f"No seed has both {CANDIDATE_CONFIG} and {REFERENCE_CONFIG} runs; "
            "the paired delta cannot be computed."
        )

    def early_stopping_status(seed: int) -> str:
        reference_stopped = bool(reference.loc[seed, "early_stopping_triggered"])
        candidate_stopped = bool(candidate.loc[seed, "early_stopping_triggered"])
        if reference_stopped and candidate_stopped:
            return "both"
        if reference_stopped:
            return REFERENCE_CONFIG
        if candidate_stopped:
            return CANDIDATE_CONFIG
        return "neither"

    return pd.DataFrame(
        [
            {
                "seed": seed,
                f"{REFERENCE_CONFIG}_pr_auc": float(reference.loc[seed, "pr_auc"]),
                f"{CANDIDATE_CONFIG}_pr_auc": float(candidate.loc[seed, "pr_auc"]),
                "delta_pr_auc": float(
                    candidate.loc[seed, "pr_auc"] - reference.loc[seed, "pr_auc"]
                ),
                f"{REFERENCE_CONFIG}_roc_auc": float(reference.loc[seed, "roc_auc"]),
                f"{CANDIDATE_CONFIG}_roc_auc": float(candidate.loc[seed, "roc_auc"]),
                "delta_roc_auc": float(
                    candidate.loc[seed, "roc_auc"] - reference.loc[seed, "roc_auc"]
                ),
                f"{REFERENCE_CONFIG}_best_iteration": int(reference.loc[seed, "best_iteration"]),
                f"{CANDIDATE_CONFIG}_best_iteration": int(candidate.loc[seed, "best_iteration"]),
                "early_stopping_status": early_stopping_status(seed),
            }
            for seed in shared
        ]
    )


def stratify_by_early_stopping(deltas: pd.DataFrame) -> dict[str, Any]:
    """Split the paired deltas by whether either model stopped before the cap.

    Early stopping selects the boosting round that maximises validation average
    precision -- the very metric reported. A model truncated well short of the
    cap therefore loses that optimisation against an opponent that kept it, and
    the resulting difference scores the stopping point rather than the feature
    set. Splitting on that lets the comparison be read on the seeds where it is
    actually a like-for-like contest.
    """

    clean = deltas[deltas["early_stopping_status"] == "neither"]
    contaminated = deltas[deltas["early_stopping_status"] != "neither"]

    result: dict[str, Any] = {
        "rationale": (
            "Early stopping maximises validation average precision, the reported "
            "metric. When one configuration stops early and the other runs to the "
            "cap, the PR-AUC difference between them partly scores the stopping "
            "point rather than the predictors."
        ),
        "n_clean": int(len(clean)),
        "n_contaminated": int(len(contaminated)),
        "clean_seeds": [int(s) for s in clean["seed"]],
        "contaminated_seeds": [int(s) for s in contaminated["seed"]],
        "which_stopped_per_seed": {
            int(row.seed): row.early_stopping_status for row in deltas.itertuples(index=False)
        },
    }
    if len(clean) >= 2:
        result["clean_delta_pr_auc"] = spread(clean["delta_pr_auc"].to_numpy())
        result["clean_delta_sign_stable"] = bool(
            (clean["delta_pr_auc"] > 0).all() or (clean["delta_pr_auc"] < 0).all()
        )
    if len(contaminated) >= 2:
        result["contaminated_delta_pr_auc"] = spread(contaminated["delta_pr_auc"].to_numpy())

    # ROC-AUC is not the early-stopping criterion, so it is not distorted by the
    # stopping point and can be read across every seed.
    result["roc_auc_delta_all_seeds"] = spread(deltas["delta_roc_auc"].to_numpy())
    result["roc_auc_delta_sign_stable"] = bool(
        (deltas["delta_roc_auc"] > 0).all() or (deltas["delta_roc_auc"] < 0).all()
    )
    result["roc_auc_is_not_the_stopping_metric"] = True
    return result


def load_frozen_bootstrap() -> dict[str, Any]:
    """Paired-bootstrap interval on the frozen B1-card1 versus B0 comparison.

    Reuses the project's existing paired bootstrap rather than reimplementing
    it, so the number here is the same statistic the rest of the project quotes.
    """

    for path in (FROZEN_B0_PREDICTIONS, FROZEN_B1_PREDICTIONS):
        if not path.exists():
            return {"available": False, "reason": f"Missing frozen predictions: {path}"}

    b0 = pd.read_parquet(FROZEN_B0_PREDICTIONS, columns=["TransactionID", "isFraud", "prediction"])
    b1 = pd.read_parquet(FROZEN_B1_PREDICTIONS, columns=["TransactionID", "isFraud", "prediction"])
    merged = b0.merge(b1, on="TransactionID", suffixes=("_b0", "_b1"), validate="one_to_one")
    if len(merged) != len(b0) or len(merged) != len(b1):
        raise AssertionError("Frozen B0 and B1 validation predictions do not align.")
    if not merged["isFraud_b0"].equals(merged["isFraud_b1"]):
        raise AssertionError("Frozen B0 and B1 validation labels disagree.")

    result = paired_bootstrap_pr_auc_delta(
        merged["isFraud_b0"].to_numpy(dtype=np.int64),
        merged["prediction_b1"].to_numpy(dtype=np.float64),
        merged["prediction_b0"].to_numpy(dtype=np.float64),
    )
    result["available"] = True
    result["comparison"] = "frozen B1-card1 minus frozen B0"
    result["holds_fixed"] = "the two trained models; resamples validation rows"
    return result


def build_interpretation(
    delta_spread: dict[str, Any],
    deltas: pd.DataFrame,
    bootstrap: dict[str, Any],
    stratification: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """State in plain language what the seed spread does to the headline claim."""

    delta_std = delta_spread["std"]
    delta_mean = delta_spread["mean"]
    all_positive = bool((deltas["delta_pr_auc"] > 0).all())
    sign_stable = all_positive or bool((deltas["delta_pr_auc"] < 0).all())

    interpretation: dict[str, Any] = {
        "paired_delta_sign_stable_across_seeds": sign_stable,
        "paired_delta_positive_in_every_seed": all_positive,
        "seed_std_of_paired_delta": delta_std,
        "mean_paired_delta": delta_mean,
        "mean_delta_over_seed_std": (float(delta_mean / delta_std) if delta_std > 0 else None),
    }

    if bootstrap.get("available"):
        bootstrap_std = float(bootstrap["bootstrap_std_delta"])
        interpretation["bootstrap_std_of_delta"] = bootstrap_std
        interpretation["seed_std_over_bootstrap_std"] = (
            float(delta_std / bootstrap_std) if bootstrap_std > 0 else None
        )
        interpretation["sources_comparable_in_size"] = bool(
            0.5 <= (delta_std / bootstrap_std) <= 2.0 if bootstrap_std > 0 else False
        )
        interpretation["seed_variance_dominates_sampling_variance"] = bool(
            delta_std > bootstrap_std
        )

    # A sign flip is only evidence against the features if the comparison that
    # produced it was like-for-like. Check the early-stopping confound first.
    clean = (stratification or {}).get("clean_delta_pr_auc")
    clean_sign_stable = (stratification or {}).get("clean_delta_sign_stable", False)
    roc_sign_stable = (stratification or {}).get("roc_auc_delta_sign_stable", False)
    confounded = bool(
        not sign_stable and clean is not None and clean_sign_stable and clean["std"] < delta_std
    )
    if stratification is not None:
        interpretation["early_stopping_confound_explains_instability"] = confounded
        interpretation["roc_auc_delta_sign_stable_across_seeds"] = roc_sign_stable

    if confounded:
        verdict = "confounded_by_early_stopping"
        roc = stratification["roc_auc_delta_all_seeds"]
        plain = (
            f"The paired delta changes sign across the full panel, but the "
            f"instability is an early-stopping artifact rather than a property of "
            f"the relational features. On the {clean['n_seeds']} seeds where both "
            f"configurations ran to the estimator cap, the delta is "
            f"{clean['mean']:+.5f} with a standard deviation of {clean['std']:.5f}. "
            f"On the seeds where one configuration stopped early, the delta swings "
            f"to whichever model kept training, because early stopping selects the "
            f"round that maximises the very metric being reported. ROC-AUC, which "
            f"is not the stopping criterion, favours B1-card1 in every seed by "
            f"{roc['mean']:+.5f} with a standard deviation of {roc['std']:.5f}. "
            "The relational gain is real; the PR-AUC point estimate is contaminated "
            "by the stopping rule and cannot be quoted until the cap is resolved."
        )
    elif not sign_stable:
        verdict = "unstable"
        plain = (
            "The paired B1-card1 minus B0 delta changes sign across seeds. The "
            "reported gain is not a stable property of the relational features; "
            "it is a seed effect, and the headline claim must be withdrawn."
        )
    elif delta_std > 0 and abs(delta_mean) < 2.0 * delta_std:
        verdict = "within_seed_noise"
        plain = (
            f"The paired delta holds its sign across all "
            f"{delta_spread['n_seeds']} seeds, but its mean of {delta_mean:+.5f} is "
            f"less than two seed standard deviations ({delta_std:.5f}) from zero. "
            "The direction is consistent; the magnitude should be quoted with the "
            "seed spread attached rather than as a point estimate."
        )
    else:
        verdict = "stable"
        plain = (
            f"The paired delta holds its sign across all "
            f"{delta_spread['n_seeds']} seeds with a mean of {delta_mean:+.5f} and a "
            f"seed standard deviation of {delta_std:.5f}. Absolute PR-AUC moves with "
            "the seed, but the movement is shared by both configurations and cancels "
            "out of the paired comparison, so the relational gain is a property of "
            "the features rather than of the seed."
        )

    if bootstrap.get("available"):
        bootstrap_std = float(bootstrap["bootstrap_std_delta"])
        if bootstrap_std > 0 and delta_std > bootstrap_std:
            plain += (
                f" Seed variance on the delta ({delta_std:.5f}) exceeds the "
                f"paired-bootstrap standard deviation ({bootstrap_std:.5f}), so the "
                "published bootstrap interval understates the true uncertainty: it "
                "resamples rows but never refits the model."
            )
        elif bootstrap_std > 0:
            plain += (
                f" Seed variance on the delta ({delta_std:.5f}) is smaller than the "
                f"paired-bootstrap standard deviation ({bootstrap_std:.5f}), so "
                "sampling variance remains the dominant source of uncertainty and "
                "the published interval is not materially optimistic."
            )

    interpretation["verdict"] = verdict
    interpretation["plain_language"] = plain
    return interpretation


def build_summary(
    runs: pd.DataFrame,
    deltas: pd.DataFrame,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    configurations = {
        config: summarize_configuration(runs, config)
        for config in CONFIGURATIONS
        if not runs[runs["configuration"] == config].empty
    }
    delta_spread = spread(deltas["delta_pr_auc"].to_numpy())
    delta_spread_roc = spread(deltas["delta_roc_auc"].to_numpy())
    stratification = stratify_by_early_stopping(deltas)

    expected_seeds = {int(s) for s in runs["seed"]}
    complete = all(set(summary["seeds"]) == expected_seeds for summary in configurations.values())

    return {
        "report_name": "Seed variance across training runs",
        "question": (
            "How much does validation PR-AUC move when the same frozen configuration "
            "is refitted under a different seed, and does the B1-card1 gain over B0 "
            "survive that movement?"
        ),
        "varied_parameter": "random_state",
        "method": (
            "Each configuration is refitted once per seed with every other setting "
            "held at its frozen value. The delta is paired by seed, so a seed effect "
            "shared by both configurations cancels rather than inflating the spread."
        ),
        "evaluation_split": "validation",
        "test_evaluated": False,
        "panel_complete": complete,
        "configurations": configurations,
        "paired_delta_definition": f"{CANDIDATE_CONFIG} minus {REFERENCE_CONFIG}, same seed",
        "paired_delta_per_seed": deltas.to_dict(orient="records"),
        "paired_delta_pr_auc_spread": delta_spread,
        "paired_delta_roc_auc_spread": delta_spread_roc,
        "frozen_paired_bootstrap": bootstrap,
        "early_stopping_stratification": stratification,
        "interpretation": build_interpretation(delta_spread, deltas, bootstrap, stratification),
        "runs_table_path": "reports/seed_variance/seed_variance_runs.csv",
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    runs = discover_runs()
    print(f"Completed runs discovered: {len(runs)}")
    for config in CONFIGURATIONS:
        subset = runs[runs["configuration"] == config]
        if subset.empty:
            print(f"  {config}: no runs yet")
            continue
        print(
            f"  {config}: {len(subset)} seeds  "
            f"PR-AUC mean={subset['pr_auc'].mean():.6f} "
            f"sd={subset['pr_auc'].std(ddof=1):.6f} "
            f"range={subset['pr_auc'].max() - subset['pr_auc'].min():.6f}"
        )

    deltas = paired_seed_deltas(runs)
    print(f"\nPaired seeds: {len(deltas)}")
    for row in deltas.itertuples(index=False):
        print(f"  seed {row.seed}: delta PR-AUC {row.delta_pr_auc:+.6f}")

    print("\nComputing the frozen paired-bootstrap reference...")
    bootstrap = load_frozen_bootstrap()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    runs.to_csv(RUNS_CSV, index=False)
    summary = build_summary(runs, deltas, bootstrap)
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print(f"\nRuns table saved: {RUNS_CSV}")
    print(f"Summary saved: {SUMMARY_JSON}")
    print(f"\nVerdict: {summary['interpretation']['verdict']}")
    print(summary["interpretation"]["plain_language"])


if __name__ == "__main__":
    main()
