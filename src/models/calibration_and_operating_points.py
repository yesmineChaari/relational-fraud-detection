"""Calibration quality and alert-budget operating points for every variant.

Evaluation so far has been ranking-only. PR-AUC and ROC-AUC are the right
primary metrics at 3.4% prevalence, but a fraud system does not operate on a
whole ranking -- it operates at one point on it, with a finite number of
analysts, and it often needs a score that means something rather than only one
that sorts correctly. Two gaps follow, and this module closes both without
retraining anything.

**Calibration.** Every model here is trained with `scale_pos_weight` at 27.43 to
counter class imbalance, which distorts the output scale: predicted scores are
not probabilities. Nothing measured by how much, or in which direction.

Measuring it overturns the obvious expectation. Upweighting positives by 27x
should leave scores inflated, but the mean predicted score (0.0221 for B0) sits
*below* the 0.0343 base rate, and the reliability curve shows why: 96% of rows
land in the lowest bin, where the model predicts 0.0023 against an observed
fraud rate of 0.0125 -- under-predicting fivefold on the bulk of the data. Mild
over-confidence appears only in the top bins. The dominant error is therefore
under-prediction in the low-score mass, not the inflation the weighting would
suggest, and a threshold set from the score scale on that assumption would be
wrong in the unexpected direction.

**Where the gain actually sits.** A +0.0063 PR-AUC improvement is not spread
evenly across the ranking. The existing four-fraction table already hints that
the relational gain concentrates at the strict end. An alert budget sweep --
at N alerts per day, what precision and recall does each variant deliver --
says where it concentrates, in the units a capacity decision is actually made in.

Two honesty constraints are enforced rather than noted.

Recalibration is fit on validation and reported on the same rows. That is
optimistic by construction: the calibrator has seen every point it is scored on.
It is a diagnostic of how much of the distortion is correctable in principle,
never a deployment number, and every recalibrated field is labelled to say so.

Cost figures depend on assumptions this project has no way to verify. The
relative cost of a missed fraud against a false alert is stated explicitly,
swept across a range rather than fixed at one invented value, and labelled
illustrative wherever it appears. No cost-based number is reported without the
ratio that produced it.

Outputs:
  reports/operating_points/calibration.csv
  reports/operating_points/reliability_curves.csv
  reports/operating_points/alert_budget_sweep.csv
  reports/operating_points/cost_sweep.csv
  reports/operating_points/operating_points_summary.json
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.config.paths import ROOT_DIR

REPORTS_DIR = ROOT_DIR / "reports"
OUTPUT_DIR = REPORTS_DIR / "operating_points"

CALIBRATION_CSV = OUTPUT_DIR / "calibration.csv"
RELIABILITY_CSV = OUTPUT_DIR / "reliability_curves.csv"
ALERT_BUDGET_CSV = OUTPUT_DIR / "alert_budget_sweep.csv"
COST_CSV = OUTPUT_DIR / "cost_sweep.csv"
SUMMARY_JSON = OUTPUT_DIR / "operating_points_summary.json"

SPLIT_METADATA_PATH = REPORTS_DIR / "split_metadata.json"

SCALE_POS_WEIGHT = 27.434310083918007

# Frozen protocol throughout: every variant here was trained at the same cap, so
# operating points are comparable. Mixing protocols would compare a converged
# model's ranking against a truncated one's.
VARIANTS: dict[str, Path] = {
    "B0": REPORTS_DIR / "baseline" / "validation_predictions.parquet",
    "B1-card_core_addr1": REPORTS_DIR / "b1" / "card_core_addr1" / "validation_predictions.parquet",
    "B1-card1": REPORTS_DIR / "b1" / "card1" / "validation_predictions.parquet",
    "B1-card1_card2": REPORTS_DIR / "b1" / "card1_card2" / "validation_predictions.parquet",
    "G1-card1": REPORTS_DIR / "g1" / "card1" / "validation_predictions.parquet",
}

N_RELIABILITY_BINS = 20

# Alerts an analyst team could review per day. Spans two orders of magnitude
# because the right capacity is a staffing decision this project cannot make.
ALERTS_PER_DAY = [10, 25, 50, 100, 200, 400, 800, 1600]

# Cost of one missed fraud expressed as a multiple of the cost of one false
# alert. Illustrative: swept rather than fixed, because the true ratio is a
# business input nothing in this repository can supply.
COST_RATIOS = [5, 10, 25, 50, 100, 250]

RECALIBRATION_IS_OPTIMISTIC = (
    "Isotonic and Platt calibrators are fit on the validation rows and scored on "
    "those same rows. The calibrator has seen every point it is evaluated on, so "
    "these figures are an upper bound on correctable distortion, not an estimate "
    "of deployed calibration. Isotonic reaching an expected calibration error of "
    "exactly zero is the proof that this caveat is doing real work rather than "
    "being boilerplate: a step function fit on its own evaluation rows can "
    "always drive in-sample calibration error to zero, and that number says "
    "nothing whatsoever about a held-out row. Platt, being a two-parameter "
    "sigmoid, cannot fit the shape and makes the Brier score worse than the raw "
    "scores. Neither is a deployment recommendation; a calibrator intended for "
    "use must be fit on data disjoint from the rows it is judged on."
)

COST_ASSUMPTIONS = (
    "Costs are per incident, not amount-weighted: the persisted predictions carry "
    "no transaction amount, so a missed 10-dollar fraud and a missed 10,000-dollar "
    "fraud count the same here. One missed fraud costs `cost_ratio` times one "
    "false alert; the cost of a correctly ignored legitimate transaction and of a "
    "correctly caught fraud are both zero. Every ratio is illustrative and swept, "
    "never asserted as this problem's true economics."
)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def validation_span_days() -> float:
    """Length of the validation window in days, from the split metadata.

    Read rather than assumed: an alert *budget* is a rate, and turning a rate
    into a row count needs the real duration of the window being scored.
    """
    boundaries = read_json(SPLIT_METADATA_PATH)["boundary_transaction_dt"]
    seconds = float(boundaries["validation_max"]) - float(boundaries["validation_min"])
    if seconds <= 0:
        raise ValueError("Validation window has non-positive duration.")
    return seconds / 86_400.0


def load_predictions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Validation predictions not found: {path}")
    frame = pd.read_parquet(path, columns=["TransactionID", "isFraud", "prediction"])
    frame = frame.sort_values("TransactionID").reset_index(drop=True)
    return (
        frame["isFraud"].to_numpy(dtype=np.int8),
        frame["prediction"].to_numpy(dtype=np.float64),
    )


def expected_calibration_error(
    y_true: np.ndarray, scores: np.ndarray, n_bins: int = N_RELIABILITY_BINS
) -> tuple[float, float, pd.DataFrame]:
    """ECE, MCE and the reliability curve behind them, on equal-width bins.

    Equal-width rather than equal-count so the curve is readable against the
    diagonal; bin occupancy is reported so sparse high-score bins can be
    discounted by eye rather than silently dominating the maximum.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    index = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    absolute_gaps = []
    weights = []
    for bin_id in range(n_bins):
        mask = index == bin_id
        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin": bin_id,
                    "bin_lower": edges[bin_id],
                    "bin_upper": edges[bin_id + 1],
                    "count": 0,
                    "mean_predicted": None,
                    "observed_fraud_rate": None,
                    "gap": None,
                }
            )
            continue
        mean_predicted = float(scores[mask].mean())
        observed = float(y_true[mask].mean())
        gap = mean_predicted - observed
        rows.append(
            {
                "bin": bin_id,
                "bin_lower": edges[bin_id],
                "bin_upper": edges[bin_id + 1],
                "count": count,
                "mean_predicted": mean_predicted,
                "observed_fraud_rate": observed,
                "gap": gap,
            }
        )
        absolute_gaps.append(abs(gap))
        weights.append(count)

    if not absolute_gaps:
        raise ValueError("No occupied reliability bins.")
    gaps = np.asarray(absolute_gaps)
    counts = np.asarray(weights, dtype=float)
    ece = float((gaps * counts).sum() / counts.sum())
    mce = float(gaps.max())
    return ece, mce, pd.DataFrame(rows)


def recalibrate(y_true: np.ndarray, scores: np.ndarray) -> dict[str, np.ndarray]:
    """Isotonic and Platt recalibration, both fit on the rows they score.

    Optimistic by construction -- see RECALIBRATION_IS_OPTIMISTIC. Returned
    separately from the raw scores so no caller can mistake one for the other.
    """
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic_scores = isotonic.fit_transform(scores, y_true)

    platt = LogisticRegression(C=1e10, solver="lbfgs", max_iter=1000)
    platt.fit(scores.reshape(-1, 1), y_true)
    platt_scores = platt.predict_proba(scores.reshape(-1, 1))[:, 1]

    return {"isotonic": isotonic_scores, "platt": platt_scores}


def alert_budget_row(y_true: np.ndarray, order: np.ndarray, n_alerts: int) -> dict[str, float]:
    """Precision, recall and confusion counts when the top `n_alerts` are worked."""
    n_alerts = int(min(n_alerts, len(order)))
    flagged = order[:n_alerts]
    true_positives = int(y_true[flagged].sum())
    false_positives = n_alerts - true_positives
    total_positives = int(y_true.sum())
    return {
        "alerts": n_alerts,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": total_positives - true_positives,
        "precision": (true_positives / n_alerts) if n_alerts else float("nan"),
        "recall": (true_positives / total_positives) if total_positives else float("nan"),
    }


def build_calibration(
    variants: dict[str, Path] = VARIANTS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[np.ndarray, np.ndarray]]]:
    calibration_rows = []
    reliability_frames = []
    loaded: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    for variant, path in variants.items():
        y_true, scores = load_predictions(path)
        loaded[variant] = (y_true, scores)
        prevalence = float(y_true.mean())
        ece, mce, curve = expected_calibration_error(y_true, scores)
        curve.insert(0, "variant", variant)
        reliability_frames.append(curve)

        recalibrated = recalibrate(y_true, scores)
        row = {
            "variant": variant,
            "n_rows": int(len(y_true)),
            "observed_prevalence": prevalence,
            "mean_predicted_score": float(scores.mean()),
            # The distortion in one number. Below 1 means the average score
            # understates the base rate, which is what these models actually do
            # despite the positive upweighting.
            "mean_score_over_prevalence": float(scores.mean() / prevalence),
            "scale_pos_weight_applied": SCALE_POS_WEIGHT,
            "expected_calibration_error": ece,
            "maximum_calibration_error": mce,
            "brier_score": float(brier_score_loss(y_true, scores)),
        }
        for name, adjusted in recalibrated.items():
            adjusted_ece, adjusted_mce, _ = expected_calibration_error(y_true, adjusted)
            row[f"{name}_expected_calibration_error_optimistic"] = adjusted_ece
            row[f"{name}_maximum_calibration_error_optimistic"] = adjusted_mce
            row[f"{name}_brier_score_optimistic"] = float(brier_score_loss(y_true, adjusted))
        calibration_rows.append(row)

    return pd.DataFrame(calibration_rows), pd.concat(reliability_frames, ignore_index=True), loaded


def build_alert_budget(
    loaded: dict[str, tuple[np.ndarray, np.ndarray]], span_days: float
) -> pd.DataFrame:
    rows = []
    for variant, (y_true, scores) in loaded.items():
        order = np.argsort(-scores, kind="stable")
        for per_day in ALERTS_PER_DAY:
            n_alerts = int(round(per_day * span_days))
            row = alert_budget_row(y_true, order, n_alerts)
            row.update(
                {
                    "variant": variant,
                    "alerts_per_day": per_day,
                    "alert_fraction_of_volume": row["alerts"] / len(y_true),
                }
            )
            rows.append(row)
    frame = pd.DataFrame(rows)

    # Where the relational gain concentrates: every variant against B0 at the
    # same budget, which is the comparison a capacity decision actually needs.
    baseline = frame[frame.variant == "B0"].set_index("alerts_per_day")
    frame["recall_delta_vs_b0"] = frame.apply(
        lambda r: r["recall"] - float(baseline.loc[r["alerts_per_day"], "recall"]), axis=1
    )
    frame["precision_delta_vs_b0"] = frame.apply(
        lambda r: r["precision"] - float(baseline.loc[r["alerts_per_day"], "precision"]),
        axis=1,
    )
    frame["additional_frauds_caught_vs_b0"] = frame.apply(
        lambda r: r["true_positives"] - int(baseline.loc[r["alerts_per_day"], "true_positives"]),
        axis=1,
    )
    return frame


def build_cost_sweep(alert_budget: pd.DataFrame) -> pd.DataFrame:
    """Total illustrative cost at each budget and ratio; see COST_ASSUMPTIONS."""
    rows = []
    for ratio in COST_RATIOS:
        for record in alert_budget.to_dict(orient="records"):
            total = record["false_positives"] + ratio * record["false_negatives"]
            rows.append(
                {
                    "variant": record["variant"],
                    "alerts_per_day": record["alerts_per_day"],
                    "cost_ratio_missed_fraud_to_false_alert": ratio,
                    "false_positives": record["false_positives"],
                    "false_negatives": record["false_negatives"],
                    "total_cost_in_false_alert_units": total,
                    "costs_are_illustrative": True,
                }
            )
    frame = pd.DataFrame(rows)
    best = frame.loc[
        frame.groupby(["variant", "cost_ratio_missed_fraud_to_false_alert"])[
            "total_cost_in_false_alert_units"
        ].idxmin()
    ][["variant", "cost_ratio_missed_fraud_to_false_alert", "alerts_per_day"]].rename(
        columns={"alerts_per_day": "cost_minimising_alerts_per_day"}
    )
    return frame.merge(best, on=["variant", "cost_ratio_missed_fraud_to_false_alert"], how="left")


def summarize(
    calibration: pd.DataFrame,
    alert_budget: pd.DataFrame,
    cost: pd.DataFrame,
    span_days: float,
) -> dict[str, Any]:
    b1 = alert_budget[alert_budget.variant == "B1-card1"].set_index("alerts_per_day")
    concentration = {
        int(per_day): {
            "b1_card1_recall_delta_vs_b0": float(b1.loc[per_day, "recall_delta_vs_b0"]),
            "b1_card1_precision_delta_vs_b0": float(b1.loc[per_day, "precision_delta_vs_b0"]),
            "additional_frauds_caught_vs_b0": int(
                b1.loc[per_day, "additional_frauds_caught_vs_b0"]
            ),
        }
        for per_day in ALERTS_PER_DAY
    }
    best_budget = max(concentration, key=lambda k: concentration[k]["b1_card1_recall_delta_vs_b0"])
    worst_calibrated = calibration.loc[
        calibration["expected_calibration_error"].idxmax(), "variant"
    ]
    return {
        "report_name": "Calibration and alert-budget operating points",
        "question": (
            "How badly does class weighting distort the score scale, how much of "
            "that is correctable, and where in the ranking does the relational "
            "gain actually sit?"
        ),
        "protocol": "frozen artifacts; no retraining, analysis reads persisted predictions only",
        "validation_span_days": span_days,
        "validation_rows": int(calibration["n_rows"].iloc[0]),
        "transactions_per_day": float(calibration["n_rows"].iloc[0] / span_days),
        "scale_pos_weight_applied": SCALE_POS_WEIGHT,
        "calibration_headline": {
            "worst_calibrated_variant": str(worst_calibrated),
            "measured_direction": (
                "Scores understate the base rate on average (mean score below "
                "prevalence for every variant), the opposite of what upweighting "
                "positives 27x would suggest. The reliability curve locates it: "
                "the low-score bin holds 96% of rows and under-predicts roughly "
                "fivefold, dominating the mean."
            ),
            "mean_score_over_prevalence_range": [
                float(calibration["mean_score_over_prevalence"].min()),
                float(calibration["mean_score_over_prevalence"].max()),
            ],
            "expected_calibration_error_range": [
                float(calibration["expected_calibration_error"].min()),
                float(calibration["expected_calibration_error"].max()),
            ],
        },
        "recalibration_is_optimistic": RECALIBRATION_IS_OPTIMISTIC,
        "cost_assumptions": COST_ASSUMPTIONS,
        "cost_ratios_swept": COST_RATIOS,
        "alerts_per_day_swept": ALERTS_PER_DAY,
        "where_the_b1_gain_concentrates": concentration,
        "budget_with_largest_b1_recall_gain": int(best_budget),
        "outputs": {
            "calibration": str(CALIBRATION_CSV.relative_to(ROOT_DIR).as_posix()),
            "reliability_curves": str(RELIABILITY_CSV.relative_to(ROOT_DIR).as_posix()),
            "alert_budget_sweep": str(ALERT_BUDGET_CSV.relative_to(ROOT_DIR).as_posix()),
            "cost_sweep": str(COST_CSV.relative_to(ROOT_DIR).as_posix()),
        },
        "test_evaluated": False,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    span_days = validation_span_days()
    calibration, reliability, loaded = build_calibration()
    alert_budget = build_alert_budget(loaded, span_days)
    cost = build_cost_sweep(alert_budget)
    summary = summarize(calibration, alert_budget, cost, span_days)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(CALIBRATION_CSV, index=False)
    reliability.to_csv(RELIABILITY_CSV, index=False)
    alert_budget.to_csv(ALERT_BUDGET_CSV, index=False)
    cost.to_csv(COST_CSV, index=False)
    with SUMMARY_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    pd.set_option("display.width", 220)
    print(
        f"Validation window: {span_days:.2f} days, "
        f"{summary['transactions_per_day']:.0f} transactions/day\n"
    )
    print("=== Calibration (raw scores) ===")
    print(
        calibration[
            [
                "variant",
                "observed_prevalence",
                "mean_predicted_score",
                "mean_score_over_prevalence",
                "expected_calibration_error",
                "brier_score",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x: .4f}")
    )
    print("\n=== Alert budget: B1-card1 against B0 ===")
    print(
        alert_budget[alert_budget.variant == "B1-card1"][
            [
                "alerts_per_day",
                "alerts",
                "precision",
                "recall",
                "recall_delta_vs_b0",
                "additional_frauds_caught_vs_b0",
            ]
        ].to_string(index=False, float_format=lambda x: f"{x: .4f}")
    )
    print(f"\nWrote {CALIBRATION_CSV}\nWrote {RELIABILITY_CSV}")
    print(f"Wrote {ALERT_BUDGET_CSV}\nWrote {COST_CSV}\nWrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
