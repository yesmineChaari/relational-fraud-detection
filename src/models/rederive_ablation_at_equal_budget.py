"""Re-derive the ablation panel at an equal estimator budget, with intervals.

The exposure screen in `compare_selection_bias` reads learning curves and so
produces point estimates only. This module produces the same corrected deltas
with the paired-bootstrap intervals every other comparison in the project is
settled by, which requires scoring each model rather than reading its curve.

No refits. Each booster on disk was saved truncated at its own `best_iteration`,
so any common budget at or below `min(best_iteration)` over the panel is
reachable by inference alone. The panel-wide budget is exactly that minimum, so
every model in the panel is scored at one identical number of trees and no arm
is read at an argmax.

Fidelity gate. Before any corrected figure is computed, every booster is scored
at its own `best_iteration` and checked against the PR-AUC its metrics file
recorded. If any run fails to reproduce, the saved model does not correspond to
the published number and nothing downstream can be trusted, so the run aborts.

Interpretation is the pre-registered ablation rule, unchanged: a feature carries
the gain if its singleton interval excludes zero on the positive side, and is
non-redundant if its leave-one-out interval excludes zero on the negative side.
Only the estimator changes.

Direction of the residual bias, which the result must be read with. A common
budget truncates whichever arm wanted more rounds, and in this panel that is
almost always the ablation variant. Truncation therefore pushes singleton
deltas down and leave-one-out deltas down, so this protocol is biased towards
finding features non-redundant exactly as the argmax protocol was biased
towards finding them redundant. The two together bound the answer; neither
settles it.

Outputs:
  reports/selection_bias/equal_budget_ablation.csv
  reports/selection_bias/equal_budget_ablation_summary.json
"""

from __future__ import annotations

import gc
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.models.compare_selection_bias import OUTPUT_DIR, ROOT_DIR, read_json
from src.models.significance import _fast_average_precision, paired_bootstrap_pr_auc_delta

FEATURE_KEYS = [
    "prior_count",
    "prior_count_24h",
    "prior_count_7d",
    "time_since_previous_hours",
]
SINGLETON = "singleton"
LEAVE_ONE_OUT = "loo"

CONVERGENCE_MODELS = ROOT_DIR / "models" / "convergence_check"
ABLATION_MODELS = ROOT_DIR / "models" / "ablation"
CONVERGENCE_REPORTS = ROOT_DIR / "reports" / "convergence_check"
ABLATION_REPORTS = ROOT_DIR / "reports" / "ablation" / "card1"

B0_REF = "b0_ref"
B1_REF = "b1_ref"

RESULTS_CSV = OUTPUT_DIR / "equal_budget_ablation.csv"
RESULTS_JSON = OUTPUT_DIR / "equal_budget_ablation_summary.json"

FIDELITY_TOLERANCE = 1e-9

OUTCOME_SINGLE_FEATURE = "SINGLE_FEATURE_CARRIES_GAIN"
OUTCOME_REDUNDANT = "REDUNDANT_SUMMARIES_ANY_ONE_SUFFICES"
OUTCOME_ADDITIVE = "ADDITIVE_CONTRIBUTIONS"
OUTCOME_NOT_LOCALISED = "GAIN_NOT_LOCALISED"


def model_paths() -> dict[str, Path]:
    paths = {
        B0_REF: CONVERGENCE_MODELS / "lightgbm_b0__stop_average_precision_cap15000_seed42.txt",
        B1_REF: CONVERGENCE_MODELS
        / "lightgbm_b1_card1__stop_average_precision_cap15000_seed42.txt",
    }
    for key in FEATURE_KEYS:
        for mode in (SINGLETON, LEAVE_ONE_OUT):
            paths[f"{mode}_{key}"] = (
                ABLATION_MODELS / f"lightgbm_ablation_card1__{mode}_{key}_cap15000_seed42.txt"
            )
    return paths


def metrics_paths() -> dict[str, Path]:
    paths = {
        B0_REF: CONVERGENCE_REPORTS
        / "b0"
        / "stop_average_precision"
        / "cap15000_seed42"
        / "metrics.json",
        B1_REF: CONVERGENCE_REPORTS
        / "b1_card1"
        / "stop_average_precision"
        / "cap15000_seed42"
        / "metrics.json",
    }
    for key in FEATURE_KEYS:
        for mode in (SINGLETON, LEAVE_ONE_OUT):
            paths[f"{mode}_{key}"] = (
                ABLATION_REPORTS / f"{mode}_{key}" / "cap15000_seed42" / "metrics.json"
            )
    return paths


def classify_panel(table: pd.DataFrame) -> tuple[list[str], list[str], str]:
    """The published ablation rule, applied verbatim to whatever estimator produced `table`.

    Kept identical to the published rule on purpose: this module changes the
    protocol, never the interpretation, so the two panels differ in exactly one
    place and a reader can attribute any change of outcome to the estimator
    alone. A feature carries the gain if its singleton interval excludes zero on
    the positive side, and is non-redundant if its leave-one-out interval
    excludes zero on the negative side -- dropping it costs something the others
    cannot replace.
    """
    carriers = sorted(
        table.loc[
            (table.ablation_mode == SINGLETON) & table.excludes_zero & (table.delta_pr_auc > 0),
            "feature",
        ]
    )
    non_redundant = sorted(
        table.loc[
            (table.ablation_mode == LEAVE_ONE_OUT) & table.excludes_zero & (table.delta_pr_auc < 0),
            "feature",
        ]
    )
    if non_redundant:
        outcome = OUTCOME_ADDITIVE
    elif len(carriers) == 1:
        outcome = OUTCOME_SINGLE_FEATURE
    elif carriers:
        outcome = OUTCOME_REDUNDANT
    else:
        outcome = OUTCOME_NOT_LOCALISED
    return carriers, non_redundant, outcome


def load_validation_frame() -> tuple[pd.DataFrame, np.ndarray]:
    """The frozen validation partition with all four relational summaries attached."""
    from src.models.train_lightgbm_ablation import (
        ABLATION_FEATURES,
        MODEL_DATASET_PATH,
        RELATION,
        _rel_output_path,
        load_frozen_b0_metadata,
    )
    from src.models.train_lightgbm_baseline import load_model_dataset, validate_split_counts
    from src.models.train_lightgbm_relational import (
        apply_category_mapping,
        attach_relational_features,
        load_feature_builder_metadata,
        load_frozen_category_mappings,
        validate_relational_merge,
    )

    model_index = pd.read_parquet(MODEL_DATASET_PATH, columns=["TransactionID", "split"])
    validate_split_counts(model_index, "model_dataset.parquet equal-budget index")
    load_feature_builder_metadata(RELATION)
    relational = pd.read_parquet(_rel_output_path(RELATION))
    merged_index = validate_relational_merge(model_index, relational, ABLATION_FEATURES)
    del relational
    gc.collect()

    train_df, validation_df, _ = load_model_dataset()
    del train_df
    gc.collect()
    validation_df = attach_relational_features(
        validation_df, merged_index, "validation", ABLATION_FEATURES
    )
    del merged_index
    gc.collect()

    categorical_columns = list(load_frozen_b0_metadata()["categorical_feature_columns"])
    mappings, _ = load_frozen_category_mappings(categorical_columns)
    for column in categorical_columns:
        validation_df[column] = apply_category_mapping(validation_df[column], mappings[column])

    return validation_df, validation_df["isFraud"].astype("int8").to_numpy()


def main() -> None:
    boosters = {
        name: lightgbm.Booster(model_file=str(path)) for name, path in model_paths().items()
    }
    trees = {name: booster.num_trees() for name, booster in boosters.items()}
    budget = min(trees.values())
    print(f"Panel-wide equal budget: {budget} trees (min over {len(trees)} runs)")

    validation_df, y = load_validation_frame()

    def score(name: str, num_iteration: int) -> np.ndarray:
        booster = boosters[name]
        return booster.predict(validation_df[booster.feature_name()], num_iteration=num_iteration)

    reported = metrics_paths()
    print("\nFidelity gate: each booster at its own best_iteration")
    for name in boosters:
        recomputed = _fast_average_precision(y, score(name, trees[name]))
        published = float(read_json(reported[name])["pr_auc"])
        if abs(recomputed - published) > FIDELITY_TOLERANCE:
            raise AssertionError(
                f"{name} does not reproduce its published PR-AUC: "
                f"recomputed {recomputed!r} vs published {published!r}. The saved "
                f"booster does not correspond to the reported number."
            )
        print(f"  {name:36s} {trees[name]:5d} trees  {recomputed:.10f}  OK")

    rows = []
    for mode in (SINGLETON, LEAVE_ONE_OUT):
        reference = B0_REF if mode == SINGLETON else B1_REF
        reference_scores = score(reference, budget)
        for key in FEATURE_KEYS:
            variant_scores = score(f"{mode}_{key}", budget)
            bootstrap = paired_bootstrap_pr_auc_delta(y, variant_scores, reference_scores)
            rows.append(
                {
                    "ablation_mode": mode,
                    "feature": key,
                    "reference": reference,
                    "equal_budget_trees": budget,
                    "delta_pr_auc": bootstrap["observed_delta"],
                    "ci_lower_95": bootstrap["ci_lower_95"],
                    "ci_upper_95": bootstrap["ci_upper_95"],
                    "excludes_zero": bootstrap["excludes_zero"],
                    "delta_roc_auc": (
                        roc_auc_score(y, variant_scores) - roc_auc_score(y, reference_scores)
                    ),
                }
            )
            print(f"  bootstrapped {mode}_{key}", flush=True)

    table = pd.DataFrame(rows)
    carriers, non_redundant, outcome = classify_panel(table)

    summary: dict[str, Any] = {
        "report_name": "Ablation panel re-derived at an equal estimator budget",
        "question": (
            "Does the published localisation survive when every model in the "
            "panel is scored at one identical number of trees?"
        ),
        "equal_budget_trees": int(budget),
        "best_iteration_by_run": {name: int(count) for name, count in trees.items()},
        "fidelity_gate_passed": True,
        "features_carrying_the_gain": carriers,
        "non_redundant_features": non_redundant,
        "outcome_code": outcome,
        "residual_bias_direction": (
            "A common budget truncates whichever arm wanted more rounds, almost "
            "always the ablation variant here. This protocol is therefore biased "
            "towards non-redundancy exactly as the argmax protocol was biased "
            "towards redundancy. The two bound the answer; neither settles it."
        ),
        "results_table_path": str(RESULTS_CSV.relative_to(ROOT_DIR).as_posix()),
        "test_evaluated": False,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_CSV, index=False)
    with RESULTS_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    pd.set_option("display.width", 220)
    print(f"\n=== Equal budget: {budget} trees ===")
    print(table.to_string(index=False, float_format=lambda x: f"{x: .5f}"))
    print(f"\ncarriers: {carriers}")
    print(f"non-redundant: {non_redundant}")
    print(f"outcome: {outcome}")
    print(f"\nWrote {RESULTS_CSV}\nWrote {RESULTS_JSON}")


if __name__ == "__main__":
    main()
