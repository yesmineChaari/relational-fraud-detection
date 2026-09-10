"""An index of every model artifact, and what each one is actually for.

Each model already has a thorough metadata manifest beside it -- feature lists,
mapping hashes, split counts, frozen parameters. What was missing is the layer
above: a reader opening `models/` sees sixty-three files and four near-identical
baseline names, with nothing saying which is the frozen baseline, which are the
alternatives it was chosen over, and which published claims rest on which file.

Two design choices keep this from becoming a stale hand-written list.

**Facts come from the manifests, opinions come from here.** Metrics, predictor
counts, caps and stopping behaviour are read from each run's own metrics and
metadata JSON at generation time, never transcribed. Only the editorial layer --
role, purpose, limitations, what the artifact must not be used for -- lives in
this file, because no manifest can carry it.

**Annotations are keyed by family, and completeness is enforced.** Forty-four
of the artifacts are panel runs: eight ablation cells, five-seed variance
panels, convergence checks, permuted nulls, fixed-budget refits. Per-file prose for those would be
noise, so families are matched by pattern. Any model file matching no family
fails the completeness check rather than being silently omitted, which is what
makes "every file has an entry" a property rather than a claim.

The ledger deliberately records what these artifacts are *not* for. They are
research models trained on a 2019 competition dataset, selected on validation,
with output scores that are not probabilities. None is a deployable fraud model
and none has been evaluated on the held-out test split.

Outputs:
  reports/ledger/experiment_ledger.csv
  reports/ledger/experiment_ledger.json
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from src.config.paths import ROOT_DIR

MODELS_DIR = ROOT_DIR / "models"
REPORTS_DIR = ROOT_DIR / "reports"
OUTPUT_DIR = REPORTS_DIR / "ledger"

LEDGER_CSV = OUTPUT_DIR / "experiment_ledger.csv"
LEDGER_JSON = OUTPUT_DIR / "experiment_ledger.json"

MODEL_SUFFIXES = {".txt", ".pt"}

# Applies to every artifact here, so it is stated once rather than repeated
# sixty-three times.
UNIVERSAL_LIMITATIONS = [
    "Trained on the 2019 IEEE-CIS competition dataset; nothing here has been "
    "validated against current fraud patterns.",
    "Selected and compared on the validation split. The held-out test split is "
    "unused, so every number in this project is a validation number.",
    "Output scores are not probabilities. Class weighting distorts the scale, "
    "and the measured distortion is under-prediction on the low-score bulk "
    "rather than the inflation the weighting would suggest.",
]

NOT_SUITABLE_FOR = (
    "Not a deployable fraud model. No production monitoring, no drift handling, "
    "no calibrated probability output, no evaluation on unseen data. Suitable "
    "only for reproducing and extending this project's measurements."
)

ESTIMATOR_CAP_LIMITATION = (
    "Trained under the 6,000-estimator cap that the convergence check found "
    "binding. This run stopped at the cap rather than converging, so its figure "
    "understates the model; quote the converged (cap 15,000) measurement instead."
)


def metrics_at(*parts: str) -> Callable[[re.Match[str]], Path]:
    """A resolver returning a fixed metrics path, ignoring the match."""

    def resolve(_: re.Match[str]) -> Path:
        return REPORTS_DIR.joinpath(*parts)

    return resolve


FAMILIES: list[dict[str, Any]] = [
    {
        "family": "b0_frozen_baseline",
        "pattern": r"^lightgbm_baseline\.txt$",
        "stage": "B0",
        "role": "frozen_baseline",
        "purpose": (
            "The tabular baseline every other number in this project is measured "
            "against. Selected over three alternatives by controlled comparison."
        ),
        "claims": [
            "The +0.00630 B1-card1 relational gain",
            "The -0.02093 G1-card1 graph deficit",
            "Every ablation and null-control delta",
        ],
        "metrics": metrics_at("baseline", "lightgbm_metrics.json"),
        "metadata": metrics_at("baseline", "baseline_metadata.json"),
    },
    {
        "family": "b0_discarded_alternative",
        "pattern": r"^lightgbm_baseline_(original_3000_weighted|unweighted_6000|weighted_6000)\.txt$",
        "stage": "B0",
        "role": "discarded_alternative",
        "purpose": (
            "A candidate from the baseline selection, kept as evidence of the "
            "comparison rather than discarded. Class weighting at 6,000 rounds "
            "was chosen: weighting materially improves PR-AUC at this 3.4% "
            "prevalence, and 6,000 rounds beat the original 3,000-round "
            "configuration. The weighted_6000 candidate is the one promoted to "
            "the frozen baseline, so it and lightgbm_baseline.txt describe the "
            "same configuration."
        ),
        "claims": ["The baseline selection decision only; no downstream claim rests on these"],
        "metrics": lambda m: REPORTS_DIR / "baseline" / "experiments" / m.group(1) / "metrics.json",
        "metadata": lambda m: REPORTS_DIR
        / "baseline"
        / "experiments"
        / m.group(1)
        / "metadata.json",
    },
    {
        "family": "b1_relational_variant",
        "pattern": r"^lightgbm_b1_(card1|card1_card2|card_core_addr1)\.txt$",
        "stage": "B1",
        "role": "relational_variant",
        "purpose": (
            "The baseline plus four hand-written history summaries over one "
            "entity definition. card1 is the variant carrying the headline "
            "relational gain; card_core_addr1 is the original relation, whose "
            "apparent signal proved to be a missingness sentinel."
        ),
        "claims": [
            "The +0.00630 relational gain (card1 only)",
            "The graph-stage recovery target",
        ],
        "metrics": lambda m: REPORTS_DIR / "b1" / m.group(1) / "metrics.json",
        "metadata": lambda m: REPORTS_DIR / "b1" / m.group(1) / "metadata.json",
    },
    {
        "family": "g1_graph_model",
        "pattern": r"^lightgbm_g1_card1\.txt$",
        "stage": "G1",
        "role": "graph_variant",
        "purpose": (
            "The baseline plus 32 GraphSAGE embedding columns over the card1 "
            "transaction graph. Underperforms both the baseline and B1-card1."
        ),
        "claims": ["The G1 deficit, and the conclusion that the graph stage did not pay off"],
        "metrics": metrics_at("g1", "card1", "metrics.json"),
        "metadata": metrics_at("g1", "card1", "metadata.json"),
    },
    {
        "family": "g1_attribution_control",
        "pattern": r"^lightgbm_g1_control_(cross_fitted|extended_budget|neighbourhood_only|shuffled_embedding)\.txt$",
        "stage": "G1",
        "role": "attribution_control",
        "purpose": (
            "Separates encoder confounds from the graph verdict: whether the "
            "deficit came from encoder overfitting, too small a budget, the "
            "neighbourhood definition, or simply from adding 32 columns. The "
            "shuffled-embedding control established that a randomly-aligned "
            "32-column block explains most of the gap by dilution alone."
        ),
        "claims": ["The attribution of the G1 deficit to dilution rather than to graph structure"],
        "metrics": lambda m: REPORTS_DIR / "g1_controls" / m.group(1) / "metrics.json",
        "metadata": lambda m: REPORTS_DIR / "g1_controls" / m.group(1) / "metadata.json",
    },
    {
        "family": "graphsage_encoder",
        "pattern": r"^graphsage_card1_encoder.*\.pt$",
        "stage": "G1",
        "role": "encoder",
        "purpose": (
            "The trained GraphSAGE encoder producing the per-transaction "
            "embeddings the G1 model consumes. Not a classifier: it emits "
            "vectors, and carries no metrics of its own beyond training loss."
        ),
        "claims": ["Inputs to the G1 model and its controls"],
        "metrics": None,
        "metadata": metrics_at("graphsage", "card1", "metadata.json"),
    },
    {
        "family": "ablation_cell",
        "pattern": r"^ablation/lightgbm_ablation_card1__(singleton|loo)_(\w+?)_cap15000_seed(\d+)\.txt$",
        "stage": "B1",
        "role": "ablation_panel",
        "purpose": (
            "One cell of the per-feature ablation localising which of the four "
            "card1 summaries carries the gain. Singleton cells add one feature "
            "to the baseline; leave-one-out cells remove one from the full set."
        ),
        "claims": [
            "The published redundancy verdict, since superseded by the "
            "fixed-budget refits; kept as the argmax endpoint of the bracket"
        ],
        "metrics": lambda m: REPORTS_DIR
        / "ablation"
        / "card1"
        / f"{m.group(1)}_{m.group(2)}"
        / f"cap15000_seed{m.group(3)}"
        / "metrics.json",
        "metadata": lambda m: REPORTS_DIR
        / "ablation"
        / "card1"
        / f"{m.group(1)}_{m.group(2)}"
        / f"cap15000_seed{m.group(3)}"
        / "metadata.json",
    },
    {
        "family": "convergence_check",
        "pattern": r"^convergence_check/lightgbm_(\w+?)__stop_(\w+?)_cap15000_seed(\d+)\.txt$",
        "stage": "protocol",
        "role": "convergence_check",
        "purpose": (
            "Re-trains a model at a 15,000-estimator cap to establish where it "
            "actually converges, and under an alternative stopping metric to "
            "test whether the stopping rule was the problem. It was not: "
            "patience on ROC-AUC stopped both models around 1,200 rounds with "
            "PR-AUC far below converged."
        ),
        "claims": [
            "The converged protocol every post-cap comparison uses",
            "The revised G1 deficit of -0.0209",
        ],
        "metrics": lambda m: REPORTS_DIR
        / "convergence_check"
        / m.group(1)
        / f"stop_{m.group(2)}"
        / f"cap15000_seed{m.group(3)}"
        / "metrics.json",
        "metadata": lambda m: REPORTS_DIR
        / "convergence_check"
        / m.group(1)
        / f"stop_{m.group(2)}"
        / f"cap15000_seed{m.group(3)}"
        / "metadata.json",
    },
    {
        "family": "seed_variance_panel",
        "pattern": r"^seed_variance/lightgbm_(\w+?)_seed(\d+)\.txt$",
        "stage": "protocol",
        "role": "seed_panel",
        "purpose": (
            "One run of the five-seed panel measuring how far refitting alone "
            "moves a paired delta. Supplies the clean-stratum noise floor of "
            "0.00051 that every other comparison is judged against."
        ),
        "claims": ["The noise floor used across the project"],
        "metrics": lambda m: REPORTS_DIR
        / "seed_variance"
        / m.group(1)
        / f"seed_{m.group(2)}"
        / "metrics.json",
        "metadata": lambda m: REPORTS_DIR
        / "seed_variance"
        / m.group(1)
        / f"seed_{m.group(2)}"
        / "metadata.json",
    },
    {
        "family": "permuted_entity_null",
        "pattern": r"^permuted_null/lightgbm_permuted_card1__cap15000_permseed(\d+)\.txt$",
        "stage": "B1",
        "role": "null_control",
        "purpose": (
            "The relational features rebuilt on a permuted card1 assignment, "
            "breaking the transaction-to-entity correspondence while preserving "
            "every distribution. Separates genuine entity history from a "
            "split-search artifact of adding four numeric columns."
        ),
        "claims": [
            "That the B1 gain is attributable to entity history rather than to "
            "split-search noise, which is what makes the result causal"
        ],
        "metrics": lambda m: REPORTS_DIR
        / "permuted_null"
        / "card1"
        / f"permseed_{m.group(1)}"
        / "cap15000"
        / "metrics.json",
        "metadata": lambda m: REPORTS_DIR
        / "permuted_null"
        / "card1"
        / f"permseed_{m.group(1)}"
        / "cap15000"
        / "metadata.json",
    },
    {
        "family": "fixed_budget_ablation",
        "pattern": r"^fixed_budget/lightgbm_fixed_card1__(\w+?)_fixed(\d+)_seed(\d+)\.txt$",
        "stage": "B1",
        "role": "fixed_budget_panel",
        "purpose": (
            "The ablation panel and its two references refit at 10,000 rounds "
            "with early stopping disabled and read at that round, so no arm's "
            "figure depends on how many rounds it was granted. The budget was "
            "registered before any run, above every optimum in the panel."
        ),
        "claims": [
            "The within-block localisation: ADDITIVE_CONTRIBUTIONS, superseding "
            "the published redundancy verdict"
        ],
        "metrics": lambda m: REPORTS_DIR
        / "fixed_budget"
        / "card1"
        / m.group(1)
        / f"fixed{m.group(2)}_seed{m.group(3)}"
        / "metrics.json",
        "metadata": lambda m: REPORTS_DIR
        / "fixed_budget"
        / "card1"
        / m.group(1)
        / f"fixed{m.group(2)}_seed{m.group(3)}"
        / "metadata.json",
    },
]

PREDICTOR_COUNT_KEYS = [
    "number_of_predictors",
    "b1_feature_count",
    "number_of_features",
    "feature_count",
]


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def discover_models() -> list[str]:
    """Every model artifact under models/, as a posix path relative to it."""
    if not MODELS_DIR.exists():
        raise FileNotFoundError(f"No models directory at {MODELS_DIR}.")
    return sorted(
        path.relative_to(MODELS_DIR).as_posix()
        for path in MODELS_DIR.rglob("*")
        if path.is_file() and path.suffix in MODEL_SUFFIXES
    )


def tracked_models() -> set[str]:
    """Model artifacts committed to git, as opposed to produced locally."""
    result = subprocess.run(
        ["git", "ls-files", "models/"],
        cwd=ROOT_DIR,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return set()
    return {
        line[len("models/") :]
        for line in result.stdout.splitlines()
        if line.startswith("models/") and Path(line).suffix in MODEL_SUFFIXES
    }


def match_family(relative_path: str) -> tuple[dict[str, Any], re.Match[str]] | None:
    for family in FAMILIES:
        matched = re.match(family["pattern"], relative_path)
        if matched:
            return family, matched
    return None


def resolve_path(spec: Any, matched: re.Match[str]) -> Path | None:
    if spec is None:
        return None
    return spec(matched)


def build_entry(relative_path: str, tracked: set[str]) -> dict[str, Any]:
    found = match_family(relative_path)
    if found is None:
        raise KeyError(
            f"Model artifact {relative_path!r} matches no ledger family. Add one "
            f"to FAMILIES rather than leaving the artifact undocumented."
        )
    family, matched = found

    metrics = read_json(resolve_path(family["metrics"], matched)) if family["metrics"] else None
    metadata = read_json(resolve_path(family["metadata"], matched)) if family["metadata"] else None

    limitations = list(UNIVERSAL_LIMITATIONS)
    cap_reached = (metrics or {}).get("estimator_cap_reached")
    if cap_reached:
        limitations.insert(0, ESTIMATOR_CAP_LIMITATION)

    predictors = None
    for key in PREDICTOR_COUNT_KEYS:
        if metadata and key in metadata:
            predictors = metadata[key]
            break

    return {
        "artifact": relative_path,
        "family": family["family"],
        "stage": family["stage"],
        "role": family["role"],
        "tracked_in_git": relative_path in tracked,
        "purpose": family["purpose"],
        "claims_resting_on_it": family["claims"],
        "training_split": "train",
        "evaluated_on_split": (metrics or {}).get("evaluation_split"),
        "number_of_predictors": predictors,
        "validation_pr_auc": (metrics or {}).get("pr_auc"),
        "validation_roc_auc": (metrics or {}).get("roc_auc"),
        "best_iteration": (metrics or {}).get("best_iteration"),
        "maximum_estimators": (metrics or {}).get("maximum_estimators"),
        "estimator_cap_reached": cap_reached,
        "early_stopping_triggered": (metrics or {}).get("early_stopping_triggered"),
        "test_evaluated": bool((metrics or {}).get("test_evaluated", False)),
        "metrics_manifest_found": metrics is not None,
        "metadata_manifest_found": metadata is not None,
        "limitations": limitations,
        "not_suitable_for": NOT_SUITABLE_FOR,
    }


def build_ledger() -> list[dict[str, Any]]:
    tracked = tracked_models()
    return [build_entry(path, tracked) for path in discover_models()]


def summarize(entries: list[dict[str, Any]]) -> dict[str, Any]:
    frame = pd.DataFrame(entries)
    return {
        "report_name": "Experiment ledger",
        "question": "What is each model artifact, and which claims rest on it?",
        "n_artifacts": len(entries),
        "n_tracked_in_git": int(frame["tracked_in_git"].sum()),
        "n_untracked": int((~frame["tracked_in_git"]).sum()),
        "artifacts_by_family": frame["family"].value_counts().to_dict(),
        "artifacts_by_role": frame["role"].value_counts().to_dict(),
        "artifacts_missing_a_metrics_manifest": sorted(
            frame.loc[~frame["metrics_manifest_found"], "artifact"]
        ),
        "any_artifact_evaluated_on_test": bool(frame["test_evaluated"].any()),
        "universal_limitations": UNIVERSAL_LIMITATIONS,
        "not_suitable_for": NOT_SUITABLE_FOR,
        "completeness": (
            "Generated by walking models/ and matching each artifact to a family. "
            "An artifact matching no family raises rather than being omitted, so "
            "the ledger cannot fall silently out of date as runs are added."
        ),
        "facts_are_read_not_transcribed": (
            "Metrics, predictor counts, caps and stopping behaviour come from "
            "each run's own manifests at generation time. Only role, purpose, "
            "limitations and intended use are authored here."
        ),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    entries = build_ledger()
    summary = summarize(entries)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    flat = pd.DataFrame(entries)
    flat["claims_resting_on_it"] = flat["claims_resting_on_it"].apply(" | ".join)
    flat["limitations"] = flat["limitations"].apply(" | ".join)
    flat.to_csv(LEDGER_CSV, index=False)
    with LEDGER_JSON.open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "artifacts": entries}, handle, indent=2)
        handle.write("\n")

    pd.set_option("display.width", 200)
    print(
        f"{summary['n_artifacts']} artifacts "
        f"({summary['n_tracked_in_git']} tracked, {summary['n_untracked']} local)\n"
    )
    counts = pd.Series(summary["artifacts_by_family"]).sort_values(ascending=False)
    print(counts.to_string())
    missing = summary["artifacts_missing_a_metrics_manifest"]
    print(f"\nwithout a metrics manifest: {len(missing)}")
    for artifact in missing:
        print(f"  {artifact}")
    print(f"\nWrote {LEDGER_CSV}\nWrote {LEDGER_JSON}")


if __name__ == "__main__":
    main()
