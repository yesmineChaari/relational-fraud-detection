"""Train-only relational candidate screening before G1.

Stage A: Structural eligibility screening -- loads existing relational-audit
         reports (entity_diagnostics.csv, graph_diagnostics.csv) and classifies
         each of the seven audited relations as problematic, viable, or promising.
         No validation or test labels are used.

Stage B: Feature-level signal diagnostics -- for each structurally viable
         relation generates the same four historical summaries used in B1
         (prior_count, prior_count_24h, prior_count_7d,
         time_since_previous_hours) using only the frozen training partition,
         then computes univariate discrimination (PR-AUC, ROC-AUC), fraud-rate
         bins, and Spearman cross-feature correlations.

         The feature computation is imported from
         src.features.build_relational_features, so the diagnostic features are
         the same code path B1 trains on and cannot drift from it.

Output: reports/relational_screening/
    relation_screening.csv               -- structural classification table
    coverage_confound.csv                -- how much signal is grouping-key missingness
    feature_discrimination.csv           -- per-feature train-only statistics
    feature_redundancy.csv               -- Spearman correlations between features
    candidate_selection.json             -- reject / shortlist / preferred decisions
    b1_feature_importance_diagnostic.csv -- B1 relational feature importance rows
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from src.config.paths import ROOT_DIR, SCREENING_CONFIG_PATH
from src.features.build_relational_features import (
    _feature_names,
    compute_relational_features,
)
from src.graph.analyze_relations import CANDIDATES

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


ENTITY_DIAG_PATH = ROOT_DIR / "reports" / "relational_audit" / "entity_diagnostics.csv"
GRAPH_DIAG_PATH = ROOT_DIR / "reports" / "relational_audit" / "graph_diagnostics.csv"

MODEL_DATASET_PATH = ROOT_DIR / "data" / "processed" / "model_dataset.parquet"
SPLIT_ASSIGNMENT_PATH = ROOT_DIR / "data" / "processed" / "split_assignment.parquet"

B1_FEATURE_IMPORTANCE_PATH = (
    ROOT_DIR / "reports" / "b1" / "card_core_addr1" / "feature_importance.csv"
)

REPORT_DIR = ROOT_DIR / "reports" / "relational_screening"

# ---------------------------------------------------------------------------
# Relation definitions
#
# CANDIDATES is imported from src.graph.analyze_relations so the audit and the
# screening can never disagree about what a relation is. Do not redefine it.
# ---------------------------------------------------------------------------

ALL_CANDIDATE_NAMES = list(CANDIDATES.keys())

# ---------------------------------------------------------------------------
SCREENING_CONFIG_SCHEMA: dict[str, tuple[str, ...]] = {
    "structural": (
        "coverage_min_pct",
        "largest_entity_share_reject_pct",
        "largest_component_reject_pct",
        "recurring_entity_min_pct",
        "group_size_median_min",
        "median_repeat_gap_must_be_finite",
    ),
    "promotion": (
        "preferred_coverage_min_pct",
        "preferred_largest_entity_max_pct",
        "max_preferred_relations",
    ),
    "signal": (
        "signal_strong_pr_auc_lift",
        "signal_moderate_pr_auc_lift",
        "require_signal_above_missingness",
    ),
}


def load_screening_config(path: Path = SCREENING_CONFIG_PATH) -> dict[str, dict[str, Any]]:
    """Screening thresholds, validated against a fixed schema.

    Every key is required and unknown keys are rejected, so a typo cannot
    silently loosen a gate -- a misspelled threshold falling back to a default
    would be a screening policy nobody chose. Keys beginning with an underscore
    are commentary and are ignored.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Screening thresholds not found at {path}. This file holds the "
            f"screening policy and is required; it is not optional configuration."
        )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    resolved: dict[str, dict[str, Any]] = {}
    for section, required_keys in SCREENING_CONFIG_SCHEMA.items():
        if section not in payload:
            raise KeyError(f"{path}: missing required section {section!r}.")
        values = {k: v for k, v in payload[section].items() if not k.startswith("_")}
        missing = sorted(set(required_keys) - set(values))
        unknown = sorted(set(values) - set(required_keys))
        if missing:
            raise KeyError(f"{path}: section {section!r} is missing {missing}.")
        if unknown:
            raise KeyError(
                f"{path}: section {section!r} has unknown keys {unknown}. Unknown "
                f"keys are rejected rather than ignored, so a typo cannot silently "
                f"leave a threshold at its old value."
            )
        resolved[section] = values
    return resolved


_SCREENING_CONFIG = load_screening_config()
_STRUCTURAL = _SCREENING_CONFIG["structural"]
_PROMOTION = _SCREENING_CONFIG["promotion"]
_SIGNAL = _SCREENING_CONFIG["signal"]


# Structural screening thresholds -- documented with rationale.
#
# COVERAGE_MIN_PCT (50%):
#   A relation covering fewer than half the training transactions provides
#   historical context for at most 50% of predictions, limiting its impact.
#
# LARGEST_ENTITY_SHARE_REJECT_PCT (10%):
#   If one entity contains >= 10% of all covered transactions the relation has
#   collapsed unrelated events into a giant super-entity whose history is
#   dominated by entity-level properties rather than card-holder behaviour.
#
# LARGEST_COMPONENT_REJECT_PCT (10%):
#   Same reasoning for graph components. A component containing >= 10% of all
#   nodes signals a hub that merges unrelated transaction streams.
#
# RECURRING_ENTITY_MIN_PCT (50%):
#   At least 50% of entities must appear more than once for there to be
#   meaningful entity-level history to summarise.
#
# GROUP_SIZE_MEDIAN_MIN (2):
#   Median group size must be >= 2; a median of 1 means most entities are
#   singletons and cannot contribute historical features.
#
# MEDIAN_REPEAT_GAP_MUST_BE_FINITE:
#   Recurring entities must have a finite median repeat gap.
# ---------------------------------------------------------------------------

COVERAGE_MIN_PCT: float = _STRUCTURAL["coverage_min_pct"]
LARGEST_ENTITY_SHARE_REJECT_PCT: float = _STRUCTURAL["largest_entity_share_reject_pct"]
LARGEST_COMPONENT_REJECT_PCT: float = _STRUCTURAL["largest_component_reject_pct"]
RECURRING_ENTITY_MIN_PCT: float = _STRUCTURAL["recurring_entity_min_pct"]
GROUP_SIZE_MEDIAN_MIN: float = _STRUCTURAL["group_size_median_min"]
MEDIAN_REPEAT_GAP_MUST_BE_FINITE: bool = _STRUCTURAL["median_repeat_gap_must_be_finite"]

# ---------------------------------------------------------------------------
# Promotion thresholds for the `preferred` decision -- documented with rationale.
#
# PREFERRED_COVERAGE_MIN_PCT (80%):
#   Stricter than the 50% structural floor. A relation we intend to build G1 on
#   should supply neighbourhood history for the large majority of transactions;
#   the B1 card_core_addr1 result showed that a 13% zero-fallback rate is enough
#   to dilute the signal below B0.
#
# PREFERRED_LARGEST_ENTITY_MAX_PCT (5%):
#   Stricter than the 10% structural reject threshold, for the same reason: at
#   the preferred tier we want no entity large enough to dominate the summaries.
#
# MAX_PREFERRED_RELATIONS (2):
#   At most two relations may be promoted, because Stage B allows at most two
#   controlled B1 experiments against the validation split. Keeping the number
#   small is what stops validation becoming a selection loop.
# ---------------------------------------------------------------------------

PREFERRED_COVERAGE_MIN_PCT: float = _PROMOTION["preferred_coverage_min_pct"]
PREFERRED_LARGEST_ENTITY_MAX_PCT: float = _PROMOTION["preferred_largest_entity_max_pct"]
MAX_PREFERRED_RELATIONS: int = _PROMOTION["max_preferred_relations"]

# ---------------------------------------------------------------------------
# Feature-signal thresholds -- expressed as PR-AUC lift over the base rate.
#
# A raw PR-AUC is meaningless without its prevalence: at a 3.5% fraud rate a
# random scorer already achieves PR-AUC ~= 0.035, so absolute cutoffs would
# label every single-feature summary "weak" by construction. These thresholds
# are therefore multiples of the train prevalence.
#
# SIGNAL_STRONG_PR_AUC_LIFT (2.0x): the feature alone doubles the base rate.
# SIGNAL_MODERATE_PR_AUC_LIFT (1.25x): a clear but modest standalone signal.
# Anything below 1.25x is reported as weak.
# ---------------------------------------------------------------------------

SIGNAL_STRONG_PR_AUC_LIFT: float = _SIGNAL["signal_strong_pr_auc_lift"]
SIGNAL_MODERATE_PR_AUC_LIFT: float = _SIGNAL["signal_moderate_pr_auc_lift"]

# ---------------------------------------------------------------------------
# Missingness-dominance rule.
#
# A relation whose grouping key is often absent hands every uncovered row a
# zero-count sentinel. Where missingness itself predicts fraud -- and on this
# dataset it does strongly for card_core_addr1 (10.2% fraud on rows with a
# missing component vs 2.5% on covered rows) -- those sentinels make the
# feature look informative while carrying no entity history at all.
#
# A relation may only be preferred if its best history summary, scored on the
# rows that actually have history, beats what you would get from a plain
# "key is missing" flag. Otherwise the relation is not contributing relational
# information; it is contributing a null indicator, which B0 can already learn
# from the raw columns. This is a measured comparison, not a fixed threshold.
# ---------------------------------------------------------------------------

REQUIRE_SIGNAL_ABOVE_MISSINGNESS: bool = _SIGNAL["require_signal_above_missingness"]

# B1 relational feature names used for importance diagnostic.
B1_RELATIONAL_FEATURES = _feature_names("card_core_addr1")

# Protected B1 artifacts -- must NOT be modified by this script.
PROTECTED_PATHS = [
    ROOT_DIR / "models" / "lightgbm_b1_card_core_addr1.txt",
    ROOT_DIR / "reports" / "b1" / "card_core_addr1" / "metrics.json",
    ROOT_DIR / "reports" / "b1" / "card_core_addr1" / "feature_importance.csv",
    ROOT_DIR / "reports" / "b1" / "card_core_addr1" / "metadata.json",
    ROOT_DIR / "data" / "processed" / "model_dataset.parquet",
    ROOT_DIR / "data" / "processed" / "split_assignment.parquet",
]


# ===========================================================================
# Stage A: Structural screening
# ===========================================================================


def load_audit_reports() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load entity and graph diagnostic CSVs produced by the relation audit."""
    if not ENTITY_DIAG_PATH.exists():
        raise FileNotFoundError(
            f"Entity diagnostics not found: {ENTITY_DIAG_PATH}. Run the relational audit first."
        )
    if not GRAPH_DIAG_PATH.exists():
        raise FileNotFoundError(
            f"Graph diagnostics not found: {GRAPH_DIAG_PATH}. Run the relational audit first."
        )

    entity_df = pd.read_csv(ENTITY_DIAG_PATH)
    graph_df = pd.read_csv(GRAPH_DIAG_PATH)

    for df, name in [(entity_df, "entity_diagnostics"), (graph_df, "graph_diagnostics")]:
        missing_relations = set(ALL_CANDIDATE_NAMES) - set(df["relation"])
        if missing_relations:
            raise ValueError(f"{name}.csv is missing relations: {sorted(missing_relations)}.")
    return entity_df, graph_df


def classify_relation(
    entity_row: pd.Series,
    graph_row: pd.Series,
) -> tuple[str, list[str]]:
    """Classify a relation as problematic, viable, or promising."""
    reasons: list[str] = []
    rejected = False

    coverage = float(entity_row["coverage_pct"])
    largest_entity = float(entity_row["largest_entity_share_pct"])
    largest_component = float(graph_row["largest_component_pct"])

    if coverage < COVERAGE_MIN_PCT:
        reasons.append(f"coverage {coverage:.2f}% < threshold {COVERAGE_MIN_PCT}%")
        rejected = True

    if largest_entity >= LARGEST_ENTITY_SHARE_REJECT_PCT:
        reasons.append(
            f"largest_entity_share {largest_entity:.3f}% >= "
            f"threshold {LARGEST_ENTITY_SHARE_REJECT_PCT}%"
        )
        rejected = True

    if largest_component >= LARGEST_COMPONENT_REJECT_PCT:
        reasons.append(
            f"largest_component_pct {largest_component:.3f}% >= "
            f"threshold {LARGEST_COMPONENT_REJECT_PCT}%"
        )
        rejected = True

    if rejected:
        return "problematic", reasons

    recurring_pct = float(entity_row["recurring_entity_pct"])
    group_median = float(entity_row["group_size_median"])
    repeat_gap = entity_row.get("median_repeat_gap_hours", np.nan)
    fraud_lift = float(graph_row.get("fraud_neighbor_lift", np.nan))

    all_preferred = True

    if recurring_pct < RECURRING_ENTITY_MIN_PCT:
        reasons.append(
            f"recurring_entity_pct {recurring_pct:.2f}% < "
            f"preferred threshold {RECURRING_ENTITY_MIN_PCT}%"
        )
        all_preferred = False

    if group_median < GROUP_SIZE_MEDIAN_MIN:
        reasons.append(
            f"group_size_median {group_median} < preferred threshold {GROUP_SIZE_MEDIAN_MIN}"
        )
        all_preferred = False

    if MEDIAN_REPEAT_GAP_MUST_BE_FINITE and (
        pd.isna(repeat_gap) or not np.isfinite(float(repeat_gap))
    ):
        reasons.append("median_repeat_gap_hours is not finite")
        all_preferred = False

    if all_preferred and not np.isnan(fraud_lift) and fraud_lift > 1.0:
        reasons.append(f"all preferred thresholds met; fraud_neighbor_lift={fraud_lift:.3f}")
        return "promising", reasons

    reasons.append("structurally valid but not all preferred thresholds met")
    return "viable", reasons


def build_structural_screening_table(
    entity_df: pd.DataFrame,
    graph_df: pd.DataFrame,
) -> pd.DataFrame:
    """Produce the structural screening summary for all seven candidates."""
    entity_idx = entity_df.set_index("relation")
    graph_idx = graph_df.set_index("relation")

    rows = []
    for relation in ALL_CANDIDATE_NAMES:
        e = entity_idx.loc[relation]
        g = graph_idx.loc[relation]
        classification, reasons = classify_relation(e, g)

        rows.append(
            {
                "relation": relation,
                "structural_classification": classification,
                "classification_reasons": "; ".join(reasons),
                "coverage_pct": float(e["coverage_pct"]),
                "covered_transactions": int(e["covered_transactions"]),
                "entity_count": int(e["entity_count"]),
                "singleton_entity_pct": float(e["singleton_entity_pct"]),
                "recurring_entity_pct": float(e["recurring_entity_pct"]),
                "repeated_transaction_pct_all": float(e["repeated_transaction_pct_all"]),
                "group_size_median": float(e["group_size_median"]),
                "group_size_p95": float(e["group_size_p95"]),
                "group_size_p99": float(e["group_size_p99"]),
                "group_size_max": int(e["group_size_max"]),
                "largest_entity_share_pct": float(e["largest_entity_share_pct"]),
                "median_entity_lifespan_days": float(e["median_entity_lifespan_days"]),
                "median_repeat_gap_hours": float(e["median_repeat_gap_hours"]),
                "participating_nodes_pct": float(g["participating_nodes_pct"]),
                "isolated_pct": float(g["isolated_pct"]),
                "largest_component_pct": float(g["largest_component_pct"]),
                "fraud_neighbor_lift": float(g["fraud_neighbor_lift"]),
                "previous_fraud_rate_given_current_fraud": float(
                    g["previous_fraud_rate_given_current_fraud"]
                ),
                "previous_fraud_rate_given_current_normal": float(
                    g["previous_fraud_rate_given_current_normal"]
                ),
            }
        )

    return pd.DataFrame(rows)


# ===========================================================================
# Stage B: Feature generation
# ===========================================================================


def feature_names_for_relation(relation: str) -> list[str]:
    """Return the four deterministic feature names for a given relation."""
    return _feature_names(relation)


def generate_relational_features(
    source_df: pd.DataFrame,
    relation: str,
    group_columns: list[str],
) -> pd.DataFrame:
    """Generate the four B1-style historical features for a given relation.

    Delegates to the B1 feature builder's kernel, so the temporal semantics --
    strictly-before timestamp blocks, inclusive-lower-bound 24h/7d windows, NaN
    recency on first occurrence, zero counts for missing grouping components --
    are the exact ones B1 trains on.
    """
    return compute_relational_features(source_df, relation, group_columns)


# ===========================================================================
# Stage B: Statistics & discrimination
# ===========================================================================


def _oriented_scores(values: np.ndarray, descending: bool) -> np.ndarray:
    """Return values oriented so that a larger score means 'more fraudulent'."""
    return -values if descending else values


def _score_pair(
    values: np.ndarray,
    labels: np.ndarray,
) -> tuple[float, float, float, str]:
    """Score one feature in both directions.

    Returns (pr_auc_best_direction, roc_auc_ascending, pr_auc_ascending, direction).

    ROC-AUC is always reported in the raw ascending direction so an inverse
    association stays visible as a value below 0.5 rather than being silently
    flipped -- card_core_addr1_prior_count is exactly such a case. PR-AUC is
    reported for whichever direction is stronger, with the direction named, as
    PR-AUC is not symmetric under a sign flip.
    """
    if len(values) < 2 or len(np.unique(labels)) != 2:
        return np.nan, np.nan, np.nan, "undetermined"
    try:
        roc_ascending = float(roc_auc_score(labels, values))
        pr_ascending = float(average_precision_score(labels, values))
        pr_descending = float(
            average_precision_score(labels, _oriented_scores(values, descending=True))
        )
    except ValueError:
        return np.nan, np.nan, np.nan, "undetermined"

    if pr_descending > pr_ascending:
        return pr_descending, roc_ascending, pr_ascending, "lower_is_fraud"
    return pr_ascending, roc_ascending, pr_ascending, "higher_is_fraud"


def compute_feature_stats(
    feature_series: pd.Series,
    label_series: pd.Series,
    feature_name: str,
    observed_mask: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute train-only descriptive and discrimination statistics.

    `observed_mask` marks the rows whose grouping key is actually present. It
    matters because a row with a missing grouping component is given a count of
    0 -- a sentinel, not a measurement -- and on this dataset missingness is
    itself strongly associated with fraud. Scoring the sentinel rows alongside
    the real ones lets a relation look informative when its history carries no
    signal at all, so discrimination is reported at two tiers:

    * `pr_auc` / `roc_auc_ascending` score **every** train row, with a NaN
      recency imputed to a sentinel above the observed maximum ("longest
      possible time since previous"). All four features of a relation are then
      scored on the same rows against the same prevalence, so these are
      comparable across features -- but they include the missingness effect.
    * `pr_auc_covered` / `roc_auc_covered_ascending` score only the rows where
      the entity is known and the feature is genuinely observed, against
      `covered_prevalence`. This is the entity-history signal with the
      missingness confound removed, and it is what candidate selection uses.

    Both tiers report the direction, and ROC-AUC is always given in the raw
    ascending sense so an inverse association stays visible as a value below
    0.5 rather than being silently flipped.
    """
    is_recency = feature_name.endswith("_time_since_previous_hours")

    values = feature_series.to_numpy(dtype=np.float64)
    labels = label_series.to_numpy(dtype=np.int32)

    non_missing_mask = ~np.isnan(values)
    n_total = len(values)
    n_non_missing = int(non_missing_mask.sum())
    missing_rate = 1.0 - n_non_missing / n_total if n_total > 0 else np.nan

    non_zero_mask = non_missing_mask & (values != 0.0)
    nonzero_rate = float(non_zero_mask.sum()) / n_total if n_total > 0 else np.nan

    vals_valid = values[non_missing_mask]

    if len(vals_valid) == 0:
        q_stats: dict[str, Any] = {
            "min": np.nan,
            "median": np.nan,
            "p90": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }
    else:
        q_stats = {
            "min": float(vals_valid.min()),
            "median": float(np.median(vals_valid)),
            "p90": float(np.quantile(vals_valid, 0.90)),
            "p95": float(np.quantile(vals_valid, 0.95)),
            "p99": float(np.quantile(vals_valid, 0.99)),
            "max": float(vals_valid.max()),
        }

    fraud_mask = labels == 1
    normal_mask = labels == 0

    vals_fraud = values[fraud_mask & non_missing_mask]
    vals_normal = values[normal_mask & non_missing_mask]

    mean_fraud = float(vals_fraud.mean()) if len(vals_fraud) else np.nan
    median_fraud = float(np.median(vals_fraud)) if len(vals_fraud) else np.nan
    mean_normal = float(vals_normal.mean()) if len(vals_normal) else np.nan
    median_normal = float(np.median(vals_normal)) if len(vals_normal) else np.nan

    prevalence = float(labels.mean()) if len(labels) else np.nan

    # All-rows scoring: impute a missing recency to just above the observed
    # maximum ("longest possible time since previous"), so every feature of a
    # relation is scored on the identical row set.
    all_row_values = values
    if is_recency and n_non_missing and n_non_missing < n_total:
        sentinel = float(vals_valid.max()) + 1.0
        all_row_values = np.where(non_missing_mask, values, sentinel)
    elif np.isnan(values).any():
        all_row_values = np.nan_to_num(values, nan=0.0)

    pr_auc, roc_ascending, pr_ascending, direction = _score_pair(all_row_values, labels)

    # Covered rows: entity known AND the feature genuinely observed.
    if observed_mask is None:
        covered_mask = non_missing_mask
    else:
        covered_mask = np.asarray(observed_mask, dtype=bool) & non_missing_mask
    cov_values = values[covered_mask]
    cov_labels = labels[covered_mask]
    cov_pr, cov_roc, _, cov_direction = _score_pair(cov_values, cov_labels)
    cov_prevalence = float(cov_labels.mean()) if len(cov_labels) else np.nan

    pr_auc_lift = (
        pr_auc / prevalence if not np.isnan(pr_auc) and prevalence and prevalence > 0 else np.nan
    )
    pr_auc_lift_covered = (
        cov_pr / cov_prevalence
        if not np.isnan(cov_pr) and cov_prevalence and cov_prevalence > 0
        else np.nan
    )
    roc_auc_strength = abs(roc_ascending - 0.5) if not np.isnan(roc_ascending) else np.nan
    roc_auc_strength_covered = abs(cov_roc - 0.5) if not np.isnan(cov_roc) else np.nan

    if is_recency:
        bins = _compute_recency_bins(values, labels)
    else:
        bins = _compute_count_bins(values, labels)

    return {
        "feature": feature_name,
        "nonzero_rate": nonzero_rate,
        "missing_rate": missing_rate,
        "prevalence": prevalence,
        **q_stats,
        "mean_fraud": mean_fraud,
        "median_fraud": median_fraud,
        "mean_normal": mean_normal,
        "median_normal": median_normal,
        "n_scored": n_total,
        "direction": direction,
        "pr_auc": pr_auc,
        "pr_auc_lift": pr_auc_lift,
        "pr_auc_ascending": pr_ascending,
        "roc_auc_ascending": roc_ascending,
        "roc_auc_strength": roc_auc_strength,
        "n_covered": int(covered_mask.sum()),
        "covered_prevalence": cov_prevalence,
        "direction_covered": cov_direction,
        "pr_auc_covered": cov_pr,
        "pr_auc_lift_covered": pr_auc_lift_covered,
        "roc_auc_covered_ascending": cov_roc,
        "roc_auc_strength_covered": roc_auc_strength_covered,
        "fraud_rate_bins": bins,
    }


def _compute_count_bins(
    values: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    bin_edges = [
        ("0", values == 0),
        ("1", values == 1),
        ("2-3", (values >= 2) & (values <= 3)),
        ("4-7", (values >= 4) & (values <= 7)),
        ("8-15", (values >= 8) & (values <= 15)),
        ("16+", values >= 16),
    ]
    bins = []
    for label, mask in bin_edges:
        valid = mask & ~np.isnan(values)
        n = int(valid.sum())
        fraud_rate = float(labels[valid].mean()) if n > 0 else np.nan
        bins.append({"bin": label, "n": n, "fraud_rate": fraud_rate})
    return bins


def _compute_recency_bins(
    values: np.ndarray,
    labels: np.ndarray,
) -> list[dict[str, Any]]:
    non_null = values[~np.isnan(values)]
    if len(non_null) == 0:
        return [{"bin": "missing (NaN)", "n": int(np.isnan(values).sum()), "fraud_rate": np.nan}]

    q25 = float(np.quantile(non_null, 0.25))
    q50 = float(np.quantile(non_null, 0.50))
    q75 = float(np.quantile(non_null, 0.75))

    bin_defs = [
        (f"<= {q25:.1f}h", ~np.isnan(values) & (values <= q25)),
        (f"{q25:.1f}h - {q50:.1f}h", ~np.isnan(values) & (values > q25) & (values <= q50)),
        (f"{q50:.1f}h - {q75:.1f}h", ~np.isnan(values) & (values > q50) & (values <= q75)),
        (f"> {q75:.1f}h", ~np.isnan(values) & (values > q75)),
        ("missing (NaN)", np.isnan(values)),
    ]
    bins = []
    for bin_label, mask in bin_defs:
        n = int(mask.sum())
        fraud_rate = float(labels[mask].mean()) if n > 0 else np.nan
        bins.append({"bin": bin_label, "n": n, "fraud_rate": fraud_rate})
    return bins


def compute_spearman_correlations(
    feature_df: pd.DataFrame,
    feature_names: list[str],
) -> list[dict[str, Any]]:
    pairs = [
        (feature_names[0], feature_names[1]),
        (feature_names[0], feature_names[2]),
        (feature_names[0], feature_names[3]),
        (feature_names[1], feature_names[2]),
        (feature_names[1], feature_names[3]),
        (feature_names[2], feature_names[3]),
    ]
    rows = []
    for col_a, col_b in pairs:
        a = feature_df[col_a].to_numpy(dtype=np.float64)
        b = feature_df[col_b].to_numpy(dtype=np.float64)
        valid = ~np.isnan(a) & ~np.isnan(b)
        n_valid = int(valid.sum())
        if n_valid >= 10:
            corr, pval = spearmanr(a[valid], b[valid])
            corr_val = float(corr)
            pval_val = float(pval)
        else:
            corr_val = np.nan
            pval_val = np.nan
        rows.append(
            {
                "feature_a": col_a,
                "feature_b": col_b,
                "spearman_r": corr_val,
                "p_value": pval_val,
                "n_valid_pairs": n_valid,
            }
        )
    return rows


def _coverage_confound_row(
    relation: str,
    covered: np.ndarray,
    labels: pd.Series,
) -> dict[str, Any]:
    """Score the grouping key's missingness on its own.

    A relation whose key is often absent gets a zero-count sentinel on those
    rows, and if missingness predicts fraud the sentinel makes the feature look
    informative. Reporting the indicator alone lets a reader see how much of a
    feature's all-rows PR-AUC is really this, and nothing to do with history.
    """
    y = labels.to_numpy(dtype=np.int32)
    missing = (~covered).astype(np.int32)
    n = len(y)
    n_covered = int(covered.sum())
    if 0 < int(missing.sum()) < n and len(np.unique(y)) == 2:
        indicator_pr = float(average_precision_score(y, missing))
        indicator_roc = float(roc_auc_score(y, missing))
    else:
        indicator_pr = np.nan
        indicator_roc = np.nan
    return {
        "relation": relation,
        "train_rows": n,
        "covered_rows": n_covered,
        "coverage_pct_train": 100.0 * n_covered / n if n else np.nan,
        "fraud_rate_covered": float(y[covered].mean()) if n_covered else np.nan,
        "fraud_rate_uncovered": (float(y[~covered].mean()) if n_covered < n else np.nan),
        "coverage_indicator_pr_auc": indicator_pr,
        "coverage_indicator_roc_auc": indicator_roc,
    }


# ===========================================================================
# B1 feature importance diagnostic
# ===========================================================================


def extract_b1_importance_diagnostic() -> pd.DataFrame:
    if not B1_FEATURE_IMPORTANCE_PATH.exists():
        raise FileNotFoundError(f"B1 feature importance not found: {B1_FEATURE_IMPORTANCE_PATH}")
    imp_df = pd.read_csv(B1_FEATURE_IMPORTANCE_PATH)
    required_cols = {"feature", "importance_gain", "importance_split"}
    if not required_cols.issubset(set(imp_df.columns)):
        raise ValueError(f"B1 feature importance CSV must contain columns {required_cols}.")

    imp_df = imp_df.sort_values("importance_gain", ascending=False).reset_index(drop=True)
    imp_df["rank_gain"] = imp_df["importance_gain"].rank(ascending=False, method="min").astype(int)
    imp_df["rank_split"] = (
        imp_df["importance_split"].rank(ascending=False, method="min").astype(int)
    )

    relational_rows = imp_df[imp_df["feature"].isin(B1_RELATIONAL_FEATURES)].copy()
    total_features = len(imp_df)
    relational_rows = relational_rows.assign(total_features_in_model=total_features)
    return relational_rows[
        [
            "feature",
            "importance_gain",
            "importance_split",
            "rank_gain",
            "rank_split",
            "total_features_in_model",
        ]
    ].reset_index(drop=True)


# ===========================================================================
# Candidate selection
# ===========================================================================


def _as_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return np.nan
    return result


def best_univariate_by_relation(
    discrimination_rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Pick each relation's strongest single feature by its covered-rows lift.

    Ranking uses `pr_auc_lift_covered` -- the PR-AUC over its own base rate on
    rows where the entity is known and the feature is observed. That is the
    entity-history signal proper. Ranking on the all-rows PR-AUC instead would
    reward relations whose grouping key is often missing, because on this
    dataset a missing card/addr component is itself a strong fraud indicator
    (10.2% vs 2.5% fraud for card_core_addr1), and the zero-count sentinel those
    rows receive smuggles that indicator into the feature.

    Rows produced before this column existed fall back to `pr_auc`.
    """
    best: dict[str, dict[str, Any]] = {}
    for row in discrimination_rows:
        relation = row["relation"]
        lift = _as_float(row.get("pr_auc_lift_covered"))
        pr_covered = _as_float(row.get("pr_auc_covered"))
        pr_all = _as_float(row.get("pr_auc"))
        # Fall back for callers that supply only the all-rows metrics.
        rank_key = lift if not np.isnan(lift) else pr_all
        if np.isnan(rank_key):
            continue
        if rank_key <= best.get(relation, {}).get("rank_key", float("-inf")):
            continue
        best[relation] = {
            "rank_key": rank_key,
            "feature": row["feature"],
            "pr_auc": pr_all,
            "pr_auc_lift": _as_float(row.get("pr_auc_lift")),
            "roc_auc": _as_float(row.get("roc_auc_ascending", row.get("roc_auc", np.nan))),
            "direction": row.get("direction", "undetermined"),
            "pr_auc_covered": pr_covered,
            "pr_auc_lift_covered": lift,
            "roc_auc_covered": _as_float(row.get("roc_auc_covered_ascending")),
            "direction_covered": row.get("direction_covered", "undetermined"),
        }
    return best


def coverage_weighted_signal(coverage_pct: float, pr_auc_lift_covered: float) -> float:
    """Excess signal over the base rate, weighted by how many rows receive it.

    A relation is only useful to the extent that (a) its history discriminates
    on the rows that have history, and (b) enough rows have history. This is
    the product of the two: coverage x (lift - 1). It ranks relations that
    have already passed the structural and missingness gates; it is a ranking
    aid, not a score that can promote a relation on its own.
    """
    if np.isnan(pr_auc_lift_covered) or np.isnan(coverage_pct):
        return np.nan
    return (coverage_pct / 100.0) * (pr_auc_lift_covered - 1.0)


def signal_strength(pr_auc_lift: float) -> str:
    """Label a standalone signal by how far its PR-AUC exceeds the base rate."""
    if np.isnan(pr_auc_lift):
        return "unknown"
    if pr_auc_lift >= SIGNAL_STRONG_PR_AUC_LIFT:
        return "strong"
    if pr_auc_lift >= SIGNAL_MODERATE_PR_AUC_LIFT:
        return "moderate"
    return "weak"


def decide_candidate_selection(
    screening_df: pd.DataFrame,
    discrimination_rows: list[dict[str, Any]],
    missingness_lift_by_relation: dict[str, float] | None = None,
) -> dict[str, Any]:
    best_by_relation = best_univariate_by_relation(discrimination_rows)
    if missingness_lift_by_relation is None:
        missingness_lift_by_relation = {}

    screening_idx = screening_df.set_index("relation")
    decisions: dict[str, dict[str, Any]] = {}

    for relation in ALL_CANDIDATE_NAMES:
        row = screening_idx.loc[relation]
        struct_class = str(row["structural_classification"])

        if struct_class == "problematic":
            decisions[relation] = {
                "decision": "reject",
                "reason": "failed structural screening",
                "structural_classification": struct_class,
            }
            continue

        best = best_by_relation.get(relation, {})
        best_pr = best.get("pr_auc", np.nan)
        best_lift = best.get("pr_auc_lift", np.nan)
        best_pr_covered = best.get("pr_auc_covered", np.nan)
        best_lift_covered = best.get("pr_auc_lift_covered", np.nan)
        ranking_lift = best_lift_covered if not np.isnan(best_lift_covered) else best_lift

        decisions[relation] = {
            "decision": "shortlist",
            "structural_classification": struct_class,
            "best_univariate_feature": best.get("feature", ""),
            "best_univariate_pr_auc": None if np.isnan(best_pr) else best_pr,
            "best_univariate_roc_auc": (
                None if np.isnan(best.get("roc_auc", np.nan)) else best["roc_auc"]
            ),
            "best_univariate_pr_auc_lift": None if np.isnan(best_lift) else best_lift,
            "best_univariate_direction": best.get("direction", "undetermined"),
            "best_univariate_pr_auc_covered": (
                None if np.isnan(best_pr_covered) else best_pr_covered
            ),
            "best_univariate_pr_auc_lift_covered": (
                None if np.isnan(best_lift_covered) else best_lift_covered
            ),
            "best_univariate_roc_auc_covered": (
                None if np.isnan(best.get("roc_auc_covered", np.nan)) else best["roc_auc_covered"]
            ),
            "overall_feature_signal_strength": signal_strength(ranking_lift),
            "coverage_pct": float(row["coverage_pct"]),
            "largest_entity_share_pct": float(row["largest_entity_share_pct"]),
            "reason": "passed structural screening",
        }

    # A relation is promotable only if it clears every measured dimension. The
    # gates are reported per relation so a rejection can be read, not guessed.
    qualified: list[str] = []
    for relation, decision in decisions.items():
        if decision["decision"] != "shortlist":
            continue
        row = screening_idx.loc[relation]
        coverage = float(row["coverage_pct"])
        largest_entity = float(row["largest_entity_share_pct"])
        struct_class = str(row["structural_classification"])
        best = best_by_relation.get(relation, {})
        lift_covered = best.get("pr_auc_lift_covered", np.nan)
        if np.isnan(lift_covered):
            lift_covered = best.get("pr_auc_lift", np.nan)
        missingness_lift = missingness_lift_by_relation.get(relation, np.nan)

        gates = {
            "structural_classification_is_promising": struct_class == "promising",
            "coverage_at_least_preferred_minimum": coverage >= PREFERRED_COVERAGE_MIN_PCT,
            "largest_entity_below_preferred_maximum": (
                largest_entity < PREFERRED_LARGEST_ENTITY_MAX_PCT
            ),
            "history_signal_at_least_moderate": bool(
                not np.isnan(lift_covered) and lift_covered >= SIGNAL_MODERATE_PR_AUC_LIFT
            ),
            "history_signal_exceeds_key_missingness": bool(
                not REQUIRE_SIGNAL_ABOVE_MISSINGNESS
                or np.isnan(missingness_lift)
                or (not np.isnan(lift_covered) and lift_covered > missingness_lift)
            ),
        }
        decision["promotion_gates"] = gates
        decision["key_missingness_pr_auc_lift"] = (
            None if np.isnan(missingness_lift) else missingness_lift
        )
        decision["coverage_weighted_signal"] = (
            None
            if np.isnan(coverage_weighted_signal(coverage, lift_covered))
            else coverage_weighted_signal(coverage, lift_covered)
        )
        if all(gates.values()):
            qualified.append(relation)
        elif not gates["history_signal_exceeds_key_missingness"]:
            decision["reason"] = (
                "passed structural screening, but its best history summary "
                f"(PR-AUC lift {lift_covered:.3f} on covered rows) does not beat a "
                f"plain 'grouping key is missing' flag (lift {missingness_lift:.3f}); "
                "the apparent signal is missingness, not entity history"
            )

    ranked = sorted(
        qualified,
        key=lambda r: (
            -(
                coverage_weighted_signal(
                    float(screening_idx.loc[r]["coverage_pct"]),
                    best_by_relation.get(r, {}).get("pr_auc_lift_covered", np.nan),
                )
            ),
            r,
        ),
    )[:MAX_PREFERRED_RELATIONS]

    for relation in ranked:
        row = screening_idx.loc[relation]
        coverage = float(row["coverage_pct"])
        largest_entity = float(row["largest_entity_share_pct"])
        lift_covered = best_by_relation.get(relation, {}).get("pr_auc_lift_covered", np.nan)
        decisions[relation]["decision"] = "preferred"
        decisions[relation]["reason"] = (
            f"passed every promotion gate and ranked top "
            f"{MAX_PREFERRED_RELATIONS} by coverage-weighted signal "
            f"({coverage_weighted_signal(coverage, lift_covered):.4f}); "
            f"coverage={coverage:.1f}% >= {PREFERRED_COVERAGE_MIN_PCT}%; "
            f"largest_entity_share={largest_entity:.3f}% < "
            f"{PREFERRED_LARGEST_ENTITY_MAX_PCT}%; "
            f"covered-rows PR-AUC lift={lift_covered:.3f}"
        )

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage_a_only": True,
        "validation_used_for_selection": False,
        "final_test_evaluated": False,
        "thresholds": {
            "coverage_min_pct": COVERAGE_MIN_PCT,
            "largest_entity_share_reject_pct": LARGEST_ENTITY_SHARE_REJECT_PCT,
            "largest_component_reject_pct": LARGEST_COMPONENT_REJECT_PCT,
            "recurring_entity_min_pct": RECURRING_ENTITY_MIN_PCT,
            "group_size_median_min": GROUP_SIZE_MEDIAN_MIN,
            "median_repeat_gap_must_be_finite": MEDIAN_REPEAT_GAP_MUST_BE_FINITE,
            "preferred_coverage_min_pct": PREFERRED_COVERAGE_MIN_PCT,
            "preferred_largest_entity_max_pct": PREFERRED_LARGEST_ENTITY_MAX_PCT,
            "max_preferred_relations": MAX_PREFERRED_RELATIONS,
            "signal_strong_pr_auc_lift": SIGNAL_STRONG_PR_AUC_LIFT,
            "signal_moderate_pr_auc_lift": SIGNAL_MODERATE_PR_AUC_LIFT,
            "require_signal_above_missingness": REQUIRE_SIGNAL_ABOVE_MISSINGNESS,
        },
        "decisions": decisions,
    }


# ===========================================================================
# I/O helpers
# ===========================================================================


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


# ===========================================================================
# Main orchestrator
# ===========================================================================


def run_screening(
    *,
    dataset_path: Path = MODEL_DATASET_PATH,
    split_path: Path = SPLIT_ASSIGNMENT_PATH,
    report_dir: Path = REPORT_DIR,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run the full train-only relational candidate screening pipeline."""

    def log(msg: str) -> None:
        if verbose:
            print(msg)

    protected_hashes_before = {str(p): _sha256(p) for p in PROTECTED_PATHS}

    log("Stage A: loading relational audit reports...")
    entity_df, graph_df = load_audit_reports()

    log("Stage A: classifying relations structurally...")
    screening_df = build_structural_screening_table(entity_df, graph_df)

    viable_relations = screening_df.loc[
        screening_df["structural_classification"] != "problematic", "relation"
    ].tolist()

    log(f"  Structurally viable relations: {viable_relations}")
    log(
        "  Rejected: "
        + str(
            screening_df.loc[
                screening_df["structural_classification"] == "problematic", "relation"
            ].tolist()
        )
    )

    all_group_columns: list[str] = []
    for cols in CANDIDATES.values():
        for c in cols:
            if c not in all_group_columns:
                all_group_columns.append(c)

    columns_to_load = list(
        dict.fromkeys(["TransactionID", "TransactionDT", "split", "isFraud"] + all_group_columns)
    )

    log("Stage B: loading model dataset...")
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Model dataset not found: {dataset_path}. Build it before running screening."
        )

    schema_cols = set(pq.read_schema(dataset_path).names)
    load_cols = [c for c in columns_to_load if c in schema_cols]
    full_df = pd.read_parquet(dataset_path, columns=load_cols)

    if "split" not in full_df.columns:
        split_df = pd.read_parquet(split_path, columns=["TransactionID", "split"])
        full_df = full_df.merge(split_df, on="TransactionID", how="left")

    # Features are generated over every row so entity history stays continuous
    # across split boundaries, exactly as in B1; only train rows are then scored,
    # and only train labels are ever read.
    train_mask = (full_df["split"] == "train").to_numpy()
    train_labels = full_df.loc[train_mask, "isFraud"].reset_index(drop=True)
    log(f"  Train partition: {int(train_mask.sum()):,} rows")
    log(f"  Fraud rate (train): {train_labels.mean():.4%}")

    log("Stage B: generating historical features for viable relations...")

    discrimination_rows: list[dict[str, Any]] = []
    redundancy_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []

    for relation in viable_relations:
        group_columns = CANDIDATES[relation]
        log(f"  {relation}: group_columns={group_columns}")

        feature_df = generate_relational_features(full_df, relation, group_columns)
        # generate_relational_features preserves full_df's row order, so the
        # train partition is the same positional mask -- no join required.
        train_feat = feature_df.loc[train_mask].reset_index(drop=True)

        # Rows whose grouping key is present. Everything else received a zero
        # count as a sentinel and must not be scored as observed history.
        covered = (
            full_df.loc[train_mask, group_columns].notna().all(axis=1).to_numpy()
            if all(c in full_df.columns for c in group_columns)
            else np.zeros(int(train_mask.sum()), dtype=bool)
        )
        coverage_rows.append(_coverage_confound_row(relation, covered, train_labels))

        names = feature_names_for_relation(relation)
        for feat in names:
            stats = compute_feature_stats(
                train_feat[feat], train_labels, feat, observed_mask=covered
            )
            stats["relation"] = relation
            discrimination_rows.append(stats)

        corr_rows = compute_spearman_correlations(train_feat, names)
        for cr in corr_rows:
            cr["relation"] = relation
        redundancy_rows.extend(corr_rows)

    log("Extracting B1 feature importance diagnostic...")
    b1_importance_df = extract_b1_importance_diagnostic()

    coverage_by_relation = {row["relation"]: row for row in coverage_rows}
    train_prevalence = float(train_labels.mean())
    missingness_lift_by_relation = {
        row["relation"]: (
            row["coverage_indicator_pr_auc"] / train_prevalence
            if not np.isnan(row["coverage_indicator_pr_auc"]) and train_prevalence > 0
            else np.nan
        )
        for row in coverage_rows
    }

    log("Building candidate selection decisions...")
    selection = decide_candidate_selection(
        screening_df, discrimination_rows, missingness_lift_by_relation
    )

    disc_rows_flat = []
    for row in discrimination_rows:
        flat = {k: v for k, v in row.items() if k != "fraud_rate_bins"}
        flat["fraud_rate_bins_json"] = json.dumps(
            row.get("fraud_rate_bins", []), ensure_ascii=False
        )
        disc_rows_flat.append(flat)
    discrimination_df = pd.DataFrame(disc_rows_flat)

    disc_col_order = [
        "relation",
        "feature",
        "nonzero_rate",
        "missing_rate",
        "prevalence",
        "min",
        "median",
        "p90",
        "p95",
        "p99",
        "max",
        "mean_fraud",
        "median_fraud",
        "mean_normal",
        "median_normal",
        "n_scored",
        "direction",
        "pr_auc",
        "pr_auc_lift",
        "pr_auc_ascending",
        "roc_auc_ascending",
        "roc_auc_strength",
        "n_covered",
        "covered_prevalence",
        "direction_covered",
        "pr_auc_covered",
        "pr_auc_lift_covered",
        "roc_auc_covered_ascending",
        "roc_auc_strength_covered",
        "fraud_rate_bins_json",
    ]
    disc_col_order = [c for c in disc_col_order if c in discrimination_df.columns]
    discrimination_df = discrimination_df[disc_col_order]

    redundancy_df = pd.DataFrame(redundancy_rows)

    best_by_relation = best_univariate_by_relation(discrimination_rows)

    summary_rows = []
    for _, srow in screening_df.iterrows():
        rel = str(srow["relation"])
        best = best_by_relation.get(rel, {})
        best_lift = best.get("rank_key", np.nan)
        summary_rows.append(
            {
                "relation": rel,
                "structural_classification": srow["structural_classification"],
                "coverage_pct": srow["coverage_pct"],
                "largest_entity_share_pct": srow["largest_entity_share_pct"],
                "largest_component_pct": srow["largest_component_pct"],
                "median_group_size": srow["group_size_median"],
                "p95_group_size": srow["group_size_p95"],
                "recurring_entity_pct": srow["recurring_entity_pct"],
                "median_repeat_gap_hours": srow["median_repeat_gap_hours"],
                "fraud_neighbor_lift": srow["fraud_neighbor_lift"],
                "best_univariate_relational_feature": best.get("feature", ""),
                "best_univariate_pr_auc": best.get("pr_auc"),
                "best_univariate_pr_auc_lift": best.get("pr_auc_lift"),
                "best_univariate_roc_auc": best.get("roc_auc"),
                "best_univariate_direction": best.get("direction", ""),
                "best_univariate_pr_auc_covered": best.get("pr_auc_covered"),
                "best_univariate_pr_auc_lift_covered": best.get("pr_auc_lift_covered"),
                "best_univariate_roc_auc_covered": best.get("roc_auc_covered"),
                "coverage_indicator_pr_auc": coverage_by_relation.get(rel, {}).get(
                    "coverage_indicator_pr_auc"
                ),
                "coverage_indicator_roc_auc": coverage_by_relation.get(rel, {}).get(
                    "coverage_indicator_roc_auc"
                ),
                "key_missingness_pr_auc_lift": missingness_lift_by_relation.get(rel),
                "coverage_weighted_signal": coverage_weighted_signal(
                    float(srow["coverage_pct"]),
                    best.get("pr_auc_lift_covered", np.nan),
                ),
                "overall_feature_signal_strength": signal_strength(best_lift),
                "candidate_decision": selection["decisions"].get(rel, {}).get("decision", ""),
            }
        )
    summary_df = pd.DataFrame(summary_rows)

    log(f"Writing reports to {report_dir}/")
    report_dir.mkdir(parents=True, exist_ok=True)

    write_csv(report_dir / "relation_screening.csv", summary_df)
    write_csv(report_dir / "coverage_confound.csv", pd.DataFrame(coverage_rows))
    write_csv(report_dir / "feature_discrimination.csv", discrimination_df)
    write_csv(report_dir / "feature_redundancy.csv", redundancy_df)
    write_json(report_dir / "candidate_selection.json", selection)
    write_csv(report_dir / "b1_feature_importance_diagnostic.csv", b1_importance_df)

    protected_hashes_after = {str(p): _sha256(p) for p in PROTECTED_PATHS}
    if protected_hashes_before != protected_hashes_after:
        changed = [
            p
            for p in protected_hashes_before
            if protected_hashes_before[p] != protected_hashes_after.get(p)
        ]
        raise AssertionError(f"Screening modified protected artifacts: {changed}")

    log("Done.")
    if verbose:
        print("\n=== Candidate Selection Summary ===")
        for relation, d in selection["decisions"].items():
            print(f"  {relation}: {d['decision']}")

    return {
        "screening_df": screening_df,
        "summary_df": summary_df,
        "discrimination_rows": discrimination_rows,
        "redundancy_df": redundancy_df,
        "b1_importance_df": b1_importance_df,
        "selection": selection,
    }


def main() -> None:
    run_screening()


if __name__ == "__main__":
    main()
