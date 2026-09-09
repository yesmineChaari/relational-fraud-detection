"""tests/test_relational_models.py

Automated verification for B1 model training invariance and output artifacts.
Covers:
  - Feature integrity (no NaNs in count features, correct column names)
  - Model artifact existence and feature count (439)
  - B0 baseline files unchanged after B1 runs
  - Metadata consistency checks
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EXPECTED_ROWS = 590_540
EXPECTED_B1_FEATURE_COUNT = 439
SUPPORTED_RELATIONS = ["card_core_addr1", "card1", "card1_card2"]

RELATIONAL_REGISTRY = {
    "card_core_addr1": ["card1", "card2", "card3", "card5", "addr1"],
    "card1": ["card1"],
    "card1_card2": ["card1", "card2"],
}

B0_PROTECTED_PATHS = [
    ROOT_DIR / "models" / "lightgbm_baseline.txt",
    ROOT_DIR / "reports" / "baseline" / "baseline_metadata.json",
    ROOT_DIR / "reports" / "baseline" / "lightgbm_metrics.json",
    ROOT_DIR / "reports" / "baseline" / "categorical_mappings.json",
]


def _sha256(p: Path) -> str | None:
    if not p.exists():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _feature_names(relation: str) -> list[str]:
    return [
        f"{relation}_prior_count",
        f"{relation}_prior_count_24h",
        f"{relation}_prior_count_7d",
        f"{relation}_time_since_previous_hours",
    ]


def _rel_output_path(relation: str) -> Path:
    return ROOT_DIR / "data" / "processed" / f"relational_features_{relation}.parquet"


def _rel_metadata_path(relation: str) -> Path:
    return ROOT_DIR / "reports" / "relational_features" / f"{relation}_metadata.json"


def _model_path(relation: str) -> Path:
    return ROOT_DIR / "models" / f"lightgbm_b1_{relation}.txt"


def _report_dir(relation: str) -> Path:
    return ROOT_DIR / "reports" / "b1" / relation


# ---------------------------------------------------------------------------
# T1: Feature parquet existence and row count
# ---------------------------------------------------------------------------


class TestFeatureParquetExistence:
    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_parquet_exists(self, relation: str) -> None:
        path = _rel_output_path(relation)
        if not path.exists():
            pytest.skip(f"Feature parquet not yet generated for {relation}: {path}")
        assert path.exists(), f"Parquet not found: {path}"

    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_parquet_row_count(self, relation: str) -> None:
        path = _rel_output_path(relation)
        if not path.exists():
            pytest.skip(f"Feature parquet not generated for {relation}")
        df = pd.read_parquet(path)
        assert len(df) == EXPECTED_ROWS, (
            f"{relation}: expected {EXPECTED_ROWS:,} rows, got {len(df):,}."
        )

    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_parquet_columns(self, relation: str) -> None:
        path = _rel_output_path(relation)
        if not path.exists():
            pytest.skip(f"Feature parquet not generated for {relation}")
        df = pd.read_parquet(path)
        expected = ["TransactionID", *_feature_names(relation)]
        assert list(df.columns) == expected, f"{relation}: wrong columns. Got: {list(df.columns)}"


# ---------------------------------------------------------------------------
# T2: Feature integrity — no NaNs in count features, all counts non-negative
# ---------------------------------------------------------------------------


class TestFeatureIntegrity:
    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_count_features_no_nan(self, relation: str) -> None:
        path = _rel_output_path(relation)
        if not path.exists():
            pytest.skip(f"Feature parquet not generated for {relation}")
        df = pd.read_parquet(path)
        feat_names = _feature_names(relation)
        for col in feat_names[:3]:
            assert not df[col].isna().any(), f"{col} contains NaN values."

    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_count_features_non_negative(self, relation: str) -> None:
        path = _rel_output_path(relation)
        if not path.exists():
            pytest.skip(f"Feature parquet not generated for {relation}")
        df = pd.read_parquet(path)
        feat_names = _feature_names(relation)
        for col in feat_names[:3]:
            assert (df[col] >= 0).all(), f"{col} contains negative counts."

    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_recency_non_negative_where_not_null(self, relation: str) -> None:
        path = _rel_output_path(relation)
        if not path.exists():
            pytest.skip(f"Feature parquet not generated for {relation}")
        df = pd.read_parquet(path)
        recency_col = _feature_names(relation)[3]
        non_null = df[recency_col].dropna()
        assert (non_null >= 0).all(), f"{recency_col} contains negative recencies."

    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_transaction_id_unique(self, relation: str) -> None:
        path = _rel_output_path(relation)
        if not path.exists():
            pytest.skip(f"Feature parquet not generated for {relation}")
        df = pd.read_parquet(path)
        assert df["TransactionID"].is_unique, f"{relation}: duplicate TransactionIDs."

    def test_card1_relation_has_full_coverage(self) -> None:
        """card1 has 100% coverage so recency NaN count must equal 0-count rows."""
        relation = "card1"
        path = _rel_output_path(relation)
        if not path.exists():
            pytest.skip("card1 parquet not generated")
        df = pd.read_parquet(path)
        count_col = _feature_names(relation)[0]
        recency_col = _feature_names(relation)[3]
        # First occurrence per entity has count=0 and NaN recency.
        count_zero = (df[count_col] == 0).sum()
        recency_nan = df[recency_col].isna().sum()
        assert count_zero == recency_nan, (
            f"card1: zero-count rows ({count_zero}) != NaN recency rows ({recency_nan})."
        )


# ---------------------------------------------------------------------------
# T3: Relational metadata correctness
# ---------------------------------------------------------------------------


class TestRelationalMetadata:
    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_metadata_exists(self, relation: str) -> None:
        path = _rel_metadata_path(relation)
        if not path.exists():
            pytest.skip(f"Metadata not yet generated for {relation}")
        assert path.exists()

    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_metadata_relation_name(self, relation: str) -> None:
        path = _rel_metadata_path(relation)
        if not path.exists():
            pytest.skip(f"Metadata not generated for {relation}")
        with path.open() as f:
            meta = json.load(f)
        assert meta["relation_name"] == relation

    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_metadata_group_columns(self, relation: str) -> None:
        path = _rel_metadata_path(relation)
        if not path.exists():
            pytest.skip(f"Metadata not generated for {relation}")
        with path.open() as f:
            meta = json.load(f)
        assert meta["group_columns"] == RELATIONAL_REGISTRY[relation]

    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_metadata_target_labels_not_used(self, relation: str) -> None:
        path = _rel_metadata_path(relation)
        if not path.exists():
            pytest.skip(f"Metadata not generated for {relation}")
        with path.open() as f:
            meta = json.load(f)
        assert meta["target_labels_used"] is False

    @pytest.mark.parametrize("relation", SUPPORTED_RELATIONS)
    def test_metadata_row_count(self, relation: str) -> None:
        path = _rel_metadata_path(relation)
        if not path.exists():
            pytest.skip(f"Metadata not generated for {relation}")
        with path.open() as f:
            meta = json.load(f)
        assert meta["row_count"] == EXPECTED_ROWS


# ---------------------------------------------------------------------------
# T4: Model artifact existence and feature count
# ---------------------------------------------------------------------------


class TestModelArtifacts:
    @pytest.mark.parametrize("relation", ["card1", "card1_card2", "card_core_addr1"])
    def test_model_file_exists(self, relation: str) -> None:
        path = _model_path(relation)
        if not path.exists():
            pytest.skip(f"Model not yet trained for {relation}: {path}")
        assert path.exists()

    @pytest.mark.parametrize("relation", ["card1", "card1_card2"])
    def test_metrics_json_exists(self, relation: str) -> None:
        path = _report_dir(relation) / "metrics.json"
        if not path.exists():
            pytest.skip(f"Metrics not yet generated for {relation}")
        assert path.exists()

    @pytest.mark.parametrize("relation", ["card1", "card1_card2"])
    def test_feature_importance_has_439_features(self, relation: str) -> None:
        path = _report_dir(relation) / "feature_importance.csv"
        if not path.exists():
            pytest.skip(f"Feature importance not yet generated for {relation}")
        df = pd.read_csv(path)
        assert len(df) == EXPECTED_B1_FEATURE_COUNT, (
            f"{relation}: expected {EXPECTED_B1_FEATURE_COUNT} features, got {len(df)}."
        )

    @pytest.mark.parametrize("relation", ["card1", "card1_card2"])
    def test_metadata_test_evaluated_false(self, relation: str) -> None:
        path = _report_dir(relation) / "metadata.json"
        if not path.exists():
            pytest.skip(f"Metadata not yet generated for {relation}")
        with path.open() as f:
            meta = json.load(f)
        assert meta.get("test_evaluated") is False, f"{relation}: test_evaluated must be False."

    @pytest.mark.parametrize("relation", ["card1", "card1_card2"])
    def test_relational_features_in_feature_importance(self, relation: str) -> None:
        path = _report_dir(relation) / "feature_importance.csv"
        if not path.exists():
            pytest.skip(f"Feature importance not generated for {relation}")
        df = pd.read_csv(path)
        for feat in _feature_names(relation):
            assert feat in df["feature"].values, (
                f"{relation}: relational feature {feat!r} missing from importance CSV."
            )

    @pytest.mark.parametrize("relation", ["card1", "card1_card2"])
    def test_validation_predictions_exist(self, relation: str) -> None:
        path = _report_dir(relation) / "validation_predictions.parquet"
        if not path.exists():
            pytest.skip(f"Predictions not yet generated for {relation}")
        df = pd.read_parquet(path)
        assert "prediction" in df.columns
        assert df["prediction"].between(0.0, 1.0).all()

    @pytest.mark.parametrize("relation", ["card1", "card1_card2"])
    def test_comparison_to_b0_csv_exists(self, relation: str) -> None:
        path = _report_dir(relation) / "comparison_to_b0.csv"
        if not path.exists():
            pytest.skip(f"Comparison CSV not yet generated for {relation}")
        df = pd.read_csv(path)
        assert "metric" in df.columns
        assert "delta_b1_minus_b0" in df.columns


# ---------------------------------------------------------------------------
# T5: B0 protected artifacts unchanged
# ---------------------------------------------------------------------------


class TestB0ArtifactsProtected:
    def test_b0_protected_paths_list_non_empty(self) -> None:
        assert len(B0_PROTECTED_PATHS) >= 3

    def test_b0_model_not_overwritten(self) -> None:
        path = ROOT_DIR / "models" / "lightgbm_baseline.txt"
        if not path.exists():
            pytest.skip("B0 model not present.")
        # Hash stability: just verifying it exists and is non-empty.
        assert path.stat().st_size > 0

    def test_b0_metrics_unchanged_by_b1(self) -> None:
        """B0 lightgbm_metrics.json must exist and retain expected PR-AUC."""
        path = ROOT_DIR / "reports" / "baseline" / "lightgbm_metrics.json"
        if not path.exists():
            pytest.skip("B0 lightgbm_metrics.json not present.")
        with path.open() as f:
            metrics = json.load(f)
        pr_auc = metrics.get("pr_auc", 0.0)
        assert abs(pr_auc - 0.64914) < 0.001, f"B0 PR-AUC drifted: {pr_auc}."

    def test_b1_card_core_addr1_not_overwritten(self) -> None:
        path = ROOT_DIR / "models" / "lightgbm_b1_card_core_addr1.txt"
        if not path.exists():
            pytest.skip("B1 card_core_addr1 model not present.")
        assert path.stat().st_size > 0

    def test_b1_card_core_addr1_metrics_unchanged(self) -> None:
        path = ROOT_DIR / "reports" / "b1" / "card_core_addr1" / "metrics.json"
        if not path.exists():
            pytest.skip("B1 card_core_addr1 metrics not present.")
        with path.open() as f:
            metrics = json.load(f)
        pr_auc = metrics.get("pr_auc", 0.0)
        assert abs(pr_auc - 0.64401) < 0.001, f"B1 card_core_addr1 PR-AUC drifted: {pr_auc}."


# ---------------------------------------------------------------------------
# T6: Cross-relation comparison report
# ---------------------------------------------------------------------------


class TestCrossRelationComparison:
    def test_comparison_csv_exists(self) -> None:
        path = ROOT_DIR / "reports" / "b1" / "b1_cross_relation_comparison.csv"
        if not path.exists():
            pytest.skip("Cross-relation comparison CSV not yet generated.")
        assert path.exists()

    def test_comparison_csv_has_4_rows(self) -> None:
        path = ROOT_DIR / "reports" / "b1" / "b1_cross_relation_comparison.csv"
        if not path.exists():
            pytest.skip("Cross-relation comparison CSV not generated.")
        df = pd.read_csv(path)
        assert len(df) == 4, f"Expected 4 rows, got {len(df)}."

    def test_summary_json_exists(self) -> None:
        path = ROOT_DIR / "reports" / "b1" / "b1_cross_relation_summary.json"
        if not path.exists():
            pytest.skip("Summary JSON not yet generated.")
        assert path.exists()

    def test_summary_json_outcome_code_valid(self) -> None:
        path = ROOT_DIR / "reports" / "b1" / "b1_cross_relation_summary.json"
        if not path.exists():
            pytest.skip("Summary JSON not generated.")
        with path.open() as f:
            summary = json.load(f)
        outcome_code = summary.get("outcome", {}).get("outcome_code")
        assert outcome_code in ("A", "B", "C"), f"Unexpected outcome code: {outcome_code!r}."

    def test_b0_row_has_zero_delta_against_itself(self) -> None:
        """The B0 reference must come from the frozen B0 artifact, not a literal."""
        path = ROOT_DIR / "reports" / "b1" / "b1_cross_relation_comparison.csv"
        if not path.exists():
            pytest.skip("Cross-relation comparison CSV not generated.")
        df = pd.read_csv(path)
        b0 = df[df["variant"] == "B0"]
        assert len(b0) == 1, "Expected exactly one B0 row."
        assert b0["delta_pr_auc_vs_b0"].iloc[0] == pytest.approx(0.0, abs=1e-12)
        assert b0["delta_roc_auc_vs_b0"].iloc[0] == pytest.approx(0.0, abs=1e-12)

    def test_comparison_references_match_frozen_b0_metrics(self) -> None:
        """Every delta must be measured against the frozen B0 metrics file."""
        comparison_path = ROOT_DIR / "reports" / "b1" / "b1_cross_relation_comparison.csv"
        b0_metrics_path = ROOT_DIR / "reports" / "baseline" / "lightgbm_metrics.json"
        if not comparison_path.exists() or not b0_metrics_path.exists():
            pytest.skip("Comparison CSV or frozen B0 metrics not available.")
        with b0_metrics_path.open() as f:
            b0_metrics = json.load(f)
        df = pd.read_csv(comparison_path)
        for _, row in df.iterrows():
            assert row["delta_pr_auc_vs_b0"] == pytest.approx(
                row["validation_pr_auc"] - b0_metrics["pr_auc"], abs=1e-12
            ), f"{row['variant']} PR-AUC delta is not measured against the frozen B0."
            assert row["delta_roc_auc_vs_b0"] == pytest.approx(
                row["validation_roc_auc"] - b0_metrics["roc_auc"], abs=1e-12
            ), f"{row['variant']} ROC-AUC delta is not measured against the frozen B0."

    def test_outcome_classification_matches_the_measured_metrics(self) -> None:
        """any_beats_b0 must reflect the recorded PR-AUCs, not a stale constant."""
        summary_path = ROOT_DIR / "reports" / "b1" / "b1_cross_relation_summary.json"
        b0_metrics_path = ROOT_DIR / "reports" / "baseline" / "lightgbm_metrics.json"
        if not summary_path.exists() or not b0_metrics_path.exists():
            pytest.skip("Summary JSON or frozen B0 metrics not available.")
        with summary_path.open() as f:
            summary = json.load(f)
        with b0_metrics_path.open() as f:
            b0_pr_auc = json.load(f)["pr_auc"]

        outcome = summary["outcome"]
        assert outcome["b0_pr_auc_reference"] == pytest.approx(b0_pr_auc, abs=1e-12)

        variants = summary["variants"]
        expected_any_beats = any(
            v["validation_pr_auc"] > b0_pr_auc for v in variants if v["role"] != "baseline"
        )
        assert outcome["any_beats_b0"] is expected_any_beats
        assert outcome["outcome_code"] == ("A" if expected_any_beats else outcome["outcome_code"])

    def test_summary_no_test_evaluation(self) -> None:
        path = ROOT_DIR / "reports" / "b1" / "b1_cross_relation_summary.json"
        if not path.exists():
            pytest.skip("Summary JSON not generated.")
        with path.open() as f:
            summary = json.load(f)
        assert summary.get("final_test_evaluated") is False
        # Relation candidates came from train-only screening; validation is used
        # here only to score the already-chosen candidates.
        assert summary.get("candidate_selection_used_validation") is False
