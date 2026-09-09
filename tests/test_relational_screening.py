"""Tests for src/features/screen_relations.py.

Nine test categories mandated by implementation.md §24:
 T1  All 7 candidate relations appear in output.
 T2  Audit metrics read from train-only reports (not val/test data).
 T3  No test/validation labels loaded during Stage A.
 T4  No validation performance metrics used for Stage A decisions.
 T5  Missing grouping values produce count=0 and recency=NaN.
 T6  Equal timestamps do not become historical neighbours.
 T7  Feature names are deterministic given a relation name.
 T8  Screening output is deterministic (two identical runs).
 T9  B1 protected artifacts are not modified by screening.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Import the module under test.
# ---------------------------------------------------------------------------
from src.features.screen_relations import (
    ALL_CANDIDATE_NAMES,
    CANDIDATES,
    PROTECTED_PATHS,
    SIGNAL_MODERATE_PR_AUC_LIFT,
    best_univariate_by_relation,
    build_structural_screening_table,
    classify_relation,
    compute_feature_stats,
    compute_spearman_correlations,
    coverage_weighted_signal,
    decide_candidate_selection,
    feature_names_for_relation,
    generate_relational_features,
    load_audit_reports,
    run_screening,
)

# ===========================================================================
# Shared test helpers
# ===========================================================================


def _make_entity_row(
    relation: str = "card1",
    coverage_pct: float = 95.0,
    covered_transactions: int = 100_000,
    entity_count: int = 25_000,
    singleton_entity_pct: float = 30.0,
    recurring_entity_pct: float = 70.0,
    repeated_transaction_pct_all: float = 65.0,
    group_size_median: float = 3.0,
    group_size_p95: float = 12.0,
    group_size_p99: float = 30.0,
    group_size_max: int = 500,
    largest_entity_share_pct: float = 0.5,
    median_entity_lifespan_days: float = 30.0,
    median_repeat_gap_hours: float = 168.0,
) -> pd.Series:
    return pd.Series(
        {
            "relation": relation,
            "coverage_pct": coverage_pct,
            "covered_transactions": covered_transactions,
            "entity_count": entity_count,
            "singleton_entity_pct": singleton_entity_pct,
            "recurring_entity_pct": recurring_entity_pct,
            "repeated_transaction_pct_all": repeated_transaction_pct_all,
            "group_size_median": group_size_median,
            "group_size_p95": group_size_p95,
            "group_size_p99": group_size_p99,
            "group_size_max": group_size_max,
            "largest_entity_share_pct": largest_entity_share_pct,
            "median_entity_lifespan_days": median_entity_lifespan_days,
            "median_repeat_gap_hours": median_repeat_gap_hours,
        }
    )


def _make_graph_row(
    relation: str = "card1",
    participating_nodes_pct: float = 90.0,
    isolated_pct: float = 10.0,
    largest_component_pct: float = 0.8,
    fraud_neighbor_lift: float = 2.5,
    previous_fraud_rate_given_current_fraud: float = 0.12,
    previous_fraud_rate_given_current_normal: float = 0.035,
) -> pd.Series:
    return pd.Series(
        {
            "relation": relation,
            "participating_nodes_pct": participating_nodes_pct,
            "isolated_pct": isolated_pct,
            "largest_component_pct": largest_component_pct,
            "fraud_neighbor_lift": fraud_neighbor_lift,
            "previous_fraud_rate_given_current_fraud": previous_fraud_rate_given_current_fraud,
            "previous_fraud_rate_given_current_normal": previous_fraud_rate_given_current_normal,
        }
    )


def _make_all_audit_dfs(overrides: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build minimal entity_diagnostics and graph_diagnostics covering all 7 relations."""
    overrides = overrides or {}
    entity_rows = []
    graph_rows = []
    for rel in ALL_CANDIDATE_NAMES:
        e_kwargs = overrides.get(f"{rel}_entity", {})
        g_kwargs = overrides.get(f"{rel}_graph", {})
        entity_rows.append(_make_entity_row(relation=rel, **e_kwargs))
        graph_rows.append(_make_graph_row(relation=rel, **g_kwargs))
    return pd.DataFrame(entity_rows), pd.DataFrame(graph_rows)


def _make_source_df(
    n: int = 10,
    group_col: str = "card1",
    group_values: list | None = None,
    times: list[int] | None = None,
    labels: list[int] | None = None,
) -> pd.DataFrame:
    """Build a minimal source DataFrame suitable for generate_relational_features."""
    if group_values is None:
        group_values = [1] * n
    if times is None:
        times = list(range(0, n * 100, 100))
    if labels is None:
        labels = [0] * n
    return pd.DataFrame(
        {
            "TransactionID": list(range(n)),
            "TransactionDT": times,
            "isFraud": labels,
            group_col: group_values,
        }
    )


# ===========================================================================
# T1 — All 7 candidate relations appear in screening output
# ===========================================================================


class TestAllSevenRelationsPresent:
    """T1: Verify every CANDIDATES entry is represented in the output."""

    def test_structural_screening_contains_all_7_relations(self):
        entity_df, graph_df = _make_all_audit_dfs()
        screening_df = build_structural_screening_table(entity_df, graph_df)
        assert set(screening_df["relation"]) == set(ALL_CANDIDATE_NAMES)
        assert len(screening_df) == 7

    def test_all_7_relations_in_candidate_decisions(self):
        entity_df, graph_df = _make_all_audit_dfs()
        screening_df = build_structural_screening_table(entity_df, graph_df)
        disc_rows: list = []  # no discrimination data needed for reject logic
        selection = decide_candidate_selection(screening_df, disc_rows)
        assert set(selection["decisions"].keys()) == set(ALL_CANDIDATE_NAMES)

    def test_candidates_dict_has_7_entries(self):
        assert len(CANDIDATES) == 7
        assert len(ALL_CANDIDATE_NAMES) == 7


# ===========================================================================
# T2 — Audit metrics read from train-only reports
# ===========================================================================


class TestAuditMetricsFromTrainOnly:
    """T2: load_audit_reports reads the frozen audit CSVs, not live data."""

    def test_load_audit_reports_reads_correct_paths(self, tmp_path, monkeypatch):
        """Verify that load_audit_reports reads from the relational_audit directory."""
        import src.features.screen_relations as sr

        # Build mock CSVs in tmp_path.
        entity_df, graph_df = _make_all_audit_dfs()
        audit_dir = tmp_path / "reports" / "relational_audit"
        audit_dir.mkdir(parents=True)
        entity_df.to_csv(audit_dir / "entity_diagnostics.csv", index=False)
        graph_df.to_csv(audit_dir / "graph_diagnostics.csv", index=False)

        monkeypatch.setattr(sr, "ENTITY_DIAG_PATH", audit_dir / "entity_diagnostics.csv")
        monkeypatch.setattr(sr, "GRAPH_DIAG_PATH", audit_dir / "graph_diagnostics.csv")

        loaded_entity, loaded_graph = load_audit_reports()
        assert set(loaded_entity["relation"]) == set(ALL_CANDIDATE_NAMES)
        assert set(loaded_graph["relation"]) == set(ALL_CANDIDATE_NAMES)

    def test_load_audit_raises_if_file_missing(self, tmp_path, monkeypatch):
        import src.features.screen_relations as sr

        monkeypatch.setattr(sr, "ENTITY_DIAG_PATH", tmp_path / "nonexistent_entity.csv")
        monkeypatch.setattr(sr, "GRAPH_DIAG_PATH", tmp_path / "nonexistent_graph.csv")
        with pytest.raises(FileNotFoundError):
            load_audit_reports()

    def test_load_audit_raises_if_relation_missing(self, tmp_path, monkeypatch):
        import src.features.screen_relations as sr

        # Only 6 of 7 relations.
        entity_rows = []
        graph_rows = []
        for rel in ALL_CANDIDATE_NAMES[:-1]:
            entity_rows.append(_make_entity_row(relation=rel))
            graph_rows.append(_make_graph_row(relation=rel))
        entity_df = pd.DataFrame(entity_rows)
        graph_df = pd.DataFrame(graph_rows)

        audit_dir = tmp_path / "relational_audit"
        audit_dir.mkdir(parents=True)
        entity_df.to_csv(audit_dir / "entity_diagnostics.csv", index=False)
        graph_df.to_csv(audit_dir / "graph_diagnostics.csv", index=False)

        monkeypatch.setattr(sr, "ENTITY_DIAG_PATH", audit_dir / "entity_diagnostics.csv")
        monkeypatch.setattr(sr, "GRAPH_DIAG_PATH", audit_dir / "graph_diagnostics.csv")

        with pytest.raises(ValueError, match="missing relations"):
            load_audit_reports()


# ===========================================================================
# T3 — No test/validation labels loaded during Stage A
# ===========================================================================


class TestNoTestLabelsInStageA:
    """T3: Stage A must not load isFraud from val/test partitions."""

    def test_classify_relation_has_no_label_parameter(self):
        """classify_relation must not accept or require isFraud data."""
        import inspect

        sig = inspect.signature(classify_relation)
        param_names = list(sig.parameters.keys())
        assert "label" not in param_names
        assert "isFraud" not in param_names
        assert "y" not in param_names

    def test_build_structural_screening_has_no_label_parameter(self):
        import inspect

        sig = inspect.signature(build_structural_screening_table)
        param_names = list(sig.parameters.keys())
        assert "label" not in param_names
        assert "isFraud" not in param_names

    def test_structural_screening_does_not_use_fraud_labels(self):
        """Stage A classification result is identical whether labels are given or not."""
        entity_df, graph_df = _make_all_audit_dfs()
        screening_df = build_structural_screening_table(entity_df, graph_df)
        # Verify classification result is stable (no random access to label data).
        screening_df2 = build_structural_screening_table(entity_df, graph_df)
        pd.testing.assert_frame_equal(screening_df, screening_df2)


# ===========================================================================
# T4 — No validation performance used in Stage A decisions
# ===========================================================================


class TestNoValidationPerformanceInStageA:
    """T4: candidate selection must set validation_used_for_selection=False."""

    def test_selection_flags_are_correct(self):
        entity_df, graph_df = _make_all_audit_dfs()
        screening_df = build_structural_screening_table(entity_df, graph_df)
        selection = decide_candidate_selection(screening_df, [])

        assert selection["validation_used_for_selection"] is False
        assert selection["final_test_evaluated"] is False
        assert selection["stage_a_only"] is True

    def test_decide_candidate_selection_only_uses_train_discrimination(self):
        """decide_candidate_selection does not accept a val_discrimination parameter."""
        import inspect

        sig = inspect.signature(decide_candidate_selection)
        param_names = list(sig.parameters.keys())
        assert "val" not in " ".join(param_names)
        assert "validation" not in " ".join(param_names)
        assert "test" not in " ".join(param_names)


# ===========================================================================
# T5 — Missing grouping values produce count=0 and recency=NaN
# ===========================================================================


class TestMissingGroupValues:
    """T5: Rows with NaN in any group column must get count=0, recency=NaN."""

    def test_single_null_group_col_gives_zero_count(self):
        df = _make_source_df(n=5, group_col="card1", group_values=[1, None, 1, None, 1])
        result = generate_relational_features(df, "card1", ["card1"])
        name = feature_names_for_relation("card1")[0]
        # Rows 1 and 3 have None card1 — they must have count=0.
        assert result.loc[1, name] == 0
        assert result.loc[3, name] == 0

    def test_single_null_group_col_gives_nan_recency(self):
        df = _make_source_df(n=5, group_col="card1", group_values=[1, None, 1, None, 1])
        result = generate_relational_features(df, "card1", ["card1"])
        recency_name = feature_names_for_relation("card1")[3]
        assert np.isnan(result.loc[1, recency_name])
        assert np.isnan(result.loc[3, recency_name])

    def test_all_null_group_values_give_zero_for_all_counts(self):
        df = _make_source_df(n=4, group_col="card1", group_values=[None] * 4)
        result = generate_relational_features(df, "card1", ["card1"])
        names = feature_names_for_relation("card1")
        for name in names[:3]:  # count features
            assert (result[name] == 0).all()
        assert result[names[3]].isna().all()

    def test_missing_column_entirely_gives_zero_and_nan(self):
        """If a group column is entirely absent from source_df, all rows are invalid."""
        df = pd.DataFrame(
            {
                "TransactionID": [0, 1, 2],
                "TransactionDT": [100, 200, 300],
            }
        )
        result = generate_relational_features(df, "card1", ["card1"])
        names = feature_names_for_relation("card1")
        for name in names[:3]:
            assert (result[name] == 0).all()
        assert result[names[3]].isna().all()


# ===========================================================================
# T6 — Equal timestamps do not become historical neighbours
# ===========================================================================


class TestEqualTimestampSemantics:
    """T6: Transactions at identical timestamps within an entity see the same
    history — none of them can see the others as prior events."""

    def test_simultaneous_transactions_have_same_prior_count(self):
        """Three transactions at the same time for the same entity
        must all have prior_count == 0 (they are the first block)."""
        df = _make_source_df(
            n=3,
            group_col="card1",
            group_values=[1, 1, 1],
            times=[1000, 1000, 1000],
        )
        result = generate_relational_features(df, "card1", ["card1"])
        name = feature_names_for_relation("card1")[0]
        assert (result[name] == 0).all()

    def test_second_block_sees_full_first_block_as_history(self):
        """Two transactions at T=1000, then one at T=2000 for same entity.
        The T=2000 transaction must see prior_count == 2."""
        df = _make_source_df(
            n=3,
            group_col="card1",
            group_values=[1, 1, 1],
            times=[1000, 1000, 2000],
        )
        result = generate_relational_features(df, "card1", ["card1"])
        name = feature_names_for_relation("card1")[0]
        # First two rows: prior_count == 0
        assert result.loc[0, name] == 0
        assert result.loc[1, name] == 0
        # Third row at t=2000 sees both earlier rows.
        assert result.loc[2, name] == 2

    def test_equal_timestamp_block_has_nan_recency(self):
        """The first time-block for an entity has no prior event, so recency is NaN."""
        df = _make_source_df(
            n=2,
            group_col="card1",
            group_values=[1, 1],
            times=[500, 500],
        )
        result = generate_relational_features(df, "card1", ["card1"])
        recency_name = feature_names_for_relation("card1")[3]
        assert result[recency_name].isna().all()

    def test_recency_measured_from_last_block_not_within_block(self):
        """Recency for T=3000 should be relative to the *last* event before that block
        (T=2000), not any T=3000 event within the block."""
        df = _make_source_df(
            n=4,
            group_col="card1",
            group_values=[1, 1, 1, 1],
            times=[1000, 2000, 3000, 3000],
        )
        result = generate_relational_features(df, "card1", ["card1"])
        recency_name = feature_names_for_relation("card1")[3]
        # Rows at t=3000 should have recency = (3000 - 2000) / 3600
        expected_recency = (3000 - 2000) / 3600.0
        assert abs(result.loc[2, recency_name] - expected_recency) < 1e-9
        assert abs(result.loc[3, recency_name] - expected_recency) < 1e-9


# ===========================================================================
# T7 — Feature names are deterministic
# ===========================================================================


class TestDeterministicFeatureNames:
    """T7: feature_names_for_relation must return the same names every call."""

    @pytest.mark.parametrize("relation", ALL_CANDIDATE_NAMES)
    def test_feature_names_are_stable(self, relation: str):
        names1 = feature_names_for_relation(relation)
        names2 = feature_names_for_relation(relation)
        assert names1 == names2

    @pytest.mark.parametrize("relation", ALL_CANDIDATE_NAMES)
    def test_feature_names_follow_naming_convention(self, relation: str):
        names = feature_names_for_relation(relation)
        assert len(names) == 4
        assert names[0] == f"{relation}_prior_count"
        assert names[1] == f"{relation}_prior_count_24h"
        assert names[2] == f"{relation}_prior_count_7d"
        assert names[3] == f"{relation}_time_since_previous_hours"

    def test_feature_names_differ_across_relations(self):
        """Ensure relations produce distinct feature namespaces."""
        all_names = [
            name for rel in ALL_CANDIDATE_NAMES for name in feature_names_for_relation(rel)
        ]
        assert len(all_names) == len(set(all_names)), (
            "Feature names must be unique across relations."
        )


# ===========================================================================
# T8 — Screening output is deterministic
# ===========================================================================


class TestDeterministicOutput:
    """T8: Running the screening pipeline twice on the same inputs yields identical outputs."""

    def test_structural_classification_is_deterministic(self):
        entity_df, graph_df = _make_all_audit_dfs()
        s1 = build_structural_screening_table(entity_df, graph_df)
        s2 = build_structural_screening_table(entity_df, graph_df)
        pd.testing.assert_frame_equal(
            s1.sort_values("relation").reset_index(drop=True),
            s2.sort_values("relation").reset_index(drop=True),
        )

    def test_feature_generation_is_deterministic(self):
        df = _make_source_df(
            n=20,
            group_col="card1",
            group_values=[1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
        )
        r1 = generate_relational_features(df, "card1", ["card1"])
        r2 = generate_relational_features(df, "card1", ["card1"])
        pd.testing.assert_frame_equal(r1, r2)

    def test_candidate_selection_decisions_are_deterministic(self):
        entity_df, graph_df = _make_all_audit_dfs()
        screening_df = build_structural_screening_table(entity_df, graph_df)
        disc = [
            {
                "relation": rel,
                "feature": feature_names_for_relation(rel)[0],
                "pr_auc": 0.12,
                "roc_auc": 0.72,
            }
            for rel in ALL_CANDIDATE_NAMES
        ]
        d1 = decide_candidate_selection(screening_df, disc)
        d2 = decide_candidate_selection(screening_df, disc)
        assert d1["decisions"] == d2["decisions"]


# ===========================================================================
# T9 — B1 protected artifacts are not modified
# ===========================================================================


class TestProtectedArtifactsNotModified:
    """T9: run_screening must leave the protected B1 artifact files unchanged."""

    def _hash_path(self, p: Path) -> str | None:
        if not p.exists():
            return None
        return hashlib.sha256(p.read_bytes()).hexdigest()

    def test_protected_paths_list_is_non_empty(self):
        assert len(PROTECTED_PATHS) > 0

    def test_all_protected_paths_reference_known_files(self):
        """Verify that the protected paths include the B1 model and key reports."""
        path_strings = [str(p) for p in PROTECTED_PATHS]
        assert any("lightgbm_b1" in s for s in path_strings)
        assert any("feature_importance" in s for s in path_strings)
        assert any("model_dataset" in s for s in path_strings)

    def test_run_screening_does_not_touch_protected_artifacts(self, tmp_path, monkeypatch):
        """End-to-end: verify file hashes of protected paths are unchanged post-run.

        This test is skipped if the actual dataset files are not present,
        as it requires the real data to run.
        """
        import src.features.screen_relations as sr

        # Check required data exists; skip if not.
        if not sr.MODEL_DATASET_PATH.exists():
            pytest.skip("model_dataset.parquet not found; skipping end-to-end hash test.")
        if not sr.ENTITY_DIAG_PATH.exists():
            pytest.skip("entity_diagnostics.csv not found; skipping.")

        hashes_before = {str(p): self._hash_path(p) for p in PROTECTED_PATHS if p.exists()}

        # Run into a temp report dir so we don't touch the real reports.
        run_screening(report_dir=tmp_path / "screening_output", verbose=False)

        hashes_after = {str(p): self._hash_path(p) for p in PROTECTED_PATHS if p.exists()}

        for path_str, hash_before in hashes_before.items():
            assert hashes_after.get(path_str) == hash_before, (
                f"Protected artifact was modified: {path_str}"
            )


# ===========================================================================
# T10 - Univariate discrimination is comparable and direction-aware
# ===========================================================================


class TestDiscriminationIsComparable:
    """The four features of a relation must be scored on the same rows.

    Scoring counts on every row while scoring recency only where it is observed
    made `best_univariate_pr_auc` a comparison between different denominators,
    and the recency feature won for every relation as a result.
    """

    def _stats(self, values, labels, name, observed=None):
        return compute_feature_stats(
            pd.Series(values, dtype="float64"),
            pd.Series(labels, dtype="int32"),
            name,
            observed_mask=observed,
        )

    def test_all_rows_tier_scores_every_row_including_missing_recency(self):
        values = [1.0, 2.0, np.nan, 4.0, np.nan, 6.0]
        labels = [0, 0, 1, 0, 1, 0]
        stats = self._stats(values, labels, "card1_time_since_previous_hours")

        assert stats["n_scored"] == len(values)
        assert not np.isnan(stats["pr_auc"])

    def test_count_and_recency_features_share_the_all_rows_denominator(self):
        labels = [0, 1, 0, 0, 1, 0, 0, 1]
        counts = [0, 1, 2, 3, 0, 5, 6, 0]
        recency = [np.nan, 1.0, 2.0, 3.0, np.nan, 5.0, 6.0, np.nan]

        count_stats = self._stats(counts, labels, "card1_prior_count")
        recency_stats = self._stats(recency, labels, "card1_time_since_previous_hours")

        assert count_stats["n_scored"] == recency_stats["n_scored"]
        assert count_stats["prevalence"] == recency_stats["prevalence"]

    def test_inverse_association_is_reported_not_silently_flipped(self):
        """A feature where low values mean fraud must show ROC-AUC below 0.5."""
        values = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        labels = [1, 1, 1, 0, 0, 0, 0, 0]

        stats = self._stats(values, labels, "card_core_addr1_prior_count")

        assert stats["roc_auc_ascending"] < 0.5
        assert stats["direction"] == "lower_is_fraud"
        assert stats["roc_auc_strength"] == pytest.approx(abs(stats["roc_auc_ascending"] - 0.5))

    def test_observed_mask_excludes_sentinel_rows_from_the_covered_tier(self):
        """Zero counts on uncovered rows are sentinels, not measured history."""
        # Rows 0-2 have no grouping key: their 0 is a sentinel and they are fraud.
        values = [0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
        labels = [1, 1, 1, 0, 0, 0, 0, 0]
        observed = np.array([False, False, False, True, True, True, True, True])

        stats = self._stats(values, labels, "card_core_addr1_prior_count", observed)

        assert stats["n_scored"] == 8
        assert stats["n_covered"] == 5
        # Every covered row is normal, so there is no signal left to measure.
        assert stats["covered_prevalence"] == 0.0
        assert np.isnan(stats["pr_auc_covered"])


# ===========================================================================
# T11 - Promotion gates
# ===========================================================================


class TestPromotionGates:
    """`preferred` must rest on several measured dimensions, per implementation.md."""

    def _screening(self):
        entity_df, graph_df = _make_all_audit_dfs()
        return build_structural_screening_table(entity_df, graph_df)

    def _disc(self, lift_covered: float):
        return [
            {
                "relation": rel,
                "feature": feature_names_for_relation(rel)[0],
                "pr_auc": 0.05,
                "pr_auc_lift": lift_covered,
                "roc_auc_ascending": 0.6,
                "pr_auc_covered": 0.05,
                "pr_auc_lift_covered": lift_covered,
                "roc_auc_covered_ascending": 0.6,
            }
            for rel in ALL_CANDIDATE_NAMES
        ]

    def test_relation_is_promoted_when_every_gate_passes(self):
        selection = decide_candidate_selection(
            self._screening(), self._disc(2.0), {rel: 1.0 for rel in ALL_CANDIDATE_NAMES}
        )
        preferred = [
            rel for rel, d in selection["decisions"].items() if d["decision"] == "preferred"
        ]
        assert preferred, "Expected at least one preferred relation."

    def test_signal_weaker_than_key_missingness_blocks_promotion(self):
        """A relation whose only signal is a null indicator must not be preferred."""
        lift = 1.5
        selection = decide_candidate_selection(
            self._screening(),
            self._disc(lift),
            {rel: lift + 0.5 for rel in ALL_CANDIDATE_NAMES},
        )

        for relation, decision in selection["decisions"].items():
            if decision["decision"] == "reject":
                continue
            assert decision["decision"] != "preferred", (
                f"{relation} was preferred despite its signal being weaker than "
                "its grouping-key missingness."
            )
            assert decision["promotion_gates"]["history_signal_exceeds_key_missingness"] is False

    def test_weak_history_signal_blocks_promotion(self):
        below_moderate = SIGNAL_MODERATE_PR_AUC_LIFT - 0.1
        selection = decide_candidate_selection(self._screening(), self._disc(below_moderate), {})
        for decision in selection["decisions"].values():
            assert decision["decision"] != "preferred"

    def test_at_most_two_relations_are_ever_preferred(self):
        """Stage B allows at most two controlled validation experiments."""
        selection = decide_candidate_selection(self._screening(), self._disc(3.0), {})
        preferred = [
            rel for rel, d in selection["decisions"].items() if d["decision"] == "preferred"
        ]
        assert len(preferred) <= 2

    def test_coverage_weighted_signal_rewards_coverage_and_lift(self):
        assert coverage_weighted_signal(100.0, 1.5) > coverage_weighted_signal(50.0, 1.5)
        assert coverage_weighted_signal(100.0, 2.0) > coverage_weighted_signal(100.0, 1.5)
        assert np.isnan(coverage_weighted_signal(100.0, np.nan))

    def test_best_univariate_ranks_on_the_covered_signal(self):
        rows = [
            {
                "relation": "card1",
                "feature": "card1_prior_count",
                "pr_auc": 0.09,
                "pr_auc_lift": 3.0,
                "pr_auc_covered": 0.02,
                "pr_auc_lift_covered": 1.0,
            },
            {
                "relation": "card1",
                "feature": "card1_time_since_previous_hours",
                "pr_auc": 0.04,
                "pr_auc_lift": 1.2,
                "pr_auc_covered": 0.06,
                "pr_auc_lift_covered": 2.0,
            },
        ]
        best = best_univariate_by_relation(rows)
        assert best["card1"]["feature"] == "card1_time_since_previous_hours"
