from __future__ import annotations

import unittest

from src.models.build_experiment_ledger import (
    FAMILIES,
    NOT_SUITABLE_FOR,
    UNIVERSAL_LIMITATIONS,
    build_entry,
    build_ledger,
    discover_models,
    match_family,
    summarize,
)


class CompletenessTests(unittest.TestCase):
    """The acceptance criterion: every artifact has an entry, enforced not claimed."""

    def setUp(self):
        self.models = discover_models()
        if not self.models:
            self.skipTest("No model artifacts present.")

    def test_every_discovered_artifact_matches_a_family(self):
        unmatched = [path for path in self.models if match_family(path) is None]
        self.assertEqual(unmatched, [], f"Undocumented model artifacts: {unmatched}")

    def test_the_ledger_covers_every_discovered_artifact(self):
        entries = build_ledger()
        self.assertEqual(sorted(entry["artifact"] for entry in entries), sorted(self.models))

    def test_an_unknown_artifact_raises_rather_than_being_skipped(self):
        # The property that keeps the ledger from silently going stale: a new
        # kind of model must be documented before it can be indexed.
        with self.assertRaises(KeyError):
            build_entry("lightgbm_some_future_stage.txt", set())

    def test_family_patterns_are_mutually_exclusive(self):
        # match_family returns the first hit, so overlapping patterns would make
        # an artifact's family depend on declaration order. Check every pattern
        # independently rather than re-deriving the same first match.
        import re

        for path in self.models:
            matched = [f["family"] for f in FAMILIES if re.match(f["pattern"], path)]
            self.assertEqual(len(matched), 1, f"{path} matched {matched}")


class EntryContentTests(unittest.TestCase):
    def setUp(self):
        if not discover_models():
            self.skipTest("No model artifacts present.")
        self.entries = build_ledger()

    def test_every_entry_states_its_limitations(self):
        for entry in self.entries:
            self.assertTrue(entry["limitations"])
            for limitation in UNIVERSAL_LIMITATIONS:
                self.assertIn(limitation, entry["limitations"])

    def test_every_entry_states_what_it_is_not_for(self):
        for entry in self.entries:
            self.assertEqual(entry["not_suitable_for"], NOT_SUITABLE_FOR)
            self.assertIn("Not a deployable fraud model", entry["not_suitable_for"])

    def test_every_entry_names_the_claims_resting_on_it(self):
        for entry in self.entries:
            self.assertTrue(entry["claims_resting_on_it"])

    def test_every_entry_declares_a_stage_and_a_role(self):
        for entry in self.entries:
            self.assertTrue(entry["stage"])
            self.assertTrue(entry["role"])

    def test_cap_bound_runs_carry_the_cap_limitation(self):
        # A run that stopped at its cap understates the model, and the ledger
        # must say so on that specific entry rather than only in prose.
        capped = [e for e in self.entries if e["estimator_cap_reached"]]
        self.assertTrue(capped, "Expected at least one cap-bound frozen run.")
        for entry in capped:
            self.assertTrue(
                any("stopped at the cap" in line for line in entry["limitations"]),
                f"{entry['artifact']} reached its cap without saying so",
            )

    def test_uncapped_runs_do_not_carry_the_cap_limitation(self):
        for entry in self.entries:
            if entry["estimator_cap_reached"] is False:
                self.assertFalse(
                    any("stopped at the cap" in line for line in entry["limitations"]),
                    f"{entry['artifact']} did not reach its cap but claims it did",
                )

    def test_no_artifact_claims_a_test_evaluation(self):
        for entry in self.entries:
            self.assertFalse(entry["test_evaluated"])

    def test_the_frozen_baseline_is_identified_exactly_once(self):
        baselines = [e for e in self.entries if e["role"] == "frozen_baseline"]
        self.assertEqual(len(baselines), 1)
        self.assertEqual(baselines[0]["artifact"], "lightgbm_baseline.txt")

    def test_discarded_baseline_alternatives_are_marked_as_such(self):
        discarded = [e for e in self.entries if e["role"] == "discarded_alternative"]
        self.assertEqual(len(discarded), 3)
        for entry in discarded:
            self.assertIn("no downstream claim", " ".join(entry["claims_resting_on_it"]))

    def test_encoders_are_not_expected_to_carry_prediction_metrics(self):
        # They emit vectors, not scores, so a missing metrics manifest is
        # correct rather than a gap in the ledger.
        encoders = [e for e in self.entries if e["role"] == "encoder"]
        self.assertTrue(encoders)
        for entry in encoders:
            self.assertFalse(entry["metrics_manifest_found"])
            self.assertIsNone(entry["validation_pr_auc"])

    def test_classifier_entries_resolve_their_metrics_manifest(self):
        for entry in self.entries:
            if entry["role"] == "encoder":
                continue
            self.assertTrue(
                entry["metrics_manifest_found"],
                f"{entry['artifact']} has no metrics manifest",
            )
            self.assertIsNotNone(entry["validation_pr_auc"])


class FactsComeFromManifestsTests(unittest.TestCase):
    def setUp(self):
        if not discover_models():
            self.skipTest("No model artifacts present.")

    def test_metrics_match_the_manifest_rather_than_a_transcribed_constant(self):
        # Read the manifest independently and compare, so a hand-edited literal
        # in the ledger source would fail here.
        import json
        from pathlib import Path

        from src.models.build_experiment_ledger import REPORTS_DIR

        entry = build_entry("lightgbm_baseline.txt", set())
        manifest = json.loads(
            Path(REPORTS_DIR / "baseline" / "lightgbm_metrics.json").read_text(encoding="utf-8")
        )
        self.assertEqual(entry["validation_pr_auc"], manifest["pr_auc"])
        self.assertEqual(entry["validation_roc_auc"], manifest["roc_auc"])

    def test_tracked_flag_distinguishes_committed_from_local_artifacts(self):
        entries = build_ledger()
        flags = {entry["tracked_in_git"] for entry in entries}
        self.assertTrue(flags.issubset({True, False}))


class SummaryTests(unittest.TestCase):
    def setUp(self):
        if not discover_models():
            self.skipTest("No model artifacts present.")
        self.summary = summarize(build_ledger())

    def test_counts_add_up(self):
        self.assertEqual(
            self.summary["n_tracked_in_git"] + self.summary["n_untracked"],
            self.summary["n_artifacts"],
        )

    def test_summary_reports_no_test_evaluation(self):
        self.assertFalse(self.summary["any_artifact_evaluated_on_test"])

    def test_summary_carries_the_universal_disclosures(self):
        self.assertEqual(self.summary["universal_limitations"], UNIVERSAL_LIMITATIONS)
        self.assertEqual(self.summary["not_suitable_for"], NOT_SUITABLE_FOR)

    def test_family_counts_cover_every_artifact(self):
        self.assertEqual(
            sum(self.summary["artifacts_by_family"].values()), self.summary["n_artifacts"]
        )


if __name__ == "__main__":
    unittest.main()
