from __future__ import annotations

import unittest

from src.config.paths import ROOT_DIR
from src.models.train_ablation_fixed_budget import (
    FIXED_BUDGET,
    LEAVE_ONE_OUT,
    MODE_REFERENCE,
    PROTOCOL,
    REFERENCES,
    SINGLETON,
    planned_runs,
    resolve_run_paths,
    run_is_complete,
    split_run_name,
)


class PreRegisteredBudgetTests(unittest.TestCase):
    def test_the_budget_is_the_pre_registered_value(self):
        # Fixed and recorded before the module was written. If this changes, the
        # pre-registration no longer describes what was run.
        self.assertEqual(FIXED_BUDGET, 10_000)

    def test_the_budget_sits_above_every_optimum_the_argmax_panel_reached(self):
        # Highest best-iteration observed anywhere in the published panel.
        self.assertGreater(FIXED_BUDGET, 9_137)

    def test_the_budget_stays_below_the_established_cap(self):
        self.assertLess(FIXED_BUDGET, 15_000)

    def test_the_protocol_declares_no_early_stopping(self):
        self.assertIn("no_early_stopping", PROTOCOL)


class PanelCompositionTests(unittest.TestCase):
    def test_the_panel_is_ten_runs(self):
        # Eight ablation cells plus two references retrained at the same budget.
        self.assertEqual(len(planned_runs()), 10)

    def test_both_references_are_retrained(self):
        # An early-stopped reference would reintroduce the asymmetry being removed.
        for reference in REFERENCES:
            self.assertIn(reference, planned_runs())
        self.assertEqual(REFERENCES, ("b0", "b1_card1"))

    def test_all_eight_ablation_cells_are_present(self):
        runs = planned_runs()
        for key in (
            "prior_count",
            "prior_count_24h",
            "prior_count_7d",
            "time_since_previous_hours",
        ):
            self.assertIn(f"{SINGLETON}_{key}", runs)
            self.assertIn(f"{LEAVE_ONE_OUT}_{key}", runs)

    def test_run_names_are_unique(self):
        runs = planned_runs()
        self.assertEqual(len(runs), len(set(runs)))

    def test_singletons_reference_b0_and_leave_one_out_references_b1(self):
        # Carried over from the published panel so the rule transfers unchanged.
        self.assertEqual(MODE_REFERENCE[SINGLETON], "b0")
        self.assertEqual(MODE_REFERENCE[LEAVE_ONE_OUT], "b1_card1")


class RunNameTests(unittest.TestCase):
    def test_a_reference_parses_with_no_feature(self):
        self.assertEqual(split_run_name("b0"), ("b0", None))
        self.assertEqual(split_run_name("b1_card1"), ("b1_card1", None))

    def test_a_cell_parses_into_mode_and_feature(self):
        self.assertEqual(
            split_run_name("singleton_prior_count_24h"), (SINGLETON, "prior_count_24h")
        )
        self.assertEqual(split_run_name("loo_prior_count_7d"), (LEAVE_ONE_OUT, "prior_count_7d"))

    def test_every_planned_run_parses(self):
        for run in planned_runs():
            split_run_name(run)

    def test_an_unknown_run_is_rejected(self):
        with self.assertRaises(ValueError):
            split_run_name("singleton_not_a_feature_at_all")
        with self.assertRaises(ValueError):
            split_run_name("nonsense")


class PathTests(unittest.TestCase):
    def test_paths_record_the_budget_so_a_smoke_run_cannot_overwrite_the_panel(self):
        panel = resolve_run_paths("b0", FIXED_BUDGET)
        smoke = resolve_run_paths("b0", 50)
        self.assertNotEqual(panel["model"], smoke["model"])
        self.assertNotEqual(panel["run_dir"], smoke["run_dir"])
        self.assertIn(str(FIXED_BUDGET), panel["model"].name)

    def test_every_run_writes_to_a_distinct_location(self):
        seen = {resolve_run_paths(run)["run_dir"] for run in planned_runs()}
        self.assertEqual(len(seen), len(planned_runs()))

    def test_outputs_never_touch_the_published_ablation_tree(self):
        # The published panel must survive this experiment untouched.
        for run in planned_runs():
            for key, path in resolve_run_paths(run).items():
                posix = path.as_posix()
                self.assertNotIn("/reports/ablation/", posix)
                self.assertNotIn("/models/ablation/", posix)

    def test_outputs_stay_inside_the_repository(self):
        for run in planned_runs():
            for path in resolve_run_paths(run).values():
                self.assertTrue(str(path).startswith(str(ROOT_DIR)))

    def test_a_nonsense_budget_or_seed_is_rejected(self):
        with self.assertRaises(ValueError):
            resolve_run_paths("b0", 0)
        with self.assertRaises(ValueError):
            resolve_run_paths("b0", FIXED_BUDGET, -1)

    def test_completeness_requires_every_artifact(self):
        # Nothing has been trained at a nonsense budget, so this must be false
        # rather than raising.
        self.assertFalse(run_is_complete("b0", 12_345))


class ProtocolIntegrityTests(unittest.TestCase):
    def test_the_trainer_installs_no_early_stopping_callback(self):
        source = (ROOT_DIR / "src" / "models" / "train_ablation_fixed_budget.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("lightgbm.early_stopping(", source)

    def test_predictions_and_the_saved_model_are_pinned_to_the_budget(self):
        source = (ROOT_DIR / "src" / "models" / "train_ablation_fixed_budget.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("predict_proba(X_validation, num_iteration=budget)", source)
        self.assertIn('save_model(str(paths["model"]), num_iteration=budget)', source)

    def test_it_reuses_the_shared_parameter_validator(self):
        source = (ROOT_DIR / "src" / "models" / "train_ablation_fixed_budget.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "validate_convergence_configuration(model, b0_metadata, seed, budget)", source
        )


if __name__ == "__main__":
    unittest.main()
