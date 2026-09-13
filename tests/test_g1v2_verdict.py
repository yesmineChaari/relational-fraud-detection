from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.paths import CONFIGS_DIR
from src.features.build_relational_features import _feature_names
from src.graph.temporal_contract import RELATION_NAME
from src.graph.train_graphsage_encoder_v2 import COUNT_BLIND, READOUT, V2Run
from src.graph.train_graphsage_variants import CROSS_FIT_FOLDS
from src.models.compare_g1v2_verdict import (
    VERDICT_NEGATIVE_STANDS,
    VERDICT_WIN,
    assert_matches_preregistration,
    evaluate_criteria,
    selection_bias_screen_from_curves,
)
from src.models.g1v2_preregistration import PREREGISTRATION_PATH, load_g1v2_preregistration
from src.models.train_ablation_fixed_budget import FIXED_BUDGET
from src.models.train_g1v2_seed_panel import (
    CONTROLS_MODULE,
    ENCODER_MODULE,
    VERDICT_MODULE,
    planned_steps,
)
from src.models.train_lightgbm_baseline import RANDOM_SEED
from src.models.train_lightgbm_g1 import embedding_feature_names
from src.models.train_lightgbm_g1v2 import (
    EXPECTED_G1V2_FEATURE_COUNT,
    PROTOCOL,
    build_g1v2_feature_manifest,
    encoder_backed_run,
)
from src.models.train_lightgbm_g1v2_controls import shuffled_run


def committed() -> dict:
    return load_g1v2_preregistration()


def write_payload(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def committed_payload() -> dict:
    return json.loads(PREREGISTRATION_PATH.read_text(encoding="utf-8"))


class PreregistrationFileTests(unittest.TestCase):
    def test_the_committed_file_loads_and_lives_under_configs(self) -> None:
        prereg = committed()
        self.assertEqual(PREREGISTRATION_PATH.parent, CONFIGS_DIR)
        self.assertEqual(prereg["design"]["encoder_seeds"], [42, 43, 44])
        self.assertEqual(prereg["design"]["primary_encoder_seed"], 42)

    def test_the_code_implements_the_registered_design(self) -> None:
        design = committed()["design"]
        self.assertEqual(design["relation"], RELATION_NAME)
        self.assertEqual(design["encoder_readout"], READOUT)
        self.assertEqual(design["cross_fit_folds"], CROSS_FIT_FOLDS)
        self.assertEqual(design["lightgbm_fixed_budget"], FIXED_BUDGET)
        self.assertEqual(design["lightgbm_seed"], RANDOM_SEED)

    def _load_modified(self, mutate) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "prereg.json"
            payload = committed_payload()
            mutate(payload)
            write_payload(path, payload)
            load_g1v2_preregistration(path)

    def test_underscore_commentary_is_ignored(self) -> None:
        def add_comment(payload: dict) -> None:
            payload["design"]["_note"] = "commentary"

        self._load_modified(add_comment)

    def test_an_unknown_key_is_rejected(self) -> None:
        def typo(payload: dict) -> None:
            payload["criteria"]["a_beats_b1_card1_"] = "typo"

        with self.assertRaises(KeyError) as caught:
            self._load_modified(typo)
        self.assertIn("unknown", str(caught.exception).lower())

    def test_a_missing_key_is_rejected(self) -> None:
        def drop(payload: dict) -> None:
            del payload["criteria"]["d_sign_consistent"]

        with self.assertRaises(KeyError):
            self._load_modified(drop)

    def test_a_single_seed_is_rejected(self) -> None:
        def one_seed(payload: dict) -> None:
            payload["design"]["encoder_seeds"] = [42]

        with self.assertRaises(ValueError):
            self._load_modified(one_seed)

    def test_duplicate_seeds_are_rejected(self) -> None:
        def duplicate(payload: dict) -> None:
            payload["design"]["encoder_seeds"] = [42, 42, 43]

        with self.assertRaises(ValueError):
            self._load_modified(duplicate)

    def test_a_primary_seed_outside_the_panel_is_rejected(self) -> None:
        def stray(payload: dict) -> None:
            payload["design"]["primary_encoder_seed"] = 7

        with self.assertRaises(ValueError):
            self._load_modified(stray)

    def test_unshared_fold_initialisation_is_rejected(self) -> None:
        def unshared(payload: dict) -> None:
            payload["design"]["shared_fold_initialisation"] = False

        with self.assertRaises(ValueError):
            self._load_modified(unshared)

    def test_an_empty_criterion_is_rejected(self) -> None:
        def blank(payload: dict) -> None:
            payload["criteria"]["b_beats_width_null"] = "  "

        with self.assertRaises(ValueError):
            self._load_modified(blank)

    def test_a_missing_file_is_rejected_with_a_clear_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                load_g1v2_preregistration(Path(tmp) / "absent.json")
        self.assertIn("not optional", str(caught.exception))


def seed_result(delta: float, lower: float, upper: float, usable: bool = True) -> dict:
    return {
        "metrics": {"pr_auc": 0.66 + delta},
        "significance": {"observed_delta": delta, "ci_lower_95": lower, "ci_upper_95": upper},
        "gate": {
            "usable": usable,
            "informativeness_gap": {"gap": 0.0 if usable else 0.05},
            "cross_fit_provenance": {"provenance_sound": True},
            "partition_alignment_diagnostic": {"mean_gap": 0.1},
        },
    }


WINNING_NULL = {"observed_delta": 0.02, "ci_lower_95": 0.01, "ci_upper_95": 0.03}


def winning_panel() -> dict:
    return {
        42: seed_result(0.006, 0.002, 0.010),
        43: seed_result(0.004, -0.001, 0.009),
        44: seed_result(0.005, 0.000, 0.010),
    }


class EvaluateCriteriaTests(unittest.TestCase):
    def test_all_four_criteria_passing_is_a_win(self) -> None:
        outcome = evaluate_criteria(committed(), winning_panel(), WINNING_NULL)
        self.assertTrue(outcome["decidable"])
        self.assertEqual(outcome["verdict"], VERDICT_WIN)
        self.assertEqual(outcome["failed_criteria"], [])
        self.assertTrue(outcome["stage2_gate"]["directional_win"])
        self.assertTrue(outcome["count_blind_attribution_permitted"])

    def test_every_criterion_carries_its_registered_rule(self) -> None:
        prereg = committed()
        outcome = evaluate_criteria(prereg, winning_panel(), WINNING_NULL)
        for key, entry in outcome["criteria"].items():
            self.assertEqual(entry["rule"], prereg["criteria"][key])

    def test_a_primary_ci_touching_zero_fails_a(self) -> None:
        panel = winning_panel()
        panel[42] = seed_result(0.006, -0.0001, 0.012)
        outcome = evaluate_criteria(committed(), panel, WINNING_NULL)
        self.assertEqual(outcome["verdict"], VERDICT_NEGATIVE_STANDS)
        self.assertEqual(outcome["failed_criteria"], ["a_beats_b1_card1"])

    def test_not_beating_the_width_null_fails_b(self) -> None:
        null = {"observed_delta": 0.001, "ci_lower_95": -0.002, "ci_upper_95": 0.004}
        outcome = evaluate_criteria(committed(), winning_panel(), null)
        self.assertEqual(outcome["failed_criteria"], ["b_beats_width_null"])

    def test_one_unusable_seed_fails_c(self) -> None:
        panel = winning_panel()
        panel[44] = seed_result(0.005, 0.000, 0.010, usable=False)
        outcome = evaluate_criteria(committed(), panel, WINNING_NULL)
        self.assertEqual(outcome["failed_criteria"], ["c_leakage_gate"])
        self.assertFalse(outcome["count_blind_attribution_permitted"])

    def test_one_seed_with_a_negative_delta_fails_d(self) -> None:
        panel = winning_panel()
        panel[43] = seed_result(-0.001, -0.005, 0.003)
        outcome = evaluate_criteria(committed(), panel, WINNING_NULL)
        self.assertEqual(outcome["failed_criteria"], ["d_sign_consistent"])

    def test_the_stage2_gate_is_reported_even_when_the_verdict_is_negative(self) -> None:
        panel = {
            42: seed_result(0.003, -0.001, 0.007),
            43: seed_result(0.001, -0.003, 0.005),
            44: seed_result(-0.001, -0.005, 0.003),
        }
        outcome = evaluate_criteria(committed(), panel, WINNING_NULL)
        self.assertEqual(outcome["verdict"], VERDICT_NEGATIVE_STANDS)
        self.assertTrue(outcome["stage2_gate"]["directional_win"])

    def test_a_missing_seed_is_not_decidable(self) -> None:
        panel = winning_panel()
        panel[44] = None
        outcome = evaluate_criteria(committed(), panel, WINNING_NULL)
        self.assertFalse(outcome["decidable"])
        self.assertIsNone(outcome["verdict"])
        self.assertIn("seed44", outcome["missing_runs"])

    def test_a_missing_width_null_is_not_decidable(self) -> None:
        outcome = evaluate_criteria(committed(), winning_panel(), None)
        self.assertFalse(outcome["decidable"])
        self.assertIn("width_null", outcome["missing_runs"])


def matching_result() -> dict:
    return {
        "metadata": {
            "protocol": PROTOCOL,
            "fixed_budget_trees": FIXED_BUDGET,
            "lightgbm_parameters": {"random_state": RANDOM_SEED},
            "encoder": {
                "random_seed": 42,
                "readout": READOUT,
                "count_mode": "with_counts",
                "cross_fit_folds": CROSS_FIT_FOLDS,
                "shared_initialisation": True,
            },
        }
    }


class PreregistrationMatchTests(unittest.TestCase):
    def test_a_matching_run_is_accepted(self) -> None:
        assert_matches_preregistration(matching_result(), committed(), 42)

    def test_a_run_with_another_readout_is_refused(self) -> None:
        result = matching_result()
        result["metadata"]["encoder"]["readout"] = "self_and_neighbourhood"
        with self.assertRaises(ValueError):
            assert_matches_preregistration(result, committed(), 42)

    def test_a_count_blind_run_is_not_evidence_for_the_registered_design(self) -> None:
        result = matching_result()
        result["metadata"]["encoder"]["count_mode"] = COUNT_BLIND
        with self.assertRaises(ValueError):
            assert_matches_preregistration(result, committed(), 42)

    def test_a_run_filed_under_the_wrong_seed_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            assert_matches_preregistration(matching_result(), committed(), 43)

    def test_an_unshared_initialisation_is_refused(self) -> None:
        result = copy.deepcopy(matching_result())
        result["metadata"]["encoder"]["shared_initialisation"] = False
        with self.assertRaises(ValueError):
            assert_matches_preregistration(result, committed(), 42)


class ManifestAndRunLayoutTests(unittest.TestCase):
    def test_manifest_is_b1_card1_then_the_embedding_block(self) -> None:
        b0 = ["a", "b"]
        card1 = _feature_names("card1")
        embedding = embedding_feature_names()
        manifest = build_g1v2_feature_manifest(b0, card1, embedding)
        self.assertEqual(manifest, [*b0, *card1, *embedding])
        self.assertEqual(EXPECTED_G1V2_FEATURE_COUNT, 471)

    def test_manifest_rejects_an_overlap(self) -> None:
        with self.assertRaises(ValueError):
            build_g1v2_feature_manifest(["embedding_00"], _feature_names("card1"), ["embedding_00"])

    def test_runs_read_their_own_encoder_artifacts(self) -> None:
        run = encoder_backed_run(43)
        encoder = V2Run(43)
        self.assertEqual(run.embeddings_path, encoder.embeddings_path)
        self.assertEqual(run.leakage_gate_path, encoder.leakage_gate_path)

    def test_no_two_runs_share_a_report_directory(self) -> None:
        runs = [
            encoder_backed_run(42),
            encoder_backed_run(43),
            encoder_backed_run(42, COUNT_BLIND),
            shuffled_run(42),
        ]
        self.assertEqual(len({run.report_dir for run in runs}), len(runs))
        self.assertEqual(len({run.model_path for run in runs}), len(runs))

    def test_the_width_null_has_no_leakage_gate(self) -> None:
        self.assertIsNone(shuffled_run(42).leakage_gate_path)


class SeedPanelTests(unittest.TestCase):
    def test_primary_seed_and_its_width_null_come_first_and_the_verdict_last(self) -> None:
        steps = planned_steps(committed())
        self.assertEqual(steps[0].args, ("--seed", "42"))
        self.assertEqual(steps[2].module, CONTROLS_MODULE)
        self.assertEqual(steps[-1].module, VERDICT_MODULE)
        self.assertFalse(steps[-1].resumable)

    def test_every_registered_seed_gets_an_encoder_and_a_lightgbm_step(self) -> None:
        steps = planned_steps(committed())
        encoder_seeds = [s.args[1] for s in steps if s.module == ENCODER_MODULE]
        self.assertEqual(sorted(encoder_seeds), ["42", "43", "44"])
        self.assertEqual(len(steps), 2 * 3 + 1 + 1)

    def test_every_training_step_resumes(self) -> None:
        for step in planned_steps(committed())[:-1]:
            self.assertTrue(step.resumable)
            self.assertIn("--skip-existing", step.command())

    def test_the_count_blind_arm_appears_only_on_request(self) -> None:
        plain = planned_steps(committed())
        with_arm = planned_steps(committed(), attribution=True)
        self.assertFalse(any("--count-blind" in s.args for s in plain))
        self.assertEqual(sum("--count-blind" in s.args for s in with_arm), 2)


class SelectionBiasScreenTests(unittest.TestCase):
    def test_equal_budget_curves_agree_and_serialise(self) -> None:
        reference = pd.DataFrame(
            {
                "iteration": np.arange(1, 101),
                "validation_average_precision": np.linspace(0.50, 0.66, 100),
                "validation_auc": np.linspace(0.90, 0.93, 100),
            }
        )
        variant = reference.assign(
            validation_average_precision=reference["validation_average_precision"] - 0.03
        )
        screen = selection_bias_screen_from_curves(reference, variant, noise_floor=0.0005)
        json.dumps(screen)
        self.assertEqual(screen["argmax_minus_matched"], 0.0)
        self.assertTrue(screen["estimators_agree"])
        self.assertAlmostEqual(screen["delta_plateau"], -0.03)
        self.assertFalse(screen["gating"])


if __name__ == "__main__":
    unittest.main()
