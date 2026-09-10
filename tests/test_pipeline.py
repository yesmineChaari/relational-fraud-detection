from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from src.config.paths import ROOT_DIR
from src.pipeline import (
    STAGE_ORDER,
    STAGES,
    Stage,
    describe_unmet,
    resolve_plan,
    run_stage,
    unmet_prerequisites,
)


def make_stage(name, outputs=(), requires=()):
    return Stage(
        name=name,
        description=f"{name} description",
        commands=(("-m", f"src.{name}"),),
        outputs=tuple(outputs),
        requires=tuple(requires),
    )


class RegistryTests(unittest.TestCase):
    def test_every_stage_the_ticket_names_is_registered(self):
        for expected in (
            "profile",
            "model_dataset",
            "temporal_split",
            "relational_audit",
            "screening",
            "relational_features",
            "train_b1",
            "compare",
        ):
            self.assertIn(expected, STAGES)

    def test_declaration_order_is_execution_order(self):
        self.assertEqual(list(STAGES), list(STAGE_ORDER))

    def test_the_split_runs_before_the_model_dataset(self):
        # The dependency the ticket had backwards: the split writes the manifest
        # the model dataset reads, so reversing them cannot work.
        self.assertIn("temporal_split", STAGES["model_dataset"].requires)
        self.assertLess(STAGE_ORDER.index("temporal_split"), STAGE_ORDER.index("model_dataset"))

    def test_every_requirement_names_a_registered_stage(self):
        for stage in STAGES.values():
            for required in stage.requires:
                self.assertIn(required, STAGES, f"{stage.name} requires unknown {required}")

    def test_no_stage_requires_something_declared_after_it(self):
        for position, name in enumerate(STAGE_ORDER):
            for required in STAGES[name].requires:
                self.assertLess(
                    STAGE_ORDER.index(required), position, f"{name} requires later {required}"
                )

    def test_stage_names_match_their_keys(self):
        for key, stage in STAGES.items():
            self.assertEqual(key, stage.name)

    def test_every_stage_has_at_least_one_command(self):
        for stage in STAGES.values():
            self.assertTrue(stage.commands)

    def test_outputs_are_inside_the_repository(self):
        for stage in STAGES.values():
            for output in stage.outputs:
                self.assertTrue(str(output).startswith(str(ROOT_DIR)))

    def test_the_verification_gate_declares_no_outputs(self):
        # A gate must never be skipped as "already complete".
        self.assertEqual(STAGES["verify_raw"].outputs, ())
        self.assertFalse(STAGES["verify_raw"].is_complete())

    def test_the_orchestrator_does_not_import_stage_logic(self):
        # It sequences modules; importing them would let it reimplement one.
        source = (ROOT_DIR / "src" / "pipeline.py").read_text(encoding="utf-8")
        for forbidden in (
            "from src.features.screen_relations",
            "from src.models.train_lightgbm_relational",
            "from src.data.build_model_dataset",
        ):
            self.assertNotIn(forbidden, source)


class CompletenessTests(unittest.TestCase):
    def test_a_stage_with_all_outputs_present_is_complete(self):
        stage = make_stage("s", outputs=[ROOT_DIR / "README.md"])
        self.assertTrue(stage.is_complete())
        self.assertEqual(stage.missing_outputs(), [])

    def test_a_stage_with_one_missing_output_is_incomplete(self):
        stage = make_stage("s", outputs=[ROOT_DIR / "README.md", ROOT_DIR / "absent.xyz"])
        self.assertFalse(stage.is_complete())
        self.assertEqual(len(stage.missing_outputs()), 1)

    def test_a_stage_with_no_outputs_is_never_complete(self):
        self.assertFalse(make_stage("s").is_complete())


class PrerequisiteTests(unittest.TestCase):
    def registry(self):
        return {
            "first": make_stage("first", outputs=[ROOT_DIR / "absent_first.xyz"]),
            "second": make_stage(
                "second", outputs=[ROOT_DIR / "absent_second.xyz"], requires=["first"]
            ),
            "third": make_stage("third", outputs=[ROOT_DIR / "README.md"], requires=["second"]),
        }

    def test_unmet_prerequisites_are_reported_transitively(self):
        with patch.dict("src.pipeline.STAGES", self.registry(), clear=True):
            unmet = unmet_prerequisites("third")
            self.assertEqual([name for name, _ in unmet], ["first", "second"])

    def test_a_satisfied_prerequisite_is_not_reported(self):
        registry = self.registry()
        registry["first"] = make_stage("first", outputs=[ROOT_DIR / "README.md"])
        with patch.dict("src.pipeline.STAGES", registry, clear=True):
            self.assertEqual([n for n, _ in unmet_prerequisites("second")], [])

    def test_an_unknown_stage_is_rejected(self):
        with self.assertRaises(KeyError):
            unmet_prerequisites("no_such_stage")

    def test_the_message_names_the_stage_not_just_the_file(self):
        # The whole point: "you skipped a step", not "file not found".
        message = describe_unmet("third", [("second", ROOT_DIR / "reports" / "x.json")])
        self.assertIn("'second'", message)
        self.assertIn("has not been run", message)
        self.assertIn("--from second", message)

    def test_a_stage_with_unmet_prerequisites_refuses_to_run(self):
        with patch.dict("src.pipeline.STAGES", self.registry(), clear=True):
            with self.assertRaises(SystemExit) as caught:
                run_stage(STAGES["third"], dry_run=True)
            self.assertIn("cannot run yet", str(caught.exception))


class PlanTests(unittest.TestCase):
    def test_the_default_plan_is_every_stage_in_order(self):
        self.assertEqual(resolve_plan(None, None, False), list(STAGE_ORDER))

    def test_a_single_stage_plan_contains_only_it(self):
        self.assertEqual(resolve_plan("screening", None, False), ["screening"])

    def test_from_runs_the_tail_of_the_pipeline(self):
        plan = resolve_plan(None, "screening", False)
        self.assertEqual(plan[0], "screening")
        self.assertEqual(plan[-1], STAGE_ORDER[-1])
        self.assertNotIn("profile", plan)

    def test_unknown_stages_are_rejected_in_both_forms(self):
        with self.assertRaises(KeyError):
            resolve_plan("nope", None, False)
        with self.assertRaises(KeyError):
            resolve_plan(None, "nope", False)


class RunStageTests(unittest.TestCase):
    def complete_registry(self):
        return {"done": make_stage("done", outputs=[ROOT_DIR / "README.md"])}

    def test_a_complete_stage_is_skipped(self):
        with patch.dict("src.pipeline.STAGES", self.complete_registry(), clear=True):
            with patch("src.pipeline.subprocess.run") as runner:
                self.assertEqual(run_stage(STAGES["done"]), "skipped")
                runner.assert_not_called()

    def test_force_reruns_a_complete_stage(self):
        with patch.dict("src.pipeline.STAGES", self.complete_registry(), clear=True):
            with patch("src.pipeline.subprocess.run") as runner:
                runner.return_value.returncode = 0
                self.assertEqual(run_stage(STAGES["done"], force=True), "ran")
                runner.assert_called_once()

    def test_a_dry_run_executes_nothing(self):
        with patch.dict("src.pipeline.STAGES", self.complete_registry(), clear=True):
            with patch("src.pipeline.subprocess.run") as runner:
                self.assertEqual(run_stage(STAGES["done"], force=True, dry_run=True), "planned")
                runner.assert_not_called()

    def test_a_failing_stage_aborts_with_its_command(self):
        with patch.dict("src.pipeline.STAGES", self.complete_registry(), clear=True):
            with patch("src.pipeline.subprocess.run") as runner:
                runner.return_value.returncode = 3
                with self.assertRaises(SystemExit) as caught:
                    run_stage(STAGES["done"], force=True)
                self.assertIn("exit 3", str(caught.exception))

    def test_stages_run_from_the_repository_root(self):
        # One stage resolves its paths relative to the working directory, so a
        # wrong cwd would silently write to the wrong place.
        with patch.dict("src.pipeline.STAGES", self.complete_registry(), clear=True):
            with patch("src.pipeline.subprocess.run") as runner:
                runner.return_value.returncode = 0
                run_stage(STAGES["done"], force=True)
                self.assertEqual(runner.call_args.kwargs["cwd"], ROOT_DIR)

    def test_every_command_of_a_multi_command_stage_runs(self):
        stage = Stage(
            name="multi",
            description="two commands",
            commands=(("-m", "a"), ("-m", "b")),
            outputs=(ROOT_DIR / "README.md",),
        )
        with patch.dict("src.pipeline.STAGES", {"multi": stage}, clear=True):
            with patch("src.pipeline.subprocess.run") as runner:
                runner.return_value.returncode = 0
                run_stage(stage, force=True)
                self.assertEqual(runner.call_count, 2)


if __name__ == "__main__":
    unittest.main()
