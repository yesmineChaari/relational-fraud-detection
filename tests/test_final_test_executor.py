"""The final test executor, exercised on synthetic data only.

No test in this module reads the real test partition: every data loader is
patched, and the end-to-end cases train a tiny booster on generated rows.
"""

from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

from src.features.build_relational_features import _feature_names
from src.models import evaluate_final_test as executor
from src.models import final_test_protocol as protocol

CARD1_FEATURES = _feature_names("card1")
B0_FEATURES = ["amount", "channel"]
MAPPINGS = {"channel": {"__MISSING__": 0, "__UNKNOWN__": 1, "app": 2, "web": 3}}
B0_METADATA = {"feature_columns": B0_FEATURES, "categorical_feature_columns": ["channel"]}


class OutcomeRuleTests(unittest.TestCase):
    def test_interval_above_zero_replicates(self):
        self.assertEqual(executor.classify_outcome(0.001, 0.01), protocol.OUTCOME_REPLICATES)

    def test_interval_below_zero_reverses(self):
        self.assertEqual(executor.classify_outcome(-0.01, -0.001), protocol.OUTCOME_REVERSES)

    def test_interval_containing_zero_is_not_confirmed(self):
        self.assertEqual(executor.classify_outcome(-0.002, 0.004), protocol.OUTCOME_NOT_CONFIRMED)

    def test_an_interval_touching_zero_is_not_confirmed(self):
        # "Entirely above zero" is strict, so a lower bound of exactly zero fails it.
        self.assertEqual(executor.classify_outcome(0.0, 0.004), protocol.OUTCOME_NOT_CONFIRMED)


class ProtocolScopeTests(unittest.TestCase):
    def test_feature_sets_cover_exactly_the_protocol_models(self):
        self.assertEqual(set(executor.all_models()), set(executor.FEATURE_SET))

    def test_the_graph_model_is_not_scored(self):
        self.assertNotIn("g1_card1", executor.all_models())
        self.assertIn("g1_card1", protocol.NOT_SCORED)

    def test_only_b0_and_b1_card1_are_primary(self):
        self.assertEqual(set(protocol.PRIMARY_MODELS), {"b0", "b1_card1"})
        self.assertEqual(protocol.PRIMARY_COMPARISON["candidate"], "b1_card1")
        self.assertEqual(protocol.PRIMARY_COMPARISON["reference"], "b0")

    def test_named_artifacts_exist_when_models_are_present(self):
        models = executor.all_models()
        if not (executor.ROOT_DIR / models["b0"]["model"]).exists():
            self.skipTest("Model artifacts are not present in this checkout.")
        for spec in models.values():
            for key in ("model", "validation_metrics", "validation_predictions"):
                self.assertTrue((executor.ROOT_DIR / spec[key]).exists(), spec[key])

    def test_train_is_never_loaded(self):
        with self.assertRaises(ValueError):
            executor.load_partition("train")


def fake_git(tracked: bool = True, modified: bool = False, commit: str = "abc123"):
    def run(args, **_kwargs):
        if args[1] == "ls-files":
            return subprocess.CompletedProcess(args, 0 if tracked else 1, "", "")
        if args[1] == "status":
            return subprocess.CompletedProcess(args, 0, " M file\n" if modified else "", "")
        return subprocess.CompletedProcess(args, 0, f"{commit}\n", "")

    return run


class ProtocolCommitGuardTests(unittest.TestCase):
    def test_an_untracked_protocol_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "not committed"):
            executor.protocol_commit_status(run=fake_git(tracked=False))

    def test_a_modified_protocol_is_refused(self):
        with self.assertRaisesRegex(RuntimeError, "uncommitted"):
            executor.protocol_commit_status(run=fake_git(modified=True))

    def test_a_clean_protocol_returns_its_commit(self):
        self.assertEqual(executor.protocol_commit_status(run=fake_git()), "abc123")


class OneShotClaimTests(unittest.TestCase):
    def test_an_existing_output_directory_refuses_the_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileExistsError):
                executor.assert_read_not_claimed(Path(tmp))

    def test_claiming_writes_a_marker_and_cannot_be_repeated(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "final_test"
            marker = executor.claim_read(output, "abc123")
            self.assertEqual(json.loads(marker.read_text())["protocol_commit"], "abc123")
            with self.assertRaises(FileExistsError):
                executor.claim_read(output, "abc123")


class ReproductionGuardTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "validation_predictions.parquet"
        self.ids = np.arange(100, 110)
        self.scores = np.linspace(0.05, 0.95, 10)
        pd.DataFrame({"TransactionID": self.ids[::-1], "prediction": self.scores[::-1]}).to_parquet(
            self.path, index=False
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_exact_scores_pass_regardless_of_row_order(self):
        result = executor.assert_reproduces_validation("m", self.ids, self.scores, self.path)
        self.assertTrue(result["identical"])

    def test_any_difference_at_all_aborts(self):
        perturbed = self.scores.copy()
        perturbed[3] += 1e-15
        with self.assertRaisesRegex(AssertionError, "does not reproduce"):
            executor.assert_reproduces_validation("m", self.ids, perturbed, self.path)


class MetricsAndGapTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.labels = (rng.random(2_000) < 0.1).astype(np.int8)
        self.scores = np.clip(self.labels * 0.3 + rng.random(2_000) * 0.7, 0, 1)

    def test_evaluate_reports_exactly_the_protocol_metric_set(self):
        metrics = executor.evaluate(self.labels, self.scores)
        self.assertTrue(metrics["test_evaluated"])
        self.assertEqual(metrics["evaluation_split"], "test")
        self.assertEqual(
            set(metrics["ranking_metrics"]), {"top_0.5_pct", "top_1_pct", "top_2_pct", "top_5_pct"}
        )

    def test_gap_is_test_minus_validation_for_every_metric(self):
        test = executor.evaluate(self.labels, self.scores)
        validation = json.loads(json.dumps(test))
        validation["pr_auc"] += 0.01
        rows = {
            row["metric"]: row for row in executor.validation_to_test_gap("m", validation, test)
        }
        self.assertEqual(len(rows), 2 + 2 * len(protocol.TOP_FRACTIONS))
        self.assertAlmostEqual(rows["pr_auc"]["test_minus_validation"], -0.01)
        self.assertAlmostEqual(rows["roc_auc"]["test_minus_validation"], 0.0)


def synthetic_partition(split: str, n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    labels = (rng.random(n) < 0.15).astype(np.int8)
    frame = pd.DataFrame(
        {
            "TransactionID": np.arange(n) + (0 if split == "validation" else 10_000),
            "TransactionDT": np.arange(n),
            "isFraud": labels,
            "split": split,
            "amount": rng.normal(size=n) + labels * 1.5,
            "channel": pd.Series(rng.choice(["app", "web", None], size=n), dtype="object"),
        }
    )
    for offset, column in enumerate(CARD1_FEATURES):
        frame[column] = rng.normal(size=n) + labels * (0.5 + offset * 0.1)
    return frame


def design(frame: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    return executor.build_design_matrix(frame, feature_set, B0_METADATA, MAPPINGS)


class EndToEndSyntheticTests(unittest.TestCase):
    """The real run() against generated data, with every loader patched."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.output = self.root / "reports" / "final_test"
        self.partitions = {
            "validation": synthetic_partition("validation", 1_500, seed=1),
            "test": synthetic_partition("test", 1_500, seed=2),
        }
        train = synthetic_partition("validation", 3_000, seed=3)
        self.models: dict[str, dict[str, str]] = {}
        for name, feature_set in executor.FEATURE_SET.items():
            model = LGBMClassifier(n_estimators=25, num_leaves=7, verbose=-1, random_state=0)
            model.fit(design(train, feature_set), train["isFraud"], categorical_feature=["channel"])
            model_path = self.root / f"{name}.txt"
            model.booster_.save_model(str(model_path))
            validation = self.partitions["validation"]
            persisted = validation[["TransactionID", "TransactionDT", "isFraud"]].copy()
            persisted["prediction"] = model.predict_proba(design(validation, feature_set))[:, 1]
            persisted.to_parquet(self.root / f"{name}_validation.parquet", index=False)
            (self.root / f"{name}_metrics.json").write_text(
                json.dumps(
                    executor.evaluate(
                        validation["isFraud"].to_numpy(), persisted["prediction"].to_numpy()
                    )
                )
            )
            self.models[name] = {
                "role": "reference" if name.startswith("b0") else "candidate",
                "model": f"{name}.txt",
                "validation_metrics": f"{name}_metrics.json",
                "validation_predictions": f"{name}_validation.parquet",
            }
        self.loaded: list[str] = []

    def tearDown(self):
        self.tmp.cleanup()

    def fake_load(self, split: str) -> pd.DataFrame:
        self.loaded.append(split)
        return self.partitions[split].drop(columns=CARD1_FEATURES)

    def fake_attach(self, partition: pd.DataFrame, split: str) -> pd.DataFrame:
        source = self.partitions[split].set_index("TransactionID")
        attached = partition.copy()
        for column in CARD1_FEATURES:
            attached[column] = source.loc[partition["TransactionID"], column].to_numpy()
        return attached

    def run_executor(self, execute: bool) -> dict:
        with (
            patch.object(executor, "protocol_commit_status", return_value="abc123"),
            patch.object(executor, "all_models", return_value=self.models),
            patch.object(executor, "load_frozen_b0_metadata", return_value=B0_METADATA),
            patch.object(executor, "load_frozen_category_mappings", return_value=(MAPPINGS, "sha")),
            patch.object(executor, "load_partition", side_effect=self.fake_load),
            patch.object(executor, "attach_card1_features", side_effect=self.fake_attach),
            patch.dict(protocol.PRIMARY_COMPARISON, {"n_resamples": 200}),
        ):
            return executor.run(execute=execute, output_dir=self.output, root=self.root)

    def test_a_dry_run_never_opens_the_test_partition(self):
        result = self.run_executor(execute=False)
        self.assertEqual(result["mode"], "dry_run")
        self.assertNotIn("test", self.loaded)
        self.assertFalse(self.output.exists())

    def test_a_validation_mismatch_aborts_before_test_is_opened(self):
        path = self.root / "b1_card1_validation.parquet"
        tampered = pd.read_parquet(path)
        tampered.loc[0, "prediction"] += 1e-9
        tampered.to_parquet(path, index=False)
        with self.assertRaisesRegex(AssertionError, "does not reproduce"):
            self.run_executor(execute=True)
        self.assertNotIn("test", self.loaded)
        self.assertFalse(self.output.exists())

    def test_execute_claims_the_read_first_and_writes_every_output(self):
        summary = self.run_executor(execute=True)
        self.assertIn(
            summary["outcome"],
            {
                protocol.OUTCOME_REPLICATES,
                protocol.OUTCOME_NOT_CONFIRMED,
                protocol.OUTCOME_REVERSES,
            },
        )
        self.assertTrue(summary["test_evaluated"])
        self.assertEqual(summary["deviations"], [])
        self.assertTrue((self.output / executor.CLAIM_MARKER).exists())
        self.assertTrue((self.output / executor.SUMMARY_NAME).exists())
        gap = pd.read_csv(self.output / executor.GAP_NAME)
        self.assertEqual(set(gap["model"]), set(self.models))
        for name in self.models:
            metrics = json.loads((self.output / name / "test_metrics.json").read_text())
            self.assertTrue(metrics["test_evaluated"])
            predictions = pd.read_parquet(self.output / name / "test_predictions.parquet")
            self.assertEqual(len(predictions), len(self.partitions["test"]))

    def test_a_second_execute_is_refused(self):
        self.run_executor(execute=True)
        with self.assertRaises(FileExistsError):
            self.run_executor(execute=True)

    def test_inputs_are_left_unchanged(self):
        before = {path: path.read_bytes() for path in self.root.iterdir() if path.is_file()}
        self.run_executor(execute=True)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content, path.name)


if __name__ == "__main__":
    unittest.main()
