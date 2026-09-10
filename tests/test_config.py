from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config.paths import (
    CONFIGS_DIR,
    MODEL_DATASET_PATH,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    RAW_IDENTITY_PATH,
    RAW_TRANSACTION_PATH,
    REPORTS_DIR,
    ROOT_DIR,
    SCREENING_CONFIG_PATH,
    SPLIT_ASSIGNMENT_PATH,
)
from src.features.screen_relations import (
    COVERAGE_MIN_PCT,
    MAX_PREFERRED_RELATIONS,
    MEDIAN_REPEAT_GAP_MUST_BE_FINITE,
    REQUIRE_SIGNAL_ABOVE_MISSINGNESS,
    SCREENING_CONFIG_SCHEMA,
    SIGNAL_MODERATE_PR_AUC_LIFT,
    load_screening_config,
)


def write_config(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def sound_payload() -> dict:
    return {section: {key: 1 for key in keys} for section, keys in SCREENING_CONFIG_SCHEMA.items()}


class RootPathTests(unittest.TestCase):
    def test_root_is_the_repository_root(self):
        self.assertTrue((ROOT_DIR / "README.md").is_file())
        self.assertTrue((ROOT_DIR / "src").is_dir())
        self.assertTrue((ROOT_DIR / "pyproject.toml").is_file())

    def test_every_anchor_sits_under_the_root(self):
        for anchor in (
            RAW_DATA_DIR,
            PROCESSED_DATA_DIR,
            REPORTS_DIR,
            MODELS_DIR,
            CONFIGS_DIR,
        ):
            self.assertEqual(anchor.parents[len(anchor.parts) - len(ROOT_DIR.parts) - 1], ROOT_DIR)

    def test_dataset_locations_are_under_processed_data(self):
        self.assertEqual(MODEL_DATASET_PATH.parent, PROCESSED_DATA_DIR)
        self.assertEqual(SPLIT_ASSIGNMENT_PATH.parent, PROCESSED_DATA_DIR)

    def test_raw_inputs_are_the_two_labelled_files(self):
        # The competition's unlabelled holdout is deliberately absent here.
        self.assertEqual(RAW_TRANSACTION_PATH.name, "train_transaction.csv")
        self.assertEqual(RAW_IDENTITY_PATH.name, "train_identity.csv")
        self.assertEqual(RAW_TRANSACTION_PATH.parent, RAW_DATA_DIR)

    def test_no_module_derives_its_own_root_any_more(self):
        # The duplication this module exists to remove: if it returns, the
        # single authoritative source has quietly stopped being authoritative.
        target = "ROOT_DIR = Path(__file__).resolve().parents[2]"
        offenders = [
            p.relative_to(ROOT_DIR).as_posix()
            for p in (ROOT_DIR / "src").rglob("*.py")
            if target in p.read_text(encoding="utf-8") and p.name != "paths.py"
        ]
        self.assertEqual(offenders, [], f"Modules re-deriving the root: {offenders}")


class ScreeningConfigTests(unittest.TestCase):
    def test_the_committed_config_loads(self):
        config = load_screening_config()
        self.assertEqual(set(config), set(SCREENING_CONFIG_SCHEMA))

    def test_the_schema_covers_twelve_thresholds(self):
        total = sum(len(keys) for keys in SCREENING_CONFIG_SCHEMA.values())
        self.assertEqual(total, 12)

    def test_module_constants_are_bound_from_the_config(self):
        config = load_screening_config()
        self.assertEqual(COVERAGE_MIN_PCT, config["structural"]["coverage_min_pct"])
        self.assertEqual(
            MEDIAN_REPEAT_GAP_MUST_BE_FINITE,
            config["structural"]["median_repeat_gap_must_be_finite"],
        )
        self.assertEqual(MAX_PREFERRED_RELATIONS, config["promotion"]["max_preferred_relations"])
        self.assertEqual(
            SIGNAL_MODERATE_PR_AUC_LIFT, config["signal"]["signal_moderate_pr_auc_lift"]
        )
        self.assertEqual(
            REQUIRE_SIGNAL_ABOVE_MISSINGNESS,
            config["signal"]["require_signal_above_missingness"],
        )

    def test_the_committed_values_are_the_published_policy(self):
        # The thresholds every published screening decision was made under.
        config = load_screening_config()
        self.assertEqual(config["structural"]["coverage_min_pct"], 50.0)
        self.assertEqual(config["promotion"]["preferred_coverage_min_pct"], 80.0)
        self.assertEqual(config["promotion"]["max_preferred_relations"], 2)
        self.assertEqual(config["signal"]["signal_strong_pr_auc_lift"], 2.0)

    def test_underscore_commentary_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screening.json"
            payload = sound_payload()
            payload["_comment"] = "top level note"
            payload["structural"]["_comment"] = ["section note"]
            write_config(path, payload)
            config = load_screening_config(path)
            self.assertNotIn("_comment", config["structural"])

    def test_a_missing_section_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screening.json"
            payload = sound_payload()
            del payload["signal"]
            write_config(path, payload)
            with self.assertRaises(KeyError):
                load_screening_config(path)

    def test_a_missing_threshold_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screening.json"
            payload = sound_payload()
            del payload["structural"]["coverage_min_pct"]
            write_config(path, payload)
            with self.assertRaises(KeyError) as caught:
                load_screening_config(path)
            self.assertIn("coverage_min_pct", str(caught.exception))

    def test_an_unknown_threshold_is_rejected_rather_than_ignored(self):
        # A typo must not silently leave the real threshold at its old value.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "screening.json"
            payload = sound_payload()
            payload["structural"]["coverage_min_pctt"] = 10.0
            write_config(path, payload)
            with self.assertRaises(KeyError) as caught:
                load_screening_config(path)
            self.assertIn("unknown", str(caught.exception).lower())

    def test_a_missing_file_is_rejected_with_a_clear_message(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError) as caught:
                load_screening_config(Path(tmp) / "absent.json")
            self.assertIn("not optional", str(caught.exception))

    def test_the_config_lives_where_the_paths_module_says(self):
        self.assertEqual(SCREENING_CONFIG_PATH.parent, CONFIGS_DIR)
        self.assertTrue(SCREENING_CONFIG_PATH.is_file())


if __name__ == "__main__":
    unittest.main()
