from __future__ import annotations

import re
import unittest

from src.config.paths import ROOT_DIR
from src.features.build_relational_features import RELATION_REGISTRY
from src.graph.analyze_relations import CANDIDATES
from src.models.build_experiment_ledger import FAMILIES, MODELS_DIR
from src.models.train_lightgbm_stage0_check import (
    METADATA_PATH,
    PURPOSE,
    RELATION,
    stage0_paths,
)


class Stage0PathTests(unittest.TestCase):
    def test_the_published_addr1_paths_are_unchanged(self) -> None:
        paths = stage0_paths("addr1")
        self.assertEqual(RELATION, "addr1")
        self.assertEqual(METADATA_PATH, paths["metadata"])
        self.assertEqual(
            paths["metadata"], ROOT_DIR / "reports" / "stage0_screening" / "addr1" / "metadata.json"
        )
        self.assertEqual(
            paths["model"], MODELS_DIR / "stage0_screening" / "lightgbm_stage0_addr1.txt"
        )

    def test_relations_never_share_artifacts(self) -> None:
        a, b = stage0_paths("addr1"), stage0_paths("device_fingerprint")
        for key in a:
            self.assertNotEqual(a[key], b[key], key)

    def test_a_relation_without_a_recorded_purpose_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            stage0_paths("card1")

    def test_every_check_has_a_purpose_and_a_feature_definition(self) -> None:
        for relation in PURPOSE:
            self.assertTrue(PURPOSE[relation].strip())
            self.assertIn(relation, RELATION_REGISTRY)

    def test_the_feature_builder_and_the_audit_define_device_fingerprint_identically(self) -> None:
        self.assertEqual(RELATION_REGISTRY["device_fingerprint"], CANDIDATES["device_fingerprint"])

    def test_every_check_model_matches_exactly_one_ledger_family(self) -> None:
        for relation in PURPOSE:
            relative = stage0_paths(relation)["model"].relative_to(MODELS_DIR).as_posix()
            matched = [f["family"] for f in FAMILIES if re.match(f["pattern"], relative)]
            self.assertEqual(matched, ["stage0_screening_check"], relative)


if __name__ == "__main__":
    unittest.main()
