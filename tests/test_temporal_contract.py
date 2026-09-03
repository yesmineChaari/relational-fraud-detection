from __future__ import annotations

import unittest

from src.graph.temporal_contract import (
    FORBIDDEN_NODE_FEATURE_COLUMNS,
    RELATION_NAME,
    TRAINING_REGIME,
    assert_no_forbidden_node_features,
    is_temporally_admissible,
    validate_sampled_neighborhood,
)


class HopAdmissibilityTests(unittest.TestCase):
    def test_strictly_earlier_node_is_admissible(self) -> None:
        self.assertTrue(is_temporally_admissible(100, 200))

    def test_equal_timestamp_is_not_admissible(self) -> None:
        self.assertFalse(is_temporally_admissible(200, 200))

    def test_later_node_is_not_admissible(self) -> None:
        self.assertFalse(is_temporally_admissible(300, 200))

    def test_admissibility_is_evaluated_against_the_same_target_at_every_hop(self) -> None:
        target_dt = 1_000
        hop1_dt = 900
        # A hop-2 node between the hop-1 node and the target is admissible
        # under target-anchoring even though it is *after* the hop-1 node.
        hop2_dt = 950
        self.assertTrue(is_temporally_admissible(hop1_dt, target_dt))
        self.assertTrue(is_temporally_admissible(hop2_dt, target_dt))


class SampledNeighborhoodValidationTests(unittest.TestCase):
    def test_all_strictly_prior_neighbors_pass(self) -> None:
        validate_sampled_neighborhood(target_dt=1_000, neighbor_dts=[100, 500, 999])

    def test_empty_neighborhood_passes(self) -> None:
        validate_sampled_neighborhood(target_dt=1_000, neighbor_dts=[])

    def test_a_tied_neighbor_raises(self) -> None:
        with self.assertRaises(AssertionError):
            validate_sampled_neighborhood(target_dt=1_000, neighbor_dts=[999, 1_000])

    def test_a_future_neighbor_raises(self) -> None:
        with self.assertRaises(AssertionError):
            validate_sampled_neighborhood(target_dt=1_000, neighbor_dts=[500, 1_500])


class NodeFeatureContractTests(unittest.TestCase):
    def test_clean_feature_list_passes(self) -> None:
        assert_no_forbidden_node_features(["TransactionDT", "TransactionAmt", "card1"])

    def test_isFraud_is_rejected(self) -> None:
        with self.assertRaises(AssertionError):
            assert_no_forbidden_node_features(["TransactionDT", "isFraud"])

    def test_every_forbidden_column_is_individually_rejected(self) -> None:
        for column in FORBIDDEN_NODE_FEATURE_COLUMNS:
            with self.assertRaises(AssertionError):
                assert_no_forbidden_node_features(["TransactionDT", column])


class ContractConstantsTests(unittest.TestCase):
    def test_relation_is_card1(self) -> None:
        self.assertEqual(RELATION_NAME, "card1")

    def test_training_regime_is_inductive(self) -> None:
        self.assertEqual(TRAINING_REGIME, "inductive")


if __name__ == "__main__":
    unittest.main()
