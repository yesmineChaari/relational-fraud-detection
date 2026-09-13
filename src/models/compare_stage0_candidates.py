"""Stage 0 verdict (G1-v2 plan): consolidate Track A and Track B into one report.

Track A -- hub-tolerant candidates, gated on a training-free lift metric alone
(`fraud_neighbor_lift` in `reports/relational_audit/graph_diagnostics.csv`). A
candidate proceeds to Stage 2 eligibility only if its lift exceeds card1's own
-- the relation the frozen G1 already uses. A scalar-feature LightGBM null
result is never a valid veto for a Track A candidate: a flat count on a
hub-dominated entity (P_emaildomain is 46.0% one value) mixes unrelated
cardholders together, a different failure mode from a graph's bounded local
sample, and says nothing about whether a capped-fanout graph could still use
the relation.

Track B -- addr1's four scalar relational features added to B1-card1's
manifest, trained at the fixed 10,000-round budget
(`src.models.train_lightgbm_stage0_check`). An independent win requires the
95% paired-bootstrap CI on PR-AUC(candidate) - PR-AUC(b1_card1_fixed_budget)
to exclude zero on the positive side.

Both bars were fixed in the G1-v2 plan before either track was run. This
module never silently drops a failing result -- every candidate's verdict is
written here, whichever way it lands.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import ROOT_DIR
from src.graph.analyze_relations import CANDIDATES, PRIMARY_RELATION
from src.models.train_lightgbm_baseline import write_json
from src.models.train_lightgbm_relational import read_json
from src.models.train_lightgbm_stage0_check import (
    METADATA_PATH as STAGE0_ADDR1_METADATA_PATH,
)
from src.models.train_lightgbm_stage0_check import (
    RELATION as STAGE0_ADDR1_RELATION,
)

GRAPH_DIAGNOSTICS_PATH = ROOT_DIR / "reports" / "relational_audit" / "graph_diagnostics.csv"
REPORT_DIR = ROOT_DIR / "reports" / "stage0_screening"
VERDICT_PATH = REPORT_DIR / "stage0_verdict.json"

# Track A: candidates screened by fraud-neighbour lift alone, no LightGBM
# training. card1 is the reference lift every candidate must beat -- it is
# the relation the frozen G1 already uses.
TRACK_A_REFERENCE_RELATION = "card1"
TRACK_A_CANDIDATES = ["email_domain", "device_info", "device_fingerprint"]


def load_track_a_lifts() -> dict[str, float]:
    if not GRAPH_DIAGNOSTICS_PATH.exists():
        raise FileNotFoundError(
            f"Graph diagnostics not found: {GRAPH_DIAGNOSTICS_PATH}. "
            "Run: python -m src.graph.analyze_relations"
        )
    graph_df = pd.read_csv(GRAPH_DIAGNOSTICS_PATH).set_index("relation")
    required = {TRACK_A_REFERENCE_RELATION, *TRACK_A_CANDIDATES}
    missing = required - set(graph_df.index)
    if missing:
        raise ValueError(f"Graph diagnostics is missing relations: {sorted(missing)}.")
    return {relation: float(graph_df.loc[relation, "fraud_neighbor_lift"]) for relation in required}


def build_track_a_verdict() -> dict[str, Any]:
    lifts = load_track_a_lifts()
    reference_lift = lifts[TRACK_A_REFERENCE_RELATION]
    candidates: dict[str, Any] = {}
    for relation in TRACK_A_CANDIDATES:
        lift = lifts[relation]
        exceeds = bool(lift > reference_lift)
        candidates[relation] = {
            "group_columns": CANDIDATES.get(relation),
            "fraud_neighbor_lift": lift,
            "card1_fraud_neighbor_lift": reference_lift,
            "exceeds_card1_lift": exceeds,
            "decision": "qualifies_for_stage2_eligibility" if exceeds else "does_not_qualify",
        }
    return {
        "track": "A",
        "description": (
            "Hub-tolerant candidates, screened by fraud-neighbour lift alone -- "
            "no LightGBM training, and a scalar-feature null result is not a "
            "valid veto for these relations (see module docstring)."
        ),
        "pre_registered_bar": (
            f"fraud_neighbor_lift must exceed {TRACK_A_REFERENCE_RELATION}'s own "
            f"{reference_lift:.4f}x to qualify for Stage 2 eligibility."
        ),
        "reference_relation": TRACK_A_REFERENCE_RELATION,
        "reference_fraud_neighbor_lift": reference_lift,
        "candidates": candidates,
        "any_candidate_qualifies": any(c["exceeds_card1_lift"] for c in candidates.values()),
    }


def build_track_b_verdict() -> dict[str, Any]:
    if not STAGE0_ADDR1_METADATA_PATH.exists():
        return {
            "track": "B",
            "relation": STAGE0_ADDR1_RELATION,
            "decidable": False,
            "reason": (
                "Stage 0 addr1 check has not been run. "
                "Run: python -m src.models.train_lightgbm_stage0_check"
            ),
        }
    metadata = read_json(STAGE0_ADDR1_METADATA_PATH)
    comparison = metadata["comparison_to_b1_card1_fixed_budget"]
    meets_bar = bool(metadata["meets_pre_registered_bar"])
    return {
        "track": "B",
        "relation": STAGE0_ADDR1_RELATION,
        "decidable": True,
        "description": (
            "addr1's 4 scalar relational features added to B1-card1's manifest, "
            "trained at the fixed 10,000-round budget against the fixed-budget "
            "B1-card1 reference."
        ),
        "pre_registered_bar": metadata["pre_registered_bar"],
        "protocol": metadata["protocol"],
        "validation_pr_auc": metadata["validation_pr_auc"],
        "observed_delta_vs_b1_card1_fixed_budget": comparison["observed_delta"],
        "ci_95_vs_b1_card1_fixed_budget": [comparison["ci_lower_95"], comparison["ci_upper_95"]],
        "meets_pre_registered_bar": meets_bar,
        "verdict": "independent_win" if meets_bar else "negative_result_stands",
    }


def build_stage0_verdict() -> dict[str, Any]:
    return {
        "report_name": "Stage 0 pre-checks (G1-v2 plan)",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "primary_relation": PRIMARY_RELATION,
        "track_a": build_track_a_verdict(),
        "track_b": build_track_b_verdict(),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    verdict = build_stage0_verdict()
    write_json(VERDICT_PATH, verdict)

    print("Stage 0 verdict\n")
    print("Track A (fraud-neighbour lift, no LightGBM):")
    for relation, candidate in verdict["track_a"]["candidates"].items():
        print(
            f"  {relation}: lift {candidate['fraud_neighbor_lift']:.4f}x -> {candidate['decision']}"
        )

    track_b = verdict["track_b"]
    if track_b.get("decidable"):
        print(
            f"\nTrack B (addr1 vs B1-card1, fixed budget): delta "
            f"{track_b['observed_delta_vs_b1_card1_fixed_budget']:+.5f} "
            f"95% CI {track_b['ci_95_vs_b1_card1_fixed_budget']} -> {track_b['verdict']}"
        )
    else:
        print(f"\nTrack B: NOT YET DECIDABLE -- {track_b['reason']}")

    print(f"\nVerdict written: {VERDICT_PATH}")


if __name__ == "__main__":
    main()
