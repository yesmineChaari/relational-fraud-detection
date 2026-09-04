"""Build the card1 transaction graph artifact for the G1 relational stage.

Materialises the structural half of the temporal contract
(src/graph/temporal_contract.py): every transaction as a node, and every
transaction's membership in its card1 entity, over all 590,540 rows with
history continuous across train, validation and test -- the same
cross-split continuity the scalar relational features already rely on.

Storage representation
-----------------------
The full pairwise transaction-to-transaction edge set for card1 is a union
of cliques, one per entity. The largest entity is close to 14,500
transactions; a clique over it alone is on the order of 105 million edges.
Materialising that is not attempted. Instead this module stores an
entity-to-transaction bipartite structure: one row per transaction recording
which card1 entity it belongs to (STORAGE_RATIONALE below). This is O(N)
edges (590,540, one per transaction) rather than O(entity_size^2), and it is
exactly what the sampler (FRM-9) needs: for any target, restrict to its
entity's rows and apply the strictly-before rule at sample time. The
temporal admissibility itself is never materialised here, by design -- see
temporal_contract.EDGE_TEMPORAL_ADMISSIBILITY.

Reference implementation note
------------------------------
src/graph/analyze_relations.py builds temporal edges for diagnostic purposes
with a k-previous-neighbour cap. That module answers a different question
(bounded-degree diagnostics for the Stage A audit) and is used here only to
reconcile figures, never as the graph this module persists.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow

from src.graph.temporal_contract import (
    EDGE_STRUCTURAL_STORAGE,
    EDGE_TEMPORAL_ADMISSIBILITY,
    FORBIDDEN_NODE_FEATURE_COLUMNS,
    GROUP_COLUMNS,
    RELATION_NAME,
)
from src.models.train_lightgbm_baseline import (
    CATEGORY_MAPPINGS_PATH,
    EXPECTED_ROWS,
    EXPECTED_SPLIT_COUNTS,
    METADATA_PATH as B0_METADATA_PATH,
    MODEL_DATASET_PATH,
    apply_category_mapping,
    assert_supported_model_dtypes,
    repository_relative,
    validate_dataset_against_manifest,
    validate_split_counts,
    write_json,
)


ROOT_DIR = Path(__file__).resolve().parents[2]
NODES_PATH = ROOT_DIR / "data" / "processed" / "graph_card1_nodes.parquet"
ENTITY_EDGES_PATH = ROOT_DIR / "data" / "processed" / "graph_card1_entity_edges.parquet"
REPORT_DIR = ROOT_DIR / "reports" / "graph"
METADATA_PATH = REPORT_DIR / "card1_transaction_graph_metadata.json"

AUDIT_ENTITY_DIAGNOSTICS_PATH = (
    ROOT_DIR / "reports" / "relational_audit" / "entity_diagnostics.csv"
)
AUDIT_GRAPH_DIAGNOSTICS_PATH = (
    ROOT_DIR / "reports" / "relational_audit" / "graph_diagnostics.csv"
)

STORAGE_RATIONALE = (
    "A full pairwise clique per card1 entity is O(entity_size^2); the "
    "largest entity is close to 14,500 transactions, which alone is on the "
    "order of 105 million edges. A fixed-lag temporal chain (as in "
    "analyze_relations.py's diagnostic audit) is bounded in size but "
    "discards exactly the information the sampler needs to make a correct, "
    "target-anchored admissibility decision for an arbitrary neighbour "
    "count. An entity-to-transaction bipartite structure is O(N) total "
    "(one row per transaction), preserves every same-entity transaction the "
    "sampler could legitimately need, and defers the actual temporal "
    "admissibility decision to sample time, exactly as "
    "temporal_contract.py specifies."
)


def file_sha256(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return payload


def load_frozen_b0_manifest() -> tuple[list[str], list[str], list[str]]:
    metadata = read_json(B0_METADATA_PATH)
    feature_columns = metadata.get("feature_columns")
    categorical_columns = metadata.get("categorical_feature_columns")
    numeric_columns = metadata.get("numeric_feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise ValueError("Frozen B0 feature manifest is missing or invalid.")
    if not isinstance(categorical_columns, list) or not isinstance(numeric_columns, list):
        raise ValueError("Frozen B0 dtype manifests are missing or invalid.")
    if set(categorical_columns) | set(numeric_columns) != set(feature_columns):
        raise ValueError("Frozen B0 dtype manifests do not cover its predictors.")
    return feature_columns, categorical_columns, numeric_columns


def load_frozen_category_mappings(
    categorical_columns: list[str],
) -> dict[str, dict[str, int]]:
    payload = read_json(CATEGORY_MAPPINGS_PATH)
    if payload.get("fit_split") != "train":
        raise ValueError("Canonical categorical mappings were not fitted on train.")
    mappings = payload.get("columns")
    if not isinstance(mappings, dict) or list(mappings) != categorical_columns:
        raise ValueError("Canonical mapping columns differ from the frozen categorical manifest.")
    return mappings


def load_all_partitions(feature_columns: list[str]) -> pd.DataFrame:
    """Load every transaction (train + validation + test), never isFraud."""
    read_columns = sorted({"TransactionID", "split", *feature_columns})
    if "isFraud" in read_columns:
        raise AssertionError("isFraud must never be read by the graph construction module.")

    df = pd.read_parquet(MODEL_DATASET_PATH, columns=read_columns)
    if len(df) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} rows; got {len(df):,}.")

    validate_split_counts(df, "model_dataset.parquet (graph construction)")
    validate_dataset_against_manifest(df)

    for column in ("TransactionDT", RELATION_NAME):
        if df[column].isna().any():
            raise ValueError(f"{column} contains missing values; card1 must have 100% coverage.")
    return df


def assign_node_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Stable node ids: position in ascending TransactionID order."""
    ordered = df.sort_values("TransactionID", kind="mergesort").reset_index(drop=True)
    ordered.insert(0, "node_id", np.arange(len(ordered), dtype=np.int64))
    return ordered


def build_node_feature_matrix(
    nodes_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    mappings: dict[str, dict[str, int]],
) -> pd.DataFrame:
    FORBIDDEN_NODE_FEATURE_COLUMNS_CHECK = FORBIDDEN_NODE_FEATURE_COLUMNS & set(feature_columns)
    if FORBIDDEN_NODE_FEATURE_COLUMNS_CHECK:
        raise AssertionError(
            f"Forbidden columns present in B0 feature manifest: "
            f"{sorted(FORBIDDEN_NODE_FEATURE_COLUMNS_CHECK)}."
        )

    features = nodes_df[feature_columns].copy()
    for column in categorical_columns:
        features[column] = apply_category_mapping(features[column], mappings[column])
    assert_supported_model_dtypes(features, "graph node features")

    if "isFraud" in features.columns:
        raise AssertionError("isFraud must never appear in the node feature matrix.")
    return features


def build_entity_edges(nodes_df: pd.DataFrame) -> pd.DataFrame:
    """Materialise entity membership: one row per transaction (bipartite)."""
    entity_values = sorted(nodes_df[RELATION_NAME].unique().tolist())
    entity_id_map = {value: index for index, value in enumerate(entity_values)}

    edges = nodes_df[["node_id", "TransactionID", "TransactionDT", "split", RELATION_NAME]].copy()
    edges["entity_id"] = edges[RELATION_NAME].map(entity_id_map).astype(np.int64)
    edges = edges.sort_values(
        ["entity_id", "TransactionDT", "node_id"], kind="mergesort"
    ).reset_index(drop=True)
    return edges[["entity_id", RELATION_NAME, "node_id", "TransactionID", "TransactionDT", "split"]]


def compute_entity_diagnostics(edges_df: pd.DataFrame) -> dict[str, Any]:
    total = len(edges_df)
    sizes = edges_df.groupby("entity_id", sort=False).size()
    entity_count = int(len(sizes))
    singleton_count = int((sizes == 1).sum())
    largest_component_size = int(sizes.max())
    return {
        "row_count": int(total),
        "entity_count": entity_count,
        "connected_components": entity_count,
        "singleton_entity_count": singleton_count,
        "singleton_entity_pct": singleton_count / entity_count * 100.0,
        "isolated_nodes": singleton_count,
        "isolated_pct": singleton_count / total * 100.0,
        "largest_component_size": largest_component_size,
        "largest_component_pct": largest_component_size / total * 100.0,
        "group_size_mean": float(sizes.mean()),
        "group_size_median": float(sizes.median()),
        "group_size_p90": float(sizes.quantile(0.90)),
        "group_size_p95": float(sizes.quantile(0.95)),
        "group_size_p99": float(sizes.quantile(0.99)),
        "group_size_max": int(sizes.max()),
    }


def reconcile_with_relational_audit(edges_df: pd.DataFrame) -> dict[str, Any]:
    """Cross-check the train-only slice of this graph against Stage A's audit.

    The relational audit (src/graph/analyze_relations.py) computed its card1
    diagnostics over exactly the frozen 413,378-row train partition, loaded
    independently from the raw CSVs. Restricting this graph's edges to
    split == "train" reproduces the identical underlying rows, so entity
    structure -- which does not depend on how the transaction-to-transaction
    edges within an entity are represented -- must match exactly.
    """
    train_edges = edges_df.loc[edges_df["split"] == "train"]
    train_diagnostics = compute_entity_diagnostics(train_edges)

    entity_audit = pd.read_csv(AUDIT_ENTITY_DIAGNOSTICS_PATH).set_index("relation").loc[RELATION_NAME]
    graph_audit = pd.read_csv(AUDIT_GRAPH_DIAGNOSTICS_PATH).set_index("relation").loc[RELATION_NAME]

    checks = {
        "entity_count": (train_diagnostics["entity_count"], int(entity_audit["entity_count"])),
        "singleton_entity_pct": (
            train_diagnostics["singleton_entity_pct"],
            float(entity_audit["singleton_entity_pct"]),
        ),
        "group_size_median": (
            train_diagnostics["group_size_median"],
            float(entity_audit["group_size_median"]),
        ),
        "group_size_max": (train_diagnostics["group_size_max"], int(entity_audit["group_size_max"])),
        "connected_components": (
            train_diagnostics["connected_components"],
            int(graph_audit["connected_components"]),
        ),
        "largest_component_size": (
            train_diagnostics["largest_component_size"],
            int(graph_audit["largest_component_size"]),
        ),
        "isolated_nodes": (train_diagnostics["isolated_nodes"], int(graph_audit["isolated_nodes"])),
    }
    mismatched = {
        name: (actual, expected)
        for name, (actual, expected) in checks.items()
        if not np.isclose(actual, expected, rtol=1e-9, atol=1e-6)
    }
    if mismatched:
        raise AssertionError(
            f"Train-only card1 entity structure does not reconcile with the "
            f"relational audit reports: {mismatched}."
        )
    return {"reconciled_against_relational_audit": True, "train_only_diagnostics": train_diagnostics}


def build_metadata(
    *,
    nodes_df: pd.DataFrame,
    edges_df: pd.DataFrame,
    feature_columns: list[str],
    categorical_columns: list[str],
    numeric_columns: list[str],
    full_diagnostics: dict[str, Any],
    reconciliation: dict[str, Any],
) -> dict[str, Any]:
    split_row_counts = {
        str(name): int(count) for name, count in nodes_df["split"].astype("string").value_counts().items()
    }
    return {
        "relation_name": RELATION_NAME,
        "group_columns": GROUP_COLUMNS,
        "storage_representation": EDGE_STRUCTURAL_STORAGE,
        "temporal_admissibility": EDGE_TEMPORAL_ADMISSIBILITY,
        "storage_rationale": STORAGE_RATIONALE,
        "temporal_contract_path": "src/graph/temporal_contract.py",
        "input_model_dataset_path": repository_relative(MODEL_DATASET_PATH),
        "input_model_dataset_sha256": file_sha256(MODEL_DATASET_PATH),
        "input_b0_metadata_path": repository_relative(B0_METADATA_PATH),
        "input_b0_metadata_sha256": file_sha256(B0_METADATA_PATH),
        "input_category_mappings_path": repository_relative(CATEGORY_MAPPINGS_PATH),
        "input_category_mappings_sha256": file_sha256(CATEGORY_MAPPINGS_PATH),
        "row_count": int(len(nodes_df)),
        "split_row_counts": split_row_counts,
        "history_across_splits": (
            "History is continuous across train, validation, and test; entity "
            "membership is never reset at split boundaries."
        ),
        "frozen_split_assignment_consumed": True,
        "node_count": int(len(nodes_df)),
        "edge_row_count": int(len(edges_df)),
        "feature_column_count": len(feature_columns),
        "categorical_feature_count": len(categorical_columns),
        "numeric_feature_count": len(numeric_columns),
        "feature_columns": feature_columns,
        "categorical_feature_columns": categorical_columns,
        "target_labels_used": False,
        "isFraud_present_in_node_features": False,
        "full_dataset_entity_diagnostics": full_diagnostics,
        "reconciled_against_relational_audit": reconciliation["reconciled_against_relational_audit"],
        "train_only_entity_diagnostics": reconciliation["train_only_diagnostics"],
        "nodes_path": repository_relative(NODES_PATH),
        "entity_edges_path": repository_relative(ENTITY_EDGES_PATH),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "pyarrow": pyarrow.__version__,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def build_and_save_transaction_graph() -> None:
    print(f"[{RELATION_NAME}] Loading frozen B0 feature manifest...")
    feature_columns, categorical_columns, numeric_columns = load_frozen_b0_manifest()
    mappings = load_frozen_category_mappings(categorical_columns)

    print(f"[{RELATION_NAME}] Loading all partitions from: {MODEL_DATASET_PATH}")
    raw_df = load_all_partitions(feature_columns)

    print(f"[{RELATION_NAME}] Assigning stable node ids...")
    raw_df = assign_node_ids(raw_df)

    print(f"[{RELATION_NAME}] Building node feature matrix ({len(feature_columns)} predictors)...")
    features = build_node_feature_matrix(raw_df, feature_columns, categorical_columns, mappings)
    nodes_df = pd.concat(
        [raw_df[["node_id", "TransactionID", "split"]].reset_index(drop=True), features],
        axis=1,
    )
    if len(nodes_df) != EXPECTED_ROWS:
        raise AssertionError("Node table row count changed during construction.")

    print(f"[{RELATION_NAME}] Materialising entity-to-transaction edges...")
    edges_df = build_entity_edges(raw_df)
    if len(edges_df) != EXPECTED_ROWS:
        raise AssertionError("Entity edge table row count does not match transaction count.")

    full_diagnostics = compute_entity_diagnostics(edges_df)
    print(f"[{RELATION_NAME}] Reconciling train-only structure against the relational audit...")
    reconciliation = reconcile_with_relational_audit(edges_df)

    NODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    nodes_df.to_parquet(NODES_PATH, index=False, engine="pyarrow", compression="snappy")
    edges_df.to_parquet(ENTITY_EDGES_PATH, index=False, engine="pyarrow", compression="snappy")

    metadata = build_metadata(
        nodes_df=nodes_df,
        edges_df=edges_df,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        full_diagnostics=full_diagnostics,
        reconciliation=reconciliation,
    )
    write_json(METADATA_PATH, metadata)

    saved_nodes = pd.read_parquet(NODES_PATH)
    if len(saved_nodes) != EXPECTED_ROWS:
        raise AssertionError("Saved node table row count changed.")
    if "isFraud" in saved_nodes.columns:
        raise AssertionError("isFraud leaked into the saved node table.")

    print(f"[{RELATION_NAME}] Nodes: {len(nodes_df):,}")
    print(f"[{RELATION_NAME}] Entities: {full_diagnostics['entity_count']:,}")
    print(f"[{RELATION_NAME}] Largest entity share: {full_diagnostics['largest_component_pct']:.4f}%")
    print(f"[{RELATION_NAME}] Reconciled against relational audit: YES")
    print(f"[{RELATION_NAME}] Nodes saved: {NODES_PATH}")
    print(f"[{RELATION_NAME}] Entity edges saved: {ENTITY_EDGES_PATH}")
    print(f"[{RELATION_NAME}] Metadata saved: {METADATA_PATH}")
    print(f"[{RELATION_NAME}] Target labels used: NO")


def main() -> None:
    build_and_save_transaction_graph()


if __name__ == "__main__":
    main()
