from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
REPORT_DIR = ROOT_DIR / "reports" / "relational_audit"

TRANSACTION_FILE = RAW_DIR / "train_transaction.csv"
IDENTITY_FILE = RAW_DIR / "train_identity.csv"
SPLIT_FILE = ROOT_DIR / "data" / "processed" / "split_assignment.parquet"

K_PREVIOUS_NEIGHBORS = 3
DENSE_GROUP_SIZE = 5

CANDIDATES: dict[str, list[str]] = {
    "card1": ["card1"],
    "card1_card2": ["card1", "card2"],
    "card_core": ["card1", "card2", "card3", "card5"],
    "card_full": ["card1", "card2", "card3", "card4", "card5", "card6"],
    "card_core_addr1": ["card1", "card2", "card3", "card5", "addr1"],
    "device_info": ["DeviceInfo"],
    "device_fingerprint": ["DeviceInfo", "id_30", "id_31", "id_33"],
}


def load_data() -> pd.DataFrame:
    transaction_columns = [
        "TransactionID",
        "TransactionDT",
        "isFraud",
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "addr1",
    ]
    identity_columns = [
        "TransactionID",
        "DeviceInfo",
        "id_30",
        "id_31",
        "id_33",
    ]

    transactions = pd.read_csv(TRANSACTION_FILE, usecols=transaction_columns)
    split_df = pd.read_parquet(
        SPLIT_FILE,
        columns=["TransactionID", "split"],
    )
    train_ids = split_df.loc[split_df["split"] == "train", "TransactionID"]
    transactions = transactions[
        transactions["TransactionID"].isin(train_ids)
    ].copy()

    identity = pd.read_csv(IDENTITY_FILE, usecols=identity_columns)
    df = transactions.merge(
        identity,
        on="TransactionID",
        how="left",
        validate="one_to_one",
    )
    df["node_id"] = np.arange(len(df), dtype=np.int64)
    return df


def build_strict_proxy(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    available = df[columns].notna().all(axis=1)
    proxy = pd.Series(pd.NA, index=df.index, dtype="string")
    if not available.any():
        return proxy

    index = df.index[available]
    key = df.loc[index, columns[0]].astype("string")
    for column in columns[1:]:
        key = key.str.cat(df.loc[index, column].astype("string"), sep="|")
    proxy.loc[index] = key
    return proxy


def analyze_entity_structure(
    name: str,
    df: pd.DataFrame,
    proxy: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    total = len(df)
    covered = proxy.notna()
    covered_count = int(covered.sum())
    if not covered_count:
        return {
            "relation": name,
            "coverage_pct": 0.0,
            "entity_count": 0,
        }, pd.DataFrame()

    counts = proxy.loc[covered].value_counts()
    entity_count = len(counts)
    recurring = counts[counts >= 2]
    group_sizes = proxy.map(counts).fillna(0).astype(np.int64)
    repeated_mask = group_sizes >= 2
    dense_mask = group_sizes >= DENSE_GROUP_SIZE

    baseline_fraud_rate = df["isFraud"].mean()
    repeated_fraud_rate = (
        df.loc[repeated_mask, "isFraud"].mean() if repeated_mask.any() else np.nan
    )
    dense_fraud_rate = (
        df.loc[dense_mask, "isFraud"].mean() if dense_mask.any() else np.nan
    )

    temporal = df.loc[covered, ["TransactionDT", "isFraud"]].copy()
    temporal["entity"] = proxy.loc[covered].astype("string")
    entity_temporal = temporal.groupby("entity", sort=False).agg(
        group_size=("TransactionDT", "size"),
        first_time=("TransactionDT", "min"),
        last_time=("TransactionDT", "max"),
        fraud_rate=("isFraud", "mean"),
    )
    entity_temporal["lifespan_days"] = (
        entity_temporal["last_time"] - entity_temporal["first_time"]
    ) / 86400.0
    recurring_temporal = entity_temporal[entity_temporal["group_size"] >= 2]

    gaps = (
        temporal.sort_values(["entity", "TransactionDT"])
        .groupby("entity", sort=False)["TransactionDT"]
        .diff()
        .dropna()
        / 3600.0
    )

    result = {
        "relation": name,
        "coverage_pct": covered_count / total * 100,
        "covered_transactions": covered_count,
        "entity_count": entity_count,
        "singleton_entity_pct": (counts == 1).sum() / entity_count * 100,
        "recurring_entity_pct": len(recurring) / entity_count * 100,
        "repeated_transaction_pct_all": recurring.sum() / total * 100,
        "repeated_transaction_pct_covered": recurring.sum() / covered_count * 100,
        "group_size_mean": counts.mean(),
        "group_size_median": counts.median(),
        "group_size_p90": counts.quantile(0.90),
        "group_size_p95": counts.quantile(0.95),
        "group_size_p99": counts.quantile(0.99),
        "group_size_max": counts.max(),
        "largest_entity_share_pct": counts.max() / covered_count * 100,
        "median_entity_lifespan_days": recurring_temporal["lifespan_days"].median(),
        "p95_entity_lifespan_days": recurring_temporal["lifespan_days"].quantile(0.95),
        "median_repeat_gap_hours": gaps.median(),
        "p95_repeat_gap_hours": gaps.quantile(0.95),
        "baseline_fraud_rate": baseline_fraud_rate,
        "repeated_group_fraud_rate": repeated_fraud_rate,
        "repeated_group_fraud_lift": (
            repeated_fraud_rate / baseline_fraud_rate
            if baseline_fraud_rate > 0 and not np.isnan(repeated_fraud_rate)
            else np.nan
        ),
        "dense_group_threshold": DENSE_GROUP_SIZE,
        "dense_group_fraud_rate": dense_fraud_rate,
        "dense_group_fraud_lift": (
            dense_fraud_rate / baseline_fraud_rate
            if baseline_fraud_rate > 0 and not np.isnan(dense_fraud_rate)
            else np.nan
        ),
    }

    top_entities = (
        entity_temporal.sort_values("group_size", ascending=False)
        .head(25)
        .reset_index()
    )
    top_entities.insert(0, "relation", name)
    return result, top_entities


def build_temporal_edges(
    df: pd.DataFrame,
    proxy: pd.Series,
    k_previous: int,
) -> np.ndarray:
    valid = proxy.notna()
    temp = pd.DataFrame(
        {
            "entity": proxy.loc[valid].astype("string"),
            "time": df.loc[valid, "TransactionDT"].to_numpy(),
            "node": df.loc[valid, "node_id"].to_numpy(),
        }
    ).sort_values(["entity", "time", "node"])
    grouped = temp.groupby("entity", sort=False)
    edge_batches = []

    for lag in range(1, k_previous + 1):
        previous_node = grouped["node"].shift(lag)
        previous_time = grouped["time"].shift(lag)
        valid_edge = previous_node.notna() & (previous_time < temp["time"])
        if not valid_edge.any():
            continue

        source = temp.loc[valid_edge, "node"].to_numpy(dtype=np.uint64)
        destination = previous_node.loc[valid_edge].to_numpy(dtype=np.uint64)
        edge_batches.append(source * np.uint64(len(df)) + destination)

    if not edge_batches:
        return np.array([], dtype=np.uint64)
    return np.unique(np.concatenate(edge_batches))


def analyze_graph(name: str, df: pd.DataFrame, edge_ids: np.ndarray) -> dict:
    n_nodes = len(df)
    n_edges = len(edge_ids)
    baseline_fraud_rate = df["isFraud"].mean()
    if not n_edges:
        return {"relation": name, "edges": 0, "isolated_pct": 100.0}

    source = (edge_ids // np.uint64(n_nodes)).astype(np.int64)
    destination = (edge_ids % np.uint64(n_nodes)).astype(np.int64)
    degree = np.bincount(
        np.concatenate([source, destination]),
        minlength=n_nodes,
    )

    rows = np.concatenate([source, destination])
    columns = np.concatenate([destination, source])
    adjacency = coo_matrix(
        (np.ones(len(rows), dtype=np.uint8), (rows, columns)),
        shape=(n_nodes, n_nodes),
    ).tocsr()
    component_count, component_labels = connected_components(
        adjacency,
        directed=False,
        return_labels=True,
    )
    component_sizes = np.bincount(component_labels)

    labels = df["isFraud"].to_numpy()
    source_labels = labels[source]
    destination_labels = labels[destination]
    current_is_fraud = source_labels == 1
    current_is_normal = source_labels == 0
    previous_fraud_rate = (
        destination_labels[current_is_fraud].mean()
        if current_is_fraud.any()
        else np.nan
    )
    previous_normal_rate = (
        destination_labels[current_is_normal].mean()
        if current_is_normal.any()
        else np.nan
    )

    return {
        "relation": name,
        "edges": n_edges,
        "participating_nodes": int((degree > 0).sum()),
        "participating_nodes_pct": (degree > 0).mean() * 100,
        "isolated_nodes": int((degree == 0).sum()),
        "isolated_pct": (degree == 0).mean() * 100,
        "degree_mean": degree.mean(),
        "degree_median": np.median(degree),
        "degree_p90": np.quantile(degree, 0.90),
        "degree_p95": np.quantile(degree, 0.95),
        "degree_p99": np.quantile(degree, 0.99),
        "degree_max": degree.max(),
        "connected_components": component_count,
        "largest_component_size": int(component_sizes.max()),
        "largest_component_pct": component_sizes.max() / n_nodes * 100,
        "label_agreement": np.mean(source_labels == destination_labels),
        "baseline_fraud_rate": baseline_fraud_rate,
        "previous_fraud_rate_given_current_fraud": previous_fraud_rate,
        "previous_fraud_rate_given_current_normal": previous_normal_rate,
        "fraud_neighbor_lift": (
            previous_fraud_rate / baseline_fraud_rate
            if baseline_fraud_rate > 0 and not np.isnan(previous_fraud_rate)
            else np.nan
        ),
        "fraud_fraud_edge_pct": np.mean(
            (source_labels == 1) & (destination_labels == 1)
        )
        * 100,
    }


def calculate_edge_overlap(edge_sets: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(edge_sets)
    matrix = pd.DataFrame(index=names, columns=names, dtype=float)
    for name_a in names:
        for name_b in names:
            if name_a == name_b:
                matrix.loc[name_a, name_b] = 1.0
                continue

            edges_a = edge_sets[name_a]
            edges_b = edge_sets[name_b]
            intersection = np.intersect1d(
                edges_a,
                edges_b,
                assume_unique=True,
            ).size
            union = len(edges_a) + len(edges_b) - intersection
            matrix.loc[name_a, name_b] = intersection / union if union else np.nan
    return matrix


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data()
    print(f"Loaded {len(df):,} training transactions; fraud rate: {df['isFraud'].mean():.4%}")

    entity_results = []
    graph_results = []
    top_entity_results = []
    edge_sets: dict[str, np.ndarray] = {}

    for name, columns in CANDIDATES.items():
        proxy = build_strict_proxy(df, columns)
        entity_result, top_entities = analyze_entity_structure(name, df, proxy)
        edge_ids = build_temporal_edges(df, proxy, K_PREVIOUS_NEIGHBORS)
        graph_result = analyze_graph(name, df, edge_ids)

        entity_results.append(entity_result)
        graph_results.append(graph_result)
        edge_sets[name] = edge_ids
        if not top_entities.empty:
            top_entity_results.append(top_entities)

        print(
            f"{name}: {entity_result['coverage_pct']:.2f}% coverage, "
            f"{len(edge_ids):,} edges, {graph_result['isolated_pct']:.2f}% isolated"
        )

    entity_df = pd.DataFrame(entity_results)
    graph_df = pd.DataFrame(graph_results)
    overlap = calculate_edge_overlap(edge_sets)

    entity_df.to_csv(REPORT_DIR / "entity_diagnostics.csv", index=False)
    graph_df.to_csv(REPORT_DIR / "graph_diagnostics.csv", index=False)
    overlap.to_csv(REPORT_DIR / "edge_overlap_jaccard.csv")
    if top_entity_results:
        pd.concat(top_entity_results, ignore_index=True).to_csv(
            REPORT_DIR / "largest_entities.csv",
            index=False,
        )

    pd.DataFrame(
        [
            {
                "relation": name,
                "columns": " + ".join(columns),
                "k_previous_neighbors": K_PREVIOUS_NEIGHBORS,
            }
            for name, columns in CANDIDATES.items()
        ]
    ).to_csv(REPORT_DIR / "candidate_definitions.csv", index=False)

    print(f"Reports written to {REPORT_DIR.resolve()}")


if __name__ == "__main__":
    main()
