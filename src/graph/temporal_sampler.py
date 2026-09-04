"""The strictly-before temporal neighbour sampler for the G1 relational stage.

Every stock neighbour sampler shipped by a graph library samples without
regard to time; using one unmodified would let a target transaction
aggregate information from transactions that had not yet happened. This is
the single place the temporal contract's admissibility rule
(src/graph/temporal_contract.py) is turned into a sampling procedure, the
same way compute_relational_features is the single place the scalar
strictly-before scan lives -- so nothing downstream can drift from it.

Speed comes from the same trick the scalar scan already uses: the entity
edges table (src/graph/build_transaction_graph.py) is sorted by
[entity_id, TransactionDT, node_id], so for any timestamp bound the
admissible same-entity candidates are a single contiguous slice, found by
one binary search per query rather than a scan.

Hop composition (contract rule 3): a target's own timestamp is threaded
through every hop of a walk explicitly, via the `target_dt` argument, and
is never replaced by an intermediate node's own timestamp as the walk
descends. sample_k_hop is the callable that enforces this in code -- it is
not left to happen to follow from repeated hop-1 filtering.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PAD_NODE_ID = -1


@dataclass(frozen=True)
class TemporalGraphIndex:
    """CSR-style index over the sorted entity edges table.

    `node_id`, `transaction_dt`, and `entity_id` are parallel arrays in
    [entity_id, TransactionDT, node_id] sort order. `entity_boundaries` is a
    length-(n_entities + 1) row-pointer array: entity `e`'s rows occupy
    positions [entity_boundaries[e], entity_boundaries[e + 1]).
    `position_of_node` maps a node_id back to its position in these arrays.
    """

    node_id: np.ndarray
    transaction_dt: np.ndarray
    entity_id: np.ndarray
    entity_boundaries: np.ndarray
    position_of_node: np.ndarray

    @property
    def n_nodes(self) -> int:
        return len(self.node_id)

    @property
    def n_entities(self) -> int:
        return len(self.entity_boundaries) - 1


def _validate_sorted_edges(edges_df: pd.DataFrame) -> None:
    required = {"entity_id", "node_id", "TransactionDT"}
    missing = required - set(edges_df.columns)
    if missing:
        raise ValueError(f"Entity edges table is missing columns: {sorted(missing)}.")
    if edges_df.empty:
        raise ValueError("Entity edges table is empty.")

    entity_id = edges_df["entity_id"].to_numpy()
    if not np.all(np.diff(entity_id) >= 0):
        raise ValueError("Entity edges table is not sorted by entity_id.")

    same_entity = np.diff(entity_id) == 0
    dt_diff = np.diff(edges_df["TransactionDT"].to_numpy())
    if not np.all(dt_diff[same_entity] >= 0):
        raise ValueError(
            "Entity edges table is not sorted by TransactionDT within an entity."
        )

    node_id = edges_df["node_id"].to_numpy()
    if node_id.min() != 0 or node_id.max() != len(node_id) - 1:
        raise ValueError("node_id must be a dense permutation of 0..N-1.")
    if len(np.unique(node_id)) != len(node_id):
        raise ValueError("node_id contains duplicates.")


def build_temporal_graph_index(edges_df: pd.DataFrame) -> TemporalGraphIndex:
    """Build the sampler's index from an entity edges table.

    `edges_df` must already be sorted by [entity_id, TransactionDT, node_id]
    (build_entity_edges in build_transaction_graph.py guarantees this); this
    is validated, not assumed.
    """
    _validate_sorted_edges(edges_df)

    node_id = edges_df["node_id"].to_numpy(dtype=np.int64, copy=True)
    transaction_dt = edges_df["TransactionDT"].to_numpy(dtype=np.float64, copy=True)
    entity_id = edges_df["entity_id"].to_numpy(dtype=np.int64, copy=True)

    n = len(edges_df)
    n_entities = int(entity_id.max()) + 1
    entity_boundaries = np.searchsorted(entity_id, np.arange(n_entities + 1))

    position_of_node = np.empty(n, dtype=np.int64)
    position_of_node[node_id] = np.arange(n, dtype=np.int64)

    return TemporalGraphIndex(
        node_id=node_id,
        transaction_dt=transaction_dt,
        entity_id=entity_id,
        entity_boundaries=entity_boundaries,
        position_of_node=position_of_node,
    )


def load_temporal_graph_index() -> TemporalGraphIndex:
    from src.graph.build_transaction_graph import ENTITY_EDGES_PATH

    if not ENTITY_EDGES_PATH.exists():
        raise FileNotFoundError(
            f"Entity edges table not found: {ENTITY_EDGES_PATH}. Run "
            "`python -m src.graph.build_transaction_graph` first."
        )
    edges_df = pd.read_parquet(
        ENTITY_EDGES_PATH, columns=["entity_id", "node_id", "TransactionDT"]
    )
    return build_temporal_graph_index(edges_df)


def admissible_neighbors(
    index: TemporalGraphIndex,
    node_id_query: int,
    target_dt: float | None = None,
) -> np.ndarray:
    """Node ids of `node_id_query`'s same-entity, strictly-earlier-than-`target_dt` peers.

    `target_dt` is the bound that governs admissibility (contract rule 3).
    Pass it explicitly and unchanged at every hop of a multi-hop walk -- it
    must always be the original walk target's own timestamp, never the
    current node's. When `target_dt` is omitted, the node's own timestamp is
    used, which is the correct (and only sensible) bound for a standalone
    hop-1 query.
    """
    position = index.position_of_node[node_id_query]
    if target_dt is None:
        target_dt = index.transaction_dt[position]

    entity = index.entity_id[position]
    start = index.entity_boundaries[entity]
    end = index.entity_boundaries[entity + 1]
    entity_times = index.transaction_dt[start:end]

    cutoff = start + int(np.searchsorted(entity_times, target_dt, side="left"))
    candidate_positions = np.arange(start, cutoff)
    # A node is never its own neighbour. This only bites when target_dt is
    # supplied externally (hop 2+): the querying node's own timestamp can
    # then be strictly before target_dt even though it can never be before
    # its own, so it must be filtered out explicitly rather than relying on
    # the cutoff to have excluded it.
    candidate_positions = candidate_positions[candidate_positions != position]
    return index.node_id[candidate_positions]


def sample_fixed_fanout(
    index: TemporalGraphIndex,
    node_id_query: int,
    fan_out: int,
    rng: np.random.Generator,
    target_dt: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample up to `fan_out` admissible neighbours, padded to a fixed shape.

    Returns (node_ids, valid_mask), both length `fan_out`. Padding slots use
    PAD_NODE_ID (-1, never a real node id) *and* a False mask entry, so an
    empty or short neighbourhood is unambiguous rather than a silent zero
    that could be mistaken for a real node 0.
    """
    if fan_out <= 0:
        raise ValueError("fan_out must be positive.")

    candidates = admissible_neighbors(index, node_id_query, target_dt)
    n_available = len(candidates)
    n_take = min(n_available, fan_out)

    padded = np.full(fan_out, PAD_NODE_ID, dtype=np.int64)
    mask = np.zeros(fan_out, dtype=bool)
    if n_take > 0:
        chosen_positions = rng.choice(n_available, size=n_take, replace=False)
        padded[:n_take] = candidates[chosen_positions]
        mask[:n_take] = True
    return padded, mask


def sample_k_hop(
    index: TemporalGraphIndex,
    target_node_id: int,
    fan_outs: list[int],
    rng: np.random.Generator,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Sample a k-hop neighbourhood, target-anchored at every hop.

    Returns one (node_ids, valid_mask) pair per hop in `fan_outs`. Hop h's
    candidates are the union, over every valid node sampled at hop h-1, of
    that node's admissible neighbours -- bounded throughout by the
    *original* target's own timestamp, never an intermediate node's. This is
    the explicit, in-code enforcement of contract rule 3: nothing here
    relies on hop-1 filtering happening to make deeper hops safe.
    """
    if not fan_outs:
        raise ValueError("fan_outs must contain at least one hop.")

    target_position = index.position_of_node[target_node_id]
    target_dt = index.transaction_dt[target_position]

    layers: list[tuple[np.ndarray, np.ndarray]] = []
    frontier = np.array([target_node_id], dtype=np.int64)
    for fan_out in fan_outs:
        sampled_chunks = [
            sample_fixed_fanout(index, int(node), fan_out, rng, target_dt=target_dt)
            for node in frontier
        ]
        if sampled_chunks:
            layer_nodes = np.concatenate([nodes for nodes, _ in sampled_chunks])
            layer_mask = np.concatenate([mask for _, mask in sampled_chunks])
        else:
            layer_nodes = np.empty(0, dtype=np.int64)
            layer_mask = np.empty(0, dtype=bool)
        layers.append((layer_nodes, layer_mask))
        frontier = layer_nodes[layer_mask]
    return layers
