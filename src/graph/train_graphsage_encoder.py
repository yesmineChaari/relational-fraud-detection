"""Inductive GraphSAGE encoder producing per-transaction embeddings (G1).

This module trains the representation the scalar relational summaries (B1)
were a lossy substitute for: a fixed-length embedding per transaction, built
by mean-aggregating a temporally admissible neighbourhood rather than a
handful of count/recency scalars.

Training objective
-------------------
Supervised on `isFraud`, restricted to train-partition targets, through a
disposable classification head stacked on top of the encoder (`GraphSAGEHead`
below). The head is trained jointly with the encoder purely to shape the
embedding space; only the encoder's output is ever persisted as the final
per-transaction embedding, and the head's weights are discarded once
extraction happens. This route is stronger than a self-supervised
neighbourhood-structure objective, but it introduces a subtlety the temporal
contract (src/graph/temporal_contract.py) already calls out: because labels
now participate in representation learning, embeddings must be produced under
the same strictly-before discipline as everything else, and the encoder must
never see validation or test labels. Concretely: gradients are computed and
backpropagated only from train-partition targets; validation labels are used
solely as a forward-pass, no-grad early-stopping monitor (mirroring how B0/B1
evaluate validation each round without training on it); test labels are never
loaded by this module at all.

Architecture (design decisions recorded per the ticket)
---------------------------------------------------------
* 2 hops, fan-out (10, 10): matches the fan-outs already exercised by the
  sampler's own performance and correctness tests (tests/test_temporal_sampler.py),
  and keeps a training step's neighbourhood small enough for CPU-only torch
  (see reports/environment/g1_dependency_stack.json for why this stack has no
  GPU).
* Aggregator: mean, over the *valid* (non-padded) sampled neighbours only.
* Embedding width: 32; hidden width: 64. Small by design -- the input
  dimension (435, the frozen B0 predictor set) already dwarfs both, so the
  bottleneck is deliberately the embedding, not the hidden layer.
* Empty admissible neighbourhood: mean over zero valid neighbours is defined
  as the zero vector (`masked_mean` below clamps its divisor rather than
  dividing by zero), never an uninitialized or arbitrary value. A node whose
  neighbourhood is empty at a given hop therefore falls back to relying on
  its own transformed features alone at that hop -- a deliberate, documented
  choice, not a silent default.
* Early stopping: monitored on validation PR-AUC (the project's own primary
  ranking metric, `train_lightgbm_baseline.evaluate_validation`), forward-pass
  only, evaluated each epoch on a fixed-seed subsample of the validation
  partition (`VALIDATION_MONITOR_SIZE`) to keep per-epoch wall-clock cost
  tractable on CPU-only hardware; the metrics ultimately persisted to
  `reports/graphsage/card1/metrics.json` are computed once, after the
  best-monitored epoch is selected, over the *full* validation partition.

Mini-batch budget
-------------------
A full pass over the 413,378 train targets, sampling 10x10 neighbours per
target with the hand-rolled (pure-Python-loop) sampler, is not free: the
sampler's own tests benchmark it near the edge of what is tractable for a
single epoch (tests/test_temporal_sampler.py::test_two_hop_sampling_is_fast_enough_for_a_training_epoch).
Running that budget for every one of several training epochs, on top of the
torch forward/backward cost, would make this module impractical to run and
re-run in this environment. Each epoch therefore draws `STEPS_PER_EPOCH`
mini-batches of `BATCH_SIZE` targets *with* replacement across steps (not a
strict partition of the train set) -- a standard stochastic mini-batch
training regime, not a full-batch epoch. This is a deliberate, recorded
trade-off, not a hidden shortcut: `training_curve.csv` and `metadata.json`
both state the resulting samples-per-epoch and the fraction of the train
partition it represents.

Feature preprocessing and the memmap cache
---------------------------------------------
The frozen node feature table (`graph_card1_nodes.parquet`, one row per
transaction, 435 B0 predictors already integer-mapped for categoricals) is
~1GB as float64 in memory -- large enough, on a memory-constrained machine, to
risk starving the rest of the training process. `prepare_feature_cache`
converts it once into a train-fitted (median-imputed, z-scored) float32 flat
binary array on disk and every subsequent read uses `numpy.load(..., mmap_mode="r")`,
so a training step only pages in the handful of rows its current batch
actually touches rather than holding the full (590,540 x 435) matrix
resident. The cache is rebuilt column-by-column (never more than one column's
worth of raw data in memory at once) so construction itself stays cheap.
"""

from __future__ import annotations

import copy
import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.graph.build_transaction_graph import (
    ENTITY_EDGES_PATH,
    NODES_PATH,
    file_sha256,
)
from src.graph.temporal_contract import (
    RELATION_NAME,
    assert_no_forbidden_node_features,
)
from src.graph.temporal_sampler import (
    PAD_NODE_ID,
    TemporalGraphIndex,
    build_temporal_graph_index,
    sample_fixed_fanout,
)
from src.models.train_lightgbm_baseline import (
    EXPECTED_ROWS,
    EXPECTED_SPLIT_COUNTS,
    MODEL_DATASET_PATH,
    evaluate_validation,
    repository_relative,
    write_json,
)

ROOT_DIR = Path(__file__).resolve().parents[2]

FEATURE_CACHE_PATH = ROOT_DIR / "data" / "processed" / "graphsage_card1_node_features.npy"
FEATURE_SCALER_PATH = ROOT_DIR / "data" / "processed" / "graphsage_card1_feature_scaler.json"
EMBEDDINGS_PATH = ROOT_DIR / "data" / "processed" / "graphsage_card1_embeddings.parquet"
MODEL_PATH = ROOT_DIR / "models" / "graphsage_card1_encoder.pt"
REPORT_DIR = ROOT_DIR / "reports" / "graphsage" / "card1"
METADATA_PATH = REPORT_DIR / "metadata.json"
TRAINING_CURVE_PATH = REPORT_DIR / "training_curve.csv"
METRICS_PATH = REPORT_DIR / "metrics.json"

RANDOM_SEED = 42
HOP_FAN_OUTS = (10, 10)
HIDDEN_DIM = 64
EMBEDDING_DIM = 32
LEARNING_RATE = 1e-3
BATCH_SIZE = 256
STEPS_PER_EPOCH = 160
MAX_EPOCHS = 15
EARLY_STOPPING_PATIENCE = 4
EARLY_STOPPING_MIN_DELTA = 1e-4
VALIDATION_MONITOR_SIZE = 15_000
INFERENCE_BATCH_SIZE = 4_096
AGGREGATOR = "mean"

# Readout modes for the final layer. The default keeps the target's own
# transformed features alongside the aggregated neighbourhood, which is
# standard GraphSAGE. `neighbourhood_only` drops the self-contribution so the
# embedding carries strictly what the neighbourhood adds -- used by the G1
# attribution controls, where the downstream tabular model already holds every
# self feature and a self-inclusive embedding cannot isolate the relational
# signal.
READOUT_SELF_AND_NEIGHBOURHOOD = "self_and_neighbourhood"
READOUT_NEIGHBOURHOOD_ONLY = "neighbourhood_only"
READOUT_MODES = (READOUT_SELF_AND_NEIGHBOURHOOD, READOUT_NEIGHBOURHOOD_ONLY)


@dataclass(frozen=True)
class FeatureScaler:
    feature_columns: list[str]
    median: np.ndarray
    mean: np.ndarray
    std: np.ndarray


def load_node_metadata() -> pd.DataFrame:
    """node_id, TransactionID, split -- row order already equals node_id."""
    meta = pd.read_parquet(NODES_PATH, columns=["node_id", "TransactionID", "split"])
    if len(meta) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} nodes; got {len(meta):,}.")
    if not (meta["node_id"].to_numpy() == np.arange(len(meta))).all():
        raise AssertionError("Node table row order does not equal node_id (0..N-1).")
    if "isFraud" in meta.columns:
        raise AssertionError("isFraud must never be read by the encoder's feature path.")

    actual_split_counts = {
        str(name): int(count)
        for name, count in meta["split"].astype("string").value_counts().items()
    }
    if actual_split_counts != EXPECTED_SPLIT_COUNTS:
        raise ValueError(
            f"Unexpected split counts in the node table. Expected "
            f"{EXPECTED_SPLIT_COUNTS}, got {actual_split_counts}."
        )
    return meta


def get_feature_columns() -> list[str]:
    schema_names = pq.ParquetFile(NODES_PATH).schema.names
    excluded = {"node_id", "TransactionID", "split"}
    feature_columns = [name for name in schema_names if name not in excluded]
    assert_no_forbidden_node_features(feature_columns)
    return feature_columns


def prepare_feature_cache(
    feature_columns: list[str],
    train_mask: np.ndarray,
) -> FeatureScaler:
    """Build (or reuse) the train-fitted, standardized float32 feature cache.

    Rebuilds column-by-column so peak memory stays O(N) for one column, never
    O(N x D) for the full table, per the module-level docstring's rationale.
    """
    n_rows = len(train_mask)
    n_features = len(feature_columns)

    if FEATURE_CACHE_PATH.exists() and FEATURE_SCALER_PATH.exists():
        with FEATURE_SCALER_PATH.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("feature_columns") == feature_columns:
            return FeatureScaler(
                feature_columns=feature_columns,
                median=np.asarray(payload["median"], dtype=np.float64),
                mean=np.asarray(payload["mean"], dtype=np.float64),
                std=np.asarray(payload["std"], dtype=np.float64),
            )

    FEATURE_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    cache = np.lib.format.open_memmap(
        FEATURE_CACHE_PATH, mode="w+", dtype=np.float32, shape=(n_rows, n_features)
    )
    median = np.empty(n_features, dtype=np.float64)
    mean = np.empty(n_features, dtype=np.float64)
    std = np.empty(n_features, dtype=np.float64)

    for i, column in enumerate(feature_columns):
        values = pd.read_parquet(NODES_PATH, columns=[column])[column].to_numpy(
            dtype=np.float64, copy=False
        )
        train_values = values[train_mask]
        train_finite = train_values[~np.isnan(train_values)]
        column_median = float(np.median(train_finite)) if len(train_finite) else 0.0
        imputed_train = np.where(np.isnan(train_values), column_median, train_values)
        column_mean = float(imputed_train.mean())
        column_std = float(imputed_train.std())
        if not np.isfinite(column_std) or column_std < 1e-6:
            column_std = 1.0

        imputed_all = np.where(np.isnan(values), column_median, values)
        cache[:, i] = ((imputed_all - column_mean) / column_std).astype(np.float32)
        median[i] = column_median
        mean[i] = column_mean
        std[i] = column_std
        del values, train_values, train_finite, imputed_train, imputed_all

    cache.flush()
    del cache

    write_json(
        FEATURE_SCALER_PATH,
        {
            "fit_split": "train",
            "feature_columns": feature_columns,
            "median": median.tolist(),
            "mean": mean.tolist(),
            "std": std.tolist(),
        },
    )
    return FeatureScaler(feature_columns=feature_columns, median=median, mean=mean, std=std)


def load_labels(meta: pd.DataFrame) -> np.ndarray:
    """isFraud indexed by node_id; NaN (never read) for test-partition nodes."""
    labels_df = pd.read_parquet(
        MODEL_DATASET_PATH,
        columns=["TransactionID", "isFraud"],
        filters=[("split", "in", ["train", "validation"])],
    )
    merged = meta.merge(labels_df, on="TransactionID", how="left", validate="one_to_one")
    if len(merged) != len(meta):
        raise AssertionError("Label merge changed row count.")
    labels_by_node_id = merged.sort_values("node_id")["isFraud"].to_numpy(dtype=np.float64)

    train_mask = (meta["split"] == "train").to_numpy()
    validation_mask = (meta["split"] == "validation").to_numpy()
    if np.isnan(labels_by_node_id[train_mask]).any():
        raise ValueError("Missing isFraud for a train-partition transaction.")
    if np.isnan(labels_by_node_id[validation_mask]).any():
        raise ValueError("Missing isFraud for a validation-partition transaction.")
    return labels_by_node_id


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Mean over the True positions of `mask` along `dim`; zero if none are True."""
    mask_f = mask.unsqueeze(-1).to(x.dtype)
    summed = (x * mask_f).sum(dim=dim)
    counts = mask_f.sum(dim=dim).clamp(min=1.0)
    return summed / counts


def l2_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    norm = x.norm(p=2, dim=dim, keepdim=True).clamp(min=eps)
    return x / norm


class TemporalGraphSAGEEncoder(nn.Module):
    """Hand-rolled 2-layer mean-aggregator GraphSAGE, target-anchored via the sampler.

    Layer 1 refines each sampled hop-1 node using its own features and the
    mean of *its* (hop-2) neighbours' raw features. Layer 2 produces the
    target's embedding from its own features and the mean of its hop-1
    nodes' layer-1 embeddings. Both layers L2-normalize their output, the
    standard GraphSAGE convention.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        readout: str = READOUT_SELF_AND_NEIGHBOURHOOD,
    ) -> None:
        super().__init__()
        if readout not in READOUT_MODES:
            raise ValueError(f"readout must be one of {READOUT_MODES}; got {readout!r}.")
        self.readout = readout
        self.layer1 = nn.Linear(input_dim * 2, hidden_dim)
        layer2_input_dim = (
            hidden_dim if readout == READOUT_NEIGHBOURHOOD_ONLY else input_dim + hidden_dim
        )
        self.layer2 = nn.Linear(layer2_input_dim, embedding_dim)

    def forward(
        self,
        x_target: torch.Tensor,
        x_hop1: torch.Tensor,
        hop1_mask: torch.Tensor,
        x_hop2: torch.Tensor,
        hop2_mask: torch.Tensor,
    ) -> torch.Tensor:
        agg2 = masked_mean(x_hop2, hop2_mask, dim=2)
        h1 = F.relu(self.layer1(torch.cat([x_hop1, agg2], dim=-1)))
        h1 = l2_normalize(h1, dim=-1)
        h1 = h1 * hop1_mask.unsqueeze(-1).to(h1.dtype)

        agg1 = masked_mean(h1, hop1_mask, dim=1)
        if self.readout == READOUT_NEIGHBOURHOOD_ONLY:
            layer2_input = agg1
        else:
            layer2_input = torch.cat([x_target, agg1], dim=-1)
        embedding = F.relu(self.layer2(layer2_input))
        return l2_normalize(embedding, dim=-1)


class GraphSAGEWithHead(nn.Module):
    """The encoder plus a disposable linear classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        readout: str = READOUT_SELF_AND_NEIGHBOURHOOD,
    ) -> None:
        super().__init__()
        self.encoder = TemporalGraphSAGEEncoder(input_dim, hidden_dim, embedding_dim, readout)
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(*args, **kwargs)
        logit = self.head(embedding).squeeze(-1)
        return embedding, logit


def sample_batch_neighborhoods(
    index: TemporalGraphIndex,
    target_node_ids: np.ndarray,
    fan_outs: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fixed-shape (B, F1) / (B, F1, F2) hop-1 / hop-2 ids and validity masks.

    Reuses `temporal_sampler.sample_fixed_fanout` directly for every hop-1 and
    hop-2 draw -- the exact, already-tested primitive that enforces the
    target-anchored strictly-before rule -- rather than `sample_k_hop`.
    `sample_k_hop`'s frontier-compaction (dropping invalid hop-1 slots before
    expanding hop-2) is correct for a single node but produces a ragged
    hop-2 count that varies per target, which cannot be stacked into a
    uniform tensor across a batch. Calling `sample_fixed_fanout` once per
    hop-1 slot (valid or not) keeps every target's hop-2 block a fixed
    (F1, F2) shape while applying the identical admissibility rule.
    """
    fan_out_1, fan_out_2 = fan_outs
    batch_size = len(target_node_ids)

    hop1_ids = np.full((batch_size, fan_out_1), PAD_NODE_ID, dtype=np.int64)
    hop1_mask = np.zeros((batch_size, fan_out_1), dtype=bool)
    hop2_ids = np.full((batch_size, fan_out_1, fan_out_2), PAD_NODE_ID, dtype=np.int64)
    hop2_mask = np.zeros((batch_size, fan_out_1, fan_out_2), dtype=bool)
    target_dts = np.empty(batch_size, dtype=np.float64)

    for b, target in enumerate(target_node_ids):
        target = int(target)
        target_position = index.position_of_node[target]
        target_dt = float(index.transaction_dt[target_position])
        target_dts[b] = target_dt

        ids1, mask1 = sample_fixed_fanout(index, target, fan_out_1, rng, target_dt=None)
        hop1_ids[b] = ids1
        hop1_mask[b] = mask1
        for j in range(fan_out_1):
            if mask1[j]:
                ids2, mask2 = sample_fixed_fanout(
                    index, int(ids1[j]), fan_out_2, rng, target_dt=target_dt
                )
                hop2_ids[b, j] = ids2
                hop2_mask[b, j] = mask2

    return hop1_ids, hop1_mask, hop2_ids, hop2_mask


def gather_features(
    feature_cache: np.ndarray,
    node_ids: np.ndarray,
    mask: np.ndarray | None = None,
) -> np.ndarray:
    """Look up rows by node id; PAD_NODE_ID (or masked-False) rows come back zeroed."""
    shape = node_ids.shape
    flat_ids = node_ids.reshape(-1)
    safe_ids = np.where(flat_ids == PAD_NODE_ID, 0, flat_ids)
    rows = np.asarray(feature_cache[safe_ids], dtype=np.float32)
    if mask is not None:
        rows = rows * mask.reshape(-1, 1).astype(np.float32)
    else:
        rows = rows * (flat_ids != PAD_NODE_ID).reshape(-1, 1).astype(np.float32)
    return rows.reshape(*shape, -1)


def build_batch_tensors(
    index: TemporalGraphIndex,
    feature_cache: np.ndarray,
    target_node_ids: np.ndarray,
    fan_outs: tuple[int, int],
    rng: np.random.Generator,
) -> dict[str, torch.Tensor]:
    hop1_ids, hop1_mask, hop2_ids, hop2_mask = sample_batch_neighborhoods(
        index, target_node_ids, fan_outs, rng
    )
    x_target = gather_features(feature_cache, target_node_ids)
    x_hop1 = gather_features(feature_cache, hop1_ids, mask=hop1_mask)
    x_hop2 = gather_features(feature_cache, hop2_ids, mask=hop2_mask)
    return {
        "x_target": torch.from_numpy(x_target),
        "x_hop1": torch.from_numpy(x_hop1),
        "hop1_mask": torch.from_numpy(hop1_mask),
        "x_hop2": torch.from_numpy(x_hop2),
        "hop2_mask": torch.from_numpy(hop2_mask),
    }


def run_inference(
    model: GraphSAGEWithHead,
    index: TemporalGraphIndex,
    feature_cache: np.ndarray,
    node_ids: np.ndarray,
    fan_outs: tuple[int, int],
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    """Forward-only embeddings for `node_ids`, batched, no gradient tracked."""
    model.eval()
    embeddings = np.empty((len(node_ids), EMBEDDING_DIM), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[start : start + batch_size]
            tensors = build_batch_tensors(index, feature_cache, batch_ids, fan_outs, rng)
            embedding, _ = model(
                tensors["x_target"],
                tensors["x_hop1"],
                tensors["hop1_mask"],
                tensors["x_hop2"],
                tensors["hop2_mask"],
            )
            embeddings[start : start + len(batch_ids)] = embedding.numpy()
    return embeddings


def run_logits(
    model: GraphSAGEWithHead,
    index: TemporalGraphIndex,
    feature_cache: np.ndarray,
    node_ids: np.ndarray,
    fan_outs: tuple[int, int],
    rng: np.random.Generator,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    logits = np.empty(len(node_ids), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[start : start + batch_size]
            tensors = build_batch_tensors(index, feature_cache, batch_ids, fan_outs, rng)
            _, logit = model(
                tensors["x_target"],
                tensors["x_hop1"],
                tensors["hop1_mask"],
                tensors["x_hop2"],
                tensors["hop2_mask"],
            )
            logits[start : start + len(batch_ids)] = logit.numpy()
    return logits


def train_encoder() -> dict[str, Any]:
    torch.manual_seed(RANDOM_SEED)
    rng = np.random.default_rng(RANDOM_SEED)

    print(f"[{RELATION_NAME}] Loading node metadata and entity-edge index...")
    meta = load_node_metadata()
    feature_columns = get_feature_columns()
    train_mask = (meta["split"] == "train").to_numpy()
    validation_mask = (meta["split"] == "validation").to_numpy()

    edges_df = pd.read_parquet(ENTITY_EDGES_PATH, columns=["entity_id", "node_id", "TransactionDT"])
    index = build_temporal_graph_index(edges_df)
    del edges_df

    print(
        f"[{RELATION_NAME}] Preparing the standardized feature cache ({len(feature_columns)} predictors)..."
    )
    prepare_feature_cache(feature_columns, train_mask)
    feature_cache = np.load(FEATURE_CACHE_PATH, mmap_mode="r")
    if feature_cache.shape != (len(meta), len(feature_columns)):
        raise AssertionError("Feature cache shape does not match the node table.")

    labels_by_node_id = load_labels(meta)
    train_node_ids = meta.loc[train_mask, "node_id"].to_numpy()
    validation_node_ids = meta.loc[validation_mask, "node_id"].to_numpy()
    test_node_ids = meta.loc[meta["split"] == "test", "node_id"].to_numpy()
    all_node_ids = meta["node_id"].to_numpy()

    monitor_size = min(VALIDATION_MONITOR_SIZE, len(validation_node_ids))
    validation_monitor_ids = rng.choice(validation_node_ids, size=monitor_size, replace=False)

    n_positive = int(labels_by_node_id[train_node_ids].sum())
    n_negative = len(train_node_ids) - n_positive
    pos_weight = torch.tensor(float(n_negative) / float(n_positive), dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = GraphSAGEWithHead(len(feature_columns), HIDDEN_DIM, EMBEDDING_DIM)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_state: dict[str, Any] | None = None
    best_val_pr_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    curve_rows: list[dict[str, Any]] = []
    stopped_epoch = 0

    print(
        f"[{RELATION_NAME}] Training: {STEPS_PER_EPOCH} steps/epoch x "
        f"{BATCH_SIZE} targets, up to {MAX_EPOCHS} epochs, patience {EARLY_STOPPING_PATIENCE}..."
    )
    for epoch in range(1, MAX_EPOCHS + 1):
        stopped_epoch = epoch
        model.train()
        epoch_losses: list[float] = []
        for _ in range(STEPS_PER_EPOCH):
            batch_ids = rng.choice(train_node_ids, size=BATCH_SIZE, replace=False)
            tensors = build_batch_tensors(index, feature_cache, batch_ids, HOP_FAN_OUTS, rng)
            y = torch.from_numpy(labels_by_node_id[batch_ids].astype(np.float32))

            optimizer.zero_grad()
            _, logit = model(
                tensors["x_target"],
                tensors["x_hop1"],
                tensors["hop1_mask"],
                tensors["x_hop2"],
                tensors["hop2_mask"],
            )
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        monitor_logits = run_logits(
            model,
            index,
            feature_cache,
            validation_monitor_ids,
            HOP_FAN_OUTS,
            rng,
            INFERENCE_BATCH_SIZE,
        )
        monitor_probs = 1.0 / (1.0 + np.exp(-monitor_logits))
        monitor_labels = labels_by_node_id[validation_monitor_ids]
        val_pr_auc = float(average_precision_score(monitor_labels, monitor_probs))
        mean_loss = float(np.mean(epoch_losses))
        curve_rows.append(
            {
                "epoch": epoch,
                "train_loss_mean": mean_loss,
                "validation_pr_auc_monitor": val_pr_auc,
            }
        )
        print(
            f"[{RELATION_NAME}] epoch {epoch:>2d}: train_loss={mean_loss:.6f} "
            f"validation_pr_auc(monitor)={val_pr_auc:.6f}"
        )

        if val_pr_auc > best_val_pr_auc + EARLY_STOPPING_MIN_DELTA:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
                print(
                    f"[{RELATION_NAME}] Early stopping at epoch {epoch} (best epoch {best_epoch})."
                )
                break

    if best_state is None:
        raise RuntimeError(
            "Training never improved on the validation monitor; no best state recorded."
        )
    model.load_state_dict(best_state)
    estimator_cap_reached = (
        stopped_epoch == MAX_EPOCHS and epochs_without_improvement < EARLY_STOPPING_PATIENCE
    )

    print(f"[{RELATION_NAME}] Scoring the full validation partition with the best encoder...")
    full_val_logits = run_logits(
        model, index, feature_cache, validation_node_ids, HOP_FAN_OUTS, rng, INFERENCE_BATCH_SIZE
    )
    full_val_probs = 1.0 / (1.0 + np.exp(-full_val_logits))
    y_validation = pd.Series(labels_by_node_id[validation_node_ids].astype("int8"))
    validation_metrics = evaluate_validation(
        y_validation, full_val_probs, run_name="graphsage_card1"
    )
    validation_metrics["model"] = "graphsage_card1_encoder_with_head"

    print(f"[{RELATION_NAME}] Extracting embeddings for all {len(all_node_ids):,} transactions...")
    embeddings = run_inference(
        model, index, feature_cache, all_node_ids, HOP_FAN_OUTS, rng, INFERENCE_BATCH_SIZE
    )

    return {
        "model": model,
        "meta": meta,
        "feature_columns": feature_columns,
        "curve_rows": curve_rows,
        "best_epoch": best_epoch,
        "best_val_pr_auc": best_val_pr_auc,
        "stopped_epoch": stopped_epoch,
        "estimator_cap_reached": estimator_cap_reached,
        "early_stopping_triggered": not estimator_cap_reached,
        "validation_metrics": validation_metrics,
        "embeddings": embeddings,
        "train_rows": int(len(train_node_ids)),
        "validation_rows": int(len(validation_node_ids)),
        "test_rows": int(len(test_node_ids)),
        "validation_monitor_size": int(monitor_size),
        "train_fraud": n_positive,
        "pos_weight": float(pos_weight.item()),
    }


def save_embeddings(meta: pd.DataFrame, embeddings: np.ndarray) -> None:
    if embeddings.shape[0] != len(meta):
        raise AssertionError("Embedding row count does not match the node table.")
    if not np.isfinite(embeddings).all():
        raise AssertionError("Embeddings contain non-finite values.")

    columns = {f"embedding_{i:02d}": embeddings[:, i] for i in range(embeddings.shape[1])}
    frame = pd.DataFrame(
        {
            "TransactionID": meta["TransactionID"].to_numpy(),
            "split": meta["split"].astype("string").to_numpy(),
            **columns,
        }
    )
    if len(frame) != EXPECTED_ROWS:
        raise AssertionError("Embedding artifact row count is not the expected transaction count.")
    if frame.isna().any().any():
        raise AssertionError("Embedding artifact contains nulls.")
    if "isFraud" in frame.columns:
        raise AssertionError("isFraud must never appear in the embedding artifact.")

    EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(EMBEDDINGS_PATH, index=False, engine="pyarrow", compression="snappy")


def build_metadata(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "relation_name": RELATION_NAME,
        "model_name": "graphsage_card1_encoder",
        "random_seed": RANDOM_SEED,
        "architecture": {
            "n_hops": len(HOP_FAN_OUTS),
            "fan_outs": list(HOP_FAN_OUTS),
            "aggregator": AGGREGATOR,
            "hidden_dim": HIDDEN_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "activation": "relu",
            "normalization": "l2_per_layer",
            "readout": READOUT_SELF_AND_NEIGHBOURHOOD,
            "empty_neighborhood_policy": (
                "masked mean over zero valid neighbours is defined as the zero "
                "vector; a node falls back to its own transformed features at "
                "that hop rather than receiving an arbitrary or uninitialized value"
            ),
            "input_dim": len(result["feature_columns"]),
        },
        "training_objective": (
            "supervised_isFraud_via_disposable_head; only the encoder's "
            "embedding output is persisted, the classification head is "
            "discarded after training"
        ),
        "training_regime": "inductive",
        "target_labels_used_for_training": True,
        "target_labels_used_in_node_features": False,
        "test_labels_used": False,
        "mini_batch_budget": {
            "steps_per_epoch": STEPS_PER_EPOCH,
            "batch_size": BATCH_SIZE,
            "samples_per_epoch": STEPS_PER_EPOCH * BATCH_SIZE,
            "train_partition_rows": result["train_rows"],
            "fraction_of_train_partition_per_epoch": (
                STEPS_PER_EPOCH * BATCH_SIZE / result["train_rows"]
            ),
            "sampling_note": (
                "Targets are drawn without replacement within a step and with "
                "replacement across steps -- a stochastic mini-batch budget, "
                "not a strict partition of the train set. Recorded here rather "
                "than presented as a full epoch."
            ),
        },
        "early_stopping": {
            "metric": "validation_pr_auc",
            "monitor_evaluation_set": "fixed_seed_subsample_of_validation_partition",
            "monitor_sample_size": result["validation_monitor_size"],
            "patience": EARLY_STOPPING_PATIENCE,
            "min_delta": EARLY_STOPPING_MIN_DELTA,
            "max_epochs": MAX_EPOCHS,
            "stopped_epoch": result["stopped_epoch"],
            "best_epoch": result["best_epoch"],
            "best_validation_pr_auc_monitor": result["best_val_pr_auc"],
            "estimator_cap_reached": result["estimator_cap_reached"],
            "early_stopping_triggered": result["early_stopping_triggered"],
            "final_metrics_evaluation_set": "full_validation_partition",
        },
        "class_weighting": {
            "train_fraud_count": result["train_fraud"],
            "train_rows": result["train_rows"],
            "pos_weight": result["pos_weight"],
        },
        "split_row_counts": {
            "train": result["train_rows"],
            "validation": result["validation_rows"],
            "test": result["test_rows"],
        },
        "reproducibility": {
            "torch_manual_seed": RANDOM_SEED,
            "numpy_generator_seed": RANDOM_SEED,
            "known_residual_nondeterminism": (
                "CPU multi-threaded floating point reduction order in torch's "
                "Linear/BCE ops can introduce sub-1e-6 differences across full "
                "training runs even under a fixed seed; a full re-run was not "
                "re-verified byte-for-byte given its multi-minute wall-clock "
                "cost. A short, fixed-seed forward+backward smoke test in "
                "tests/test_graphsage_encoder.py verifies bit-identical loss "
                "and gradients for a single step, the same verification "
                "approach already used for the torch install check "
                "(src/graph/environment_check.py)."
            ),
        },
        "input_model_dataset_path": repository_relative(MODEL_DATASET_PATH),
        "input_model_dataset_sha256": file_sha256(MODEL_DATASET_PATH),
        "input_nodes_path": repository_relative(NODES_PATH),
        "input_nodes_sha256": file_sha256(NODES_PATH),
        "input_entity_edges_path": repository_relative(ENTITY_EDGES_PATH),
        "input_entity_edges_sha256": file_sha256(ENTITY_EDGES_PATH),
        "feature_scaler_path": repository_relative(FEATURE_SCALER_PATH),
        "feature_scaler_sha256": file_sha256(FEATURE_SCALER_PATH),
        "feature_columns": result["feature_columns"],
        "feature_column_count": len(result["feature_columns"]),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "embeddings_path": repository_relative(EMBEDDINGS_PATH),
        "embedding_row_count": EXPECTED_ROWS,
        "embedding_has_no_nulls": True,
        "embedding_has_no_target_column": True,
        "model_path": repository_relative(MODEL_PATH),
        "metrics_path": repository_relative(METRICS_PATH),
        "training_curve_path": repository_relative(TRAINING_CURVE_PATH),
        "depends_on": [
            "src/graph/temporal_sampler.py",
            "src/graph/build_transaction_graph.py",
            "reports/environment/g1_dependency_stack.json",
        ],
    }


def build_and_train_graphsage_encoder() -> None:
    result = train_encoder()

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    torch.save(result["model"].state_dict(), MODEL_PATH)
    save_embeddings(result["meta"], result["embeddings"])
    write_json(METRICS_PATH, result["validation_metrics"])
    pd.DataFrame(result["curve_rows"]).to_csv(TRAINING_CURVE_PATH, index=False)

    metadata = build_metadata(result)
    write_json(METADATA_PATH, metadata)

    saved_embeddings = pd.read_parquet(EMBEDDINGS_PATH)
    if len(saved_embeddings) != EXPECTED_ROWS:
        raise AssertionError("Saved embedding artifact row count changed.")
    if "isFraud" in saved_embeddings.columns:
        raise AssertionError("isFraud leaked into the saved embedding artifact.")

    print(
        f"\n[{RELATION_NAME}] Best epoch: {result['best_epoch']} / stopped at {result['stopped_epoch']}"
    )
    print(
        f"[{RELATION_NAME}] Early stopping triggered: {'YES' if result['early_stopping_triggered'] else 'NO'}"
    )
    print(f"[{RELATION_NAME}] Full validation PR-AUC: {result['validation_metrics']['pr_auc']:.8f}")
    print(
        f"[{RELATION_NAME}] Full validation ROC-AUC: {result['validation_metrics']['roc_auc']:.8f}"
    )
    print(f"[{RELATION_NAME}] Embeddings saved: {EMBEDDINGS_PATH}")
    print(f"[{RELATION_NAME}] Model saved: {MODEL_PATH}")
    print(f"[{RELATION_NAME}] Metrics saved: {METRICS_PATH}")
    print(f"[{RELATION_NAME}] Training curve saved: {TRAINING_CURVE_PATH}")
    print(f"[{RELATION_NAME}] Metadata saved: {METADATA_PATH}")
    print(f"[{RELATION_NAME}] Test labels used: NO")


def main() -> None:
    build_and_train_graphsage_encoder()


if __name__ == "__main__":
    main()
