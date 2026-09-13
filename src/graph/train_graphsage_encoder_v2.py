"""Cardinality-aware GraphSAGE encoder on card1 (G1-v2, Stage 1).

The frozen encoder cannot tell a card with 10 prior transactions from one with
500: `sample_fixed_fanout` computes the true number of admissible neighbours
and discards it, and `masked_mean` is invariant to how many neighbours
contributed. card1's median entity has 4 prior transactions but its p95 is 84
and its p99 about 600, so the 10-neighbour cap hides exactly the history depth
`prior_count` -- B1-card1's strongest non-redundant feature -- carries. Here
`count_admissible_neighbors` supplies the true count and it enters both layers
as log1p(count).

What the counts add, stated so a positive result cannot be over-read:

* The target count *is* `prior_count`, which LightGBM already receives in
  G1-v2's 471-column manifest. It is not new information. It is there so the
  encoder can condition its aggregation on how much history stands behind a
  mean of at most 10 draws -- an interaction a count-blind embedding cannot
  hand LightGBM.
* The hop-1 count is each sampled neighbour's own history depth, bounded by the
  neighbour's timestamp (leak-safe: everything counted precedes the neighbour,
  which precedes the target). The target-anchored bound used for hop 2 would be
  degenerate here: card1 is one flat relation, every hop-1 neighbour shares the
  target's entity, and its target-anchored count is the target's count minus
  one for every neighbour alike. Even the own-timestamp count is only the
  neighbour's rank in the shared card timeline -- recency position, not an
  independent history. An independent neighbour history needs a second
  relation, which is Stage 2's question.

The readout is neighbourhood_only, as pre-registered in
configs/g1_v2_preregistration.json: the only readout that reached B0 parity in
the attribution controls, and the downstream model already holds every target
feature.

Cross-fitting shares one initialisation. The earlier cross-fitted control seeded
each fold encoder separately, so an embedding column meant a different direction
either side of the train/validation boundary. Here one random state is
snapshotted once and loaded into the full-train encoder and every fold encoder
before any of them trains, and each records the fingerprint of the state it
started from so the claim can be checked from its metadata.

After embedding, `informativeness_gap` and `cross_fit_provenance` run on the
block and leakage_gate.json is written whatever they find; a block failing
either is marked unusable. Budget, fan-outs, widths, feature cache and fold
protocol are the attribution controls' own, so against `cross_fitted` only the
count inputs, the readout and the shared initialisation change. Nothing here
writes to the frozen encoder, its embeddings or any control artifact.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

from src.config.paths import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from src.graph.build_transaction_graph import ENTITY_EDGES_PATH, NODES_PATH, file_sha256
from src.graph.leakage_checks import cross_fit_provenance, informativeness_gap
from src.graph.temporal_contract import RELATION_NAME
from src.graph.temporal_sampler import TemporalGraphIndex, count_admissible_neighbors
from src.graph.train_graphsage_encoder import (
    AGGREGATOR,
    EMBEDDING_DIM,
    FEATURE_SCALER_PATH,
    HIDDEN_DIM,
    HOP_FAN_OUTS,
    LEARNING_RATE,
    READOUT_MODES,
    READOUT_NEIGHBOURHOOD_ONLY,
    gather_features,
    l2_normalize,
    masked_mean,
    sample_batch_neighborhoods,
)
from src.graph.train_graphsage_variants import (
    CONTROL_BUDGET,
    CONTROL_INFERENCE_BATCH_SIZE,
    CROSS_FIT_FOLDS,
    EncoderBudget,
    EncoderContext,
    assign_cross_fit_folds,
    build_encoder_context,
    embedding_frame,
)
from src.models.compare_g1_controls import embedding_partition_alignment
from src.models.train_lightgbm_baseline import (
    EXPECTED_ROWS,
    MODEL_DATASET_PATH,
    evaluate_validation,
    repository_relative,
    write_json,
)

REPORT_ROOT = REPORTS_DIR / "graphsage" / "card1_v2"

READOUT = READOUT_NEIGHBOURHOOD_ONLY
DEFAULT_SEED = 42

WITH_COUNTS = "with_counts"
COUNT_BLIND = "count_blind"
COUNT_MODES = (WITH_COUNTS, COUNT_BLIND)


@dataclass(frozen=True)
class V2Run:
    """One cross-fitted G1-v2 encoder run and where its artifacts go."""

    seed: int
    count_mode: str = WITH_COUNTS

    def __post_init__(self) -> None:
        if self.count_mode not in COUNT_MODES:
            raise ValueError(f"count_mode must be one of {COUNT_MODES}; got {self.count_mode!r}.")
        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

    @property
    def name(self) -> str:
        prefix = "" if self.count_mode == WITH_COUNTS else f"{COUNT_BLIND}_"
        return f"{prefix}seed{self.seed}"

    @property
    def report_dir(self) -> Path:
        return REPORT_ROOT / self.name

    @property
    def embeddings_path(self) -> Path:
        return PROCESSED_DATA_DIR / f"graphsage_card1_v2_embeddings_{self.name}.parquet"

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / f"graphsage_card1_v2_encoder_{self.name}.pt"

    def fold_model_path(self, fold: int) -> Path:
        return MODELS_DIR / f"graphsage_card1_v2_encoder_{self.name}_fold{fold}.pt"

    @property
    def metadata_path(self) -> Path:
        return self.report_dir / "encoder_metadata.json"

    @property
    def metrics_path(self) -> Path:
        return self.report_dir / "encoder_metrics.json"

    @property
    def training_curve_path(self) -> Path:
        return self.report_dir / "encoder_training_curve.csv"

    @property
    def leakage_gate_path(self) -> Path:
        return self.report_dir / "leakage_gate.json"

    def is_complete(self) -> bool:
        return all(
            path.exists()
            for path in (
                self.embeddings_path,
                self.metadata_path,
                self.metrics_path,
                self.training_curve_path,
                self.leakage_gate_path,
            )
        )


class CardinalityAwareGraphSAGEEncoder(nn.Module):
    """TemporalGraphSAGEEncoder with log1p(true admissible count) fed in at both layers."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        readout: str = READOUT,
    ) -> None:
        super().__init__()
        if readout not in READOUT_MODES:
            raise ValueError(f"readout must be one of {READOUT_MODES}; got {readout!r}.")
        self.readout = readout
        self.layer1 = nn.Linear(input_dim * 2 + 1, hidden_dim)
        self_dim = 0 if readout == READOUT_NEIGHBOURHOOD_ONLY else input_dim
        self.layer2 = nn.Linear(self_dim + hidden_dim + 1, embedding_dim)

    def forward(
        self,
        x_target: torch.Tensor,
        x_hop1: torch.Tensor,
        hop1_mask: torch.Tensor,
        x_hop2: torch.Tensor,
        hop2_mask: torch.Tensor,
        hop1_count: torch.Tensor,
        target_count: torch.Tensor,
    ) -> torch.Tensor:
        agg2 = masked_mean(x_hop2, hop2_mask, dim=2)
        hop1_depth = torch.log1p(hop1_count).unsqueeze(-1)
        h1 = F.relu(self.layer1(torch.cat([x_hop1, agg2, hop1_depth], dim=-1)))
        h1 = l2_normalize(h1, dim=-1)
        h1 = h1 * hop1_mask.unsqueeze(-1).to(h1.dtype)

        agg1 = masked_mean(h1, hop1_mask, dim=1)
        parts = [agg1, torch.log1p(target_count).unsqueeze(-1)]
        if self.readout != READOUT_NEIGHBOURHOOD_ONLY:
            parts.insert(0, x_target)
        embedding = F.relu(self.layer2(torch.cat(parts, dim=-1)))
        return l2_normalize(embedding, dim=-1)


class CardinalityAwareGraphSAGEWithHead(nn.Module):
    """The cardinality-aware encoder plus a disposable linear classification head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        embedding_dim: int,
        readout: str = READOUT,
    ) -> None:
        super().__init__()
        self.encoder = CardinalityAwareGraphSAGEEncoder(
            input_dim, hidden_dim, embedding_dim, readout
        )
        self.head = nn.Linear(embedding_dim, 1)

    def forward(self, **tensors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.encoder(**tensors)
        return embedding, self.head(embedding).squeeze(-1)


def state_fingerprint(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def initial_encoder_state(input_dim: int, readout: str, seed: int) -> dict[str, torch.Tensor]:
    """The one random initialisation every encoder of a cross-fitted run starts from."""
    torch.manual_seed(seed)
    model = CardinalityAwareGraphSAGEWithHead(input_dim, HIDDEN_DIM, EMBEDDING_DIM, readout)
    return copy.deepcopy(model.state_dict())


def sample_batch_neighborhoods_v2(
    index: TemporalGraphIndex,
    target_node_ids: np.ndarray,
    fan_outs: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """`sample_batch_neighborhoods` plus the true counts its fan-out cap hides.

    Sampling is delegated unchanged and counting draws no randomness, so for a
    given generator state the neighbourhood is exactly the frozen encoder's.
    Returns hop-1/hop-2 ids and masks, then hop-1 counts (B, F1) -- zero in
    padded slots -- and target counts (B,).
    """
    hop1_ids, hop1_mask, hop2_ids, hop2_mask = sample_batch_neighborhoods(
        index, target_node_ids, fan_outs, rng
    )
    target_count = np.array(
        [count_admissible_neighbors(index, int(target)) for target in target_node_ids],
        dtype=np.float32,
    )
    hop1_count = np.zeros(hop1_ids.shape, dtype=np.float32)
    for b, j in zip(*np.nonzero(hop1_mask)):
        hop1_count[b, j] = count_admissible_neighbors(index, int(hop1_ids[b, j]))
    return hop1_ids, hop1_mask, hop2_ids, hop2_mask, hop1_count, target_count


def build_batch_tensors_v2(
    index: TemporalGraphIndex,
    feature_cache: np.ndarray,
    target_node_ids: np.ndarray,
    fan_outs: tuple[int, int],
    rng: np.random.Generator,
    count_mode: str = WITH_COUNTS,
) -> dict[str, torch.Tensor]:
    if count_mode not in COUNT_MODES:
        raise ValueError(f"count_mode must be one of {COUNT_MODES}; got {count_mode!r}.")
    hop1_ids, hop1_mask, hop2_ids, hop2_mask, hop1_count, target_count = (
        sample_batch_neighborhoods_v2(index, target_node_ids, fan_outs, rng)
    )
    if count_mode == COUNT_BLIND:
        hop1_count = np.zeros_like(hop1_count)
        target_count = np.zeros_like(target_count)
    return {
        "x_target": torch.from_numpy(gather_features(feature_cache, target_node_ids)),
        "x_hop1": torch.from_numpy(gather_features(feature_cache, hop1_ids, mask=hop1_mask)),
        "hop1_mask": torch.from_numpy(hop1_mask),
        "x_hop2": torch.from_numpy(gather_features(feature_cache, hop2_ids, mask=hop2_mask)),
        "hop2_mask": torch.from_numpy(hop2_mask),
        "hop1_count": torch.from_numpy(hop1_count),
        "target_count": torch.from_numpy(target_count),
    }


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))


def run_forward_v2(
    model: CardinalityAwareGraphSAGEWithHead,
    index: TemporalGraphIndex,
    feature_cache: np.ndarray,
    node_ids: np.ndarray,
    rng: np.random.Generator,
    batch_size: int,
    count_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward-only (embeddings, logits) for `node_ids`, batched, no gradient tracked."""
    model.eval()
    embeddings = np.empty((len(node_ids), EMBEDDING_DIM), dtype=np.float32)
    logits = np.empty(len(node_ids), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(node_ids), batch_size):
            batch_ids = node_ids[start : start + batch_size]
            tensors = build_batch_tensors_v2(
                index, feature_cache, batch_ids, HOP_FAN_OUTS, rng, count_mode
            )
            embedding, logit = model(**tensors)
            embeddings[start : start + len(batch_ids)] = embedding.numpy()
            logits[start : start + len(batch_ids)] = logit.numpy()
    return embeddings, logits


def train_encoder_v2(
    ctx: EncoderContext,
    *,
    initial_state: dict[str, torch.Tensor],
    readout: str,
    budget: EncoderBudget,
    seed: int,
    label: str,
    target_node_ids: np.ndarray,
    count_mode: str,
    fixed_epochs: int | None = None,
    inference_batch_size: int = CONTROL_INFERENCE_BATCH_SIZE,
) -> dict[str, Any]:
    """Train one encoder+head from `initial_state` on `target_node_ids`.

    `fixed_epochs=None` scores every epoch on the whole validation partition and
    restores the best one; `fixed_epochs=k` trains exactly k epochs unmonitored,
    the schedule the cross-fitting folds use.
    """
    rng = np.random.default_rng(seed)
    model = CardinalityAwareGraphSAGEWithHead(
        len(ctx.feature_columns), HIDDEN_DIM, EMBEDDING_DIM, readout
    )
    model.load_state_dict(initial_state)
    initial_fingerprint = state_fingerprint(model.state_dict())
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ctx.pos_weight, dtype=torch.float32))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    monitored = fixed_epochs is None
    n_epochs = budget.max_epochs if monitored else int(fixed_epochs)
    if not 1 <= n_epochs <= budget.max_epochs:
        raise ValueError("A schedule must run between one epoch and the budget's cap.")
    validation_labels = ctx.labels_by_node_id[ctx.validation_node_ids]

    best_state: dict[str, Any] | None = None
    best_val_pr_auc = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0
    curve_rows: list[dict[str, Any]] = []
    stopped_epoch = 0

    schedule = (
        f"up to {n_epochs} monitored epochs, patience {budget.patience}"
        if monitored
        else f"a fixed {n_epochs}-epoch schedule (no monitor)"
    )
    print(
        f"[{label}] Training: {budget.steps_per_epoch} steps/epoch x {budget.batch_size} "
        f"targets, {schedule}, {len(target_node_ids):,} eligible targets, counts={count_mode}..."
    )
    for epoch in range(1, n_epochs + 1):
        stopped_epoch = epoch
        model.train()
        epoch_losses: list[float] = []
        for _ in range(budget.steps_per_epoch):
            batch_ids = rng.choice(target_node_ids, size=budget.batch_size, replace=False)
            tensors = build_batch_tensors_v2(
                ctx.index, ctx.feature_cache, batch_ids, HOP_FAN_OUTS, rng, count_mode
            )
            y = torch.from_numpy(ctx.labels_by_node_id[batch_ids].astype(np.float32))
            optimizer.zero_grad()
            _, logit = model(**tensors)
            loss = criterion(logit, y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(epoch_losses))
        if not monitored:
            curve_rows.append(
                {"epoch": epoch, "train_loss_mean": mean_loss, "validation_pr_auc_monitor": np.nan}
            )
            print(f"[{label}] epoch {epoch:>2d}: train_loss={mean_loss:.6f}")
            continue

        _, monitor_logits = run_forward_v2(
            model,
            ctx.index,
            ctx.feature_cache,
            ctx.validation_node_ids,
            rng,
            inference_batch_size,
            count_mode,
        )
        val_pr_auc = float(average_precision_score(validation_labels, _sigmoid(monitor_logits)))
        curve_rows.append(
            {"epoch": epoch, "train_loss_mean": mean_loss, "validation_pr_auc_monitor": val_pr_auc}
        )
        print(
            f"[{label}] epoch {epoch:>2d}: train_loss={mean_loss:.6f} "
            f"validation_pr_auc(full)={val_pr_auc:.6f}"
        )
        if val_pr_auc > best_val_pr_auc + budget.min_delta:
            best_val_pr_auc = val_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= budget.patience:
                print(f"[{label}] Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    if monitored:
        if best_state is None:
            raise RuntimeError(f"[{label}] Training never improved on the validation monitor.")
        model.load_state_dict(best_state)
        epoch_cap_reached = (
            stopped_epoch == budget.max_epochs and epochs_without_improvement < budget.patience
        )
    else:
        best_epoch = stopped_epoch
        best_val_pr_auc = float("nan")
        epoch_cap_reached = False

    return {
        "model": model,
        "label": label,
        "curve_rows": curve_rows,
        "best_epoch": int(best_epoch),
        "stopped_epoch": int(stopped_epoch),
        "best_validation_pr_auc_monitor": float(best_val_pr_auc),
        "epoch_cap_reached": bool(epoch_cap_reached),
        "early_stopping_triggered": bool(monitored and not epoch_cap_reached),
        "monitored": bool(monitored),
        "target_row_count": int(len(target_node_ids)),
        "targets_drawn": int(budget.targets_per_epoch * stopped_epoch),
        "initial_state_sha256": initial_fingerprint,
    }


def fit_cross_fitted_encoders(
    ctx: EncoderContext,
    *,
    seed: int,
    count_mode: str,
    label: str,
    readout: str = READOUT,
    budget: EncoderBudget = CONTROL_BUDGET,
    n_folds: int = CROSS_FIT_FOLDS,
    inference_batch_size: int = CONTROL_INFERENCE_BATCH_SIZE,
) -> dict[str, Any]:
    """Full-train encoder for validation/test rows, fold encoders for train rows, one shared init.

    Fold encoders train for the full-train encoder's selected epoch count rather
    than early-stopping on the shared validation monitor themselves, the same
    schedule the earlier cross-fitted control used.
    """
    initial_state = initial_encoder_state(len(ctx.feature_columns), readout, seed)
    shared_fingerprint = state_fingerprint(initial_state)
    common: dict[str, Any] = {
        "initial_state": initial_state,
        "readout": readout,
        "budget": budget,
        "count_mode": count_mode,
        "inference_batch_size": inference_batch_size,
    }

    full = train_encoder_v2(
        ctx,
        seed=seed,
        label=f"{label}:full_train",
        target_node_ids=ctx.train_node_ids,
        **common,
    )
    _, validation_logits = run_forward_v2(
        full["model"],
        ctx.index,
        ctx.feature_cache,
        ctx.validation_node_ids,
        np.random.default_rng(seed + 1),
        inference_batch_size,
        count_mode,
    )
    y_validation = pd.Series(ctx.labels_by_node_id[ctx.validation_node_ids].astype("int8"))
    metrics = evaluate_validation(y_validation, _sigmoid(validation_logits), run_name=label)
    metrics["model"] = f"{label}_encoder_with_head"

    fold_epochs = int(full["best_epoch"])
    folds = assign_cross_fit_folds(ctx.train_node_ids, n_folds, seed)
    embeddings = np.zeros((len(ctx.meta), EMBEDDING_DIM), dtype=np.float32)
    embedded = np.zeros(len(ctx.meta), dtype=bool)

    inference_ids = np.concatenate([ctx.validation_node_ids, ctx.test_node_ids])
    print(f"[{label}:full_train] Embedding {len(inference_ids):,} validation/test transactions...")
    inference_embeddings, _ = run_forward_v2(
        full["model"],
        ctx.index,
        ctx.feature_cache,
        inference_ids,
        np.random.default_rng(seed + 2),
        inference_batch_size,
        count_mode,
    )
    embeddings[inference_ids] = inference_embeddings
    embedded[inference_ids] = True

    fold_results: list[dict[str, Any]] = []
    for fold in range(n_folds):
        held_out = ctx.train_node_ids[folds == fold]
        fit_on = ctx.train_node_ids[folds != fold]
        fold_seed = seed + 100 + fold
        result = train_encoder_v2(
            ctx,
            seed=fold_seed,
            label=f"{label}:fold{fold}",
            target_node_ids=fit_on,
            fixed_epochs=fold_epochs,
            **common,
        )
        print(f"[{label}:fold{fold}] Embedding {len(held_out):,} held-out train transactions...")
        fold_embeddings, _ = run_forward_v2(
            result["model"],
            ctx.index,
            ctx.feature_cache,
            held_out,
            np.random.default_rng(fold_seed + 2),
            inference_batch_size,
            count_mode,
        )
        embeddings[held_out] = fold_embeddings
        embedded[held_out] = True
        result["held_out_rows"] = int(len(held_out))
        fold_results.append(result)

    if not embedded.all():
        raise AssertionError(f"{int((~embedded).sum()):,} nodes were left unembedded.")
    fold_sizes = [int((folds == fold).sum()) for fold in range(n_folds)]
    held_out_total = sum(result["held_out_rows"] for result in fold_results)
    if sum(fold_sizes) != len(ctx.train_node_ids) or held_out_total != len(ctx.train_node_ids):
        raise AssertionError("Cross-fit folds do not partition the train partition.")
    started_from = {
        full["initial_state_sha256"],
        *(r["initial_state_sha256"] for r in fold_results),
    }
    if started_from != {shared_fingerprint}:
        raise AssertionError("An encoder did not start from the shared initialisation.")

    cross_fitting = {
        "n_folds": n_folds,
        "fold_assignment": "fixed_seed_permutation_unstratified",
        "fold_assignment_seed": seed,
        "fold_sizes": fold_sizes,
        "fold_schedule": "fixed_epochs_from_full_train_best_epoch",
        "fold_epochs": fold_epochs,
        "train_rows_embedded_by_held_out_encoder": int(held_out_total),
        "inference_rows_embedded_by_full_train_encoder": int(len(inference_ids)),
        "shared_initialisation": True,
        "initial_state_sha256": shared_fingerprint,
        "feature_scaler_fit_scope": "full_train_partition",
        "feature_scaler_note": (
            "Standardization statistics are fitted once on the whole train partition "
            "and shared by every fold encoder. This is label-free preprocessing and "
            "matches the frozen encoder; it is recorded rather than treated as "
            "cross-fitted."
        ),
    }
    return {
        "full": full,
        "folds": fold_results,
        "metrics": metrics,
        "embeddings": embeddings,
        "cross_fitting": cross_fitting,
    }


def build_leakage_gate(
    ctx: EncoderContext,
    embeddings: np.ndarray,
    cross_fitting: dict[str, Any],
    embeddings_path: Path,
) -> dict[str, Any]:
    labels = ctx.labels_by_node_id
    gap = informativeness_gap(
        embeddings[ctx.train_node_ids],
        labels[ctx.train_node_ids].astype(np.int8),
        embeddings[ctx.validation_node_ids],
        labels[ctx.validation_node_ids].astype(np.int8),
    )
    provenance = cross_fit_provenance(
        cross_fitting,
        int(len(ctx.train_node_ids)),
        int(len(ctx.validation_node_ids) + len(ctx.test_node_ids)),
    )
    return {
        "usable": bool(gap["parity_holds"]) and bool(provenance["provenance_sound"]),
        "rule": (
            "Usable only if informativeness_gap parity holds and cross_fit_provenance "
            "is sound. A block that is not usable is not credited by the verdict."
        ),
        "informativeness_gap": gap,
        "cross_fit_provenance": provenance,
        "partition_alignment_diagnostic": {
            **embedding_partition_alignment(embeddings_path),
            "gating": False,
            "reference": (
                "embedding_partition_alignment in reports/g1_controls/"
                "g1_control_summary.json: single-encoder blocks versus the unaligned "
                "cross-fitted block"
            ),
        },
        "embeddings_path": repository_relative(embeddings_path),
        "embeddings_sha256": file_sha256(embeddings_path),
    }


def build_encoder_v2_metadata(
    ctx: EncoderContext,
    run: V2Run,
    outcome: dict[str, Any],
    gate: dict[str, Any],
) -> dict[str, Any]:
    full = outcome["full"]
    encoder = full["model"].encoder
    encoders = [
        {
            "role": "full_train",
            "model_path": repository_relative(run.model_path),
            "train_target_rows": full["target_row_count"],
            "best_epoch": full["best_epoch"],
            "stopped_epoch": full["stopped_epoch"],
            "monitored": True,
            "initial_state_sha256": full["initial_state_sha256"],
            "embedded_splits": ["validation", "test"],
        }
    ]
    for fold, result in enumerate(outcome["folds"]):
        encoders.append(
            {
                "role": f"fold_{fold}",
                "model_path": repository_relative(run.fold_model_path(fold)),
                "train_target_rows": result["target_row_count"],
                "held_out_rows": result["held_out_rows"],
                "best_epoch": result["best_epoch"],
                "stopped_epoch": result["stopped_epoch"],
                "monitored": False,
                "initial_state_sha256": result["initial_state_sha256"],
                "embedded_splits": [f"train_fold_{fold}"],
            }
        )
    return {
        "relation_name": RELATION_NAME,
        "model_name": f"graphsage_card1_v2_encoder_{run.name}",
        "stage": "g1_v2_stage1_cardinality_aware",
        "random_seed": run.seed,
        "count_mode": run.count_mode,
        "architecture": {
            "n_hops": len(HOP_FAN_OUTS),
            "fan_outs": list(HOP_FAN_OUTS),
            "aggregator": AGGREGATOR,
            "hidden_dim": HIDDEN_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "activation": "relu",
            "normalization": "l2_per_layer",
            "readout": encoder.readout,
            "input_dim": len(ctx.feature_columns),
            "layer1_input_dim": encoder.layer1.in_features,
            "layer2_input_dim": encoder.layer2.in_features,
            "cardinality_inputs": {
                "transform": "log1p",
                "target": (
                    "count_admissible_neighbors(target) under the target's own "
                    "timestamp; equal to B1-card1's prior_count"
                ),
                "hop1": (
                    "count_admissible_neighbors(neighbour) under the neighbour's own "
                    "timestamp; its rank in the shared card timeline"
                ),
                "zeroed": run.count_mode == COUNT_BLIND,
            },
            "empty_neighborhood_policy": (
                "masked mean over zero valid neighbours is the zero vector, and a "
                "padded hop-1 slot carries a zero count"
            ),
        },
        "training_objective": (
            "supervised_isFraud_via_disposable_head; only the encoder's embedding "
            "output is persisted, the classification head is discarded after training"
        ),
        "training_regime": "inductive",
        "target_labels_used_for_training": True,
        "target_labels_used_in_node_features": False,
        "test_labels_used": False,
        "mini_batch_budget": {
            **CONTROL_BUDGET.describe(len(ctx.train_node_ids)),
            "targets_drawn_by_full_train_encoder": full["targets_drawn"],
        },
        "early_stopping": {
            "metric": "validation_pr_auc",
            "monitor_evaluation_set": "full_validation_partition",
            "patience": CONTROL_BUDGET.patience,
            "min_delta": CONTROL_BUDGET.min_delta,
            "max_epochs": CONTROL_BUDGET.max_epochs,
            "stopped_epoch": full["stopped_epoch"],
            "best_epoch": full["best_epoch"],
            "best_validation_pr_auc_monitor": full["best_validation_pr_auc_monitor"],
            "estimator_cap_reached": full["epoch_cap_reached"],
            "early_stopping_triggered": full["early_stopping_triggered"],
        },
        "cross_fitting": outcome["cross_fitting"],
        "encoders": encoders,
        "standalone_validation_pr_auc": float(outcome["metrics"]["pr_auc"]),
        "leakage_gate": {
            "usable": gate["usable"],
            "path": repository_relative(run.leakage_gate_path),
        },
        "class_weighting": {
            "train_fraud_count": ctx.train_fraud_count,
            "train_rows": int(len(ctx.train_node_ids)),
            "pos_weight": ctx.pos_weight,
        },
        "split_row_counts": {
            "train": int(len(ctx.train_node_ids)),
            "validation": int(len(ctx.validation_node_ids)),
            "test": int(len(ctx.test_node_ids)),
        },
        "input_model_dataset_sha256": file_sha256(MODEL_DATASET_PATH),
        "input_nodes_sha256": file_sha256(NODES_PATH),
        "input_entity_edges_sha256": file_sha256(ENTITY_EDGES_PATH),
        "feature_scaler_sha256": file_sha256(FEATURE_SCALER_PATH),
        "feature_column_count": len(ctx.feature_columns),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "embeddings_path": repository_relative(run.embeddings_path),
        "embedding_row_count": EXPECTED_ROWS,
        "embedding_has_no_nulls": True,
        "embedding_has_no_target_column": True,
        "model_path": repository_relative(run.model_path),
        "metrics_path": repository_relative(run.metrics_path),
        "training_curve_path": repository_relative(run.training_curve_path),
    }


def run_encoder_v2(
    run: V2Run,
    ctx: EncoderContext | None = None,
    skip_existing: bool = False,
) -> None:
    if skip_existing and run.is_complete():
        print(f"[v2/{run.name}] Already complete; skipping.")
        return
    if ctx is None:
        ctx = build_encoder_context()

    run.report_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    run.embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    label = f"v2/{run.name}"
    outcome = fit_cross_fitted_encoders(ctx, seed=run.seed, count_mode=run.count_mode, label=label)

    torch.save(outcome["full"]["model"].state_dict(), run.model_path)
    for fold, result in enumerate(outcome["folds"]):
        torch.save(result["model"].state_dict(), run.fold_model_path(fold))
    embedding_frame(ctx.meta, outcome["embeddings"]).to_parquet(
        run.embeddings_path, index=False, engine="pyarrow", compression="snappy"
    )
    curve_rows = [{"encoder": "full_train", **row} for row in outcome["full"]["curve_rows"]]
    for fold, result in enumerate(outcome["folds"]):
        curve_rows.extend({"encoder": f"fold_{fold}", **row} for row in result["curve_rows"])
    pd.DataFrame(curve_rows).to_csv(run.training_curve_path, index=False)
    write_json(run.metrics_path, outcome["metrics"])

    print(f"[{label}] Running the leakage gate...")
    gate = build_leakage_gate(
        ctx, outcome["embeddings"], outcome["cross_fitting"], run.embeddings_path
    )
    write_json(run.leakage_gate_path, gate)
    write_json(run.metadata_path, build_encoder_v2_metadata(ctx, run, outcome, gate))

    gap = gate["informativeness_gap"]
    alignment = gate["partition_alignment_diagnostic"]
    print(f"[{label}] Best epoch: {outcome['full']['best_epoch']} (folds trained that many)")
    print(f"[{label}] Standalone validation PR-AUC: {outcome['metrics']['pr_auc']:.6f}")
    print(
        f"[{label}] Probe ROC-AUC train {gap['train_probe_roc_auc']:.4f} vs validation "
        f"{gap['validation_probe_roc_auc']:.4f} (gap {gap['gap']:+.4f})"
    )
    print(f"[{label}] Partition alignment mean gap: {alignment['mean_gap']:.3f} (diagnostic)")
    print(f"[{label}] Leakage gate: {'USABLE' if gate['usable'] else 'NOT USABLE'}")
    print(f"[{label}] Embeddings saved: {run.embeddings_path}")
    print(f"[{label}] Test labels used: NO")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.graph.train_graphsage_encoder_v2",
        description="Train one cross-fitted, cardinality-aware G1-v2 encoder.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--count-blind",
        action="store_true",
        help="Zero every count input: the pre-registered attribution arm.",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    run = V2Run(seed=args.seed, count_mode=COUNT_BLIND if args.count_blind else WITH_COUNTS)
    run_encoder_v2(run, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
