"""Encoder variants for the G1 attribution controls.

The first G1 run scored below both B0 and B1-card1. Three properties of that
run each offer an explanation for the gap that has nothing to do with graph
structure being unhelpful, and the frozen run cannot tell them apart:

1. The embedding is not a relational feature block. The encoder's input is the
   same 435 raw predictors LightGBM already receives, and the standard
   GraphSAGE readout concatenates the target's own transformed features onto
   the aggregated neighbourhood. The persisted block therefore mixes signal the
   downstream model already holds with whatever the neighbourhood adds.
2. The encoder is fit on the same rows the downstream model trains on, with a
   supervised objective and no cross-fitting. Train-row embeddings carry
   information from those rows' own labels; validation-row embeddings do not.
   A block whose informativeness does not transfer is exactly what the frozen
   run's importance table shows.
3. The encoder is barely trained: best epoch 4, ~0.40 passes over the train
   partition for the persisted weights, early-stopped on a 15,000-row
   validation subsample after four epochs each worth a tenth of a pass.

This module produces the encoder-side controls that separate those confounds
from the graph verdict. Every control shares one budget
(`CONTROL_BUDGET`), so a control's effect is read against `extended_budget`
rather than against the frozen run, and only one thing changes at a time:

* `extended_budget`    -- frozen architecture and fitting, raised budget,
                          selection on the full validation partition. Isolates
                          confound 3, and is the reference the other two are
                          read against.
* `neighbourhood_only` -- `extended_budget` plus a readout that drops the
                          target's own features. Isolates confound 1.
* `cross_fitted`       -- `extended_budget` plus K-fold cross-fitting over the
                          train partition. Isolates confound 2.

The shuffled-embedding control needs no encoder at all and lives with the
downstream runner (src/models/train_lightgbm_g1_controls.py).

Nothing here touches the frozen encoder, its embeddings, or any B0/B1/G1
artifact: every variant writes its own model, embedding table and report
directory.

Cross-fitting protocol
------------------------
K disjoint folds over the train partition, assigned by a fixed-seed
permutation. Fold k's rows are embedded by an encoder trained on the other
K-1 folds; validation and test rows are embedded by an encoder trained on the
whole train partition. Train-time and inference-time embeddings then carry
comparable informativeness, which is the property the frozen run lacks.

The fold encoders train for a *fixed* number of epochs -- the best epoch the
full-train encoder selected -- rather than early-stopping individually. Two
reasons: selecting a stopping point per fold on the shared validation
partition would let K models each peek at the same monitor, and a fixed
schedule keeps the K+1 encoders comparable in budget. It also removes K
full-validation monitor passes, which dominate wall-clock here.

Known, deliberate limitation: the standardization statistics in the shared
feature cache (`prepare_feature_cache`) are fitted on the full train
partition, so a fold encoder's *inputs* are scaled using statistics that saw
its held-out fold. This is unsupervised, label-free preprocessing, it is
identical to what the frozen encoder does, and rebuilding a 435-column cache
per fold would cost more than the control is worth. It is recorded in the
variant metadata rather than left implicit.
"""

from __future__ import annotations

import copy
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

from src.config.paths import ROOT_DIR
from src.graph.build_transaction_graph import (
    ENTITY_EDGES_PATH,
    NODES_PATH,
    file_sha256,
)
from src.graph.temporal_contract import RELATION_NAME
from src.graph.temporal_sampler import (
    TemporalGraphIndex,
    build_temporal_graph_index,
)
from src.graph.train_graphsage_encoder import (
    AGGREGATOR,
    EMBEDDING_DIM,
    FEATURE_CACHE_PATH,
    FEATURE_SCALER_PATH,
    HIDDEN_DIM,
    HOP_FAN_OUTS,
    LEARNING_RATE,
    READOUT_MODES,
    READOUT_NEIGHBOURHOOD_ONLY,
    READOUT_SELF_AND_NEIGHBOURHOOD,
    GraphSAGEWithHead,
    build_batch_tensors,
    get_feature_columns,
    load_labels,
    load_node_metadata,
    prepare_feature_cache,
    run_inference,
    run_logits,
)
from src.models.train_lightgbm_baseline import (
    EXPECTED_ROWS,
    MODEL_DATASET_PATH,
    evaluate_validation,
    repository_relative,
    write_json,
)

CONTROL_REPORT_ROOT = ROOT_DIR / "reports" / "g1_controls"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODELS_DIR = ROOT_DIR / "models"

RANDOM_SEED = 42


@dataclass(frozen=True)
class EncoderBudget:
    """One encoder's training budget, stated in the unit that actually matters.

    `steps_per_epoch * batch_size * max_epochs` targets is the honest ceiling;
    "epochs" alone is meaningless here because an epoch is a stochastic
    mini-batch budget, not a pass over the train partition. Both are recorded.
    """

    steps_per_epoch: int
    batch_size: int
    max_epochs: int
    patience: int
    min_delta: float
    monitor: str

    def __post_init__(self) -> None:
        if min(self.steps_per_epoch, self.batch_size, self.max_epochs, self.patience) <= 0:
            raise ValueError("Budget sizes and patience must all be positive.")
        if self.min_delta < 0.0:
            raise ValueError("min_delta must be non-negative.")
        if self.monitor != "full_validation_partition":
            raise ValueError(
                "The controls select on the full validation partition; "
                f"got monitor={self.monitor!r}."
            )

    @property
    def targets_per_epoch(self) -> int:
        return self.steps_per_epoch * self.batch_size

    def describe(self, train_rows: int) -> dict[str, Any]:
        return {
            "steps_per_epoch": self.steps_per_epoch,
            "batch_size": self.batch_size,
            "targets_per_epoch": self.targets_per_epoch,
            "max_epochs": self.max_epochs,
            "max_targets": self.targets_per_epoch * self.max_epochs,
            "train_partition_rows": int(train_rows),
            "passes_per_epoch": self.targets_per_epoch / train_rows,
            "max_passes": self.targets_per_epoch * self.max_epochs / train_rows,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "monitor_evaluation_set": self.monitor,
        }


# One budget shared by every encoder control, so a control's effect is read
# against `extended_budget` and not against the frozen run's under-training.
# Against the frozen encoder this is 2x the targets per epoch, a cap of 10 such
# epochs instead of 15 small ones, a patience worth twice as many targets, and
# selection on all 88,581 validation rows instead of a 15,000-row subsample --
# a ceiling of ~2.0 passes over the train partition against the 0.40 passes
# behind the persisted frozen weights.
CONTROL_BUDGET = EncoderBudget(
    steps_per_epoch=320,
    batch_size=256,
    max_epochs=10,
    patience=4,
    min_delta=1e-4,
    monitor="full_validation_partition",
)

CROSS_FIT_FOLDS = 3

# Inference batches are memory-bound, not compute-bound: a batch of B targets
# materializes a (B, 10, 10, 435) float32 hop-2 feature block, and the masked
# mean over it allocates a second one of the same size. At the frozen encoder's
# 4,096 that is ~1.4GB of transient allocation per batch, which is enough to
# push this machine into swap while three encoders and a 6,000-round LightGBM
# run share it. 1,024 keeps the peak near 350MB at essentially the same
# throughput -- the per-row cost is dominated by the sampler, not by batching.
CONTROL_INFERENCE_BATCH_SIZE = 1_024


@dataclass(frozen=True)
class EncoderVariant:
    name: str
    readout: str
    cross_fit_folds: int | None
    budget: EncoderBudget
    seed: int
    isolates: str
    description: str

    def __post_init__(self) -> None:
        if self.readout not in READOUT_MODES:
            raise ValueError(f"readout must be one of {READOUT_MODES}; got {self.readout!r}.")
        if self.cross_fit_folds is not None and self.cross_fit_folds < 2:
            raise ValueError("cross_fit_folds must be at least 2 when cross-fitting.")

    @property
    def report_dir(self) -> Path:
        return CONTROL_REPORT_ROOT / self.name

    @property
    def embeddings_path(self) -> Path:
        return PROCESSED_DIR / f"graphsage_card1_embeddings_{self.name}.parquet"

    @property
    def model_path(self) -> Path:
        return MODELS_DIR / f"graphsage_card1_encoder_{self.name}.pt"

    def fold_model_path(self, fold: int) -> Path:
        return MODELS_DIR / f"graphsage_card1_encoder_{self.name}_fold{fold}.pt"

    @property
    def metadata_path(self) -> Path:
        return self.report_dir / "encoder_metadata.json"

    @property
    def metrics_path(self) -> Path:
        return self.report_dir / "encoder_metrics.json"

    @property
    def training_curve_path(self) -> Path:
        return self.report_dir / "encoder_training_curve.csv"


ENCODER_VARIANTS: dict[str, EncoderVariant] = {
    variant.name: variant
    for variant in (
        EncoderVariant(
            name="extended_budget",
            readout=READOUT_SELF_AND_NEIGHBOURHOOD,
            cross_fit_folds=None,
            budget=CONTROL_BUDGET,
            seed=RANDOM_SEED,
            isolates="encoder training budget and monitor size",
            description=(
                "Frozen G1 architecture and fitting procedure, retrained under the "
                "raised control budget with the stopping point selected on the full "
                "validation partition. Reference run for the other encoder controls."
            ),
        ),
        EncoderVariant(
            name="neighbourhood_only",
            readout=READOUT_NEIGHBOURHOOD_ONLY,
            cross_fit_folds=None,
            budget=CONTROL_BUDGET,
            seed=RANDOM_SEED,
            isolates="self-contribution in the readout",
            description=(
                "Control budget with the target's own features dropped from the final "
                "readout, so the embedding carries what the tabular model does not "
                "already hold rather than a re-encoding of the same 435 predictors."
            ),
        ),
        EncoderVariant(
            name="cross_fitted",
            readout=READOUT_SELF_AND_NEIGHBOURHOOD,
            cross_fit_folds=CROSS_FIT_FOLDS,
            budget=CONTROL_BUDGET,
            seed=RANDOM_SEED,
            isolates="label information in train-row embeddings",
            description=(
                "Control budget with K-fold cross-fitting over the train partition: "
                "each train fold is embedded by an encoder that never saw its labels, "
                "validation and test by a full-train encoder."
            ),
        ),
    )
}


@dataclass
class EncoderContext:
    """Everything an encoder run needs that is identical across variants."""

    meta: pd.DataFrame
    feature_columns: list[str]
    index: TemporalGraphIndex
    feature_cache: np.ndarray
    labels_by_node_id: np.ndarray
    train_node_ids: np.ndarray
    validation_node_ids: np.ndarray
    test_node_ids: np.ndarray
    all_node_ids: np.ndarray
    pos_weight: float
    train_fraud_count: int


def build_encoder_context() -> EncoderContext:
    print(f"[{RELATION_NAME}] Loading node metadata and entity-edge index...")
    meta = load_node_metadata()
    feature_columns = get_feature_columns()
    train_mask = (meta["split"] == "train").to_numpy()

    edges_df = pd.read_parquet(ENTITY_EDGES_PATH, columns=["entity_id", "node_id", "TransactionDT"])
    index = build_temporal_graph_index(edges_df)
    del edges_df

    print(
        f"[{RELATION_NAME}] Preparing the standardized feature cache "
        f"({len(feature_columns)} predictors)..."
    )
    prepare_feature_cache(feature_columns, train_mask)
    feature_cache = np.load(FEATURE_CACHE_PATH, mmap_mode="r")
    if feature_cache.shape != (len(meta), len(feature_columns)):
        raise AssertionError("Feature cache shape does not match the node table.")

    labels_by_node_id = load_labels(meta)
    train_node_ids = meta.loc[train_mask, "node_id"].to_numpy()
    validation_node_ids = meta.loc[meta["split"] == "validation", "node_id"].to_numpy()
    test_node_ids = meta.loc[meta["split"] == "test", "node_id"].to_numpy()

    train_fraud_count = int(labels_by_node_id[train_node_ids].sum())
    n_negative = len(train_node_ids) - train_fraud_count
    return EncoderContext(
        meta=meta,
        feature_columns=feature_columns,
        index=index,
        feature_cache=feature_cache,
        labels_by_node_id=labels_by_node_id,
        train_node_ids=train_node_ids,
        validation_node_ids=validation_node_ids,
        test_node_ids=test_node_ids,
        all_node_ids=meta["node_id"].to_numpy(),
        pos_weight=float(n_negative) / float(train_fraud_count),
        train_fraud_count=train_fraud_count,
    )


def assign_cross_fit_folds(train_node_ids: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold index per entry of `train_node_ids`, a fixed-seed disjoint partition.

    Deliberately *not* stratified by label or by entity. Stratifying on the
    label would let fold membership depend on the very quantity cross-fitting
    exists to keep out of the held-out rows' embeddings, and grouping by
    entity would change what a fold encoder can see through the graph rather
    than only which rows' labels it trained on -- a second change on top of the
    one this control is isolating.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be at least 2.")
    if n_folds > len(train_node_ids):
        raise ValueError("n_folds cannot exceed the number of train rows.")
    rng = np.random.default_rng(seed)
    shuffled_positions = rng.permutation(len(train_node_ids))
    folds = np.empty(len(train_node_ids), dtype=np.int64)
    folds[shuffled_positions] = np.arange(len(train_node_ids)) % n_folds
    return folds


def train_single_encoder(
    ctx: EncoderContext,
    *,
    readout: str,
    budget: EncoderBudget,
    seed: int,
    label: str,
    target_node_ids: np.ndarray,
    fixed_epochs: int | None = None,
) -> dict[str, Any]:
    """Train one encoder+head on `target_node_ids`.

    `fixed_epochs=None` runs the full monitored schedule: every epoch is scored
    on the whole validation partition and the best-scoring epoch's weights are
    restored. `fixed_epochs=k` trains exactly k epochs with no monitor at all,
    the schedule the cross-fitting folds use.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(ctx.pos_weight, dtype=torch.float32))
    model = GraphSAGEWithHead(len(ctx.feature_columns), HIDDEN_DIM, EMBEDDING_DIM, readout)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    monitored = fixed_epochs is None
    n_epochs = budget.max_epochs if monitored else fixed_epochs
    if not monitored and fixed_epochs > budget.max_epochs:
        raise ValueError("A fixed schedule cannot exceed the budget's epoch cap.")

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
        f"[{label}] Training: {budget.steps_per_epoch} steps/epoch x "
        f"{budget.batch_size} targets, {schedule}, "
        f"{len(target_node_ids):,} eligible targets..."
    )
    for epoch in range(1, n_epochs + 1):
        stopped_epoch = epoch
        model.train()
        epoch_losses: list[float] = []
        for _ in range(budget.steps_per_epoch):
            batch_ids = rng.choice(target_node_ids, size=budget.batch_size, replace=False)
            tensors = build_batch_tensors(
                ctx.index, ctx.feature_cache, batch_ids, HOP_FAN_OUTS, rng
            )
            y = torch.from_numpy(ctx.labels_by_node_id[batch_ids].astype(np.float32))

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

        mean_loss = float(np.mean(epoch_losses))
        if not monitored:
            curve_rows.append(
                {
                    "epoch": epoch,
                    "train_loss_mean": mean_loss,
                    "validation_pr_auc_monitor": np.nan,
                }
            )
            print(f"[{label}] epoch {epoch:>2d}: train_loss={mean_loss:.6f}")
            continue

        monitor_logits = run_logits(
            model,
            ctx.index,
            ctx.feature_cache,
            ctx.validation_node_ids,
            HOP_FAN_OUTS,
            rng,
            CONTROL_INFERENCE_BATCH_SIZE,
        )
        monitor_probs = 1.0 / (1.0 + np.exp(-monitor_logits))
        val_pr_auc = float(
            average_precision_score(ctx.labels_by_node_id[ctx.validation_node_ids], monitor_probs)
        )
        curve_rows.append(
            {
                "epoch": epoch,
                "train_loss_mean": mean_loss,
                "validation_pr_auc_monitor": val_pr_auc,
            }
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
    }


def standalone_validation_metrics(
    ctx: EncoderContext, model: GraphSAGEWithHead, seed: int, run_name: str
) -> dict[str, Any]:
    """The encoder+head's own validation ranking quality, full partition.

    Reported for the same reason the frozen run reports it: an encoder that
    ranks fraud far worse than LightGBM standalone is evidence about the
    encoder, and it is the number the budget control is trying to move.
    """
    rng = np.random.default_rng(seed + 1)
    logits = run_logits(
        model,
        ctx.index,
        ctx.feature_cache,
        ctx.validation_node_ids,
        HOP_FAN_OUTS,
        rng,
        CONTROL_INFERENCE_BATCH_SIZE,
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_validation = pd.Series(ctx.labels_by_node_id[ctx.validation_node_ids].astype("int8"))
    metrics = evaluate_validation(y_validation, probs, run_name=run_name)
    metrics["model"] = f"{run_name}_encoder_with_head"
    return metrics


def embed_nodes(
    ctx: EncoderContext,
    model: GraphSAGEWithHead,
    node_ids: np.ndarray,
    seed: int,
    label: str,
) -> np.ndarray:
    print(f"[{label}] Extracting embeddings for {len(node_ids):,} transactions...")
    rng = np.random.default_rng(seed + 2)
    return run_inference(
        model,
        ctx.index,
        ctx.feature_cache,
        node_ids,
        HOP_FAN_OUTS,
        rng,
        CONTROL_INFERENCE_BATCH_SIZE,
    )


def embedding_frame(meta: pd.DataFrame, embeddings: np.ndarray) -> pd.DataFrame:
    """The exact artifact schema the frozen embeddings use, validated the same way."""
    if embeddings.shape != (len(meta), EMBEDDING_DIM):
        raise AssertionError("Embedding matrix shape does not match the node table.")
    if not np.isfinite(embeddings).all():
        raise AssertionError("Embeddings contain non-finite values.")

    frame = pd.DataFrame(
        {
            "TransactionID": meta["TransactionID"].to_numpy(),
            "split": meta["split"].astype("string").to_numpy(),
            **{f"embedding_{i:02d}": embeddings[:, i] for i in range(embeddings.shape[1])},
        }
    )
    if len(frame) != EXPECTED_ROWS:
        raise AssertionError("Embedding artifact row count is not the expected count.")
    if frame.isna().any().any():
        raise AssertionError("Embedding artifact contains nulls.")
    if "isFraud" in frame.columns:
        raise AssertionError("isFraud must never appear in an embedding artifact.")
    return frame


def _run_plain_variant(ctx: EncoderContext, variant: EncoderVariant) -> dict[str, Any]:
    result = train_single_encoder(
        ctx,
        readout=variant.readout,
        budget=variant.budget,
        seed=variant.seed,
        label=variant.name,
        target_node_ids=ctx.train_node_ids,
    )
    model = result["model"]
    metrics = standalone_validation_metrics(ctx, model, variant.seed, variant.name)
    embeddings = embed_nodes(ctx, model, ctx.all_node_ids, variant.seed, variant.name)

    torch.save(model.state_dict(), variant.model_path)
    return {
        "run": result,
        "metrics": metrics,
        "embeddings": embeddings,
        "encoders": [
            {
                "role": "full_train",
                "model_path": repository_relative(variant.model_path),
                "train_target_rows": result["target_row_count"],
                "best_epoch": result["best_epoch"],
                "stopped_epoch": result["stopped_epoch"],
                "monitored": True,
                "embedded_splits": ["train", "validation", "test"],
            }
        ],
        "cross_fitting": None,
    }


def _run_cross_fitted_variant(ctx: EncoderContext, variant: EncoderVariant) -> dict[str, Any]:
    n_folds = int(variant.cross_fit_folds)
    full = train_single_encoder(
        ctx,
        readout=variant.readout,
        budget=variant.budget,
        seed=variant.seed,
        label=f"{variant.name}:full_train",
        target_node_ids=ctx.train_node_ids,
    )
    full_model = full["model"]
    metrics = standalone_validation_metrics(ctx, full_model, variant.seed, variant.name)
    torch.save(full_model.state_dict(), variant.model_path)

    fold_epochs = int(full["best_epoch"])
    folds = assign_cross_fit_folds(ctx.train_node_ids, n_folds, variant.seed)

    embeddings = np.zeros((len(ctx.meta), EMBEDDING_DIM), dtype=np.float32)
    embedded = np.zeros(len(ctx.meta), dtype=bool)

    inference_ids = np.concatenate([ctx.validation_node_ids, ctx.test_node_ids])
    inference_embeddings = embed_nodes(
        ctx, full_model, inference_ids, variant.seed, f"{variant.name}:full_train"
    )
    embeddings[inference_ids] = inference_embeddings
    embedded[inference_ids] = True

    encoders: list[dict[str, Any]] = [
        {
            "role": "full_train",
            "model_path": repository_relative(variant.model_path),
            "train_target_rows": full["target_row_count"],
            "best_epoch": full["best_epoch"],
            "stopped_epoch": full["stopped_epoch"],
            "monitored": True,
            "embedded_splits": ["validation", "test"],
        }
    ]
    curve_rows = [{"encoder": "full_train", **row} for row in full["curve_rows"]]

    for fold in range(n_folds):
        held_out = ctx.train_node_ids[folds == fold]
        fit_on = ctx.train_node_ids[folds != fold]
        label = f"{variant.name}:fold{fold}"
        fold_result = train_single_encoder(
            ctx,
            readout=variant.readout,
            budget=variant.budget,
            seed=variant.seed + 100 + fold,
            label=label,
            target_node_ids=fit_on,
            fixed_epochs=fold_epochs,
        )
        fold_model = fold_result["model"]
        fold_embeddings = embed_nodes(ctx, fold_model, held_out, variant.seed + 100 + fold, label)
        embeddings[held_out] = fold_embeddings
        embedded[held_out] = True

        fold_path = variant.fold_model_path(fold)
        torch.save(fold_model.state_dict(), fold_path)
        encoders.append(
            {
                "role": f"fold_{fold}",
                "model_path": repository_relative(fold_path),
                "train_target_rows": fold_result["target_row_count"],
                "held_out_rows": int(len(held_out)),
                "best_epoch": fold_result["best_epoch"],
                "stopped_epoch": fold_result["stopped_epoch"],
                "monitored": False,
                "embedded_splits": [f"train_fold_{fold}"],
            }
        )
        curve_rows.extend({"encoder": f"fold_{fold}", **row} for row in fold_result["curve_rows"])
        del fold_model

    if not embedded.all():
        raise AssertionError(
            f"Cross-fitted embedding left {int((~embedded).sum()):,} nodes unembedded."
        )

    fold_sizes = [int((folds == fold).sum()) for fold in range(n_folds)]
    if sum(fold_sizes) != len(ctx.train_node_ids):
        raise AssertionError("Cross-fit folds do not partition the train partition.")

    return {
        "run": full,
        "metrics": metrics,
        "embeddings": embeddings,
        "encoders": encoders,
        "curve_rows_override": curve_rows,
        "cross_fitting": {
            "n_folds": n_folds,
            "fold_assignment": "fixed_seed_permutation_unstratified",
            "fold_assignment_seed": variant.seed,
            "fold_sizes": fold_sizes,
            "fold_schedule": "fixed_epochs_from_full_train_best_epoch",
            "fold_epochs": fold_epochs,
            "train_rows_embedded_by_held_out_encoder": int(len(ctx.train_node_ids)),
            "inference_rows_embedded_by_full_train_encoder": int(len(inference_ids)),
            "feature_scaler_fit_scope": "full_train_partition",
            "feature_scaler_note": (
                "Standardization statistics are fitted once on the whole train "
                "partition and shared by every fold encoder. This is label-free "
                "preprocessing and matches the frozen encoder; it is recorded "
                "rather than treated as cross-fitted."
            ),
        },
    }


def build_variant_metadata(
    ctx: EncoderContext,
    variant: EncoderVariant,
    outcome: dict[str, Any],
) -> dict[str, Any]:
    run = outcome["run"]
    return {
        "relation_name": RELATION_NAME,
        "model_name": f"graphsage_card1_encoder_{variant.name}",
        "control_variant": variant.name,
        "control_isolates": variant.isolates,
        "control_description": variant.description,
        "random_seed": variant.seed,
        "architecture": {
            "n_hops": len(HOP_FAN_OUTS),
            "fan_outs": list(HOP_FAN_OUTS),
            "aggregator": AGGREGATOR,
            "hidden_dim": HIDDEN_DIM,
            "embedding_dim": EMBEDDING_DIM,
            "activation": "relu",
            "normalization": "l2_per_layer",
            "readout": variant.readout,
            "empty_neighborhood_policy": (
                "masked mean over zero valid neighbours is defined as the zero "
                "vector; a node falls back to its own transformed features at "
                "that hop rather than receiving an arbitrary or uninitialized value"
            ),
            "input_dim": len(ctx.feature_columns),
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
            **variant.budget.describe(len(ctx.train_node_ids)),
            "targets_drawn_by_persisted_encoder": run["targets_drawn"],
            "sampling_note": (
                "Targets are drawn without replacement within a step and with "
                "replacement across steps -- a stochastic mini-batch budget, "
                "not a strict partition of the train set."
            ),
        },
        "early_stopping": {
            "metric": "validation_pr_auc",
            "monitor_evaluation_set": "full_validation_partition",
            "monitor_sample_size": int(len(ctx.validation_node_ids)),
            "patience": variant.budget.patience,
            "min_delta": variant.budget.min_delta,
            "max_epochs": variant.budget.max_epochs,
            "stopped_epoch": run["stopped_epoch"],
            "best_epoch": run["best_epoch"],
            "best_validation_pr_auc_monitor": run["best_validation_pr_auc_monitor"],
            "estimator_cap_reached": run["epoch_cap_reached"],
            "early_stopping_triggered": run["early_stopping_triggered"],
            "final_metrics_evaluation_set": "full_validation_partition",
        },
        "cross_fitting": outcome["cross_fitting"],
        "encoders": outcome["encoders"],
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
        "input_model_dataset_path": repository_relative(MODEL_DATASET_PATH),
        "input_model_dataset_sha256": file_sha256(MODEL_DATASET_PATH),
        "input_nodes_path": repository_relative(NODES_PATH),
        "input_nodes_sha256": file_sha256(NODES_PATH),
        "input_entity_edges_path": repository_relative(ENTITY_EDGES_PATH),
        "input_entity_edges_sha256": file_sha256(ENTITY_EDGES_PATH),
        "feature_scaler_path": repository_relative(FEATURE_SCALER_PATH),
        "feature_scaler_sha256": file_sha256(FEATURE_SCALER_PATH),
        "feature_columns": ctx.feature_columns,
        "feature_column_count": len(ctx.feature_columns),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "embeddings_path": repository_relative(variant.embeddings_path),
        "embedding_row_count": EXPECTED_ROWS,
        "embedding_has_no_nulls": True,
        "embedding_has_no_target_column": True,
        "model_path": repository_relative(variant.model_path),
        "metrics_path": repository_relative(variant.metrics_path),
        "training_curve_path": repository_relative(variant.training_curve_path),
        "depends_on": [
            "src/graph/train_graphsage_encoder.py",
            "src/graph/temporal_sampler.py",
            "src/graph/build_transaction_graph.py",
        ],
    }


def variant_is_complete(variant: EncoderVariant) -> bool:
    """True when this variant's embedding block and reports are all on disk.

    An encoder run costs tens of minutes, so `--skip-existing` lets an
    interrupted sweep pick up where it stopped instead of recomputing blocks
    that are already published.
    """
    return all(
        path.exists()
        for path in (
            variant.embeddings_path,
            variant.metadata_path,
            variant.metrics_path,
            variant.training_curve_path,
        )
    )


def run_variant(
    variant_name: str,
    ctx: EncoderContext | None = None,
    skip_existing: bool = False,
) -> None:
    if variant_name not in ENCODER_VARIANTS:
        raise KeyError(
            f"Unknown encoder variant {variant_name!r}; known: {sorted(ENCODER_VARIANTS)}."
        )
    variant = ENCODER_VARIANTS[variant_name]
    if skip_existing and variant_is_complete(variant):
        print(f"[{variant.name}] Already complete; skipping.")
        return
    if ctx is None:
        ctx = build_encoder_context()

    variant.report_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    variant.embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    if variant.cross_fit_folds is None:
        outcome = _run_plain_variant(ctx, variant)
    else:
        outcome = _run_cross_fitted_variant(ctx, variant)

    frame = embedding_frame(ctx.meta, outcome["embeddings"])
    frame.to_parquet(variant.embeddings_path, index=False, engine="pyarrow", compression="snappy")

    curve_rows = outcome.get("curve_rows_override", outcome["run"]["curve_rows"])
    pd.DataFrame(curve_rows).to_csv(variant.training_curve_path, index=False)
    write_json(variant.metrics_path, outcome["metrics"])
    write_json(variant.metadata_path, build_variant_metadata(ctx, variant, outcome))

    saved = pd.read_parquet(variant.embeddings_path)
    if len(saved) != EXPECTED_ROWS:
        raise AssertionError("Saved embedding artifact row count changed.")
    if "isFraud" in saved.columns:
        raise AssertionError("isFraud leaked into the saved embedding artifact.")

    run = outcome["run"]
    print(f"\n[{variant.name}] Readout: {variant.readout}")
    print(f"[{variant.name}] Cross-fit folds: {variant.cross_fit_folds or 'none'}")
    print(f"[{variant.name}] Best epoch: {run['best_epoch']} / stopped at {run['stopped_epoch']}")
    print(
        f"[{variant.name}] Early stopping triggered: {'YES' if run['early_stopping_triggered'] else 'NO'}"
    )
    print(f"[{variant.name}] Standalone validation PR-AUC:  {outcome['metrics']['pr_auc']:.8f}")
    print(f"[{variant.name}] Standalone validation ROC-AUC: {outcome['metrics']['roc_auc']:.8f}")
    print(f"[{variant.name}] Embeddings saved: {variant.embeddings_path}")
    print(f"[{variant.name}] Reports saved: {variant.report_dir}")
    print(f"[{variant.name}] Test labels used: NO")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train a G1 attribution-control encoder variant.")
    parser.add_argument(
        "--variant",
        action="append",
        choices=sorted(ENCODER_VARIANTS),
        help="Variant to run; repeatable. Defaults to every variant, in order.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip a variant whose embedding block and reports already exist.",
    )
    args = parser.parse_args()
    names = args.variant or ["extended_budget", "neighbourhood_only", "cross_fitted"]

    if args.skip_existing and all(variant_is_complete(ENCODER_VARIANTS[name]) for name in names):
        print("Every requested variant is already complete; nothing to do.")
        return

    ctx = build_encoder_context()
    for name in names:
        run_variant(name, ctx, skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
