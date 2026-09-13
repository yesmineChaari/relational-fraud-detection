"""Post-hoc diagnostic: does aligning the cross-fitted encoders explain the G1-v2 loss?

The G1-v2 verdict (reports/g1_v2/g1_v2_verdict.json) is negative, and the real
embedding block scored below a within-split permutation of itself. The leading
explanation is a basis shift: train rows are embedded by three fold encoders
and validation rows by the full-train encoder, and the partition alignment gap
stayed at 0.40-0.61 despite the shared initialisation, against at most ~0.23
for any single-encoder block. This module tests that explanation on the saved
encoders of one seed without retraining any of them.

Each fold encoder's output is mapped into the full-train encoder's coordinates
by an affine orthogonal Procrustes fit: centre, rotate, re-centre. The fit uses
train rows embedded by both encoders over identical sampled neighbourhoods, so
the encoders are the only difference, and it reads no labels; a single 32x32
rotation fitted on tens of thousands of rows has no capacity to carry any one
row's label. Validation and test rows keep their full-train embeddings, so the
evaluation side of the block is exactly the published one.

The rule, fixed before the diagnostic was run: the misalignment explanation is
supported only if the aligned block (i) brings the partition alignment gap
within the largest gap of any single-encoder block in
reports/g1_controls/g1_control_summary.json and (ii) beats the unaligned block
with a paired-bootstrap 95% CI entirely above zero. It is read on validation
after the verdict, so it cannot revise the verdict; it decides only whether
Stage 2 includes the alignment step.

Two steps, each its own process -- `build` (encoder inference) and `evaluate`
(a fixed-budget LightGBM fit, ~4.5 GB peak) -- so they never share memory.
"""

from __future__ import annotations

import argparse
import platform
from typing import Any

import numpy as np
import pandas as pd
import torch

from src.config.paths import PROCESSED_DATA_DIR, REPORTS_DIR
from src.graph.train_graphsage_encoder import EMBEDDING_DIM, HIDDEN_DIM
from src.graph.train_graphsage_encoder_v2 import (
    READOUT,
    WITH_COUNTS,
    CardinalityAwareGraphSAGEWithHead,
    V2Run,
    build_leakage_gate,
    run_forward_v2,
)
from src.graph.train_graphsage_variants import (
    CONTROL_INFERENCE_BATCH_SIZE,
    EncoderContext,
    assign_cross_fit_folds,
    build_encoder_context,
    embedding_frame,
)
from src.models.g1v2_preregistration import load_g1v2_preregistration
from src.models.significance import compare_variants
from src.models.train_lightgbm_baseline import repository_relative, write_json
from src.models.train_lightgbm_g1 import embedding_feature_names
from src.models.train_lightgbm_g1v2 import (
    REPORT_ROOT,
    G1V2Run,
    encoder_backed_run,
    train_g1v2_run,
)
from src.models.train_lightgbm_g1v2_controls import shuffled_run
from src.models.train_lightgbm_relational import file_sha256, read_json

ANCHOR_ROWS_PER_FOLD = 20_000
CONTROL_SUMMARY_PATH = REPORTS_DIR / "g1_controls" / "g1_control_summary.json"
UNALIGNED_CONTROL = "cross_fitted"

RULE = (
    "The misalignment explanation is supported only if the aligned block (i) brings "
    "the partition alignment mean gap within the largest mean gap of any "
    "single-encoder block in reports/g1_controls/g1_control_summary.json and (ii) "
    "beats the unaligned block with a paired-bootstrap 95% CI entirely above zero."
)


def diagnostic_path(seed: int):
    return REPORTS_DIR / "g1_v2" / f"alignment_diagnostic_seed{seed}.json"


def aligned_run(seed: int) -> G1V2Run:
    name = f"aligned_seed{seed}"
    return G1V2Run(
        name=name,
        embeddings_path=PROCESSED_DATA_DIR / f"graphsage_card1_v2_embeddings_{name}.parquet",
        encoder_metadata_path=REPORT_ROOT / name / "encoder_metadata.json",
        leakage_gate_path=REPORT_ROOT / name / "leakage_gate.json",
        description=(
            f"Post-hoc diagnostic: the seed-{seed} G1-v2 block with each fold encoder's "
            f"output aligned to the full-train encoder by orthogonal Procrustes."
        ),
    )


def fit_affine_procrustes(source: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    """The orthogonal map plus translation taking `source` rows closest to `target` rows."""
    if source.ndim != 2 or source.shape != target.shape:
        raise ValueError("source and target must be 2-D arrays of the same shape.")
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    u, _, vt = np.linalg.svd((source - source_mean).T @ (target - target_mean))
    return {"source_mean": source_mean, "rotation": u @ vt, "target_mean": target_mean}


def apply_alignment(embeddings: np.ndarray, alignment: dict[str, np.ndarray]) -> np.ndarray:
    centred = np.asarray(embeddings, dtype=np.float64) - alignment["source_mean"]
    return centred @ alignment["rotation"] + alignment["target_mean"]


def single_encoder_alignment_ceiling(path=CONTROL_SUMMARY_PATH) -> float:
    """The largest alignment gap any single-encoder block showed, read from its report."""
    controls = read_json(path)["controls"]
    gaps = [
        float(payload["embedding_partition_alignment"]["mean_gap"])
        for name, payload in controls.items()
        if name != UNALIGNED_CONTROL
    ]
    if not gaps:
        raise ValueError(f"No single-encoder alignment gaps found in {path}.")
    return max(gaps)


def load_encoder(path, ctx: EncoderContext) -> CardinalityAwareGraphSAGEWithHead:
    model = CardinalityAwareGraphSAGEWithHead(
        len(ctx.feature_columns), HIDDEN_DIM, EMBEDDING_DIM, READOUT
    )
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a, np.float64) - np.asarray(b, np.float64)) ** 2)))


def build_aligned_block(seed: int) -> None:
    source = V2Run(seed)
    if not source.is_complete():
        raise FileNotFoundError(f"The G1-v2 block for seed {seed} is not complete.")
    run = aligned_run(seed)
    source_metadata = read_json(source.metadata_path)
    cross_fitting = source_metadata["cross_fitting"]

    ctx = build_encoder_context()
    n_folds = int(cross_fitting["n_folds"])
    folds = assign_cross_fit_folds(
        ctx.train_node_ids, n_folds, int(cross_fitting["fold_assignment_seed"])
    )
    if [int((folds == fold).sum()) for fold in range(n_folds)] != cross_fitting["fold_sizes"]:
        raise AssertionError("Recomputed fold assignment does not match the published run.")

    feat_names = embedding_feature_names()
    frame = pd.read_parquet(source.embeddings_path, columns=["TransactionID", *feat_names])
    if not np.array_equal(frame["TransactionID"].to_numpy(), ctx.meta["TransactionID"].to_numpy()):
        raise AssertionError("Embedding rows are not in node order.")
    embeddings = frame[feat_names].to_numpy(dtype=np.float64, copy=True)
    del frame

    full = load_encoder(source.model_path, ctx)
    anchor_rng = np.random.default_rng(seed + 500)
    fold_reports: list[dict[str, Any]] = []
    for fold in range(n_folds):
        held_out = ctx.train_node_ids[folds == fold]
        anchor = anchor_rng.choice(
            held_out, size=min(ANCHOR_ROWS_PER_FOLD, len(held_out)), replace=False
        )
        fold_model = load_encoder(source.fold_model_path(fold), ctx)
        sampling_seed = seed + 600 + fold
        print(f"[aligned_seed{seed}] fold {fold}: embedding {len(anchor):,} anchor rows twice...")
        fold_anchor, _ = run_forward_v2(
            fold_model,
            ctx.index,
            ctx.feature_cache,
            anchor,
            np.random.default_rng(sampling_seed),
            CONTROL_INFERENCE_BATCH_SIZE,
            WITH_COUNTS,
        )
        full_anchor, _ = run_forward_v2(
            full,
            ctx.index,
            ctx.feature_cache,
            anchor,
            np.random.default_rng(sampling_seed),
            CONTROL_INFERENCE_BATCH_SIZE,
            WITH_COUNTS,
        )
        alignment = fit_affine_procrustes(fold_anchor, full_anchor)
        embeddings[held_out] = apply_alignment(embeddings[held_out], alignment)
        fold_reports.append(
            {
                "fold": fold,
                "anchor_rows": int(len(anchor)),
                "anchor_rmse_before": _rmse(fold_anchor, full_anchor),
                "anchor_rmse_after": _rmse(apply_alignment(fold_anchor, alignment), full_anchor),
                "rotation_orthogonality_error": float(
                    np.abs(
                        alignment["rotation"].T @ alignment["rotation"] - np.eye(EMBEDDING_DIM)
                    ).max()
                ),
            }
        )
        print(
            f"[aligned_seed{seed}] fold {fold}: anchor RMSE "
            f"{fold_reports[-1]['anchor_rmse_before']:.4f} -> "
            f"{fold_reports[-1]['anchor_rmse_after']:.4f}"
        )
        del fold_model

    aligned = embeddings.astype(np.float32)
    run.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    embedding_frame(ctx.meta, aligned).to_parquet(
        run.embeddings_path, index=False, engine="pyarrow", compression="snappy"
    )
    gate = build_leakage_gate(ctx, aligned, cross_fitting, run.embeddings_path)
    run.leakage_gate_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(run.leakage_gate_path, gate)

    metadata = {
        **source_metadata,
        "model_name": f"graphsage_card1_v2_embeddings_{run.name}",
        "alignment": {
            "method": "affine_orthogonal_procrustes_per_fold_onto_full_train_encoder",
            "anchor": (
                "train rows held out by the fold, embedded by the fold encoder and the "
                "full-train encoder over identical sampled neighbourhoods; no labels read"
            ),
            "anchor_rows_per_fold": ANCHOR_ROWS_PER_FOLD,
            "validation_and_test_rows": "unchanged full-train embeddings",
            "source_embeddings_path": repository_relative(source.embeddings_path),
            "source_embeddings_sha256": file_sha256(source.embeddings_path),
            "folds": fold_reports,
        },
        "leakage_gate": {
            "usable": gate["usable"],
            "path": repository_relative(run.leakage_gate_path),
        },
        "embeddings_path": repository_relative(run.embeddings_path),
        "versions": {**source_metadata.get("versions", {}), "python": platform.python_version()},
    }
    write_json(run.encoder_metadata_path, metadata)
    print(
        f"[aligned_seed{seed}] Partition alignment mean gap: "
        f"{gate['partition_alignment_diagnostic']['mean_gap']:.3f}  "
        f"leakage gate: {'USABLE' if gate['usable'] else 'NOT USABLE'}"
    )


def evaluate_aligned_block(seed: int) -> dict[str, Any]:
    run = aligned_run(seed)
    for path in (run.embeddings_path, run.encoder_metadata_path, run.leakage_gate_path):
        if not path.exists():
            raise FileNotFoundError(f"Run the build step first; missing {path}.")
    train_g1v2_run(run, skip_existing=True)

    predictions = run.paths()["validation_predictions"]
    unaligned = encoder_backed_run(seed)
    null = shuffled_run(seed)
    aligned_vs_unaligned = compare_variants(
        f"g1v2_{run.name}",
        predictions,
        f"g1v2_{unaligned.name}",
        unaligned.paths()["validation_predictions"],
    )
    aligned_vs_null = compare_variants(
        f"g1v2_{run.name}", predictions, f"g1v2_{null.name}", null.paths()["validation_predictions"]
    )

    ceiling = single_encoder_alignment_ceiling()
    gap_before = float(
        read_json(V2Run(seed).leakage_gate_path)["partition_alignment_diagnostic"]["mean_gap"]
    )
    gate = read_json(run.leakage_gate_path)
    gap_after = float(gate["partition_alignment_diagnostic"]["mean_gap"])
    realigned = gap_after <= ceiling
    improves = float(aligned_vs_unaligned["ci_lower_95"]) > 0.0

    report = {
        "report_name": f"G1-v2 alignment diagnostic, encoder seed {seed}",
        "post_hoc": True,
        "revises_the_verdict": False,
        "rule": RULE,
        "partition_alignment": {
            "mean_gap_before": gap_before,
            "mean_gap_after": gap_after,
            "single_encoder_ceiling": ceiling,
            "within_single_encoder_range": realigned,
        },
        "aligned_vs_b1_card1": read_json(run.paths()["significance"]),
        "aligned_vs_unaligned": aligned_vs_unaligned,
        "aligned_vs_width_null": aligned_vs_null,
        "leakage_gate_usable": bool(gate["usable"]),
        "misalignment_explanation_supported": bool(realigned and improves),
        "consequence_for_stage2": (
            "include the alignment step"
            if realigned and improves
            else "alignment does not explain the loss; do not add it to Stage 2"
        ),
        "evaluation_split": "validation",
        "test_evaluated": False,
    }
    write_json(diagnostic_path(seed), report)
    print(
        f"\nAlignment gap {gap_before:.3f} -> {gap_after:.3f} (single-encoder ceiling {ceiling:.3f})"
        f"\nAligned vs unaligned: {aligned_vs_unaligned['observed_delta']:+.5f} "
        f"[{aligned_vs_unaligned['ci_lower_95']:+.5f}, {aligned_vs_unaligned['ci_upper_95']:+.5f}]"
        f"\nAligned vs B1-card1: {report['aligned_vs_b1_card1']['observed_delta']:+.5f}"
        f"\nMisalignment explanation supported: {report['misalignment_explanation_supported']}"
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.diagnose_g1v2_alignment",
        description="Align the cross-fitted G1-v2 encoders post hoc, then re-evaluate.",
    )
    parser.add_argument("--step", choices=("build", "evaluate"), required=True)
    parser.add_argument("--seed", type=int, default=None, help="Defaults to the primary seed.")
    args = parser.parse_args(argv)
    seed = args.seed
    if seed is None:
        seed = load_g1v2_preregistration()["design"]["primary_encoder_seed"]
    if args.step == "build":
        build_aligned_block(seed)
    else:
        evaluate_aligned_block(seed)


if __name__ == "__main__":
    main()
