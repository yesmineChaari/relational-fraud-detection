"""Width null for G1-v2: the primary seed's embedding block, permuted within each split.

Thirty-two extra columns cost the frozen LightGBM configuration something even
when they carry nothing -- the original shuffled control put most of G1's loss
against B0 down to exactly that. This rebuilds the same null on the G1-v2 block
(whole rows permuted inside each split, so every split keeps its marginals and
loses only row alignment) and trains it through `train_g1v2_run`, identical in
every other respect to the real block's run. Criterion (b) of
configs/g1_v2_preregistration.json compares the two directly in
src/models/compare_g1v2_verdict.py.
"""

from __future__ import annotations

import argparse
import platform

import numpy as np
import pandas as pd

from src.config.paths import PROCESSED_DATA_DIR
from src.graph.train_graphsage_encoder_v2 import V2Run
from src.models.g1v2_preregistration import load_g1v2_preregistration
from src.models.train_lightgbm_baseline import (
    EXPECTED_ROWS,
    EXPECTED_SPLIT_COUNTS,
    repository_relative,
    write_json,
)
from src.models.train_lightgbm_g1 import embedding_feature_names
from src.models.train_lightgbm_g1_controls import permute_block_within_split
from src.models.train_lightgbm_g1v2 import REPORT_ROOT, G1V2Run, train_g1v2_run
from src.models.train_lightgbm_relational import file_sha256, read_json

SHUFFLE_SEED = 42


def shuffled_run(primary_seed: int) -> G1V2Run:
    source = V2Run(primary_seed)
    name = f"shuffled_{source.name}"
    return G1V2Run(
        name=name,
        embeddings_path=PROCESSED_DATA_DIR / f"graphsage_card1_v2_embeddings_{name}.parquet",
        encoder_metadata_path=REPORT_ROOT / name / "encoder_metadata.json",
        leakage_gate_path=None,
        description=f"Width null: the {source.name} G1-v2 block permuted within each split.",
    )


def write_shuffled_block(run: G1V2Run, source: V2Run, seed: int = SHUFFLE_SEED) -> None:
    feat_names = embedding_feature_names()
    frame = pd.read_parquet(source.embeddings_path)
    expected_columns = ["TransactionID", "split", *feat_names]
    if list(frame.columns) != expected_columns:
        raise ValueError(f"Source embedding columns must be exactly {expected_columns}.")
    if len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS:,} source embedding rows.")
    shuffled = permute_block_within_split(
        frame, feat_names, seed, expected_split_counts=dict(EXPECTED_SPLIT_COUNTS)
    )
    run.embeddings_path.parent.mkdir(parents=True, exist_ok=True)
    shuffled.to_parquet(run.embeddings_path, index=False, engine="pyarrow", compression="snappy")
    unmoved = int(
        np.all(shuffled[feat_names].to_numpy() == frame[feat_names].to_numpy(), axis=1).sum()
    )

    metadata = {
        **read_json(source.metadata_path),
        "model_name": f"graphsage_card1_v2_embeddings_{run.name}",
        "control_variant": "shuffled_embedding",
        "training_objective": (
            "none: this block is not trained. It is the source G1-v2 block with its "
            "rows permuted within each split."
        ),
        "permutation": {
            "scope": "within_split",
            "seed": seed,
            "unit": "whole 32-column row",
            "source_embeddings_path": repository_relative(source.embeddings_path),
            "source_embeddings_sha256": file_sha256(source.embeddings_path),
            "rows_left_in_original_position": unmoved,
            "per_split_marginals_preserved": True,
        },
        "leakage_gate": {"not_applicable": "permutation null; no encoder was trained"},
        "embeddings_path": repository_relative(run.embeddings_path),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    run.encoder_metadata_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(run.encoder_metadata_path, metadata)
    print(f"[{run.name}] Shuffled block written ({unmoved:,} of {EXPECTED_ROWS:,} rows unmoved)")


def run_width_null(skip_existing: bool = False) -> None:
    primary = load_g1v2_preregistration()["design"]["primary_encoder_seed"]
    source = V2Run(primary)
    if not source.is_complete():
        raise FileNotFoundError(
            f"The primary G1-v2 block ({source.name}) is not complete. "
            f"Run: python -m src.graph.train_graphsage_encoder_v2 --seed {primary}"
        )
    run = shuffled_run(primary)
    if skip_existing and run.is_complete():
        print(f"[{run.name}] Already complete; skipping.")
        return
    if not run.embeddings_path.exists() or not run.encoder_metadata_path.exists():
        write_shuffled_block(run, source)
    train_g1v2_run(run)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.train_lightgbm_g1v2_controls",
        description="Train the G1-v2 width null (primary block permuted within split).",
    )
    parser.add_argument("--skip-existing", action="store_true")
    args = parser.parse_args(argv)
    run_width_null(skip_existing=args.skip_existing)


if __name__ == "__main__":
    main()
