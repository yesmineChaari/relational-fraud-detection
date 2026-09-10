"""Leakage checks for the graph stage that temporal assertions cannot see.

The sampler's temporal contract is already asserted directly and thoroughly: no
neighbour at any hop shares or exceeds its target's timestamp, hop-2 is anchored
to the original target rather than to the intermediate node, and the guards are
exercised on adversarial fixtures. Those tests pass, and they passed against the
encoder that produced the disappointing graph result.

The failure mode that actually cost the stage sits outside that surface. The
encoder is fit on the train partition with an `isFraud` head and then used to
embed those same rows, so a train row's embedding can carry information from
that row's own label while a validation row's embedding cannot. Nothing about
that is a temporal violation -- every neighbour is still strictly in the past --
and no amount of admissibility testing will detect it. What it produces is a
train/inference *informativeness gap*: the embedding block looks far more
predictive on the rows it was fitted through than on rows it merely scored, and
downstream that surfaces as an unexplained regression rather than as an error.

Two checks are defined here.

`cross_fit_provenance` verifies the structural claim: under cross-fitting, every
train row's embedding was produced by an encoder that never saw that row's
label, and the folds genuinely partition the train partition. This is checkable
from the run's own recorded provenance without re-running anything.

`informativeness_gap` measures the symptom directly. A small probe is
cross-validated *within* each split separately and its ranking quality compared.
Probing within a split rather than across splits is the point: if each train
row's embedding encodes that row's own label, the contamination is per-row and
survives the probe's own train/test division, so it shows up as elevated
within-train ranking quality. A clean pipeline leaves the two splits comparable;
a contaminated one does not.

The tolerance is deliberately a stated constant rather than a tuned one, and the
check is designed to fail against the configuration that produced the problem. A
test that passes against a broken implementation is not a test.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

# A probe gap wider than this is treated as evidence of label information in the
# embeddings of the rows the encoder was fitted through. Stated rather than
# tuned, and deliberately loose: a cross-validated ROC-AUC carries sampling
# scatter of its own, so the threshold has to clear that scatter rather than the
# far smaller seed-to-seed noise of a fitted model.
INFORMATIVENESS_GAP_TOLERANCE = 0.02

# The tolerance above is only meaningful once each split carries enough
# positives for its probe AUC to be stable. Cross-validated AUC scatter falls
# roughly as 1/sqrt(positives): at ~600 positives it is around +/-0.03, which
# alone would breach the tolerance on embeddings carrying no label information
# whatsoever. Below this floor the comparison is refused rather than reported,
# because a false leakage alarm is worse than an absent check.
MIN_POSITIVES_PER_SPLIT = 1_500

PROBE_FOLDS = 4
PROBE_MAX_ITER = 200
PROBE_SEED = 42

PARITY_HOLDS = "TRAIN_INFERENCE_PARITY_HOLDS"
PARITY_VIOLATED = "TRAIN_INFERENCE_PARITY_VIOLATED"


def probe_auc(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_folds: int = PROBE_FOLDS,
    seed: int = PROBE_SEED,
    min_positives: int = MIN_POSITIVES_PER_SPLIT,
) -> float:
    """Cross-validated ROC-AUC of a linear probe on the embedding block alone.

    Cross-validated *within* the supplied rows. A probe fitted and scored on the
    same rows would measure the probe's capacity rather than the embedding's
    content, and a probe fitted on one split and scored on another would confound
    label contamination with ordinary distribution shift between splits.
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int8)
    if embeddings.ndim != 2:
        raise ValueError("Embeddings must be a 2-D array.")
    if len(embeddings) != len(labels):
        raise ValueError("Embeddings and labels must have equal length.")
    positives = int(labels.sum())
    if positives < n_folds or positives == len(labels):
        raise ValueError(
            f"Need at least {n_folds} positives and at least one negative to probe; "
            f"got {positives} positives of {len(labels)} rows."
        )
    if positives < min_positives:
        raise ValueError(
            f"Probe needs at least {min_positives:,} positives for its AUC to be "
            f"stable enough to compare across splits; got {positives:,}. Comparing "
            f"below this floor produces leakage alarms from sampling scatter alone."
        )

    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    scores = np.empty(len(labels), dtype=np.float64)
    for fit_index, score_index in splitter.split(embeddings, labels):
        model = LogisticRegression(max_iter=PROBE_MAX_ITER)
        model.fit(embeddings[fit_index], labels[fit_index])
        scores[score_index] = model.predict_proba(embeddings[score_index])[:, 1]
    return float(roc_auc_score(labels, scores))


def informativeness_gap(
    train_embeddings: np.ndarray,
    train_labels: np.ndarray,
    validation_embeddings: np.ndarray,
    validation_labels: np.ndarray,
    tolerance: float = INFORMATIVENESS_GAP_TOLERANCE,
    n_folds: int = PROBE_FOLDS,
    seed: int = PROBE_SEED,
    min_positives: int = MIN_POSITIVES_PER_SPLIT,
) -> dict[str, Any]:
    """Compare within-split probe quality on train against validation rows.

    A positive gap means the embedding block is more predictive on the rows the
    encoder was fitted through than on rows it only scored, which is the
    signature of label information reaching train-row embeddings.
    """
    if tolerance <= 0:
        raise ValueError("Tolerance must be positive.")
    train_auc = probe_auc(
        train_embeddings,
        train_labels,
        n_folds=n_folds,
        seed=seed,
        min_positives=min_positives,
    )
    validation_auc = probe_auc(
        validation_embeddings,
        validation_labels,
        n_folds=n_folds,
        seed=seed,
        min_positives=min_positives,
    )
    gap = train_auc - validation_auc
    holds = gap <= tolerance
    return {
        "train_probe_roc_auc": train_auc,
        "validation_probe_roc_auc": validation_auc,
        "gap": gap,
        "tolerance": tolerance,
        "parity_holds": holds,
        "outcome_code": PARITY_HOLDS if holds else PARITY_VIOLATED,
    }


def cross_fit_provenance(
    cross_fitting: dict[str, Any] | None,
    train_row_count: int,
    inference_row_count: int,
) -> dict[str, Any]:
    """Verify that no train row was embedded by an encoder that saw its label.

    Read from a run's own recorded provenance. The structural claim has three
    parts, and all three must hold: the folds partition the train partition
    exactly, every train row was embedded by the encoder that held it out, and
    the rows outside the train partition were embedded by the full-train encoder
    (which is sound for them, since it never saw their labels either).
    """
    if cross_fitting is None:
        return {
            "cross_fitted": False,
            "provenance_sound": False,
            "reason": (
                "No cross-fitting record. Every train row was embedded by an "
                "encoder fitted on that row's own label."
            ),
        }

    fold_sizes = list(cross_fitting.get("fold_sizes") or [])
    n_folds = int(cross_fitting.get("n_folds", 0))
    embedded_by_held_out = int(cross_fitting.get("train_rows_embedded_by_held_out_encoder", -1))
    embedded_by_full = int(cross_fitting.get("inference_rows_embedded_by_full_train_encoder", -1))

    failures = []
    if n_folds < 2:
        failures.append(f"n_folds must be at least 2; got {n_folds}.")
    if len(fold_sizes) != n_folds:
        failures.append(f"Expected {n_folds} fold sizes; got {len(fold_sizes)}.")
    if sum(fold_sizes) != train_row_count:
        failures.append(
            f"Fold sizes sum to {sum(fold_sizes):,}, not the {train_row_count:,} train rows."
        )
    if embedded_by_held_out != train_row_count:
        failures.append(
            f"{embedded_by_held_out:,} train rows embedded by a held-out encoder, "
            f"expected all {train_row_count:,}."
        )
    if embedded_by_full != inference_row_count:
        failures.append(
            f"{embedded_by_full:,} inference rows embedded by the full-train encoder, "
            f"expected {inference_row_count:,}."
        )

    return {
        "cross_fitted": True,
        "n_folds": n_folds,
        "fold_sizes": fold_sizes,
        "provenance_sound": not failures,
        "failures": failures,
    }
