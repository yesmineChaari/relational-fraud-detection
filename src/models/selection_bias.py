"""Selection-bias estimators over persisted validation learning curves.

Every headline number in this project is the maximum of a noisy validation
curve. Training uses patience on `average_precision` and reports the PR-AUC at
that curve's argmax. A run continues precisely because the curve keeps setting
new maxima, so a model whose curve plateaus slowly is granted more rounds *and*
draws its reported maximum from more samples of a noisy statistic. Two arms of a
comparison are therefore not scored under the same amount of selection, and the
arm granted more rounds is flattered.

This is not the estimator-cap defect. That was "the cap was binding". This is
present at a cap no run reaches, where early stopping triggers cleanly for both
arms. It is a property of reporting an argmax.

Three estimators of the same paired delta are defined here:

`argmax`
    What the training pipeline reports: each arm's own curve maximum. Upper
    bound on the delta for the arm granted more rounds.

`matched`
    Each arm's maximum over a common budget. Removes the unequal-draws
    advantage, but truncating to the shorter curve penalises the arm that
    legitimately wanted more rounds, so this is a lower bound.

`plateau`
    Each arm's mean level over a common tail window. Reads the level the curve
    settled at rather than its best draw, so no argmax is taken at all. The
    least selection-contaminated of the three, and the one that answers "is this
    model better" rather than "did this model get a luckier maximum".

The truth for any single comparison lies between `matched` and `argmax`. Where
they agree within the noise floor, the comparison is unexposed and the reported
figure stands. Where they disagree, `exposure_gap` says why: the bias tracks how
far apart the two arms stopped.

Every function here is pure and reads only already-written learning curves;
callers own writing whatever report artifact the comparison produces. Nothing in
this module refits, and nothing writes to a frozen artifact.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

ARGMAX = "argmax"
MATCHED = "matched"
PLATEAU = "plateau"
ESTIMATORS = [ARGMAX, MATCHED, PLATEAU]

DEFAULT_PLATEAU_WINDOW = 2000

AVERAGE_PRECISION_COLUMN = "validation_average_precision"
AUC_COLUMN = "validation_auc"
ITERATION_COLUMN = "iteration"


def _metric_values(curve: pd.DataFrame, metric_column: str) -> np.ndarray:
    """The metric column as a float array, validated against a curve's shape."""
    if metric_column not in curve.columns:
        raise ValueError(
            f"Learning curve is missing metric column {metric_column!r}; got {list(curve.columns)}."
        )
    if ITERATION_COLUMN not in curve.columns:
        raise ValueError(f"Learning curve is missing {ITERATION_COLUMN!r}.")
    values = curve[metric_column].to_numpy(dtype=float)
    if len(values) == 0:
        raise ValueError("Learning curve is empty.")
    if not np.isfinite(values).all():
        raise ValueError(f"Learning curve column {metric_column!r} has non-finite values.")
    return values


def common_length(reference_curve: pd.DataFrame, variant_curve: pd.DataFrame) -> int:
    """Rounds both arms actually reached -- the widest budget scoring both fairly."""
    return int(min(len(reference_curve), len(variant_curve)))


def exposure_gap(
    reference_curve: pd.DataFrame,
    variant_curve: pd.DataFrame,
) -> dict[str, Any]:
    """How unequally the two arms were selected, before any delta is computed.

    This is the cheap screen: a comparison whose arms stopped within a few
    hundred rounds of each other cannot be materially distorted by the argmax,
    however large its delta. One whose arms stopped thousands of rounds apart
    should not be read from `argmax` alone.
    """
    n_reference = int(len(reference_curve))
    n_variant = int(len(variant_curve))
    gap = n_variant - n_reference
    longer = "variant" if gap > 0 else ("reference" if gap < 0 else "equal")
    return {
        "reference_rounds": n_reference,
        "variant_rounds": n_variant,
        "round_gap": gap,
        "absolute_round_gap": abs(gap),
        "longer_trained": longer,
        "round_ratio": (
            float(max(n_reference, n_variant) / min(n_reference, n_variant))
            if min(n_reference, n_variant) > 0
            else float("inf")
        ),
        "common_rounds": common_length(reference_curve, variant_curve),
    }


def delta_at(
    reference_curve: pd.DataFrame,
    variant_curve: pd.DataFrame,
    estimator: str,
    metric_column: str = AVERAGE_PRECISION_COLUMN,
    plateau_window: int = DEFAULT_PLATEAU_WINDOW,
) -> float:
    """Paired variant-minus-reference delta under one of the three estimators."""
    if estimator not in ESTIMATORS:
        raise ValueError(f"Unknown estimator {estimator!r}; expected one of {ESTIMATORS}.")
    reference = _metric_values(reference_curve, metric_column)
    variant = _metric_values(variant_curve, metric_column)

    if estimator == ARGMAX:
        return float(variant.max() - reference.max())

    n = min(len(reference), len(variant))
    if estimator == MATCHED:
        return float(variant[:n].max() - reference[:n].max())

    window = min(plateau_window, n)
    if window <= 0:
        raise ValueError("Plateau window resolved to zero rounds.")
    return float(variant[n - window : n].mean() - reference[n - window : n].mean())


def selected_iteration(
    curve: pd.DataFrame,
    metric_column: str = AVERAGE_PRECISION_COLUMN,
    budget: int | None = None,
) -> int:
    """The iteration a stopping rule on `metric_column` would select.

    With `budget`, the argmax is taken over the first `budget` rounds only --
    what the run would have reported had it been granted that many rounds.
    """
    values = _metric_values(curve, metric_column)
    if budget is not None:
        values = values[:budget]
        if len(values) == 0:
            raise ValueError("Budget resolved to zero rounds.")
    return int(curve[ITERATION_COLUMN].to_numpy()[int(np.argmax(values))])


def off_metric_delta_at_selection(
    reference_curve: pd.DataFrame,
    variant_curve: pd.DataFrame,
    stop_metric_column: str = AVERAGE_PRECISION_COLUMN,
    off_metric_column: str = AUC_COLUMN,
    budget: int | None = None,
) -> float:
    """Delta on a metric the stopping rule does not watch, at the selected round.

    The decisive check on whether extra rounds bought real learning. A model
    that genuinely improved by training longer improves on both metrics; one
    whose advantage is a luckier draw of the stopping metric does not. Reading
    the off-metric at the *same* selected iteration keeps the comparison
    paired -- this is not a second argmax.
    """
    reference_stop = _metric_values(reference_curve, stop_metric_column)
    variant_stop = _metric_values(variant_curve, stop_metric_column)
    reference_off = _metric_values(reference_curve, off_metric_column)
    variant_off = _metric_values(variant_curve, off_metric_column)

    if budget is not None:
        reference_stop, reference_off = reference_stop[:budget], reference_off[:budget]
        variant_stop, variant_off = variant_stop[:budget], variant_off[:budget]

    return float(
        variant_off[int(np.argmax(variant_stop))] - reference_off[int(np.argmax(reference_stop))]
    )


def compare_estimators(
    reference_curve: pd.DataFrame,
    variant_curve: pd.DataFrame,
    metric_column: str = AVERAGE_PRECISION_COLUMN,
    off_metric_column: str = AUC_COLUMN,
    plateau_window: int = DEFAULT_PLATEAU_WINDOW,
    noise_floor: float | None = None,
) -> dict[str, Any]:
    """All three estimators plus the exposure screen for one paired comparison.

    `estimators_agree` is the decision this module exists to support: when the
    argmax and matched deltas sit within the noise floor of each other, the
    reported figure is not distorted by unequal selection and needs no
    correction. It is deliberately keyed on the noise floor the seed-variance
    panel measured rather than on a threshold invented here.
    """
    exposure = exposure_gap(reference_curve, variant_curve)
    deltas = {
        estimator: delta_at(
            reference_curve,
            variant_curve,
            estimator,
            metric_column=metric_column,
            plateau_window=plateau_window,
        )
        for estimator in ESTIMATORS
    }
    result: dict[str, Any] = {
        **exposure,
        "plateau_window": min(plateau_window, exposure["common_rounds"]),
        "delta_argmax": deltas[ARGMAX],
        "delta_matched": deltas[MATCHED],
        "delta_plateau": deltas[PLATEAU],
        "argmax_minus_matched": deltas[ARGMAX] - deltas[MATCHED],
        "off_metric_delta_at_selection": off_metric_delta_at_selection(
            reference_curve,
            variant_curve,
            stop_metric_column=metric_column,
            off_metric_column=off_metric_column,
        ),
        "sign_flips_under_matching": (
            np.sign(deltas[ARGMAX]) != np.sign(deltas[MATCHED])
            and deltas[ARGMAX] != 0.0
            and deltas[MATCHED] != 0.0
        ),
    }
    if noise_floor is not None:
        if noise_floor <= 0:
            raise ValueError("Noise floor must be positive.")
        result["noise_floor"] = float(noise_floor)
        result["estimators_agree"] = bool(abs(result["argmax_minus_matched"]) <= noise_floor)
        result["shift_in_noise_floors"] = float(result["argmax_minus_matched"] / noise_floor)
    return result
