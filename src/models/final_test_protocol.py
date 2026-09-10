"""The one-shot final test protocol, fixed before the test partition is opened.

The test partition -- 88,581 rows, the chronologically last 15% -- has never been
read. Every artifact in the repository records `test_evaluated: false`. That is
the one guarantee this project can spend only once, so everything about the read
is decided here, committed, and only then executed.

This module holds decisions, not code that touches data. The executor imports it
and refuses to run unless this file is committed and unmodified, so the protocol
cannot drift between the review and the read.

**What is scored.** B0 and B1-card1, and nothing else. The graph model lost to
both on validation, so the test partition has nothing to tell us about it, and
scoring it would spend the guarantee for no answer. The ablation, seed,
convergence and null-control panels are validation instruments and are not scored.

**Which artifacts.** The primary pair is the converged protocol (cap 15,000,
patience on average precision, seed 42), because the gate on this read explicitly
waited for the estimator-cap decision on the grounds that it would change every
model scored here. The frozen 6,000-round pair is scored as a secondary,
descriptive row for continuity with the originally named models; no claim rests
on it.

**What counts as a result.** One pre-registered comparison, B1-card1 against B0,
on the paired bootstrap every validation interval in this project used. Its three
outcomes are fixed below. Everything else is reported, not tested.

**How results are treated.** A test figure below its validation figure is the
expected outcome for a validation-selected model and is the honest headline, not
something to explain away. The validation-to-test gap is itself a finding: it
measures what validation-guided selection cost.

**What is forbidden afterwards.** No model, threshold, feature or protocol change
informed by these results. Further work motivated by them is a new stage against
a newly held-out partition.
"""

from __future__ import annotations

PROTOCOL_VERSION = 1

# Rows the read must find, and the only split it may touch.
TEST_SPLIT = "test"
EXPECTED_TEST_ROWS = 88_581

# ---------------------------------------------------------------------------
# Models, named before any test data is loaded. Not revised after the fact.
# ---------------------------------------------------------------------------

PRIMARY_MODELS: dict[str, dict[str, str]] = {
    "b0": {
        "role": "reference",
        "model": "models/convergence_check/lightgbm_b0__stop_average_precision_cap15000_seed42.txt",
        "validation_metrics": (
            "reports/convergence_check/b0/stop_average_precision/cap15000_seed42/metrics.json"
        ),
        "validation_predictions": (
            "reports/convergence_check/b0/stop_average_precision/cap15000_seed42/"
            "validation_predictions.parquet"
        ),
    },
    "b1_card1": {
        "role": "candidate",
        "model": (
            "models/convergence_check/lightgbm_b1_card1__stop_average_precision_cap15000_seed42.txt"
        ),
        "validation_metrics": (
            "reports/convergence_check/b1_card1/stop_average_precision/cap15000_seed42/metrics.json"
        ),
        "validation_predictions": (
            "reports/convergence_check/b1_card1/stop_average_precision/cap15000_seed42/"
            "validation_predictions.parquet"
        ),
    },
}

# Descriptive only. Reported beside the primary pair; no outcome is read from it.
SECONDARY_MODELS: dict[str, dict[str, str]] = {
    "b0_frozen": {
        "role": "reference",
        "model": "models/lightgbm_baseline.txt",
        "validation_metrics": "reports/baseline/lightgbm_metrics.json",
        "validation_predictions": "reports/baseline/validation_predictions.parquet",
    },
    "b1_card1_frozen": {
        "role": "candidate",
        "model": "models/lightgbm_b1_card1.txt",
        "validation_metrics": "reports/b1/card1/metrics.json",
        "validation_predictions": "reports/b1/card1/validation_predictions.parquet",
    },
}

NOT_SCORED: dict[str, str] = {
    "g1_card1": (
        "Lost to B0 and B1-card1 on validation with intervals excluding zero; the "
        "test partition cannot change that conclusion, so scoring it spends the "
        "guarantee for no answer."
    ),
    "validation panels": (
        "Ablation, fixed-budget, seed-variance, convergence and permuted-null runs "
        "are instruments for validation questions, not candidate models."
    ),
}

# ---------------------------------------------------------------------------
# Metrics, fixed in advance: the set every other stage reports. No additions.
# ---------------------------------------------------------------------------

METRICS = ["pr_auc", "roc_auc"]
TOP_FRACTIONS = [0.005, 0.01, 0.02, 0.05]  # precision and recall at each

# ---------------------------------------------------------------------------
# The one pre-registered comparison and its outcomes.
# ---------------------------------------------------------------------------

PRIMARY_COMPARISON = {
    "candidate": "b1_card1",
    "reference": "b0",
    "metric": "pr_auc",
    "method": "paired bootstrap over test rows, same indices applied to both models",
    "n_resamples": 10_000,
    "seed": 42,
    "interval": "95% percentile",
    # For the reader, not for the rule: the validation result being tested.
    "validation_delta": 0.006304055057609892,
    "validation_ci_95": [0.002387257418836225, 0.010179266133185674],
}

OUTCOME_REPLICATES = "GAIN_REPLICATES_ON_TEST"
OUTCOME_NOT_CONFIRMED = "GAIN_NOT_CONFIRMED_ON_TEST"
OUTCOME_REVERSES = "GAIN_REVERSES_ON_TEST"

OUTCOME_RULE = (
    "Applied to the primary comparison only. If the 95% interval on the test "
    "PR-AUC delta lies entirely above zero, the relational gain replicates. If it "
    "contains zero, the gain is not confirmed on test and the project's headline "
    "is reported as a validation-only finding. If it lies entirely below zero, "
    "the gain reverses on test. The outcome is reported whichever way it falls, "
    "and its magnitude is not compared to the validation delta as a pass mark: a "
    "smaller test delta is expected under validation-guided selection."
)

# ---------------------------------------------------------------------------
# Reporting and conduct.
# ---------------------------------------------------------------------------

REPORT_VALIDATION_TO_TEST_GAP = True  # per model, per metric, test minus validation

TREATMENT = (
    "A test figure below validation is the expected, correct outcome for a "
    "validation-selected model and is reported as the headline number."
)

FORBIDDEN_AFTERWARDS = [
    "Any change to a model, its features, or its training configuration informed by test results.",
    "Any decision threshold or operating point tuned on test predictions.",
    "Any metric added, dropped or re-weighted after the results are seen.",
    "A second read of the test partition under a revised protocol. Work motivated "
    "by these results is a new stage against a newly held-out partition.",
]

# ---------------------------------------------------------------------------
# Execution guards the executor must enforce before any test row is scored.
# ---------------------------------------------------------------------------

EXECUTION_GUARDS = [
    "This file is committed and has no uncommitted modification.",
    "The output directory does not exist yet: the read happens once.",
    "Exactly 88,581 test rows are loaded and every one carries split == 'test'.",
    "Models are loaded as saved. Nothing is refit, retuned or re-thresholded.",
    "Each model first re-scores the validation rows through the same inference "
    "path and must reproduce its persisted validation predictions exactly; any "
    "mismatch aborts before a single test row is scored.",
    "B1-card1's relational features for test rows come from the same builder and "
    "strictly-before rule as train and validation, with the frozen categorical "
    "mappings applied unchanged.",
    "test_evaluated flips to true only in the artifacts this run writes.",
    "Any deviation from this protocol is recorded in the output summary rather "
    "than silently absorbed.",
]

OUTPUT_DIR = "reports/final_test"
