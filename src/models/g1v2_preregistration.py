"""The G1-v2 Stage 1 pre-registration, loaded against a fixed schema.

configs/g1_v2_preregistration.json states the Stage 1 design and its four
success criteria before any Stage 1 run exists. Every key is required and
unknown keys are rejected -- the same policy as the screening thresholds -- so a
typo cannot quietly turn a registered criterion into an unregistered one. Keys
beginning with an underscore are commentary and are ignored.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config.paths import CONFIGS_DIR

PREREGISTRATION_PATH = CONFIGS_DIR / "g1_v2_preregistration.json"

PREREGISTRATION_SCHEMA: dict[str, tuple[str, ...]] = {
    "design": (
        "relation",
        "encoder_readout",
        "encoder_seeds",
        "primary_encoder_seed",
        "cross_fit_folds",
        "shared_fold_initialisation",
        "lightgbm_fixed_budget",
        "lightgbm_seed",
        "reference",
    ),
    "criteria": (
        "a_beats_b1_card1",
        "b_beats_width_null",
        "c_leakage_gate",
        "d_sign_consistent",
    ),
    "stage2_gate": ("directional_win",),
    "diagnostics": ("partition_alignment", "count_blind_attribution"),
}


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_design(design: dict[str, Any], path: Path) -> None:
    seeds = design["encoder_seeds"]
    if not isinstance(seeds, list) or not all(_is_int(s) and s >= 0 for s in seeds):
        raise ValueError(f"{path}: encoder_seeds must be a list of non-negative integers.")
    if len(seeds) < 2:
        raise ValueError(
            f"{path}: at least two encoder seeds are required -- refit noise in this "
            f"project exceeds bootstrap CI width, so one seed is not evidence."
        )
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"{path}: encoder_seeds contains duplicates.")
    if design["primary_encoder_seed"] not in seeds:
        raise ValueError(f"{path}: primary_encoder_seed must be one of encoder_seeds.")
    for key, minimum in (
        ("cross_fit_folds", 2),
        ("lightgbm_fixed_budget", 1),
        ("lightgbm_seed", 0),
    ):
        if not _is_int(design[key]) or design[key] < minimum:
            raise ValueError(f"{path}: {key} must be an integer of at least {minimum}.")
    if design["shared_fold_initialisation"] is not True:
        raise ValueError(
            f"{path}: shared_fold_initialisation must be true -- without it the fold "
            f"encoders' latent bases disagree and the block is known to be defective."
        )
    for key in ("relation", "encoder_readout", "reference"):
        if not isinstance(design[key], str) or not design[key].strip():
            raise ValueError(f"{path}: {key} must be a non-empty string.")


def load_g1v2_preregistration(path: Path = PREREGISTRATION_PATH) -> dict[str, dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"G1-v2 pre-registration not found at {path}. It fixes the Stage 1 design "
            f"and success criteria before any run and is not optional."
        )
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    resolved: dict[str, dict[str, Any]] = {}
    for section, required_keys in PREREGISTRATION_SCHEMA.items():
        if section not in payload:
            raise KeyError(f"{path}: missing required section {section!r}.")
        values = {k: v for k, v in payload[section].items() if not k.startswith("_")}
        missing = sorted(set(required_keys) - set(values))
        unknown = sorted(set(values) - set(required_keys))
        if missing:
            raise KeyError(f"{path}: section {section!r} is missing {missing}.")
        if unknown:
            raise KeyError(
                f"{path}: section {section!r} has unknown keys {unknown}. Unknown keys are "
                f"rejected so a typo cannot register a criterion nobody chose."
            )
        resolved[section] = values

    _validate_design(resolved["design"], path)
    for section in ("criteria", "stage2_gate", "diagnostics"):
        for key, statement in resolved[section].items():
            if not isinstance(statement, str) or not statement.strip():
                raise ValueError(f"{path}: {section}.{key} must be a non-empty statement.")
    return resolved
