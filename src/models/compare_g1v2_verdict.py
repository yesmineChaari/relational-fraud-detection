"""The pre-registered G1-v2 verdict.

Reads every run the pre-registration names -- one per encoder seed, plus the
width null -- and applies the four criteria fixed in
configs/g1_v2_preregistration.json before any Stage 1 run existed. All four must
hold for a win; any failure is reported as negative_result_stands with the
failing criteria named. A run whose design does not match the pre-registration
(readout, budget, seeds, fold count, shared initialisation) is refused outright
rather than read, because it is not evidence for the design that was
registered. Seeds are never re-drawn and nothing here trains anything.

The Stage 2 gate and the diagnostics -- partition alignment and the count-blind
attribution arm -- are reported beside the verdict and cannot change it.
"""

from __future__ import annotations

import platform
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from src.config.paths import REPORTS_DIR
from src.graph.train_graphsage_encoder_v2 import COUNT_BLIND, READOUT, WITH_COUNTS
from src.models.compare_selection_bias import load_noise_floor
from src.models.g1v2_preregistration import PREREGISTRATION_PATH, load_g1v2_preregistration
from src.models.selection_bias import compare_estimators
from src.models.significance import DEFAULT_N_RESAMPLES, compare_variants
from src.models.train_ablation_fixed_budget import (
    resolve_run_paths as resolve_fixed_budget_run_paths,
)
from src.models.train_lightgbm_baseline import repository_relative, write_json
from src.models.train_lightgbm_g1v2 import (
    PROTOCOL,
    REFERENCE_RUN_NAME,
    G1V2Run,
    encoder_backed_run,
)
from src.models.train_lightgbm_g1v2_controls import shuffled_run
from src.models.train_lightgbm_relational import file_sha256, read_json

VERDICT_PATH = REPORTS_DIR / "g1_v2" / "g1_v2_verdict.json"
SEED_TABLE_PATH = REPORTS_DIR / "g1_v2" / "g1_v2_seed_table.csv"

VERDICT_WIN = "g1v2_beats_b1_card1"
VERDICT_NEGATIVE_STANDS = "negative_result_stands"


def load_run_result(run: G1V2Run) -> dict[str, Any] | None:
    paths = run.paths()
    required = ("metrics", "metadata", "significance", "validation_predictions")
    if not all(paths[key].exists() for key in required):
        return None
    return {
        "metrics": read_json(paths["metrics"]),
        "metadata": read_json(paths["metadata"]),
        "significance": read_json(paths["significance"]),
        "validation_predictions_path": paths["validation_predictions"],
    }


def assert_matches_preregistration(
    result: dict[str, Any], prereg: dict[str, Any], encoder_seed: int
) -> None:
    metadata = result["metadata"]
    encoder = metadata["encoder"]
    design = prereg["design"]
    checks = {
        "protocol": (metadata["protocol"], PROTOCOL),
        "lightgbm_fixed_budget": (metadata["fixed_budget_trees"], design["lightgbm_fixed_budget"]),
        "lightgbm_seed": (metadata["lightgbm_parameters"]["random_state"], design["lightgbm_seed"]),
        "encoder_seed": (encoder["random_seed"], encoder_seed),
        "encoder_readout": (encoder["readout"], design["encoder_readout"]),
        "count_mode": (encoder["count_mode"], WITH_COUNTS),
        "cross_fit_folds": (encoder["cross_fit_folds"], design["cross_fit_folds"]),
        "shared_fold_initialisation": (
            encoder["shared_initialisation"],
            design["shared_fold_initialisation"],
        ),
    }
    mismatched = {
        key: {"run": actual, "preregistered": expected}
        for key, (actual, expected) in checks.items()
        if actual != expected
    }
    if mismatched:
        raise ValueError(
            f"Run for encoder seed {encoder_seed} does not match the pre-registration: "
            f"{mismatched}. It is not evidence for the registered design."
        )


def evaluate_criteria(
    prereg: dict[str, Any],
    seed_results: dict[int, dict[str, Any] | None],
    real_vs_shuffled: dict[str, Any] | None,
) -> dict[str, Any]:
    """Apply the four registered criteria. Pure: reads only what it is handed."""
    design = prereg["design"]
    seeds = list(design["encoder_seeds"])
    primary = design["primary_encoder_seed"]

    missing = [f"seed{seed}" for seed in seeds if seed_results.get(seed) is None]
    if real_vs_shuffled is None:
        missing.append("width_null")
    if missing:
        return {
            "decidable": False,
            "verdict": None,
            "missing_runs": missing,
            "rules": prereg["criteria"],
        }

    deltas = {seed: float(seed_results[seed]["significance"]["observed_delta"]) for seed in seeds}
    primary_significance = seed_results[primary]["significance"]
    gates = {seed: seed_results[seed]["gate"] for seed in seeds}

    criteria = {
        "a_beats_b1_card1": {
            "passed": float(primary_significance["ci_lower_95"]) > 0.0,
            "evidence": {
                "encoder_seed": primary,
                "observed_delta": float(primary_significance["observed_delta"]),
                "ci_95": [
                    float(primary_significance["ci_lower_95"]),
                    float(primary_significance["ci_upper_95"]),
                ],
            },
        },
        "b_beats_width_null": {
            "passed": float(real_vs_shuffled["ci_lower_95"]) > 0.0,
            "evidence": {
                "observed_delta": float(real_vs_shuffled["observed_delta"]),
                "ci_95": [
                    float(real_vs_shuffled["ci_lower_95"]),
                    float(real_vs_shuffled["ci_upper_95"]),
                ],
            },
        },
        "c_leakage_gate": {
            "passed": all(bool(gate["usable"]) for gate in gates.values()),
            "evidence": {
                f"seed{seed}": {
                    "usable": bool(gate["usable"]),
                    "probe_gap": float(gate["informativeness_gap"]["gap"]),
                    "provenance_sound": bool(gate["cross_fit_provenance"]["provenance_sound"]),
                }
                for seed, gate in gates.items()
            },
        },
        "d_sign_consistent": {
            "passed": all(delta > 0.0 for delta in deltas.values()),
            "evidence": {f"seed{seed}": delta for seed, delta in deltas.items()},
        },
    }
    for key, entry in criteria.items():
        entry["passed"] = bool(entry["passed"])
        entry["rule"] = prereg["criteria"][key]
    failed = [key for key, entry in criteria.items() if not entry["passed"]]

    mean_delta = float(np.mean(list(deltas.values())))
    return {
        "decidable": True,
        "verdict": VERDICT_NEGATIVE_STANDS if failed else VERDICT_WIN,
        "failed_criteria": failed,
        "criteria": criteria,
        "seed_panel": {
            "observed_delta_by_seed": {f"seed{seed}": delta for seed, delta in deltas.items()},
            "mean_observed_delta": mean_delta,
            "std_observed_delta": float(np.std(list(deltas.values()), ddof=1)),
        },
        "stage2_gate": {
            "rule": prereg["stage2_gate"]["directional_win"],
            "directional_win": mean_delta > 0.0,
        },
        "count_blind_attribution_permitted": not failed,
    }


def attribution_diagnostic(prereg: dict[str, Any], primary: dict[str, Any]) -> dict[str, Any]:
    seed = prereg["design"]["primary_encoder_seed"]
    blind_run = encoder_backed_run(seed, COUNT_BLIND)
    blind = load_run_result(blind_run)
    if blind is None:
        return {"run": False, "rule": prereg["diagnostics"]["count_blind_attribution"]}
    with_counts_vs_blind = compare_variants(
        candidate_label=f"g1v2_seed{seed}",
        candidate_predictions_path=primary["validation_predictions_path"],
        reference_label=f"g1v2_{blind_run.name}",
        reference_predictions_path=blind["validation_predictions_path"],
    )
    return {
        "run": True,
        "gating": False,
        "rule": prereg["diagnostics"]["count_blind_attribution"],
        "count_blind_delta_vs_b1_card1": float(blind["significance"]["observed_delta"]),
        "with_counts_vs_count_blind": with_counts_vs_blind,
    }


def build_seed_table(
    prereg: dict[str, Any], seed_results: dict[int, dict[str, Any] | None]
) -> pd.DataFrame:
    rows = []
    for seed in prereg["design"]["encoder_seeds"]:
        result = seed_results.get(seed)
        if result is None:
            rows.append({"encoder_seed": seed, "complete": False})
            continue
        significance = result["significance"]
        gate = result["gate"]
        rows.append(
            {
                "encoder_seed": seed,
                "complete": True,
                "pr_auc": float(result["metrics"]["pr_auc"]),
                "delta_vs_b1_card1": float(significance["observed_delta"]),
                "ci_lower_95": float(significance["ci_lower_95"]),
                "ci_upper_95": float(significance["ci_upper_95"]),
                "leakage_gate_usable": bool(gate["usable"]),
                "probe_gap": float(gate["informativeness_gap"]["gap"]),
                "partition_alignment_mean_gap": float(
                    gate["partition_alignment_diagnostic"]["mean_gap"]
                ),
            }
        )
    return pd.DataFrame(rows)


def selection_bias_screen_from_curves(
    reference_curve: pd.DataFrame,
    variant_curve: pd.DataFrame,
    noise_floor: float,
) -> dict[str, Any]:
    """The argmax/matched/plateau screen on one comparison, JSON-safe.

    Both arms here train the same fixed budget, so the argmax and matched
    estimates coincide by construction; the plateau delta is the informative read.
    """
    screen = compare_estimators(reference_curve, variant_curve, noise_floor=noise_floor)
    return {
        **{
            key: bool(value) if isinstance(value, np.bool_) else value
            for key, value in screen.items()
        },
        "gating": False,
    }


def selection_bias_screen(run: G1V2Run) -> dict[str, Any] | None:
    reference_path = resolve_fixed_budget_run_paths(REFERENCE_RUN_NAME)["learning_curve"]
    variant_path = run.paths()["learning_curve"]
    if not reference_path.exists() or not variant_path.exists():
        return None
    noise_floor, source = load_noise_floor()
    return {
        **selection_bias_screen_from_curves(
            pd.read_csv(reference_path), pd.read_csv(variant_path), noise_floor
        ),
        "noise_floor_source": source,
        "reference_learning_curve": repository_relative(reference_path),
        "variant_learning_curve": repository_relative(variant_path),
    }


def build_verdict() -> tuple[dict[str, Any], pd.DataFrame]:
    prereg = load_g1v2_preregistration()
    design = prereg["design"]
    if design["encoder_readout"] != READOUT:
        raise ValueError(
            f"The encoder module's readout ({READOUT}) is not the registered one "
            f"({design['encoder_readout']})."
        )

    seed_results: dict[int, dict[str, Any] | None] = {}
    for seed in design["encoder_seeds"]:
        run = encoder_backed_run(seed)
        result = load_run_result(run)
        if result is not None:
            assert_matches_preregistration(result, prereg, seed)
            result["gate"] = read_json(run.leakage_gate_path)
        seed_results[seed] = result

    primary_seed = design["primary_encoder_seed"]
    primary = seed_results[primary_seed]
    null_run = shuffled_run(primary_seed)
    null = load_run_result(null_run)
    real_vs_shuffled = None
    if primary is not None and null is not None:
        print("Paired bootstrap: real block vs width null...")
        real_vs_shuffled = compare_variants(
            candidate_label=f"g1v2_seed{primary_seed}",
            candidate_predictions_path=primary["validation_predictions_path"],
            reference_label=f"g1v2_{null_run.name}",
            reference_predictions_path=null["validation_predictions_path"],
            n_resamples=DEFAULT_N_RESAMPLES,
        )

    outcome = evaluate_criteria(prereg, seed_results, real_vs_shuffled)
    verdict = {
        "report_name": "G1-v2 Stage 1 verdict (cardinality-aware card1 encoder)",
        "preregistration_path": repository_relative(PREREGISTRATION_PATH),
        "preregistration_sha256": file_sha256(PREREGISTRATION_PATH),
        "evaluation_split": "validation",
        "test_evaluated": False,
        **outcome,
        "width_null": {
            "run": null_run.name,
            "delta_vs_b1_card1": (
                None if null is None else float(null["significance"]["observed_delta"])
            ),
            "real_vs_shuffled": real_vs_shuffled,
        },
        "selection_bias_screen": selection_bias_screen(encoder_backed_run(primary_seed)),
        "count_blind_attribution": (
            attribution_diagnostic(prereg, primary)
            if outcome.get("verdict") == VERDICT_WIN
            else {"run": False, "rule": prereg["diagnostics"]["count_blind_attribution"]}
        ),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return verdict, build_seed_table(prereg, seed_results)


def main() -> None:
    verdict, table = build_verdict()
    VERDICT_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_json(VERDICT_PATH, verdict)
    table.to_csv(SEED_TABLE_PATH, index=False)

    print("\nG1-v2 seed panel\n")
    print(table.to_string(index=False))
    if not verdict["decidable"]:
        print(f"\nVerdict: NOT YET DECIDABLE -- missing {verdict['missing_runs']}")
    else:
        for key, entry in verdict["criteria"].items():
            print(f"  {key}: {'PASS' if entry['passed'] else 'FAIL'}")
        print(f"\nVerdict: {verdict['verdict']}")
        print(f"Stage 2 directional gate: {verdict['stage2_gate']['directional_win']}")
    print(f"\nVerdict written: {VERDICT_PATH}")


if __name__ == "__main__":
    main()
