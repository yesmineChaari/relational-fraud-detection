"""Run the pre-registered G1-v2 panel end to end, one process per step.

A step is a cross-fitted encoder run (roughly 80-90 minutes) or a fixed-budget
LightGBM run (roughly 20 minutes, ~4.5 GB peak). Each runs in its own
interpreter, strictly in sequence, so no two ever share memory: on this machine
a parallel or single-process sweep is an out-of-memory failure, not a speed-up.
Every training step resumes with --skip-existing, so re-invoking the panel after
an interruption picks up where it stopped.

Seeds come from configs/g1_v2_preregistration.json, never from the command
line: the panel is the registered one or it is not the panel. The count-blind
attribution arm is refused unless the verdict has already passed all four
criteria, as registered.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from typing import Any

from src.models.g1v2_preregistration import load_g1v2_preregistration

ENCODER_MODULE = "src.graph.train_graphsage_encoder_v2"
LIGHTGBM_MODULE = "src.models.train_lightgbm_g1v2"
CONTROLS_MODULE = "src.models.train_lightgbm_g1v2_controls"
VERDICT_MODULE = "src.models.compare_g1v2_verdict"


@dataclass(frozen=True)
class PanelStep:
    label: str
    module: str
    args: tuple[str, ...] = ()
    resumable: bool = True

    def command(self) -> list[str]:
        resume = ("--skip-existing",) if self.resumable else ()
        return [sys.executable, "-m", self.module, *self.args, *resume]


def planned_steps(prereg: dict[str, Any], attribution: bool = False) -> list[PanelStep]:
    """Primary seed first, its width null straight after, the other seeds, then the verdict."""
    design = prereg["design"]
    primary = design["primary_encoder_seed"]
    ordered = [primary, *(seed for seed in design["encoder_seeds"] if seed != primary)]

    steps: list[PanelStep] = []
    for seed in ordered:
        steps.append(PanelStep(f"encoder seed {seed}", ENCODER_MODULE, ("--seed", str(seed))))
        steps.append(PanelStep(f"lightgbm seed {seed}", LIGHTGBM_MODULE, ("--seed", str(seed))))
        if seed == primary:
            steps.append(PanelStep(f"width null (seed {seed} permuted)", CONTROLS_MODULE))
    if attribution:
        blind = ("--seed", str(primary), "--count-blind")
        steps.append(PanelStep(f"encoder count-blind seed {primary}", ENCODER_MODULE, blind))
        steps.append(PanelStep(f"lightgbm count-blind seed {primary}", LIGHTGBM_MODULE, blind))
    steps.append(PanelStep("verdict", VERDICT_MODULE, resumable=False))
    return steps


def attribution_permitted() -> bool:
    # Imported here so the driver does not hold torch in memory while its
    # child processes train.
    from src.models.compare_g1v2_verdict import VERDICT_PATH, VERDICT_WIN
    from src.models.train_lightgbm_relational import read_json

    return VERDICT_PATH.exists() and read_json(VERDICT_PATH).get("verdict") == VERDICT_WIN


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.models.train_g1v2_seed_panel",
        description="Run the pre-registered G1-v2 panel, one process per step.",
    )
    parser.add_argument("--list", action="store_true", help="Show the steps and exit.")
    parser.add_argument(
        "--attribution",
        action="store_true",
        help="Also run the count-blind arm; refused unless the verdict already passed.",
    )
    args = parser.parse_args(argv)

    if args.attribution and not attribution_permitted():
        raise SystemExit(
            "The count-blind attribution arm is pre-registered to run only after all "
            "four criteria pass, and the current verdict has not passed them."
        )
    steps = planned_steps(load_g1v2_preregistration(), attribution=args.attribution)
    if args.list:
        for index, step in enumerate(steps, start=1):
            print(f"{index:>2}. {step.label}: {' '.join(step.command()[1:])}")
        return

    for index, step in enumerate(steps, start=1):
        print(f"\n=== [{index}/{len(steps)}] {step.label} ===", flush=True)
        subprocess.run(step.command(), check=True)
    print("\nG1-v2 panel complete. Test set evaluated: NO")


if __name__ == "__main__":
    main()
