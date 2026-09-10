"""One entrypoint that runs the pipeline, or any stage, with prerequisites checked.

Reproducing this project meant running a dozen modules in an order that existed
only in the README's numbering. Nothing enforced it. Running the relational
trainer before the feature builder failed somewhere inside a merge; running the
screening before the audit failed on a missing CSV. Each module validates its own
inputs well, but the failure surfaced late and read as a file error rather than
"you skipped a step".

This module fixes the ordering in one place and checks it before any work starts.

Why a Python entrypoint rather than a Makefile or a task runner. The project is
Python-only and is developed on Windows, where `make` is not a given; a Makefile
would add a toolchain dependency to the one command a newcomer is most likely to
run first. A task runner would add a third-party dependency to a pinned stack for
a job the standard library does adequately.

Why subprocesses rather than importing each stage and calling `main()`. One stage
(`data/inspect_raw.py`) resolves its paths relative to the working directory, so
running it in-process from elsewhere would silently write to the wrong place.
Launching every stage with `cwd` set to the repository root makes that correct
without editing a working module, keeps a crashing stage from taking the
orchestrator down with it, and keeps this file honestly a sequencer: it never
imports stage logic, so it cannot accidentally reimplement any of it.

Idempotency, and an honest limit on it. A stage whose outputs all exist is
skipped unless forced. It would be better to skip only when the *inputs* are also
unchanged, and the graph stages do record input hashes in their manifests — but
none of the stages sequenced here do. Output-existence is therefore the available
signal, and `--force` is the escape hatch. Presenting hash-based freshness we
cannot actually compute would be worse than saying plainly what this checks.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from src.config.paths import (
    MODEL_DATASET_PATH,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
    ROOT_DIR,
    SPLIT_ASSIGNMENT_PATH,
)

RELATIONS = ("card1", "card1_card2")


@dataclass(frozen=True)
class Stage:
    """One pipeline step: what to run, what it produces, what it needs first."""

    name: str
    description: str
    commands: tuple[tuple[str, ...], ...]
    outputs: tuple[Path, ...]
    requires: tuple[str, ...] = field(default=())

    def is_complete(self) -> bool:
        return bool(self.outputs) and all(path.exists() for path in self.outputs)

    def missing_outputs(self) -> list[Path]:
        return [path for path in self.outputs if not path.exists()]


def _relational_feature_outputs() -> tuple[Path, ...]:
    return tuple(
        PROCESSED_DATA_DIR / f"relational_features_{relation}.parquet" for relation in RELATIONS
    )


def _b1_model_outputs() -> tuple[Path, ...]:
    return tuple(ROOT_DIR / "models" / f"lightgbm_b1_{relation}.txt" for relation in RELATIONS)


# Declaration order is execution order. The dependency the ticket had backwards:
# the temporal split *writes* the split manifest and the model dataset *reads*
# it, so the split must run first.
STAGES: dict[str, Stage] = {
    "verify_raw": Stage(
        name="verify_raw",
        description="Check the raw inputs are present, complete and the expected files.",
        commands=(("-m", "src.data.verify_raw_inputs"),),
        # A gate, not a producer. Its own outputs are empty, so it is never
        # skipped as "already complete" -- verifying twice costs seconds.
        outputs=(),
    ),
    "profile": Stage(
        name="profile",
        description="Column-level profile of every raw column.",
        commands=(("data/inspect_raw.py",),),
        outputs=(REPORTS_DIR / "data_profile" / "transaction_columns.csv",),
        requires=("verify_raw",),
    ),
    "temporal_split": Stage(
        name="temporal_split",
        description="Carve the train/validation/test partitions by time.",
        commands=(("-m", "src.data.make_temporal_split"),),
        outputs=(SPLIT_ASSIGNMENT_PATH, REPORTS_DIR / "split_summary.csv"),
        requires=("verify_raw",),
    ),
    "model_dataset": Stage(
        name="model_dataset",
        description="Join transactions and identity into the modelling frame.",
        commands=(("-m", "src.data.build_model_dataset"),),
        outputs=(MODEL_DATASET_PATH,),
        requires=("temporal_split",),
    ),
    "relational_audit": Stage(
        name="relational_audit",
        description="Entity and graph diagnostics for the candidate relations.",
        commands=(("-m", "src.graph.analyze_relations"),),
        outputs=(
            REPORTS_DIR / "relational_audit" / "entity_diagnostics.csv",
            REPORTS_DIR / "relational_audit" / "graph_diagnostics.csv",
        ),
        requires=("verify_raw",),
    ),
    "screening": Stage(
        name="screening",
        description="Train-only structural and signal screening of the relations.",
        commands=(("-m", "src.features.screen_relations"),),
        outputs=(REPORTS_DIR / "relational_screening" / "candidate_selection.json",),
        requires=("relational_audit", "model_dataset"),
    ),
    "relational_features": Stage(
        name="relational_features",
        description="Build the four history summaries for each preferred relation.",
        commands=tuple(
            ("-m", "src.features.build_relational_features", "--relation", relation)
            for relation in RELATIONS
        ),
        outputs=_relational_feature_outputs(),
        requires=("screening",),
    ),
    "train_b1": Stage(
        name="train_b1",
        description="Train the relational variant per relation.",
        commands=tuple(
            ("-m", "src.models.train_lightgbm_relational", "--relation", relation)
            for relation in RELATIONS
        ),
        outputs=_b1_model_outputs(),
        requires=("relational_features",),
    ),
    "compare": Stage(
        name="compare",
        description="Paired-bootstrap intervals and the cross-relation comparison.",
        commands=(
            ("-m", "src.models.compare_b1_significance"),
            ("-m", "src.models.compare_b1_variants"),
        ),
        outputs=(
            REPORTS_DIR / "b1" / "b1_significance.json",
            REPORTS_DIR / "b1" / "b1_cross_relation_summary.json",
        ),
        requires=("train_b1",),
    ),
}

STAGE_ORDER: tuple[str, ...] = tuple(STAGES)


def unmet_prerequisites(name: str) -> list[tuple[str, Path]]:
    """Upstream stages that have not produced their outputs, transitively.

    Returns the *stage* that is missing rather than the file, because "you have
    not run the screening" is actionable and "candidate_selection.json not
    found" is a puzzle.
    """
    if name not in STAGES:
        raise KeyError(f"Unknown stage {name!r}. Known stages: {', '.join(STAGE_ORDER)}.")

    unmet: list[tuple[str, Path]] = []
    seen: set[str] = set()

    def walk(stage_name: str) -> None:
        for required in STAGES[stage_name].requires:
            if required in seen:
                continue
            seen.add(required)
            walk(required)
            stage = STAGES[required]
            missing = stage.missing_outputs()
            if stage.outputs and missing:
                unmet.append((required, missing[0]))

    walk(name)
    return unmet


def resolve_plan(selected: str | None, run_from: str | None, run_all: bool) -> list[str]:
    """The stages to run, in declaration order."""
    if run_all:
        return list(STAGE_ORDER)
    if run_from is not None:
        if run_from not in STAGES:
            raise KeyError(f"Unknown stage {run_from!r}.")
        return list(STAGE_ORDER[STAGE_ORDER.index(run_from) :])
    if selected is not None:
        if selected not in STAGES:
            raise KeyError(f"Unknown stage {selected!r}.")
        return [selected]
    return list(STAGE_ORDER)


def describe_unmet(name: str, unmet: list[tuple[str, Path]]) -> str:
    lines = [f"Stage {name!r} cannot run yet:"]
    for stage_name, missing in unmet:
        lines.append(
            f"  - stage {stage_name!r} has not been run "
            f"(missing {missing.relative_to(ROOT_DIR).as_posix()})"
        )
    first = unmet[0][0]
    lines.append(f"Run it first:  python -m src.pipeline --from {first}")
    return "\n".join(lines)


def run_stage(stage: Stage, force: bool = False, dry_run: bool = False) -> str:
    """Run one stage. Returns 'skipped', 'ran' or 'planned'."""
    unmet = unmet_prerequisites(stage.name)
    if unmet:
        raise SystemExit(describe_unmet(stage.name, unmet))

    if stage.is_complete() and not force:
        print(f"[{stage.name}] already complete; skipping (use --force to rerun).")
        return "skipped"

    for command in stage.commands:
        printable = " ".join(("python", *command))
        if dry_run:
            print(f"[{stage.name}] would run: {printable}")
            continue
        print(f"[{stage.name}] {printable}")
        # cwd is the repository root so a stage resolving paths relatively
        # writes where it is expected to, whatever directory the user is in.
        result = subprocess.run([sys.executable, *command], cwd=ROOT_DIR, check=False)
        if result.returncode != 0:
            raise SystemExit(f"[{stage.name}] failed: {printable} (exit {result.returncode})")
    return "planned" if dry_run else "ran"


def print_stages() -> None:
    print(f"{'stage':22s} {'status':10s} description")
    for name in STAGE_ORDER:
        stage = STAGES[name]
        if not stage.outputs:
            status = "gate"
        else:
            status = "complete" if stage.is_complete() else "pending"
        print(f"{name:22s} {status:10s} {stage.description}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="python -m src.pipeline",
        description="Run the pipeline, or any stage, with prerequisites checked first.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--stage", help="Run exactly this stage.")
    group.add_argument("--from", dest="run_from", help="Run this stage and everything after it.")
    group.add_argument("--all", action="store_true", help="Run every stage (the default).")
    parser.add_argument("--list", action="store_true", help="Show stages and their status.")
    parser.add_argument("--force", action="store_true", help="Rerun stages already complete.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running.")
    args = parser.parse_args(argv)

    if args.list:
        print_stages()
        return

    plan = resolve_plan(args.stage, args.run_from, args.all)
    print(f"Plan: {' -> '.join(plan)}\n")
    outcomes = {name: run_stage(STAGES[name], args.force, args.dry_run) for name in plan}
    ran = sum(1 for outcome in outcomes.values() if outcome == "ran")
    skipped = sum(1 for outcome in outcomes.values() if outcome == "skipped")
    print(f"\nDone. {ran} stage(s) ran, {skipped} skipped.")


if __name__ == "__main__":
    main()
