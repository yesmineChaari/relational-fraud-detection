"""The one place the repository root and its shared locations are defined.

Twenty-nine modules previously derived `ROOT_DIR` from their own `__file__` and
rebuilt the same dataset paths independently. That worked, but it meant every
new stage added another copy of the same three lines, and a module moved between
packages would silently resolve to the wrong root — `parents[2]` is correct from
`src/models/x.py` and wrong from `src/x.py`, and nothing catches the difference
until a path fails to exist.

Deliberately narrow. This module holds the root, the directory anchors beneath
it, and the dataset locations that more than one stage refers to. It does **not**
absorb every path constant in the project: a module's own report and artifact
paths belong with the module that writes them, and pulling them here would turn
one file into a registry of everything and couple every stage to every other.

It deliberately holds no helper functions either. `repository_relative` lives
in the baseline trainer, where sixteen modules already import it and rely on
its raising behaviour for paths outside the repository; a second, differently
behaved copy here would recreate the duplication this module exists to remove.

It also holds no thresholds, no hyperparameters and no expected row counts.
Those are invariants the code asserts against rather than settings, and the
friction of changing them is the point.
"""

from __future__ import annotations

from pathlib import Path

# src/config/paths.py -> src/config -> src -> repository root
ROOT_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

REPORTS_DIR = ROOT_DIR / "reports"
MODELS_DIR = ROOT_DIR / "models"
CONFIGS_DIR = ROOT_DIR / "configs"

# Raw inputs. The competition's unlabelled holdout files are deliberately absent:
# this project's test partition is carved from the labelled training file.
RAW_TRANSACTION_PATH = RAW_DATA_DIR / "train_transaction.csv"
RAW_IDENTITY_PATH = RAW_DATA_DIR / "train_identity.csv"

# Built once and consumed by every downstream stage.
MODEL_DATASET_PATH = PROCESSED_DATA_DIR / "model_dataset.parquet"
SPLIT_ASSIGNMENT_PATH = PROCESSED_DATA_DIR / "split_assignment.parquet"

SCREENING_CONFIG_PATH = CONFIGS_DIR / "screening.json"
