"""Verify the raw inputs before anything downstream reads them.

`data/raw/` is gitignored — the inputs are roughly 1.3 GB and are not ours to
redistribute — so a clean checkout has an empty directory and the pipeline's
first step fails somewhere deep inside a parser. This module turns that into an
explicit check with a message naming exactly what is wrong.

It verifies three things per required file, cheapest first, so a reader gets the
most obvious answer rather than the most precise one:

1. **Presence.** The file is missing entirely, which on a clean checkout is the
   expected state and is worth saying plainly rather than diagnosing.
2. **Row count.** Catches a truncated or partially downloaded file, which is the
   common failure and produces confusing downstream errors rather than obvious
   ones.
3. **Checksum.** Catches a file that is the right length but not the right
   content — a different revision, a re-exported copy, a corrupted transfer.

The recorded digests are of the exact files that produced every committed
report. A checksum mismatch does not necessarily mean the data is wrong, but it
does mean the published numbers cannot be expected to reproduce byte for byte,
and that is worth knowing before spending hours of training rather than after.

Row counting reads newlines from the raw bytes rather than parsing CSV. These
files have no embedded newlines inside quoted fields, and a full parse of a
683 MB file to answer "is it the right length" would cost more than the check
is worth.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any

from src.data.build_model_dataset import IDENTITY_PATH, ROOT_DIR, TRANSACTION_PATH

READ_CHUNK_BYTES = 8 << 20

DATASET_NAME = "IEEE-CIS Fraud Detection"
DATASET_SOURCE = "https://www.kaggle.com/competitions/ieee-fraud-detection/data"

# Measured from the files that produced every committed report.
REQUIRED_FILES: dict[str, dict[str, Any]] = {
    "train_transaction.csv": {
        "path": TRANSACTION_PATH,
        "sha256": "3a5c83ab6b3cc13dcabe5ffa9f522307fd5f7f7b6e6f6a60c32284ca6283d642",
        "data_rows": 590_540,
        "bytes": 683_351_067,
        "role": "Transactions and the isFraud label. Every split in this project is carved from it.",
    },
    "train_identity.csv": {
        "path": IDENTITY_PATH,
        "sha256": "b63c725d8377be90a995268d97f347c17d456b95db45807adcf9f59cd603c37c",
        "data_rows": 144_233,
        "bytes": 26_529_680,
        "role": "Device and identity attributes, left-joined onto transactions where present.",
    },
}

# Present in the competition download and deliberately unused. Recorded because
# mistaking them for this project's test partition is an easy and costly error:
# they are the competition's unlabelled holdout and carry no isFraud column at
# all. This project's test partition is carved from the labelled training file.
UNUSED_COMPETITION_FILES = {
    "test_transaction.csv": "Competition holdout, unlabelled. Not this project's test split.",
    "test_identity.csv": "Competition holdout identity rows, unlabelled.",
    "sample_submission.csv": "Competition submission template. No role here.",
}

LICENCE_NOTE = (
    "The dataset is distributed by Kaggle under the IEEE-CIS Fraud Detection "
    "competition rules, which govern access and redistribution. It is therefore "
    "not vendored into this repository and data/raw/ is gitignored. Download it "
    "yourself from the competition page after accepting those rules; this "
    "project redistributes none of it."
)


def file_sha256(path: Path, chunk_bytes: int = READ_CHUNK_BYTES) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def count_data_rows(path: Path, chunk_bytes: int = READ_CHUNK_BYTES) -> int:
    """Data rows, excluding the header, counted from raw newlines."""
    newlines = 0
    ends_with_newline = True
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            newlines += chunk.count(b"\n")
            ends_with_newline = chunk.endswith(b"\n")
    # A file without a trailing newline still holds one more row than newlines.
    total_lines = newlines if ends_with_newline else newlines + 1
    return max(total_lines - 1, 0)


def verify_file(name: str, expected: dict[str, Any], check_checksum: bool = True) -> list[str]:
    """Failures for one required file, cheapest check first, empty when sound."""
    path: Path = expected["path"]
    if not path.exists():
        return [
            f"{name}: missing. Expected at {path}. Download it from "
            f"{DATASET_SOURCE} and place it in data/raw/."
        ]

    failures = []
    actual_rows = count_data_rows(path)
    if actual_rows != expected["data_rows"]:
        failures.append(
            f"{name}: {actual_rows:,} data rows, expected {expected['data_rows']:,}. "
            f"The file looks truncated or partially downloaded."
        )

    actual_bytes = path.stat().st_size
    if actual_bytes != expected["bytes"]:
        failures.append(f"{name}: {actual_bytes:,} bytes, expected {expected['bytes']:,}.")

    if check_checksum:
        actual_digest = file_sha256(path)
        if actual_digest != expected["sha256"]:
            failures.append(
                f"{name}: sha256 {actual_digest} does not match the recorded "
                f"{expected['sha256']}. The committed reports were produced from a "
                f"file with the recorded digest, so they may not reproduce exactly."
            )
    return failures


def verify(
    required: dict[str, dict[str, Any]] | None = None,
    check_checksum: bool = True,
) -> dict[str, Any]:
    required = REQUIRED_FILES if required is None else required
    failures: list[str] = []
    for name, expected in required.items():
        failures.extend(verify_file(name, expected, check_checksum=check_checksum))
    return {
        "dataset": DATASET_NAME,
        "source": DATASET_SOURCE,
        "required_files": sorted(required),
        "unused_competition_files": sorted(UNUSED_COMPETITION_FILES),
        "checksums_verified": check_checksum,
        "ok": not failures,
        "failures": failures,
    }


def main() -> None:
    quick = "--skip-checksums" in sys.argv
    print(f"Verifying raw inputs for {DATASET_NAME} under {ROOT_DIR / 'data' / 'raw'}")
    if quick:
        print("Checksums skipped by request; presence and row counts still checked.")
    result = verify(check_checksum=not quick)

    if result["ok"]:
        for name, expected in REQUIRED_FILES.items():
            print(f"  OK  {name:24s} {expected['data_rows']:>9,} rows")
        print("\nRaw inputs verified. The pipeline can run.")
        return

    print("\nRaw inputs are not usable:\n")
    for failure in result["failures"]:
        print(f"  - {failure}")
    print(f"\n{LICENCE_NOTE}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
