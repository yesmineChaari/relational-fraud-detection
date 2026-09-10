from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

from src.data.verify_raw_inputs import (
    DATASET_SOURCE,
    LICENCE_NOTE,
    REQUIRED_FILES,
    UNUSED_COMPETITION_FILES,
    count_data_rows,
    file_sha256,
    verify,
    verify_file,
)


def write_csv(path: Path, data_rows: int, trailing_newline: bool = True) -> None:
    body = "TransactionID,isFraud\n" + "".join(f"{i},0\n" for i in range(data_rows))
    if not trailing_newline and body.endswith("\n"):
        body = body[:-1]
    path.write_text(body, encoding="utf-8")


def spec_for(path: Path, data_rows: int) -> dict:
    return {
        "path": path,
        "sha256": file_sha256(path),
        "data_rows": data_rows,
        "bytes": path.stat().st_size,
        "role": "fixture",
    }


class RowCountingTests(unittest.TestCase):
    def test_header_is_excluded_from_the_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            write_csv(path, 10)
            self.assertEqual(count_data_rows(path), 10)

    def test_a_file_without_a_trailing_newline_is_not_undercounted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            write_csv(path, 10, trailing_newline=False)
            self.assertEqual(count_data_rows(path), 10)

    def test_a_header_only_file_has_no_data_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            write_csv(path, 0)
            self.assertEqual(count_data_rows(path), 0)

    def test_counting_is_independent_of_chunk_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            write_csv(path, 500)
            # A chunk boundary must not land between a row and its newline.
            self.assertEqual(count_data_rows(path, chunk_bytes=7), 500)
            self.assertEqual(count_data_rows(path, chunk_bytes=1 << 20), 500)


class ChecksumTests(unittest.TestCase):
    def test_digest_matches_hashlib(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            write_csv(path, 20)
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(file_sha256(path), expected)

    def test_digest_is_independent_of_chunk_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            write_csv(path, 200)
            self.assertEqual(file_sha256(path, chunk_bytes=3), file_sha256(path))


class VerifyFileTests(unittest.TestCase):
    def test_a_sound_file_produces_no_failures(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            write_csv(path, 50)
            self.assertEqual(verify_file("f.csv", spec_for(path, 50)), [])

    def test_a_missing_file_is_reported_with_where_to_get_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "absent.csv"
            spec = {"path": path, "sha256": "x", "data_rows": 1, "bytes": 1, "role": ""}
            failures = verify_file("absent.csv", spec)
            self.assertEqual(len(failures), 1)
            self.assertIn("missing", failures[0])
            self.assertIn(DATASET_SOURCE, failures[0])

    def test_a_truncated_file_is_caught_by_the_row_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            write_csv(path, 50)
            spec = spec_for(path, 50)
            write_csv(path, 30)  # truncate after recording the expectation
            failures = verify_file("f.csv", spec)
            self.assertTrue(any("truncated or partially downloaded" in f for f in failures))

    def test_same_length_different_content_is_caught_by_the_checksum(self):
        # The case row counts and byte sizes both miss: right shape, wrong file.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            write_csv(path, 50)
            spec = spec_for(path, 50)
            body = path.read_text(encoding="utf-8").replace("0\n", "1\n", 1)
            path.write_text(body, encoding="utf-8")
            self.assertEqual(path.stat().st_size, spec["bytes"])
            self.assertEqual(count_data_rows(path), 50)
            failures = verify_file("f.csv", spec)
            self.assertTrue(any("sha256" in f for f in failures))

    def test_checksums_can_be_skipped_for_a_fast_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            write_csv(path, 50)
            spec = spec_for(path, 50)
            body = path.read_text(encoding="utf-8").replace("0\n", "1\n", 1)
            path.write_text(body, encoding="utf-8")
            self.assertEqual(verify_file("f.csv", spec, check_checksum=False), [])
            self.assertTrue(verify_file("f.csv", spec, check_checksum=True))


class VerifyTests(unittest.TestCase):
    def test_failures_from_every_required_file_are_collected(self):
        with tempfile.TemporaryDirectory() as tmp:
            required = {
                name: {
                    "path": Path(tmp) / name,
                    "sha256": "x",
                    "data_rows": 1,
                    "bytes": 1,
                    "role": "",
                }
                for name in ("a.csv", "b.csv")
            }
            result = verify(required)
            self.assertFalse(result["ok"])
            self.assertEqual(len(result["failures"]), 2)

    def test_a_sound_set_reports_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "f.csv"
            write_csv(path, 12)
            result = verify({"f.csv": spec_for(path, 12)})
            self.assertTrue(result["ok"])
            self.assertEqual(result["failures"], [])
            self.assertTrue(result["checksums_verified"])


class RecordedExpectationsTests(unittest.TestCase):
    def test_both_required_files_are_declared(self):
        self.assertEqual(sorted(REQUIRED_FILES), ["train_identity.csv", "train_transaction.csv"])

    def test_recorded_row_counts_match_the_documented_dataset(self):
        self.assertEqual(REQUIRED_FILES["train_transaction.csv"]["data_rows"], 590_540)
        self.assertEqual(REQUIRED_FILES["train_identity.csv"]["data_rows"], 144_233)

    def test_every_required_file_records_a_full_sha256(self):
        for expected in REQUIRED_FILES.values():
            self.assertEqual(len(expected["sha256"]), 64)
            self.assertTrue(all(c in "0123456789abcdef" for c in expected["sha256"]))

    def test_the_unused_competition_files_are_named(self):
        # Mistaking the competition holdout for this project's test split is the
        # error the ticket called out; the distinction is recorded in code.
        self.assertIn("test_transaction.csv", UNUSED_COMPETITION_FILES)
        self.assertIn("sample_submission.csv", UNUSED_COMPETITION_FILES)
        self.assertIn("unlabelled", UNUSED_COMPETITION_FILES["test_transaction.csv"].lower())

    def test_the_licence_note_states_no_redistribution(self):
        self.assertIn("redistribut", LICENCE_NOTE.lower())
        self.assertIn("gitignored", LICENCE_NOTE)

    def test_the_real_files_verify_when_present(self):
        missing = [n for n, e in REQUIRED_FILES.items() if not e["path"].exists()]
        if missing:
            self.skipTest(f"Raw inputs not present: {missing}")
        result = verify(check_checksum=False)
        self.assertTrue(result["ok"], result["failures"])


if __name__ == "__main__":
    unittest.main()
