"""Independent tiny-data tests for reproducible, immutable evaluation subsets."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import random
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from src.evaluation.subsets import build_subset, main, write_subset
from src.evaluation.validate import load_context


class SubsetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.data = self.root / "processed"
        self.data.mkdir()
        self.catalog = self.data / "catalog.csv"
        self.profiles = self.data / "validation_profiles.jsonl"
        self.targets = self.data / "validation_targets.csv"
        self.output = self.root / "configs" / "validation_users_v1.json"
        self._write_csv(self.catalog, ["movie_id", "title"], [[10, "A"], [20, "B"], [30, "C"]])
        # Deliberately unsorted source IDs: the sampler must sort before sampling.
        self.records = [
            {"user_id": user_id, "split": "validation", "history_movie_ids": [90],
             "history_latest_timestamp": 99}
            for user_id in [40, 10, 30, 20]
        ]
        self._write_profiles()
        self.rows = [[10, 10, 4, 101, 1], [20, 20, 5, 102, 1],
                     [30, 30, 4.5, 103, 1], [40, 10, 4, 104, 1],
                     # Out-of-catalog-only user is legitimately absent from profiles.
                     [50, 999, 5, 105, 0]]
        self._write_targets()
        self._refresh_manifest()

    @staticmethod
    def _write_csv(path, headers, rows):
        with path.open("w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(headers)
            writer.writerows(rows)

    def _write_profiles(self):
        self.profiles.write_text("".join(json.dumps(record) + "\n" for record in self.records), encoding="utf-8")

    def _write_targets(self):
        self._write_csv(self.targets, ["user_id", "movie_id", "rating", "timestamp", "in_catalog"], self.rows)

    def _refresh_manifest(self, *, version="tiny-data-v1"):
        files = [self.catalog, self.profiles, self.targets]
        stats = self.data / "statistics.json"
        if stats.exists():
            files.append(stats)
        manifest = {
            "version": version,
            "file_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        }
        (self.data / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    def _build(self, **kwargs):
        return build_subset(self.data, size=kwargs.pop("size", 2), **kwargs)

    def _cli(self, *extra):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = main(["--data-dir", str(self.data), "--size", "2", "--output", str(self.output), *extra])
        return result, stdout.getvalue(), stderr.getvalue()

    def test_seeded_selection_uses_sorted_eligible_population_and_sorted_output(self):
        result = self._build(seed=17)
        self.assertEqual(result["user_ids"], sorted(random.Random(17).sample([10, 20, 30, 40], 2)))
        self.assertEqual(result["selection"]["eligible_users"], 4)
        self.assertEqual(result["selection"]["size"], 2)
        self.assertEqual(result["selection"]["seed"], 17)
        self.assertEqual(result["selection"]["algorithm"], "python.random.Random.sample_from_sorted_user_ids_v1")
        self.assertEqual(result["subset_version"], "evaluation-users-v1")
        self.assertEqual(result["dataset_version"], "tiny-data-v1")
        self.assertEqual(result["split"], "validation")

    def test_payload_and_file_hashes_match_documented_serialization(self):
        result = self._build()
        ids_bytes = (json.dumps(result["user_ids"], separators=(",", ":")) + "\n").encode("utf-8")
        self.assertEqual(result["user_ids_sha256"], hashlib.sha256(ids_bytes).hexdigest())
        actual_hash = write_subset(self.output, result)
        expected = (json.dumps(result, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
        self.assertEqual(self.output.read_bytes(), expected)
        self.assertEqual(actual_hash, hashlib.sha256(expected).hexdigest())
        self.assertNotIn("file_sha256", result)
        for name, path in {"manifest": self.data / "manifest.json", "catalog": self.catalog,
                           "profiles": self.profiles, "targets": self.targets}.items():
            self.assertEqual(result["source_sha256"][name], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_same_inputs_reproduce_same_payload_and_idempotent_output(self):
        first = self._build()
        initial_hash = write_subset(self.output, first)
        before = self.output.stat().st_mtime_ns
        self.assertEqual(first, self._build())
        self.assertEqual(initial_hash, write_subset(self.output, self._build()))
        self.assertEqual(before, self.output.stat().st_mtime_ns)

    def test_changed_existing_output_is_not_overwritten(self):
        first = self._build()
        write_subset(self.output, first)
        before = self.output.read_bytes()
        changed = self._build(seed=99)
        with self.assertRaisesRegex(ValueError, "new versioned output filename"):
            write_subset(self.output, changed)
        self.assertEqual(before, self.output.read_bytes())

    def test_future_test_targets_and_train_ratings_are_never_read(self):
        first = self._build()
        (self.data / "test_targets.csv").write_bytes(b"not a valid CSV\x00")
        (self.data / "train_ratings.csv").mkdir()
        self.assertEqual(first, self._build())

    def test_profile_text_quality_and_source_order_do_not_change_ids(self):
        first = self._build()["user_ids"]
        self.records.reverse()
        for record in self.records:
            record["text"] = ""
            record["history_count"] = 1000000
        self._write_profiles()
        self._refresh_manifest()
        self.assertEqual(first, self._build()["user_ids"])

    def test_size_and_seed_require_exact_integer_types(self):
        for value in [0, -1, 5, True, 2.0, "2"]:
            with self.subTest(size=value), self.assertRaises(ValueError):
                self._build(size=value)
        for value in [True, 2.0, "2", None]:
            with self.subTest(seed=value), self.assertRaises(ValueError):
                self._build(seed=value)
        self.assertEqual(len(self._build(size=4, seed=-1)["user_ids"]), 4)

    def test_invalid_split_and_missing_manifest_fail(self):
        with self.assertRaisesRegex(ValueError, "split"):
            self._build(split="train")
        (self.data / "manifest.json").unlink()
        with self.assertRaises(OSError):
            self._build()

    def test_nonempty_arbitrary_dataset_version_is_allowed(self):
        self._refresh_manifest(version="team-revision-2")
        self.assertEqual(self._build()["dataset_version"], "team-revision-2")
        for value in ["", " ", 3, None, True]:
            with self.subTest(version=value):
                self._refresh_manifest(version=value)
                with self.assertRaisesRegex(ValueError, "nonempty string version"):
                    self._build()

    def test_source_checksum_drift_fails(self):
        self.profiles.write_text(self.profiles.read_text() + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            self._build()

    def test_missing_manifest_hash_fails(self):
        path = self.data / "manifest.json"
        manifest = json.loads(path.read_text())
        del manifest["file_sha256"][self.profiles.name]
        path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "missing or invalid SHA-256"):
            self._build()

    def test_duplicate_json_keys_fail_even_with_matching_checksum(self):
        self.profiles.write_text('{"user_id":10,"user_id":20}\n', encoding="utf-8")
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self._build()

    def test_profile_ids_reject_booleans_and_duplicates(self):
        for replacement in [True, 0, "10"]:
            with self.subTest(replacement=replacement):
                old = self.records[0]["user_id"]
                self.records[0]["user_id"] = replacement
                self._write_profiles()
                self._refresh_manifest()
                with self.assertRaisesRegex(ValueError, "positive integer user_id"):
                    self._build()
                self.records[0]["user_id"] = old
        self.records.append(dict(self.records[0]))
        self._write_profiles()
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate profile"):
            self._build()

    def test_bad_histories_are_rejected(self):
        for history in [[True], [90, 90], [-1], None]:
            with self.subTest(history=history):
                self.records[0]["history_movie_ids"] = history
                self._write_profiles()
                self._refresh_manifest()
                with self.assertRaisesRegex(ValueError, "history_movie_ids"):
                    self._build()

    def test_profile_and_target_eligible_users_must_match(self):
        self.records.pop()
        self._write_profiles()
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "missing profiles=\\[20\\]"):
            self._build()

    def test_profiles_without_in_catalog_targets_are_rejected(self):
        self.records.append({"user_id": 50, "split": "validation", "history_movie_ids": [90]})
        self._write_profiles()
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "profiles without eligible targets=\\[50\\]"):
            self._build()

    def test_profile_split_and_target_overlap_are_rejected(self):
        self.records[0]["split"] = "test"
        self._write_profiles()
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "profile split"):
            self._build()
        self.records[0]["split"] = "validation"
        self.records[0]["history_movie_ids"] = [10, 90]
        self._write_profiles()
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "already watched"):
            self._build()

    def test_target_values_and_membership_are_checked(self):
        original = list(self.rows[0])
        cases = [(0, "True", "user_id"), (1, 0, "movie_id"),
                 (2, "nan", "finite"), (2, 3.5, "finite"), (2, 5.1, "finite"),
                 (4, 2, "in_catalog must be"), (4, 0, "disagrees")]
        for index, value, message in cases:
            with self.subTest(index=index, value=value):
                self.rows[0] = list(original)
                self.rows[0][index] = value
                self._write_targets()
                self._refresh_manifest()
                with self.assertRaisesRegex(ValueError, message):
                    self._build()

    def test_duplicate_targets_and_catalog_ids_fail(self):
        self.rows.append(list(self.rows[0]))
        self._write_targets()
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate positive target"):
            self._build()
        self.rows.pop()
        self._write_targets()
        self._write_csv(self.catalog, ["movie_id"], [[10], [10]])
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "duplicate catalog"):
            self._build()

    def test_optional_statistics_verify_checksum_counts_and_cutoff(self):
        stats = {"catalog_movies": 3, "validation_cutoff_unix": 100,
                 "evaluation": {"validation": {"users_with_in_catalog_positive": 4}}}
        path = self.data / "statistics.json"
        path.write_text(json.dumps(stats), encoding="utf-8")
        self._refresh_manifest()
        result = self._build()
        self.assertEqual(result["source_sha256"]["statistics"], hashlib.sha256(path.read_bytes()).hexdigest())
        stats["evaluation"]["validation"]["users_with_in_catalog_positive"] = 5
        path.write_text(json.dumps(stats), encoding="utf-8")
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "evaluable-user count"):
            self._build()
        stats["evaluation"]["validation"]["users_with_in_catalog_positive"] = 4
        path.write_text(json.dumps(stats), encoding="utf-8")
        self.records[0]["history_latest_timestamp"] = 100
        self._write_profiles()
        self._refresh_manifest()
        with self.assertRaisesRegex(ValueError, "must precede"):
            self._build()

    def test_cli_creates_validator_compatible_subset_and_reports_full_file_hash(self):
        status, stdout, stderr = self._cli()
        self.assertEqual(status, 0, stderr)
        report = json.loads(stdout)
        self.assertEqual(report["selected_users"], 2)
        self.assertEqual(report["file_sha256"], hashlib.sha256(self.output.read_bytes()).hexdigest())
        context = load_context(self.catalog, self.profiles, self.output, "validation")
        self.assertEqual(context.expected_user_ids, set(json.loads(self.output.read_text())["user_ids"]))

    def test_cli_invalid_input_is_readable_exit_2_and_preserves_old_file(self):
        status, _, stderr = self._cli()
        self.assertEqual(status, 0, stderr)
        before = self.output.read_bytes()
        status, stdout, stderr = self._cli("--seed", "2")
        self.assertEqual(status, 2)
        self.assertEqual(stdout, "")
        self.assertIn("new versioned output filename", stderr)
        self.assertNotIn("Traceback", stderr)
        self.assertEqual(self.output.read_bytes(), before)

    def test_cli_cannot_overwrite_a_data_input_or_hardlink(self):
        before = self.catalog.read_bytes()
        status, _, stderr = self._cli("--output", str(self.catalog))
        self.assertEqual(status, 2)
        self.assertIn("protect processed data inputs", stderr)
        self.assertEqual(self.catalog.read_bytes(), before)
        alias = self.root / "catalog-alias.json"
        alias.hardlink_to(self.catalog)
        status, _, stderr = self._cli("--output", str(alias))
        self.assertEqual(status, 2)
        self.assertIn("alias a data input", stderr)
        self.assertEqual(self.catalog.read_bytes(), before)

    def test_test_split_reads_its_own_population(self):
        profiles = self.data / "test_profiles.jsonl"
        targets = self.data / "test_targets.csv"
        profiles.write_text(json.dumps({"user_id": 80, "split": "test", "history_movie_ids": [90]}) + "\n", encoding="utf-8")
        self._write_csv(targets, ["user_id", "movie_id", "rating", "in_catalog"], [[80, 30, 5, 1]])
        manifest_path = self.data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        for path in [profiles, targets]:
            manifest["file_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        self.assertEqual(self._build(split="test", size=1)["user_ids"], [80])


if __name__ == "__main__":
    unittest.main()
