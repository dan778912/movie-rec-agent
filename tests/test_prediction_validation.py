"""Test prediction-contract failures with tiny, explicitly synthetic inputs."""

import copy
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.validate import (
    PredictionValidationError,
    ValidationContext,
    load_context,
    validate_predictions,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evaluation_validation"


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class PredictionValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        self.context = load_context(
            FIXTURES / "catalog.csv",
            FIXTURES / "profiles.jsonl",
            FIXTURES / "users.json",
            "validation",
        )
        self.records = read_jsonl(FIXTURES / "embedding_valid.jsonl")

    def write_records(self, records, name="predictions.jsonl"):
        path = self.work / name
        path.write_text(
            "".join(json.dumps(record, allow_nan=False) + "\n" for record in records),
            encoding="utf-8",
        )
        return path

    def write_text(self, content):
        path = self.work / "predictions.jsonl"
        path.write_text(content, encoding="utf-8")
        return path

    def validate(self, path, method="embedding", candidate_pool_path=None):
        return validate_predictions(
            path,
            self.context,
            method=method,
            required_count=3,
            candidate_pool_path=candidate_pool_path,
        )

    def assert_invalid(self, path, code, **kwargs):
        original = path.read_bytes()
        with self.assertRaises(PredictionValidationError) as raised:
            self.validate(path, **kwargs)
        report = raised.exception.report
        self.assertEqual(report["status"], "invalid")
        self.assertGreater(report["error_count"], 0)
        self.assertIn(code, report["errors_by_code"])
        self.assertTrue(any(error["code"] == code for error in report["errors"]))
        for error in report["errors"]:
            self.assertTrue({"code", "message", "line", "user_id"} <= error.keys())
            self.assertTrue(error["message"])
        self.assertTrue(str(raised.exception))
        self.assertEqual(path.read_bytes(), original)
        return report

    def test_valid_embedding_preserves_original_records_and_inputs(self):
        path = FIXTURES / "embedding_valid.jsonl"
        original = path.read_bytes()
        histories = copy.deepcopy(self.context.histories)
        result = self.validate(path)
        self.assertEqual(result.records, {record["user_id"]: record for record in self.records})
        self.assertEqual(result.report["status"], "valid")
        self.assertEqual(result.report["expected_users"], 2)
        self.assertEqual(result.report["users_with_records"], 2)
        self.assertEqual(result.report["valid_users"], 2)
        self.assertEqual(result.report["prediction_user_coverage"], 1.0)
        self.assertEqual(result.report["missing_users"], [])
        self.assertEqual(result.report["unexpected_users"], [])
        self.assertEqual(result.report["error_count"], 0)
        self.assertEqual(result.report["errors"], [])
        self.assertEqual(path.read_bytes(), original)
        self.assertEqual(self.context.histories, histories)

    def test_each_baseline_method_is_supported(self):
        for method in ("popularity", "bm25", "embedding"):
            with self.subTest(method=method):
                records = copy.deepcopy(self.records)
                for record in records:
                    record["method"] = method
                self.assertEqual(self.validate(self.write_records(records), method).report["status"], "valid")

    def test_rank_is_authoritative_scores_can_be_tied_zero_or_negative(self):
        records = copy.deepcopy(self.records)
        records[0]["recommendations"][0]["score"] = -10
        records[0]["recommendations"][1]["score"] = 0
        records[0]["recommendations"][2]["score"] = 0
        records[0]["recommendations"][0]["explanation"] = "Synthetic explanation"
        result = self.validate(self.write_records(records))
        self.assertEqual(result.records[1], records[0])

    def test_watched_check_uses_complete_history_even_with_empty_liked_movies(self):
        report = self.assert_invalid(FIXTURES / "watched_invalid.jsonl", "watched_movie")
        errors = [error for error in report["errors"] if error["code"] == "watched_movie"]
        self.assertEqual(errors[0]["line"], 1)
        self.assertEqual(errors[0]["user_id"], 1)
        self.assertIn("90", errors[0]["message"])
        self.assertEqual(report["valid_users"], 1)
        self.assertEqual(report["prediction_user_coverage"], 0.5)

    def test_missing_unexpected_and_duplicate_users_are_rejected(self):
        cases = [
            (self.records[:1], "missing_users"),
            (self.records + [dict(self.records[0], user_id=3)], "unexpected_user"),
            (self.records + [copy.deepcopy(self.records[0])], "duplicate_user"),
        ]
        for records, code in cases:
            with self.subTest(code=code):
                report = self.assert_invalid(self.write_records(records), code)
                if code == "missing_users":
                    self.assertEqual(report["missing_users"], [2])
                    self.assertEqual(report["prediction_user_coverage"], 0.5)
                if code == "unexpected_user":
                    self.assertEqual(report["unexpected_users"], [3])

    def test_duplicate_user_is_invalid_even_when_record_is_identical(self):
        report = self.assert_invalid(self.write_records(self.records + self.records[:1]), "duplicate_user")
        self.assertLess(report["prediction_user_coverage"], 1.0)

    def test_empty_prediction_file_does_not_silently_evaluate_zero_users(self):
        report = self.assert_invalid(self.write_text(""), "missing_users")
        self.assertEqual(report["missing_users"], [1, 2])
        self.assertEqual(report["prediction_user_coverage"], 0.0)

    def test_wrong_split_or_method_is_rejected(self):
        for field, value, code in (("split", "test", "wrong_split"), ("method", "bm25", "wrong_method")):
            with self.subTest(field=field):
                records = copy.deepcopy(self.records)
                records[0][field] = value
                self.assert_invalid(self.write_records(records), code)

    def test_missing_required_record_and_item_fields_are_rejected(self):
        for field in ("user_id", "split", "method", "recommendations"):
            with self.subTest(record_field=field):
                records = copy.deepcopy(self.records)
                del records[0][field]
                self.assert_invalid(self.write_records(records), "missing_field")
        for field in ("movie_id", "score", "rank"):
            with self.subTest(item_field=field):
                records = copy.deepcopy(self.records)
                del records[0]["recommendations"][0][field]
                self.assert_invalid(self.write_records(records), "missing_field")

    def test_user_ids_require_positive_integer_and_reject_bool(self):
        for value in (True, 0, -1, 1.0, "1", None):
            with self.subTest(value=value):
                records = copy.deepcopy(self.records)
                records[0]["user_id"] = value
                self.assert_invalid(self.write_records(records), "invalid_user_id")

    def test_movie_ids_require_positive_integer_and_reject_bool(self):
        for value in (True, 0, -1, 10.0, "10", None):
            with self.subTest(value=value):
                records = copy.deepcopy(self.records)
                records[0]["recommendations"][0]["movie_id"] = value
                self.assert_invalid(self.write_records(records), "invalid_movie_id")

    def test_unknown_movie_and_duplicate_movies_are_rejected(self):
        for movie_id, code in ((999, "unknown_movie"), (20, "duplicate_movie")):
            with self.subTest(code=code):
                records = copy.deepcopy(self.records)
                records[0]["recommendations"][0]["movie_id"] = movie_id
                self.assert_invalid(self.write_records(records), code)

    def test_rank_must_match_array_position_and_be_positive_integer(self):
        for rank in (True, 0, -1, 1.0, "1", None, 2):
            with self.subTest(rank=rank):
                records = copy.deepcopy(self.records)
                records[0]["recommendations"][0]["rank"] = rank
                self.assert_invalid(self.write_records(records), "invalid_rank")

    def test_invalid_scores_and_overflow_are_rejected(self):
        for score in (True, "0.9", None, [], {}):
            with self.subTest(score=score):
                records = copy.deepcopy(self.records)
                records[0]["recommendations"][0]["score"] = score
                self.assert_invalid(self.write_records(records), "invalid_score")
        text = (FIXTURES / "embedding_valid.jsonl").read_text(encoding="utf-8")
        self.assert_invalid(self.write_text(text.replace('"score":0.9', '"score":1e999', 1)), "invalid_score")

    def test_prediction_length_and_collection_types_are_checked(self):
        for recommendations, code in (([], "wrong_length"), (None, "invalid_recommendations"), ({}, "invalid_recommendations"), ("movies", "invalid_recommendations")):
            with self.subTest(recommendations=recommendations):
                records = copy.deepcopy(self.records)
                records[0]["recommendations"] = recommendations
                self.assert_invalid(self.write_records(records), code)
        records = copy.deepcopy(self.records)
        records[0]["recommendations"][0] = 10
        self.assert_invalid(self.write_records(records), "invalid_item")

    def test_strict_json_rejects_malformed_duplicate_keys_and_nonfinite_constants(self):
        valid_text = (FIXTURES / "embedding_valid.jsonl").read_text(encoding="utf-8")
        cases = (
            "{broken json}\n",
            valid_text.replace('"user_id":1', '"user_id":1,"user_id":1', 1),
            valid_text.replace('"score":0.9', '"score":0.9,"score":0.8', 1),
            valid_text.replace('"score":0.9', '"score":NaN', 1),
            valid_text.replace('"score":0.9', '"score":Infinity', 1),
            valid_text.replace('"score":0.9', '"score":-Infinity', 1),
        )
        for text in cases:
            with self.subTest(text=text[:100]):
                self.assert_invalid(self.write_text(text), "invalid_json")

    def test_nonobject_records_and_blank_lines_are_invalid(self):
        self.assert_invalid(self.write_text("[]\n"), "invalid_record")
        text = (FIXTURES / "embedding_valid.jsonl").read_text(encoding="utf-8")
        self.assert_invalid(self.write_text("\n" + text), "invalid_json")

    def test_collects_errors_from_multiple_lines_instead_of_silently_cleaning(self):
        records = copy.deepcopy(self.records)
        records[0]["recommendations"][0]["movie_id"] = 90
        records[1]["recommendations"][0]["movie_id"] = 999
        report = self.assert_invalid(self.write_records(records), "watched_movie")
        self.assertIn("unknown_movie", report["errors_by_code"])
        self.assertGreaterEqual(report["error_count"], 2)
        self.assertEqual(report["valid_users"], 0)

    def test_p1_can_reorder_but_must_preserve_exact_candidate_sets(self):
        pool = FIXTURES / "embedding_valid.jsonl"
        result = self.validate(FIXTURES / "p1_valid.jsonl", method="p1", candidate_pool_path=pool)
        self.assertEqual(result.report["status"], "valid")
        self.assertEqual(result.records[1]["recommendations"][0]["movie_id"], 30)
        report = self.assert_invalid(
            FIXTURES / "p1_invalid_pool.jsonl",
            "candidate_pool_mismatch",
            method="p1",
            candidate_pool_path=pool,
        )
        self.assertEqual(report["valid_users"], 1)

    def test_p1_rejects_invalid_frozen_candidate_file(self):
        # The pool itself must obey catalog/history/schema rules before comparison.
        with self.assertRaises(ValueError) as raised:
            self.validate(
                FIXTURES / "p1_valid.jsonl",
                method="p1",
                candidate_pool_path=FIXTURES / "watched_invalid.jsonl",
            )
        self.assertNotIsInstance(raised.exception, PredictionValidationError)
        self.assertIn("candidate pool is invalid", str(raised.exception))

    def test_invalid_validator_options_and_missing_files_are_input_errors(self):
        valid = FIXTURES / "embedding_valid.jsonl"
        for method in ("unknown", "Embedding", ""):
            with self.subTest(method=method):
                with self.assertRaises(ValueError):
                    self.validate(valid, method=method)
        with self.assertRaises(ValueError):
            self.validate(FIXTURES / "p1_valid.jsonl", method="p1")
        with self.assertRaises(ValueError):
            self.validate(valid, candidate_pool_path=valid)
        for count in (0, -1, True, 3.0):
            with self.subTest(count=count):
                with self.assertRaises((TypeError, ValueError)):
                    validate_predictions(valid, self.context, method="embedding", required_count=count)
        with self.assertRaises(OSError):
            self.validate(self.work / "does-not-exist.jsonl")


class ContextLoadingTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)
        for name in ("catalog.csv", "profiles.jsonl", "users.json"):
            (self.work / name).write_bytes((FIXTURES / name).read_bytes())

    def load(self):
        return load_context(self.work / "catalog.csv", self.work / "profiles.jsonl", self.work / "users.json", "validation")

    def test_loads_complete_watched_histories_and_frozen_user_population(self):
        context = self.load()
        self.assertIsInstance(context, ValidationContext)
        self.assertEqual(context.catalog_ids, set(range(10, 100, 10)))
        self.assertEqual(context.histories, {1: {90}, 2: {80}})
        self.assertEqual(context.expected_user_ids, {1, 2})
        self.assertEqual(context.split, "validation")

    def test_histories_can_contain_movies_excluded_from_recommendable_catalog(self):
        rows = read_jsonl(FIXTURES / "profiles.jsonl")
        rows[0]["history_movie_ids"].append(999)
        (self.work / "profiles.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
        )
        self.assertEqual(self.load().histories[1], {90, 999})

    def test_profile_population_can_be_larger_than_frozen_subset(self):
        # Real profiles contain many users; only the selected IDs need predictions.
        (self.work / "users.json").write_text(
            json.dumps({"split": "validation", "user_ids": [1]}), encoding="utf-8"
        )
        context = self.load()
        self.assertEqual(context.expected_user_ids, {1})
        self.assertEqual(context.histories[1], {90})

    def test_frozen_user_list_binds_catalog_and_profile_contents(self):
        users = {
            "split": "validation", "user_ids": [1, 2],
            "source_sha256": {
                name: hashlib.sha256((self.work / filename).read_bytes()).hexdigest()
                for name, filename in (("catalog", "catalog.csv"), ("profiles", "profiles.jsonl"))
            },
        }
        (self.work / "users.json").write_text(json.dumps(users), encoding="utf-8")
        self.assertEqual(self.load().expected_user_ids, {1, 2})
        # Even a harmless text change means this is a different frozen input.
        for filename, label in (("catalog.csv", "catalog"), ("profiles.jsonl", "profiles")):
            with self.subTest(filename=filename):
                path = self.work / filename
                original = path.read_bytes()
                if label == "catalog":
                    path.write_bytes(original.replace(b"Synthetic Movie A", b"Renamed Synthetic Movie A", 1))
                else:
                    path.write_bytes(original.replace(b'"liked_movies":[]', b'"liked_movies": []', 1))
                self.assertNotEqual(path.read_bytes(), original)
                with self.assertRaisesRegex(ValueError, f"{label} hash does not match"):
                    self.load()
                path.write_bytes(original)

    def test_malformed_frozen_source_hashes_are_rejected(self):
        for bindings in (None, [], {}, {"catalog": "not-a-hash", "profiles": "a" * 64}):
            with self.subTest(bindings=bindings):
                users = {"split": "validation", "user_ids": [1, 2], "source_sha256": bindings}
                (self.work / "users.json").write_text(json.dumps(users), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source_sha256"):
                    self.load()

    def test_catalog_requires_movie_id_column_and_unique_positive_integer_ids(self):
        for text in ("id,title\n10,A\n", "movie_id\n10\n10\n", "movie_id\n0\n", "movie_id\n10.0\n"):
            with self.subTest(text=text):
                (self.work / "catalog.csv").write_text(text, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.load()

    def test_users_file_requires_exact_split_and_unique_positive_integer_ids(self):
        cases = (
            {"split": "test", "user_ids": [1, 2]},
            {"split": "validation", "user_ids": [1, 1]},
            {"split": "validation", "user_ids": [True, 2]},
            {"split": "validation", "user_ids": [0, 2]},
            {"split": "validation", "user_ids": "1,2"},
            {"split": "validation", "user_ids": [1, 3]},
            {"split": "validation"},
        )
        for users in cases:
            with self.subTest(users=users):
                (self.work / "users.json").write_text(json.dumps(users), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.load()

    def test_profiles_require_unique_users_correct_split_and_complete_valid_history(self):
        original = read_jsonl(FIXTURES / "profiles.jsonl")
        cases = [original + original[:1]]
        for field, value in (
            ("split", "test"),
            ("user_id", True),
            ("history_movie_ids", "90"),
            ("history_movie_ids", [90, 90]),
            ("history_movie_ids", [True]),
            ("history_movie_ids", [0]),
        ):
            rows = copy.deepcopy(original)
            rows[0][field] = value
            cases.append(rows)
        rows = copy.deepcopy(original)
        del rows[0]["history_movie_ids"]
        cases.append(rows)
        for rows in cases:
            with self.subTest(rows=rows):
                (self.work / "profiles.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.load()


class ValidatorCommandTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.work = Path(self.temp.name)

    def run_cli(self, predictions, *extra, method="embedding"):
        return subprocess.run(
            [
                sys.executable, "-m", "src.evaluation.validate",
                "--catalog", str(FIXTURES / "catalog.csv"),
                "--profiles", str(FIXTURES / "profiles.jsonl"),
                "--users", str(FIXTURES / "users.json"),
                "--predictions", str(predictions),
                "--split", "validation",
                "--method", method,
                "--required-count", "3",
                *extra,
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_valid_cli_exit_zero_and_saves_machine_readable_report(self):
        report_path = self.work / "valid_report.json"
        prediction = FIXTURES / "embedding_valid.jsonl"
        before = prediction.read_bytes()
        result = self.run_cli(prediction, "--report", str(report_path))
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "valid")
        self.assertEqual(report["error_count"], 0)
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)
        self.assertEqual(prediction.read_bytes(), before)

    def test_invalid_cli_exit_one_and_preserves_prediction_file(self):
        report_path = self.work / "invalid_report.json"
        prediction = FIXTURES / "watched_invalid.jsonl"
        before = prediction.read_bytes()
        result = self.run_cli(prediction, "--report", str(report_path))
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["status"], "invalid")
        self.assertIn("watched_movie", report["errors_by_code"])
        self.assertTrue(result.stderr.strip())
        self.assertEqual(json.loads(report_path.read_text(encoding="utf-8")), report)
        self.assertEqual(prediction.read_bytes(), before)

    def test_missing_input_cli_exit_two_with_readable_error(self):
        result = self.run_cli(self.work / "missing.jsonl")
        self.assertEqual(result.returncode, 2)
        self.assertTrue(result.stderr.strip())

    def test_invalid_p1_pool_is_input_error_without_misleading_prediction_report(self):
        report_path = self.work / "not_evaluated.json"
        pool_path = FIXTURES / "watched_invalid.jsonl"
        original_pool = pool_path.read_bytes()
        result = self.run_cli(
            FIXTURES / "p1_valid.jsonl", "--candidate-pool", str(pool_path),
            "--report", str(report_path), method="p1",
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("candidate pool is invalid", result.stderr)
        self.assertIn(str(pool_path), result.stderr)
        self.assertEqual(result.stdout, "")
        self.assertFalse(report_path.exists())
        self.assertEqual(pool_path.read_bytes(), original_pool)

    def test_report_cannot_overwrite_prediction_input(self):
        prediction_path = self.work / "prediction.jsonl"
        original = (FIXTURES / "embedding_valid.jsonl").read_bytes()
        prediction_path.write_bytes(original)
        result = self.run_cli(prediction_path, "--report", str(prediction_path))
        self.assertEqual(result.returncode, 2)
        self.assertIn("must not overwrite an input file", result.stderr)
        self.assertEqual(prediction_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
