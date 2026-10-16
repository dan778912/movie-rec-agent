"""Independent hand-calculated scoring and end-to-end artifact checks.

Every user/movie here is synthetic. Assertions compare known calculations rather
than invoking the metric helpers to manufacture expected results.
"""

import contextlib
import copy
import csv
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.evaluation.evaluate import evaluate, main, write_results
from src.evaluation.validate import PredictionValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evaluation_scoring"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, payload):
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class EvaluationPipelineTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.data = self.work / "data"
        shutil.copytree(FIXTURES, self.data)
        self.config_path = self.work / "config.json"
        self.predictions = self.work / "predictions.jsonl"
        self.predictions.write_bytes((FIXTURES / "embedding.jsonl").read_bytes())
        self.config = read_json(FIXTURES / "config.json")
        self.config.update(data_dir=str(self.data), users_file=str(self.data / "users.json"))
        self.save_config()

    def save_config(self):
        save_json(self.config_path, self.config)

    def score(self, **kwargs):
        return evaluate(self.config_path, self.predictions, method="embedding",
                        repository_root=self.work, **kwargs)

    def rebind_sources(self):
        """Refresh integrity metadata so semantic-rejection tests reach parsing."""
        manifest_path = self.data / "manifest.json"
        manifest = read_json(manifest_path)
        for name in manifest["file_sha256"]:
            manifest["file_sha256"][name] = digest(self.data / name)
        save_json(manifest_path, manifest)
        users_path = self.data / "users.json"
        users = read_json(users_path)
        users["source_sha256"] = {
            "manifest": digest(manifest_path), "catalog": digest(self.data / "catalog.csv"),
            "profiles": digest(self.data / "validation_profiles.jsonl"),
            "targets": digest(self.data / "validation_targets.csv"),
        }
        save_json(users_path, users)
        self.config.update(manifest_sha256=digest(manifest_path), users_sha256=digest(users_path))
        self.save_config()

    def target_rows(self):
        with (self.data / "validation_targets.csv").open(newline="", encoding="utf-8") as source:
            return list(csv.DictReader(source))

    def save_targets(self, rows):
        with (self.data / "validation_targets.csv").open("w", newline="", encoding="utf-8") as target:
            writer = csv.DictWriter(target, fieldnames=["user_id", "movie_id", "rating", "in_catalog"],
                                    lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        self.rebind_sources()

    def cli(self, *, predictions=None, method="embedding", output=None, extra=()):
        output = output or self.work / "results"
        args = ["--config", str(self.config_path), "--predictions", str(predictions or self.predictions),
                "--method", method, "--output-dir", str(output), *extra]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(args)
        return code, stdout.getvalue(), stderr.getvalue(), output

    def test_committed_demo_is_runnable_with_repository_relative_paths(self):
        result = evaluate(FIXTURES / "config.json", FIXTURES / "embedding.jsonl", method="embedding")
        self.assertEqual(result.summary["user_count"], 2)
        self.assertEqual(result.summary["metrics"], {
            "recall_at_2": 0.5, "ndcg_at_2": 0.5, "hit_rate_at_2": 1.0, "recall_at_3": 0.75,
        })
        self.assertFalse(result.summary["bootstrap_applied"])
        self.assertEqual(result.summary["config_status"], "draft_for_team_review")

    def test_complete_truth_counts_missing_candidate_and_excludes_out_of_catalog(self):
        result = self.score()
        one, two = result.per_user
        self.assertEqual((one["user_id"], two["user_id"]), (1, 2))
        self.assertEqual((one["relevant_targets"], two["relevant_targets"]), (2, 2))
        self.assertEqual((one["positive_targets"], two["positive_targets"]), (3, 2))
        self.assertEqual(one["out_of_catalog_positive_targets"], 1)
        self.assertEqual(one["recall_at_3"], 1.0)
        self.assertEqual(two["recall_at_3"], 0.5)
        self.assertEqual(result.summary["prediction_user_coverage"], 1.0)

    def test_full_split_coverage_includes_user_with_only_out_of_catalog_positives(self):
        coverage = self.score().summary["coverage"]
        self.assertEqual(coverage["full_split"]["positive_targets"], 6)
        self.assertEqual(coverage["full_split"]["in_catalog_positive_targets"], 4)
        self.assertEqual(coverage["full_split"]["out_of_catalog_positive_targets"], 2)
        self.assertAlmostEqual(coverage["full_split"]["catalog_positive_coverage"], 2 / 3)
        self.assertEqual(coverage["selected_users"]["positive_targets"], 5)
        self.assertEqual(coverage["selected_users"]["catalog_positive_coverage"], 0.8)

    def test_users_join_by_id_and_output_is_sorted_independent_of_file_order(self):
        first = self.score()
        self.predictions.write_text("\n".join(reversed(self.predictions.read_text().splitlines())) + "\n")
        second = self.score()
        self.assertEqual(first.per_user, second.per_user)
        self.assertEqual(first.summary["metrics"], second.summary["metrics"])
        self.assertEqual([row["user_id"] for row in second.per_user], [1, 2])

    def test_p1_improves_top_ranks_without_changing_full_pool_recall(self):
        before = self.score()
        after = evaluate(self.config_path, FIXTURES / "p1.jsonl", method="p1",
                         candidate_pool_path=self.predictions, repository_root=self.work)
        self.assertEqual(after.summary["metrics"]["recall_at_2"], 0.75)
        self.assertAlmostEqual(after.summary["metrics"]["ndcg_at_2"], 0.8065735963827292)
        self.assertEqual(after.summary["metrics"]["recall_at_3"], before.summary["metrics"]["recall_at_3"])
        self.assertEqual(after.summary["source_sha256"]["candidate_pool"], digest(self.predictions))

    def test_macro_average_differs_from_pooled_recall_when_truth_sizes_differ(self):
        self.save_targets([row for row in self.target_rows() if (row["user_id"], row["movie_id"]) != ("1", "20")])
        result = self.score()
        self.assertEqual(result.summary["metrics"]["recall_at_2"], 0.75)
        self.assertNotAlmostEqual(result.summary["metrics"]["recall_at_2"], 2 / 3)

    def test_all_baseline_methods_use_the_same_metric_contract(self):
        records = [json.loads(line) for line in self.predictions.read_text().splitlines()]
        for method in ("popularity", "bm25", "embedding"):
            with self.subTest(method=method):
                for record in records:
                    record["method"] = method
                self.predictions.write_text("".join(json.dumps(row) + "\n" for row in records))
                result = evaluate(self.config_path, self.predictions, method=method, repository_root=self.work)
                self.assertEqual(result.summary["method"], method)
                self.assertEqual(result.summary["metrics"]["ndcg_at_2"], 0.5)

    def test_scores_do_not_resort_authoritative_ranks(self):
        original = self.score().summary["metrics"]
        records = [json.loads(line) for line in self.predictions.read_text().splitlines()]
        for record in records:
            for recommendation in record["recommendations"]:
                recommendation["score"] = recommendation["rank"] * -100 if recommendation["rank"] == 1 else 100
        self.predictions.write_text("".join(json.dumps(row) + "\n" for row in records))
        self.assertEqual(self.score().summary["metrics"], original)

    def test_hit_rate_can_be_disabled_without_changing_recall_or_ndcg(self):
        self.config["include_hit_rate"] = False
        self.save_config()
        self.assertEqual(self.score().summary["metrics"], {
            "recall_at_2": 0.5, "ndcg_at_2": 0.5, "recall_at_3": 0.75,
        })

    def test_invalid_predictions_fail_without_creating_output(self):
        code, stdout, stderr, output = self.cli(predictions=FIXTURES / "invalid_predictions.jsonl")
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout)["status"], "invalid")
        self.assertIn("watched", stderr.lower())
        self.assertFalse(output.exists())

    def test_missing_prediction_user_is_rejected_instead_of_averaging_remaining_user(self):
        self.predictions.write_text(self.predictions.read_text().splitlines()[0] + "\n")
        with self.assertRaises(PredictionValidationError):
            self.score()

    def test_cli_success_writes_three_parseable_consistent_artifacts(self):
        code, stdout, stderr, output = self.cli()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(set(path.name for path in output.iterdir()), {"summary.json", "per_user.csv", "validation.json"})
        self.assertEqual(json.loads(stdout), read_json(output / "summary.json"))
        self.assertEqual(read_json(output / "validation.json")["status"], "valid")
        with (output / "per_user.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([int(row["user_id"]) for row in rows], [1, 2])
        self.assertEqual([float(row["recall_at_3"]) for row in rows], [1.0, 0.5])

    def test_bad_configuration_cli_returns_input_error_without_output(self):
        self.config["averaging"] = "micro"
        self.save_config()
        code, stdout, stderr, output = self.cli()
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("averaging", stderr)
        self.assertFalse(output.exists())

    def test_unsupported_semantics_and_invalid_cutoffs_are_rejected(self):
        original = copy.deepcopy(self.config)
        cases = [("averaging", "micro"), ("relevance", "graded"),
                 ("ground_truth", "retrieved_candidates_only"), ("positive_rating_at_least", 3.5),
                 ("failure_policy", "skip_invalid_users"), ("path_resolution", "current_directory"),
                 ("p1_candidate_policy", "allow_new_movies"), ("ranking_k", [4]), ("retrieval_k", [4]),
                 ("ranking_k", [True]), ("retrieval_k", [0]), ("ranking_k", [2, 2]),
                 ("include_hit_rate", 1), ("sample_size", True), ("candidate_pool_size", 2)]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                self.config = copy.deepcopy(original)
                self.config[key] = value
                self.save_config()
                with self.assertRaises(ValueError):
                    self.score()

    def test_target_byte_drift_is_rejected_before_scoring(self):
        with (self.data / "validation_targets.csv").open("a") as target:
            target.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.score()

    def test_missing_targets_are_an_input_error(self):
        (self.data / "validation_targets.csv").unlink()
        code, stdout, stderr, output = self.cli()
        self.assertEqual(code, 2)
        self.assertIn("validation_targets.csv", stderr)
        self.assertEqual(stdout, "")
        self.assertFalse(output.exists())

    def test_manifest_dataset_version_must_match_configuration(self):
        path = self.data / "manifest.json"
        manifest = read_json(path)
        manifest["version"] = "different-data"
        save_json(path, manifest)
        self.rebind_sources()
        with self.assertRaisesRegex(ValueError, "version"):
            self.score()

    def test_manifest_positive_threshold_must_match_evaluation(self):
        path = self.data / "manifest.json"
        manifest = read_json(path)
        manifest["config"]["positive_rating_at_least"] = 3.5
        save_json(path, manifest)
        self.rebind_sources()
        with self.assertRaisesRegex(ValueError, "threshold"):
            self.score()

    def test_saved_subset_version_seed_size_and_provenance_are_checked(self):
        path = self.data / "users.json"
        original = read_json(path)
        cases = [("dataset_version", "different-data"), ("split", "test"),
                 ("source_sha256", {**original["source_sha256"], "targets": "0" * 64}),
                 ("selection", {**original["selection"], "size": 1}),
                 ("selection", {**original["selection"], "seed": 123}),
                 ("user_ids_sha256", "0" * 64)]
        for key, value in cases:
            with self.subTest(key=key):
                users = copy.deepcopy(original)
                users[key] = value
                save_json(path, users)
                self.config["users_sha256"] = digest(path)
                self.save_config()
                with self.assertRaises(ValueError):
                    self.score()

    def test_sample_size_cannot_silently_differ_from_selected_users(self):
        self.config["sample_size"] = 1
        self.save_config()
        with self.assertRaisesRegex(ValueError, "sample_size"):
            self.score()

    def test_malformed_semantic_targets_are_rejected_even_with_matching_hashes(self):
        original = self.target_rows()
        cases = []
        for field, value in (("rating", "NaN"), ("rating", "Infinity"), ("rating", "3.5"),
                             ("rating", "5.5"), ("in_catalog", "2"), ("in_catalog", "0"),
                             ("user_id", "1.5"), ("movie_id", "0")):
            rows = copy.deepcopy(original)
            rows[0][field] = value
            cases.append((f"{field}={value}", rows))
        cases.append(("duplicate", original + [copy.deepcopy(original[0])]))
        cases.append(("history overlap", original + [{"user_id": "1", "movie_id": "90", "rating": "4", "in_catalog": "1"}]))
        cases.append(("missing profile", original + [{"user_id": "4", "movie_id": "10", "rating": "4", "in_catalog": "1"}]))
        cases.append(("no eligible truth", [row for row in original if row["user_id"] != "1"]))
        for name, rows in cases:
            with self.subTest(name=name):
                self.save_targets(rows)
                with self.assertRaises(ValueError):
                    self.score()

    def test_explicit_repository_root_resolves_relative_data_paths(self):
        self.config.update(data_dir="data", users_file="data/users.json")
        self.save_config()
        self.assertEqual(self.score().summary["metrics"]["recall_at_3"], 0.75)

    def test_provenance_records_actual_inputs_and_current_source(self):
        summary = self.score().summary
        expected = {"config": self.config_path, "manifest": self.data / "manifest.json",
                    "catalog": self.data / "catalog.csv", "profiles": self.data / "validation_profiles.jsonl",
                    "targets": self.data / "validation_targets.csv", "users": self.data / "users.json",
                    "predictions": self.predictions}
        for name, path in expected.items():
            self.assertEqual(summary["source_sha256"][name], digest(path))
        self.assertEqual(set(summary["implementation"]["source_sha256"]), {"evaluate", "metrics", "validate"})
        self.assertIn("python_version", summary["runtime"])

    def test_results_are_deterministic_and_identical_rerun_preserves_files(self):
        output = self.work / "results"
        first, second = self.score(), self.score()
        self.assertEqual(first.summary, second.summary)
        write_results(output, first)
        original = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
        write_results(output, second)
        self.assertEqual(original, {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()})

    def test_different_existing_results_are_preserved_and_rejected(self):
        result = self.score()
        output = self.work / "results"
        write_results(output, result)
        original = {path.name: path.read_bytes() for path in output.iterdir()}
        result.summary["metrics"]["recall_at_2"] = 0.123
        with self.assertRaisesRegex(ValueError, "existing results differ"):
            write_results(output, result)
        self.assertEqual(original, {path.name: path.read_bytes() for path in output.iterdir()})

    def test_output_inside_data_dir_and_directory_alias_are_rejected(self):
        result = self.score()
        alias = self.work / "data_alias"
        alias.symlink_to(self.data, target_is_directory=True)
        for output in (self.data, self.data / "results", alias / "results"):
            with self.subTest(output=output):
                with self.assertRaises(ValueError):
                    write_results(output, result)
        self.assertFalse((self.data / "results").exists())

    def test_output_file_aliases_cannot_overwrite_input(self):
        result = self.score()
        original = self.predictions.read_bytes()
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                output = self.work / kind
                output.mkdir()
                if kind == "symlink":
                    (output / "summary.json").symlink_to(self.predictions)
                else:
                    os.link(self.predictions, output / "summary.json")
                with self.assertRaises(ValueError):
                    write_results(output, result)
                self.assertEqual(self.predictions.read_bytes(), original)
                self.assertEqual({path.name for path in output.iterdir()}, {"summary.json"})

    def test_p1_requires_original_candidate_pool(self):
        with self.assertRaises(ValueError):
            evaluate(self.config_path, FIXTURES / "p1.jsonl", method="p1", repository_root=self.work)


if __name__ == "__main__":
    unittest.main()
