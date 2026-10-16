"""Check real integration boundaries with explicitly synthetic full exports.

Invalid excluded users and discarded tails must fail before selection. The
independent scoring example then exercises preparation through the final table.
"""

import contextlib
import copy
import hashlib
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bootstrap import compare_runs, write_comparison
from src.evaluation.evaluate import evaluate, write_results
from src.evaluation.prepare import export_users, main, prepare_predictions, write_prepared
from src.evaluation.report import build_table, write_table
from src.evaluation.validate import PredictionValidationError


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evaluation_preparation"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def save_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PredictionPreparationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.data = self.work / "data"
        shutil.copytree(FIXTURES, self.data)
        self.config_path = self.work / "config.json"
        self.config = read_json(FIXTURES / "config.json")
        self.config.update(data_dir=str(self.data), users_file=str(self.data / "users.json"))
        self.save_config()
        self.source = self.work / "source.jsonl"
        self.source.write_bytes((FIXTURES / "full_embedding.jsonl").read_bytes())

    def save_config(self):
        save_json(self.config_path, self.config)

    def prepare(self, **overrides):
        options = dict(method="embedding", source_count=4, source_users="all",
                       source_label="synthetic frozen embedding", repository_root=self.work)
        options.update(overrides)
        return prepare_predictions(self.config_path, self.source, **options)

    def write_records(self, records, path=None):
        path = path or self.source
        path.write_text("".join(json.dumps(row, allow_nan=False) + "\n" for row in records),
                        encoding="utf-8")
        return path

    def rebind_sources(self):
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

    def cli(self, *, command="predictions", extra=(), output=None):
        output = output or self.work / "prepared"
        args = [command, "--config", str(self.config_path), "--output-dir", str(output)]
        if command == "predictions":
            args += ["--source", str(self.source), "--method", "embedding", "--source-count", "4",
                     "--source-users", "all", "--source-label", "synthetic frozen embedding"]
        args += list(extra)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(args)
        return code, stdout.getvalue(), stderr.getvalue(), output

    def p1(self, *, pool=None, count=3):
        return prepare_predictions(
            self.config_path, FIXTURES / "full_p1.jsonl", method="p1", source_count=count,
            source_users="all", source_label="synthetic P1 over frozen pool",
            candidate_pool_path=pool or FIXTURES / "full_candidate_pool.jsonl", repository_root=self.work,
        )

    def test_committed_fixture_runs_and_selects_exact_frozen_users(self):
        result = prepare_predictions(
            FIXTURES / "config.json", FIXTURES / "full_embedding.jsonl", method="embedding",
            source_count=4, source_users="all", source_label="synthetic fixture",
        )
        self.assertEqual(result.summary["status"], "prepared")
        self.assertEqual(result.summary["source_user_count"], 3)
        self.assertEqual(result.summary["selected_users"], 2)
        self.assertEqual(result.summary["source_count"], 4)
        self.assertEqual(result.summary["output_count"], 3)
        self.assertEqual([row["user_id"] for row in result.records], [1, 2])
        self.assertEqual(result.source_validation["expected_users"], 3)
        self.assertEqual(result.source_validation["declared_population"], "all")
        self.assertEqual(result.source_validation["expected_user_ids_sha256"],
                         hashlib.sha256(b"[1,2,3]\n").hexdigest())
        self.assertEqual(result.derived_validation["expected_users"], 2)
        self.assertEqual(result.derived_validation["status"], "valid")
        self.assertEqual(result.derived_validation["source_sha256"]["users"], digest(FIXTURES / "users.json"))

    def test_preserves_prefix_scores_ranks_explanations_and_record_metadata(self):
        records = read_jsonl(self.source)
        records[2]["recommendations"][0]["score"] = -10
        records[2]["recommendations"][1]["score"] = 0
        records[2]["recommendations"][2]["score"] = 0
        self.write_records(records)
        originals = {row["user_id"]: row for row in records}
        result = self.prepare()
        for row in result.records:
            source = originals[row["user_id"]]
            self.assertEqual(row, {**source, "recommendations": source["recommendations"][:3]})
        self.assertEqual(result.records[0]["recommendations"][0]["explanation"],
                         "Synthetic explanation kept verbatim")
        self.assertEqual([item["movie_id"] for item in result.records[0]["recommendations"]], [10, 30, 20])

    def test_preparation_does_not_change_any_source_bytes(self):
        sources = [self.source, self.config_path, *self.data.iterdir()]
        before = {path: path.read_bytes() for path in sources if path.is_file()}
        write_prepared(self.work / "prepared", self.prepare())
        self.assertEqual(before, {path: path.read_bytes() for path in before})

    def test_invalid_unselected_user_is_rejected_before_filtering(self):
        records = read_jsonl(self.source)
        records[1]["recommendations"][0]["movie_id"] = 70  # User 3 watched this.
        self.write_records(records)
        with self.assertRaises(PredictionValidationError) as raised:
            self.prepare()
        self.assertIn("watched_movie", raised.exception.report["errors_by_code"])

    def test_invalid_discarded_tail_is_rejected_before_truncation(self):
        records = read_jsonl(self.source)
        records[2]["recommendations"][3]["movie_id"] = 90  # Would disappear from top 3.
        self.write_records(records)
        with self.assertRaises(PredictionValidationError) as raised:
            self.prepare()
        self.assertIn("watched_movie", raised.exception.report["errors_by_code"])

    def test_missing_extra_and_duplicate_source_users_are_not_silently_ignored(self):
        originals = read_jsonl(self.source)
        for records in (originals[:1] + originals[2:], originals + originals[:1],
                        originals + [{**originals[0], "user_id": 4}]):
            with self.subTest(users=[row["user_id"] for row in records]):
                self.write_records(records)
                with self.assertRaises(PredictionValidationError):
                    self.prepare()

    def test_selected_source_mode_requires_exactly_selected_population(self):
        rows = read_jsonl(self.source)
        with self.assertRaises(PredictionValidationError):
            self.prepare(source_users="selected")
        self.write_records([row for row in rows if row["user_id"] in (1, 2)])
        self.assertEqual(self.prepare(source_users="selected").summary["source_user_count"], 2)
        with self.assertRaises(PredictionValidationError):
            self.prepare(source_users="all")

    def test_source_count_is_explicit_and_cannot_hide_short_or_long_rows(self):
        for count in (3, 5):
            with self.subTest(count=count), self.assertRaises(PredictionValidationError):
                self.prepare(source_count=count)
        for count in (2, 0, True, 4.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                self.prepare(source_count=count)

    def test_all_baseline_methods_preserve_their_method_without_conversion(self):
        originals = read_jsonl(self.source)
        for method in ("popularity", "bm25", "embedding"):
            with self.subTest(method=method):
                self.write_records([{**row, "method": method} for row in originals])
                result = self.prepare(method=method)
                self.assertEqual({row["method"] for row in result.records}, {method})
        with self.assertRaises(PredictionValidationError):
            self.prepare(method="bm25")

    def test_strict_json_rejects_duplicate_keys_nonfinite_and_bad_tail(self):
        original = self.source.read_text(encoding="utf-8")
        for content in (original.replace('"user_id":2', '"user_id":2,"user_id":2', 1),
                        original.replace('"score":-1', '"score":NaN', 1), original + "{broken json}\n"):
            with self.subTest(content=content[-25:]):
                self.source.write_text(content, encoding="utf-8")
                with self.assertRaises(PredictionValidationError):
                    self.prepare()

    def test_p1_preserves_exact_pool_and_candidate_output_matches_embedding_preparation(self):
        embedding = self.prepare()
        p1 = self.p1()
        self.assertEqual(p1.candidate_records, embedding.records)
        self.assertEqual(p1.summary["output_sha256"]["candidate_pool"],
                         embedding.summary["output_sha256"]["predictions"])
        embedding_dir, p1_dir = self.work / "embedding", self.work / "p1"
        write_prepared(embedding_dir, embedding)
        write_prepared(p1_dir, p1)
        self.assertEqual((embedding_dir / "predictions.jsonl").read_bytes(),
                         (p1_dir / "candidate_pool.jsonl").read_bytes())
        self.assertEqual([item["movie_id"] for item in p1.records[0]["recommendations"]], [10, 20, 30])

    def test_p1_longer_source_is_rejected_instead_of_truncated(self):
        with self.assertRaisesRegex(ValueError, "cannot be truncated"):
            self.p1(count=4)

    def test_selected_only_p1_accepts_selected_pool_and_rejects_full_population_pool(self):
        embedding = self.prepare()
        embedding_dir = self.work / "selected_embedding"
        write_prepared(embedding_dir, embedding)
        self.write_records([row for row in read_jsonl(FIXTURES / "full_p1.jsonl") if row["user_id"] in (1, 2)])
        result = self.prepare(method="p1", source_count=3, source_users="selected",
                              candidate_pool_path=embedding_dir / "predictions.jsonl")
        self.assertEqual(result.candidate_records, embedding.records)
        self.assertEqual(result.summary["source_user_count"], 2)
        with self.assertRaisesRegex(ValueError, "candidate pool is invalid"):
            self.prepare(method="p1", source_count=3, source_users="selected",
                         candidate_pool_path=FIXTURES / "full_candidate_pool.jsonl")

    def test_p1_rejects_changed_pool_membership_for_unselected_user(self):
        records = read_jsonl(FIXTURES / "full_candidate_pool.jsonl")
        records[1]["recommendations"][0]["movie_id"] = 50
        pool = self.write_records(records, self.work / "pool.jsonl")
        with self.assertRaises(PredictionValidationError):
            self.p1(pool=pool)

    def test_p1_rejects_invalid_pool_records_and_incomplete_pool_population(self):
        original = read_jsonl(FIXTURES / "full_candidate_pool.jsonl")
        watched = copy.deepcopy(original)
        watched[1]["recommendations"][2]["movie_id"] = 70
        for records in (watched, original[:1] + original[2:]):
            with self.subTest(users=[row["user_id"] for row in records]):
                pool = self.write_records(records, self.work / "pool.jsonl")
                with self.assertRaisesRegex(ValueError, "candidate pool is invalid"):
                    self.p1(pool=pool)

    def test_candidate_pool_is_required_only_for_p1(self):
        with self.assertRaisesRegex(ValueError, "only for p1"):
            self.prepare(candidate_pool_path=FIXTURES / "full_candidate_pool.jsonl")
        with self.assertRaisesRegex(ValueError, "requires"):
            self.prepare(method="p1", source_count=3)

    def test_expected_source_hash_is_checked_and_recorded(self):
        expected = digest(self.source)
        result = self.prepare(expected_source_sha256=expected)
        self.assertTrue(result.summary["expected_source_sha256_verified"])
        self.assertEqual(result.summary["source_sha256"]["predictions"], expected)
        for value in ("0" * 64, "not-a-hash", expected.upper()):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.prepare(expected_source_sha256=value)

    def test_optional_owner_metadata_is_preserved_and_explicitly_unverified(self):
        path = FIXTURES / "source_metadata.json"
        result = self.prepare(source_metadata_path=path)
        self.assertEqual(result.summary["source_metadata"], read_json(path))
        self.assertEqual(result.summary["source_sha256"]["source_metadata"], digest(path))
        self.assertEqual(result.summary["source_metadata_verification"], "owner_supplied_unverified")
        self.assertEqual(result.summary["source_label_verification"], "owner_supplied_unverified")

    def test_metadata_must_be_a_strict_json_object_and_label_nonempty(self):
        path = self.work / "metadata.json"
        for content in ('[]', '{"model":"a","model":"b"}', '{"cost":NaN}', '{bad}'):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValueError):
                    self.prepare(source_metadata_path=path)
        for label in ("", "  ", None):
            with self.subTest(label=label), self.assertRaises(ValueError):
                self.prepare(source_label=label)

    def test_frozen_user_file_config_and_data_drift_are_rejected(self):
        original_config = copy.deepcopy(self.config)
        for key, value in (("manifest_sha256", "0" * 64), ("users_sha256", "0" * 64),
                           ("dataset_version", "wrong-version"), ("sample_size", 3),
                           ("selection_seed", 12345)):
            with self.subTest(key=key):
                self.config = {**original_config, key: value}
                self.save_config()
                with self.assertRaises(ValueError):
                    self.prepare()
        self.config = original_config
        self.save_config()
        with (self.data / "validation_targets.csv").open("a", encoding="utf-8") as target:
            target.write("\n")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            self.prepare()

    def test_rebound_inconsistent_target_population_is_still_rejected(self):
        targets = self.data / "validation_targets.csv"
        targets.write_text(targets.read_text(encoding="utf-8").replace("3,20,4.0,1\n", ""), encoding="utf-8")
        self.rebind_sources()
        with self.assertRaises(ValueError):
            self.prepare()

    def test_rebound_inconsistent_frozen_selection_metadata_is_rejected(self):
        path = self.data / "users.json"
        original = read_json(path)
        for key, value in (("user_ids_sha256", "0" * 64), ("dataset_version", "other")):
            with self.subTest(key=key):
                save_json(path, {**original, key: value})
                self.config["users_sha256"] = digest(path)
                self.save_config()
                with self.assertRaises(ValueError):
                    self.prepare()

    def test_saved_artifacts_are_complete_and_stable_across_repeated_preparation(self):
        first, second = self.prepare(), self.prepare()
        self.assertEqual(first, second)
        output = self.work / "prepared"
        write_prepared(output, first)
        expected = {"predictions.jsonl", "preparation.json", "source_validation.json", "derived_validation.json"}
        self.assertEqual({path.name for path in output.iterdir()}, expected)
        before = {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()}
        write_prepared(output, second)
        self.assertEqual(before, {path.name: (path.read_bytes(), path.stat().st_mtime_ns) for path in output.iterdir()})
        report = read_json(output / "derived_validation.json")
        self.assertEqual(report["predictions"], "derived predictions.jsonl")
        self.assertNotIn("evaluation-preparation-", json.dumps(first.summary) + json.dumps(report))
        self.assertEqual(digest(output / "predictions.jsonl"), first.summary["output_sha256"]["predictions"])

    def test_writer_refuses_existing_different_or_partial_results(self):
        output = self.work / "prepared"
        result = self.prepare()
        output.mkdir()
        preserved = output / "notes.txt"
        preserved.write_text("keep me", encoding="utf-8")
        with self.assertRaises(ValueError):
            write_prepared(output, result)
        self.assertEqual(preserved.read_text(), "keep me")
        self.assertEqual(list(output.iterdir()), [preserved])

    def test_writer_rejects_changed_records_that_disagree_with_output_hash(self):
        result = self.prepare()
        result.records[0]["recommendations"][0]["movie_id"] = 40
        output = self.work / "prepared"
        with self.assertRaisesRegex(ValueError, "hash"):
            write_prepared(output, result)
        self.assertFalse(output.exists())

    def test_writer_protects_data_directory_and_source_file(self):
        result = self.prepare()
        source_bytes = self.source.read_bytes()
        for output in (self.data / "prepared", self.data, self.source):
            with self.subTest(output=output), self.assertRaises(ValueError):
                write_prepared(output, result)
        self.assertEqual(self.source.read_bytes(), source_bytes)

    def test_writer_rejects_symlink_and_hardlink_input_aliases(self):
        result = self.prepare()
        output = self.work / "prepared"
        output.mkdir()
        destination = output / "predictions.jsonl"
        original = self.source.read_bytes()
        destination.symlink_to(self.source)
        with self.assertRaises(ValueError):
            write_prepared(output, result)
        destination.unlink()
        os.link(self.source, destination)
        with self.assertRaises(ValueError):
            write_prepared(output, result)
        self.assertEqual(self.source.read_bytes(), original)
        linked_directory = self.work / "linked_directory"
        linked_directory.symlink_to(output, target_is_directory=True)
        with self.assertRaises(ValueError):
            write_prepared(linked_directory, result)

    def test_plain_user_export_preserves_frozen_file_and_records_provenance(self):
        frozen_path = self.data / "users.json"
        before = frozen_path.read_bytes()
        output = self.work / "plain_users"
        summary = export_users(self.config_path, output, repository_root=self.work)
        self.assertEqual(summary["status"], "users_exported")
        self.assertEqual(summary["selected_users"], 2)
        exported = read_json(output / "user_ids.json")
        self.assertEqual(exported, [1, 2])
        self.assertIsInstance(exported, list)  # Members 3 and 4 iterate this directly.
        self.assertEqual(summary["output_sha256"]["user_ids"], digest(output / "user_ids.json"))
        self.assertEqual(read_json(output / "user_ids.provenance.json"), summary)
        self.assertEqual(frozen_path.read_bytes(), before)
        self.assertEqual(export_users(self.config_path, output, repository_root=self.work), summary)

    def test_plain_user_export_rejects_overwrite_and_invalid_configuration(self):
        with self.assertRaises(ValueError):
            export_users(self.config_path, self.data / "plain", repository_root=self.work)
        output = self.work / "plain"
        export_users(self.config_path, output, repository_root=self.work)
        (output / "user_ids.json").write_text("[3]\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            export_users(self.config_path, output, repository_root=self.work)
        self.config["users_sha256"] = "0" * 64
        self.save_config()
        with self.assertRaises(ValueError):
            export_users(self.config_path, self.work / "other", repository_root=self.work)

    def test_cli_success_saves_artifacts_and_prints_the_saved_summary(self):
        code, stdout, stderr, output = self.cli()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual(json.loads(stdout), read_json(output / "preparation.json"))
        code, stdout, stderr, users = self.cli(command="users", output=self.work / "plain")
        self.assertEqual(code, 0, stderr)
        self.assertEqual(json.loads(stdout)["status"], "users_exported")
        self.assertEqual(read_json(users / "user_ids.json"), [1, 2])

    def test_cli_invalid_predictions_exit_one_without_creating_output(self):
        records = read_jsonl(self.source)
        records[1]["recommendations"][0]["movie_id"] = 70
        self.write_records(records)
        code, stdout, stderr, output = self.cli()
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout)["status"], "invalid")
        self.assertIn("watched", stderr.lower())
        self.assertFalse(output.exists())

    def test_cli_bad_configuration_or_source_hash_exit_two_without_output(self):
        code, stdout, stderr, output = self.cli(extra=("--expected-source-sha256", "0" * 64))
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertIn("SHA-256", stderr)
        self.assertFalse(output.exists())
        self.config["averaging"] = "micro"
        self.save_config()
        code, stdout, stderr, output = self.cli()
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertFalse(output.exists())

    def test_prepared_artifacts_score_compare_and_generate_known_synthetic_table(self):
        embedding_dir, p1_dir = self.work / "prepared_embedding", self.work / "prepared_p1"
        write_prepared(embedding_dir, self.prepare())
        write_prepared(p1_dir, self.p1())
        embedding = evaluate(self.config_path, embedding_dir / "predictions.jsonl", method="embedding",
                             repository_root=self.work)
        p1 = evaluate(self.config_path, p1_dir / "predictions.jsonl", method="p1",
                      candidate_pool_path=p1_dir / "candidate_pool.jsonl", repository_root=self.work)
        self.assertEqual(embedding.summary["metrics"], {
            "recall_at_2": 0.5, "ndcg_at_2": 0.5, "hit_rate_at_2": 1.0, "recall_at_3": 0.75,
        })
        self.assertEqual(p1.summary["metrics"]["recall_at_2"], 0.75)
        self.assertAlmostEqual(p1.summary["metrics"]["ndcg_at_2"], 0.8065735963827292)
        baseline, challenger = self.work / "scored_embedding", self.work / "scored_p1"
        write_results(baseline, embedding)
        write_results(challenger, p1)
        comparison = compare_runs(baseline, challenger)
        compared = self.work / "comparison"
        write_comparison(compared, comparison)
        table = build_table([baseline, challenger], comparison_dirs=[compared], kind="synthetic")
        output = self.work / "table"
        write_table(output, table)
        markdown = (output / "table.md").read_text(encoding="utf-8")
        self.assertIn("0.806574", markdown)
        self.assertIn("0.750000", markdown)
        self.assertIn("synthetic", markdown.lower())


if __name__ == "__main__":
    unittest.main()
