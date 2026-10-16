"""Reporting checks using independently hand-checked synthetic scorer outputs.

These fixtures verify presentation and provenance, not model performance.
"""

import contextlib
import csv
import io
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bootstrap import compare_runs, write_comparison
from src.evaluation.evaluate import evaluate, write_results
from src.evaluation.report import build_table, main, write_table


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evaluation_scoring"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class ExperimentTableTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.data = self.work / "data"
        shutil.copytree(FIXTURES, self.data)
        self.config_path = self.work / "config.json"
        config = read_json(self.data / "config.json")
        config.update(data_dir=str(self.data), users_file=str(self.data / "users.json"))
        save_json(self.config_path, config)
        self.embedding = self.work / "embedding"
        self.p1 = self.work / "p1"
        write_results(self.embedding, evaluate(self.config_path, self.data / "embedding.jsonl",
                                              method="embedding", repository_root=self.work))
        write_results(self.p1, evaluate(self.config_path, self.data / "p1.jsonl", method="p1",
                                       candidate_pool_path=self.data / "embedding.jsonl",
                                       repository_root=self.work))
        self.comparison = self.work / "p1_vs_embedding"
        write_comparison(self.comparison, compare_runs(self.embedding, self.p1))

    def table(self, *, comparisons=False, runs=None, kind="synthetic"):
        return build_table(runs if runs is not None else [self.embedding, self.p1],
                           comparison_dirs=[self.comparison] if comparisons else [], kind=kind)

    def mutate_json(self, path, mutation):
        value = read_json(path)
        mutation(value)
        save_json(path, value)

    def read_csv(self, path):
        with path.open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            return list(reader.fieldnames), list(reader)

    def save_csv(self, path, columns, rows):
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def cli(self, *, output=None, comparisons=True, kind="synthetic"):
        output = output or self.work / "table"
        arguments = ["--runs", str(self.embedding), str(self.p1)]
        if comparisons:
            arguments.extend(["--comparisons", str(self.comparison)])
        arguments.extend(["--kind", kind, "--output-dir", str(output)])
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(arguments)
        return code, stdout.getvalue(), stderr.getvalue(), output

    def make_baseline(self, method):
        predictions = self.data / f"{method}.jsonl"
        records = [json.loads(line) for line in (self.data / "embedding.jsonl").read_text().splitlines()]
        for record in records:
            record["method"] = method
        predictions.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        directory = self.work / method
        write_results(directory, evaluate(self.config_path, predictions, method=method,
                                           repository_root=self.work))
        return directory

    def test_rows_match_hand_calculated_synthetic_means(self):
        result = self.table()
        self.assertEqual(result.summary["status"], "table_generated")
        self.assertEqual(result.summary["kind"], "synthetic")
        self.assertEqual(result.summary["run_count"], 2)
        self.assertEqual(result.summary["methods"], ["embedding", "p1"])
        self.assertEqual(result.summary["missing_methods"], ["popularity", "bm25"])
        self.assertEqual([row["method"] for row in result.rows], ["embedding", "p1"])
        before, after = result.rows
        for key, value in {"recall_at_2": 0.5, "ndcg_at_2": 0.5,
                           "hit_rate_at_2": 1.0, "recall_at_3": 0.75}.items():
            self.assertEqual(before[key], value)
        self.assertEqual(after["recall_at_2"], 0.75)
        self.assertAlmostEqual(after["ndcg_at_2"], 0.8065735963827292)
        self.assertEqual(after["recall_at_3"], 0.75)
        self.assertEqual(after["hit_rate_at_2"], 1.0)

    def test_all_four_methods_have_canonical_order(self):
        popularity = self.make_baseline("popularity")
        bm25 = self.make_baseline("bm25")
        result = self.table(runs=[self.p1, bm25, self.embedding, popularity])
        self.assertEqual([row["method"] for row in result.rows], ["popularity", "bm25", "embedding", "p1"])
        self.assertEqual(result.summary["missing_methods"], [])

    def test_catalog_coverage_populations_have_separate_labels(self):
        for row in self.table().rows:
            self.assertEqual(row["user_count"], 2)
            self.assertEqual(row["prediction_user_coverage"], 1.0)
            self.assertEqual(row["selected_users_catalog_positive_coverage"], 0.8)
            self.assertAlmostEqual(row["full_split_catalog_positive_coverage"], 2 / 3)
            self.assertEqual(row["selected_users_positive_targets"], 5)
            self.assertEqual(row["selected_users_in_catalog_positive_targets"], 4)
            self.assertEqual(row["full_split_positive_targets"], 6)

    def test_comparisons_are_optional_without_fabricated_intervals(self):
        result = self.table()
        self.assertEqual(result.summary["comparison_count"], 0)
        self.assertEqual(result.comparisons, [])
        self.assertFalse(any("ci_" in key for row in result.rows for key in row))

    def test_comparison_rows_match_saved_paired_intervals(self):
        result = self.table(comparisons=True)
        self.assertEqual(result.summary["comparison_count"], 1)
        rows = {row["metric"]: row for row in result.comparisons}
        recall = rows["recall_at_2"]
        self.assertEqual(recall["baseline_method"], "embedding")
        self.assertEqual(recall["challenger_method"], "p1")
        self.assertEqual(recall["mean_difference"], 0.25)
        self.assertEqual((recall["ci_lower"], recall["ci_upper"]), (0.0, 0.5))
        self.assertEqual((recall["confidence"], recall["resamples"], recall["seed"]),
                         (0.95, 10000, 20261002))
        self.assertAlmostEqual(rows["ndcg_at_2"]["mean_difference"], 0.3065735963827292)
        self.assertAlmostEqual(rows["ndcg_at_2"]["ci_lower"], 0.2262943855309167)
        self.assertAlmostEqual(rows["ndcg_at_2"]["ci_upper"], 0.38685280723454163)
        for metric in ("recall_at_3", "hit_rate_at_2"):
            self.assertEqual((rows[metric]["mean_difference"], rows[metric]["ci_lower"], rows[metric]["ci_upper"]),
                             (0.0, 0.0, 0.0))

    def test_markdown_labels_synthetic_and_draft_and_csv_keeps_precision(self):
        result = self.table(comparisons=True)
        self.assertIn("synthetic", result.markdown.lower())
        self.assertIn("draft_for_team_review", result.markdown)
        self.assertIn("0.806574", result.markdown)
        self.assertIn("0.306574", result.markdown)
        self.assertEqual(result.summary["config_status"], "draft_for_team_review")
        output = self.work / "table"
        write_table(output, result)
        _, rows = self.read_csv(output / "table.csv")
        self.assertAlmostEqual(float(rows[1]["ndcg_at_2"]), 0.8065735963827292, places=14)
        self.assertNotEqual(rows[1]["ndcg_at_2"], "0.806574")

    def test_input_order_does_not_change_rows_or_display(self):
        first = self.table(comparisons=True)
        second = build_table([self.p1, self.embedding], comparison_dirs=[self.comparison], kind="synthetic")
        self.assertEqual(first.rows, second.rows)
        self.assertEqual(first.comparisons, second.comparisons)
        self.assertEqual(first.markdown, second.markdown)

    def test_relocated_report_bundles_need_no_original_inputs(self):
        moved = self.work / "portable"
        moved.mkdir()
        for source in (self.embedding, self.p1, self.comparison):
            shutil.copytree(source, moved / source.name)
        shutil.rmtree(self.data)
        self.config_path.unlink()
        for source in (self.embedding, self.p1, self.comparison):
            shutil.rmtree(source)
        result = build_table([moved / "embedding", moved / "p1"],
                             comparison_dirs=[moved / "p1_vs_embedding"], kind="synthetic")
        self.assertEqual(result.summary["run_count"], 2)
        self.assertEqual(result.summary["comparison_count"], 1)
        write_table(moved / "table", result)
        self.assertTrue((moved / "table" / "table.md").is_file())

    def test_empty_run_list_and_duplicate_method_are_rejected(self):
        for runs in ([], [self.embedding, self.embedding]):
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                self.table(runs=runs)

    def test_unsupported_method_is_rejected(self):
        self.mutate_json(self.embedding / "summary.json", lambda value: value.update(method="unknown"))
        with self.assertRaises(ValueError):
            self.table()

    def test_mismatched_frozen_experiment_and_scorer_are_rejected(self):
        path = self.p1 / "summary.json"
        original = path.read_bytes()
        mutations = [lambda value: value["source_sha256"].update(targets="0" * 64),
                     lambda value: value["implementation"]["source_sha256"].update(metrics="0" * 64),
                     lambda value: value["settings"]["bootstrap"].update(seed=18)]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                path.write_bytes(original)
                self.mutate_json(path, mutation)
                with self.assertRaises(ValueError):
                    self.table()

    def test_different_user_ids_are_rejected(self):
        path = self.p1 / "per_user.csv"
        columns, rows = self.read_csv(path)
        rows[-1]["user_id"] = "999"
        self.save_csv(path, columns, rows)
        with self.assertRaises(ValueError):
            self.table()

    def test_different_per_user_truth_counts_are_rejected(self):
        path = self.p1 / "per_user.csv"
        columns, rows = self.read_csv(path)
        for index, change in enumerate((1, -1)):
            for key in ("relevant_targets", "positive_targets"):
                rows[index][key] = str(int(rows[index][key]) + change)
        self.save_csv(path, columns, rows)
        with self.assertRaises(ValueError):
            self.table()

    def test_p1_requires_embedding_and_matching_pool_hash(self):
        with self.assertRaises(ValueError):
            self.table(runs=[self.p1])
        for name in ("summary.json", "validation.json"):
            self.mutate_json(self.p1 / name,
                             lambda value: value["source_sha256"].update(candidate_pool="0" * 64))
        with self.assertRaises(ValueError):
            self.table()

    def test_invalid_validation_report_is_rejected(self):
        self.mutate_json(self.embedding / "validation.json", lambda value: value.update(status="invalid"))
        with self.assertRaises(ValueError):
            self.table()

    def test_invalid_kind_and_mislabeled_synthetic_experiment_are_rejected(self):
        for kind in ("experiment", "unknown", "", None):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.table(kind=kind)

    def test_experiment_kind_accepts_non_synthetic_dataset_and_preserves_draft(self):
        for directory in (self.embedding, self.p1):
            def update(value):
                value["dataset_version"] = "ml25m-global-v1"
                value["settings"]["dataset_version"] = "ml25m-global-v1"
            self.mutate_json(directory / "summary.json", update)
        result = self.table(kind="experiment")
        self.assertEqual(result.summary["kind"], "experiment")
        self.assertIn("draft_for_team_review", result.markdown)

    def test_comparison_requires_both_included_run_methods(self):
        with self.assertRaises(ValueError):
            self.table(runs=[self.embedding], comparisons=True)

    def test_duplicate_comparison_is_rejected(self):
        with self.assertRaises(ValueError):
            build_table([self.embedding, self.p1], comparison_dirs=[self.comparison, self.comparison],
                        kind="synthetic")

    def test_saved_comparison_values_are_rederived_and_tampering_is_rejected(self):
        path = self.comparison / "comparison.json"
        original = path.read_bytes()
        mutations = [lambda value: value["metrics"]["recall_at_2"].update(mean_difference=0.125),
                     lambda value: value["metrics"]["recall_at_2"].update(ci_lower=0.1),
                     lambda value: value["metrics"]["recall_at_2"].update(ci_upper=0.75),
                     lambda value: value["metrics"]["ndcg_at_2"].update(baseline_mean=0.25),
                     lambda value: value["bootstrap"].update(seed=1),
                     lambda value: value.update(difference_direction="baseline_minus_challenger"),
                     lambda value: value.update(user_count=3)]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                path.write_bytes(original)
                self.mutate_json(path, mutation)
                with self.assertRaises(ValueError):
                    self.table(comparisons=True)

    def test_all_six_comparison_source_report_hashes_must_match(self):
        path = self.comparison / "comparison.json"
        original = path.read_bytes()
        for key in ("baseline_summary", "baseline_per_user", "baseline_validation",
                    "challenger_summary", "challenger_per_user", "challenger_validation"):
            with self.subTest(key=key):
                path.write_bytes(original)
                self.mutate_json(path, lambda value, key=key: value["source_sha256"].update({key: "0" * 64}))
                with self.assertRaises(ValueError):
                    self.table(comparisons=True)

    def test_tampered_per_user_comparison_rows_are_rejected(self):
        path = self.comparison / "per_user_differences.csv"
        original = path.read_bytes()
        for column, value in (("difference_recall_at_2", "0.125"),
                              ("baseline_recall_at_2", "0.125"), ("user_id", "999"),
                              ("challenger_method", "embedding")):
            with self.subTest(column=column):
                path.write_bytes(original)
                columns, rows = self.read_csv(path)
                rows[0][column] = value
                self.save_csv(path, columns, rows)
                with self.assertRaises(ValueError):
                    self.table(comparisons=True)

    def test_comparison_csv_row_order_is_keyed_by_user_id(self):
        first = self.table(comparisons=True)
        path = self.comparison / "per_user_differences.csv"
        columns, rows = self.read_csv(path)
        self.save_csv(path, columns, list(reversed(rows)))
        second = self.table(comparisons=True)
        self.assertEqual(first.comparisons, second.comparisons)

    def test_cli_writes_four_files_and_reports_success(self):
        code, stdout, stderr, output = self.cli()
        self.assertEqual((code, stderr), (0, ""))
        summary = json.loads(stdout)
        self.assertEqual(summary["status"], "table_generated")
        self.assertEqual({path.name for path in output.iterdir()},
                         {"table.csv", "table.md", "differences.csv", "report.json"})
        self.assertEqual(read_json(output / "report.json"), summary)

    def test_cli_invalid_input_creates_no_output_and_returns_two(self):
        self.mutate_json(self.p1 / "summary.json", lambda value: value["metrics"].update(recall_at_2=0.1))
        code, stdout, stderr, output = self.cli()
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr)
        self.assertFalse(output.exists())

    def test_saved_table_is_immutable_and_identical_rerun_is_accepted(self):
        result = self.table(comparisons=True)
        output = self.work / "table"
        write_table(output, result)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        write_table(output, result)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})
        (output / "table.md").write_text("Preserve my existing report.\n", encoding="utf-8")
        with self.assertRaises(ValueError):
            write_table(output, result)
        self.assertEqual((output / "table.md").read_text(), "Preserve my existing report.\n")
        self.assertEqual((output / "table.csv").read_bytes(), before["table.csv"])

    def test_outputs_cannot_replace_or_nest_inside_inputs(self):
        result = self.table(comparisons=True)
        before = (self.embedding / "summary.json").read_bytes()
        for destination in (self.embedding, self.embedding / "nested", self.comparison / "nested",
                            self.data / "nested", self.embedding / "summary.json"):
            with self.subTest(destination=destination), self.assertRaises(ValueError):
                write_table(destination, result)
        self.assertEqual((self.embedding / "summary.json").read_bytes(), before)

    def test_symlinks_and_hardlinks_cannot_alias_input_files(self):
        result = self.table(comparisons=True)
        source = self.embedding / "summary.json"
        before = source.read_bytes()
        linked_dir = self.work / "linked-dir"
        linked_dir.symlink_to(self.embedding, target_is_directory=True)
        with self.assertRaises(ValueError):
            write_table(linked_dir / "nested", result)
        for kind in ("symlink", "hardlink"):
            with self.subTest(kind=kind):
                output = self.work / kind
                output.mkdir()
                destination = output / "table.csv"
                if kind == "symlink":
                    destination.symlink_to(source)
                else:
                    os.link(source, destination)
                with self.assertRaises(ValueError):
                    write_table(output, result)
                self.assertEqual(source.read_bytes(), before)

    def test_single_baseline_table_has_no_placeholder_models_or_comparisons(self):
        result = self.table(runs=[self.embedding])
        self.assertEqual(len(result.rows), 1)
        self.assertEqual(result.rows[0]["method"], "embedding")
        self.assertEqual(result.summary["missing_methods"], ["popularity", "bm25", "p1"])
        self.assertEqual(result.comparisons, [])
        output = self.work / "single"
        write_table(output, result)
        _, rows = self.read_csv(output / "differences.csv")
        self.assertEqual(rows, [])


if __name__ == "__main__":
    unittest.main()
