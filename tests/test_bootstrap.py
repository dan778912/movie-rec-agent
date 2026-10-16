"""Independent paired-bootstrap arithmetic and saved-run integrity checks.

The two-user examples are synthetic implementation checks. They are deliberately
too small to support claims about real recommendation performance.
"""

import contextlib
import copy
import csv
import io
import json
import random
import shutil
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bootstrap import compare_runs, main, paired_bootstrap, write_comparison
from src.evaluation.evaluate import evaluate, write_results


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evaluation_scoring"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


class PairedBootstrapArithmeticTest(unittest.TestCase):
    def test_constant_user_difference_has_zero_width_interval(self):
        result = paired_bootstrap({1: 0.0, 2: 0.5}, {1: 0.25, 2: 0.75})
        self.assertEqual(result["baseline_mean"], 0.25)
        self.assertEqual(result["challenger_mean"], 0.5)
        self.assertEqual(result["mean_difference"], 0.25)
        self.assertEqual((result["ci_lower"], result["ci_upper"]), (0.25, 0.25))
        self.assertEqual(result["user_count"], 2)
        self.assertEqual(result["resamples"], 10000)
        self.assertEqual(result["seed"], 20261002)
        self.assertEqual(result["confidence"], 0.95)

    def test_identical_methods_have_zero_difference_and_interval(self):
        result = paired_bootstrap({1: 0.0, 2: 1.0}, {2: 1.0, 1: 0.0}, resamples=12)
        self.assertEqual(result["mean_difference"], 0.0)
        self.assertEqual((result["ci_lower"], result["ci_upper"]), (0.0, 0.0))

    def test_single_user_resamples_the_same_difference(self):
        result = paired_bootstrap({71: 0.75}, {71: 0.25}, resamples=3)
        self.assertEqual(result["user_count"], 1)
        self.assertEqual(result["mean_difference"], -0.5)
        self.assertEqual((result["ci_lower"], result["ci_upper"]), (-0.5, -0.5))

    def test_pairing_uses_user_id_instead_of_mapping_insertion_order(self):
        first = paired_bootstrap({1: 0.0, 2: 0.5}, {2: 0.75, 1: 0.25}, resamples=50)
        second = paired_bootstrap({2: 0.5, 1: 0.0}, {1: 0.25, 2: 0.75}, resamples=50)
        self.assertEqual(first, second)
        self.assertEqual((first["ci_lower"], first["ci_upper"]), (0.25, 0.25))

    def test_small_resample_count_uses_linear_percentile_interpolation(self):
        # Seed 7 draws user-index triples 101, 200, 201, 202. Their means are
        # 1/3, 1/3, 1/2, 2/3. The 25th and 75th percentile indices are .75 and
        # 2.25, so the expected endpoints are 1/3 and 13/24.
        result = paired_bootstrap({1: 0.0, 2: 0.0, 3: 0.0},
                                  {1: 0.0, 2: 0.5, 3: 1.0},
                                  resamples=4, confidence=0.5, seed=7)
        self.assertAlmostEqual(result["mean_difference"], 0.5)
        self.assertAlmostEqual(result["ci_lower"], 1 / 3)
        self.assertAlmostEqual(result["ci_upper"], 13 / 24)

    def test_fixed_seed_is_reproducible_without_mutating_global_rng(self):
        before = random.getstate()
        kwargs = {"resamples": 37, "seed": -17, "confidence": 0.8}
        first = paired_bootstrap({1: 0.1, 2: 0.8, 3: 0.4}, {1: 0.7, 2: 0.4, 3: 0.6}, **kwargs)
        self.assertEqual(first, paired_bootstrap({1: 0.1, 2: 0.8, 3: 0.4},
                                              {1: 0.7, 2: 0.4, 3: 0.6}, **kwargs))
        self.assertEqual(random.getstate(), before)
        self.assertIn("percentile_convention", first)
        self.assertIn("rng", first)

    def test_empty_or_different_populations_are_rejected(self):
        for baseline, challenger in (({}, {}), ({1: 0.5}, {}),
                                     ({1: 0.5}, {2: 0.5}), ({1: 0.5}, {1: 0.5, 2: 0.5})):
            with self.subTest(baseline=baseline, challenger=challenger), self.assertRaises(ValueError):
                paired_bootstrap(baseline, challenger)

    def test_invalid_ids_or_scores_are_rejected(self):
        for user in (True, 0, -1, "1", 1.5):
            with self.subTest(user=user), self.assertRaises(ValueError):
                paired_bootstrap({user: 0.5}, {user: 0.75})
        for score in (True, "0.5", None, float("nan"), float("inf"), -0.1, 1.1):
            for side in ("baseline", "challenger"):
                with self.subTest(score=score, side=side), self.assertRaises(ValueError):
                    arguments = {"baseline": {1: 0.5}, "challenger": {1: 0.75}}
                    arguments[side][1] = score
                    paired_bootstrap(**arguments)

    def test_invalid_bootstrap_settings_are_rejected(self):
        cases = [("resamples", value) for value in (True, 0, -1, 1.5, "20")]
        cases += [("seed", value) for value in (True, 1.5, "7", None)]
        cases += [("confidence", value) for value in
                  (True, 0, 1, -0.1, 1.1, "0.95", float("nan"), float("inf"))]
        for key, value in cases:
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                paired_bootstrap({1: 0.5}, {1: 0.75}, **{key: value})

    def test_huge_numeric_values_fail_cleanly(self):
        for side in ("baseline", "challenger"):
            with self.subTest(side=side), self.assertRaises(ValueError):
                values = {"baseline": {1: 0.5}, "challenger": {1: 0.75}}
                values[side][1] = 10 ** 500
                paired_bootstrap(**values)
        with self.assertRaises(ValueError):
            paired_bootstrap({1: 0.5}, {1: 0.75}, confidence=10 ** 500)


class BootstrapRunComparisonTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.data = self.work / "data"
        shutil.copytree(FIXTURES, self.data)
        self.config_path = self.work / "config.json"
        config = read_json(FIXTURES / "config.json")
        config.update(data_dir=str(self.data), users_file=str(self.data / "users.json"))
        save_json(self.config_path, config)
        self.baseline = self.work / "embedding"
        self.challenger = self.work / "p1"
        before = evaluate(self.config_path, self.data / "embedding.jsonl", method="embedding",
                          repository_root=self.work)
        after = evaluate(self.config_path, self.data / "p1.jsonl", method="p1",
                         candidate_pool_path=self.data / "embedding.jsonl", repository_root=self.work)
        write_results(self.baseline, before)
        write_results(self.challenger, after)

    def compare(self):
        return compare_runs(self.baseline, self.challenger)

    def mutate_json(self, name, mutation, *, directory=None):
        path = (directory or self.challenger) / name
        value = read_json(path)
        mutation(value)
        save_json(path, value)

    def read_csv(self, directory=None):
        with ((directory or self.challenger) / "per_user.csv").open(newline="", encoding="utf-8") as source:
            reader = csv.DictReader(source)
            return list(reader.fieldnames), list(reader)

    def save_csv(self, columns, rows, *, directory=None):
        with ((directory or self.challenger) / "per_user.csv").open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def cli(self, *, output=None):
        output = output or self.work / "comparison"
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--baseline-dir", str(self.baseline), "--challenger-dir", str(self.challenger),
                         "--output-dir", str(output)])
        return code, stdout.getvalue(), stderr.getvalue(), output

    def test_synthetic_comparison_matches_hand_calculated_paired_intervals(self):
        result = self.compare()
        summary = result.summary
        self.assertEqual(summary["status"], "compared")
        self.assertEqual(summary["difference_direction"], "challenger_minus_baseline")
        self.assertEqual((summary["baseline_method"], summary["challenger_method"]), ("embedding", "p1"))
        self.assertEqual(summary["user_count"], 2)
        ndcg = summary["metrics"]["ndcg_at_2"]
        self.assertAlmostEqual(ndcg["mean_difference"], 0.3065735963827292)
        self.assertAlmostEqual(ndcg["ci_lower"], 0.2262943855309167)
        self.assertAlmostEqual(ndcg["ci_upper"], 0.38685280723454163)
        recall = summary["metrics"]["recall_at_2"]
        self.assertEqual(recall["mean_difference"], 0.25)
        self.assertEqual((recall["ci_lower"], recall["ci_upper"]), (0.0, 0.5))
        for name in ("recall_at_3", "hit_rate_at_2"):
            self.assertEqual(summary["metrics"][name]["mean_difference"], 0.0)
            self.assertEqual(summary["metrics"][name]["ci_lower"], 0.0)
            self.assertEqual(summary["metrics"][name]["ci_upper"], 0.0)
        self.assertEqual([row["user_id"] for row in result.per_user], [1, 2])
        self.assertAlmostEqual(result.per_user[0]["difference_ndcg_at_2"], 0.38685280723454163)

    def test_per_user_pairing_is_independent_of_csv_row_order(self):
        first = self.compare()
        columns, rows = self.read_csv()
        self.save_csv(columns, list(reversed(rows)))
        second = self.compare()
        self.assertEqual(first.per_user, second.per_user)
        self.assertEqual(first.summary["metrics"], second.summary["metrics"])

    def test_reversing_methods_negates_the_difference_and_interval(self):
        forward = self.compare().summary["metrics"]
        reverse = compare_runs(self.challenger, self.baseline).summary["metrics"]
        for metric in forward:
            with self.subTest(metric=metric):
                self.assertAlmostEqual(reverse[metric]["mean_difference"],
                                       -forward[metric]["mean_difference"])
                self.assertAlmostEqual(reverse[metric]["ci_lower"], -forward[metric]["ci_upper"])
                self.assertAlmostEqual(reverse[metric]["ci_upper"], -forward[metric]["ci_lower"])

    def test_portable_reports_do_not_require_original_data_or_predictions(self):
        moved = self.work / "portable"
        moved.mkdir()
        shutil.copytree(self.baseline, moved / "embedding")
        shutil.copytree(self.challenger, moved / "p1")
        shutil.rmtree(self.data)
        self.config_path.unlink()
        result = compare_runs(moved / "embedding", moved / "p1")
        self.assertEqual(result.summary["user_count"], 2)
        self.assertAlmostEqual(result.summary["metrics"]["recall_at_2"]["mean_difference"], 0.25)

    def test_mismatched_frozen_experiment_provenance_is_rejected(self):
        path = self.challenger / "summary.json"
        original = path.read_bytes()
        mutations = [lambda value: value.update(dataset_version="different-data"),
                     lambda value: value.update(evaluation_version="different-evaluation"),
                     lambda value: value.update(split="test")]
        mutations += [lambda value, key=key: value["source_sha256"].update({key: "0" * 64})
                      for key in ("config", "manifest", "catalog", "profiles", "targets", "users")]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                path.write_bytes(original)
                self.mutate_json("summary.json", mutation)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_mismatched_scoring_implementation_is_rejected(self):
        self.mutate_json("summary.json", lambda value:
                         value["implementation"]["source_sha256"].update(metrics="0" * 64))
        with self.assertRaises(ValueError):
            self.compare()

    def test_mismatched_metric_sets_are_rejected(self):
        self.mutate_json("summary.json", lambda value: value["metrics"].pop("recall_at_3"))
        with self.assertRaises(ValueError):
            self.compare()

    def test_different_settings_or_invalid_bootstrap_settings_are_rejected(self):
        path = self.challenger / "summary.json"
        original = path.read_bytes()
        mutations = [lambda value: value["settings"].update(ranking_k=[1]),
                     lambda value: value["settings"]["bootstrap"].update(seed=3),
                     lambda value: value["settings"]["bootstrap"].update(method="independent"),
                     lambda value: value["settings"]["bootstrap"].update(resamples=True),
                     lambda value: value["settings"]["bootstrap"].update(confidence=1.0)]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                path.write_bytes(original)
                self.mutate_json("summary.json", mutation)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_unsupported_rules_are_rejected_even_when_both_runs_agree(self):
        originals = {directory: (directory / "summary.json").read_bytes()
                     for directory in (self.baseline, self.challenger)}
        for key, value in (("positive_rating_at_least", 3.0),
                           ("positive_rating_at_least", True),
                           ("path_resolution", "config_parent")):
            with self.subTest(key=key, value=value):
                for directory, original in originals.items():
                    (directory / "summary.json").write_bytes(original)
                    self.mutate_json("summary.json", lambda summary: summary["settings"].update({key: value}),
                                     directory=directory)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_missing_duplicate_or_unexpected_csv_users_are_rejected(self):
        columns, rows = self.read_csv()
        altered = copy.deepcopy(rows)
        altered[1]["user_id"] = "999"
        for label, changed in (("missing", rows[:1]), ("duplicate", rows + rows[:1]),
                               ("different user", altered)):
            with self.subTest(label=label):
                self.save_csv(columns, changed)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_wrong_csv_method_split_ids_or_scores_are_rejected(self):
        columns, original = self.read_csv()
        cases = [("method", "embedding"), ("split", "test"), ("user_id", "1.5"),
                 ("relevant_targets", "0"), ("prediction_count", "2"),
                 ("ndcg_at_2", "NaN"), ("ndcg_at_2", "Infinity"), ("ndcg_at_2", "1.1")]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                rows = copy.deepcopy(original)
                rows[0][key] = value
                self.save_csv(columns, rows)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_csv_duplicate_columns_or_malformed_row_are_rejected(self):
        path = self.challenger / "per_user.csv"
        original = path.read_text()
        lines = original.splitlines()
        cases = [lines[0] + ",user_id\n" + "\n".join(line + ",1" for line in lines[1:]) + "\n",
                 original + "1,p1\n", original + lines[1] + ",unexpected\n"]
        for index, text in enumerate(cases):
            with self.subTest(index=index):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_summary_means_cannot_disagree_with_per_user_scores(self):
        self.mutate_json("summary.json", lambda value: value["metrics"].update(ndcg_at_2=0.1))
        with self.assertRaises(ValueError):
            self.compare()

    def test_paired_users_must_have_matching_target_counts(self):
        columns, rows = self.read_csv()
        rows[0]["positive_targets"] = "4"
        rows[0]["out_of_catalog_positive_targets"] = "2"
        self.save_csv(columns, rows)
        self.mutate_json("summary.json", lambda value: value["coverage"]["selected_users"].update(
            positive_targets=6, out_of_catalog_positive_targets=2, catalog_positive_coverage=2 / 3))
        with self.assertRaises(ValueError):
            self.compare()

    def test_p1_pool_must_be_the_actual_embedding_baseline_predictions(self):
        different_hash = "f" * 64
        self.mutate_json("summary.json", lambda value:
                         value["source_sha256"].update(candidate_pool=different_hash))
        self.mutate_json("validation.json", lambda value:
                         value["source_sha256"].update(candidate_pool=different_hash))
        with self.assertRaises(ValueError):
            self.compare()

    def test_invalid_or_inconsistent_validation_reports_are_rejected(self):
        path = self.challenger / "validation.json"
        original = path.read_bytes()
        mutations = [lambda value: value.update(status="invalid"),
                     lambda value: value.update(prediction_user_coverage=0.5),
                     lambda value: value.update(valid_users=1),
                     lambda value: value.update(missing_users=[2]),
                     lambda value: value.update(unexpected_users=[99]),
                     lambda value: value.update(error_count=1),
                     lambda value: value.update(method="embedding"),
                     lambda value: value.update(required_count=2),
                     lambda value: value["source_sha256"].update(predictions="0" * 64)]
        for index, mutation in enumerate(mutations):
            with self.subTest(index=index):
                path.write_bytes(original)
                self.mutate_json("validation.json", mutation)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_duplicate_json_keys_and_nonfinite_constants_are_rejected(self):
        path = self.challenger / "summary.json"
        original = path.read_text()
        for text in ('{"status":"invalid",' + original[1:], original.replace('"user_count":2', '"user_count":NaN')):
            with self.subTest(text=text[:40]):
                path.write_text(text)
                with self.assertRaises(ValueError):
                    self.compare()

    def test_cli_writes_two_consistent_comparison_artifacts(self):
        code, stdout, stderr, output = self.cli()
        self.assertEqual(code, 0, stderr)
        self.assertEqual(stderr, "")
        self.assertEqual({path.name for path in output.iterdir()},
                         {"comparison.json", "per_user_differences.csv"})
        self.assertEqual(json.loads(stdout), read_json(output / "comparison.json"))
        with (output / "per_user_differences.csv").open(newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual([int(row["user_id"]) for row in rows], [1, 2])
        self.assertAlmostEqual(float(rows[0]["difference_ndcg_at_2"]), 0.38685280723454163)
        self.assertEqual(float(rows[1]["baseline_recall_at_3"]), 0.5)
        self.assertEqual(float(rows[1]["challenger_recall_at_3"]), 0.5)

    def test_cli_invalid_input_does_not_create_output(self):
        self.mutate_json("summary.json", lambda value: value.update(user_count=3))
        code, stdout, stderr, output = self.cli()
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        self.assertTrue(stderr)
        self.assertFalse(output.exists())

    def test_saved_results_are_immutable_and_identical_reruns_are_accepted(self):
        result = self.compare()
        output = self.work / "comparison"
        write_comparison(output, result)
        original = {path.name: path.read_bytes() for path in output.iterdir()}
        write_comparison(output, result)
        self.assertEqual(original, {path.name: path.read_bytes() for path in output.iterdir()})
        (output / "comparison.json").write_text("preserve me\n")
        with self.assertRaises(ValueError):
            write_comparison(output, result)
        self.assertEqual((output / "comparison.json").read_text(), "preserve me\n")

    def test_output_must_not_alias_report_inputs(self):
        result = self.compare()
        original = (self.baseline / "summary.json").read_bytes()
        for output in (self.baseline, self.baseline / "summary.json"):
            with self.subTest(output=output), self.assertRaises(ValueError):
                write_comparison(output, result)
        self.assertEqual((self.baseline / "summary.json").read_bytes(), original)
        output = self.work / "aliased_output"
        output.mkdir()
        (output / "comparison.json").symlink_to(self.baseline / "summary.json")
        with self.assertRaises(ValueError):
            write_comparison(output, result)
        self.assertEqual((self.baseline / "summary.json").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
