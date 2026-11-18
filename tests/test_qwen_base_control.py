"""Synthetic end-to-end checks for the separately named untuned Qwen control."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bootstrap import compare_runs, write_comparison
from src.evaluation.evaluate import evaluate, write_results
from src.evaluation.prepare import prepare_predictions
from src.evaluation.report import build_table
from src.evaluation.validate import PredictionValidationError


FIXTURES = Path(__file__).resolve().parent / "fixtures"


class QwenBaseControlTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.data = self.work / "data"
        shutil.copytree(FIXTURES / "evaluation_scoring", self.data)
        self.config_path = self.work / "config.json"
        config = json.loads((self.data / "config.json").read_text(encoding="utf-8"))
        config.update(data_dir=str(self.data), users_file=str(self.data / "users.json"))
        self.config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
        self.pool_path = self.data / "embedding.jsonl"
        self.qwen_path = self.work / "qwen_base.jsonl"
        self._write_qwen_from(self.pool_path)

    def _write_qwen_from(self, path):
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        for record in records:
            record["method"] = "qwen_base"
        self.qwen_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")

    def _score(self, method, predictions, output):
        result = evaluate(self.config_path, predictions, method=method,
                          candidate_pool_path=self.pool_path if method != "embedding" else None,
                          repository_root=self.work)
        write_results(output, result)
        return result

    def test_scores_base_control_as_distinct_fixed_pool_method(self):
        result = self._score("qwen_base", self.qwen_path, self.work / "base")
        self.assertEqual(result.summary["method"], "qwen_base")
        self.assertEqual(result.summary["metrics"]["ndcg_at_2"], 0.5)
        self.assertEqual(result.summary["user_count"], 2)
        self.assertIn("candidate_pool", result.summary["source_sha256"])
        self.assertEqual(result.validation_report["method"], "qwen_base")

    def test_rejects_control_with_movie_outside_frozen_b3_pool(self):
        records = [json.loads(line) for line in self.qwen_path.read_text(encoding="utf-8").splitlines()]
        user_one = next(record for record in records if record["user_id"] == 1)
        user_one["recommendations"][2]["movie_id"] = 40  # valid catalog movie, outside this user's B3 pool
        self.qwen_path.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        with self.assertRaises(PredictionValidationError) as raised:
            evaluate(self.config_path, self.qwen_path, method="qwen_base",
                     candidate_pool_path=self.pool_path, repository_root=self.work)
        self.assertIn("candidate_pool_mismatch", raised.exception.report["errors_by_code"])

    def test_pairs_base_with_p1_and_reports_separate_rows(self):
        b3 = self.work / "b3"
        base = self.work / "base"
        p1 = self.work / "p1"
        self._score("embedding", self.pool_path, b3)
        self._score("qwen_base", self.qwen_path, base)
        self._score("p1", self.data / "p1.jsonl", p1)
        comparison = compare_runs(base, p1)
        self.assertEqual((comparison.summary["baseline_method"], comparison.summary["challenger_method"]),
                         ("qwen_base", "p1"))
        self.assertAlmostEqual(comparison.summary["metrics"]["ndcg_at_2"]["mean_difference"],
                               0.8065735963827292 - 0.5)
        reverse = compare_runs(p1, base)
        self.assertAlmostEqual(reverse.summary["metrics"]["ndcg_at_2"]["mean_difference"],
                               0.5 - 0.8065735963827292)
        comparison_dir = self.work / "base_vs_p1"
        write_comparison(comparison_dir, comparison)
        table = build_table([p1, b3, base], comparison_dirs=[comparison_dir], kind="synthetic")
        self.assertEqual(table.summary["methods"], ["embedding", "qwen_base", "p1"])
        self.assertEqual(table.summary["missing_methods"], ["popularity", "bm25", "p2"])
        self.assertEqual([row["method"] for row in table.rows], ["embedding", "qwen_base", "p1"])
        self.assertTrue(any(row["baseline_method"] == "qwen_base" and row["challenger_method"] == "p1"
                            for row in table.comparisons))

    def test_table_rejects_base_scored_against_different_b3_source(self):
        b3 = self.work / "b3"
        base = self.work / "base"
        p1 = self.work / "p1"
        self._score("embedding", self.pool_path, b3)
        self._score("p1", self.data / "p1.jsonl", p1)
        alternate_pool = self.work / "alternate_b3.jsonl"
        records = [json.loads(line) for line in self.pool_path.read_text(encoding="utf-8").splitlines()]
        records.reverse()  # equivalent users and IDs, different source bytes
        alternate_pool.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        result = evaluate(self.config_path, self.qwen_path, method="qwen_base",
                          candidate_pool_path=alternate_pool, repository_root=self.work)
        write_results(base, result)
        with self.assertRaisesRegex(ValueError, "candidate pool hashes differ"):
            compare_runs(base, p1)
        with self.assertRaisesRegex(ValueError, "candidate pool hash does not match"):
            build_table([base, b3], kind="synthetic")

    def test_preparation_preserves_base_pool_and_forbids_truncation(self):
        prep_data = self.work / "prep_data"
        shutil.copytree(FIXTURES / "evaluation_preparation", prep_data)
        config = json.loads((prep_data / "config.json").read_text(encoding="utf-8"))
        config.update(data_dir=str(prep_data), users_file=str(prep_data / "users.json"))
        prep_config = self.work / "prep_config.json"
        prep_config.write_text(json.dumps(config) + "\n", encoding="utf-8")
        source = self.work / "full_qwen_base.jsonl"
        records = [json.loads(line) for line in (prep_data / "full_p1.jsonl").read_text(encoding="utf-8").splitlines()]
        for record in records:
            record["method"] = "qwen_base"
        source.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
        options = dict(method="qwen_base", source_count=3, source_users="all",
                       source_label="synthetic untuned Qwen", candidate_pool_path=prep_data / "full_candidate_pool.jsonl",
                       repository_root=self.work)
        prepared = prepare_predictions(prep_config, source, **options)
        self.assertEqual(prepared.summary["method"], "qwen_base")
        self.assertEqual(prepared.summary["selected_users"], 2)
        self.assertEqual(len(prepared.candidate_records), 2)
        with self.assertRaisesRegex(ValueError, "cannot be truncated"):
            prepare_predictions(prep_config, source, **{**options, "source_count": 4})


if __name__ == "__main__":
    unittest.main()
