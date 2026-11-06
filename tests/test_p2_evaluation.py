"""Synthetic Phase 2 trace, scoring, and comparison contract checks."""

import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from src.evaluation.bootstrap import compare_runs
from src.evaluation.evaluate import evaluate, write_results
from src.evaluation.report import build_table
from src.evaluation.prepare import prepare_predictions
from src.evaluation.validate import (PredictionValidationError, ValidationContext,
                                     load_context, validate_predictions)


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evaluation_scoring"


def write_jsonl(path, records):
    path.write_text("".join(json.dumps(record, separators=(",", ":")) + "\n" for record in records),
                    encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def explanation_event(ids):
    return {"action": "explain_recommendations",
            "explanations": [{"movie_id": movie_id, "text": f"Grounded synthetic reason for {movie_id}"}
                             for movie_id in ids]}


class P2EvaluationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.data = self.work / "data"
        shutil.copytree(FIXTURES, self.data)
        self.pool = self.data / "embedding.jsonl"
        self.config_path = self.work / "config.json"
        config = json.loads((FIXTURES / "config.json").read_text(encoding="utf-8"))
        config.update(data_dir=str(self.data), users_file=str(self.data / "users.json"))
        self.config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
        self.context = load_context(self.data / "catalog.csv", self.data / "validation_profiles.jsonl",
                                    self.data / "users.json", "validation")
        self.predictions_path = self.work / "p2.jsonl"
        self.trace_path = self.work / "trace.jsonl"
        self.predictions = [
            {"user_id": 1, "split": "validation", "method": "p2", "recommendations": [
                {"movie_id": movie_id, "rank": rank, "score": 4 - rank}
                for rank, movie_id in enumerate((20, 10, 30), 1)]},
            {"user_id": 2, "split": "validation", "method": "p2", "recommendations": [
                {"movie_id": movie_id, "rank": rank, "score": 4 - rank}
                for rank, movie_id in enumerate((50, 40, 60), 1)]},
        ]
        self.traces = [
            {"schema_version": "p2-trace-v1", "user_id": 1, "split": "validation", "method": "p2",
             "initial_candidate_movie_ids": [10, 30, 20],
             "events": [
                 {"action": "assess_sufficiency", "sufficient": True},
                 {"action": "evaluate_candidates", "candidate_movie_ids": [10, 30, 20],
                  "pool_sufficient": True},
                 {"action": "rank_candidates", "movie_ids": [20, 10, 30]},
                 explanation_event([20, 10, 30]),
             ],
             "final_candidate_movie_ids": [10, 30, 20], "termination_reason": "recommend"},
            {"schema_version": "p2-trace-v1", "user_id": 2, "split": "validation", "method": "p2",
             "initial_candidate_movie_ids": [60, 40, 70],
             "events": [
                 {"action": "assess_sufficiency", "sufficient": True},
                 {"action": "evaluate_candidates", "candidate_movie_ids": [60, 40, 70],
                  "pool_sufficient": False},
                 {"action": "retrieve_candidates", "added_movie_ids": [50], "index_version": "synthetic-v1"},
                 {"action": "evaluate_candidates", "candidate_movie_ids": [60, 40, 70, 50],
                  "pool_sufficient": True},
                 {"action": "rank_candidates", "movie_ids": [50, 40, 60]},
                 explanation_event([50, 40, 60]),
             ],
             "final_candidate_movie_ids": [60, 40, 70, 50], "termination_reason": "recommend"},
        ]
        self.save()

    def save(self):
        write_jsonl(self.predictions_path, self.predictions)
        write_jsonl(self.trace_path, self.traces)

    def validate(self):
        return validate_predictions(self.predictions_path, self.context, method="p2",
                                    required_count=3, candidate_pool_path=self.pool,
                                    trace_path=self.trace_path)

    def invalid(self, code):
        with self.assertRaises(PredictionValidationError) as raised:
            self.validate()
        self.assertIn(code, raised.exception.report["errors_by_code"])
        return raised.exception.report

    def test_valid_expansion_binds_all_hashes_and_improves_final_recall(self):
        validated = self.validate()
        self.assertEqual(validated.report["status"], "valid")
        self.assertEqual(set(validated.traces), {1, 2})
        self.assertEqual(validated.report["trace_max_control_events"], 8)
        self.assertEqual(validated.report["trace_max_events"], 10)
        self.assertEqual(validated.report["source_sha256"]["trace"], sha256(self.trace_path))
        self.assertEqual(validated.report["source_sha256"]["candidate_pool"], sha256(self.pool))
        p2 = evaluate(self.config_path, self.predictions_path, method="p2",
                      candidate_pool_path=self.pool, trace_path=self.trace_path,
                      repository_root=self.work)
        b3 = evaluate(self.config_path, self.pool, method="embedding", repository_root=self.work)
        self.assertEqual(p2.summary["metrics"]["recall_at_3"], 1.0)
        self.assertEqual(b3.summary["metrics"]["recall_at_3"], 0.75)
        self.assertEqual(p2.summary["source_sha256"]["trace"], sha256(self.trace_path))

    def test_p1_p2_paired_bootstrap_and_table(self):
        b3_dir, p1_dir, p2_dir = (self.work / name for name in ("b3", "p1", "p2"))
        write_results(b3_dir, evaluate(self.config_path, self.pool, method="embedding",
                                       repository_root=self.work))
        write_results(p1_dir, evaluate(self.config_path, self.data / "p1.jsonl", method="p1",
                                       candidate_pool_path=self.pool, repository_root=self.work))
        write_results(p2_dir, evaluate(self.config_path, self.predictions_path, method="p2",
                                       candidate_pool_path=self.pool, trace_path=self.trace_path,
                                       repository_root=self.work))
        comparison = compare_runs(p1_dir, p2_dir)
        self.assertEqual((comparison.summary["baseline_method"], comparison.summary["challenger_method"]),
                         ("p1", "p2"))
        self.assertEqual(comparison.summary["metrics"]["recall_at_3"]["mean_difference"], 0.25)
        self.assertEqual(compare_runs(b3_dir, p2_dir).summary["challenger_method"], "p2")
        table = build_table([p2_dir, b3_dir, p1_dir], kind="synthetic")
        self.assertEqual(table.summary["methods"], ["embedding", "p1", "p2"])
        self.assertEqual(table.rows[-1]["recall_at_3"], 1.0)

    def test_p2_report_trace_hash_tampering_is_rejected(self):
        p1_dir, p2_dir = self.work / "p1", self.work / "p2"
        write_results(p1_dir, evaluate(self.config_path, self.data / "p1.jsonl", method="p1",
                                       candidate_pool_path=self.pool, repository_root=self.work))
        write_results(p2_dir, evaluate(self.config_path, self.predictions_path, method="p2",
                                       candidate_pool_path=self.pool, trace_path=self.trace_path,
                                       repository_root=self.work))
        report_path = p2_dir / "validation.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report["source_sha256"]["trace"] = "0" * 64
        report_path.write_text(json.dumps(report) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source hashes disagree"):
            compare_runs(p1_dir, p2_dir)

    def test_cross_version_archive_is_rejected(self):
        p1_dir, p2_dir = self.work / "p1", self.work / "p2"
        write_results(p1_dir, evaluate(self.config_path, self.data / "p1.jsonl", method="p1",
                                       candidate_pool_path=self.pool, repository_root=self.work))
        write_results(p2_dir, evaluate(self.config_path, self.predictions_path, method="p2",
                                       candidate_pool_path=self.pool, trace_path=self.trace_path,
                                       repository_root=self.work))
        summary_path = p1_dir / "summary.json"
        old = json.loads(summary_path.read_text(encoding="utf-8"))
        old["implementation"]["source_sha256"]["validate"] = "0" * 64
        summary_path.write_text(json.dumps(old) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "scoring implementation"):
            compare_runs(p1_dir, p2_dir)

    def test_declared_p2_policy_cannot_disagree_with_validator(self):
        config = json.loads(self.config_path.read_text(encoding="utf-8"))
        config["p2_trace_schema"] = "unknown-trace"
        self.config_path.write_text(json.dumps(config) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "config.p2_trace_schema"):
            evaluate(self.config_path, self.predictions_path, method="p2",
                     candidate_pool_path=self.pool, trace_path=self.trace_path,
                     repository_root=self.work)

    def test_fallback_may_omit_assessments_but_must_preserve_b3_order(self):
        self.predictions[0]["recommendations"] = [
            {"movie_id": movie_id, "rank": rank, "score": 4 - rank}
            for rank, movie_id in enumerate((10, 30, 20), 1)]
        self.traces[0]["events"] = [
            {"action": "rank_candidates", "movie_ids": [10, 30, 20]},
            explanation_event([10, 30, 20]),
        ]
        self.traces[0].update(termination_reason="fallback", fallback_reason="model parse error",
                              fallback_source="frozen_b3_order")
        self.save()
        self.assertEqual(self.validate().report["status"], "valid")
        self.predictions[0]["recommendations"] = [
            {"movie_id": movie_id, "rank": rank, "score": 4 - rank}
            for rank, movie_id in enumerate((20, 10, 30), 1)]
        self.save()
        self.invalid("trace_fallback_mismatch")

    def test_invalid_pool_provenance_and_movie_safety(self):
        cases = [
            (lambda t: t[1].update(initial_candidate_movie_ids=[40, 60, 70]), "trace_initial_pool_mismatch"),
            (lambda t: t[1]["events"][2].update(added_movie_ids=[80]), "trace_watched_movie"),
            (lambda t: t[1]["events"][2].update(added_movie_ids=[999]), "trace_unknown_movie"),
            (lambda t: t[1]["events"][2].update(added_movie_ids=[40]), "trace_pool_mismatch"),
            (lambda t: t[1].update(final_candidate_movie_ids=[60, 40, 70]), "trace_final_pool_mismatch"),
            (lambda t: t[1]["events"][3].update(candidate_movie_ids=[60, 40, 70]), "trace_pool_mismatch"),
        ]
        original = copy.deepcopy(self.traces)
        for mutation, code in cases:
            with self.subTest(code=code):
                self.traces = copy.deepcopy(original)
                mutation(self.traces)
                self.save()
                self.invalid(code)

    def test_invalid_decisions_action_order_loop_and_explanations(self):
        original = copy.deepcopy(self.traces)
        cases = [
            (lambda t: t[0]["events"][0].update(sufficient="true"), "trace_invalid_decision"),
            (lambda t: t[1]["events"][1].update(pool_sufficient=True), "trace_action_order"),
            (lambda t: t[1]["events"][3].update(pool_sufficient=False), "trace_unresolved_decision"),
            (lambda t: t[0]["events"].insert(1, {"action": "ask_clarification", "question": "Why?",
                                               "answer": "Comedy", "answer_source": "simulator"}), "trace_action_order"),
            (lambda t: t[0]["events"].insert(0, {"action": "invented_action"}), "trace_invalid_action"),
            (lambda t: t[0]["events"].extend([{"action": "assess_sufficiency", "sufficient": True}] * 7),
             "trace_loop_limit"),
            (lambda t: t[0]["events"][-1]["explanations"][0].update(movie_id=30),
             "trace_explanation_mismatch"),
            (lambda t: t[0]["events"][-1]["explanations"][0].update(text=""),
             "trace_invalid_explanations"),
        ]
        for mutation, code in cases:
            with self.subTest(code=code):
                self.traces = copy.deepcopy(original)
                mutation(self.traces)
                self.save()
                self.invalid(code)

    def test_clarification_requires_reassessment_before_pool_evaluation(self):
        original = copy.deepcopy(self.traces[0]["events"])
        self.traces[0]["events"] = [
            {"action": "assess_sufficiency", "sufficient": False},
            {"action": "ask_clarification", "question": "What mood?", "answer": "Lighthearted",
             "answer_source": "offline_simulator"},
            {"action": "assess_sufficiency", "sufficient": True},
            *original[1:],
        ]
        self.save()
        self.assertEqual(self.validate().report["status"], "valid")
        del self.traces[0]["events"][2]
        self.save()
        self.invalid("trace_action_order")

    def test_cli_requires_trace_and_preparation_rejects_p2(self):
        command = [sys.executable, "-m", "src.evaluation.validate", "--catalog", str(self.data / "catalog.csv"),
                   "--profiles", str(self.data / "validation_profiles.jsonl"),
                   "--users", str(self.data / "users.json"), "--predictions", str(self.predictions_path),
                   "--split", "validation", "--method", "p2", "--required-count", "3",
                   "--candidate-pool", str(self.pool), "--trace", str(self.trace_path)]
        result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                text=True, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["status"], "valid")
        result = subprocess.run(command[:-2], cwd=Path(__file__).resolve().parents[1], capture_output=True,
                                text=True, check=False)
        self.assertEqual(result.returncode, 2)
        self.assertIn("requires --trace", result.stderr)
        with self.assertRaisesRegex(ValueError, "trace-aware"):
            prepare_predictions(self.config_path, self.predictions_path, method="p2",
                                source_count=3, source_users="selected", source_label="synthetic",
                                repository_root=self.work)

    def test_optional_prediction_explanation_must_agree_with_trace(self):
        self.predictions[0]["recommendations"][0]["explanation"] = "Different text"
        self.save()
        self.invalid("trace_explanation_mismatch")
        self.predictions[0]["recommendations"][0]["explanation"] = ""
        self.save()
        self.invalid("invalid_explanation")

    def test_missing_or_duplicate_trace_rows_are_rejected(self):
        original = copy.deepcopy(self.traces)
        self.traces = original[:1]
        self.save()
        self.invalid("trace_missing_users")
        self.traces = original + [copy.deepcopy(original[0])]
        self.save()
        self.invalid("trace_duplicate_user")
        self.traces = original
        self.save()
        raw = self.trace_path.read_text(encoding="utf-8")
        self.trace_path.write_text(raw.replace('"user_id":1', '"user_id":1,"user_id":1', 1),
                                   encoding="utf-8")
        self.invalid("trace_invalid_json")

    def test_required_inputs_and_non_p2_trace_rejection(self):
        with self.assertRaisesRegex(ValueError, "requires --trace"):
            validate_predictions(self.predictions_path, self.context, method="p2",
                                 required_count=3, candidate_pool_path=self.pool)
        with self.assertRaisesRegex(ValueError, "requires --candidate-pool"):
            validate_predictions(self.predictions_path, self.context, method="p2",
                                 required_count=3, trace_path=self.trace_path)
        with self.assertRaisesRegex(ValueError, "only for p2"):
            validate_predictions(self.pool, self.context, method="embedding",
                                 required_count=3, trace_path=self.trace_path)

    def test_full_fifty_predictions_expand_pool_and_explain_top_ten(self):
        context = ValidationContext(catalog_ids=set(range(1, 61)), histories={1: {60}},
                                    expected_user_ids={1}, split="validation")
        initial = list(range(1, 51))
        final = list(range(2, 51)) + [59]
        pool_path = self.work / "fifty_pool.jsonl"
        prediction_path = self.work / "fifty_p2.jsonl"
        trace_path = self.work / "fifty_trace.jsonl"
        write_jsonl(pool_path, [{"user_id": 1, "split": "validation", "method": "embedding",
                                "recommendations": [{"movie_id": movie_id, "rank": rank, "score": 51 - rank}
                                                    for rank, movie_id in enumerate(initial, 1)]}])
        write_jsonl(prediction_path, [{"user_id": 1, "split": "validation", "method": "p2",
                                      "recommendations": [{"movie_id": movie_id, "rank": rank,
                                                           "score": 51 - rank}
                                                          for rank, movie_id in enumerate(final, 1)]}])
        write_jsonl(trace_path, [{"schema_version": "p2-trace-v1", "user_id": 1,
                                  "split": "validation", "method": "p2",
                                  "initial_candidate_movie_ids": initial,
                                  "events": [
                                      {"action": "assess_sufficiency", "sufficient": True},
                                      {"action": "evaluate_candidates", "candidate_movie_ids": initial,
                                       "pool_sufficient": False},
                                      {"action": "retrieve_candidates", "added_movie_ids": [59],
                                       "index_version": "synthetic-v1"},
                                      {"action": "evaluate_candidates", "candidate_movie_ids": initial + [59],
                                       "pool_sufficient": True},
                                      {"action": "rank_candidates", "movie_ids": final},
                                      explanation_event(final[:10]),
                                  ],
                                  "final_candidate_movie_ids": initial + [59],
                                  "termination_reason": "recommend"}])
        result = validate_predictions(prediction_path, context, method="p2", required_count=50,
                                      candidate_pool_path=pool_path, trace_path=trace_path)
        self.assertEqual(result.report["status"], "valid")
        self.assertEqual(len(result.traces[1]["events"][-1]["explanations"]), 10)


if __name__ == "__main__":
    unittest.main()
