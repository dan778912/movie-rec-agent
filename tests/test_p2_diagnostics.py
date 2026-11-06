"""Hand-calculated P2 pool diagnostics on the synthetic scoring fixture."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.p2_diagnostics import analyze_p2, main, write_diagnostics
from src.evaluation.validate import PredictionValidationError


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evaluation_scoring"


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")


class P2DiagnosticsTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.config = FIXTURES / "config.json"
        self.pool = FIXTURES / "embedding.jsonl"
        self.predictions = self.work / "p2.jsonl"
        self.trace = self.work / "trace.jsonl"
        original = [json.loads(line) for line in self.pool.read_text(encoding="utf-8").splitlines()]
        self.prediction_rows = [{**record, "method": "p2"} for record in original]
        # User 1's frozen pool contains both positives (10, 20). User 2's pool
        # contains 40 but misses 50. P2 adds 50, then omits it from the final
        # three ranks, so expanded-pool recall exceeds final-ranking recall.
        self.trace_rows = [
            {
                "schema_version": "p2-trace-v1", "user_id": 1, "split": "validation", "method": "p2",
                "initial_candidate_movie_ids": [10, 30, 20],
                "events": [
                    {"action": "assess_sufficiency", "sufficient": False},
                    {"action": "ask_clarification", "question": "A genre?", "answer": "Comedy",
                     "answer_source": "synthetic_pre_cutoff_history"},
                    {"action": "assess_sufficiency", "sufficient": True},
                    {"action": "evaluate_candidates", "candidate_movie_ids": [10, 30, 20],
                     "pool_sufficient": True},
                    {"action": "rank_candidates", "movie_ids": [10, 30, 20]},
                    {"action": "explain_recommendations", "explanations": [
                        {"movie_id": 10, "text": "Catalog movie 10."},
                        {"movie_id": 30, "text": "Catalog movie 30."},
                        {"movie_id": 20, "text": "Catalog movie 20."},
                    ]},
                ],
                "final_candidate_movie_ids": [10, 30, 20], "termination_reason": "recommend",
            },
            {
                "schema_version": "p2-trace-v1", "user_id": 2, "split": "validation", "method": "p2",
                "initial_candidate_movie_ids": [60, 40, 70],
                "events": [
                    {"action": "assess_sufficiency", "sufficient": True},
                    {"action": "evaluate_candidates", "candidate_movie_ids": [60, 40, 70],
                     "pool_sufficient": False},
                    {"action": "retrieve_candidates", "added_movie_ids": [50],
                     "index_version": "synthetic-index-v1"},
                    {"action": "evaluate_candidates", "candidate_movie_ids": [60, 40, 70, 50],
                     "pool_sufficient": True},
                    {"action": "rank_candidates", "movie_ids": [60, 40, 70]},
                    {"action": "explain_recommendations", "explanations": [
                        {"movie_id": 60, "text": "Catalog movie 60."},
                        {"movie_id": 40, "text": "Catalog movie 40."},
                        {"movie_id": 70, "text": "Catalog movie 70."},
                    ]},
                ],
                "final_candidate_movie_ids": [60, 40, 70, 50], "termination_reason": "recommend",
            },
        ]
        self.save()

    def save(self):
        _write_jsonl(self.predictions, self.prediction_rows)
        _write_jsonl(self.trace, self.trace_rows)

    def analyze(self):
        return analyze_p2(self.config, self.pool, self.predictions, self.trace)

    def test_hand_calculated_pool_and_action_diagnostics(self):
        result = self.analyze()
        one, two = result.per_user
        self.assertEqual([row["user_id"] for row in result.per_user], [1, 2])
        self.assertEqual((one["initial_pool_hits"], one["expanded_pool_hits"],
                          one["final_ranking_hits"]), (2, 2, 2))
        self.assertEqual((two["initial_pool_hits"], two["expanded_pool_hits"],
                          two["final_ranking_hits"]), (1, 2, 1))
        self.assertEqual((two["initial_pool_recall"], two["expanded_pool_recall"],
                          two["final_ranking_recall"]), (0.5, 1.0, 0.5))
        self.assertIsNone(one["added_candidate_yield"])
        self.assertEqual(two["added_candidate_yield"], 1.0)
        aggregate = result.summary["aggregate"]
        self.assertEqual(aggregate["initial_pool_recall_macro"], 0.75)
        self.assertEqual(aggregate["expanded_pool_recall_macro"], 1.0)
        self.assertEqual(aggregate["final_ranking_recall_macro"], 0.75)
        self.assertEqual(aggregate["expanded_minus_initial_pool_recall_macro"], 0.25)
        self.assertEqual((aggregate["added_candidate_count"], aggregate["added_relevant_count"],
                          aggregate["added_candidate_yield"]), (1, 1, 1.0))
        self.assertEqual((aggregate["clarification_user_rate"], aggregate["retrieval_user_rate"],
                          aggregate["fallback_user_rate"]), (0.5, 0.5, 0.0))
        self.assertEqual(aggregate["final_pool_size"]["p50"], 3.5)
        self.assertAlmostEqual(aggregate["final_pool_size"]["p95"], 3.95)
        self.assertEqual(result.summary["source_sha256"]["candidate_pool"],
                         hashlib.sha256(self.pool.read_bytes()).hexdigest())

    def test_fallback_stays_in_denominator_and_no_added_yield_is_null(self):
        fallback = copy.deepcopy(self.trace_rows[0])
        fallback["events"] = fallback["events"][-2:]
        fallback.update(termination_reason="fallback", fallback_reason="synthetic timeout",
                        fallback_source="frozen_b3_order")
        self.trace_rows[0] = fallback
        direct = copy.deepcopy(self.trace_rows[1])
        direct["events"] = [direct["events"][0],
                            {"action": "evaluate_candidates", "candidate_movie_ids": [60, 40, 70],
                             "pool_sufficient": True}, *direct["events"][-2:]]
        direct["final_candidate_movie_ids"] = [60, 40, 70]
        self.trace_rows[1] = direct
        self.save()
        result = self.analyze()
        aggregate = result.summary["aggregate"]
        self.assertEqual(result.summary["user_count"], 2)
        self.assertEqual(aggregate["fallback_user_count"], 1)
        self.assertEqual(aggregate["fallback_user_rate"], 0.5)
        self.assertEqual(aggregate["users_with_retrieval"], 0)
        self.assertEqual(aggregate["added_candidate_count"], 0)
        self.assertIsNone(aggregate["added_candidate_yield"])
        self.assertEqual(aggregate["initial_pool_recall_macro"], aggregate["expanded_pool_recall_macro"])

    def test_invalid_trace_cannot_produce_diagnostics(self):
        self.trace_rows[1]["initial_candidate_movie_ids"] = [40, 60, 70]
        self.save()
        with self.assertRaises(PredictionValidationError):
            self.analyze()
        output = self.work / "invalid-output"
        stdout, stderr = io.StringIO(), io.StringIO()
        args = ["--config", str(self.config), "--candidate-pool", str(self.pool),
                "--predictions", str(self.predictions), "--trace", str(self.trace),
                "--output-dir", str(output)]
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(args)
        self.assertEqual(code, 1)
        self.assertEqual(json.loads(stdout.getvalue())["status"], "invalid")
        self.assertFalse(output.exists())

    def test_missing_trace_user_is_not_silently_dropped(self):
        self.trace_rows.pop()
        self.save()
        with self.assertRaises(PredictionValidationError):
            self.analyze()

    def test_cli_writes_two_immutable_hashed_outputs(self):
        output = self.work / "diagnostics"
        args = ["--config", str(self.config), "--candidate-pool", str(self.pool),
                "--predictions", str(self.predictions), "--trace", str(self.trace),
                "--output-dir", str(output)]
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            self.assertEqual(main(args), 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual({item.name for item in output.iterdir()}, {"summary.json", "per_user.csv"})
        self.assertEqual(json.loads(stdout.getvalue()), json.loads((output / "summary.json").read_text()))
        before = {item.name: item.read_bytes() for item in output.iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertEqual(before, {item.name: item.read_bytes() for item in output.iterdir()})
        (output / "per_user.csv").write_text("changed", encoding="utf-8")
        with self.assertRaises(ValueError):
            write_diagnostics(output, self.analyze())

    def test_processed_data_directory_is_protected(self):
        with self.assertRaisesRegex(ValueError, "outside the processed data"):
            write_diagnostics(FIXTURES / "forbidden-output", self.analyze())


if __name__ == "__main__":
    unittest.main()
