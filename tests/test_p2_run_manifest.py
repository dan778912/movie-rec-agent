"""Synthetic checks for the Phase 2 producer run receipt."""

import contextlib
import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.evaluate import _canonical_bytes
from src.evaluation.p2_run_manifest import main, validate_run_manifest


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "evaluation_scoring"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _jsonl(path, rows):
    path.write_bytes(b"".join(_canonical_bytes(row) for row in rows))


def _trace(user_id, initial, events, final):
    return {
        "schema_version": "p2-trace-v1", "user_id": user_id,
        "split": "validation", "method": "p2",
        "initial_candidate_movie_ids": initial, "events": events,
        "final_candidate_movie_ids": final, "termination_reason": "recommend",
    }


def _finish(ranking):
    return [
        {"action": "rank_candidates", "movie_ids": ranking},
        {"action": "explain_recommendations", "explanations": [
            {"movie_id": movie_id, "text": f"Synthetic explanation for {movie_id}."}
            for movie_id in ranking
        ]},
    ]


class P2RunManifestTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.config = FIXTURES / "config.json"
        self.pool = FIXTURES / "embedding.jsonl"
        self.predictions = self.work / "p2.jsonl"
        self.trace = self.work / "trace.jsonl"
        self.manifest = self.work / "run.json"
        source = [json.loads(line) for line in self.pool.read_text(encoding="utf-8").splitlines()]
        self.prediction_rows = [{**row, "method": "p2"} for row in source]
        self.trace_rows = [
            _trace(1, [10, 30, 20], [
                {"action": "assess_sufficiency", "sufficient": False},
                {"action": "ask_clarification", "question": "Genre?", "answer": "Comedy",
                 "answer_source": "synthetic-simulator-v1"},
                {"action": "assess_sufficiency", "sufficient": True},
                {"action": "evaluate_candidates", "candidate_movie_ids": [10, 30, 20],
                 "pool_sufficient": True},
                *_finish([10, 30, 20]),
            ], [10, 30, 20]),
            _trace(2, [60, 40, 70], [
                {"action": "assess_sufficiency", "sufficient": True},
                {"action": "evaluate_candidates", "candidate_movie_ids": [60, 40, 70],
                 "pool_sufficient": False},
                {"action": "retrieve_candidates", "added_movie_ids": [50],
                 "index_version": "synthetic-index-v1"},
                {"action": "evaluate_candidates", "candidate_movie_ids": [60, 40, 70, 50],
                 "pool_sufficient": True},
                *_finish([60, 40, 70]),
            ], [60, 40, 70, 50]),
        ]
        self.save_inputs()
        sources = {
            "config": self.config,
            "dataset_manifest": FIXTURES / "manifest.json",
            "catalog": FIXTURES / "catalog.csv",
            "profiles": FIXTURES / "validation_profiles.jsonl",
            "targets": FIXTURES / "validation_targets.csv",
            "users": FIXTURES / "users.json",
            "candidate_pool": self.pool,
        }
        catalog_hash = _hash(sources["catalog"])
        self.receipt = {
            "schema_version": "p2-run-manifest-v1",
            "run_id": "synthetic-validation-v1",
            "policy": {
                "dataset_version": "synthetic-evaluation-v1", "split": "validation",
                "user_count": 2, "code_commit": "a" * 40, "code_dirty": False,
                "source_sha256": {key: _hash(path) for key, path in sources.items()},
                "model": {
                    "p1_checkpoint_id": "synthetic-p1", "p1_checkpoint_revision": "v1",
                    "p2_checkpoint_id": "synthetic-p2", "p2_checkpoint_revision": "v2",
                    "tokenizer_id": "synthetic-tokenizer", "training_sample_version": "samples-v1",
                    "prompt_version": "prompt-v1",
                    "decoding": {"temperature": 0, "top_p": 1, "max_new_tokens": 200,
                                 "seed": 0, "score_kind": "rank_surrogate"},
                },
                "retrieval": {
                    "index_version": "synthetic-index-v1", "index_manifest_sha256": "b" * 64,
                    "ingestion_catalog_sha256": catalog_hash,
                    "embedding_model_id": "synthetic-embeddings", "max_candidate_pool_size": 10,
                    "watched_exclusion": True,
                },
                "simulator": {
                    "version": "synthetic-simulator-v1", "code_sha256": "c" * 64,
                    "answer_source": "synthetic-simulator-v1",
                    "pre_cutoff_source": "training history only",
                },
                "agent": {
                    "parser_version": "parser-v1", "max_control_events": 8,
                    "max_serialized_events": 10, "timeout_ms": 1000,
                    "fallback_source": "frozen_b3_order",
                    "fallback_triggers": ["timeout", "invalid_action", "loop_cap"],
                    "explanation_policy_version": "catalog-grounding-v1",
                },
                "runtime": {
                    "hardware": "synthetic CPU", "concurrency": 1,
                    "measurement_version": "timer-v1",
                    "latency_scope": "offline_episode_excludes_simulator_wait",
                },
            },
            "policy_sha256": "",
            "output_sha256": {"predictions": _hash(self.predictions), "trace": _hash(self.trace)},
            "episodes": [
                {"user_id": 1, "termination_reason": "recommend", "fallback_reason": None,
                 "fallback_trigger": None,
                 "model_call_count": 2, "retrieval_call_count": 0,
                 "informative_answer_count": 1, "noninformative_answer_count": 0,
                 "invalid_action_count": 0, "timed_out": False, "loop_cap_hit": False,
                 "latency_ms": {"model": 40, "retrieval": 0, "end_to_end": 50},
                 "cost_usd": 0.01},
                {"user_id": 2, "termination_reason": "recommend", "fallback_reason": None,
                 "fallback_trigger": None,
                 "model_call_count": 3, "retrieval_call_count": 1,
                 "informative_answer_count": 0, "noninformative_answer_count": 0,
                 "invalid_action_count": 1, "timed_out": False, "loop_cap_hit": False,
                 "latency_ms": {"model": 60, "retrieval": 20, "end_to_end": 100},
                 "cost_usd": 0.02},
            ],
            "aggregate": {
                "user_count": 2, "fallback_user_count": 0,
                "fallback_by_trigger": {},
                "invalid_action_user_count": 1, "timeout_user_count": 0,
                "loop_cap_user_count": 0, "model_call_count": 5,
                "retrieval_call_count": 1, "informative_answer_count": 1,
                "noninformative_answer_count": 0, "invalid_action_count": 1,
                "latency_ms": {
                    "model": {"p50": 50, "p95": 59},
                    "retrieval": {"p50": 10, "p95": 19},
                    "end_to_end": {"p50": 75, "p95": 97.5},
                },
                "cost_usd": 0.03,
            },
        }
        self.save_receipt()

    def save_inputs(self):
        _jsonl(self.predictions, self.prediction_rows)
        _jsonl(self.trace, self.trace_rows)

    def save_receipt(self):
        self.receipt["policy_sha256"] = hashlib.sha256(_canonical_bytes(self.receipt["policy"])).hexdigest()
        self.manifest.write_bytes(_canonical_bytes(self.receipt))

    def validate(self):
        return validate_run_manifest(self.manifest, self.config, self.pool, self.predictions, self.trace)

    def test_valid_receipt_checks_every_user_and_hand_calculated_aggregates(self):
        report = self.validate()
        self.assertEqual((report["status"], report["user_count"]), ("valid", 2))
        self.assertEqual(report["aggregate"]["invalid_action_user_count"], 1)
        self.assertEqual(report["aggregate"]["latency_ms"]["end_to_end"], {"p50": 75.0, "p95": 97.5})
        self.assertEqual(report["source_sha256"]["targets"], _hash(FIXTURES / "validation_targets.csv"))
        self.assertEqual(report["run_manifest_sha256"], _hash(self.manifest))
        self.assertIn("measurement accuracy", report["owner_assertions_unverified"])

    def test_changed_source_and_policy_digest_are_rejected(self):
        self.predictions.write_bytes(self.predictions.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "output_sha256.predictions"):
            self.validate()
        self.save_inputs()
        self.receipt["policy"]["model"]["prompt_version"] = "changed-after-freeze"
        self.manifest.write_bytes(_canonical_bytes(self.receipt))
        with self.assertRaisesRegex(ValueError, "policy_sha256"):
            self.validate()

    def test_missing_and_duplicate_episode_are_rejected(self):
        self.receipt["episodes"].pop()
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "episode user coverage mismatch"):
            self.validate()
        self.receipt["episodes"].append(copy.deepcopy(self.receipt["episodes"][0]))
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "duplicate episode"):
            self.validate()

    def test_aggregate_and_trace_policy_mismatches_are_rejected(self):
        self.receipt["aggregate"]["fallback_user_count"] = 1
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "aggregate.fallback_user_count"):
            self.validate()
        self.receipt["aggregate"]["fallback_user_count"] = 0
        self.receipt["policy"]["retrieval"]["index_version"] = "wrong-index"
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "index_version disagrees"):
            self.validate()
        self.receipt["policy"]["retrieval"]["index_version"] = "synthetic-index-v1"
        self.receipt["policy"]["simulator"]["answer_source"] = "wrong-simulator"
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "answer_source disagrees"):
            self.validate()

    def test_fallback_trigger_and_counts_match_trace(self):
        trace = self.trace_rows[0]
        trace["events"] = _finish([10, 30, 20])
        trace["termination_reason"] = "fallback"
        trace["fallback_reason"] = "synthetic timeout"
        trace["fallback_source"] = "frozen_b3_order"
        self.save_inputs()
        self.receipt["output_sha256"] = {"predictions": _hash(self.predictions), "trace": _hash(self.trace)}
        episode = self.receipt["episodes"][0]
        episode.update(termination_reason="fallback", fallback_reason="synthetic timeout",
                       fallback_trigger="timeout", model_call_count=0,
                       informative_answer_count=0, timed_out=True)
        aggregate = self.receipt["aggregate"]
        aggregate.update(fallback_user_count=1, fallback_by_trigger={"timeout": 1},
                         timeout_user_count=1, model_call_count=3, informative_answer_count=0)
        self.save_receipt()
        self.assertEqual(self.validate()["aggregate"]["fallback_by_trigger"], {"timeout": 1})
        episode["fallback_trigger"] = "undeclared"
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "fallback_trigger is absent"):
            self.validate()

    def test_invalid_timing_count_and_json_are_rejected(self):
        self.receipt["episodes"][0]["latency_ms"]["end_to_end"] = 10
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "stage latency exceeds"):
            self.validate()
        self.receipt["episodes"][0]["latency_ms"]["end_to_end"] = 50
        self.receipt["episodes"][0]["model_call_count"] = True
        self.save_receipt()
        with self.assertRaisesRegex(ValueError, "excluding booleans"):
            self.validate()
        self.manifest.write_text('{"schema_version":"p2-run-manifest-v1","schema_version":"again"}', encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "duplicate JSON key"):
            self.validate()

    def test_cli_returns_valid_report(self):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(["--manifest", str(self.manifest), "--config", str(self.config),
                         "--candidate-pool", str(self.pool), "--predictions", str(self.predictions),
                         "--trace", str(self.trace)])
        self.assertEqual(code, 0)
        self.assertEqual(stderr.getvalue(), "")
        self.assertEqual(json.loads(stdout.getvalue())["status"], "valid")


if __name__ == "__main__":
    unittest.main()
