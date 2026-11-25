import json
import hashlib
import shutil
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path

from src.agent.config import AgentConfig
from src.app.service_b.offline import EpisodeFailed, run, run_episode, source_snapshot
from src.app.service_b.runtime import Resources
from src.evaluation.validate import ValidationContext, validate_predictions
from src.retrieval.online import canonical, file_hash
from tests.service_b_helpers import ScriptedModel, ScriptedRetriever, catalog


class OfflineProducerTest(unittest.TestCase):
    def test_full_export_and_receipt_validate_and_require_matching_approval(self):
        self.export_fixture()

    def test_template_explanations_export_valid_complete_manifest_and_counted_summary(self):
        from tests.test_explanation_fallback import ExplanationModel
        root = self.export_fixture(model=ExplanationModel(), template=True)
        summary_path = root / "export" / "explanation_fallback_summary.json"
        summary = json.loads(summary_path.read_text())
        self.assertEqual(summary["completed_users"], 2)
        self.assertEqual(summary["catalog_metadata_fallback_user_count"], 2)
        self.assertEqual(summary["ranking_fallback_user_count"], 0)
        manifest = json.loads((root / "export" / "run_manifest.json").read_text())
        self.assertIn(file_hash(summary_path), manifest["owner_notes"])
        self.assertIn(file_hash(root / "export" / "telemetry.jsonl"), manifest["owner_notes"])
        self.assertEqual(manifest["aggregate"]["invalid_action_count"], 2)

    def export_fixture(self, model=None, template=False):
        # Reuse Member 2's synthetic input package and independent receipt validator.
        from tests import test_p2_run_manifest as fixtures
        helper = fixtures.P2RunManifestTest()
        helper.setUp()
        self.addCleanup(helper.doCleanups)
        receipt = helper.receipt
        root = helper.work
        data = root / "data"
        shutil.copytree(fixtures.FIXTURES, data)
        profiles_path = data / "validation_profiles.jsonl"
        profiles_path.write_bytes(b"".join(canonical({**json.loads(line), "text": "Comedy"})
                                           for line in profiles_path.read_text().splitlines()))
        manifest_path = data / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["file_sha256"]["validation_profiles.jsonl"] = file_hash(profiles_path)
        manifest_path.write_bytes(canonical(manifest))
        users_path = data / "users.json"
        users = json.loads(users_path.read_text())
        users["source_sha256"].update(profiles=file_hash(profiles_path), manifest=file_hash(manifest_path))
        users_path.write_bytes(canonical(users))
        config = json.loads(helper.config.read_text())
        config.update(data_dir=str(data), users_file=str(users_path), users_sha256=file_hash(users_path),
                      manifest_sha256=file_hash(manifest_path))
        config_path = root / "config.json"
        config_path.write_bytes(canonical(config))
        index = root / "index.json"
        index.write_text('{}')
        service = root / "service.json"
        service.write_text('{}')
        policy = receipt["policy"]
        policy["source_sha256"].update(config=file_hash(config_path), dataset_manifest=file_hash(manifest_path),
            profiles=file_hash(profiles_path), users=file_hash(users_path))
        policy["retrieval"]["index_manifest_sha256"] = file_hash(index)
        settings = AgentConfig(label="synthetic-only", ranking_size=3, max_pool_size=10,
                               explanation_failure_policy="template" if template else "fail")
        from src.agent.manifest import agent_policy
        policy["agent"] = agent_policy(settings, timeout_ms=10000)
        model = model or ScriptedModel()
        model.identity = {"files": {"synthetic": "a" * 64}}
        resources = Resources(model, ScriptedRetriever(), catalog(), settings,
                              {"request_timeout_seconds": 10, "index": {"manifest": str(index)}})
        resources.retriever.index_version = "synthetic-index-v1"
        # The synthetic catalog uses a sparse movie ID set.
        bundle = {"policy": policy, "policy_sha256": hashlib.sha256(canonical(policy)).hexdigest(),
            "runtime_config_sha256": file_hash(service), "source_snapshot": source_snapshot(),
            "agent_config": settings.to_dict(), "checkpoint_files": model.identity["files"]}
        freeze = root / "freeze.json"
        freeze.write_bytes(canonical(bundle))
        args = SimpleNamespace(config=str(config_path), candidate_pool=str(helper.pool), service_config=str(service),
            freeze=str(freeze), approval_record=None, smoke=0, hourly_rate=1, output_dir=str(root / "export"), run_id="synthetic-producer-v1")
        with self.assertRaisesRegex(ValueError, "approval-record"):
            run(args)
        approval = root / "approval.json"
        approval.write_bytes(canonical({"approved": True, "approved_by": "synthetic test", "approved_at": "2026-11-20",
                                        "split": "validation", "freeze_sha256": file_hash(freeze)}))
        args.approval_record = str(approval)
        with patch('src.app.service_b.offline.load_resources', return_value=resources), patch(
                'src.app.service_b.offline.histories_for_simulator', return_value={1: {}, 2: {}}):
            result = run(args)
        self.assertEqual(result["status"], "validated")
        self.assertTrue((root / "export" / "run_manifest.json").exists())
        return root

    def test_complete_clarification_expansion_episode_passes_authoritative_validator(self):
        resources = Resources(ScriptedModel(ask=True, widen=True), ScriptedRetriever(), catalog(),
            AgentConfig(label="synthetic-only"), {"request_timeout_seconds": 10})
        profile = {"user_id": 1, "text": "I like Comedy", "history_movie_ids": [1]}
        facts = {"favorite_genre": [{"kind": "favorite_genres", "genres": ["Comedy"],
                                      "source_movie_ids": [1], "source_max_timestamp": 1}]}
        initial = list(range(2, 52))
        pred, trace, measurement, audit = run_episode(resources, profile, initial, facts, "validation", hourly_rate=1)
        self.assertEqual(len(pred["recommendations"]), 50)
        self.assertEqual(len(trace["final_candidate_movie_ids"]), 79)
        self.assertEqual(trace["initial_candidate_movie_ids"], initial)
        self.assertEqual(measurement["informative_answer_count"], 1)
        self.assertEqual(measurement["retrieval_call_count"], 1)
        self.assertGreater(measurement["latency_ms"]["end_to_end"], measurement["latency_ms"]["model"])
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "pred.jsonl").write_bytes(canonical(pred))
            (root / "trace.jsonl").write_bytes(canonical(trace))
            (root / "b3.jsonl").write_bytes(canonical({"user_id": 1, "split": "validation", "method": "embedding",
                "recommendations": [{"movie_id": mid, "score": 1.0, "rank": i} for i, mid in enumerate(initial, 1)]}))
            validated = validate_predictions(root / "pred.jsonl", ValidationContext(set(catalog()), {1: {1}}, {1}, "validation"),
                method="p2", required_count=50, candidate_pool_path=root / "b3.jsonl", trace_path=root / "trace.jsonl")
            self.assertEqual(validated.report["error_count"], 0)

    def test_invalid_attempt_is_counted_and_fallback_preserves_frozen_order(self):
        resources = Resources(ScriptedModel(invalid_once=True), ScriptedRetriever(), catalog(), AgentConfig(),
                              {"request_timeout_seconds": 10})
        initial = list(range(2, 52))
        pred, trace, measurement, audit = run_episode(resources, {"user_id": 1, "text": "Comedy", "history_movie_ids": [1]},
                                                     initial, {}, "validation")
        self.assertEqual(trace["termination_reason"], "fallback")
        self.assertEqual([r["movie_id"] for r in pred["recommendations"]], initial)
        self.assertEqual(measurement["invalid_action_count"], 1)

    def test_explanation_failure_is_retained_as_failure_not_fabricated_trace(self):
        class Broken(ScriptedModel):
            def generate(self, prompt):
                if 'Output format: {"action": "explain_recommendations"' in prompt:
                    return "bad JSON"
                return super().generate(prompt)
        resources = Resources(Broken(), ScriptedRetriever(), catalog(), AgentConfig(), {"request_timeout_seconds": 10})
        with self.assertRaises(EpisodeFailed) as failure:
            run_episode(resources, {"user_id": 1, "text": "Comedy", "history_movie_ids": []}, list(range(1, 51)), {}, "validation")
        self.assertEqual(failure.exception.evidence["telemetry"]["termination_reason"], "failed")


if __name__ == "__main__":
    unittest.main()
