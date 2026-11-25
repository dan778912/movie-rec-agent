"""First-validation explanation policy, exercised through the actual offline producer."""
import hashlib
import json
import re
import tempfile
import unittest
from pathlib import Path

from src.agent.config import AgentConfig
from src.app.service_b.explanations import summarize_explanations
from src.app.service_b.offline import run_episode
from src.app.service_b.runtime import Resources
from src.evaluation.validate import ValidationContext, validate_predictions
from src.retrieval.online import canonical
from tests.service_b_helpers import ScriptedModel, ScriptedRetriever, catalog

MALFORMED = '{"action":"explain_recommendations","explanations":['
TIMEOUT_MESSAGE = "synthetic explanation deadline: " + "x" * 250


class ExplanationModel(ScriptedModel):
    def __init__(self, failure="malformed", **kwargs):
        super().__init__(**kwargs)
        self.failure = failure

    def generate(self, prompt):
        if 'Output format: {"action": "explain_recommendations"' not in prompt:
            return super().generate(prompt)
        if self.failure == "timeout":
            raise TimeoutError(TIMEOUT_MESSAGE)
        if self.failure == "malformed":
            return MALFORMED
        ids = [int(i) for i in re.findall(r"^\[(\d+)\]", prompt, re.M)]
        if self.failure == "wrong_ids":
            ids = [99999] + ids[1:]
        text = 'It is tagged "quantum unicorn".' if self.failure == "unsupported" else "It is listed under Comedy."
        return json.dumps({"action": "explain_recommendations", "explanations": [
            {"movie_id": i, "text": text} for i in ids]})


class ExplanationFallbackTest(unittest.TestCase):
    def episode(self, model):
        resources = Resources(model, ScriptedRetriever(), catalog(), AgentConfig(
            label="synthetic-template-policy", explanation_failure_policy="template", grounding_policy="flag"),
            {"request_timeout_seconds": 10})
        return run_episode(resources, {"user_id": 1, "text": "Comedy", "history_movie_ids": [1]},
                           list(range(2, 52)), {}, "validation")

    def validate(self, prediction, trace):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            pool = {"user_id": 1, "split": "validation", "method": "embedding", "recommendations": [
                {"movie_id": i, "rank": n, "score": 1.0} for n, i in enumerate(range(2, 52), 1)]}
            for name, row in (("p2.jsonl", prediction), ("trace.jsonl", trace), ("b3.jsonl", pool)):
                (root / name).write_bytes(canonical(row))
            result = validate_predictions(root / "p2.jsonl", ValidationContext(set(catalog()), {1: {1}}, {1}, "validation"),
                method="p2", required_count=50, candidate_pool_path=root / "b3.jsonl", trace_path=root / "trace.jsonl")
            self.assertEqual(result.report["error_count"], 0)
            self.assertEqual(result.traces[1]["events"][-1]["source"], trace["events"][-1]["source"])

    def test_bad_explanation_preserves_expanded_pool_ranking_and_valid_trace(self):
        for failure in ("malformed", "wrong_ids", "timeout"):
            with self.subTest(failure=failure):
                pred, trace, measurement, audit = self.episode(ExplanationModel(failure, widen=True))
                ranking = [r["movie_id"] for r in pred["recommendations"]]
                self.assertEqual(ranking, list(reversed(trace["final_candidate_movie_ids"]))[:50])
                self.assertNotEqual(ranking, list(range(2, 52)))
                self.assertEqual(trace["termination_reason"], "recommend")
                event = trace["events"][-1]
                self.assertEqual(event["source"], "catalog_metadata_fallback")
                self.assertEqual([e["movie_id"] for e in event["explanations"]], ranking[:10])
                self.assertTrue(all(e["text"] == "It is listed under Comedy." for e in event["explanations"]))
                self.assertEqual(measurement["invalid_action_count"], int(failure != "timeout"))
                self.assertEqual(measurement["timed_out"], failure == "timeout")
                self.assertEqual(measurement["model_call_count"], len(audit["model_call_audit"]))
                last_call = audit["model_call_audit"][-1]
                if failure == "timeout":
                    self.assertEqual(last_call["error_message"], TIMEOUT_MESSAGE)
                    self.assertEqual(last_call["error_type"], "TimeoutError")
                elif failure == "malformed":
                    self.assertEqual(last_call["raw_sha256"], hashlib.sha256(MALFORMED.encode()).hexdigest())
                summary = summarize_explanations([trace], {1: audit})
                self.assertEqual(summary["catalog_metadata_fallback_user_count"], 1)
                self.assertEqual(summary["ranking_fallback_user_count"], 0)
                self.validate(pred, trace)

    def test_ranking_and_explanation_fallbacks_remain_distinct(self):
        pred, trace, measurement, audit = self.episode(ExplanationModel(invalid_once=True))
        self.assertEqual(trace["termination_reason"], "fallback")
        self.assertEqual([r["movie_id"] for r in pred["recommendations"]], list(range(2, 52)))
        self.assertEqual(trace["events"][-1]["source"], "catalog_metadata_fallback")
        self.assertNotEqual(trace["fallback_reason"], trace["events"][-1]["explanation_fallback_reason"])
        self.assertEqual(measurement["invalid_action_count"], 2)
        summary = summarize_explanations([trace], {1: audit})
        self.assertEqual(summary["ranking_fallback_user_count"], 1)
        self.assertEqual(summary["catalog_metadata_fallback_user_count"], 1)
        self.validate(pred, trace)

    def test_parseable_unsupported_text_is_kept_for_audit(self):
        pred, trace, measurement, audit = self.episode(ExplanationModel("unsupported"))
        event = trace["events"][-1]
        self.assertEqual(event["source"], "model")
        self.assertTrue(all(e["text"] == 'It is tagged "quantum unicorn".' for e in event["explanations"]))
        self.assertTrue(any(r["status"] == "unsupported" for g in audit["telemetry"]["grounding"] for r in g["results"]))
        self.assertEqual(measurement["invalid_action_count"], 0)
        self.assertEqual(summarize_explanations([trace], {1: audit})["catalog_metadata_fallback_user_count"], 0)

    def test_summary_rejects_a_substitution_without_failed_attempt_evidence(self):
        pred, trace, measurement, audit = self.episode(ExplanationModel())
        audit["telemetry"]["decisions"] = []
        with self.assertRaisesRegex(ValueError, "failed-attempt evidence"):
            summarize_explanations([trace], {1: audit})


if __name__ == "__main__":
    unittest.main()
