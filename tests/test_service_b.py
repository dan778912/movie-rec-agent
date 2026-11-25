import concurrent.futures
import json
import threading
import time
import unittest
from dataclasses import replace

from fastapi.testclient import TestClient
from src.agent.config import AgentConfig
from src.app.service_b.main import create_app
from src.app.service_b.runtime import Resources
from tests.service_b_helpers import ScriptedModel, ScriptedRetriever, catalog


class ServiceBTest(unittest.TestCase):
    def resources(self, model=None, timeout=2):
        return Resources(model or ScriptedModel(), ScriptedRetriever(), catalog(),
            AgentConfig(label="synthetic-only"), {"request_timeout_seconds": timeout}, "test")

    def test_service_a_payload_question_replay_and_exclusion(self):
        with TestClient(create_app(self.resources(ScriptedModel(ask=True)))) as client:
            request = {"profile_text": "I like Comedy", "history_movie_ids": [1, 2, 99999], "clarifications": []}
            first = client.post("/decide", json=request)
            self.assertEqual(first.status_code, 200)
            self.assertEqual(first.json()["action"], "ask")
            request["clarifications"] = [{"question": first.json()["question"], "answer": "Comedy"}]
            result = client.post("/decide", json=request)
            self.assertEqual(result.status_code, 200, result.text)
            items = result.json()["items"]
            self.assertEqual(len(items), 10)
            self.assertFalse(set(i["movie_id"] for i in items) & {1, 2, 99999})
            self.assertTrue(result.headers["X-Request-ID"])
            # No conversation memory leaks into a separate request.
            self.assertEqual(client.post("/decide", json={**request, "clarifications": []}).json()["action"], "ask")

    def test_invalid_model_attempt_uses_original_order_fallback(self):
        with TestClient(create_app(self.resources(ScriptedModel(invalid_once=True)))) as client:
            result = client.post("/decide", json={"profile_text": "Comedy", "history_movie_ids": []})
            self.assertEqual(result.status_code, 200, result.text)
            self.assertEqual([i["movie_id"] for i in result.json()["items"]], list(range(1, 11)))

    def test_v1_signed_resume_preserves_full_episode_and_rejects_tampering(self):
        r = self.resources(ScriptedModel(ask=True, widen=True))
        with TestClient(create_app(r, state_secret="test-only-secret-" * 3)) as client:
            payload = {"profile_text": "Comedy", "history_movie_ids": [1], "clarifications": []}
            first = client.post("/v1/decide", json=payload)
            self.assertEqual(first.status_code, 200, first.text)
            token = first.json()["continuation"]
            bad = client.post("/v1/resume", json={"continuation": token + "0", "answer": "Comedy"})
            self.assertEqual(bad.status_code, 422)
            second = client.post("/v1/resume", json={"continuation": token, "answer": "Comedy"})
            self.assertEqual(second.status_code, 200, second.text)
            trace = second.json()["decision_trace"]
            self.assertEqual(second.json()["trace_id"], first.json()["trace_id"])
            self.assertEqual(trace["initial_candidate_movie_ids"], first.json()["decision_trace"]["initial_candidate_movie_ids"])
            self.assertEqual(sum(e["action"] == "ask_clarification" for e in trace["events"]), 1)
            self.assertTrue(any(e["action"] == "retrieve_candidates" for e in trace["events"]))
            self.assertEqual(trace["events"][-1]["action"], "explain_recommendations")
            self.assertIsNone(second.json()["continuation"])
            r.retriever.index_version = "changed-index"
            self.assertEqual(client.post("/v1/resume", json={"continuation": token, "answer": "Comedy"}).status_code, 422)

    def test_v1_requires_secret_and_rejects_expired_state(self):
        r = self.resources(ScriptedModel(ask=True))
        request = {"profile_text": "Comedy", "history_movie_ids": []}
        with TestClient(create_app(r)) as client:
            self.assertEqual(client.post("/v1/decide", json=request).status_code, 503)
            self.assertEqual(client.post("/decide", json=request).status_code, 200)
        with TestClient(create_app(r, state_secret="test-only-secret-" * 3)) as client:
            first = client.post("/v1/decide", json=request).json()
            from unittest.mock import patch
            with patch('src.app.service_b.continuation.time.time', return_value=time.time() + 7200):
                self.assertEqual(client.post("/v1/resume", json={"continuation": first["continuation"], "answer": "Comedy"}).status_code, 422)

    def test_v1_failure_returns_trace_and_correlation_id(self):
        class BrokenExplanation(ScriptedModel):
            def generate(self, prompt):
                if 'Output format: {"action": "explain_recommendations"' in prompt:
                    return "bad JSON"
                return super().generate(prompt)
        with TestClient(create_app(self.resources(BrokenExplanation()), state_secret="test-only-secret-" * 3)) as client:
            response = client.post("/v1/decide", json={"profile_text": "Comedy", "history_movie_ids": []})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"]["decision_trace"]["termination_reason"], "failed")
            self.assertTrue(response.headers["X-Request-ID"])

    def test_enabled_template_policy_serves_valid_rank_and_provenance(self):
        from tests.test_explanation_fallback import ExplanationModel, MALFORMED
        import hashlib
        r = self.resources(ExplanationModel(widen=True))
        r.agent_config = replace(r.agent_config, explanation_failure_policy="template")
        with TestClient(create_app(r, state_secret="test-only-secret-" * 3)) as client:
            payload = {"profile_text": "Comedy", "history_movie_ids": [1]}
            legacy = client.post("/decide", json=payload)
            self.assertEqual(legacy.status_code, 200, legacy.text)
            self.assertEqual([i["movie_id"] for i in legacy.json()["items"]], list(range(80, 70, -1)))
            versioned = client.post("/v1/decide", json=payload)
            self.assertEqual(versioned.status_code, 200, versioned.text)
            result = versioned.json()
            self.assertEqual(result["decision_trace"]["termination_reason"], "recommend")
            self.assertEqual(result["decision_trace"]["events"][-1]["source"], "catalog_metadata_fallback")
            self.assertEqual(result["model_call_audit"][-1]["raw_sha256"], hashlib.sha256(MALFORMED.encode()).hexdigest())

    def test_unavailable_model_recovers_for_following_requests(self):
        class Recoverable(ScriptedModel):
            def generate(self, prompt):
                if not hasattr(self, "failed_once"):
                    self.failed_once = True
                    self.ready = False
                    raise TimeoutError("synthetic failure")
                if not self.ready:
                    raise RuntimeError("worker stopped")
                return super().generate(prompt)
            def recover(self):
                self.ready = True
                return True
        r = self.resources(Recoverable())
        with TestClient(create_app(r)) as client:
            payload = {"profile_text": "Comedy", "history_movie_ids": []}
            self.assertEqual(client.post("/decide", json=payload).status_code, 503)
            for _ in range(50):
                if client.get("/health/ready").status_code == 200:
                    break
                time.sleep(.01)
            response = client.post("/decide", json=payload)
            self.assertEqual(response.status_code, 200, response.text)

    def test_strict_payload_validation(self):
        with TestClient(create_app(self.resources())) as client:
            for ids in ([True], ["1"], [-1], [1, 1]):
                result = client.post("/decide", json={"profile_text": "Comedy", "history_movie_ids": ids})
                self.assertEqual(result.status_code, 422)

    def test_readiness_exposes_mode_and_worker_failure(self):
        r = self.resources()
        with TestClient(create_app(r)) as client:
            self.assertEqual(client.get("/health/live").status_code, 200)
            self.assertEqual(client.get("/health/ready").json()["mode"], "test")
            r.model.ready = False
            self.assertEqual(client.get("/health/ready").status_code, 503)
            self.assertEqual(client.post("/decide", json={"profile_text": "x", "history_movie_ids": []}).status_code, 503)

    def test_readiness_detects_index_outage(self):
        r = self.resources()
        r.retriever.check_health = lambda: False
        with TestClient(create_app(r)) as client:
            response = client.get("/health/ready")
            self.assertEqual(response.status_code, 503)
            self.assertFalse(response.json()["index_ready"])

    def test_initial_retrieval_failure_is_explicit(self):
        r = self.resources()
        def broken(**kwargs):
            raise RuntimeError("unavailable")
        r.retriever.initial_pool = broken
        with TestClient(create_app(r)) as client:
            response = client.post("/decide", json={"profile_text": "x", "history_movie_ids": []})
            self.assertEqual(response.status_code, 503)
            self.assertEqual(response.json()["detail"]["code"], "initial_retrieval_failed")

    def test_timeout_keeps_admission_until_worker_finishes(self):
        class SlowModel(ScriptedModel):
            def generate(self, prompt):
                time.sleep(.12)
                return super().generate(prompt)
        with TestClient(create_app(self.resources(SlowModel(), timeout=.02))) as client:
            payload = {"profile_text": "Comedy", "history_movie_ids": []}
            self.assertEqual(client.post("/decide", json=payload).status_code, 504)
            self.assertEqual(client.post("/decide", json=payload).status_code, 429)
            time.sleep(.6)


if __name__ == "__main__":
    unittest.main()
