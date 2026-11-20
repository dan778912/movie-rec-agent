"""Exercise Member 5's unchanged HTTP client against the real Service B ASGI app."""
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from tests.fixtures.service_a_client_v1 import HttpServiceBClient
from tests import test_service_b as service_fixture
from tests.service_b_helpers import ScriptedModel
from src.app.service_b.main import create_app


class ServiceACompatibilityTest(unittest.TestCase):
    def test_original_member5_client_can_ask_then_recommend(self):
        resources = service_fixture.ServiceBTest().resources(ScriptedModel(ask=True))
        with TestClient(create_app(resources)) as transport:
            def send(url, **kwargs):
                kwargs.pop("timeout", None)  # ASGI transport has no socket deadline.
                return transport.post(url, **kwargs)
            with patch('httpx.post', side_effect=send):
                client = HttpServiceBClient(base_url="http://testserver")
                request = {"profile_text": "Comedy", "history_movie_ids": [1, 2], "clarifications": []}
                first = client.decide(**request)
                self.assertEqual(first["action"], "ask")
                request["clarifications"] = [{"question": first["question"], "answer": "Comedy"}]
                second = client.decide(**request)
                self.assertEqual(second["action"], "recommend")
                self.assertEqual(len(second["items"]), 10)
                self.assertFalse({i["movie_id"] for i in second["items"]} & {1, 2})
