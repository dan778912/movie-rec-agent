"""Signed, expiring continuation state; Service B keeps no conversation database."""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

from src.agent.state import EpisodeState
from src.retrieval.online import canonical


class InvalidContinuation(ValueError):
    pass


class Continuations:
    def __init__(self, secret, resource_identity, ttl=3600):
        if not isinstance(secret, str) or len(secret.encode()) < 32:
            raise ValueError("SERVICE_B_STATE_SECRET needs at least 32 bytes for the v1 continuation API")
        self.secret, self.identity, self.ttl = secret.encode(), resource_identity, ttl

    def encode(self, state, trace_id):
        payload = canonical({"version": "service-b-continuation-v1", "trace_id": trace_id,
            "expires_at": int(time.time()) + self.ttl, "resource_identity": self.identity,
            "state": state.to_dict()})
        body = base64.urlsafe_b64encode(payload).decode().rstrip("=")
        signature = hmac.new(self.secret, body.encode(), hashlib.sha256).hexdigest()
        return body + "." + signature

    def decode(self, token):
        try:
            if len(token) > 2_000_000:
                raise ValueError("oversized continuation")
            body, signature = token.rsplit(".", 1)
            expected = hmac.new(self.secret, body.encode(), hashlib.sha256).hexdigest()
            if not hmac.compare_digest(expected, signature):
                raise ValueError("invalid signature")
            record = json.loads(base64.urlsafe_b64decode(body + "=" * (-len(body) % 4)))
            if record["version"] != "service-b-continuation-v1" or record["expires_at"] <= time.time():
                raise ValueError("expired or incompatible continuation")
            if record["resource_identity"] != self.identity:
                raise ValueError("model, index, or policy changed; start a new episode")
            state = EpisodeState.from_dict(record["state"])
            if state.phase != "await_answer":
                raise ValueError("episode is not waiting for an answer")
            return state, record["trace_id"]
        except Exception as error:
            raise InvalidContinuation(str(error)) from error
