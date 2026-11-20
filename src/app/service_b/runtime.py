"""Shared live/offline resources and request-local instrumentation."""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass
import hashlib
from pathlib import Path

from src.agent.config import AgentConfig
from src.profiles.agent_text import PROMPT_VERSION
from src.retrieval.online import OnlineRetriever, SentenceEncoder, canonical, chroma_client
from .model import ProcessModel


class Timed:
    def __init__(self, target):
        self.target, self.seconds, self.calls, self.timed_out = target, 0.0, 0, False

    def __getattr__(self, name):
        member = getattr(self.target, name)
        if not callable(member):
            return member
        def call(*args, **kwargs):
            start = time.perf_counter()
            self.calls += 1
            try:
                return member(*args, **kwargs)
            except TimeoutError:
                self.timed_out = True
                raise
            finally:
                self.seconds += time.perf_counter() - start
        return call


@dataclass
class Resources:
    model: object
    retriever: object
    catalog: dict
    agent_config: AgentConfig
    config: dict
    mode: str = "real"

    @property
    def ready(self):
        return bool(getattr(self.model, "ready", True))

    @property
    def identity(self):
        return hashlib.sha256(canonical({"model": getattr(self.model, "identity", {"mode": self.mode}),
            "model_settings": getattr(self.model, "settings", {}), "prompt_version": PROMPT_VERSION,
            "index": self.retriever.index_version, "agent": self.agent_config.to_dict()})).hexdigest()

    def close(self):
        if hasattr(self.model, "close"):
            self.model.close()


def load_resources(config_path):
    config = json.loads(Path(config_path).read_text())
    timeout = config.get("request_timeout_seconds")
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("request_timeout_seconds must be finite and positive")
    if config["mode"] == "mock":
        from src.agent.example import CATALOG, DemoModel, DemoRetriever
        return Resources(DemoModel(), DemoRetriever(), CATALOG, AgentConfig(**config.get("agent", {})), config, "mock")
    if config["mode"] != "real":
        raise ValueError("mode must be real or mock")
    index = config["index"]
    manifest = json.loads(Path(index["manifest"]).read_text())
    encoder = SentenceEncoder(manifest["embedding"], device=index.get("device", "cpu"))
    retriever = OnlineRetriever(chroma_client(index.get("path", "data/indexes/chroma"),
        os.environ.get("CHROMA_HOST", index.get("host")), index.get("port", 8000)),
        index["manifest"], index["documents"], encoder)
    model = ProcessModel(config["model"])
    try:
        return Resources(model, retriever, retriever.catalog, AgentConfig(**config.get("agent", {})), config)
    except BaseException:
        model.close()
        raise
