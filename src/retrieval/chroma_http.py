"""Bounded transport adapter for the pinned Chroma 1.3.5 client.

The upstream synchronous transport uses timeout=None. Keep this one small override
isolated and covered by a transport test when upgrading Chroma.
"""
import math
import os

import httpx


class TimeoutProxy:
    def __init__(self, target):
        self.target = target

    def __getattr__(self, name):
        member = getattr(self.target, name)
        if not callable(member):
            return member
        def call(*args, **kwargs):
            try:
                result = member(*args, **kwargs)
            except httpx.TimeoutException as error:
                raise TimeoutError("Chroma HTTP transport timed out") from error
            return TimeoutProxy(result) if name in ("create_collection", "get_collection") else result
        return call


def bounded_http_client(host, port, settings):
    import chromadb
    timeout = float(os.environ.get("CHROMA_REQUEST_TIMEOUT_SECONDS", "5"))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("CHROMA_REQUEST_TIMEOUT_SECONDS must be finite and positive")
    # Pinned Chroma exposes no public timeout argument. Its collection operations
    # share this session. Initialization happens before the timeout is installed.
    client = chromadb.HttpClient(host=host, port=port, settings=settings)
    client._server._session.timeout = httpx.Timeout(timeout)
    return TimeoutProxy(client)
