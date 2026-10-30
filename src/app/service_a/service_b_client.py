"""The Service A <-> Service B contract.

This is provisional: Member 3 has not shipped Service B yet, and the exact
agent action JSON (Member 4's contract) is not agreed. `ServiceBClient` is the
interface Service A codes against; `HttpServiceBClient` is what calls the real
thing once it exists; `MockServiceBClient` is a deterministic stand-in so
Service A is runnable and testable today.

Request shape sent to Service B (`decide`):
    {
      "profile_text": str,            # src/profiles/text.py user_profile() output
      "history_movie_ids": [int, ...], # full watched set, for exclusion
      "clarifications": [{"question": str, "answer": str}, ...]
    }

Response shape (one of):
    {"action": "ask", "question": str}
    {"action": "recommend", "items": [
        {"movie_id": int, "score": float, "rank": int, "explanation": str}, ...
    ]}
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import Protocol, TypedDict


class ClarificationTurn(TypedDict):
    question: str
    answer: str


class CandidateMovie(TypedDict):
    movie_id: int
    title: str


class ServiceBClient(Protocol):
    def decide(
        self,
        *,
        profile_text: str,
        history_movie_ids: list[int],
        clarifications: list[ClarificationTurn],
    ) -> dict: ...


class MockServiceBClient:
    """Deterministic placeholder for the real agent + retrieval + reranking
    pipeline (Members 3/4/6). Asks one clarifying question when the watched
    history looks thin and no clarification has happened yet; otherwise
    returns `top_k` candidates from `candidate_source` as recommendations.
    Never re-asks after one clarification round, so the mock always
    terminates.
    """

    def __init__(
        self,
        candidate_source: Callable[[Iterable[int]], list[CandidateMovie]],
        min_history_for_sufficiency: int = 3,
        top_k: int = 10,
    ) -> None:
        self._candidate_source = candidate_source
        self._min_history_for_sufficiency = min_history_for_sufficiency
        self._top_k = top_k

    def decide(
        self,
        *,
        profile_text: str,
        history_movie_ids: list[int],
        clarifications: list[ClarificationTurn],
    ) -> dict:
        if not clarifications and len(history_movie_ids) < self._min_history_for_sufficiency:
            return {
                "action": "ask",
                "question": (
                    "You've only rated a couple of movies so far — want to tell me "
                    "a genre or mood you're in the mood for?"
                ),
            }

        candidates = self._candidate_source(history_movie_ids)[: self._top_k]
        items = [
            {
                "movie_id": candidate["movie_id"],
                "score": round(1.0 - 0.05 * rank, 4),
                "rank": rank + 1,
                "explanation": (
                    f"Placeholder justification: picked from the candidate pool for a "
                    f"user whose profile is \"{profile_text[:80]}...\"."
                    if len(profile_text) > 80
                    else f"Placeholder justification: picked from the candidate pool for a user whose profile is \"{profile_text}\"."
                ),
            }
            for rank, candidate in enumerate(candidates)
        ]
        return {"action": "recommend", "items": items}


class HttpServiceBClient:
    """Calls the real Service B once Member 3 ships it."""

    def __init__(self, base_url: str | None = None, timeout: float = 30.0) -> None:
        self._base_url = (base_url or os.environ.get("SERVICE_B_BASE_URL", "http://localhost:8001")).rstrip("/")
        self._timeout = timeout

    def decide(
        self,
        *,
        profile_text: str,
        history_movie_ids: list[int],
        clarifications: list[ClarificationTurn],
    ) -> dict:
        import httpx

        response = httpx.post(
            f"{self._base_url}/decide",
            json={
                "profile_text": profile_text,
                "history_movie_ids": history_movie_ids,
                "clarifications": clarifications,
            },
            timeout=self._timeout,
        )
        response.raise_for_status()
        return response.json()
