"""Core P1 candidate reranking logic."""

from __future__ import annotations

import math
import re
from typing import Any, Protocol


class CandidateScorer(Protocol):
    """Interface implemented by a trained reranking model."""

    def score(self, profile: str, candidate_document: str) -> float:
        """Return a higher score for a better user/movie match."""

    def explain(self, profile: str, candidate_document: str) -> str:
        """Return a short explanation grounded in the supplied text."""


class KeywordOverlapScorer:
    """Lightweight scorer used only for tests and pipeline smoke checks."""

    _stopwords = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "the",
        "to",
        "with",
    }

    @classmethod
    def _terms(cls, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if len(token) > 2 and token not in cls._stopwords
        }

    def score(self, profile: str, candidate_document: str) -> float:
        profile_terms = self._terms(profile)
        movie_terms = self._terms(candidate_document)

        if not profile_terms or not movie_terms:
            return 0.0

        return len(profile_terms & movie_terms) / len(movie_terms)

    def explain(self, profile: str, candidate_document: str) -> str:
        shared = sorted(
            self._terms(profile) & self._terms(candidate_document)
        )

        if not shared:
            return (
                "This candidate was selected from the retrieved movie pool, "
                "but no direct text overlap was found."
            )

        terms = ", ".join(shared[:5])
        return f"This movie matches the profile through: {terms}."


def rerank_candidate_pools(
    candidate_pools: list[dict[str, Any]],
    profiles: dict[int, dict[str, Any]],
    movie_documents: dict[int, dict[str, Any]],
    scorer: CandidateScorer,
) -> list[dict[str, Any]]:
    """Rerank each frozen candidate pool without changing its movie IDs."""
    output: list[dict[str, Any]] = []

    for pool in candidate_pools:
        user_id = pool["user_id"]

        if user_id not in profiles:
            raise ValueError(f"Missing profile for user {user_id}")

        profile_text = profiles[user_id]["text"]
        scored: list[dict[str, Any]] = []

        for original in pool["recommendations"]:
            movie_id = original["movie_id"]

            if movie_id not in movie_documents:
                raise ValueError(
                    f"Missing movie document for movie {movie_id}"
                )

            document_text = movie_documents[movie_id]["text"]
            score = float(scorer.score(profile_text, document_text))

            if not math.isfinite(score):
                raise ValueError(
                    f"Model returned a nonfinite score for movie {movie_id}"
                )

            explanation = scorer.explain(
                profile_text,
                document_text,
            ).strip()

            if not explanation:
                raise ValueError(
                    f"Model returned an empty explanation for movie {movie_id}"
                )

            scored.append(
                {
                    "movie_id": movie_id,
                    "score": score,
                    "original_rank": original["rank"],
                    "explanation": explanation,
                }
            )

        scored.sort(
            key=lambda item: (
                -item["score"],
                item["original_rank"],
                item["movie_id"],
            )
        )

        recommendations = []

        for new_rank, item in enumerate(scored, start=1):
            recommendations.append(
                {
                    "movie_id": item["movie_id"],
                    "score": item["score"],
                    "rank": new_rank,
                    "explanation": item["explanation"],
                }
            )

        output.append(
            {
                "user_id": user_id,
                "split": pool["split"],
                "method": "p1",
                "recommendations": recommendations,
            }
        )

    return output