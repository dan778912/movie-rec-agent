"""Core P1 candidate reranking logic."""

from __future__ import annotations

import math
import re
from typing import Any, Protocol, Sequence


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

    def score_many(
        self,
        profile: str,
        candidate_documents: Sequence[str],
    ) -> list[float]:
        """Score multiple candidates while preserving input order."""
        return [
            self.score(profile, candidate_document)
            for candidate_document in candidate_documents
        ]

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


def _score_candidates(
    scorer: CandidateScorer,
    profile_text: str,
    document_texts: Sequence[str],
) -> list[float]:
    """Use batched scoring when supported, otherwise score individually."""
    score_many = getattr(scorer, "score_many", None)

    if callable(score_many):
        scores = [
            float(score)
            for score in score_many(profile_text, document_texts)
        ]
    else:
        scores = [
            float(scorer.score(profile_text, document_text))
            for document_text in document_texts
        ]

    if len(scores) != len(document_texts):
        raise ValueError(
            "Model returned a different number of scores than candidates"
        )

    return scores


def rerank_candidate_pools(
    candidate_pools: list[dict[str, Any]],
    profiles: dict[int, dict[str, Any]],
    movie_documents: dict[int, dict[str, Any]],
    scorer: CandidateScorer,
) -> list[dict[str, Any]]:
    """Rerank each frozen candidate pool without changing its movie IDs."""
    output: list[dict[str, Any]] = []
    generate_explanations = bool(
        getattr(scorer, "generate_explanations", True)
    )

    for pool in candidate_pools:
        user_id = pool["user_id"]

        if user_id not in profiles:
            raise ValueError(f"Missing profile for user {user_id}")

        profile_text = profiles[user_id]["text"]
        originals = pool["recommendations"]
        document_texts: list[str] = []

        for original in originals:
            movie_id = original["movie_id"]

            if movie_id not in movie_documents:
                raise ValueError(
                    f"Missing movie document for movie {movie_id}"
                )

            document_texts.append(movie_documents[movie_id]["text"])

        candidate_scores = _score_candidates(
            scorer,
            profile_text,
            document_texts,
        )
        scored: list[dict[str, Any]] = []

        for original, document_text, score in zip(
            originals,
            document_texts,
            candidate_scores,
            strict=True,
        ):
            movie_id = original["movie_id"]

            if not math.isfinite(score):
                raise ValueError(
                    f"Model returned a nonfinite score for movie {movie_id}"
                )

            item: dict[str, Any] = {
                "movie_id": movie_id,
                "score": score,
                "original_rank": original["rank"],
            }

            if generate_explanations:
                explanation = scorer.explain(
                    profile_text,
                    document_text,
                ).strip()

                if not explanation:
                    raise ValueError(
                        "Model returned an empty explanation for movie "
                        f"{movie_id}"
                    )

                item["explanation"] = explanation

            scored.append(item)

        scored.sort(
            key=lambda item: (
                -item["score"],
                item["original_rank"],
                item["movie_id"],
            )
        )

        recommendations: list[dict[str, Any]] = []

        for new_rank, item in enumerate(scored, start=1):
            recommendation: dict[str, Any] = {
                "movie_id": item["movie_id"],
                "score": item["score"],
                "rank": new_rank,
            }

            if "explanation" in item:
                recommendation["explanation"] = item["explanation"]

            recommendations.append(recommendation)

        output.append(
            {
                "user_id": user_id,
                "split": pool["split"],
                "method": "p1",
                "recommendations": recommendations,
            }
        )

    return output