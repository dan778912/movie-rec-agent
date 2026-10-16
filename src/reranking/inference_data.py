"""Load profiles, movie documents, and frozen candidate pools."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .training_data import read_jsonl


def load_profiles(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load canonical user profiles indexed by user ID."""
    profiles: dict[int, dict[str, Any]] = {}

    for record in read_jsonl(path):
        user_id = record.get("user_id")
        profile_text = record.get("text")

        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError("Every profile must have a positive integer user_id")

        if not isinstance(profile_text, str) or not profile_text.strip():
            raise ValueError(f"Profile for user {user_id} has no text")

        if user_id in profiles:
            raise ValueError(f"Duplicate profile for user {user_id}")

        profiles[user_id] = record

    if not profiles:
        raise ValueError(f"No profiles found in {path}")

    return profiles


def load_movie_documents(path: str | Path) -> dict[int, dict[str, Any]]:
    """Load canonical movie documents indexed by movie ID."""
    documents: dict[int, dict[str, Any]] = {}

    for record in read_jsonl(path):
        movie_id = record.get("movie_id")
        document_text = record.get("text")

        if not isinstance(movie_id, int) or movie_id <= 0:
            raise ValueError(
                "Every movie document must have a positive integer movie_id"
            )

        if not isinstance(document_text, str) or not document_text.strip():
            raise ValueError(f"Movie {movie_id} has no document text")

        if movie_id in documents:
            raise ValueError(f"Duplicate movie document {movie_id}")

        documents[movie_id] = record

    if not documents:
        raise ValueError(f"No movie documents found in {path}")

    return documents


def load_candidate_pools(
    path: str | Path,
    expected_split: str = "validation",
) -> list[dict[str, Any]]:
    """Load and validate Member 5's frozen candidate pools."""
    pools: list[dict[str, Any]] = []
    seen_users: set[int] = set()

    for line_number, record in enumerate(read_jsonl(path), start=1):
        user_id = record.get("user_id")
        split = record.get("split")
        method = record.get("method")
        recommendations = record.get("recommendations")

        if not isinstance(user_id, int) or user_id <= 0:
            raise ValueError(
                f"Candidate line {line_number} has an invalid user_id"
            )

        if user_id in seen_users:
            raise ValueError(f"Duplicate candidate pool for user {user_id}")

        if split != expected_split:
            raise ValueError(
                f"Candidate line {line_number} has split {split!r}; "
                f"expected {expected_split!r}"
            )

        if method != "embedding":
            raise ValueError(
                f"Candidate line {line_number} must use method 'embedding'"
            )

        if not isinstance(recommendations, list) or not recommendations:
            raise ValueError(
                f"Candidate line {line_number} has no recommendations"
            )

        movie_ids: set[int] = set()

        for expected_rank, recommendation in enumerate(
            recommendations,
            start=1,
        ):
            movie_id = recommendation.get("movie_id")
            rank = recommendation.get("rank")

            if not isinstance(movie_id, int) or movie_id <= 0:
                raise ValueError(
                    f"User {user_id} has an invalid candidate movie_id"
                )

            if movie_id in movie_ids:
                raise ValueError(
                    f"User {user_id} has duplicate movie {movie_id}"
                )

            if rank != expected_rank:
                raise ValueError(
                    f"User {user_id} has invalid candidate rank {rank}"
                )

            movie_ids.add(movie_id)

        seen_users.add(user_id)
        pools.append(record)

    if not pools:
        raise ValueError(f"No candidate pools found in {path}")

    return pools