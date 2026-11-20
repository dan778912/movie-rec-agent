"""Load and validate Member 1's P2 prompt-completion samples."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

P2_TASKS = (
    "assess_sufficiency",
    "ask_clarification",
    "evaluate_candidates",
    "retrieve_candidates",
    "rank_candidates",
    "explain_recommendations",
)

ACTION_FIELDS = {
    "assess_sufficiency": ("sufficient",),
    "ask_clarification": ("question",),
    "evaluate_candidates": ("pool_sufficient",),
    "retrieve_candidates": ("query",),
    "rank_candidates": ("movie_ids",),
    "explain_recommendations": ("explanations",),
}

REQUIRED_SAMPLE_FIELDS = {
    "completion",
    "episode_id",
    "prompt",
    "prompt_version",
    "provenance",
    "sample_id",
    "schema_version",
    "split",
    "state",
    "step",
    "target",
    "task",
    "user_id",
}


class P2TrainingDataError(ValueError):
    """Raised when a P2 training record violates the handoff contract."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 checksum of a file."""
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)

    return digest.hexdigest()


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}

    for key, value in pairs:
        if key in result:
            raise P2TrainingDataError(
                f"Completion contains duplicate JSON key: {key}"
            )
        result[key] = value

    return result


def _reject_nonfinite_number(value: str) -> None:
    raise P2TrainingDataError(
        f"Completion contains non-finite JSON number: {value}"
    )


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _validate_action(action: Any) -> None:
    if not isinstance(action, dict):
        raise P2TrainingDataError(
            "Completion must contain one JSON object"
        )

    task = action.get("action")

    if task not in P2_TASKS:
        raise P2TrainingDataError(
            f"Completion has unknown action: {task!r}"
        )

    expected_fields = {"action", *ACTION_FIELDS[task]}

    if set(action) != expected_fields:
        raise P2TrainingDataError(
            f"{task} fields must be exactly "
            f"{sorted(expected_fields)}"
        )

    if task == "assess_sufficiency":
        if type(action["sufficient"]) is not bool:
            raise P2TrainingDataError(
                "sufficient must be a Boolean"
            )

    elif task == "ask_clarification":
        if not _nonempty_string(action["question"]):
            raise P2TrainingDataError(
                "question must be a nonempty string"
            )

    elif task == "evaluate_candidates":
        if type(action["pool_sufficient"]) is not bool:
            raise P2TrainingDataError(
                "pool_sufficient must be a Boolean"
            )

    elif task == "retrieve_candidates":
        if not _nonempty_string(action["query"]):
            raise P2TrainingDataError(
                "query must be a nonempty string"
            )

    elif task == "rank_candidates":
        movie_ids = action["movie_ids"]

        if (
            not isinstance(movie_ids, list)
            or not movie_ids
            or not all(_positive_integer(value) for value in movie_ids)
        ):
            raise P2TrainingDataError(
                "movie_ids must be a nonempty list of positive integers"
            )

        if len(set(movie_ids)) != len(movie_ids):
            raise P2TrainingDataError(
                "movie_ids must not contain duplicates"
            )

    elif task == "explain_recommendations":
        explanations = action["explanations"]

        if not isinstance(explanations, list) or not explanations:
            raise P2TrainingDataError(
                "explanations must be a nonempty list"
            )

        for explanation in explanations:
            if (
                not isinstance(explanation, dict)
                or set(explanation) != {"movie_id", "text"}
            ):
                raise P2TrainingDataError(
                    "Each explanation must contain exactly "
                    "movie_id and text"
                )

            if not _positive_integer(explanation["movie_id"]):
                raise P2TrainingDataError(
                    "Explanation movie_id must be a positive integer"
                )

            if not _nonempty_string(explanation["text"]):
                raise P2TrainingDataError(
                    "Explanation text must be nonempty"
                )


def parse_completion(text: str) -> dict[str, Any]:
    """Parse and validate one strict P2 completion."""
    if not _nonempty_string(text):
        raise P2TrainingDataError("Completion must be nonempty")

    try:
        action = json.loads(
            text,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite_number,
        )
    except json.JSONDecodeError as error:
        raise P2TrainingDataError(
            f"Completion is invalid JSON: {error.msg}"
        ) from error

    _validate_action(action)
    return action


def _canonical_completion(action: dict[str, Any]) -> str:
    ordered = {"action": action["action"]}

    for field in ACTION_FIELDS[action["action"]]:
        ordered[field] = action[field]

    return json.dumps(ordered, ensure_ascii=False)


def load_p2_samples(
    path: Path,
    *,
    expected_split: str,
    expected_prompt_version: str,
) -> list[dict[str, Any]]:
    """Load one frozen P2 train or dev JSONL file."""
    if expected_split not in {"train", "dev"}:
        raise ValueError("expected_split must be train or dev")

    records: list[dict[str, Any]] = []
    sample_ids: set[str] = set()

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                raise P2TrainingDataError(
                    f"{path}:{line_number}: blank lines are not allowed"
                )

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise P2TrainingDataError(
                    f"{path}:{line_number}: invalid JSON: {error.msg}"
                ) from error

            if not isinstance(record, dict):
                raise P2TrainingDataError(
                    f"{path}:{line_number}: record must be an object"
                )

            missing = REQUIRED_SAMPLE_FIELDS - set(record)

            if missing:
                raise P2TrainingDataError(
                    f"{path}:{line_number}: missing fields "
                    f"{sorted(missing)}"
                )

            sample_id = record["sample_id"]

            if not _nonempty_string(sample_id):
                raise P2TrainingDataError(
                    f"{path}:{line_number}: invalid sample_id"
                )

            if sample_id in sample_ids:
                raise P2TrainingDataError(
                    f"{path}:{line_number}: duplicate sample_id "
                    f"{sample_id!r}"
                )

            if record["split"] != expected_split:
                raise P2TrainingDataError(
                    f"{path}:{line_number}: expected split "
                    f"{expected_split!r}, got {record['split']!r}"
                )

            if record["prompt_version"] != expected_prompt_version:
                raise P2TrainingDataError(
                    f"{path}:{line_number}: expected prompt version "
                    f"{expected_prompt_version!r}, got "
                    f"{record['prompt_version']!r}"
                )

            if not _nonempty_string(record["prompt"]):
                raise P2TrainingDataError(
                    f"{path}:{line_number}: prompt must be nonempty"
                )

            action = parse_completion(record["completion"])

            if record["task"] != action["action"]:
                raise P2TrainingDataError(
                    f"{path}:{line_number}: task does not match "
                    "completion action"
                )

            canonical = _canonical_completion(action)

            if record["completion"] != canonical:
                raise P2TrainingDataError(
                    f"{path}:{line_number}: completion is not in "
                    "canonical serialized form"
                )

            if not _positive_integer(record["user_id"]):
                raise P2TrainingDataError(
                    f"{path}:{line_number}: user_id must be positive"
                )

            sample_ids.add(sample_id)
            records.append(
                {
                    "sample_id": sample_id,
                    "episode_id": record["episode_id"],
                    "user_id": record["user_id"],
                    "task": record["task"],
                    "prompt": record["prompt"],
                    "completion": record["completion"],
                }
            )

    if not records:
        raise P2TrainingDataError(f"No records found in {path}")

    return records


def validate_user_disjointness(
    training_records: list[dict[str, Any]],
    development_records: list[dict[str, Any]],
) -> None:
    """Require train and dev users to be disjoint."""
    training_users = {
        record["user_id"] for record in training_records
    }
    development_users = {
        record["user_id"] for record in development_records
    }
    overlap = training_users & development_users

    if overlap:
        preview = sorted(overlap)[:10]
        raise P2TrainingDataError(
            "Train and dev users overlap; first overlapping IDs: "
            f"{preview}"
        )


def task_counts(
    records: list[dict[str, Any]],
) -> dict[str, int]:
    """Return deterministic counts for all six tasks."""
    counts = Counter(record["task"] for record in records)
    return {
        task: counts.get(task, 0)
        for task in P2_TASKS
    }