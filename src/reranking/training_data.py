"""Load Member 3's P1 pointwise training examples."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator

from .prompts import build_ranking_prompt

VALID_LABELS = {"relevant", "not_relevant"}
REQUIRED_FIELDS = {
    "example_id",
    "user_id",
    "movie_id",
    "split",
    "profile",
    "candidate_document",
    "label",
}


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Read nonempty JSON objects from a JSONL file."""
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON on line {line_number} of {path}"
                ) from error

            if not isinstance(record, dict):
                raise ValueError(
                    f"Line {line_number} of {path} must contain a JSON object"
                )

            yield record


def load_training_examples(path: str | Path) -> list[dict[str, Any]]:
    """Validate and load Member 3's training-example file."""
    examples: list[dict[str, Any]] = []

    for line_number, record in enumerate(read_jsonl(path), start=1):
        missing = REQUIRED_FIELDS - record.keys()
        if missing:
            raise ValueError(
                f"Training line {line_number} is missing: {sorted(missing)}"
            )

        if record["split"] != "train":
            raise ValueError(
                f"Training line {line_number} has split {record['split']!r}"
            )

        if record["label"] not in VALID_LABELS:
            raise ValueError(
                f"Training line {line_number} has invalid label "
                f"{record['label']!r}"
            )

        if not isinstance(record["profile"], str) or not record["profile"].strip():
            raise ValueError(
                f"Training line {line_number} has an empty profile"
            )

        if (
            not isinstance(record["candidate_document"], str)
            or not record["candidate_document"].strip()
        ):
            raise ValueError(
                f"Training line {line_number} has an empty candidate document"
            )

        examples.append(record)

    if not examples:
        raise ValueError(f"No training examples found in {path}")

    return examples


def to_sft_records(
    examples: list[dict[str, Any]],
) -> list[dict[str, str]]:
    """Convert validated examples into prompt/completion records."""
    return [
        {
            "prompt": build_ranking_prompt(
                example["profile"],
                example["candidate_document"],
            ),
            "completion": example["label"],
        }
        for example in examples
    ]