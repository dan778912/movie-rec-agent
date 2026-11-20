"""Tests for the Member 6 P2 training-data loader."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.reranking.p2_training_data import (
    P2TrainingDataError,
    load_p2_samples,
    parse_completion,
    task_counts,
    validate_user_disjointness,
)


def make_record(
    *,
    sample_id: str,
    user_id: int,
    split: str,
    task: str,
    completion: str,
) -> dict:
    """Create a minimal synthetic record with the handoff fields."""
    return {
        "completion": completion,
        "episode_id": f"{split}:{user_id}:1",
        "prompt": f"Synthetic prompt for {task}.",
        "prompt_version": "p2-prompt-v1",
        "provenance": {},
        "sample_id": sample_id,
        "schema_version": "p2-sample-v1",
        "split": split,
        "state": {},
        "step": 1,
        "target": {},
        "task": task,
        "user_id": user_id,
    }


def write_jsonl(path: Path, records: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            file.write("\n")


class P2TrainingDataTests(unittest.TestCase):
    def test_loads_matching_train_and_dev_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path = root / "train.jsonl"
            dev_path = root / "dev.jsonl"

            write_jsonl(
                train_path,
                [
                    make_record(
                        sample_id="train:1:1",
                        user_id=1,
                        split="train",
                        task="assess_sufficiency",
                        completion=(
                            '{"action": "assess_sufficiency", '
                            '"sufficient": true}'
                        ),
                    ),
                    make_record(
                        sample_id="train:1:2",
                        user_id=1,
                        split="train",
                        task="rank_candidates",
                        completion=(
                            '{"action": "rank_candidates", '
                            '"movie_ids": [20, 10]}'
                        ),
                    ),
                ],
            )
            write_jsonl(
                dev_path,
                [
                    make_record(
                        sample_id="dev:2:1",
                        user_id=2,
                        split="dev",
                        task="ask_clarification",
                        completion=(
                            '{"action": "ask_clarification", '
                            '"question": "Which genres do you prefer?"}'
                        ),
                    )
                ],
            )

            training = load_p2_samples(
                train_path,
                expected_split="train",
                expected_prompt_version="p2-prompt-v1",
            )
            development = load_p2_samples(
                dev_path,
                expected_split="dev",
                expected_prompt_version="p2-prompt-v1",
            )

            validate_user_disjointness(training, development)

            self.assertEqual(len(training), 2)
            self.assertEqual(len(development), 1)
            self.assertEqual(
                task_counts(training)["rank_candidates"],
                1,
            )

    def test_rejects_task_and_completion_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            write_jsonl(
                path,
                [
                    make_record(
                        sample_id="train:1:1",
                        user_id=1,
                        split="train",
                        task="ask_clarification",
                        completion=(
                            '{"action": "assess_sufficiency", '
                            '"sufficient": true}'
                        ),
                    )
                ],
            )

            with self.assertRaisesRegex(
                P2TrainingDataError,
                "task does not match",
            ):
                load_p2_samples(
                    path,
                    expected_split="train",
                    expected_prompt_version="p2-prompt-v1",
                )

    def test_rejects_noncanonical_completion(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.jsonl"
            write_jsonl(
                path,
                [
                    make_record(
                        sample_id="train:1:1",
                        user_id=1,
                        split="train",
                        task="assess_sufficiency",
                        completion=(
                            '{"action":"assess_sufficiency",'
                            '"sufficient":true}'
                        ),
                    )
                ],
            )

            with self.assertRaisesRegex(
                P2TrainingDataError,
                "canonical",
            ):
                load_p2_samples(
                    path,
                    expected_split="train",
                    expected_prompt_version="p2-prompt-v1",
                )

    def test_rejects_duplicate_ranked_movie(self) -> None:
        with self.assertRaisesRegex(
            P2TrainingDataError,
            "duplicates",
        ):
            parse_completion(
                '{"action": "rank_candidates", '
                '"movie_ids": [10, 10]}'
            )

    def test_rejects_overlapping_train_and_dev_users(self) -> None:
        training = [{"user_id": 7}]
        development = [{"user_id": 7}]

        with self.assertRaisesRegex(
            P2TrainingDataError,
            "overlap",
        ):
            validate_user_disjointness(
                training,
                development,
            )


if __name__ == "__main__":
    unittest.main()