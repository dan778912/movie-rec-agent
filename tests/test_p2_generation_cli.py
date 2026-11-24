"""Tests for selecting P2 raw-output generation samples."""

from __future__ import annotations

import unittest

from src.reranking.p2_training_data import P2_TASKS
from src.reranking.run_p2_generation import select_examples


def record(task: str, number: int) -> dict:
    return {
        "sample_id": f"{task}:{number}",
        "task": task,
    }


class P2GenerationCliTests(unittest.TestCase):
    def test_selects_requested_count_for_every_task(self) -> None:
        records = []

        for number in range(3):
            for task in P2_TASKS:
                records.append(record(task, number))

        selected = select_examples(
            records,
            examples_per_task=2,
        )

        self.assertEqual(
            len(selected),
            2 * len(P2_TASKS),
        )

        for task in P2_TASKS:
            selected_ids = [
                item["sample_id"]
                for item in selected
                if item["task"] == task
            ]
            self.assertEqual(
                selected_ids,
                [f"{task}:0", f"{task}:1"],
            )

    def test_rejects_missing_task_samples(self) -> None:
        records = [
            record(
                "assess_sufficiency",
                0,
            )
        ]

        with self.assertRaisesRegex(
            ValueError,
            "Not enough samples",
        ):
            select_examples(
                records,
                examples_per_task=1,
            )

    def test_rejects_nonpositive_request(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "must be positive",
        ):
            select_examples(
                [],
                examples_per_task=0,
            )


if __name__ == "__main__":
    unittest.main()