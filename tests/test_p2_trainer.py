"""Tests for P2 token auditing and checkpoint shortlisting."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.reranking.train_p2_qwen import (
    build_length_report,
    checkpoint_candidates,
)


class FakeTokenizer:
    """Small tokenizer replacement for dependency-free unit tests."""

    eos_token = "<eos>"

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[int]]:
        if truncation:
            raise AssertionError(
                "The P2 trainer must never request truncation"
            )

        token_count = len(text.split())

        if add_special_tokens:
            token_count += 1

        return {
            "input_ids": list(range(token_count))
        }


def sample(
    sample_id: str,
    task: str,
    prompt: str,
    completion: str,
) -> dict:
    return {
        "sample_id": sample_id,
        "task": task,
        "prompt": prompt,
        "completion": completion,
    }


class P2TrainerTests(unittest.TestCase):
    def test_length_report_flags_records_without_truncating(self) -> None:
        training = [
            sample(
                "train:1",
                "rank_candidates",
                "one two three four five",
                "six seven",
            )
        ]
        development = [
            sample(
                "dev:1",
                "assess_sufficiency",
                "one two",
                "three",
            )
        ]

        report = build_length_report(
            training,
            development,
            FakeTokenizer(),
            max_length=6,
        )

        self.assertFalse(
            report["fits_configured_length"]
        )
        self.assertEqual(
            report["over_length_count"],
            1,
        )
        self.assertEqual(
            report["over_length"][0]["sample_id"],
            "train:1",
        )

    def test_length_report_accepts_records_that_fit(self) -> None:
        records = [
            sample(
                "train:1",
                "assess_sufficiency",
                "one two",
                "three",
            )
        ]

        report = build_length_report(
            records,
            records,
            FakeTokenizer(),
            max_length=20,
        )

        self.assertTrue(
            report["fits_configured_length"]
        )
        self.assertEqual(
            report["over_length_count"],
            0,
        )

    def test_checkpoint_shortlist_uses_loss_then_earlier_step(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory)

            for step in (100, 200, 300):
                (
                    checkpoint_root
                    / f"checkpoint-{step}"
                ).mkdir()

            history = [
                {
                    "step": 200,
                    "eval_loss": 0.50,
                },
                {
                    "step": 100,
                    "eval_loss": 0.50,
                },
                {
                    "step": 300,
                    "eval_loss": 0.40,
                },
            ]

            report = checkpoint_candidates(
                history,
                checkpoint_root,
                shortlist_size=2,
            )

            self.assertEqual(
                [
                    item["step"]
                    for item in report["shortlist"]
                ],
                [300, 100],
            )
            self.assertEqual(
                report["tie_break"],
                "earlier_training_step",
            )


if __name__ == "__main__":
    unittest.main()