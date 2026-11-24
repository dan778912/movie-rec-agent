"""Tests for the Member 6 P2 training configuration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.reranking.p2_config import load_p2_config


def valid_config() -> dict:
    return {
        "version": "p2-qwen-lora-v1",
        "base_model": "Qwen/Qwen2.5-1.5B-Instruct",
        "model_revision": (
            "775b11afaf83e0dc75bd5abaf90133e47b3ec082"
        ),
        "initial_model": "p1-qwen-lora-v1",
        "prompt_version": "p2-prompt-v1",
        "seed": 20261001,
        "max_length": 6656,
        "allow_truncation": False,
        "learning_rate": 0.0001,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": 16,
        "logging_steps": 10,
        "eval_steps": 250,
        "save_steps": 250,
        "save_total_limit": 6,
        "checkpoint_metric": "eval_loss",
        "shortlist_size": 3,
    }


def write_config(path: Path, config: dict) -> None:
    path.write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )


class P2ConfigTests(unittest.TestCase):
    def test_valid_configuration_loads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            write_config(path, valid_config())

            config = load_p2_config(path)

            self.assertEqual(config["max_length"], 6656)
            self.assertEqual(config["shortlist_size"], 3)
            self.assertFalse(config["allow_truncation"])

    def test_truncation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = valid_config()
            config["allow_truncation"] = True
            write_config(path, config)

            with self.assertRaisesRegex(
                ValueError,
                "does not permit",
            ):
                load_p2_config(path)

    def test_shortlist_must_fit_saved_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = valid_config()
            config["save_total_limit"] = 2
            write_config(path, config)

            with self.assertRaisesRegex(
                ValueError,
                "at least shortlist_size",
            ):
                load_p2_config(path)

    def test_checkpoint_metric_is_frozen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = valid_config()
            config["checkpoint_metric"] = "training_loss"
            write_config(path, config)

            with self.assertRaisesRegex(
                ValueError,
                "must be eval_loss",
            ):
                load_p2_config(path)


if __name__ == "__main__":
    unittest.main()