"""Configuration loading and validation for P1 fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "version",
    "base_model",
    "model_revision",
    "seed",
    "max_length",
    "validation_fraction",
    "learning_rate",
    "num_train_epochs",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "lora_r",
    "lora_alpha",
    "lora_dropout",
    "target_modules",
    "logging_steps",
    "eval_steps",
    "save_steps",
    "save_total_limit",
}


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate the P1 LoRA configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("The reranking configuration must be a JSON object")

    missing = REQUIRED_FIELDS - config.keys()
    if missing:
        raise ValueError(f"Configuration is missing: {sorted(missing)}")

    if not isinstance(config["base_model"], str) or not config["base_model"]:
        raise ValueError("base_model must be a nonempty string")

    if (
        not isinstance(config["model_revision"], str)
        or not config["model_revision"]
    ):
        raise ValueError("model_revision must be a nonempty string")

    if not isinstance(config["seed"], int):
        raise ValueError("seed must be an integer")

    if not isinstance(config["max_length"], int) or config["max_length"] <= 0:
        raise ValueError("max_length must be a positive integer")

    validation_fraction = config["validation_fraction"]
    if not isinstance(validation_fraction, (int, float)) or not (
        0 < validation_fraction < 1
    ):
        raise ValueError("validation_fraction must be between 0 and 1")

    if (
        not isinstance(config["learning_rate"], (int, float))
        or config["learning_rate"] <= 0
    ):
        raise ValueError("learning_rate must be positive")

    if (
        not isinstance(config["target_modules"], list)
        or not config["target_modules"]
        or not all(
            isinstance(module, str) and module
            for module in config["target_modules"]
        )
    ):
        raise ValueError("target_modules must be a nonempty string list")

    return config