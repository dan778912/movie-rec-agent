"""Configuration loading and validation for P2 continued fine-tuning."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_FIELDS = {
    "version",
    "base_model",
    "model_revision",
    "initial_model",
    "prompt_version",
    "seed",
    "max_length",
    "allow_truncation",
    "learning_rate",
    "num_train_epochs",
    "per_device_train_batch_size",
    "per_device_eval_batch_size",
    "gradient_accumulation_steps",
    "logging_steps",
    "eval_steps",
    "save_steps",
    "save_total_limit",
    "checkpoint_metric",
    "shortlist_size",
}


def _positive_integer(value: Any) -> bool:
    return type(value) is int and value > 0


def _nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def load_p2_config(path: str | Path) -> dict[str, Any]:
    """Load and validate one P2 training configuration."""
    with Path(path).open("r", encoding="utf-8") as file:
        config = json.load(file)

    if not isinstance(config, dict):
        raise ValueError("The P2 configuration must be a JSON object")

    missing = REQUIRED_FIELDS - set(config)

    if missing:
        raise ValueError(
            f"P2 configuration is missing: {sorted(missing)}"
        )

    for field in (
        "version",
        "base_model",
        "model_revision",
        "initial_model",
        "prompt_version",
        "checkpoint_metric",
    ):
        if not _nonempty_string(config[field]):
            raise ValueError(f"{field} must be a nonempty string")

    if type(config["seed"]) is not int:
        raise ValueError("seed must be an integer")

    for field in (
        "max_length",
        "per_device_train_batch_size",
        "per_device_eval_batch_size",
        "gradient_accumulation_steps",
        "logging_steps",
        "eval_steps",
        "save_steps",
        "save_total_limit",
        "shortlist_size",
    ):
        if not _positive_integer(config[field]):
            raise ValueError(f"{field} must be a positive integer")

    if type(config["allow_truncation"]) is not bool:
        raise ValueError("allow_truncation must be a Boolean")

    if config["allow_truncation"]:
        raise ValueError(
            "P2 does not permit silent prompt truncation"
        )

    if (
        not isinstance(config["learning_rate"], (int, float))
        or isinstance(config["learning_rate"], bool)
        or config["learning_rate"] <= 0
    ):
        raise ValueError("learning_rate must be positive")

    if (
        not isinstance(config["num_train_epochs"], (int, float))
        or isinstance(config["num_train_epochs"], bool)
        or config["num_train_epochs"] <= 0
    ):
        raise ValueError("num_train_epochs must be positive")

    if config["checkpoint_metric"] != "eval_loss":
        raise ValueError(
            "The stage-one checkpoint metric must be eval_loss"
        )

    if config["save_steps"] % config["eval_steps"] != 0:
        raise ValueError(
            "save_steps must be a multiple of eval_steps"
        )

    if config["save_total_limit"] < config["shortlist_size"]:
        raise ValueError(
            "save_total_limit must be at least shortlist_size"
        )

    return config