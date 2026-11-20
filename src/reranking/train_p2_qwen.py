"""Continue fine-tuning the P1 Qwen LoRA adapter for P2 actions."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

from .p2_config import load_p2_config
from .p2_training_data import (
    load_p2_samples,
    sha256_file,
    task_counts,
    validate_user_disjointness,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Continue fine-tuning the P1 LoRA adapter on the six "
            "structured P2 agent actions."
        )
    )
    parser.add_argument(
        "--training-data",
        required=True,
        type=Path,
        help="Member 1's frozen p2_samples_train.jsonl.",
    )
    parser.add_argument(
        "--development-data",
        required=True,
        type=Path,
        help="Member 1's frozen p2_samples_dev.jsonl.",
    )
    parser.add_argument(
        "--initial-adapter",
        required=True,
        type=Path,
        help="Path to the trained P1 LoRA adapter.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p2_qwen_lora_v1.json"),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--max-train-examples",
        type=int,
        default=None,
        help="Optional prefix limit for a documented smoke test.",
    )
    parser.add_argument(
        "--max-dev-examples",
        type=int,
        default=None,
        help="Optional dev prefix limit for a documented smoke test.",
    )
    parser.add_argument(
        "--length-check-only",
        action="store_true",
        help=(
            "Measure real tokenizer lengths and exit without loading "
            "the model or using a GPU."
        ),
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training without CUDA. This will be very slow.",
    )
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    """Write one deterministic, readable JSON file."""
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def git_value(*arguments: str) -> str | None:
    """Return one Git value, or None outside a usable repository."""
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    return result.stdout.strip()


def git_provenance() -> dict[str, Any]:
    """Record the source revision and whether tracked files changed."""
    commit = git_value("rev-parse", "HEAD")
    status = git_value("status", "--short")

    return {
        "commit": commit,
        "tracked_worktree_clean": status == "" if status is not None else None,
        "status": status,
    }


def directory_file_hashes(path: Path) -> dict[str, str]:
    """Hash every regular file directly inside an adapter directory."""
    return {
        item.name: sha256_file(item)
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def percentile(values: list[int], fraction: float) -> int:
    """Return a nearest-rank percentile for a nonempty integer list."""
    ordered = sorted(values)
    index = max(0, math.ceil(fraction * len(ordered)) - 1)
    return ordered[index]


def summarize_lengths(lengths: list[int]) -> dict[str, int]:
    """Summarize token lengths without external dependencies."""
    return {
        "count": len(lengths),
        "minimum": min(lengths),
        "p50": percentile(lengths, 0.50),
        "p95": percentile(lengths, 0.95),
        "maximum": max(lengths),
    }


def tokenize_record(
    record: dict[str, Any],
    tokenizer: Any,
) -> dict[str, Any]:
    """Tokenize one prompt and completion without truncation."""
    completion_text = record["completion"] + tokenizer.eos_token

    prompt_ids = tokenizer(
        record["prompt"],
        add_special_tokens=True,
        truncation=False,
    )["input_ids"]
    completion_ids = tokenizer(
        completion_text,
        add_special_tokens=False,
        truncation=False,
    )["input_ids"]

    return {
        "sample_id": record["sample_id"],
        "task": record["task"],
        "prompt_tokens": len(prompt_ids),
        "completion_tokens": len(completion_ids),
        "total_tokens": len(prompt_ids) + len(completion_ids),
    }


def build_length_report(
    training_records: list[dict[str, Any]],
    development_records: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
) -> dict[str, Any]:
    """Measure all records with the real tokenizer."""
    report: dict[str, Any] = {
        "max_length": max_length,
        "truncation_allowed": False,
        "splits": {},
        "over_length": [],
    }

    for split, records in (
        ("train", training_records),
        ("dev", development_records),
    ):
        measured = [
            tokenize_record(record, tokenizer)
            for record in records
        ]
        totals = [item["total_tokens"] for item in measured]
        prompts = [item["prompt_tokens"] for item in measured]
        completions = [
            item["completion_tokens"] for item in measured
        ]

        by_task: dict[str, list[int]] = defaultdict(list)

        for item in measured:
            by_task[item["task"]].append(item["total_tokens"])

            if item["total_tokens"] > max_length:
                report["over_length"].append(
                    {
                        "split": split,
                        **item,
                        "tokens_over": (
                            item["total_tokens"] - max_length
                        ),
                    }
                )

        report["splits"][split] = {
            "records": len(records),
            "prompt_tokens": summarize_lengths(prompts),
            "completion_tokens": summarize_lengths(completions),
            "total_tokens": summarize_lengths(totals),
            "total_tokens_by_task": {
                task: summarize_lengths(values)
                for task, values in sorted(by_task.items())
            },
        }

    report["over_length_count"] = len(report["over_length"])
    report["fits_configured_length"] = (
        report["over_length_count"] == 0
    )
    return report


def limit_records(
    records: list[dict[str, Any]],
    maximum: int | None,
    option_name: str,
) -> list[dict[str, Any]]:
    """Apply an optional deterministic prefix limit."""
    if maximum is None:
        return records

    if maximum <= 0:
        raise ValueError(f"{option_name} must be positive")

    return records[:maximum]


def checkpoint_candidates(
    log_history: list[dict[str, Any]],
    checkpoint_root: Path,
    shortlist_size: int,
) -> dict[str, Any]:
    """Rank evaluated, retained checkpoints by dev loss."""
    candidates = []

    for event in log_history:
        if "eval_loss" not in event or "step" not in event:
            continue

        step = int(event["step"])
        checkpoint_path = checkpoint_root / f"checkpoint-{step}"

        candidates.append(
            {
                "step": step,
                "eval_loss": float(event["eval_loss"]),
                "checkpoint": str(checkpoint_path),
                "retained": checkpoint_path.exists(),
            }
        )

    candidates.sort(
        key=lambda item: (
            item["eval_loss"],
            item["step"],
        )
    )

    retained = [
        item for item in candidates
        if item["retained"]
    ]

    return {
        "selection_stage": "training_period_dev",
        "metric": "eval_loss",
        "greater_is_better": False,
        "tie_break": "earlier_training_step",
        "requested_shortlist_size": shortlist_size,
        "evaluated_candidates": candidates,
        "shortlist": retained[:shortlist_size],
    }


def main() -> None:
    args = parse_args()
    config = load_p2_config(args.config)

    if not args.training_data.is_file():
        raise FileNotFoundError(args.training_data)

    if not args.development_data.is_file():
        raise FileNotFoundError(args.development_data)

    if not args.initial_adapter.is_dir():
        raise FileNotFoundError(args.initial_adapter)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}"
        )

    training_records = load_p2_samples(
        args.training_data,
        expected_split="train",
        expected_prompt_version=config["prompt_version"],
    )
    development_records = load_p2_samples(
        args.development_data,
        expected_split="dev",
        expected_prompt_version=config["prompt_version"],
    )
    validate_user_disjointness(
        training_records,
        development_records,
    )

    training_records = limit_records(
        training_records,
        args.max_train_examples,
        "--max-train-examples",
    )
    development_records = limit_records(
        development_records,
        args.max_dev_examples,
        "--max-dev-examples",
    )

    if len(training_records) < 2:
        raise ValueError(
            "At least two P2 training records are required"
        )

    if not development_records:
        raise ValueError(
            "At least one P2 development record is required"
        )

    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise SystemExit(
            "P2 dependencies are missing. Install "
            "requirements-reranking.txt."
        ) from error

    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.initial_adapter,
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    length_report = build_length_report(
        training_records,
        development_records,
        tokenizer,
        int(config["max_length"]),
    )
    write_json(
        args.output_dir / "token_lengths.json",
        length_report,
    )

    print(
        "Measured token lengths. Maximum train length:",
        length_report["splits"]["train"]["total_tokens"]["maximum"],
    )
    print(
        "Measured token lengths. Maximum dev length:",
        length_report["splits"]["dev"]["total_tokens"]["maximum"],
    )

    if not length_report["fits_configured_length"]:
        preview = length_report["over_length"][:10]
        raise ValueError(
            f"{length_report['over_length_count']} records exceed "
            f"max_length={config['max_length']}. No records were "
            f"truncated. First records: {preview}"
        )

    if args.length_check_only:
        print(
            "Length check passed. No model was loaded and no "
            "training was performed."
        )
        return

    try:
        import torch
        from peft import PeftConfig, PeftModel
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise SystemExit(
            "P2 dependencies are missing. Install "
            "requirements-reranking.txt."
        ) from error

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA GPU not detected. Run training on GPU hardware, "
            "or pass --allow-cpu only for an intentional slow test."
        )

    set_seed(int(config["seed"]))

    peft_config = PeftConfig.from_pretrained(
        args.initial_adapter
    )

    if peft_config.base_model_name_or_path != config["base_model"]:
        raise ValueError(
            "P1 adapter base model does not match the P2 config: "
            f"{peft_config.base_model_name_or_path!r} != "
            f"{config['base_model']!r}"
        )

    class PromptCompletionDataset(Dataset):
        def __init__(
            self,
            records: list[dict[str, Any]],
        ) -> None:
            self.records = records

        def __len__(self) -> int:
            return len(self.records)

        def __getitem__(
            self,
            index: int,
        ) -> dict[str, list[int]]:
            record = self.records[index]
            completion_text = (
                record["completion"] + tokenizer.eos_token
            )

            prompt_ids = tokenizer(
                record["prompt"],
                add_special_tokens=True,
                truncation=False,
            )["input_ids"]
            completion_ids = tokenizer(
                completion_text,
                add_special_tokens=False,
                truncation=False,
            )["input_ids"]

            total_length = len(prompt_ids) + len(completion_ids)

            if total_length > int(config["max_length"]):
                raise ValueError(
                    f"Sample {record['sample_id']} has "
                    f"{total_length} tokens, exceeding "
                    f"max_length={config['max_length']}"
                )

            input_ids = prompt_ids + completion_ids
            labels = (
                [-100] * len(prompt_ids)
                + completion_ids
            )

            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }

    def collate(
        batch: list[dict[str, list[int]]],
    ) -> dict[str, Any]:
        longest = max(
            len(item["input_ids"]) for item in batch
        )

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            padding = longest - len(item["input_ids"])
            input_ids.append(
                item["input_ids"]
                + [tokenizer.pad_token_id] * padding
            )
            attention_masks.append(
                item["attention_mask"] + [0] * padding
            )
            labels.append(
                item["labels"] + [-100] * padding
            )

        return {
            "input_ids": torch.tensor(
                input_ids,
                dtype=torch.long,
            ),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(
                labels,
                dtype=torch.long,
            ),
        }

    use_bfloat16 = (
        torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
    )
    model_dtype = (
        torch.bfloat16
        if use_bfloat16
        else torch.float16
        if torch.cuda.is_available()
        else torch.float32
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        revision=config["model_revision"],
        dtype=model_dtype,
    )
    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()

    if hasattr(base_model, "enable_input_require_grads"):
        base_model.enable_input_require_grads()

    model = PeftModel.from_pretrained(
        base_model,
        args.initial_adapter,
        is_trainable=True,
    )
    model.config.use_cache = False

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    if trainable_parameters <= 0:
        raise RuntimeError(
            "The loaded P1 adapter has no trainable parameters"
        )

    checkpoint_root = args.output_dir / "checkpoints"

    training_arguments = TrainingArguments(
        output_dir=str(checkpoint_root),
        seed=int(config["seed"]),
        data_seed=int(config["seed"]),
        learning_rate=float(config["learning_rate"]),
        num_train_epochs=float(config["num_train_epochs"]),
        per_device_train_batch_size=int(
            config["per_device_train_batch_size"]
        ),
        per_device_eval_batch_size=int(
            config["per_device_eval_batch_size"]
        ),
        gradient_accumulation_steps=int(
            config["gradient_accumulation_steps"]
        ),
        logging_strategy="steps",
        logging_steps=int(config["logging_steps"]),
        eval_strategy="steps",
        eval_steps=int(config["eval_steps"]),
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=int(config["save_total_limit"]),
        load_best_model_at_end=True,
        metric_for_best_model=config["checkpoint_metric"],
        greater_is_better=False,
        bf16=use_bfloat16,
        fp16=torch.cuda.is_available() and not use_bfloat16,
        report_to="none",
        remove_unused_columns=False,
        group_by_length=True,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=PromptCompletionDataset(
            training_records
        ),
        eval_dataset=PromptCompletionDataset(
            development_records
        ),
        data_collator=collate,
    )

    train_result = trainer.train()

    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    candidate_report = checkpoint_candidates(
        trainer.state.log_history,
        checkpoint_root,
        int(config["shortlist_size"]),
    )
    write_json(
        args.output_dir / "checkpoint_candidates.json",
        candidate_report,
    )

    run_summary = {
        "status": "completed",
        "training_mode": (
            "continued_fine_tuning_from_p1_adapter"
        ),
        "config": config,
        "git": git_provenance(),
        "inputs": {
            "training_data": {
                "path": str(args.training_data),
                "sha256": sha256_file(
                    args.training_data
                ),
            },
            "development_data": {
                "path": str(args.development_data),
                "sha256": sha256_file(
                    args.development_data
                ),
            },
            "initial_p1_adapter": {
                "path": str(args.initial_adapter),
                "file_sha256": directory_file_hashes(
                    args.initial_adapter
                ),
            },
        },
        "examples": {
            "training": len(training_records),
            "development": len(development_records),
            "training_task_counts": task_counts(
                training_records
            ),
            "development_task_counts": task_counts(
                development_records
            ),
            "max_train_examples_override": (
                args.max_train_examples
            ),
            "max_dev_examples_override": (
                args.max_dev_examples
            ),
        },
        "token_lengths": length_report,
        "model": {
            "base_model": config["base_model"],
            "model_revision": config["model_revision"],
            "starting_adapter": config["initial_model"],
            "trainable_parameters": trainable_parameters,
            "total_parameters": total_parameters,
        },
        "runtime": {
            "device": (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            ),
            "dtype": str(model_dtype),
            "cuda_device_count": (
                torch.cuda.device_count()
                if torch.cuda.is_available()
                else 0
            ),
            "cuda_device_name": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "train_metrics": train_result.metrics,
        "best_model_checkpoint": (
            trainer.state.best_model_checkpoint
        ),
        "best_metric": trainer.state.best_metric,
        "log_history": trainer.state.log_history,
        "checkpoint_candidates_file": (
            "checkpoint_candidates.json"
        ),
        "adapter_directory": str(adapter_dir),
    }
    write_json(
        args.output_dir / "run.json",
        run_summary,
    )

    print(f"Saved P2 LoRA adapter to {adapter_dir}")
    print(
        "Saved checkpoint candidates to "
        f"{args.output_dir / 'checkpoint_candidates.json'}"
    )
    print(
        f"Saved training report to "
        f"{args.output_dir / 'run.json'}"
    )


if __name__ == "__main__":
    main()