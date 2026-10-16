"""Fine-tune Qwen for P1 pointwise relevance scoring with LoRA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from .config import load_config
from .training_data import load_training_examples, to_sft_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune the P1 Qwen reranker with LoRA."
    )
    parser.add_argument("--training-data", required=True, type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/p1_qwen_lora_v1.json"),
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional limit for a documented smoke test.",
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training without CUDA. This will be very slow.",
    )
    return parser.parse_args()


def split_records(
    records: list[dict[str, str]],
    validation_fraction: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Create a deterministic train/development split."""
    if len(records) < 2:
        raise ValueError("At least two examples are required for training")

    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)

    validation_size = max(1, round(len(shuffled) * validation_fraction))
    validation_size = min(validation_size, len(shuffled) - 1)

    validation_records = shuffled[:validation_size]
    training_records = shuffled[validation_size:]
    return training_records, validation_records


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}"
        )

    examples = load_training_examples(args.training_data)

    if args.max_examples is not None:
        if args.max_examples < 2:
            raise ValueError("--max-examples must be at least 2")
        examples = examples[: args.max_examples]

    records = to_sft_records(examples)
    training_records, validation_records = split_records(
        records,
        float(config["validation_fraction"]),
        int(config["seed"]),
    )

    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from torch.utils.data import Dataset
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as error:
        raise SystemExit(
            "Reranking dependencies are missing. Install them with "
            "'python -m pip install -r requirements-reranking.txt'."
        ) from error

    if not torch.cuda.is_available() and not args.allow_cpu:
        raise SystemExit(
            "CUDA GPU not detected. Run this on GPU hardware, or pass "
            "--allow-cpu only if you intentionally accept very slow training."
        )

    set_seed(int(config["seed"]))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        revision=config["model_revision"],
        use_fast=True,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    max_length = int(config["max_length"])

    class PromptCompletionDataset(Dataset):
        def __init__(self, items: list[dict[str, str]]) -> None:
            self.items = items

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int) -> dict[str, list[int]]:
            item = self.items[index]
            completion = item["completion"] + tokenizer.eos_token

            prompt_ids = tokenizer(
                item["prompt"],
                add_special_tokens=True,
            )["input_ids"]
            completion_ids = tokenizer(
                completion,
                add_special_tokens=False,
            )["input_ids"]

            available_prompt_length = max_length - len(completion_ids)
            if available_prompt_length <= 0:
                raise ValueError(
                    "The completion is longer than the configured max_length"
                )

            prompt_ids = prompt_ids[-available_prompt_length:]
            input_ids = prompt_ids + completion_ids
            labels = [-100] * len(prompt_ids) + completion_ids

            return {
                "input_ids": input_ids,
                "attention_mask": [1] * len(input_ids),
                "labels": labels,
            }

    def collate(
        batch: list[dict[str, list[int]]],
    ) -> dict[str, Any]:
        longest = max(len(item["input_ids"]) for item in batch)

        input_ids = []
        attention_masks = []
        labels = []

        for item in batch:
            padding = longest - len(item["input_ids"])
            input_ids.append(
                item["input_ids"] + [tokenizer.pad_token_id] * padding
            )
            attention_masks.append(
                item["attention_mask"] + [0] * padding
            )
            labels.append(item["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(
                attention_masks,
                dtype=torch.long,
            ),
            "labels": torch.tensor(labels, dtype=torch.long),
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

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        revision=config["model_revision"],
        torch_dtype=model_dtype,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type="CAUSAL_LM",
        r=int(config["lora_r"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        target_modules=list(config["target_modules"]),
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    training_arguments = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=use_bfloat16,
        fp16=torch.cuda.is_available() and not use_bfloat16,
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=PromptCompletionDataset(training_records),
        eval_dataset=PromptCompletionDataset(validation_records),
        data_collator=collate,
    )

    train_result = trainer.train()

    adapter_dir = args.output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(adapter_dir)

    run_summary = {
        "status": "completed",
        "config": config,
        "training_data": str(args.training_data),
        "examples_total": len(records),
        "examples_training": len(training_records),
        "examples_validation": len(validation_records),
        "max_examples_override": args.max_examples,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "dtype": str(model_dtype),
        "train_metrics": train_result.metrics,
        "log_history": trainer.state.log_history,
        "adapter_directory": str(adapter_dir),
    }
    write_json(args.output_dir / "run.json", run_summary)

    print(f"Saved LoRA adapter to {adapter_dir}")
    print(f"Saved training report to {args.output_dir / 'run.json'}")


if __name__ == "__main__":
    main()