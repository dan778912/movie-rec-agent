"""Generate and validate raw P2 action outputs from saved samples."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .p2_generator import (
    P2ActionGenerator,
    P2GenerationError,
    validate_generated_action,
)
from .p2_training_data import (
    P2_TASKS,
    load_p2_samples,
    sha256_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reload a P2 adapter and generate raw outputs for "
            "the six structured actions."
        )
    )
    parser.add_argument(
        "--samples",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--split",
        choices=("train", "dev"),
        required=True,
    )
    parser.add_argument(
        "--adapter",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )
    parser.add_argument(
        "--examples-per-task",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--base-model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
    )
    parser.add_argument(
        "--model-revision",
        default=(
            "775b11afaf83e0dc75bd5abaf90133e47b3ec082"
        ),
    )
    parser.add_argument(
        "--prompt-version",
        default="p2-prompt-v1",
    )
    parser.add_argument(
        "--max-input-tokens",
        type=int,
        default=6144,
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=1024,
    )
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
    )
    return parser.parse_args()


def select_examples(
    records: list[dict[str, Any]],
    examples_per_task: int,
) -> list[dict[str, Any]]:
    """Select a deterministic prefix for every P2 task."""
    if examples_per_task <= 0:
        raise ValueError(
            "examples_per_task must be positive"
        )

    counts: Counter[str] = Counter()
    selected = []

    for record in records:
        task = record["task"]

        if counts[task] >= examples_per_task:
            continue

        selected.append(record)
        counts[task] += 1

        if all(
            counts[required] >= examples_per_task
            for required in P2_TASKS
        ):
            break

    missing = {
        task: examples_per_task - counts[task]
        for task in P2_TASKS
        if counts[task] < examples_per_task
    }

    if missing:
        raise ValueError(
            f"Not enough samples for every task: {missing}"
        )

    return selected


def write_json(path: Path, value: Any) -> None:
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


def adapter_hashes(path: Path) -> dict[str, str]:
    """Hash the adapter and tokenizer files in the artifact root."""
    return {
        item.name: sha256_file(item)
        for item in sorted(path.iterdir())
        if item.is_file()
    }


def main() -> None:
    args = parse_args()

    if args.output.exists():
        raise FileExistsError(
            f"Output already exists: {args.output}"
        )

    manifest_path = args.output.with_suffix(
        args.output.suffix + ".manifest.json"
    )

    if manifest_path.exists():
        raise FileExistsError(
            f"Manifest already exists: {manifest_path}"
        )

    records = load_p2_samples(
        args.samples,
        expected_split=args.split,
        expected_prompt_version=args.prompt_version,
    )
    selected = select_examples(
        records,
        args.examples_per_task,
    )

    generator = P2ActionGenerator(
        args.adapter,
        base_model=args.base_model,
        model_revision=args.model_revision,
        max_input_tokens=args.max_input_tokens,
        max_new_tokens=args.max_new_tokens,
        allow_cpu=args.allow_cpu,
    )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    temporary = args.output.with_name(
        args.output.name + ".tmp"
    )

    validity: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    started = time.perf_counter()

    try:
        with temporary.open(
            "w",
            encoding="utf-8",
            newline="\n",
        ) as file:
            for index, record in enumerate(
                selected,
                start=1,
            ):
                prompt_tokens = generator.count_input_tokens(
                    record["prompt"]
                )
                call_started = time.perf_counter()
                raw_output = generator.generate(
                    record["prompt"]
                )
                latency_ms = (
                    time.perf_counter() - call_started
                ) * 1000.0

                valid = True
                validation_error = None
                parsed_action = None

                try:
                    parsed_action = validate_generated_action(
                        raw_output,
                        expected_action=record["task"],
                    )
                except P2GenerationError as error:
                    valid = False
                    validation_error = str(error)

                result = {
                    "sample_id": record["sample_id"],
                    "user_id": record["user_id"],
                    "task": record["task"],
                    "prompt_tokens": prompt_tokens,
                    "raw_output": raw_output,
                    "valid": valid,
                    "validation_error": validation_error,
                    "parsed_action": parsed_action,
                    "latency_ms": latency_ms,
                }
                file.write(
                    json.dumps(
                        result,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                file.write("\n")
                file.flush()

                validity[
                    "valid" if valid else "invalid"
                ] += 1
                task_counts[record["task"]] += 1

                print(
                    f"[{index}/{len(selected)}] "
                    f"{record['task']}: "
                    f"{'valid' if valid else 'invalid'}",
                    flush=True,
                )

        os.replace(temporary, args.output)
    finally:
        if temporary.exists():
            temporary.unlink()

    total_seconds = time.perf_counter() - started

    manifest = {
        "schema_version": "p2-raw-output-manifest-v1",
        "samples": {
            "path": str(args.samples),
            "sha256": sha256_file(args.samples),
            "split": args.split,
            "examples_per_task": args.examples_per_task,
        },
        "adapter": {
            "path": str(args.adapter),
            "file_sha256": adapter_hashes(args.adapter),
        },
        "generation": generator.settings(),
        "records": len(selected),
        "task_counts": {
            task: task_counts.get(task, 0)
            for task in P2_TASKS
        },
        "validity": {
            "valid": validity.get("valid", 0),
            "invalid": validity.get("invalid", 0),
        },
        "total_seconds": total_seconds,
        "output": {
            "path": str(args.output),
            "sha256": sha256_file(args.output),
        },
    }
    write_json(manifest_path, manifest)

    print(
        f"Wrote {len(selected)} raw P2 outputs to "
        f"{args.output}"
    )
    print(
        f"Valid: {validity.get('valid', 0)}; "
        f"invalid: {validity.get('invalid', 0)}"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()