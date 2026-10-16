"""Run the P1 reranking pipeline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from .inference_data import (
    load_candidate_pools,
    load_movie_documents,
    load_profiles,
)
from .qwen_scorer import QwenLoRAScorer
from .reranker import KeywordOverlapScorer, rerank_candidate_pools


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    """Write deterministic JSONL output atomically."""
    if path.exists():
        raise FileExistsError(
            f"Output already exists: {path}. Choose a new path."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")

    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            for record in records:
                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
                file.write("\n")

        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rerank frozen embedding candidates for P1."
    )
    parser.add_argument("--profiles", required=True, type=Path)
    parser.add_argument("--movies", required=True, type=Path)
    parser.add_argument("--candidates", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)

    parser.add_argument(
        "--split",
        choices=("validation", "test"),
        default="validation",
    )
    parser.add_argument(
        "--adapter",
        type=Path,
        default=None,
        help="Path to the trained Qwen LoRA adapter.",
    )
    parser.add_argument(
        "--smoke-keyword-scorer",
        action="store_true",
        help=(
            "Use the lightweight keyword scorer for pipeline testing only. "
            "This is not the official P1 model."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.adapter is not None and args.smoke_keyword_scorer:
        raise SystemExit(
            "Choose either --adapter or --smoke-keyword-scorer, not both."
        )

    if args.adapter is not None:
        scorer = QwenLoRAScorer(args.adapter)
    elif args.smoke_keyword_scorer:
        scorer = KeywordOverlapScorer()
    else:
        raise SystemExit(
            "Select a trained model with --adapter, or use "
            "--smoke-keyword-scorer for a pipeline-only test."
        )

    profiles = load_profiles(args.profiles)
    movies = load_movie_documents(args.movies)
    candidates = load_candidate_pools(
        args.candidates,
        expected_split=args.split,
    )

    predictions = rerank_candidate_pools(
        candidate_pools=candidates,
        profiles=profiles,
        movie_documents=movies,
        scorer=scorer,
    )

    write_jsonl(args.output, predictions)
    print(
        f"Wrote {len(predictions)} P1 prediction records "
        f"to {args.output}"
    )


if __name__ == "__main__":
    main()