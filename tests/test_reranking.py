"""Synthetic tests for the Member 6 P1 reranking pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.reranking.config import load_config
from src.reranking.inference_data import (
    load_candidate_pools,
    load_movie_documents,
    load_profiles,
)
from src.reranking.reranker import (
    KeywordOverlapScorer,
    rerank_candidate_pools,
)
from src.reranking.train_qwen import split_records
from src.reranking.training_data import (
    load_training_examples,
    to_sft_records,
)


def write_jsonl(path: Path, records: list[dict]) -> None:
    """Write synthetic JSONL fixture records."""
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record))
            file.write("\n")


class RerankingTests(unittest.TestCase):
    def test_member3_training_examples_convert_to_sft(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "training.jsonl"
            write_jsonl(
                path,
                [
                    {
                        "example_id": "1:10:100",
                        "user_id": 1,
                        "movie_id": 10,
                        "split": "train",
                        "profile": "Likes science fiction and space movies.",
                        "candidate_document": (
                            "Space Journey. Genres: Science Fiction."
                        ),
                        "label": "relevant",
                    },
                    {
                        "example_id": "1:20:101",
                        "user_id": 1,
                        "movie_id": 20,
                        "split": "train",
                        "profile": "Likes science fiction and space movies.",
                        "candidate_document": (
                            "Quiet Romance. Genres: Romance."
                        ),
                        "label": "not_relevant",
                    },
                ],
            )

            examples = load_training_examples(path)
            sft_records = to_sft_records(examples)

            self.assertEqual(len(sft_records), 2)
            self.assertEqual(sft_records[0]["completion"], "relevant")
            self.assertEqual(
                sft_records[1]["completion"],
                "not_relevant",
            )
            self.assertIn(
                "Likes science fiction",
                sft_records[0]["prompt"],
            )

    def test_reranker_reorders_but_preserves_candidate_movies(self) -> None:
        profiles = {
            1: {
                "user_id": 1,
                "split": "validation",
                "text": "Likes space science fiction adventures.",
                "history_movie_ids": [99],
            }
        }

        movies = {
            10: {
                "movie_id": 10,
                "text": "A romantic historical drama.",
            },
            20: {
                "movie_id": 20,
                "text": "A space science fiction adventure.",
            },
            30: {
                "movie_id": 30,
                "text": "A courtroom mystery.",
            },
        }

        pools = [
            {
                "user_id": 1,
                "split": "validation",
                "method": "embedding",
                "recommendations": [
                    {
                        "movie_id": 10,
                        "score": 0.9,
                        "rank": 1,
                    },
                    {
                        "movie_id": 30,
                        "score": 0.8,
                        "rank": 2,
                    },
                    {
                        "movie_id": 20,
                        "score": 0.7,
                        "rank": 3,
                    },
                ],
            }
        ]

        predictions = rerank_candidate_pools(
            pools,
            profiles,
            movies,
            KeywordOverlapScorer(),
        )

        recommendations = predictions[0]["recommendations"]
        output_movies = [
            item["movie_id"]
            for item in recommendations
        ]

        self.assertEqual(predictions[0]["method"], "p1")
        self.assertEqual(output_movies[0], 20)
        self.assertEqual(set(output_movies), {10, 20, 30})
        self.assertEqual(
            [item["rank"] for item in recommendations],
            [1, 2, 3],
        )
        self.assertTrue(
            all(item["explanation"] for item in recommendations)
        )

    def test_loaders_accept_matching_synthetic_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            profiles_path = root / "profiles.jsonl"
            movies_path = root / "movies.jsonl"
            candidates_path = root / "candidates.jsonl"

            write_jsonl(
                profiles_path,
                [
                    {
                        "user_id": 1,
                        "split": "validation",
                        "text": "Likes comedy.",
                        "history_movie_ids": [5],
                    }
                ],
            )

            write_jsonl(
                movies_path,
                [
                    {
                        "movie_id": 10,
                        "text": "A comedy movie.",
                    },
                    {
                        "movie_id": 20,
                        "text": "A drama movie.",
                    },
                ],
            )

            write_jsonl(
                candidates_path,
                [
                    {
                        "user_id": 1,
                        "split": "validation",
                        "method": "embedding",
                        "recommendations": [
                            {
                                "movie_id": 10,
                                "score": 0.9,
                                "rank": 1,
                            },
                            {
                                "movie_id": 20,
                                "score": 0.8,
                                "rank": 2,
                            },
                        ],
                    }
                ],
            )

            profiles = load_profiles(profiles_path)
            movies = load_movie_documents(movies_path)
            pools = load_candidate_pools(candidates_path)

            self.assertIn(1, profiles)
            self.assertEqual(set(movies), {10, 20})
            self.assertEqual(pools[0]["user_id"], 1)

    def test_duplicate_candidate_movie_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad-candidates.jsonl"

            write_jsonl(
                path,
                [
                    {
                        "user_id": 1,
                        "split": "validation",
                        "method": "embedding",
                        "recommendations": [
                            {
                                "movie_id": 10,
                                "score": 0.9,
                                "rank": 1,
                            },
                            {
                                "movie_id": 10,
                                "score": 0.8,
                                "rank": 2,
                            },
                        ],
                    }
                ],
            )

            with self.assertRaisesRegex(ValueError, "duplicate movie"):
                load_candidate_pools(path)

    def test_qwen_lora_config_is_valid(self) -> None:
        config = load_config("configs/p1_qwen_lora_v1.json")

        self.assertEqual(
            config["base_model"],
            "Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.assertEqual(config["seed"], 20261001)
        self.assertGreater(config["lora_r"], 0)

    def test_training_split_is_deterministic(self) -> None:
        records = [
            {
                "prompt": f"Example {index}",
                "completion": "relevant",
            }
            for index in range(20)
        ]

        first_train, first_validation = split_records(
            records,
            validation_fraction=0.1,
            seed=20261001,
        )
        second_train, second_validation = split_records(
            records,
            validation_fraction=0.1,
            seed=20261001,
        )

        self.assertEqual(first_train, second_train)
        self.assertEqual(first_validation, second_validation)
        self.assertEqual(len(first_train), 18)
        self.assertEqual(len(first_validation), 2)


if __name__ == "__main__":
    unittest.main()