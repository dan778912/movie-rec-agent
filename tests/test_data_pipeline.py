import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.data.build import build
from src.data.verify import verify


def write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(header)
        writer.writerows(rows)


class DataPipelineTest(unittest.TestCase):
    def test_global_cutoffs_profiles_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            raw.mkdir()
            ratings = []
            for step in range(30):
                for user_id in range(1, 5):
                    movie_id = (step + (user_id - 1) * 4) % 30 + 1
                    rating = 5.0 if step % 3 != 0 else 2.0
                    ratings.append((user_id, movie_id, rating, step * 4 + user_id))
            write_csv(raw / "ratings.csv", ("userId", "movieId", "rating", "timestamp"), ratings)
            write_csv(raw / "movies.csv", ("movieId", "title", "genres"),
                      [(i, f"Movie {i} (2000)", "Drama|Comedy") for i in range(1, 31)])
            write_csv(raw / "tags.csv", ("userId", "movieId", "tag", "timestamp"),
                      [(1, 1, "funny", 3), (2, 1, "future secret", 120)])
            write_csv(raw / "genome-tags.csv", ("tagId", "tag"), [(1, "warm"), (2, "dark")])
            write_csv(raw / "genome-scores.csv", ("movieId", "tagId", "relevance"),
                      [(i, j, 0.9 if j == 1 else 0.1) for i in range(1, 31) for j in (1, 2)])
            config = {
                "version": "fixture-v1", "minimum_training_ratings_per_user": 20,
                "minimum_training_ratings_per_catalog_movie": 2,
                "test_quantile": 0.9, "validation_quantile_within_pretest": 0.9,
                "genome_tags_per_movie": 1, "user_tags_per_movie": 1,
                "liked_movies_per_profile": 3, "disliked_movies_per_profile": 2,
            }
            output = root / "processed"
            stats = build(raw, output, config)
            self.assertEqual(stats["source_ratings"], 120)
            self.assertEqual(stats["eligible_users"], 4)
            self.assertGreater(stats["evaluation"]["test"]["users_with_in_catalog_positive"], 0)
            self.assertEqual(verify(output)["version"], "fixture-v1")
            first_hashes = json.loads((output / "manifest.json").read_text())["file_sha256"]
            self.assertEqual(len(first_hashes), 10)
            self.assertFalse(any(name.startswith("_build.sqlite3") for name in first_hashes))
            build(raw, output, config)
            self.assertEqual(first_hashes, json.loads((output / "manifest.json").read_text())["file_sha256"])
            with (output / "test_profiles.jsonl").open(encoding="utf-8") as source:
                for line in source:
                    profile = json.loads(line)
                    self.assertLess(profile["history_latest_timestamp"], stats["test_cutoff_unix"])
                    self.assertNotIn("future secret", profile["text"])


if __name__ == "__main__":
    unittest.main()
