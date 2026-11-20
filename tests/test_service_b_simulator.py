import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.app.service_b.offline import histories_for_simulator
from src.app.service_b.simulator import load_simulator_metadata
from src.data.build import build
from tests.test_data_pipeline import write_csv


class SimulatorMetadataTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.raw, self.data = root / "raw", root / "data"
        self.raw.mkdir()
        ratings = [(u, (step + (u - 1) * 4) % 30 + 1, 5.0, step * 4 + u)
                   for step in range(30) for u in range(1, 5)]
        ratings[0] = (1, 999, 5.0, 1)  # watched once: outside the recommendation catalog
        write_csv(self.raw / "ratings.csv", ("userId", "movieId", "rating", "timestamp"), ratings)
        write_csv(self.raw / "movies.csv", ("movieId", "title", "genres"),
                  [(i, f"Movie {i} (2000)", "Comedy") for i in list(range(1, 31)) + [999]])
        write_csv(self.raw / "tags.csv", ("userId", "movieId", "tag", "timestamp"), [])
        write_csv(self.raw / "genome-tags.csv", ("tagId", "tag"), [(1, "warm")])
        write_csv(self.raw / "genome-scores.csv", ("movieId", "tagId", "relevance"),
                  [(i, 1, .9) for i in list(range(1, 31)) + [999]])
        build(self.raw, self.data, {"version": "fixture-v1", "minimum_training_ratings_per_user": 20,
            "minimum_training_ratings_per_catalog_movie": 2, "test_quantile": .9,
            "validation_quantile_within_pretest": .9, "genome_tags_per_movie": 1,
            "user_tags_per_movie": 1, "liked_movies_per_profile": 3, "disliked_movies_per_profile": 2})

    def test_out_of_catalog_watched_movie_remains_answerable_without_becoming_retrievable(self):
        metadata = load_simulator_metadata(self.data, self.raw)
        with (self.data / "catalog.csv").open() as source:
            catalog = {int(row["movie_id"]) for row in csv.DictReader(source)}
        self.assertIn(999, metadata)
        self.assertNotIn(999, catalog)
        with (self.data / "train_ratings.csv").open() as source:
            ids = {int(r["movie_id"]) for r in csv.DictReader(source) if int(r["user_id"]) == 1}
        profiles = {1: {"user_id": 1, "text": "Comedy", "history_movie_ids": sorted(ids), "liked_movies": []}}
        facts = histories_for_simulator({"profiles": self.data / "validation_profiles.jsonl",
            "dataset_manifest": self.data / "manifest.json"}, [1], "validation", profiles, catalog, self.raw)
        self.assertTrue(any(f["movie_id"] == 999 for f in facts[1]["liked_example"]))

    def test_mismatched_raw_metadata_fails_before_simulation(self):
        with (self.raw / "movies.csv").open("a") as out:
            out.write("1000,Injected movie,Drama\n")
        with self.assertRaisesRegex(ValueError, "checksum mismatch"):
            load_simulator_metadata(self.data, self.raw)


if __name__ == "__main__":
    unittest.main()
