import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.data.build import sha256, write_jsonl
from src.retrieval.bm25 import BM25Index, document_text, tokenize
from src.retrieval.run_bm25 import (
    diagnostic_metrics, index_command, load_profiles, predict, tune, validate_record, write_json,
)


class BM25ScoringTest(unittest.TestCase):
    def test_hand_calculated_scores_and_binary_query_frequency(self):
        index = BM25Index.build([
            {"movie_id": 3, "text": "space space"},
            {"movie_id": 1, "text": "space comedy"},
            {"movie_id": 2, "text": "romance"},
        ], ["text"])
        ranked = index.search("space", set(), 3, k1=1.2, b=0.75)
        idf = math.log(1 + 1.5 / 2.5)
        expected = idf * (2 * 2.2) / (2 + 1.2 * (0.25 + 0.75 * 2 / (5 / 3)))
        self.assertEqual([item["movie_id"] for item in ranked], [3, 1, 2])
        self.assertAlmostEqual(ranked[0]["score"], expected)
        self.assertEqual(ranked, index.search("space space space", set(), 3))

    def test_watched_filter_ties_and_zero_match_fallback(self):
        index = BM25Index.build([
            {"movie_id": i, "text": "space"} for i in (8, 2, 4)
        ], ["text"])
        self.assertEqual([r["movie_id"] for r in index.search("space", {2, 999}, 100)], [4, 8])
        for query in ("", "missing", "The movies 5/5"):
            ranked = index.search(query, {4}, 100)
            self.assertEqual([r["movie_id"] for r in ranked], [2, 8])
            self.assertTrue(all(r["score"] == 0 for r in ranked))
        self.assertEqual(index.search("space", {2, 4, 8}, 10), [])

    def test_unicode_tokenizer_and_explicit_document_fields(self):
        self.assertEqual(tokenize("HIGHLY rated Movies: Amélie (2001); SCI-fi 4.5/5"),
                         ["amélie", "sci", "fi"])
        doc = {"title": "Space", "genres": ["Drama"], "genome_tags": ["space travel"],
               "user_tags": [], "text": "Space. Genres: Drama. Genome tags: space travel."}
        self.assertEqual(document_text(doc, ["genres", "genome_tags"]), "Drama space travel")
        with self.assertRaises(ValueError):
            document_text(doc, ["text", "title"])
        with self.assertRaises(ValueError):
            document_text(doc, ["plot"])

    def test_empty_documents_and_parameter_validation(self):
        index = BM25Index.build([{"movie_id": 1, "text": ""}], ["text"])
        self.assertEqual(index.search("space", set())[0]["score"], 0)
        for k1, b in ((0, 0.5), (-1, 0), (1, 1.1), (float("nan"), 0), (1, float("inf"))):
            with self.assertRaises(ValueError):
                index.search("space", set(), k1=k1, b=b)
        with self.assertRaises(ValueError):
            index.search("space", set(), 0)
        with self.assertRaises(ValueError):
            BM25Index.build([{"movie_id": 1, "text": "x"}] * 2, ["text"])

    def test_json_index_roundtrip_preserves_ranking(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "index.json"
            index = BM25Index.build([{"movie_id": i, "text": text}
                                     for i, text in ((1, "space travel"), (3, "romance"), (2, "space"))],
                                    ["text"])
            index.save(path)
            self.assertEqual(index.search("space travel", {1}),
                             BM25Index.load(path).search("space travel", {1}))


class BM25DiagnosticTest(unittest.TestCase):
    def test_metrics_against_hand_calculated_rankings(self):
        self.assertEqual(diagnostic_metrics([1, 2], {1, 2})["ndcg@10"], 1)
        bad = diagnostic_metrics([9, 8], {1, 2})
        self.assertTrue(all(value == 0 for value in bad.values()))
        partial = diagnostic_metrics([9, 1], {1, 2})
        self.assertEqual(partial["recall@10"], 0.5)
        self.assertAlmostEqual(partial["ndcg@10"], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
        self.assertEqual(diagnostic_metrics(list(range(1, 101)), {20, 50, 100}),
                         {"recall@10": 0, "recall@20": 1 / 3, "recall@50": 2 / 3,
                          "recall@100": 1, "ndcg@10": 0})

    def test_prediction_validator_rejects_watched_duplicate_invalid_rank_and_score(self):
        profile = {"user_id": 1, "split": "validation", "history_movie_ids": [1]}
        valid = {"user_id": 1, "split": "validation", "method": "bm25",
                 "recommendations": [{"movie_id": 2, "score": 1.0, "rank": 1},
                                     {"movie_id": 3, "score": 0.0, "rank": 2}]}
        validate_record(valid, profile, {1, 2, 3}, 2)
        for key, value in (("movie_id", 1), ("movie_id", 2), ("movie_id", 99),
                           ("rank", 1), ("score", float("nan")), ("score", 2)):
            record = json.loads(json.dumps(valid))
            record["recommendations"][1][key] = value
            with self.assertRaises(ValueError):
                validate_record(record, profile, {1, 2, 3}, 2)


class BM25WorkflowTest(unittest.TestCase):
    def fixture(self, root):
        root.mkdir()
        documents = [{"movie_id": i, "title": title, "genres": [genre], "genome_tags": [tag],
                      "user_tags": [], "text": f"{title}. Genres: {genre}. Genome tags: {tag}."}
                     for i, title, genre, tag in ((1, "Space", "Sci-Fi", "moon"),
                                                   (2, "Moon", "Sci-Fi", "space"),
                                                   (3, "Love", "Romance", "romantic"),
                                                   (4, "Stars", "Sci-Fi", "space"))]
        write_jsonl(root / "movie_documents.jsonl", documents)
        (root / "catalog.csv").write_text("movie_id\n1\n2\n3\n4\n", encoding="utf-8")
        for split in ("validation", "test"):
            write_jsonl(root / f"{split}_profiles.jsonl", [
                {"user_id": i, "split": split, "text": text, "history_movie_ids": [watched]}
                for i, text, watched in ((5, "space moon", 1), (8, "romantic", 2))])
        (root / "validation_targets.csv").write_text(
            "user_id,movie_id,rating,timestamp,in_catalog\n5,2,5,100,1\n8,3,4,100,1\n8,99,5,100,0\n",
            encoding="utf-8")
        # No test targets or rating files: this workflow must never require them.
        write_json(root / "manifest.json", {"version": "fixture-v1", "file_sha256": {
            path.name: sha256(path) for path in root.iterdir()}})

    def test_tuning_export_reload_and_unseen_test_predictions_without_test_targets(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = root / "data"
            self.fixture(data)
            grid = root / "grid.json"
            write_json(grid, {"seed": 7, "validation_users": 2, "top_k": 100,
                              "objective": "ndcg@10", "k1": [0.6, 1.2], "b": [0.75],
                              "fields": [["text"], ["title", "genres"]]})
            config = root / "frozen.json"
            args = SimpleNamespace(data=data, grid=grid, output_dir=root / "tune",
                                   user_ids=None, frozen_config=config)
            tune(args)
            first_config = config.read_bytes()
            tune(args)
            self.assertEqual(config.read_bytes(), first_config)
            index_path = root / "index.json"
            index_command(SimpleNamespace(data=data, config=config, output=index_path))
            for split in ("validation", "test"):
                output = root / f"{split}.jsonl"
                predict_args = SimpleNamespace(data=data, split=split, config=config, index=index_path,
                                               user_ids=None, output=output, report_dir=root / split)
                predict(predict_args)
                first = output.read_bytes()
                predict(predict_args)
                self.assertEqual(first, output.read_bytes())
                records = [json.loads(line) for line in output.read_text().splitlines()]
                self.assertEqual([r["user_id"] for r in records], [5, 8])
                summary = json.loads((root / split / "summary.json").read_text())
                self.assertEqual(summary["invalid_predictions"], 0)
                self.assertEqual("diagnostic_metrics" in summary, split == "validation")
            # Configuration/index mismatch must fail instead of silently changing retrieval.
            frozen = json.loads(config.read_text())
            frozen["fields"] = ["user_tags"]
            write_json(config, frozen)
            with self.assertRaisesRegex(ValueError, "Index and frozen configuration differ"):
                predict(predict_args)

    def test_input_checksum_mismatch_and_missing_subset_user_fail(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(temp)
            data = root / "data"
            self.fixture(data)
            subset = root / "users.json"
            write_json(subset, [99])
            with self.assertRaisesRegex(ValueError, "missing profile users"):
                load_profiles(data, "validation", subset)
            (data / "movie_documents.jsonl").write_text("{}\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                tune(SimpleNamespace(data=data))


if __name__ == "__main__":
    unittest.main()
