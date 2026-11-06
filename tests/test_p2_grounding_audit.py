"""Manual explanation audit preparation on a complete synthetic P2 export."""

import contextlib
import csv
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.evaluation.p2_grounding_audit import main, prepare_audit, write_audit
from src.evaluation.validate import PredictionValidationError


FIXTURES = Path(__file__).resolve().parent / "fixtures" / "evaluation_validation"


def _jsonl(path, records):
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records), encoding="utf-8")


class P2GroundingAuditTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.work = Path(temporary.name)
        self.catalog = self.work / "catalog.csv"
        movie_genres = ((10, "Comedy"), (20, "Drama"), (30, "Comedy"),
                        (40, "Drama"), (50, "Comedy"), (60, "Action"),
                        (70, "Drama"), (80, "Action"), (90, "Comedy"))
        self.catalog.write_text(
            "movie_id,title,genres\n" + "".join(
                f"{movie_id},Synthetic Movie {movie_id},{genre}\n"
                for movie_id, genre in movie_genres
            ), encoding="utf-8"
        )
        self.movie_documents = self.work / "movie_documents.jsonl"
        self.document_rows = [
            {"movie_id": movie_id, "title": f"Synthetic Movie {movie_id}",
             "genres": [genre], "genome_tags": ["family saga"] if movie_id == 70 else [],
             "user_tags": ["classic"] if movie_id == 70 else [],
             "text": f"Synthetic Movie {movie_id}. Genres: {genre}."
                     + (" Genome tags: family saga. User tags: classic." if movie_id == 70 else "")}
            for movie_id, genre in movie_genres
        ]
        _jsonl(self.movie_documents, self.document_rows)
        self.profiles = self.work / "profiles.jsonl"
        self.profile_rows = [
            {"user_id": 1, "split": "validation", "history_movie_ids": [90],
             "text": "Liked a comedy."},
            {"user_id": 2, "split": "validation", "history_movie_ids": [80],
             "text": "Liked a drama."},
        ]
        _jsonl(self.profiles, self.profile_rows)
        self.users = FIXTURES / "users.json"
        self.pool = FIXTURES / "embedding_valid.jsonl"
        self.predictions = self.work / "p2.jsonl"
        self.trace = self.work / "trace.jsonl"
        self.prediction_rows = [
            {"user_id": user_id, "split": "validation", "method": "p2",
             "recommendations": [
                 {"movie_id": movie_id, "score": 4 - rank, "rank": rank}
                 for rank, movie_id in enumerate(ids, start=1)
             ]}
            for user_id, ids in ((1, [10, 20, 30]), (2, [70, 40, 50]))
        ]
        self.trace_rows = [
            {"schema_version": "p2-trace-v1", "user_id": 1, "split": "validation",
             "method": "p2", "initial_candidate_movie_ids": [10, 20, 30],
             "events": [
                 {"action": "assess_sufficiency", "sufficient": True},
                 {"action": "evaluate_candidates", "candidate_movie_ids": [10, 20, 30],
                  "pool_sufficient": True},
                 {"action": "rank_candidates", "movie_ids": [10, 20, 30]},
                 {"action": "explain_recommendations", "explanations": [
                     {"movie_id": 10, "text": "You liked comedy."},
                     {"movie_id": 20, "text": "This is a drama."},
                     {"movie_id": 30, "text": "This has comedy."},
                 ]},
             ],
             "final_candidate_movie_ids": [10, 20, 30], "termination_reason": "recommend"},
            {"schema_version": "p2-trace-v1", "user_id": 2, "split": "validation",
             "method": "p2", "initial_candidate_movie_ids": [40, 50, 60],
             "events": [
                 {"action": "assess_sufficiency", "sufficient": False},
                 {"action": "ask_clarification", "question": "Preferred genre?",
                  "answer": "Drama", "answer_source": "synthetic_pre_cutoff_history"},
                 {"action": "assess_sufficiency", "sufficient": True},
                 {"action": "evaluate_candidates", "candidate_movie_ids": [40, 50, 60],
                  "pool_sufficient": False},
                 {"action": "retrieve_candidates", "added_movie_ids": [70],
                  "index_version": "synthetic-index-v1"},
                 {"action": "evaluate_candidates", "candidate_movie_ids": [40, 50, 60, 70],
                  "pool_sufficient": True},
                 {"action": "rank_candidates", "movie_ids": [70, 40, 50]},
                 {"action": "explain_recommendations", "explanations": [
                     {"movie_id": 70, "text": "This is a drama."},
                     {"movie_id": 40, "text": "You asked for drama."},
                     {"movie_id": 50, "text": "This is a comedy."},
                 ]},
             ],
             "final_candidate_movie_ids": [40, 50, 60, 70], "termination_reason": "recommend"},
        ]
        self.save()

    def save(self):
        _jsonl(self.predictions, self.prediction_rows)
        _jsonl(self.trace, self.trace_rows)

    def prepare(self, **kwargs):
        return prepare_audit(
            self.catalog, self.movie_documents, self.profiles, self.users, "validation", self.pool,
            self.predictions, self.trace, sample_size=kwargs.get("sample_size", 2),
            seed=kwargs.get("seed", 20261002), required_count=3,
        )

    def test_complete_top_three_with_review_fields_and_provenance(self):
        audit = self.prepare()
        self.assertEqual(audit.manifest["review_row_count"], 6)
        self.assertEqual(set(audit.manifest["selected_user_ids_in_selection_order"]), {1, 2})
        self.assertEqual(audit.manifest["judgment_status"], "unreviewed")
        by_user_rank = {(row["user_id"], row["rank"]): row for row in audit.rows}
        added = by_user_rank[(2, 1)]
        self.assertEqual((added["movie_id"], added["initial_pool_member"],
                          added["retrieved_pool_member"], added["final_pool_member"]),
                         (70, 0, 1, 1))
        self.assertEqual(added["catalog_genres"], "Drama")
        self.assertEqual(added["movie_document_text"],
                         "Synthetic Movie 70. Genres: Drama. Genome tags: family saga. User tags: classic.")
        self.assertEqual(json.loads(added["genome_tags_json"]), ["family saga"])
        self.assertEqual(json.loads(added["user_tags_json"]), ["classic"])
        self.assertEqual(added["visible_profile_text"], "Liked a drama.")
        self.assertEqual(json.loads(added["clarification_turns_json"])[0]["answer"], "Drama")
        self.assertEqual(added["grounding_label"], "")
        self.assertEqual(by_user_rank[(1, 1)]["retrieved_pool_member"], 0)

    def test_seeded_selection_depends_only_on_user_ids_and_seed(self):
        first = self.prepare(sample_size=1)
        self.prediction_rows.reverse()
        self.trace_rows.reverse()
        self.save()
        second = self.prepare(sample_size=1)
        self.assertEqual(first.manifest["selected_user_ids_in_selection_order"],
                         second.manifest["selected_user_ids_in_selection_order"])
        self.assertEqual([row["movie_id"] for row in first.rows],
                         [row["movie_id"] for row in second.rows])

    def test_cli_immutable_package_and_hash(self):
        output = self.work / "audit-package"
        args = ["--catalog", str(self.catalog), "--movie-documents", str(self.movie_documents),
                "--profiles", str(self.profiles),
                "--users", str(self.users), "--split", "validation",
                "--candidate-pool", str(self.pool), "--predictions", str(self.predictions),
                "--trace", str(self.trace), "--required-count", "3", "--sample-size", "2",
                "--output-dir", str(output)]
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertEqual({item.name for item in output.iterdir()}, {"manifest.json", "review.csv"})
        manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["review_csv_sha256"],
                         hashlib.sha256((output / "review.csv").read_bytes()).hexdigest())
        self.assertEqual(manifest["source_sha256"]["movie_documents"],
                         hashlib.sha256(self.movie_documents.read_bytes()).hexdigest())
        with (output / "review.csv").open(encoding="utf-8", newline="") as source:
            rows = list(csv.DictReader(source))
        self.assertEqual(len(rows), 6)
        before = {item.name: item.read_bytes() for item in output.iterdir()}
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(args), 0)
        self.assertEqual(before, {item.name: item.read_bytes() for item in output.iterdir()})
        (output / "review.csv").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "existing audit differs"):
            write_audit(output, self.prepare())

    def test_invalid_trace_and_missing_profile_text_fail_closed(self):
        self.trace_rows[1]["initial_candidate_movie_ids"] = [50, 40, 60]
        self.save()
        with self.assertRaises(PredictionValidationError):
            self.prepare()
        self.trace_rows[1]["initial_candidate_movie_ids"] = [40, 50, 60]
        self.save()
        self.profile_rows[1].pop("text")
        _jsonl(self.profiles, self.profile_rows)
        with self.assertRaisesRegex(ValueError, "lacks visible text"):
            self.prepare()

    def test_invalid_sample_size_and_input_protection(self):
        with self.assertRaisesRegex(ValueError, "sample_size"):
            self.prepare(sample_size=3)
        with self.assertRaisesRegex(ValueError, "outside"):
            write_audit(self.catalog, self.prepare())

    def test_document_mismatch_and_missing_document_fail_closed(self):
        self.document_rows[6]["genres"] = ["Comedy"]
        _jsonl(self.movie_documents, self.document_rows)
        with self.assertRaisesRegex(ValueError, "disagrees with catalog"):
            self.prepare()
        self.document_rows.pop(6)
        _jsonl(self.movie_documents, self.document_rows)
        with self.assertRaisesRegex(ValueError, "do not exactly cover"):
            self.prepare()


if __name__ == "__main__":
    unittest.main()
