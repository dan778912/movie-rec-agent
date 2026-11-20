import json
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np
from src.retrieval.online import OnlineRetriever, chroma_client, ingest


class TinyEncoder:
    settings = {"model_id": "synthetic", "revision": "a" * 40, "max_seq_length": 256, "batch_size": 64, "seed": 0}

    def encode(self, texts):
        return np.array([[1., 0.] if "Comedy" in t else [0., 1.] for t in texts], dtype=np.float32)


class OnlineRetrievalTest(unittest.TestCase):
    def test_http_transport_has_explicit_timeout_and_reports_timeout(self):
        import httpx
        from src.retrieval.chroma_http import bounded_http_client
        from types import SimpleNamespace
        target = SimpleNamespace(_server=SimpleNamespace(_session=SimpleNamespace()))
        with patch('chromadb.HttpClient', return_value=target):
            client = bounded_http_client("localhost", 8000, None)
            self.assertEqual(target._server._session.timeout.read, 5)
        def slow():
            raise httpx.ReadTimeout("slow")
        target.heartbeat = slow
        with self.assertRaises(TimeoutError):
            client.heartbeat()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.documents = self.root / "documents.jsonl"
        self.csv = self.root / "catalog.csv"
        self.manifest = self.root / "index.json"
        self.encoder = TinyEncoder()
        records = [{"movie_id": i, "title": f"Movie {i}", "text": "Comedy" if i < 4 else "Drama",
                    "genres": ["Comedy" if i < 4 else "Drama"], "genome_tags": [], "user_tags": []}
                   for i in range(1, 7)]
        self.documents.write_text("\n".join(json.dumps(r) for r in records))
        self.csv.write_text("movie_id\n" + "\n".join(str(i) for i in range(1, 7)))
        self.client = chroma_client(self.root / "chroma")
        self.receipt = ingest(self.client, self.documents, self.csv, self.manifest, self.encoder)
        self.retriever = OnlineRetriever(self.client, self.manifest, self.documents, self.encoder, pool_size=2)

    def test_real_chroma_search_excludes_watched_and_existing_candidates(self):
        pool = self.retriever.initial_pool(profile_text="Comedy", clarifications=[], history_movie_ids=[1, 999])
        self.assertEqual(pool.movie_ids, [2, 3])
        more = self.retriever.search(query="Drama", exclude_movie_ids=[1, 2, 3, 999], max_new=50)
        self.assertEqual(more.movie_ids, [4, 5, 6])
        self.assertEqual(more.index_version, pool.index_version)

    def test_clarification_changes_query_and_empty_catalog_remainder_is_explicit(self):
        original = self.encoder.encode
        seen = []
        def capture(texts):
            seen.extend(texts)
            return original(texts)
        self.encoder.encode = capture
        result = self.retriever.initial_pool(profile_text="Drama", clarifications=[{"question": "Genre?", "answer": "Comedy"}],
                                             history_movie_ids=[])
        self.assertTrue(seen[0].startswith("Answer: Comedy"))
        # ANN may select any two of three tied Comedy vectors; sort only the returned ties.
        self.assertEqual(len(result.movie_ids), 2)
        self.assertTrue(set(result.movie_ids) <= {1, 2, 3})
        self.assertEqual(result.movie_ids, sorted(result.movie_ids))
        self.assertEqual(self.retriever.search(query="Drama", exclude_movie_ids=list(range(1, 7)), max_new=10).movie_ids, [])

    def test_reopen_persistent_index_and_reject_mismatched_settings(self):
        reopened = OnlineRetriever(chroma_client(self.root / "chroma"), self.manifest, self.documents, self.encoder)
        self.assertEqual(reopened.search(query="Drama", exclude_movie_ids=[], max_new=3).movie_ids, [4, 5, 6])
        other = TinyEncoder()
        other.settings = {**other.settings, "max_seq_length": 128}
        with self.assertRaisesRegex(ValueError, "configuration"):
            OnlineRetriever(self.client, self.manifest, self.documents, other)
        self.documents.write_text(self.documents.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            OnlineRetriever(self.client, self.manifest, self.documents, self.encoder)

    def test_immutable_publication_and_corrupt_collection_detected(self):
        with self.assertRaises(FileExistsError):
            ingest(self.client, self.documents, self.csv, self.manifest, self.encoder)
        self.client.get_collection(self.receipt["index_version"]).delete(ids=["1"])
        with self.assertRaisesRegex(ValueError, "completeness"):
            OnlineRetriever(self.client, self.manifest, self.documents, self.encoder)


if __name__ == "__main__":
    unittest.main()
