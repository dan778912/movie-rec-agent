"""Dependency-free BM25 over Member 1's canonical movie documents.

Uses positive Robertson IDF and binary query term frequency. Indexing never
reads ratings or targets; BM25 statistics come solely from the frozen catalog.
"""

from __future__ import annotations

import heapq
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.data.build import sha256

TOKENIZER_VERSION = "unicode-casefold-stopwords-v1"
# Fixed, untuned English function words and shared text-template scaffolding.
# Pure numbers (ratings and years) are omitted from documents and queries alike.
STOPWORDS = frozenset("""
a an and are as at be been being by for from had has have he her hers him his
i in into is it its me my of on or our ours she that the their them they this
to was we were what when where which who will with you your
highly low rated movies movie available history no genres genre genome user
tags tag unspecified none
""".split())
TOKEN_PATTERN = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
ALLOWED_FIELDS = frozenset({"text", "title", "genres", "genome_tags", "user_tags"})


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_PATTERN.findall(text.casefold())
            if token not in STOPWORDS and not token.isdecimal()]


def document_text(document: dict, fields: list[str]) -> str:
    if not fields or len(set(fields)) != len(fields) or set(fields) - ALLOWED_FIELDS:
        raise ValueError(f"Invalid document fields: {fields}")
    if "text" in fields and fields != ["text"]:
        raise ValueError("Use canonical text alone, or explicitly select metadata fields")
    values = []
    for field in fields:
        value = document[field]
        values.append(" ".join(value) if isinstance(value, list) else value)
    return " ".join(values)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


class BM25Index:
    def __init__(self, movie_ids: list[int], lengths: list[int], postings: dict,
                 metadata: dict):
        if not movie_ids or len(movie_ids) != len(set(movie_ids)):
            raise ValueError("Index requires a nonempty catalog of unique movie IDs")
        self.movie_ids = movie_ids
        self.lengths = lengths
        self.postings = postings
        self.metadata = metadata
        self.average_length = sum(lengths) / len(lengths)
        self.idf = {term: math.log1p((len(movie_ids) - len(rows) + 0.5) / (len(rows) + 0.5))
                    for term, rows in postings.items()}
        self._parameters = None
        self._weights = None

    @classmethod
    def build(cls, documents: list[dict], fields: list[str], metadata: dict | None = None):
        movie_ids, lengths = [], []
        postings = defaultdict(list)
        for position, document in enumerate(sorted(documents, key=lambda row: row["movie_id"])):
            counts = Counter(tokenize(document_text(document, fields)))
            movie_ids.append(document["movie_id"])
            lengths.append(sum(counts.values()))
            for term, count in sorted(counts.items()):
                postings[term].append((position, count))
        return cls(movie_ids, lengths, dict(postings), {
            **(metadata or {}), "format": "bm25-index-v1", "fields": fields,
            "tokenizer": TOKENIZER_VERSION,
        })

    def prepare(self, k1: float = 1.2, b: float = 0.75) -> None:
        if not math.isfinite(k1) or k1 <= 0 or not math.isfinite(b) or not 0 <= b <= 1:
            raise ValueError("BM25 requires finite k1 > 0 and 0 <= b <= 1")
        if self._parameters == (k1, b):
            return
        average = self.average_length or 1.0
        norms = [k1 * (1 - b + b * length / average) for length in self.lengths]
        self._weights = {
            term: [(position, self.idf[term] * count * (k1 + 1) / (count + norms[position]))
                   for position, count in rows]
            for term, rows in self.postings.items()
        }
        self._parameters = (k1, b)

    def search(self, query: str, watched: set[int], top_k: int = 100,
               k1: float = 1.2, b: float = 0.75) -> list[dict]:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k must be a positive integer")
        self.prepare(k1, b)
        scores = defaultdict(float)
        # Stable summation order makes scores and tie handling reproducible.
        for term in sorted(set(tokenize(query))):
            for position, weight in self._weights.get(term, ()):
                scores[position] += weight
        candidates = ((self.movie_ids[position], scores.get(position, 0.0))
                      for position in range(len(self.movie_ids))
                      if self.movie_ids[position] not in watched)
        ranked = heapq.nsmallest(top_k, candidates, key=lambda pair: (-pair[1], pair[0]))
        return [{"movie_id": movie_id, "score": score, "rank": rank}
                for rank, (movie_id, score) in enumerate(ranked, 1)]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"metadata": self.metadata, "movie_ids": self.movie_ids,
                                    "lengths": self.lengths, "postings": self.postings},
                                   sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        if (data["metadata"].get("format") != "bm25-index-v1"
                or data["metadata"].get("tokenizer") != TOKENIZER_VERSION):
            raise ValueError("Incompatible BM25 index; rebuild it with this tokenizer")
        return cls(data["movie_ids"], data["lengths"], data["postings"], data["metadata"])


def build_index(data_dir: Path, fields: list[str]) -> BM25Index:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    documents_path = data_dir / "movie_documents.jsonl"
    digest = sha256(documents_path)
    if digest != manifest["file_sha256"]["movie_documents.jsonl"]:
        raise ValueError("Movie document checksum differs from the dataset manifest")
    return BM25Index.build(read_jsonl(documents_path), fields, {
        "dataset_version": manifest["version"], "documents_sha256": digest,
    })
