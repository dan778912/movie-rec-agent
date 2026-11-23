"""Build the versioned movie-catalog seed for Service A and the online vector index.

One JSON record per catalog movie with the canonical document text, display
fields, and training-only rating aggregates. Field names are a superset of
Service A's ``movies`` table (``movie_id``, ``title``, ``genres``, ``text``,
``genome_tags``, ``user_tags``, ``rating_avg``, ``rating_count``) so the seed
can be loaded without re-scanning the 20M-row training file. Aggregates use
``train_ratings.csv`` only; no validation or test rating is read.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from src.data.build import csv_rows, sha256, write_jsonl
from src.data.p2_samples import year_of
from src.profiles.agent_text import display_tags

SEED_VERSION = "catalog-seed-v1"
DATA = Path("data/processed/ml25m-global-v1")
RAW = Path("data/raw/ml-25m")
OUTPUT = Path("data/processed/p2-v2")
CONSUMED = ("catalog.csv", "movie_documents.jsonl", "train_ratings.csv")
_ARTICLE = re.compile(r"^(?P<main>.+), (?P<article>The|A|An)$")


def display_title(title: str) -> str:
    """``"Matrix, The (1999)"`` -> ``"The Matrix"``; alternate titles in parentheses are kept."""
    base = re.sub(r"\s*\(\d{4}(?:[-–]\d{0,4})?\)\s*$", "", title).strip()
    head, sep, tail = base.partition(" (")
    match = _ARTICLE.match(head)
    if match:
        head = f"{match['article']} {match['main']}"
    return head + (sep + tail if sep else "")


def _imdb(value: str) -> str | None:
    return f"tt{value}" if value else None


def build_seed(data: Path, raw: Path, output: Path) -> dict:
    started = time.perf_counter()
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    for name in CONSUMED:
        if sha256(data / name) != manifest["file_sha256"][name]:
            raise ValueError(f"Input checksum mismatch: {name}")
    catalog = {int(r["movie_id"]): r for r in csv_rows(data / "catalog.csv")}
    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    positives: dict[int, int] = {}
    for row in csv_rows(data / "train_ratings.csv"):
        movie_id = int(row["movie_id"])
        if movie_id in catalog:
            rating = float(row["rating"])
            sums[movie_id] = sums.get(movie_id, 0.0) + rating
            counts[movie_id] = counts.get(movie_id, 0) + 1
            positives[movie_id] = positives.get(movie_id, 0) + (rating >= 4)
    links = {int(r["movieId"]): r for r in csv_rows(raw / "links.csv")}
    rank = {mid: i for i, mid in enumerate(sorted(catalog, key=lambda m: (-int(catalog[m]["train_positives"]), m)), 1)}
    with (data / "movie_documents.jsonl").open(encoding="utf-8") as source:
        documents = [json.loads(line) for line in source]
    records = []
    for doc in documents:
        movie_id = doc["movie_id"]
        row = catalog[movie_id]
        if counts.get(movie_id) != int(row["train_ratings"]) or positives.get(movie_id) != int(row["train_positives"]):
            raise AssertionError(f"Training aggregates disagree with catalog.csv for {movie_id}")
        link = links.get(movie_id, {})
        records.append({
            "movie_id": movie_id, "title": doc["title"], "display_title": display_title(doc["title"]),
            "year": year_of(doc["title"]), "genres": doc["genres"], "genome_tags": doc["genome_tags"],
            "user_tags": doc["user_tags"], "display_tags": display_tags(doc["genome_tags"], doc["user_tags"]),
            "text": doc["text"], "rating_avg": round(sums[movie_id] / counts[movie_id], 4),
            "rating_count": counts[movie_id], "train_positives": positives[movie_id],
            "popularity_rank": rank[movie_id], "imdb_id": _imdb(link.get("imdbId", "")),
            "tmdb_id": int(link["tmdbId"]) if link.get("tmdbId") else None,
        })
    if len(records) != len(catalog):
        raise AssertionError("Seed does not cover the catalog")
    output.mkdir(parents=True, exist_ok=True)
    path = output / "catalog_seed.jsonl"
    write_jsonl(path, records)
    report = {
        "seed_version": SEED_VERSION, "data_version": manifest["version"],
        "data_manifest_sha256": sha256(data / "manifest.json"),
        "consumed_sha256": {name: manifest["file_sha256"][name] for name in CONSUMED},
        "links_sha256": sha256(raw / "links.csv"),
        "movies": len(records), "output_sha256": sha256(path),
        "missing_year": sum(r["year"] is None for r in records),
        "missing_genres": sum(not r["genres"] for r in records),
        "missing_genome_tags": sum(not r["genome_tags"] for r in records),
        "missing_user_tags": sum(not r["user_tags"] for r in records),
        "missing_tmdb_id": sum(r["tmdb_id"] is None for r in records),
        "aggregate_scope": "train_ratings.csv only (eligible users, before the validation cutoff)",
        "runtime_seconds": round(time.perf_counter() - started, 1),
    }
    (output / "catalog_seed.report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                                     encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--raw", type=Path, default=RAW)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    print(json.dumps(build_seed(args.data, args.raw, args.output), indent=2))


if __name__ == "__main__":
    main()
