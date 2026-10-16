"""Build frozen MovieLens 25M data artifacts with global time cutoffs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from src.profiles.text import movie_document, user_profile


def csv_rows(path: Path):
    with path.open("r", encoding="utf-8", newline="") as source:
        yield from csv.DictReader(source)


def write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8", newline="\n") as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _connect(path: Path) -> sqlite3.Connection:
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("PRAGMA cache_size=-131072")
    return con


def _import_ratings(con: sqlite3.Connection, path: Path) -> int:
    con.execute("CREATE TABLE ratings(user_id INTEGER, movie_id INTEGER, rating REAL, ts INTEGER)")
    batch = []
    count = 0
    for row in csv_rows(path):
        batch.append((int(row["userId"]), int(row["movieId"]), float(row["rating"]), int(row["timestamp"])))
        if len(batch) == 50000:
            con.executemany("INSERT INTO ratings VALUES (?,?,?,?)", batch)
            count += len(batch)
            batch.clear()
    if batch:
        con.executemany("INSERT INTO ratings VALUES (?,?,?,?)", batch)
        count += len(batch)
    con.commit()
    con.execute("CREATE INDEX rating_time ON ratings(ts)")
    con.execute("CREATE INDEX rating_user_time ON ratings(user_id,ts,movie_id)")
    return count


def _cutoffs(con: sqlite3.Connection, total: int, test_quantile: float, val_quantile: float):
    test_offset = int((total - 1) * test_quantile)
    test_cutoff = con.execute("SELECT ts FROM ratings ORDER BY ts LIMIT 1 OFFSET ?", (test_offset,)).fetchone()[0]
    pretest = con.execute("SELECT COUNT(*) FROM ratings WHERE ts < ?", (test_cutoff,)).fetchone()[0]
    val_offset = int((pretest - 1) * val_quantile)
    val_cutoff = con.execute("SELECT ts FROM ratings WHERE ts < ? ORDER BY ts LIMIT 1 OFFSET ?", (test_cutoff, val_offset)).fetchone()[0]
    if not val_cutoff < test_cutoff:
        raise ValueError("Dataset timestamps do not permit distinct validation/test cutoffs")
    return val_cutoff, test_cutoff


def _movies(path: Path):
    return {int(row["movieId"]): (row["title"], row["genres"]) for row in csv_rows(path)}


def _genome_tags(raw: Path, top_k: int):
    names = {int(row["tagId"]): row["tag"].strip() for row in csv_rows(raw / "genome-tags.csv")}
    best = {}
    for row in csv_rows(raw / "genome-scores.csv"):
        movie_id = int(row["movieId"])
        tag_id = int(row["tagId"])
        score = float(row["relevance"])
        heap = best.setdefault(movie_id, [])
        pair = (score, -tag_id)
        if len(heap) < top_k:
            heapq.heappush(heap, pair)
        elif pair > heap[0]:
            heapq.heapreplace(heap, pair)
    return {
        movie_id: [names[-tag_id] for _, tag_id in sorted(heap, reverse=True)]
        for movie_id, heap in best.items()
    }


def _user_tags(raw: Path, before: int, catalog: set[int], top_k: int):
    counts = {}
    for row in csv_rows(raw / "tags.csv"):
        movie_id = int(row["movieId"])
        if movie_id not in catalog or int(row["timestamp"]) >= before:
            continue
        tag = re.sub(r"\s+", " ", row["tag"].strip().lower())
        if tag:
            counts.setdefault(movie_id, Counter())[tag] += 1
    return {
        movie_id: [tag for tag, _ in sorted(counter.items(), key=lambda pair: (-pair[1], pair[0]))[:top_k]]
        for movie_id, counter in counts.items()
    }


def _split_where(split: str, val_cutoff: int, test_cutoff: int):
    if split == "train":
        return "r.ts < ?", (val_cutoff,)
    if split == "validation":
        return "r.ts >= ? AND r.ts < ?", (val_cutoff, test_cutoff)
    return "r.ts >= ?", (test_cutoff,)


def _export_split(con: sqlite3.Connection, output: Path, split: str, val_cutoff: int, test_cutoff: int):
    where, params = _split_where(split, val_cutoff, test_cutoff)
    query = ("SELECT r.user_id,r.movie_id,r.rating,r.ts FROM ratings r "
             "JOIN eligible e ON e.user_id=r.user_id WHERE " + where + " ORDER BY r.user_id,r.ts,r.movie_id")
    count = positive = 0
    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("user_id", "movie_id", "rating", "timestamp"))
        for user_id, movie_id, rating, ts in con.execute(query, params):
            writer.writerow((user_id, movie_id, f"{rating:g}", ts))
            count += 1
            positive += rating >= 4.0
    return {"interactions": count, "positives": positive}


def _targets(con: sqlite3.Connection, output: Path, split: str, val_cutoff: int, test_cutoff: int):
    where, params = _split_where(split, val_cutoff, test_cutoff)
    query = ("SELECT r.user_id,r.movie_id,r.rating,r.ts,c.movie_id IS NOT NULL "
             "FROM ratings r JOIN eligible e ON e.user_id=r.user_id "
             "LEFT JOIN catalog c ON c.movie_id=r.movie_id WHERE " + where +
             " AND r.rating >= 4 ORDER BY r.user_id,r.ts,r.movie_id")
    users = set()
    counts = Counter()
    with output.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target, lineterminator="\n")
        writer.writerow(("user_id", "movie_id", "rating", "timestamp", "in_catalog"))
        for user_id, movie_id, rating, ts, in_catalog in con.execute(query, params):
            writer.writerow((user_id, movie_id, f"{rating:g}", ts, int(in_catalog)))
            counts["positive_targets"] += 1
            counts["positive_targets_in_catalog"] += bool(in_catalog)
            if in_catalog:
                users.add(user_id)
    counts["users_with_in_catalog_positive"] = len(users)
    return sorted(users), dict(counts)


def _profile_items(history: list[tuple], movies: dict, genome: dict, user_tags: dict, limit: int):
    items = []
    for movie_id, rating, ts in history[:limit]:
        title, genres = movies[movie_id]
        items.append({
            "movie_id": movie_id,
            "title": title,
            "rating": rating,
            "timestamp": ts,
            "genres": [] if genres == "(no genres listed)" else genres.split("|"),
            "tags": (genome.get(movie_id, [])[:3] + user_tags.get(movie_id, [])[:2]),
        })
    return items


def _export_profiles(con: sqlite3.Connection, output: Path, split: str, users: list[int], cutoff: int,
                     movies: dict, genome: dict, user_tags: dict, config: dict):
    def records():
        for user_id in users:
            history = list(con.execute(
                "SELECT movie_id,rating,ts FROM ratings WHERE user_id=? AND ts < ? ORDER BY ts,movie_id",
                (user_id, cutoff)))
            if len(history) < config["minimum_training_ratings_per_user"]:
                raise AssertionError(f"History too short for user {user_id}")
            if any(ts >= cutoff for _, _, ts in history):
                raise AssertionError("Future interaction entered a profile")
            liked = sorted((row for row in history if row[1] >= 4), key=lambda row: (-row[1], -row[2], row[0]))
            disliked = sorted((row for row in history if row[1] <= 2.5), key=lambda row: (row[1], -row[2], row[0]))
            likes = _profile_items(liked, movies, genome, user_tags, config["liked_movies_per_profile"])
            dislikes = _profile_items(disliked, movies, genome, user_tags, config["disliked_movies_per_profile"])
            yield {
                "user_id": user_id,
                "split": split,
                "text": user_profile(likes, dislikes),
                "history_movie_ids": sorted({row[0] for row in history}),
                "history_count": len(history),
                "history_latest_timestamp": max(row[2] for row in history),
                "liked_movies": likes,
                "disliked_movies": dislikes,
            }
    write_jsonl(output, records())


def build(raw: Path, output: Path, config: dict):
    for name in ("ratings.csv", "movies.csv", "tags.csv", "genome-tags.csv", "genome-scores.csv"):
        if not (raw / name).is_file():
            raise FileNotFoundError(raw / name)
    output.mkdir(parents=True, exist_ok=True)
    db = output / "_build.sqlite3"
    if db.exists():
        db.unlink()
    con = _connect(db)
    try:
        total = _import_ratings(con, raw / "ratings.csv")
        val_cutoff, test_cutoff = _cutoffs(con, total, config["test_quantile"],
                                           config["validation_quantile_within_pretest"])
        min_user = config["minimum_training_ratings_per_user"]
        con.execute("CREATE TABLE eligible(user_id INTEGER PRIMARY KEY)")
        con.execute("INSERT INTO eligible SELECT user_id FROM ratings WHERE ts < ? "
                    "GROUP BY user_id HAVING COUNT(*) >= ?", (val_cutoff, min_user))
        con.execute("CREATE TABLE catalog(movie_id INTEGER PRIMARY KEY, train_ratings INTEGER, train_positives INTEGER)")
        con.execute("INSERT INTO catalog SELECT r.movie_id,COUNT(*),SUM(r.rating >= 4) "
                    "FROM ratings r JOIN eligible e ON e.user_id=r.user_id WHERE r.ts < ? "
                    "GROUP BY r.movie_id HAVING COUNT(*) >= ?",
                    (val_cutoff, config["minimum_training_ratings_per_catalog_movie"]))
        catalog_rows = con.execute("SELECT * FROM catalog ORDER BY movie_id").fetchall()
        catalog = {row[0] for row in catalog_rows}
        movies = _movies(raw / "movies.csv")
        missing_movies = catalog - movies.keys()
        if missing_movies:
            raise ValueError(f"Catalog IDs absent from movies.csv: {sorted(missing_movies)[:10]}")
        genome = _genome_tags(raw, config["genome_tags_per_movie"])
        user_tags = _user_tags(raw, val_cutoff, catalog, config["user_tags_per_movie"])
        with (output / "catalog.csv").open("w", encoding="utf-8", newline="") as target:
            writer = csv.writer(target, lineterminator="\n")
            writer.writerow(("movie_id", "title", "genres", "train_ratings", "train_positives"))
            for movie_id, count, positives in catalog_rows:
                writer.writerow((movie_id, *movies[movie_id], count, positives))
        write_jsonl(output / "movie_documents.jsonl", (
            {"movie_id": movie_id, "title": movies[movie_id][0],
             "genres": [] if movies[movie_id][1] == "(no genres listed)" else movies[movie_id][1].split("|"),
             "genome_tags": genome.get(movie_id, []), "user_tags": user_tags.get(movie_id, []),
             "text": movie_document(*movies[movie_id], genome.get(movie_id, []), user_tags.get(movie_id, []))}
            for movie_id in sorted(catalog)))
        split_stats = {}
        for split in ("train", "validation", "test"):
            split_stats[split] = _export_split(con, output / f"{split}_ratings.csv", split, val_cutoff, test_cutoff)
        eval_stats = {}
        for split, cutoff in (("validation", val_cutoff), ("test", test_cutoff)):
            users, counts = _targets(con, output / f"{split}_targets.csv", split, val_cutoff, test_cutoff)
            _export_profiles(con, output / f"{split}_profiles.jsonl", split, users, cutoff,
                             movies, genome, user_tags, config)
            counts["catalog_positive_coverage"] = (
                counts.get("positive_targets_in_catalog", 0) / counts["positive_targets"]
                if counts.get("positive_targets", 0) else None)
            eval_stats[split] = counts
        user_count = con.execute("SELECT COUNT(*) FROM eligible").fetchone()[0]
        train_catalog_interactions = con.execute(
            "SELECT COUNT(*) FROM ratings r JOIN eligible e ON e.user_id=r.user_id "
            "JOIN catalog c ON c.movie_id=r.movie_id WHERE r.ts < ?", (val_cutoff,)).fetchone()[0]
        stats = {
            "source_ratings": total,
            "eligible_users": user_count,
            "catalog_movies": len(catalog),
            "train_catalog_interactions": train_catalog_interactions,
            "train_catalog_density": train_catalog_interactions / (user_count * len(catalog)) if user_count and catalog else None,
            "train_catalog_sparsity": 1 - train_catalog_interactions / (user_count * len(catalog)) if user_count and catalog else None,
            "splits": split_stats,
            "evaluation": eval_stats,
            "validation_cutoff_unix": val_cutoff,
            "validation_cutoff_utc": datetime.fromtimestamp(val_cutoff, timezone.utc).isoformat(),
            "test_cutoff_unix": test_cutoff,
            "test_cutoff_utc": datetime.fromtimestamp(test_cutoff, timezone.utc).isoformat(),
        }
        (output / "statistics.json").write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        files = sorted(p for p in output.iterdir() if p.is_file() and p.name != "manifest.json" and not p.name.startswith(db.name))
        manifest = {
            "version": config["version"], "config": config,
            "raw_sha256": {name: sha256(raw / name) for name in
                           ("ratings.csv", "movies.csv", "tags.csv", "genome-tags.csv", "genome-scores.csv")},
            "file_sha256": {p.name: sha256(p) for p in files},
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return stats
    finally:
        con.close()
        db.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/ml-25m"))
    parser.add_argument("--output", type=Path, default=Path("data/processed/ml25m-global-v1"))
    parser.add_argument("--config", type=Path, default=Path("configs/data_v1.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    print(json.dumps(build(args.raw, args.output, config), indent=2))


if __name__ == "__main__":
    main()
