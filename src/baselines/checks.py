"""Check Member 3 handoff artifacts against their source data; no metric computation."""
from __future__ import annotations

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import json
from pathlib import Path
import time

from src.baselines.common import DATA, CONFIG, check_inputs, jsonl, read_config, save_json
from src.baselines.training import user_histories
from src.data.build import csv_rows, sha256
from src.profiles.text import user_profile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_training(data, path, config):
    started = time.perf_counter()
    manifest = check_inputs(data, ["train_ratings.csv", "movie_documents.jsonl", "statistics.json"])
    cutoff = json.loads((data / "statistics.json").read_text())["validation_cutoff_unix"]
    docs = {r["movie_id"]: r for r in jsonl(data / "movie_documents.jsonl")}
    by_user = defaultdict(list)
    ids = set()
    labels = Counter()
    for record in jsonl(path):
        uid, mid = record["user_id"], record["movie_id"]
        key = (uid, mid)
        require(key not in ids, f"Duplicate training user/movie: {key}")
        ids.add(key)
        require(record["split"] == "train", "Non-training example")
        require(mid in docs, f"Candidate outside shared catalog: {mid}")
        by_user[uid].append(record)
    require(bool(by_user), "Empty training export")
    require(len(by_user) <= config["training_user_limit"], "Export exceeds configured user limit")
    remaining = set(by_user)
    for uid, history in user_histories(data / "train_ratings.csv", cutoff):
        if uid not in by_user:
            continue
        remaining.remove(uid)
        records = by_user[uid]
        require(len(records) <= config["examples_per_user"], f"Too many targets for user {uid}")
        times = [r["timestamp"] for r in history]
        targets = {(r["movie_id"], r["timestamp"]): r["rating"] for r in history}
        for record in records:
            mid, ts, rating = record["movie_id"], record["target_timestamp"], record["target_rating"]
            require(targets.get((mid, ts)) == rating, f"Target absent from training: {uid}/{mid}")
            require(record["example_id"] == f"{uid}:{mid}:{ts}", "Incorrect example ID")
            require(rating >= 4 or rating <= 2.5, "Ambiguous target rating")
            label = "relevant" if rating >= 4 else "not_relevant"
            require(record["label"] == label, f"Incorrect label for {uid}/{mid}")
            require(record["negative_source"] == ("explicit_rating" if rating <= 2.5 else None),
                    "Incorrect negative provenance")
            earlier = history[:bisect_left(times, ts)]
            require(len(earlier) >= config["minimum_history"], "Insufficient earlier history")
            require(record["history_count"] == len(earlier), "Incorrect history count")
            require(record["history_latest_timestamp"] == earlier[-1]["timestamp"] < ts,
                    "Future history or incorrect latest timestamp")
            require(mid not in {r["movie_id"] for r in earlier}, "Already-rated training target")
            known = [r for r in earlier if r["movie_id"] in docs]
            liked = sorted((r for r in known if r["rating"] >= 4),
                           key=lambda r: (-r["rating"], -r["timestamp"], r["movie_id"]))
            disliked = sorted((r for r in known if r["rating"] <= 2.5),
                              key=lambda r: (r["rating"], -r["timestamp"], r["movie_id"]))
            def render_items(rows, limit):
                result = []
                for row in rows[:limit]:
                    doc = docs[row["movie_id"]]
                    result.append({**row, "title": doc["title"], "genres": doc["genres"],
                                   "tags": doc["genome_tags"][:3] + doc["user_tags"][:2]})
                return result
            liked = render_items(liked, manifest["config"]["liked_movies_per_profile"])
            disliked = render_items(disliked, manifest["config"]["disliked_movies_per_profile"])
            require(record["liked_movies"] == liked and record["disliked_movies"] == disliked,
                    f"Incorrect selected history for {uid}/{mid}")
            require(record["profile"] == user_profile(liked, disliked), "Noncanonical profile text")
            require(record["candidate_document"] == docs[mid]["text"], "Noncanonical candidate text")
            labels[label] += 1
    require(not remaining, f"Users absent from training: {sorted(remaining)[:10]}")
    return {"artifact": str(path), "output_sha256": sha256(path), "error_count": 0,
            "examples_checked": sum(labels.values()), "users_checked": len(by_user),
            "labels": dict(labels), "runtime_seconds": time.perf_counter() - started,
            "scope": "Source ratings, strictly earlier history, labels, selected movies, canonical text, duplicates and export limits."}


def check_popularity(data, path, config, split="validation", user_ids=None):
    started = time.perf_counter()
    require(split in ("validation", "test"), "Invalid split")
    check_inputs(data, ["catalog.csv", f"{split}_profiles.jsonl"])
    profiles = {r["user_id"]: r for r in jsonl(data / f"{split}_profiles.jsonl")}
    expected_users = set(profiles) if user_ids is None else set(user_ids)
    require(bool(expected_users) and expected_users <= profiles.keys(), "Invalid evaluation user subset")
    scores = {int(r["movie_id"]): int(r["train_positives"]) for r in csv_rows(data / "catalog.csv")}
    order = sorted(scores, key=lambda mid: (-scores[mid], mid))
    seen_users = set()
    count = 0
    for record in jsonl(path):
        uid = record["user_id"]
        require(uid in expected_users and uid not in seen_users, "Unexpected or duplicate B0 user")
        seen_users.add(uid)
        require(record["method"] == "popularity" and record["split"] == split, "Incorrect method/split")
        watched = set(profiles[uid]["history_movie_ids"])
        expected = [mid for mid in order if mid not in watched][:config["popularity_k"]]
        actual = record["recommendations"]
        require([r["movie_id"] for r in actual] == expected, f"Incorrect popularity order or watched removal: {uid}")
        for rank, rec in enumerate(actual, 1):
            require(rec["rank"] == rank and rec["score"] == scores[rec["movie_id"]], "Incorrect B0 score/rank")
        count += len(actual)
    require(seen_users == expected_users, "Missing B0 evaluation users")
    return {"artifact": str(path), "output_sha256": sha256(path), "error_count": 0,
            "users_checked": len(seen_users), "recommendations_checked": count,
            "runtime_seconds": time.perf_counter() - started,
            "scope": "B0 counts, exact expected order, watched removal, ranks, duplicate and missing users; no metrics."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind", choices=("training", "popularity"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--user-ids", type=Path, help="JSON user-ID array matching a B0 subset export")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.kind == "training":
        report = check_training(args.data, args.input, config)
    else:
        subset = json.loads(args.user_ids.read_text()) if args.user_ids else None
        report = check_popularity(args.data, args.input, config, args.split, subset)
    save_json(args.output, report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
