"""Export chronological, explicitly labelled profile-candidate examples for P1."""
from __future__ import annotations

import argparse
from bisect import insort
from collections import Counter
from itertools import groupby
import json
from pathlib import Path
import random
import time

from src.baselines.common import DATA, CONFIG, check_inputs, jsonl, provenance, read_config, save_json
from src.data.build import csv_rows, sha256, write_jsonl
from src.profiles.text import user_profile


def user_histories(path, cutoff):
    """Stream the shared user/time-ordered file, retaining only one user at a time."""
    previous_user = -1
    for uid, rows in groupby(csv_rows(path), lambda r: int(r["user_id"])):
        if uid <= previous_user:
            raise ValueError("train_ratings.csv must be sorted by user_id")
        previous_user = uid
        history = []
        previous_key = (-1, -1)
        for row in rows:
            mid, ts, rating = int(row["movie_id"]), int(row["timestamp"]), float(row["rating"])
            if (ts, mid) < previous_key or ts >= cutoff:
                raise ValueError(f"Out-of-order or non-training rating for user {uid}")
            if rating not in (0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5):
                raise ValueError(f"Invalid rating for user {uid}: {rating}")
            history.append({"movie_id": mid, "rating": rating, "timestamp": ts})
            previous_key = (ts, mid)
        yield uid, history


def eligible_targets(history, docs, minimum, counts):
    seen = set()
    previous_count = 0
    candidates = []
    for _, batch in groupby(enumerate(history), lambda pair: pair[1]["timestamp"]):
        batch = list(batch)
        for index, row in batch:
            counts["ratings"] += 1
            rating, mid = row["rating"], row["movie_id"]
            reason = ("ambiguous" if 2.5 < rating < 4 else
                      "outside_catalog" if mid not in docs else
                      "short_history" if previous_count < minimum else
                      "already_seen" if mid in seen else None)
            if reason:
                counts["skipped_" + reason] += 1
                continue
            label = "relevant" if rating >= 4 else "not_relevant"
            counts["eligible_" + label] += 1
            candidates.append(index)
        seen.update(row["movie_id"] for _, row in batch)
        previous_count += len(batch)
    return candidates


def profile_examples(uid, history, selected, docs, profile_config):
    """Keep the same top-rated/recency ordering and formatter as Member 1."""
    likes, dislikes = [], []
    seen = set()
    history_count = 0
    latest = None
    like_limit = profile_config["liked_movies_per_profile"]
    dislike_limit = profile_config["disliked_movies_per_profile"]
    def item(row):
        doc = docs[row["movie_id"]]
        return {**row, "title": doc["title"], "genres": doc["genres"],
                "tags": doc["genome_tags"][:3] + doc["user_tags"][:2]}
    for _, batch in groupby(enumerate(history), lambda pair: pair[1]["timestamp"]):
        batch = list(batch)
        for index, row in batch:
            if index not in selected:
                continue
            liked = [item(history[i]) for _, i in likes]
            disliked = [item(history[i]) for _, i in dislikes]
            yield {
                "example_id": f"{uid}:{row['movie_id']}:{row['timestamp']}",
                "user_id": uid, "movie_id": row["movie_id"], "split": "train",
                "profile": user_profile(liked, disliked),
                "candidate_document": docs[row["movie_id"]]["text"],
                "label": "relevant" if row["rating"] >= 4 else "not_relevant",
                "target_rating": row["rating"], "target_timestamp": row["timestamp"],
                "negative_source": "explicit_rating" if row["rating"] <= 2.5 else None,
                "history_count": history_count, "history_latest_timestamp": latest,
                "liked_movies": liked, "disliked_movies": disliked,
            }
        # Commit the timestamp group only after exporting all its targets.
        for index, row in batch:
            mid, rating, ts = row["movie_id"], row["rating"], row["timestamp"]
            if mid in seen:
                raise ValueError(f"Duplicate user/movie interaction: {uid}/{mid}")
            seen.add(mid)
            history_count += 1
            latest = ts
            if mid not in docs:
                continue
            if rating >= 4:
                insort(likes, ((-rating, -ts, mid), index))
                del likes[like_limit:]
            elif rating <= 2.5:
                insort(dislikes, ((rating, -ts, mid), index))
                del dislikes[dislike_limit:]


def build_examples(data, output, config):
    started = time.perf_counter()
    manifest = check_inputs(data, ["train_ratings.csv", "movie_documents.jsonl", "statistics.json"])
    cutoff = json.loads((data / "statistics.json").read_text())["validation_cutoff_unix"]
    docs = {r["movie_id"]: r for r in jsonl(data / "movie_documents.jsonl")}
    # Uniform reservoir of qualifying users; classification counts cover all users.
    # Store target indices rather than millions of fully rendered examples.
    reservoir = []
    population = Counter()
    eligible_users = 0
    rng = random.Random(config["seed"])
    for uid, history in user_histories(data / "train_ratings.csv", cutoff):
        population["users"] += 1
        candidates = eligible_targets(history, docs, config["minimum_history"], population)
        if not candidates:
            continue
        eligible_users += 1
        # Per-user RNG makes target selection independent of reservoir replacements.
        user_rng = random.Random(f"{config['seed']}:{uid}")
        selected = set(user_rng.sample(candidates, min(len(candidates), config["examples_per_user"])))
        entry = (uid, selected)
        if len(reservoir) < config["training_user_limit"]:
            reservoir.append(entry)
        else:
            index = rng.randrange(eligible_users)
            if index < config["training_user_limit"]:
                reservoir[index] = entry
    chosen = dict(reservoir)
    labels = Counter()
    users = set()
    errors = 0
    def examples():
        for uid, history in user_histories(data / "train_ratings.csv", cutoff):
            if uid not in chosen:
                continue
            for record in profile_examples(uid, history, chosen[uid], docs, manifest["config"]):
                if record["history_latest_timestamp"] >= record["target_timestamp"]:
                    raise AssertionError("Future rating entered training profile")
                labels[record["label"]] += 1
                users.add(uid)
                yield record
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, examples())
    if not labels:
        raise ValueError("No training examples produced; check data and minimum history")
    report = {
        **provenance(data, config), "schema_version": "p1-pointwise-v1",
        "training_population": dict(population), "eligible_users": eligible_users,
        "selected_users": len(users), "selected_user_ids": sorted(users),
        "exported_labels": {name: labels[name] for name in ("relevant", "not_relevant")},
        "examples": sum(labels.values()),
        "positive_fraction": labels["relevant"] / sum(labels.values()),
        "output_sha256": sha256(output), "error_count": errors,
        "runtime_seconds": time.perf_counter() - started,
        "limitations": [
            "Explicit rating supervision only; no weak negatives or automatic class balancing.",
            "This file has pointwise relevance labels, not listwise ranking or explanation targets.",
            "Metadata uses the shared static genome and training-cutoff tag aggregates, not per-target snapshots.",
            "Out-of-catalog rated movies count as history but cannot be rendered without shared movie documents.",
        ],
    }
    save_json(output.with_suffix(".report.json"), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--output", type=Path, default=Path("outputs/predictions/p1-training.jsonl"))
    parser.add_argument("--users", type=int, help="Override user limit for a smoke test")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.users is not None:
        if args.users < 1:
            parser.error("--users must be positive")
        config["training_user_limit"] = args.users
    report = build_examples(args.data, args.output, config)
    print(json.dumps({k: v for k, v in report.items() if k not in ("selected_user_ids", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
