"""Verify processed checksums and the no-future-profile invariant."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.data.build import sha256


def verify(output: Path) -> dict:
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    stats = json.loads((output / "statistics.json").read_text(encoding="utf-8"))
    for name, expected in manifest["file_sha256"].items():
        actual = sha256(output / name)
        if actual != expected:
            raise AssertionError(f"Checksum mismatch: {name}")
    catalog = set()
    with (output / "catalog.csv").open(encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            catalog.add(int(row["movie_id"]))
    if len(catalog) != stats["catalog_movies"]:
        raise AssertionError("Catalog count mismatch")
    result = {"version": manifest["version"], "checksums_verified": len(manifest["file_sha256"])}
    profile_histories = {}
    for split in ("validation", "test"):
        cutoff = stats[f"{split}_cutoff_unix"]
        profiles = 0
        expected = {}
        with (output / f"{split}_profiles.jsonl").open(encoding="utf-8") as source:
            for line in source:
                record = json.loads(line)
                if record["split"] != split or record["history_latest_timestamp"] >= cutoff:
                    raise AssertionError(f"Future history in {split} profile for user {record['user_id']}")
                seen = set(record["history_movie_ids"])
                if len(seen) != len(record["history_movie_ids"]):
                    raise AssertionError("Duplicate history movie")
                for item in record["liked_movies"] + record["disliked_movies"]:
                    if item["timestamp"] >= cutoff or item["movie_id"] not in seen:
                        raise AssertionError("Selected profile item is not in available history")
                expected[record["user_id"]] = seen
                profiles += 1
        if profiles != stats["evaluation"][split].get("users_with_in_catalog_positive", 0):
            raise AssertionError("Profile count mismatch")
        result[f"{split}_profiles_verified"] = profiles
        profile_histories[split] = expected
    actual = {
        split: {user_id: set() for user_id in profiles}
        for split, profiles in profile_histories.items()
    }
    for filename, recipients in (("train_ratings.csv", ("validation", "test")),
                                 ("validation_ratings.csv", ("test",))):
        with (output / filename).open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                user_id, movie_id = int(row["user_id"]), int(row["movie_id"])
                for split in recipients:
                    if user_id in actual[split]:
                        actual[split][user_id].add(movie_id)
    for split in ("validation", "test"):
        for user_id, expected in profile_histories[split].items():
            if actual[split][user_id] != expected:
                raise AssertionError(f"{split} history mismatch for user {user_id}")
        with (output / f"{split}_ratings.csv").open(encoding="utf-8", newline="") as source:
            for row in csv.DictReader(source):
                user_id, movie_id = int(row["user_id"]), int(row["movie_id"])
                if user_id in profile_histories[split] and movie_id in profile_histories[split][user_id]:
                    raise AssertionError(f"{split} interaction already appears in profile for user {user_id}")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/processed/ml25m-global-v1"))
    args = parser.parse_args()
    print(json.dumps(verify(args.output), indent=2))


if __name__ == "__main__":
    main()
