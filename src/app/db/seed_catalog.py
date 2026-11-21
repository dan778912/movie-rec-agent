"""Load Member 1's frozen catalog into the `movies` table.

    python -m src.app.db.seed_catalog --source data/processed/ml25m-global-v1

Reads `movie_documents.jsonl` (one canonical document per catalog movie) and
`train_ratings.csv` (real per-user ratings, training split only) and upserts
each row by `movie_id`, so re-running after a catalog refresh is safe.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.app.db.database import SessionLocal, engine
from src.app.db.models import Base, Movie

BATCH_SIZE = 500


def _rating_aggregates(source: Path) -> dict[int, tuple[float, int]]:
    """mean and count of TRAINING ratings per movie_id. Training only, by the
    same split Member 1 froze — not validation/test, so this cannot leak
    held-out labels. ~20M rows; one pass, O(num_movies) memory."""

    sums: dict[int, float] = {}
    counts: dict[int, int] = {}
    path = source / "train_ratings.csv"
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            movie_id = int(row["movie_id"])
            rating = float(row["rating"])
            sums[movie_id] = sums.get(movie_id, 0.0) + rating
            counts[movie_id] = counts.get(movie_id, 0) + 1
    return {movie_id: (sums[movie_id] / counts[movie_id], counts[movie_id]) for movie_id in sums}


def _iter_movies(source: Path):
    ratings = _rating_aggregates(source)
    path = source / "movie_documents.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            movie_id = record["movie_id"]
            rating_avg, rating_count = ratings.get(movie_id, (None, None))
            yield {
                "movie_id": movie_id,
                "title": record["title"],
                "genres": "|".join(record.get("genres", [])),
                "text": record["text"],
                "genome_tags": record.get("genome_tags", []),
                "user_tags": record.get("user_tags", []),
                "rating_avg": rating_avg,
                "rating_count": rating_count,
            }


def seed(source: Path) -> int:
    Base.metadata.create_all(bind=engine, tables=[Movie.__table__])
    db = SessionLocal()
    count = 0
    try:
        for record in _iter_movies(source):
            db.merge(Movie(**record))
            count += 1
            if count % BATCH_SIZE == 0:
                db.commit()
        db.commit()
    finally:
        db.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("data/processed/ml25m-global-v1"),
        help="Directory containing Member 1's movie_documents.jsonl",
    )
    args = parser.parse_args()
    total = seed(args.source)
    print(f"Seeded {total} movies from {args.source}")


if __name__ == "__main__":
    main()
