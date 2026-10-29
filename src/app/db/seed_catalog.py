"""Load Member 1's frozen catalog into the `movies` table.

    python -m src.app.db.seed_catalog --source data/processed/ml25m-global-v1

Reads `movie_documents.jsonl` (one canonical document per catalog movie) and
upserts each row by `movie_id`, so re-running after a catalog refresh is safe.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.app.db.database import SessionLocal, engine
from src.app.db.models import Base, Movie

BATCH_SIZE = 500


def _iter_movies(source: Path):
    path = source / "movie_documents.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            yield {
                "movie_id": record["movie_id"],
                "title": record["title"],
                "genres": "|".join(record.get("genres", [])),
                "text": record["text"],
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
