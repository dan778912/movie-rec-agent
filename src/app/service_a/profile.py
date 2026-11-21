"""Builds live profile text from a session's watched items, using the exact
same formatter Member 1's offline pipeline uses (src/profiles/text.py), so
online and offline profiles stay comparable. Mirrors the item construction
and liked/disliked selection in src/data/build.py (_profile_items,
_export_profiles): top N liked (rating >= 4) by rating desc/recency desc/id,
top M disliked (rating <= 2.5) by rating asc/recency desc/id, each carrying
up to 3 genome tags + 2 user tags. N/M match configs/data_v1.json defaults.
"""

from __future__ import annotations

from sqlalchemy.orm import Session as DbSession

from src.app.db import repository as repo
from src.app.db.models import Movie, WatchedItem
from src.profiles.text import user_profile

LIKED_LIMIT = 10
DISLIKED_LIMIT = 5


def _timestamp(item: WatchedItem) -> float:
    moment = item.rated_at or item.created_at
    return moment.timestamp()


def _profile_item(item: WatchedItem, movie: Movie | None) -> dict:
    genres: list[str] = []
    tags: list[str] = []
    if movie is not None:
        genres = [] if movie.genres in ("", "(no genres listed)") else movie.genres.split("|")
        tags = list(movie.genome_tags[:3]) + list(movie.user_tags[:2])
    return {"title": item.title, "rating": item.rating, "genres": genres, "tags": tags}


def build_profile_text(db: DbSession, session_id: int) -> str:
    watched = repo.get_watched_items(db, session_id)
    movies = repo.get_movies_by_ids(db, (item.movie_id for item in watched))

    liked = sorted(
        (item for item in watched if item.rating >= 4.0),
        key=lambda item: (-item.rating, -_timestamp(item), item.movie_id),
    )[:LIKED_LIMIT]
    disliked = sorted(
        (item for item in watched if item.rating <= 2.5),
        key=lambda item: (item.rating, -_timestamp(item), item.movie_id),
    )[:DISLIKED_LIMIT]

    liked_items = [_profile_item(item, movies.get(item.movie_id)) for item in liked]
    disliked_items = [_profile_item(item, movies.get(item.movie_id)) for item in disliked]
    return user_profile(liked_items, disliked_items)
