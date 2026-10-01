"""Deterministic, shared text formatting for MovieLens metadata and histories."""

from __future__ import annotations


def movie_document(title: str, genres: str, genome_tags: list[str], user_tags: list[str]) -> str:
    parts = [title, f"Genres: {', '.join(genres.split('|')) if genres != '(no genres listed)' else 'unspecified'}"]
    if genome_tags:
        parts.append("Genome tags: " + ", ".join(genome_tags))
    if user_tags:
        parts.append("User tags: " + ", ".join(user_tags))
    return ". ".join(parts) + "."


def user_profile(liked: list[dict], disliked: list[dict]) -> str:
    if liked:
        likes = "; ".join(
            f"{item['title']} ({item['rating']:g}/5; "
            f"genres: {', '.join(item['genres']) or 'unspecified'}; "
            f"tags: {', '.join(item['tags']) or 'none'})"
            for item in liked
        )
        text = "Highly rated movies: " + likes + "."
    else:
        text = "No highly rated movies in the available history."
    if disliked:
        dislikes = "; ".join(f"{item['title']} ({item['rating']:g}/5)" for item in disliked)
        text += " Low rated movies: " + dislikes + "."
    return text
