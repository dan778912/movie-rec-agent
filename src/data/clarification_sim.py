"""Deterministic clarification-answer simulator: ``pre_cutoff_history_sim_v1``.

MovieLens has no conversational answers. This simulator answers an agent's
clarifying question using **only the user's available history** (ratings
strictly before the relevant cutoff) and prefers facts that the visible profile
text does not already show, as required by the Phase 2 protocol. It never
receives held-out targets. Member 1 uses it to write P2 training episodes; the
same functions can drive offline P2 evaluation so training and evaluation
answers come from one frozen implementation.
"""

from __future__ import annotations

import re
from collections import Counter

SIMULATOR_VERSION = "pre_cutoff_history_sim_v1"

# Fixed preference order for tie-breaking between intents.
INTENTS = ("liked_example", "favorite_genre", "themes", "disliked", "era")

QUESTION_TEMPLATES = {
    "liked_example": (
        "Can you name a movie you really enjoyed?",
        "What is one of your favorite movies?",
        "Is there a film you loved that I should know about?",
    ),
    "favorite_genre": (
        "Which movie genres do you enjoy most?",
        "What genres do you usually reach for when picking a movie?",
        "Which genres would you like more recommendations from?",
    ),
    "themes": (
        "What kinds of stories or themes do you like in a movie?",
        "Are there particular moods or themes you look for?",
        "What themes make a movie appealing to you?",
    ),
    "disliked": (
        "Are there genres or movies you would rather avoid?",
        "Is there anything you dislike in movies, such as a genre or a film you did not enjoy?",
        "Which movies or genres do you tend not to like?",
    ),
    "era": (
        "Do you prefer recent releases or older classics?",
        "Is there a particular decade of movies you enjoy most?",
        "Do you lean toward older films or newer ones?",
    ),
}

# Checked in order; the first matching intent wins.
_INTENT_PATTERNS = (
    ("disliked", re.compile(r"avoid|dislike|don't like|do not like|not enjoy|didn't enjoy|did not enjoy"
                            r"|not like|hate|tend not|turn(s)? you off")),
    ("era", re.compile(r"decade|\bera\b|older|newer|recent releases|classics?\b|old movies|new movies"
                       r"|time period|\byears?\b")),
    ("liked_example", re.compile(r"name a (movie|film)|favou?rite (movie|film)|(movie|film) you (really )?"
                                 r"(enjoyed|loved|liked)|film you loved|one of your favou?rite")),
    ("themes", re.compile(r"theme|stor(y|ies)|mood|tone|topic|subject|kinds? of movies?")),
    ("favorite_genre", re.compile(r"genre")),
)

UNRECOGNIZED_ANSWER = "I'm not sure how to answer that."
EXHAUSTED_ANSWER = "I don't have anything more to add about that."
NO_PREFERENCE_ANSWER = {
    "liked_example": "I can't think of another movie I loved right now.",
    "favorite_genre": "I don't have a strong genre preference.",
    "themes": "I don't look for any particular theme.",
    "disliked": "There is nothing in particular I avoid.",
    "era": "I enjoy movies from many different decades.",
}

LIKED_AT_LEAST = 4.0
DISLIKED_AT_MOST = 2.5


def classify_question(question: str) -> str | None:
    """Map free-form question text to a recognized intent, or ``None``."""
    text = question.lower()
    for intent, pattern in _INTENT_PATTERNS:
        if pattern.search(text):
            return intent
    return None


def question_for(intent: str, variant: int) -> str:
    templates = QUESTION_TEMPLATES[intent]
    return templates[variant % len(templates)]


def _join(values: list[str]) -> str:
    return values[0] if len(values) == 1 else ", ".join(values[:-1]) + " and " + values[-1]


def build_fact_pool(history: list[tuple], meta: dict, visible_ids: set[int], visible_text: str,
                    cutoff: int | None = None) -> dict[str, list[dict]]:
    """Ordered answerable facts per intent from available history only.

    ``history`` holds ``(movie_id, rating, timestamp)`` rows, all strictly before
    ``cutoff`` when one is given. ``visible_ids`` are movies already rendered in
    the profile; they are never revealed as example titles.
    """
    if cutoff is not None and any(ts >= cutoff for _, _, ts in history):
        raise ValueError("Simulator history contains a rating at or after the cutoff")
    lowered = visible_text.lower()
    known = [row for row in history if row[0] in meta]
    liked = [row for row in known if row[1] >= LIKED_AT_LEAST]
    pool: dict[str, list[dict]] = {intent: [] for intent in INTENTS}

    hidden_liked = sorted((row for row in liked if row[0] not in visible_ids),
                          key=lambda row: (-row[1], -row[2], row[0]))
    pool["liked_example"] = [
        {"kind": "liked_movie", "movie_id": mid, "title": meta[mid]["title"], "rating": rating,
         "source_movie_ids": [mid], "source_max_timestamp": ts}
        for mid, rating, ts in hidden_liked
    ]

    genre_support: dict[str, list[tuple]] = {}
    for row in liked:
        for genre in meta[row[0]]["genres"]:
            genre_support.setdefault(genre, []).append(row)
    genres = sorted(genre_support, key=lambda g: (-len(genre_support[g]), g))
    for start in range(0, len(genres), 2):
        chunk = genres[start:start + 2]
        rows = sorted({row for g in chunk for row in genre_support[g]}, key=lambda r: (r[2], r[0]))
        pool["favorite_genre"].append({
            "kind": "favorite_genres", "genres": chunk,
            "support": [len(genre_support[g]) for g in chunk],
            "source_movie_ids": [r[0] for r in rows], "source_max_timestamp": max(r[2] for r in rows)})

    tag_support: dict[str, list[tuple]] = {}
    for row in liked:
        for tag in dict.fromkeys(t.lower() for t in meta[row[0]]["tags"] if '"' not in t):
            tag_support.setdefault(tag, []).append(row)
    tags = sorted((t for t, rows in tag_support.items() if len(rows) >= 2),
                  key=lambda t: (t in lowered, -len(tag_support[t]), t))
    for start in range(0, len(tags), 2):
        chunk = tags[start:start + 2]
        rows = sorted({row for t in chunk for row in tag_support[t]}, key=lambda r: (r[2], r[0]))
        pool["themes"].append({
            "kind": "themes", "tags": chunk, "support": [len(tag_support[t]) for t in chunk],
            "source_movie_ids": [r[0] for r in rows], "source_max_timestamp": max(r[2] for r in rows)})

    rated_by_genre: dict[str, list[tuple]] = {}
    for row in known:
        for genre in meta[row[0]]["genres"]:
            rated_by_genre.setdefault(genre, []).append(row)
    favorite = set(genres[:2])
    disliked_genres = []
    for genre, rows in rated_by_genre.items():
        mean = sum(r[1] for r in rows) / len(rows)
        if len(rows) >= 3 and mean <= 2.75 and genre not in favorite:
            disliked_genres.append((mean, -len(rows), genre, rows))
    for mean, _, genre, rows in sorted(disliked_genres):
        rows = sorted(rows, key=lambda r: (r[2], r[0]))
        pool["disliked"].append({
            "kind": "disliked_genre", "genre": genre, "mean_rating": round(mean, 4), "count": len(rows),
            "source_movie_ids": [r[0] for r in rows], "source_max_timestamp": max(r[2] for r in rows)})
    hidden_disliked = sorted((row for row in known if row[1] <= DISLIKED_AT_MOST and row[0] not in visible_ids),
                             key=lambda row: (row[1], -row[2], row[0]))
    pool["disliked"].extend(
        {"kind": "disliked_movie", "movie_id": mid, "title": meta[mid]["title"], "rating": rating,
         "source_movie_ids": [mid], "source_max_timestamp": ts}
        for mid, rating, ts in hidden_disliked)

    dated = [row for row in liked if meta[row[0]]["decade"]]
    if len(dated) >= 3:
        decades = Counter(meta[row[0]]["decade"] for row in dated)
        decade, count = sorted(decades.items(), key=lambda pair: (-pair[1], pair[0]))[0]
        if count / len(dated) >= 0.4:
            rows = sorted((r for r in dated if meta[r[0]]["decade"] == decade), key=lambda r: (r[2], r[0]))
            pool["era"].append({
                "kind": "favorite_decade", "decade": decade, "share": round(count / len(dated), 4),
                "source_movie_ids": [r[0] for r in rows], "source_max_timestamp": max(r[2] for r in rows)})
    return pool


def answer_text(fact: dict) -> str:
    kind = fact["kind"]
    if kind == "liked_movie":
        return f"I really enjoyed {fact['title']} ({fact['rating']:g}/5)."
    if kind == "favorite_genres":
        return f"I mostly enjoy {_join(fact['genres'])} movies."
    if kind == "themes":
        return "I'm drawn to movies tagged " + _join([f'"{t}"' for t in fact["tags"]]) + "."
    if kind == "disliked_genre":
        return f"I usually don't enjoy {fact['genre']} movies."
    if kind == "disliked_movie":
        return f"I didn't enjoy {fact['title']} ({fact['rating']:g}/5)."
    if kind == "favorite_decade":
        return f"I mostly enjoy movies from the {fact['decade']}."
    raise ValueError(f"Unknown fact kind: {kind}")


def fact_terms(fact: dict) -> list[str]:
    """Strings a fact reveals; used to decide whether it is new to the visible text."""
    kind = fact["kind"]
    if kind in ("liked_movie", "disliked_movie"):
        return [fact["title"]]
    if kind == "favorite_genres":
        return list(fact["genres"])
    if kind == "themes":
        return list(fact["tags"])
    if kind == "disliked_genre":
        return [fact["genre"]]
    return [fact["decade"]]


def next_fact(intent: str, pool: dict[str, list[dict]], previous_questions: list[str]) -> dict | None:
    """Fact the simulator would reveal next for ``intent`` given earlier questions."""
    used = sum(1 for question in previous_questions if classify_question(question) == intent)
    facts = pool.get(intent, [])
    return facts[used] if used < len(facts) else None


def answer_question(question: str, pool: dict[str, list[dict]], previous_questions: list[str],
                    visible_text: str) -> dict:
    """Answer one question. Repeating an intent reveals that intent's next fact."""
    intent = classify_question(question)
    if intent is None:
        return {"answer": UNRECOGNIZED_ANSWER, "intent": None, "informative": False, "fact": None,
                "new_fact": False, "source_movie_ids": [], "simulator_version": SIMULATOR_VERSION}
    fact = next_fact(intent, pool, previous_questions)
    if fact is None:
        used = any(classify_question(q) == intent for q in previous_questions)
        text = EXHAUSTED_ANSWER if used else NO_PREFERENCE_ANSWER[intent]
        return {"answer": text, "intent": intent, "informative": False, "fact": None,
                "new_fact": False, "source_movie_ids": [], "simulator_version": SIMULATOR_VERSION}
    lowered = visible_text.lower()
    return {
        "answer": answer_text(fact), "intent": intent, "informative": True, "fact": fact,
        "new_fact": any(term.lower() not in lowered for term in fact_terms(fact)),
        "source_movie_ids": list(fact["source_movie_ids"]), "simulator_version": SIMULATOR_VERSION,
    }
