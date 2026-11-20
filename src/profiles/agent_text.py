"""Deterministic prompt text for the P2 agent's six structured actions.

Training samples (Member 1) and live/offline inference (Member 4's agent) must
render identical prompts, so both import this module. Changing any string here
requires a new ``PROMPT_VERSION``.
"""

from __future__ import annotations

import json

PROMPT_VERSION = "p2-prompt-v1"

ACTIONS = (
    "assess_sufficiency",
    "ask_clarification",
    "evaluate_candidates",
    "retrieve_candidates",
    "rank_candidates",
    "explain_recommendations",
)

SYSTEM_TEXT = (
    "You are a movie recommendation agent. Use only the user information and MovieLens catalog "
    "metadata given below. Reply with exactly one JSON object and nothing else."
)

TASK_TEXT = {
    "assess_sufficiency": (
        "Decide whether the preference information above is sufficient to recommend movies, "
        "or whether a clarifying question is needed first.",
        '{"action": "assess_sufficiency", "sufficient": true or false}',
    ),
    "ask_clarification": (
        "The preference information is not sufficient. Ask one short clarifying question that would "
        "most improve the recommendations. Do not repeat a question that was already asked.",
        '{"action": "ask_clarification", "question": "<question>"}',
    ),
    "evaluate_candidates": (
        "Decide whether the candidate pool is good enough to rank for this user, or whether a wider "
        "retrieval is needed.",
        '{"action": "evaluate_candidates", "pool_sufficient": true or false}',
    ),
    "retrieve_candidates": (
        "The candidate pool is not good enough. Write a search query describing the movies to retrieve, "
        "using only the preferences above.",
        '{"action": "retrieve_candidates", "query": "<search query>"}',
    ),
    "rank_candidates": (
        "Rank every candidate from most to least likely to be enjoyed by the user. "
        "Include each movie ID exactly once.",
        '{"action": "rank_candidates", "movie_ids": [<movie_id>, ...]}',
    ),
    "explain_recommendations": (
        "Write one short justification for each recommended movie, in the given order. "
        "Cite only titles, ratings, genres, tags, and answers shown above.",
        '{"action": "explain_recommendations", "explanations": '
        '[{"movie_id": <movie_id>, "text": "<justification>"}, ...]}',
    ),
}

# Fields each action carries, in serialization order (``action`` first).
ACTION_FIELDS = {
    "assess_sufficiency": ("sufficient",),
    "ask_clarification": ("question",),
    "evaluate_candidates": ("pool_sufficient",),
    "retrieve_candidates": ("query",),
    "rank_candidates": ("movie_ids",),
    "explain_recommendations": ("explanations",),
}

CANDIDATE_TASKS = {"evaluate_candidates", "retrieve_candidates", "rank_candidates"}


def display_tags(genome_tags: list[str], user_tags: list[str]) -> list[str]:
    """Same tag slice as canonical profile items: three genome tags then two user tags."""
    return list(genome_tags[:3]) + list(user_tags[:2])


def candidate_line(movie: dict) -> str:
    """One compact catalog line: ``[id] Title (Year) | genres: ... | tags: ...``."""
    genres = ", ".join(movie["genres"]) or "unspecified"
    tags = ", ".join(movie["tags"]) or "none"
    return f"[{movie['movie_id']}] {movie['title']} | genres: {genres} | tags: {tags}"


def render_prompt(task: str, profile_text: str, clarifications: list[dict],
                  candidates: list[dict] | None = None) -> str:
    """Render the model input for one agent decision.

    ``clarifications`` are ``{"question", "answer"}`` dicts in turn order.
    ``candidates`` are dicts with ``movie_id``, ``title``, ``genres`` and ``tags``;
    they are required for candidate tasks and for explanations.
    """
    if task not in TASK_TEXT:
        raise ValueError(f"Unknown task: {task}")
    instruction, output_format = TASK_TEXT[task]
    parts = [SYSTEM_TEXT, "", "User profile:", profile_text, "", "Clarifications:"]
    if clarifications:
        for turn in clarifications:
            parts.append(f"Q: {turn['question']}")
            parts.append(f"A: {turn['answer']}")
    else:
        parts.append("None so far.")
    if task in CANDIDATE_TASKS or task == "explain_recommendations":
        if not candidates:
            raise ValueError(f"{task} requires candidates")
        header = ("Recommended movies in rank order" if task == "explain_recommendations"
                  else "Candidate pool")
        parts.extend(["", f"{header} ({len(candidates)} movies):"])
        parts.extend(candidate_line(movie) for movie in candidates)
    elif candidates:
        raise ValueError(f"{task} does not take candidates")
    parts.extend(["", f"Task: {instruction}", f"Output format: {output_format}"])
    return "\n".join(parts)


def serialize_action(action: dict) -> str:
    """Canonical completion text: ``action`` first, then fields in contract order."""
    name = action["action"]
    ordered = {"action": name}
    for field in ACTION_FIELDS[name]:
        ordered[field] = action[field]
    if set(action) != set(ordered):
        raise ValueError(f"Unexpected fields for {name}: {sorted(set(action) - set(ordered))}")
    return json.dumps(ordered, ensure_ascii=False)
