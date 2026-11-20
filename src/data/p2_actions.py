"""Reference grammar and validators for P2 agent actions and episodes.

The action vocabulary and ordering rules mirror the proposed ``p2-trace-v1``
contract (``docs/phase2_interface_contract.md`` on ``member2-phase2``). Member 1
uses these checks on every training target; Member 4 may reuse them for
inference-time parsing. They check structure and binding, not truthfulness.
"""

from __future__ import annotations

import json

from src.profiles.agent_text import ACTION_FIELDS, ACTIONS

CONTROL_ACTIONS = ("assess_sufficiency", "ask_clarification", "evaluate_candidates", "retrieve_candidates")
MAX_CONTROL_EVENTS = 8


class ActionError(ValueError):
    """Raised when model output or a training target violates the action grammar."""


def _reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ActionError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value):
    raise ActionError(f"Non-finite JSON number: {value}")


def _positive_int(value) -> bool:
    return type(value) is int and value > 0


def _nonempty_str(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def parse_action(text: str) -> dict:
    """Strictly parse one JSON action object (no surrounding prose, no duplicate keys)."""
    try:
        action = json.loads(text, object_pairs_hook=_reject_duplicates, parse_constant=_reject_constant)
    except json.JSONDecodeError as error:
        raise ActionError(f"Invalid JSON: {error.msg}") from error
    validate_action(action)
    return action


def validate_action(action, *, candidate_ids: list[int] | None = None,
                    explain_ids: list[int] | None = None) -> None:
    """Check field names and types; optionally bind rank/explanation IDs to the prompt."""
    if not isinstance(action, dict):
        raise ActionError("Action must be a JSON object")
    name = action.get("action")
    if name not in ACTIONS:
        raise ActionError(f"Unknown action: {name!r}")
    expected = {"action", *ACTION_FIELDS[name]}
    if set(action) != expected:
        raise ActionError(f"{name} fields must be exactly {sorted(expected)}, got {sorted(action)}")
    if name == "assess_sufficiency" and type(action["sufficient"]) is not bool:
        raise ActionError("sufficient must be a Boolean")
    if name == "evaluate_candidates" and type(action["pool_sufficient"]) is not bool:
        raise ActionError("pool_sufficient must be a Boolean")
    if name == "ask_clarification" and not _nonempty_str(action["question"]):
        raise ActionError("question must be a nonempty string")
    if name == "retrieve_candidates" and not _nonempty_str(action["query"]):
        raise ActionError("query must be a nonempty string")
    if name == "rank_candidates":
        ids = action["movie_ids"]
        if not isinstance(ids, list) or not ids or not all(_positive_int(i) for i in ids):
            raise ActionError("movie_ids must be a nonempty list of positive integers")
        if len(set(ids)) != len(ids):
            raise ActionError("movie_ids contains duplicates")
        if candidate_ids is not None and sorted(ids) != sorted(candidate_ids):
            raise ActionError("movie_ids must be a permutation of the candidate pool")
    if name == "explain_recommendations":
        items = action["explanations"]
        if not isinstance(items, list) or not items:
            raise ActionError("explanations must be a nonempty list")
        for item in items:
            if not isinstance(item, dict) or set(item) != {"movie_id", "text"}:
                raise ActionError("Each explanation needs exactly movie_id and text")
            if not _positive_int(item["movie_id"]) or not _nonempty_str(item["text"]):
                raise ActionError("Explanation movie_id must be a positive integer and text nonempty")
        if explain_ids is not None and [item["movie_id"] for item in items] != list(explain_ids):
            raise ActionError("Explanations must cover the ranked movies in order")


def validate_episode(events: list[dict], initial_pool: list[int], *,
                     allow_fallback: bool = False) -> dict:
    """Validate ordering rules of one offline episode; returns summary counts.

    Rules follow ``p2-trace-v1``: at most eight control events; ``ask`` only
    after an insufficient assessment; ``evaluate`` only after a sufficient one
    and bound to the whole current pool; ``retrieve`` only after an insufficient
    pool judgment; a new assessment, question, or retrieval invalidates the pool
    judgment; exactly one ``rank`` then one ``explain`` event close the episode.
    """
    pool = list(initial_pool)
    sufficient = None
    pool_ok = None
    control = 0
    ranked = None
    explained = False
    counts = {name: 0 for name in ACTIONS}
    for index, event in enumerate(events):
        name = event.get("action")
        if name not in ACTIONS:
            raise ActionError(f"Event {index}: unknown action {name!r}")
        counts[name] += 1
        if name in CONTROL_ACTIONS:
            control += 1
            if ranked is not None:
                raise ActionError(f"Event {index}: control event after ranking")
            if control > MAX_CONTROL_EVENTS:
                raise ActionError("More than eight control events")
        if name == "assess_sufficiency":
            if type(event.get("sufficient")) is not bool:
                raise ActionError(f"Event {index}: sufficient must be Boolean")
            sufficient = event["sufficient"]
            pool_ok = None
        elif name == "ask_clarification":
            if sufficient is not False:
                raise ActionError(f"Event {index}: question without an insufficient assessment")
            for field in ("question", "answer", "answer_source"):
                if not _nonempty_str(event.get(field)):
                    raise ActionError(f"Event {index}: {field} must be nonempty")
            sufficient = None
            pool_ok = None
        elif name == "evaluate_candidates":
            if sufficient is not True:
                raise ActionError(f"Event {index}: pool judged without a sufficient assessment")
            if event.get("candidate_movie_ids") != pool:
                raise ActionError(f"Event {index}: candidate_movie_ids differ from current pool")
            if type(event.get("pool_sufficient")) is not bool:
                raise ActionError(f"Event {index}: pool_sufficient must be Boolean")
            pool_ok = event["pool_sufficient"]
        elif name == "retrieve_candidates":
            if pool_ok is not False:
                raise ActionError(f"Event {index}: retrieval without an insufficient pool judgment")
            added = event.get("added_movie_ids")
            if not isinstance(added, list) or not added or not all(_positive_int(i) for i in added):
                raise ActionError(f"Event {index}: added_movie_ids must be nonempty positive integers")
            if set(added) & set(pool) or len(set(added)) != len(added):
                raise ActionError(f"Event {index}: added IDs must be fresh and distinct")
            if not _nonempty_str(event.get("index_version")):
                raise ActionError(f"Event {index}: index_version must be nonempty")
            pool.extend(added)
            pool_ok = None
        elif name == "rank_candidates":
            if ranked is not None:
                raise ActionError(f"Event {index}: second ranking")
            if not allow_fallback and not (sufficient is True and pool_ok is True):
                raise ActionError(f"Event {index}: ranking before sufficient information and pool")
            ids = event.get("movie_ids")
            if not isinstance(ids, list) or not ids or len(set(ids)) != len(ids):
                raise ActionError(f"Event {index}: invalid ranked IDs")
            if not set(ids) <= set(pool) | set(event.get("injected_movie_ids", [])):
                raise ActionError(f"Event {index}: ranked IDs outside the final pool")
            ranked = ids
        elif name == "explain_recommendations":
            if ranked is None or explained:
                raise ActionError(f"Event {index}: explanation must follow exactly one ranking")
            items = event.get("explanations")
            expected = ranked[:min(10, len(ranked))]
            if not isinstance(items, list) or [item.get("movie_id") for item in items] != expected:
                raise ActionError(f"Event {index}: explanations must cover the top ranked IDs in order")
            if not all(_nonempty_str(item.get("text")) for item in items):
                raise ActionError(f"Event {index}: empty explanation text")
            explained = True
    if ranked is None or not explained:
        raise ActionError("Episode must end with one ranking and one explanation")
    return {"control_events": control, "final_pool_size": len(pool), **counts}
