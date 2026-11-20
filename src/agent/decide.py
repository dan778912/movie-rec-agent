"""The callable entry point for Service B: one request in, "ask" or "recommend" out.

``decide`` and ``resume`` connect the pure controller (``controller.py``) to two collaborators the
caller supplies. The shapes below are PROPOSED for Member 3 to accept or change (decisions A7, A10);
nothing here has been agreed, and no real model or index exists yet.

    ModelClient.generate(prompt) -> str
        Owned by Member 3/6. Anything it raises (timeout, out of memory, bad checkpoint) is treated
        as a failed model call and handled like unusable output. Timeouts are the client's job.

    Retriever.initial_pool(profile_text, clarifications, history_movie_ids) -> RetrievalResult
    Retriever.search(query, exclude_movie_ids, max_new) -> RetrievalResult
        Owned by Member 3. ``initial_pool`` must return the B3-style ordered candidates with no
        watched movie. Whether clarification answers change that pool is the retriever's call; the
        controller never edits ``profile_text``. ``search`` must also exclude watched movies, but
        the controller re-filters defensively and records what it dropped.

    catalog: Mapping[int, Mapping]   movie_id -> {"movie_id", "title", "genres": [..], "tags": [..]}
        The documents shown to the model. Every retrieved movie needs an entry; a gap raises
        ``MissingCandidateError`` (a data bug, not something to paper over).

State is explicit in both directions. ``decide`` rebuilds an episode from the request, including the
clarifications Service A replays, so Service B keeps no memory between calls. ``resume`` instead
continues from a state returned by an earlier call, which preserves its full event history.

This module is NOT the offline evaluation runner (the loop over saved users that simulates answers
and writes prediction/trace/manifest files). Who owns that is still open (decision A12).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from src.agent import controller as ctl
from src.agent.config import AgentConfig
from src.agent.state import EpisodeState

MAX_STEPS = 100  # a safety net: the controller's own limits end an episode in far fewer steps


class RequestError(ValueError):
    """The request is malformed. A service would answer it with a client error."""


class InitialRetrievalError(RuntimeError):
    """The retriever could not produce the starting pool (the original exception is ``__cause__``).

    Unlike a later search failure, this has no fallback: the frozen B3 order the fallback would use is
    exactly what the retriever failed to supply, so there is nothing honest to recommend. A service
    should answer with its own error for this, and tell it apart from a malformed request."""


class ModelClient(Protocol):
    def generate(self, prompt: str) -> str: ...


@dataclass(frozen=True)
class RetrievalResult:
    movie_ids: list[int]
    index_version: str


class Retriever(Protocol):
    def initial_pool(self, *, profile_text: str, clarifications: Sequence[Mapping],
                     history_movie_ids: Sequence[int]) -> RetrievalResult: ...

    def search(self, *, query: str, exclude_movie_ids: Sequence[int], max_new: int) -> RetrievalResult: ...


@dataclass(frozen=True)
class Decision:
    """``response`` has the shape Service A's provisional client expects (``ask`` / ``recommend``),
    plus a proposed ``error`` form. ``state`` is the explicit state after this call."""

    response: dict
    state: EpisodeState


def decide(request: Mapping, *, model: ModelClient, retriever: Retriever, catalog: Mapping[int, Mapping],
           config: AgentConfig | None = None) -> Decision:
    """Handle one call. ``request`` has Service A's fields: ``profile_text``, ``history_movie_ids`` and
    ``clarifications`` (a list of ``{"question", "answer"}``); ``user_id`` and ``split`` are optional."""
    profile_text, history, turns, user_id, split = _read_request(request)
    try:
        pool = retriever.initial_pool(profile_text=profile_text, clarifications=turns, history_movie_ids=history)
    except Exception as error:
        raise InitialRetrievalError(f"initial retrieval failed: {type(error).__name__}: {error}") from error
    state = ctl.start_episode(profile_text, pool.movie_ids, history, config=config, clarifications=turns,
                              initial_index_version=pool.index_version, user_id=user_id, split=split)
    return advance(state, model=model, retriever=retriever, catalog=catalog)


def resume(state: EpisodeState | Mapping, answer: str, answer_source: str, *, model: ModelClient,
           retriever: Retriever, catalog: Mapping[int, Mapping]) -> Decision:
    """Continue from a state that was waiting for an answer. A dict is validated as untrusted input."""
    if not isinstance(state, EpisodeState):
        state = EpisodeState.from_dict(state)
    return advance(ctl.apply_answer(state, answer, answer_source), model=model, retriever=retriever, catalog=catalog)


def advance(state: EpisodeState, *, model: ModelClient, retriever: Retriever,
            catalog: Mapping[int, Mapping]) -> Decision:
    """Drive the controller until it needs a user answer or the episode ends."""
    for _ in range(MAX_STEPS):
        step = ctl.next_step(state)
        if step.kind == "model":
            try:
                raw = model.generate(ctl.build_prompt(state, catalog))
            except ctl.MissingCandidateError:
                raise  # a catalog gap is a data bug, not a model failure
            except Exception as error:  # any model failure is an ordinary, recorded outcome
                state = ctl.apply_model_error(state, f"{type(error).__name__}: {error}")
            else:
                state = ctl.apply_model_output(state, raw, catalog=catalog)
        elif step.kind == "answer":
            return Decision({"action": "ask", "question": step.question}, state)
        elif step.kind == "retrieval":
            exclude = sorted(set(state.history_movie_ids) | set(state.pool))
            try:
                found = retriever.search(query=step.query, exclude_movie_ids=exclude, max_new=step.max_new)
            except Exception as error:  # a retrieval failure is likewise recorded, then falls back
                state = ctl.apply_retrieval_failure(state, f"{type(error).__name__}: {error}")
            else:
                state = ctl.apply_retrieval(state, found.movie_ids, found.index_version)
        else:
            return Decision(_final_response(state), state)
    raise RuntimeError("controller did not finish; this indicates a bug")


def _read_request(request: Mapping) -> tuple[str, list[int], list[dict], int | None, str | None]:
    """Check everything the episode state would later refuse, so a bad request is a RequestError and
    not an internal TransitionError."""
    if not isinstance(request, Mapping):
        raise RequestError("request must be an object")
    profile_text, history = request.get("profile_text"), request.get("history_movie_ids")
    turns = request.get("clarifications", [])
    if not isinstance(profile_text, str):
        raise RequestError("profile_text must be a string")
    if not isinstance(history, list) or not all(type(i) is int and i > 0 for i in history):
        raise RequestError("history_movie_ids must be a list of positive integers")
    if len(set(history)) != len(history):
        raise RequestError("history_movie_ids must not repeat a movie")
    if not isinstance(turns, list) or not all(
        isinstance(t, Mapping) and isinstance(t.get("question"), str) and isinstance(t.get("answer"), str)
        and t["question"].strip() and t["answer"].strip() for t in turns
    ):
        raise RequestError("clarifications must be a list of {question, answer} with nonempty text")
    if not all(t.get("answer_source") is None or _nonempty_text(t["answer_source"]) for t in turns):
        raise RequestError("a clarification's answer_source, when given, must be a nonempty string")
    user_id, split = request.get("user_id"), request.get("split")
    if user_id is not None and not (type(user_id) is int and user_id > 0):
        raise RequestError("user_id, when given, must be a positive integer")
    if split is not None and not _nonempty_text(split):
        raise RequestError("split, when given, must be a nonempty string")
    return profile_text, list(history), [dict(t) for t in turns], user_id, split


def _nonempty_text(value) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _final_response(state: EpisodeState) -> dict:
    if state.termination_reason == "failed":
        # No usable explanation, so there is nothing honest to show. The caller must surface a failure.
        return {"action": "error", "code": "failed", "reason": state.reason}
    explained = next(e for e in reversed(state.events) if e["action"] == "explain_recommendations")
    texts = {item["movie_id"]: item["text"] for item in explained["explanations"]}
    size = len(state.ranked)
    # The model gives an order, not scores, so the score is a rank surrogate: 1.0 for rank 1, falling
    # linearly. It is not a probability and should be labelled rank_surrogate in any manifest.
    items = [{"movie_id": movie_id, "score": round(1.0 - index / size, 6), "rank": index + 1,
              "explanation": texts[movie_id]}
             for index, movie_id in enumerate(state.ranked[: state.config.explain_count])]
    return {"action": "recommend", "items": items}
