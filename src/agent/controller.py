"""Controller transitions for one P2 agent episode.

This is a "sans-IO" state machine: it never calls a model, a retriever, a database or the network.
The caller (Service B, or an offline runner, or a test) does the I/O and feeds results back:

    state = start_episode(profile_text, initial_pool, history_ids, config=...)
    while True:
        step = next_step(state)
        if step.kind == "model":       # render prompt -> call model -> hand back raw text
            state = apply_model_output(state, model(build_prompt(state, catalog)))
        elif step.kind == "answer":    # show step.question to the user (or the simulator)
            state = apply_answer(state, answer, answer_source)
        elif step.kind == "retrieval": # search for step.query, at most step.max_new new movies
            state = apply_retrieval(state, movie_ids, index_version)
        else:                          # step.kind == "done"
            break

Every function returns a NEW state and leaves its input untouched. All limits live here, not in the
prompt: the model can ask for a third question, but this code will not allow it. Limits and policies
come from ``AgentConfig`` and are unratified (see ``docs/member4_agent_decisions.md``).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from src.agent.actions import AgentActionError, parse_model_output
from src.agent.config import AgentConfig
from src.agent.grounding import UNGROUNDED, UNSUPPORTED, Evidence, check_explanations, repair_text
from src.agent.state import CALLER_PHASES, PHASE_TASK, EpisodeState, StateError
from src.profiles.agent_text import CANDIDATE_TASKS, render_prompt

FALLBACK_SOURCE = "frozen_b3_order"  # the only fallback source the proposed v1 contract names
DEFAULT_ANSWER_SOURCE = "client_supplied"  # for clarifications replayed from an earlier call

# Why a step could not continue. Recorded in ``failures`` and in ``reason``.
CLARIFICATION_BUDGET = "clarification_budget_exhausted"
RETRIEVAL_BUDGET = "retrieval_budget_exhausted"
POOL_FULL = "pool_size_cap_reached"
EVENT_CAP = "control_event_cap_reached"
NO_NEW_CANDIDATES = "retrieval_returned_no_new_candidates"


class TransitionError(RuntimeError):
    """The caller used a function in the wrong phase or passed unusable arguments (a caller bug)."""


class MissingCandidateError(KeyError):
    """The catalog has no document for a movie the prompt must show."""


@dataclass(frozen=True)
class NextStep:
    kind: str  # "model" | "answer" | "retrieval" | "done"
    task: str | None = None  # kind == "model": which of the six actions to request
    question: str | None = None  # kind == "answer": the question to put to the user
    query: str | None = None  # kind == "retrieval": the model's search text
    max_new: int | None = None  # kind == "retrieval": room left in the pool
    attempt: int = 1  # kind == "model": 1 for the first try, 2 after one unusable output, ...


# --------------------------------------------------------------------------- starting an episode
def start_episode(
    profile_text: str,
    initial_pool: Sequence[int],
    history_movie_ids: Sequence[int],
    *,
    config: AgentConfig | None = None,
    clarifications: Sequence[Mapping] = (),
    initial_index_version: str | None = None,
    user_id: int | None = None,
    split: str | None = None,
) -> EpisodeState:
    """Begin an episode, or resume one from replayed clarifications.

    ``clarifications`` are the ``{"question", "answer"}`` turns an earlier call already collected
    (Service A resends them on every call because Service B is stateless). They count against the
    budget, so a resumed conversation cannot be given fresh questions. Each earlier question took
    one assessment and one question event, which is why ``control_events`` starts at twice their
    number. Those earlier events are not reconstructed: the live trace is assembled across calls.

    If that leaves no room for the one assessment and one pool evaluation any recommendation needs,
    no valid episode can follow, so the episode is treated as budget-exhausted before any model call
    (the configured ``exhaustion_policy`` decides what that means), instead of running judgments
    that would breach the event cap.
    """
    config = config or AgentConfig()
    turns = []
    for turn in clarifications:
        question, answer = turn.get("question"), turn.get("answer")
        if not (isinstance(question, str) and question.strip() and isinstance(answer, str) and answer.strip()):
            raise TransitionError("each clarification needs a nonempty question and answer")
        turns.append({"question": question, "answer": answer,
                      "answer_source": turn.get("answer_source") or DEFAULT_ANSWER_SOURCE})
    state = EpisodeState(
        config=config,
        profile_text=profile_text,
        history_movie_ids=list(history_movie_ids),
        initial_pool=list(initial_pool),
        pool=list(initial_pool),
        clarifications=turns,
        resumed_clarifications=len(turns),
        control_events=2 * len(turns),
        initial_index_version=initial_index_version,
        user_id=user_id,
        split=split,
    )
    try:
        state.check_invariants()
        EpisodeState.from_dict(state.to_dict())  # also checks types and duplicates
    except StateError as error:
        raise TransitionError(f"cannot start an episode: {error}") from error
    if state.control_events + 2 > config.max_control_events:  # an assessment and a pool evaluation
        _on_budget_exhausted(state, EVENT_CAP)
    return state


# --------------------------------------------------------------------------- what to do next
def next_step(state: EpisodeState) -> NextStep:
    phase = state.phase
    if phase in PHASE_TASK:
        return NextStep("model", task=PHASE_TASK[phase], attempt=state.step_invalid_count + 1)
    if phase == "await_answer":
        return NextStep("answer", question=state.pending_question)
    if phase == "await_retrieval":
        room = state.config.max_pool_size - len(state.pool)
        return NextStep("retrieval", query=state.pending_query, max_new=max(room, 0))
    return NextStep("done")


def build_prompt(state: EpisodeState, catalog: Mapping[int, Mapping]) -> str:
    """Render the model input for the current step with Member 1's shared ``render_prompt``.

    Only the profile text, the clarification questions/answers and the displayed candidates' own
    catalog fields (title, genres, tags) reach the prompt. The watched-history list is used for
    exclusion only and is never shown.
    """
    task = _model_task(state)
    if task in CANDIDATE_TASKS:
        shown = state.pool
    elif task == "explain_recommendations":
        shown = state.ranked[: state.config.explain_count]
    else:
        shown = None
    candidates = None
    if shown is not None:
        missing = [movie_id for movie_id in shown if movie_id not in catalog]
        if missing:
            raise MissingCandidateError(f"no catalog entry for movies {missing[:5]}")
        candidates = [catalog[movie_id] for movie_id in shown]
    turns = [{"question": turn["question"], "answer": turn["answer"]} for turn in state.clarifications]
    return render_prompt(task, state.profile_text, turns, candidates)


# --------------------------------------------------------------------------- model output
def apply_model_output(state: EpisodeState, raw_text: str, *, catalog: Mapping[int, Mapping] | None = None) -> EpisodeState:
    """Parse and apply one model output for the step ``next_step`` requested.

    ``catalog`` (movie_id -> document with title, genres, tags) is needed only for the explanation
    step, where the grounding screen compares the text with the movies that were displayed. It is
    required there unless ``grounding_policy`` is ``"off"``.
    """
    task = _model_task(state)
    new = state.copy()
    config = new.config
    digest = hashlib.sha256(raw_text.encode("utf-8")).hexdigest() if isinstance(raw_text, str) else None
    try:
        action = parse_model_output(
            raw_text,
            task,
            pool_ids=new.pool,
            explain_ids=None if new.ranked is None else new.ranked[: config.explain_count],
            previous_questions=[turn["question"] for turn in new.clarifications],
            reject_repeated_questions=config.reject_repeated_questions,
        )
    except AgentActionError as error:
        return _on_invalid(new, task, digest, error)

    if task == "explain_recommendations":
        return _apply_explanation(new, action, digest, catalog)
    new.decisions.append({"index": len(new.decisions), "source": "model", "task": task,
                          "raw_sha256": digest, "valid": True, "error": None, "parsed": action})
    new.step_invalid_count = 0
    _HANDLERS[task](new, action)
    return new


def apply_model_error(state: EpisodeState, reason: str) -> EpisodeState:
    """The model call itself failed (exception, timeout). It follows the same path as an unusable
    output: retries if configured, then the contract fallback, or "failed" for the explanation."""
    task = _model_task(state)
    detail = (str(reason) or "unspecified")[:200]
    return _on_invalid(state.copy(), task, None, AgentActionError("model_error", detail))


def _on_assess(state: EpisodeState, action: dict) -> None:
    state.sufficient = action["sufficient"]
    state.pool_ok = None
    state.control_events += 1
    state.events.append({"action": "assess_sufficiency", "sufficient": state.sufficient})
    if state.sufficient:
        state.phase = "evaluate"
        return
    blocker = _cannot_ask(state)
    if blocker:
        _on_budget_exhausted(state, blocker)
    else:
        state.phase = "ask"


def _on_ask(state: EpisodeState, action: dict) -> None:
    # The event is recorded only when the answer arrives: the trace event carries both.
    state.pending_question = action["question"]
    state.phase = "await_answer"


def _on_evaluate(state: EpisodeState, action: dict) -> None:
    state.pool_ok = action["pool_sufficient"]
    state.control_events += 1
    state.events.append({"action": "evaluate_candidates", "candidate_movie_ids": list(state.pool),
                         "pool_sufficient": state.pool_ok})
    if state.pool_ok:
        state.phase = "rank"
        return
    blocker = _cannot_retrieve(state)
    if blocker:
        _on_budget_exhausted(state, blocker)
    else:
        state.phase = "retrieve"


def _on_retrieve(state: EpisodeState, action: dict) -> None:
    # Likewise recorded after the retrieval, once the added movies and index version are known.
    state.pending_query = action["query"]
    state.phase = "await_retrieval"


def _on_rank(state: EpisodeState, action: dict) -> None:
    # The model ranks the whole pool (the prompt says so); the trace and prediction keep the top
    # ``ranking_size`` (the proposed v1 contract's 50). Truncating is decision A5.
    state.ranked = action["movie_ids"][: state.config.ranking_size]
    state.events.append({"action": "rank_candidates", "movie_ids": list(state.ranked)})
    state.phase = "explain"


_HANDLERS = {
    "assess_sufficiency": _on_assess,
    "ask_clarification": _on_ask,
    "evaluate_candidates": _on_evaluate,
    "retrieve_candidates": _on_retrieve,
    "rank_candidates": _on_rank,
}  # explanations are handled by _apply_explanation, which also runs the grounding screen


def _apply_explanation(state: EpisodeState, action: dict, digest, catalog) -> EpisodeState:
    """Screen the explanations against what the model was shown, apply ``grounding_policy``, record.

    Only "unsupported" results trigger repair or rejection; "needs_review", "insufficient_evidence"
    and "no_checkable_claim" are recorded for the audit and never change the text.
    """
    policy = state.config.grounding_policy
    explanations = [dict(item) for item in action["explanations"]]
    attempt = state.step_invalid_count + 1
    if policy == "off":
        state.grounding.append({"attempt": attempt, "policy": policy, "outcome": "unchecked", "results": []})
    else:
        if catalog is None:
            raise TransitionError("catalog is required to screen explanations (or set grounding_policy='off')")
        shown_ids = state.ranked[: state.config.explain_count]
        missing = [movie_id for movie_id in shown_ids if movie_id not in catalog]
        if missing:
            raise MissingCandidateError(f"no catalog entry for movies {missing[:5]}")
        evidence = Evidence(state.profile_text, tuple(turn["answer"] for turn in state.clarifications),
                            {movie_id: catalog[movie_id] for movie_id in shown_ids})
        results = check_explanations(explanations, evidence)
        flagged = [index for index, result in enumerate(results) if result["status"] == UNSUPPORTED]
        outcome = "kept"
        if flagged and policy == "reject":
            outcome = "rejected"
        elif flagged and policy == "repair":
            outcome = "repaired"
            for index in flagged:
                results[index]["repaired"] = True
                results[index]["original_sha256"] = hashlib.sha256(explanations[index]["text"].encode("utf-8")).hexdigest()
                explanations[index]["text"] = repair_text(catalog[explanations[index]["movie_id"]])
        state.grounding.append({"attempt": attempt, "policy": policy, "outcome": outcome, "results": results})
        if outcome == "rejected":
            message = f"{len(flagged)} explanation(s) cite evidence that was not shown"
            return _on_invalid(state, "explain_recommendations", digest, AgentActionError(UNGROUNDED, message))
    state.decisions.append({"index": len(state.decisions), "source": "model", "task": "explain_recommendations",
                            "raw_sha256": digest, "valid": True, "error": None, "parsed": action})
    state.step_invalid_count = 0
    state.events.append({"action": "explain_recommendations", "explanations": explanations})
    if state.termination_reason is None:
        state.termination_reason = "recommend"
    state.phase = "done"
    return state


def _on_invalid(state: EpisodeState, task: str, digest, error: AgentActionError) -> EpisodeState:
    state.decisions.append({"index": len(state.decisions), "source": "model", "task": task,
                            "raw_sha256": digest, "valid": False,
                            "error": {"code": error.code, "message": error.message}, "parsed": None})
    state.failures.append({"code": error.code, "stage": task, "detail": error.message})
    state.step_invalid_count += 1
    if state.step_invalid_count <= state.config.max_invalid_actions_per_step:
        return state  # same phase: the caller may try once more
    state.step_invalid_count = 0
    if task == "explain_recommendations":
        # No contract rule says what to do with a ranking whose explanations are unusable, and
        # inventing text here would defeat the grounding audit. Report it (decision A4).
        _finish_failed(state, f"explanation_invalid:{error.code}")
    elif error.code == "model_error":
        _to_fallback(state, f"model_error:{task}")  # the call itself failed; the output was never judged
    else:
        _to_fallback(state, f"invalid_action:{task}:{error.code}")
    return state


# --------------------------------------------------------------------------- caller-supplied results
def apply_answer(state: EpisodeState, answer: str, answer_source: str) -> EpisodeState:
    """Record the answer to the pending question and return to the sufficiency check."""
    if state.phase != "await_answer":
        raise TransitionError(f"no question is pending (phase is {state.phase!r})")
    if not (isinstance(answer, str) and answer.strip()):
        raise TransitionError("answer must be a nonempty string")
    if not (isinstance(answer_source, str) and answer_source.strip()):
        raise TransitionError("answer_source must be a nonempty string")
    new = state.copy()
    question = new.pending_question
    new.clarifications.append({"question": question, "answer": answer, "answer_source": answer_source})
    new.events.append({"action": "ask_clarification", "question": question, "answer": answer,
                       "answer_source": answer_source})
    new.control_events += 1
    new.pending_question = None
    new.sufficient = None  # new information: sufficiency and the pool must be judged again
    new.pool_ok = None
    new.phase = "assess"
    return new


def apply_retrieval(state: EpisodeState, candidate_ids: Sequence[int], index_version: str) -> EpisodeState:
    """Add retrieved movies to the pool.

    The retrieval attempt is consumed even if it adds nothing. IDs already in the pool, in the
    watched history, repeated, or beyond the pool-size cap are dropped and counted in
    ``retrieval_log``. Who is responsible for this filtering is still open (decision A7), so the
    controller does not rely on the retriever to do it.
    """
    if state.phase != "await_retrieval":
        raise TransitionError(f"no retrieval is pending (phase is {state.phase!r})")
    if not (isinstance(index_version, str) and index_version.strip()):
        raise TransitionError("index_version must be a nonempty string")
    if not isinstance(candidate_ids, (list, tuple)) or not all(
        type(item) is int and item > 0 for item in candidate_ids
    ):
        raise TransitionError("candidate_ids must be a list of positive integers")

    new = state.copy()
    watched, in_pool, seen = set(new.history_movie_ids), set(new.pool), set()
    fresh, counts = [], {"duplicate": 0, "already_in_pool": 0, "watched": 0, "over_capacity": 0}
    for movie_id in candidate_ids:
        if movie_id in seen:
            counts["duplicate"] += 1
        elif movie_id in in_pool:
            counts["already_in_pool"] += 1
        elif movie_id in watched:
            counts["watched"] += 1
        else:
            fresh.append(movie_id)
        seen.add(movie_id)
    room = max(new.config.max_pool_size - len(new.pool), 0)
    counts["over_capacity"] = max(len(fresh) - room, 0)
    fresh = fresh[:room]

    new.retrievals += 1
    new.retrieval_log.append({"query": new.pending_query, "index_version": index_version,
                              "returned": len(candidate_ids), "added": len(fresh),
                              **{f"dropped_{name}": count for name, count in counts.items()}})
    new.pending_query = None
    if not fresh:
        new.failures.append({"code": "retrieval_empty", "stage": "retrieve_candidates",
                             "detail": NO_NEW_CANDIDATES})
        _on_budget_exhausted(new, NO_NEW_CANDIDATES)
        return new
    new.control_events += 1
    new.events.append({"action": "retrieve_candidates", "added_movie_ids": list(fresh),
                       "index_version": index_version})
    new.pool.extend(fresh)
    new.pool_ok = None  # the judgment was about the old pool
    new.phase = "evaluate"
    return new


def apply_retrieval_failure(state: EpisodeState, reason: str) -> EpisodeState:
    """The retriever raised or timed out. The proposed contract allows the frozen-B3 fallback for a
    retrieval error, and this applies regardless of ``exhaustion_policy`` (that policy covers
    spent budgets, not errors)."""
    if state.phase != "await_retrieval":
        raise TransitionError(f"no retrieval is pending (phase is {state.phase!r})")
    new = state.copy()
    detail = (str(reason) or "unspecified")[:200]
    new.retrievals += 1
    new.pending_query = None
    new.failures.append({"code": "retrieval_error", "stage": "retrieve_candidates", "detail": detail})
    _to_fallback(new, f"retrieval_error:{detail}")
    return new


# --------------------------------------------------------------------------- limits and endings
def _cannot_ask(state: EpisodeState) -> str | None:
    """Why another question is not allowed, or None. A question only helps if the assessment and
    pool evaluation that must follow it still fit under the event cap."""
    if state.asks >= state.config.max_clarifications:
        return CLARIFICATION_BUDGET
    if state.control_events + 3 > state.config.max_control_events:  # ask + assess + evaluate
        return EVENT_CAP
    return None


def _cannot_retrieve(state: EpisodeState) -> str | None:
    if state.retrievals >= state.config.max_retrievals:
        return RETRIEVAL_BUDGET
    if len(state.pool) >= state.config.max_pool_size:
        return POOL_FULL
    if state.control_events + 2 > state.config.max_control_events:  # retrieve + evaluate
        return EVENT_CAP
    return None


def _on_budget_exhausted(state: EpisodeState, reason: str) -> None:
    """A judgment is still "no" and no budget remains. Which rule applies is decision A1."""
    state.failures.append({"code": "budget_exhausted", "stage": state.phase, "detail": reason})
    if state.config.exhaustion_policy == "fallback_b3_order":
        _to_fallback(state, reason)
    else:  # rank_available_pool
        state.termination_reason = "budget_exhausted_rank"
        state.reason = reason
        state.phase = "rank"


def _to_fallback(state: EpisodeState, reason: str) -> None:
    """The proposed v1 fallback: the frozen B3 order, still followed by explanations."""
    state.ranked = state.initial_pool[: state.config.ranking_size]
    state.events.append({"action": "rank_candidates", "movie_ids": list(state.ranked)})
    state.decisions.append({"index": len(state.decisions), "source": "fallback", "task": "rank_candidates",
                            "raw_sha256": None, "valid": True, "error": None, "parsed": None})
    state.termination_reason = "fallback"
    state.reason = reason
    state.fallback_source = FALLBACK_SOURCE
    state.pending_question = state.pending_query = None
    state.phase = "explain"


def _finish_failed(state: EpisodeState, reason: str) -> None:
    state.termination_reason = "failed"
    state.reason = reason
    state.pending_question = state.pending_query = None
    state.phase = "done"


def _model_task(state: EpisodeState) -> str:
    if state.phase in CALLER_PHASES or state.phase == "done":
        raise TransitionError(f"no model action is expected in phase {state.phase!r}")
    return PHASE_TASK[state.phase]
