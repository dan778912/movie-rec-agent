"""JSON-serializable state of one agent episode.

The controller (``controller.py``) never keeps hidden memory. Everything it needs to pause for a
clarification answer and resume later lives in this object, so a stateless Service B can return
it, or rebuild it from replayed clarifications (``start_episode``), and continue correctly.

State that arrives from outside a process is untrusted, so ``from_dict`` checks structure and the
invariants the transitions rely on. This is a structural check, not a proof of how the state was
produced.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields

from src.agent.config import AgentConfig, ConfigError
from src.data.p2_actions import CONTROL_ACTIONS
from src.profiles.agent_text import ACTIONS

STATE_VERSION = "p2-agent-state-v1"

# phase -> the model task the controller needs next (the six actions come from agent_text).
PHASE_TASK = {
    "assess": "assess_sufficiency",
    "ask": "ask_clarification",
    "evaluate": "evaluate_candidates",
    "retrieve": "retrieve_candidates",
    "rank": "rank_candidates",
    "explain": "explain_recommendations",
}
assert set(PHASE_TASK.values()) == set(ACTIONS), "phase map must cover the shared action list exactly"

# Phases where the caller, not the model, must act next.
CALLER_PHASES = ("await_answer", "await_retrieval")
PHASES = (*PHASE_TASK, *CALLER_PHASES, "done")

# "recommend" and "fallback" are the only reasons defined by the proposed v1 trace contract.
# "budget_exhausted_rank" (rank_available_pool policy) and "failed" (no usable explanation) are
# this module's own labels: the proposed v1 evaluator does not define them (decision A13).
CONTRACT_TERMINATIONS = ("recommend", "fallback")
TERMINATIONS = (*CONTRACT_TERMINATIONS, "budget_exhausted_rank", "failed")


class StateError(ValueError):
    """Raised when a state object is malformed or violates an invariant."""


def _pos_int(value) -> bool:
    return type(value) is int and value > 0


def _pos_int_list(name: str, value, *, nonempty: bool = False) -> list[int]:
    if not isinstance(value, list) or not all(_pos_int(item) for item in value):
        raise StateError(f"{name} must be a list of positive integers")
    if len(set(value)) != len(value):
        raise StateError(f"{name} contains duplicates")
    if nonempty and not value:
        raise StateError(f"{name} must not be empty")
    return list(value)


def _nonneg_int(name: str, value) -> int:
    if type(value) is not int or value < 0:
        raise StateError(f"{name} must be a nonnegative integer")
    return value


def _opt_str(name: str, value) -> str | None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise StateError(f"{name} must be null or a nonempty string")
    return value


def _opt_bool(name: str, value) -> bool | None:
    if value is not None and type(value) is not bool:
        raise StateError(f"{name} must be null or a Boolean")
    return value


def _dict_list(name: str, value) -> list[dict]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise StateError(f"{name} must be a list of objects")
    return copy.deepcopy(value)


@dataclass
class EpisodeState:
    config: AgentConfig
    profile_text: str
    history_movie_ids: list[int]  # complete watched history, used only to exclude movies
    initial_pool: list[int]  # the frozen B3 order the episode started from
    pool: list[int]  # initial_pool followed by every retrieved addition, in order

    # Questions answered so far: {"question", "answer", "answer_source"}.
    clarifications: list[dict] = field(default_factory=list)
    # How many of them were replayed from an earlier call (their events are not in ``events``).
    resumed_clarifications: int = 0
    pending_question: str | None = None  # asked, awaiting an answer (phase await_answer)
    pending_query: str | None = None  # model's retrieval query, awaiting results (await_retrieval)

    sufficient: bool | None = None  # latest sufficiency judgment; reset by a new question
    pool_ok: bool | None = None  # latest pool judgment; reset by a question or a retrieval
    retrievals: int = 0  # retrieval attempts consumed, including empty ones
    control_events: int = 0  # counted against config.max_control_events (includes replayed ones)
    step_invalid_count: int = 0  # unusable outputs at the current step

    ranked: list[int] | None = None  # final ranking, at most config.ranking_size movies
    phase: str = "assess"

    events: list[dict] = field(default_factory=list)  # p2-trace-v1 events recorded by this episode
    decisions: list[dict] = field(default_factory=list)  # one entry per model call or fallback
    failures: list[dict] = field(default_factory=list)  # rejected outputs, exhausted budgets, errors
    retrieval_log: list[dict] = field(default_factory=list)  # what each retrieval returned and kept
    grounding: list[dict] = field(default_factory=list)  # one record per screened explanation set
    initial_index_version: str | None = None  # index that produced initial_pool, if the caller knows it

    termination_reason: str | None = None
    reason: str | None = None  # why the episode did not end in a plain "recommend"
    fallback_source: str | None = None

    user_id: int | None = None  # offline traces need these two; live calls may omit them
    split: str | None = None
    schema_version: str = STATE_VERSION

    @property
    def asks(self) -> int:
        """Questions answered so far, replayed ones included."""
        return len(self.clarifications)

    # ------------------------------------------------------------------ serialization
    def to_dict(self) -> dict:
        data = {"schema_version": self.schema_version, "config": self.config.to_dict()}
        for item in fields(self):
            if item.name not in data:
                data[item.name] = copy.deepcopy(getattr(self, item.name))
        return data

    @classmethod
    def from_dict(cls, data) -> "EpisodeState":
        if not isinstance(data, dict):
            raise StateError("state must be an object")
        expected = {item.name for item in fields(cls)}
        if set(data) != expected:
            missing, extra = sorted(expected - set(data)), sorted(set(data) - expected)
            raise StateError(f"state fields differ: missing {missing}, unexpected {extra}")
        if data["schema_version"] != STATE_VERSION:
            raise StateError(f"unsupported state version: {data['schema_version']!r}")
        try:
            config = AgentConfig.from_dict(data["config"])
        except ConfigError as error:
            raise StateError(f"invalid config: {error}") from error
        if not isinstance(data["profile_text"], str):
            raise StateError("profile_text must be a string")
        clarifications = _dict_list("clarifications", data["clarifications"])
        for turn in clarifications:
            if set(turn) != {"question", "answer", "answer_source"} or not all(
                isinstance(value, str) and value.strip() for value in turn.values()
            ):
                raise StateError("each clarification needs nonempty question, answer and answer_source")
        ranked = data["ranked"]
        state = cls(
            config=config,
            profile_text=data["profile_text"],
            history_movie_ids=_pos_int_list("history_movie_ids", data["history_movie_ids"]),
            initial_pool=_pos_int_list("initial_pool", data["initial_pool"], nonempty=True),
            pool=_pos_int_list("pool", data["pool"], nonempty=True),
            clarifications=clarifications,
            resumed_clarifications=_nonneg_int("resumed_clarifications", data["resumed_clarifications"]),
            pending_question=_opt_str("pending_question", data["pending_question"]),
            pending_query=_opt_str("pending_query", data["pending_query"]),
            sufficient=_opt_bool("sufficient", data["sufficient"]),
            pool_ok=_opt_bool("pool_ok", data["pool_ok"]),
            retrievals=_nonneg_int("retrievals", data["retrievals"]),
            control_events=_nonneg_int("control_events", data["control_events"]),
            step_invalid_count=_nonneg_int("step_invalid_count", data["step_invalid_count"]),
            ranked=None if ranked is None else _pos_int_list("ranked", ranked, nonempty=True),
            phase=data["phase"],
            events=_dict_list("events", data["events"]),
            decisions=_dict_list("decisions", data["decisions"]),
            failures=_dict_list("failures", data["failures"]),
            retrieval_log=_dict_list("retrieval_log", data["retrieval_log"]),
            grounding=_dict_list("grounding", data["grounding"]),
            initial_index_version=_opt_str("initial_index_version", data["initial_index_version"]),
            termination_reason=_opt_str("termination_reason", data["termination_reason"]),
            reason=_opt_str("reason", data["reason"]),
            fallback_source=_opt_str("fallback_source", data["fallback_source"]),
            user_id=data["user_id"],
            split=_opt_str("split", data["split"]),
            schema_version=data["schema_version"],
        )
        if state.user_id is not None and not _pos_int(state.user_id):
            raise StateError("user_id must be null or a positive integer")
        state.check_invariants()
        return state

    # ------------------------------------------------------------------ invariants
    def check_invariants(self) -> None:
        """Rules every reachable state satisfies. Raises StateError on the first violation."""
        if self.phase not in PHASES:
            raise StateError(f"unknown phase: {self.phase!r}")
        if self.termination_reason is not None and self.termination_reason not in TERMINATIONS:
            raise StateError(f"unknown termination_reason: {self.termination_reason!r}")
        if self.phase == "done" and self.termination_reason is None:
            raise StateError("a finished episode needs a termination_reason")
        if set(self.history_movie_ids) & set(self.pool):
            raise StateError("pool contains a movie from the watched history")
        if self.pool[: len(self.initial_pool)] != self.initial_pool:
            raise StateError("pool must start with the initial pool, in order")
        if len(self.pool) > max(self.config.max_pool_size, len(self.initial_pool)):
            raise StateError("pool is larger than max_pool_size")
        if self.retrievals > self.config.max_retrievals:
            raise StateError("more retrievals than the configured budget")
        if self.phase == "await_answer" and self.pending_question is None:
            raise StateError("await_answer needs a pending_question")
        if self.phase != "await_answer" and self.pending_question is not None:
            raise StateError("pending_question is only allowed in await_answer")
        if self.phase == "await_retrieval" and self.pending_query is None:
            raise StateError("await_retrieval needs a pending_query")
        if self.phase != "await_retrieval" and self.pending_query is not None:
            raise StateError("pending_query is only allowed in await_retrieval")
        needs_ranking = self.phase in ("explain", "done") and self.termination_reason != "failed"
        if needs_ranking and self.ranked is None:
            raise StateError("explain and done phases need a ranking")
        if self.ranked is not None and not set(self.ranked) <= set(self.pool):
            raise StateError("ranked contains a movie outside the pool")
        if self.ranked is not None and len(self.ranked) > self.config.ranking_size:
            raise StateError("ranked is longer than ranking_size")
        if self.resumed_clarifications > len(self.clarifications):
            raise StateError("resumed_clarifications exceeds the clarifications present")
        self._check_records()
        self._check_counters_agree_with_records()

    def _check_records(self) -> None:
        """Shape of the records that later code (telemetry, the run manifest) reads without checking."""
        for event in self.events:
            if event.get("action") not in ACTIONS:
                raise StateError(f"event has a missing or unknown action: {event.get('action')!r}")
        for position, entry in enumerate(self.decisions):
            if entry.get("index") != position or entry.get("source") not in ("model", "fallback"):
                raise StateError(f"decision {position} has a wrong index or an unknown source")
            if entry.get("task") not in ACTIONS or type(entry.get("valid")) is not bool:
                raise StateError(f"decision {position} needs a known task and a Boolean valid")
            error = entry.get("error")
            if entry["valid"] and error is not None:
                raise StateError(f"decision {position} is valid but carries an error")
            if not entry["valid"] and not (
                isinstance(error, dict) and isinstance(error.get("code"), str) and isinstance(error.get("message"), str)
            ):
                raise StateError(f"decision {position} is invalid but has no error code and message")
        for position, failure in enumerate(self.failures):
            if not all(isinstance(failure.get(key), str) for key in ("code", "stage", "detail")):
                raise StateError(f"failure {position} needs text code, stage and detail")

    def _check_counters_agree_with_records(self) -> None:
        """Every counter the controller keeps has a record behind it, so editing one number alone is
        caught. (Rewriting the counter and its records together is not; that needs a signature.)"""
        control = sum(1 for event in self.events if event["action"] in CONTROL_ACTIONS)
        if self.control_events != 2 * self.resumed_clarifications + control:
            raise StateError("control_events disagrees with the events and replayed turns recorded")
        asked = sum(1 for event in self.events if event["action"] == "ask_clarification")
        if len(self.clarifications) != self.resumed_clarifications + asked:
            raise StateError("clarifications disagree with the question events and replayed turns recorded")
        retrieved = sum(1 for event in self.events if event["action"] == "retrieve_candidates")
        if self.retrievals < max(retrieved, len(self.retrieval_log)):
            raise StateError("retrievals is lower than the retrievals recorded")
        if self.step_invalid_count > self.config.max_invalid_actions_per_step:
            raise StateError("step_invalid_count exceeds the allowed retries")

    def copy(self) -> "EpisodeState":
        return copy.deepcopy(self)
