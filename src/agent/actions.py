"""Parsing of one model output for the step the controller asked for.

The grammar itself (field names, types, strict JSON, duplicate keys) is Member 1's
``src.data.p2_actions``; it is imported, not re-implemented. This module adds only what the
*controller* knows and a stand-alone grammar check cannot:

* the model must return the action requested for this step, not just any valid action;
* ranking and explanation IDs must match the pool/ranking the controller actually showed;
* a clarifying question must not repeat an earlier one.

Parsing is strict on purpose: the training completions are exactly one JSON object, and silently
repairing near-misses (code fences, prose) would hide format failures the evaluation should count.
Whether to relax this after seeing real checkpoint outputs is decision A9.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from src.data.p2_actions import ActionError, parse_action, validate_action

# Stable reason codes recorded in decisions/failures.
INVALID_ACTION = "invalid_action"  # not valid JSON, or violates the shared grammar
UNEXPECTED_ACTION = "unexpected_action"  # a valid action, but not the one requested
UNBOUND_IDS = "unbound_ids"  # ranking/explanation IDs differ from what was shown
REPEATED_QUESTION = "repeated_question"


class AgentActionError(ValueError):
    """A model output the controller cannot use. ``code`` is one of the constants above."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def normalize_question(text: str) -> str:
    """Case, punctuation and spacing do not make a question new."""
    return " ".join(re.sub(r"[^\w\s]", " ", text).casefold().split())


def parse_model_output(
    raw_text,
    expected_task: str,
    *,
    pool_ids: list[int] | None = None,
    explain_ids: list[int] | None = None,
    previous_questions: Iterable[str] = (),
    reject_repeated_questions: bool = True,
) -> dict:
    """Return the validated action, or raise :class:`AgentActionError`.

    ``pool_ids`` is required for ``rank_candidates`` (the output must be a permutation of it) and
    ``explain_ids`` for ``explain_recommendations`` (the explanations must cover exactly these
    movies, in this order).
    """
    if not isinstance(raw_text, str):
        raise AgentActionError(INVALID_ACTION, "model output must be text")
    try:
        action = parse_action(raw_text)
    except ActionError as error:
        raise AgentActionError(INVALID_ACTION, str(error)) from error

    if action["action"] != expected_task:
        raise AgentActionError(
            UNEXPECTED_ACTION, f"expected {expected_task!r} but the model returned {action['action']!r}"
        )

    if expected_task == "rank_candidates" and pool_ids is None:
        raise ValueError("pool_ids is required to validate a ranking")
    if expected_task == "explain_recommendations" and explain_ids is None:
        raise ValueError("explain_ids is required to validate explanations")
    try:
        validate_action(
            action,
            candidate_ids=pool_ids if expected_task == "rank_candidates" else None,
            explain_ids=explain_ids if expected_task == "explain_recommendations" else None,
        )
    except ActionError as error:
        raise AgentActionError(UNBOUND_IDS, str(error)) from error

    if expected_task == "ask_clarification" and reject_repeated_questions:
        seen = {normalize_question(question) for question in previous_questions}
        if normalize_question(action["question"]) in seen:
            raise AgentActionError(REPEATED_QUESTION, "the question repeats an earlier one")
    return action
