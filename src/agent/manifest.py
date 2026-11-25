"""The parts of Member 2's proposed ``p2-run-manifest-v1`` that the controller can know.

The manifest (``docs/phase2_run_manifest.md`` on ``origin/member2-phase2``, "proposed v1") asks for a
``policy.agent`` object and one telemetry object per user. Some of those values come from the
controller's own state: how many model calls were made, how many outputs were unusable, why a fallback
happened. Others can only come from whoever runs the episode: wall-clock latency, cost, whether a call
timed out, and whether a simulated answer was informative. This module computes the first kind and takes
the second kind as explicit arguments, so nothing here pretends to measure what the controller cannot see.

This is NOT the offline evaluation runner (decision A12). It does no I/O, loops over no users and writes
no files. It only gives whichever component builds the manifest the fields that are Member 4's to supply.

Every definition below is a Member 4 proposal (decision A17) for Member 2 to accept or change:

* ``fallback_trigger`` is a short stable code derived from the exact ``fallback_reason`` text.
* ``invalid_action_count`` counts model outputs that were parsed and rejected. A failed model call
  (``model_error``) is not an invalid action; it is counted in ``model_call_count`` and, when it was a
  timeout, reported by the caller as ``timed_out``.
* ``loop_cap_hit`` is true when any question, retrieval, pool-size or event limit stopped a "no" judgment.
"""

from __future__ import annotations

from collections.abc import Mapping

from src.agent.actions import INVALID_ACTION, REPEATED_QUESTION, UNBOUND_IDS, UNEXPECTED_ACTION
from src.agent.config import AgentConfig
from src.agent.controller import (CLARIFICATION_BUDGET, EVENT_CAP, FALLBACK_SOURCE, NO_NEW_CANDIDATES, POOL_FULL,
                                  RETRIEVAL_BUDGET)
from src.agent.state import EXPLANATION_SOURCE_FALLBACK, EpisodeState
from src.agent.trace import TraceError, is_contract_termination

PARSER_VERSION = "p2-agent-parser-v1"  # names the strict grammar in docs/member4_action_reference.md
GROUNDING_SCREEN_VERSION = "p2-grounding-screen-v1"  # names src/agent/grounding.py's checks
MODEL_ERROR = "model_error"
RETRIEVAL_ERROR = "retrieval_error"

# The complete set of codes a run can hand to ``policy.agent.fallback_triggers``.
FALLBACK_TRIGGERS = (
    INVALID_ACTION, UNEXPECTED_ACTION, UNBOUND_IDS, REPEATED_QUESTION,  # an unusable model output
    MODEL_ERROR,  # the model call itself failed
    RETRIEVAL_ERROR, NO_NEW_CANDIDATES,  # the retriever raised, or returned nothing usable
    CLARIFICATION_BUDGET, RETRIEVAL_BUDGET, POOL_FULL, EVENT_CAP,  # a limit stopped a "no" judgment
)
_LIMIT_REASONS = {CLARIFICATION_BUDGET, RETRIEVAL_BUDGET, POOL_FULL, EVENT_CAP, NO_NEW_CANDIDATES}


def explanation_policy_version(config: AgentConfig) -> str:
    """Names the grounding screen, the policy applied to what it flags and, when it is on, the rule for an
    explanation step with no usable output. The default (``fail``) leaves the string as it always was."""
    version = f"{GROUNDING_SCREEN_VERSION}/{config.grounding_policy}"
    if config.explanation_failure_policy == "template":
        version += f"/{EXPLANATION_SOURCE_FALLBACK}"  # Member 2's label for the explanation fallback
    return version


def agent_policy(config: AgentConfig, *, timeout_ms: int) -> dict:
    """The ``policy.agent`` object. ``timeout_ms`` is the caller's: the model client enforces it, so the
    controller has no value of its own to report."""
    if type(timeout_ms) is not int or timeout_ms <= 0:
        raise ValueError("timeout_ms must be a positive integer")
    return {
        "parser_version": PARSER_VERSION,
        "explanation_policy_version": explanation_policy_version(config),
        "max_control_events": config.max_control_events,
        "max_serialized_events": config.max_control_events + 2,  # plus exactly one rank and one explain event
        "timeout_ms": timeout_ms,
        "fallback_source": FALLBACK_SOURCE,
        "fallback_triggers": list(FALLBACK_TRIGGERS),
    }


def fallback_trigger(reason: str) -> str:
    """The stable code for a ``fallback_reason``. Raises on text it does not know, so a new reason added
    to the controller cannot silently reach a manifest without a trigger."""
    if reason in _LIMIT_REASONS:
        return reason
    prefix, _, rest = reason.partition(":")
    if prefix == INVALID_ACTION:  # invalid_action:<step>:<code>
        code = rest.rpartition(":")[2]
        if code in (INVALID_ACTION, UNEXPECTED_ACTION, UNBOUND_IDS, REPEATED_QUESTION):
            return code
    elif prefix in (MODEL_ERROR, RETRIEVAL_ERROR) and rest:  # model_error:<step>, retrieval_error:<detail>
        return prefix
    raise ValueError(f"no fallback trigger is defined for reason {reason!r}")


def controller_counts(state: EpisodeState) -> dict:
    """What the state alone shows. A state rebuilt from replayed clarifications (``decide``) has no memory
    of model calls made while earlier turns were served, so whole-episode counts should come from a
    state that was carried through every call (``resume``) or be summed by the caller."""
    model = [entry for entry in state.decisions if entry["source"] == "model"]
    return {
        "model_call_count": len(model),
        "invalid_action_count": sum(1 for entry in model if not entry["valid"] and entry["error"]["code"] != MODEL_ERROR),
        "retrieval_call_count": state.retrievals,  # a call that failed or added nothing still counts
        "clarification_event_count": sum(1 for event in state.events if event["action"] == "ask_clarification"),
        "loop_cap_hit": any(failure["code"] == "budget_exhausted" for failure in state.failures),
    }


def manifest_episode(state: EpisodeState, *, informative_answer_count: int, noninformative_answer_count: int,
                     timed_out: bool, latency_ms: Mapping[str, float], cost_usd: float) -> dict:
    """One ``episodes`` entry. The caller supplies what it alone can know (answerability, timing, cost).

    Raises ``TraceError`` for a state that has no valid trace (unfinished, failed, or ended with a label
    outside the contract), and ``ValueError`` if the answerability counts do not add up to the number of
    clarification events, the same check the manifest validator makes.
    """
    if state.phase != "done":
        raise TraceError(f"episode is not finished (phase {state.phase!r})")
    if not is_contract_termination(state):
        raise TraceError(f"termination {state.termination_reason!r} has no manifest entry")
    if type(state.user_id) is not int or state.user_id <= 0:
        raise TraceError("a manifest entry needs a positive integer user_id")
    counts = controller_counts(state)
    if informative_answer_count + noninformative_answer_count != counts["clarification_event_count"]:
        raise ValueError(
            f"answerability counts ({informative_answer_count} + {noninformative_answer_count}) must equal "
            f"the {counts['clarification_event_count']} clarification event(s)")
    fallback = state.termination_reason == "fallback"
    return {
        "user_id": state.user_id,
        "termination_reason": state.termination_reason,
        "fallback_reason": state.reason if fallback else None,
        "fallback_trigger": fallback_trigger(state.reason) if fallback else None,
        "model_call_count": counts["model_call_count"],
        "retrieval_call_count": counts["retrieval_call_count"],
        "informative_answer_count": informative_answer_count,
        "noninformative_answer_count": noninformative_answer_count,
        "invalid_action_count": counts["invalid_action_count"],
        "timed_out": bool(timed_out),
        "loop_cap_hit": counts["loop_cap_hit"],
        "latency_ms": dict(latency_ms),
        "cost_usd": cost_usd,
    }
