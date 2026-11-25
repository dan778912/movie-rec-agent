"""Convert a finished episode into the proposed ``p2-trace-v1`` record, plus separate telemetry.

``to_trace_record`` follows ``docs/phase2_interface_contract.md`` on ``origin/member2-phase2``
(status there: "proposed v1"). It emits only the event fields that document lists, including the
explanation event's ``source`` and ``explanation_fallback_reason`` (first real validation run). Everything else
the manifest asks for (raw-output digests, validation results, retrieval queries, drop counts,
configuration) goes in ``telemetry_record``, so the trace stays valid against the contract.

A trace is a record of what happened, not a claim that it was good: a structurally valid trace
does not show that model calls were autonomous, that a simulated answer used only pre-cutoff
history, or that an explanation is true. Those need provenance and audit under the protocol.
"""

from __future__ import annotations

import copy

from src.agent.state import (CONTRACT_TERMINATIONS, EXPLANATION_SOURCE_FALLBACK, EXPLANATION_SOURCE_MODEL,
                             EpisodeState)

TRACE_SCHEMA_VERSION = "p2-trace-v1"
TELEMETRY_SCHEMA_VERSION = "p2-agent-telemetry-v0"  # this module's own, unratified


class TraceError(ValueError):
    """No valid trace can be produced for this state."""


def to_trace_record(state: EpisodeState) -> dict:
    """The offline per-user trace. Raises TraceError unless the episode finished and has ranking
    and explanation events, a user_id and a split."""
    if state.phase != "done":
        raise TraceError(f"episode is not finished (phase {state.phase!r})")
    if state.termination_reason == "failed":
        raise TraceError(f"episode failed ({state.reason}); report the failure instead of a trace")
    if state.user_id is None or state.split is None:
        raise TraceError("an offline trace needs user_id and split")
    record = {
        "schema_version": TRACE_SCHEMA_VERSION,
        "user_id": state.user_id,
        "split": state.split,
        "method": "p2",
        "initial_candidate_movie_ids": list(state.initial_pool),
        "events": copy.deepcopy(state.events),
        "final_candidate_movie_ids": list(state.pool),
        "termination_reason": state.termination_reason,
    }
    if state.termination_reason == "fallback":
        record["fallback_reason"] = state.reason
        record["fallback_source"] = state.fallback_source
    return record


def is_contract_termination(state: EpisodeState) -> bool:
    """True when the termination reason is one the proposed v1 contract defines."""
    return state.termination_reason in CONTRACT_TERMINATIONS


def explanation_source(state: EpisodeState) -> str | None:
    """Where the explanations in the trace came from, in Member 2's words: ``"model"`` (the model's text,
    after any grounding policy), ``"catalog_metadata_fallback"`` (written from catalog fields because the
    model gave nothing usable, ``explanation_failure_policy="template"``), or None (no explanation event:
    unfinished or failed). It is the same value the trace's explanation event carries as ``source``.

    Derived from the decision records, so it also works for a state stored before the event had a
    ``source`` field."""
    entry = _last_explanation_decision(state)
    if entry is None:
        return None
    return EXPLANATION_SOURCE_FALLBACK if entry["source"] == "fallback" else EXPLANATION_SOURCE_MODEL


def explanation_fallback_reason(state: EpisodeState) -> str | None:
    """Why the explanations are catalog text (the same string as the event's ``explanation_fallback_reason``),
    or None when the model wrote them or there is no explanation event."""
    entry = _last_explanation_decision(state)
    return entry["reason"] if entry is not None and entry["source"] == "fallback" else None


def _last_explanation_decision(state: EpisodeState) -> dict | None:
    for entry in reversed(state.decisions):
        if entry["task"] == "explain_recommendations" and entry["valid"]:
            return entry
    return None


def telemetry_record(state: EpisodeState) -> dict:
    """Audit data kept beside, not inside, the trace."""
    return {
        "schema_version": TELEMETRY_SCHEMA_VERSION,
        "user_id": state.user_id,
        "split": state.split,
        "config": state.config.to_dict(),
        "termination_reason": state.termination_reason,
        "reason": state.reason,
        "contract_termination": is_contract_termination(state),
        "explanation_source": explanation_source(state),
        "explanation_fallback_reason": explanation_fallback_reason(state),
        "control_events": state.control_events,
        "clarifications_used": state.asks,
        "clarifications_replayed": state.resumed_clarifications,
        "retrievals_used": state.retrievals,
        "initial_index_version": state.initial_index_version,
        "decisions": copy.deepcopy(state.decisions),
        "failures": copy.deepcopy(state.failures),
        "retrieval_log": copy.deepcopy(state.retrieval_log),
        # Deterministic screen results only. They are not the human audit and never say "supported".
        "grounding": copy.deepcopy(state.grounding),
    }
