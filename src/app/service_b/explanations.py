"""Count explanation substitutions separately from B3 ranking fallbacks."""
from collections import Counter

from src.agent.state import EXPLANATION_SOURCE_FALLBACK, EXPLANATION_SOURCE_MODEL


def summarize_explanations(traces, evidence):
    users, reasons = [], Counter()
    for trace in traces:
        uid = trace["user_id"]
        events = [e for e in trace["events"] if e["action"] == "explain_recommendations"]
        if len(events) != 1:
            raise ValueError(f"Expected one explanation event for user {uid}")
        event = events[0]
        source = event.get("source")
        if source not in (EXPLANATION_SOURCE_MODEL, EXPLANATION_SOURCE_FALLBACK):
            raise ValueError(f"Unknown explanation source for user {uid}")
        reason = event.get("explanation_fallback_reason")
        if source == EXPLANATION_SOURCE_FALLBACK:
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(f"Missing explanation fallback reason for user {uid}")
            failed = [d for d in evidence[uid]["telemetry"]["decisions"]
                      if d["source"] == "model" and d["task"] == "explain_recommendations" and not d["valid"]]
            if not failed or not failed[-1].get("error"):
                raise ValueError(f"Explanation substitution has no failed-attempt evidence for user {uid}")
            reasons[reason] += 1
        elif reason is not None:
            raise ValueError(f"Model explanation unexpectedly has a fallback reason for user {uid}")
        users.append({"user_id": uid, "explanation_source": source,
                      "explanation_fallback_reason": reason,
                      "ranking_termination_reason": trace["termination_reason"]})
    if len({u["user_id"] for u in users}) != len(users):
        raise ValueError("Repeated user in explanation summary")
    fallback_count = sum(reasons.values())
    return {"schema_version": "p2-explanation-summary-v1", "completed_users": len(users),
            "catalog_metadata_fallback_user_count": fallback_count,
            "model_explanation_user_count": len(users) - fallback_count,
            "ranking_fallback_user_count": sum(t["termination_reason"] == "fallback" for t in traces),
            "catalog_metadata_fallback_by_reason": dict(sorted(reasons.items())),
            "users": sorted(users, key=lambda u: u["user_id"])}
