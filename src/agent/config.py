"""Limits and policies for the agent controller.

**Every default below is UNRATIFIED.** None has been approved by the team. Each is a proposal
taken from a named source, so changing one is a configuration change, not a code change.
``docs/member4_agent_decisions.md`` lists each open decision and what the code does meanwhile.

The values are stored inside every :class:`~src.agent.state.EpisodeState`, so a conversation that
pauses for a clarification answer keeps the limits it started with, and a trace shows exactly
which limits produced it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

# What to do when a budget runs out while the latest judgment is still "no":
#   fallback_b3_order   - Member 2's proposed v1 contract: rank the frozen B3 order and mark the
#                         trace ``termination_reason: "fallback"``.
#   rank_available_pool - Member 1 / midterm section 3.4: "proceed to ranking the available
#                         candidates", ranked by the model over the current pool.
# These two rules conflict and neither has been ratified (decision A1).
EXHAUSTION_POLICIES = ("fallback_b3_order", "rank_available_pool")

# The product shows ten recommendations, and the shared validator requires exactly this many
# explanations (or fewer only when fewer movies are ranked).
EXPLAINED_MOVIES = 10


class ConfigError(ValueError):
    """Raised when a limit or policy value is invalid."""


def _require_int(name: str, value, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}, got {value!r}")


@dataclass(frozen=True)
class AgentConfig:
    # Free-text provenance recorded in traces. Rename it when you freeze a real configuration.
    label: str = "unratified-defaults"

    # Source for the next three: Member 1's training design (midterm section 3.4):
    # at most 2 questions, 1 wider retrieval, a pool growing from 50 to at most 100.
    max_clarifications: int = 2
    max_retrievals: int = 1
    max_pool_size: int = 100

    # Source: proposed v1 contract, "at most eight control events". Eight is exactly the worst
    # case of the budgets above: 3 assessments + 2 questions + 2 pool evaluations + 1 retrieval.
    max_control_events: int = 8

    # Source: proposed v1 contract. The ranking event lists the final prediction's 50 movies.
    # Explanations cover the first ``min(10, ranking size)`` of them; that ten is fixed by the
    # contract and by Member 1's shared validator, so it is derived (``explain_count``), not set.
    ranking_size: int = 50

    # See EXHAUSTION_POLICIES. The default is the one Member 2's proposed validator can accept,
    # NOT a team decision.
    exhaustion_policy: str = "fallback_b3_order"

    # Extra attempts allowed after an unusable model output at the same step. 0 means the first
    # unusable output goes straight to the fallback. With greedy decoding a retry repeats the same
    # prompt and would usually produce the same output, so 0 is the least presumptuous default.
    max_invalid_actions_per_step: int = 0

    # The ask prompt says "Do not repeat a question". When True the controller enforces it.
    reject_repeated_questions: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ConfigError("label must be a nonempty string")
        for name in ("max_clarifications", "max_retrievals", "max_invalid_actions_per_step"):
            _require_int(name, getattr(self, name), 0)
        for name in ("max_pool_size", "max_control_events", "ranking_size"):
            _require_int(name, getattr(self, name), 1)
        if self.exhaustion_policy not in EXHAUSTION_POLICIES:
            raise ConfigError(f"exhaustion_policy must be one of {EXHAUSTION_POLICIES}")
        if type(self.reject_repeated_questions) is not bool:
            raise ConfigError("reject_repeated_questions must be a Boolean")

    @property
    def explain_count(self) -> int:
        """How many top-ranked movies get an explanation (the contract's ``min(10, required_count)``)."""
        return min(EXPLAINED_MOVIES, self.ranking_size)

    @property
    def worst_case_control_events(self) -> int:
        """Control events needed to use every budget: (asks + 1) assessments, asks, (retrievals + 1)
        pool evaluations and the retrievals themselves."""
        return 2 * self.max_clarifications + 2 * self.max_retrievals + 2

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data) -> "AgentConfig":
        """Strict loader: every field must be present, so a stored state never silently picks up
        a newer default."""
        if not isinstance(data, dict):
            raise ConfigError("config must be an object")
        expected = set(cls.__dataclass_fields__)
        if set(data) != expected:
            missing, extra = sorted(expected - set(data)), sorted(set(data) - expected)
            raise ConfigError(f"config fields differ: missing {missing}, unexpected {extra}")
        return cls(**data)
