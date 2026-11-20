"""The manifest fields the controller supplies (src/agent/manifest.py).

Every fallback reason here is produced by driving the real controller, not typed in, so the trigger
mapping is tested against what the controller actually says. The key sets are Member 2's proposed
``p2-run-manifest-v1`` as of ``origin/member2-phase2`` at ``9032fd1``; the validator itself lives on that
branch, so these tests check shape and the definitions in this module, and a one-off run against the real
validator is recorded in docs/member4_agent_decisions.md (A17).
"""

from __future__ import annotations

import unittest
from dataclasses import replace

from src.agent import controller as ctl
from src.agent.config import AgentConfig
from src.agent.manifest import (FALLBACK_TRIGGERS, PARSER_VERSION, agent_policy, controller_counts,
                                explanation_policy_version, fallback_trigger, manifest_episode)
from src.agent.trace import TraceError, to_trace_record
from tests.test_agent_controller import ScriptedModel, Unavailable, drive

EPISODE_KEYS = {"user_id", "termination_reason", "fallback_reason", "fallback_trigger", "model_call_count",
                "retrieval_call_count", "informative_answer_count", "noninformative_answer_count",
                "invalid_action_count", "timed_out", "loop_cap_hit", "latency_ms", "cost_usd"}
AGENT_KEYS = {"parser_version", "explanation_policy_version", "max_control_events", "max_serialized_events",
              "timeout_ms", "fallback_source", "fallback_triggers"}
POOL, HISTORY = list(range(1, 51)), [900]
LATENCY = {"model": 10.0, "retrieval": 5.0, "end_to_end": 20.0}


def start(config=None, pool=POOL):
    return ctl.start_episode("profile", pool, HISTORY, config=config or AgentConfig(), user_id=7, split="validation")


def raise_unavailable(step):
    raise Unavailable("index offline")


def finish(case, *, decisions=(), raw=None, retrieval=None, config=None, pool=POOL, state=None):
    return drive(case, state or start(config, pool), ScriptedModel(decisions=decisions, raw=raw), retrieval=retrieval)


def episode(state, answerable=None):
    asked = controller_counts(state)["clarification_event_count"]
    informative = asked if answerable is None else answerable
    return manifest_episode(state, informative_answer_count=informative, noninformative_answer_count=asked - informative,
                            timed_out=False, latency_ms=LATENCY, cost_usd=0.0)


class TriggerCase(unittest.TestCase):
    def reasons(self):
        """One real fallback per kind of trigger, produced by the controller."""
        wrong_action = '{"action": "evaluate_candidates", "pool_sufficient": true}'
        short_ranking = '{"action": "rank_candidates", "movie_ids": [1, 2]}'
        same_question = '{"action": "ask_clarification", "question": "Same?"}'
        model_failed = finish(self, state=ctl.apply_model_error(start(), "TimeoutError: no answer in time"))
        yield "invalid_action", finish(self, raw={"assess_sufficiency": ["nonsense"]})
        yield "unexpected_action", finish(self, raw={"assess_sufficiency": [wrong_action]})
        yield "unbound_ids", finish(self, decisions=[True, True], raw={"rank_candidates": [short_ranking]})
        yield "repeated_question", finish(self, decisions=[False, False], raw={"ask_clarification": [same_question] * 2})
        yield "model_error", model_failed
        yield "retrieval_error", finish(self, decisions=[True, False], retrieval=raise_unavailable)
        yield "retrieval_returned_no_new_candidates", finish(self, decisions=[True, False], retrieval=lambda step: [])
        yield "clarification_budget_exhausted", finish(self, decisions=[False, False, False])
        yield "retrieval_budget_exhausted", finish(self, decisions=[True, False, False])
        yield "pool_size_cap_reached", finish(self, decisions=[True, False], pool=list(range(1, 101)))
        yield "control_event_cap_reached", finish(self, decisions=[False], config=AgentConfig(max_control_events=3))


class TriggerMappingTests(TriggerCase):
    def test_every_declared_trigger_is_reachable_and_every_reachable_one_is_declared(self):
        seen = {}
        for expected, state in self.reasons():
            self.assertEqual(state.termination_reason, "fallback", expected)
            seen[expected] = fallback_trigger(state.reason)
        self.assertEqual(set(seen.values()), set(FALLBACK_TRIGGERS))
        for expected, trigger in seen.items():
            self.assertIn(trigger, FALLBACK_TRIGGERS)
            self.assertTrue(expected.startswith(trigger) or trigger == expected, (expected, trigger))

    def test_the_declared_list_is_valid_for_a_manifest(self):
        self.assertEqual(len(set(FALLBACK_TRIGGERS)), len(FALLBACK_TRIGGERS), "no duplicates")
        self.assertTrue(all(isinstance(code, str) and code.strip() for code in FALLBACK_TRIGGERS))

    def test_unknown_reason_text_raises_instead_of_reaching_a_manifest(self):
        for reason in ("something_new", "invalid_action:assess_sufficiency:weird", "model_error:", "", "retrieval_error"):
            with self.assertRaises(ValueError, msg=reason):
                fallback_trigger(reason)


class CountTests(TriggerCase):
    def test_a_plain_recommendation(self):
        state = finish(self)
        self.assertEqual(controller_counts(state), {
            "model_call_count": 4, "invalid_action_count": 0, "retrieval_call_count": 0,
            "clarification_event_count": 0, "loop_cap_hit": False})

    def test_unusable_outputs_are_counted_and_retries_add_calls(self):
        config = AgentConfig(max_invalid_actions_per_step=1)
        state = finish(self, raw={"assess_sufficiency": ["nonsense"]}, config=config)
        self.assertEqual(state.termination_reason, "recommend")
        counts = controller_counts(state)
        self.assertEqual((counts["model_call_count"], counts["invalid_action_count"]), (5, 1))

    def test_a_failed_model_call_is_a_call_but_not_an_invalid_action(self):
        config = AgentConfig(max_invalid_actions_per_step=1)
        state = finish(self, state=ctl.apply_model_error(start(config), "TimeoutError: slow"))
        self.assertEqual(state.termination_reason, "recommend")
        counts = controller_counts(state)
        self.assertEqual((counts["model_call_count"], counts["invalid_action_count"]), (5, 0))

    def test_retrieval_calls_count_even_when_they_fail_or_add_nothing(self):
        for retrieval in (raise_unavailable, lambda step: []):
            state = finish(self, decisions=[True, False], retrieval=retrieval)
            self.assertEqual(controller_counts(state)["retrieval_call_count"], 1)
        state = finish(self, decisions=[True, False, True])
        counts = controller_counts(state)
        self.assertEqual(counts["retrieval_call_count"], 1)
        self.assertFalse(counts["loop_cap_hit"])

    def test_loop_cap_hit_is_true_for_a_limit_and_false_for_an_error(self):
        for expected, state in self.reasons():
            hit = controller_counts(state)["loop_cap_hit"]
            limit = expected in ("clarification_budget_exhausted", "retrieval_budget_exhausted", "pool_size_cap_reached",
                                 "control_event_cap_reached", "retrieval_returned_no_new_candidates")
            self.assertEqual(hit, limit, expected)

    def test_clarifications_are_counted_from_events_not_from_the_replay_list(self):
        state = finish(self, decisions=[False, True])
        self.assertEqual(controller_counts(state)["clarification_event_count"], 1)
        # A turn replayed into a new episode is in the clarification list but has no trace event, and the
        # validator compares answerability counts with trace events.
        replayed = ctl.start_episode("profile", POOL, HISTORY, user_id=7, split="validation",
                                     clarifications=[{"question": "Q?", "answer": "A"}])
        replayed = finish(self, state=replayed)
        self.assertEqual(replayed.asks, 1)
        self.assertEqual(controller_counts(replayed)["clarification_event_count"], 0)
        self.assertEqual(episode(replayed)["informative_answer_count"], 0)


class EpisodeEntryTests(TriggerCase):
    def test_a_recommendation_entry(self):
        entry = episode(finish(self))
        self.assertEqual(set(entry), EPISODE_KEYS)
        self.assertEqual((entry["termination_reason"], entry["fallback_reason"], entry["fallback_trigger"]),
                         ("recommend", None, None))
        self.assertGreaterEqual(entry["model_call_count"], 1)

    def test_a_fallback_entry_carries_the_exact_trace_reason_and_a_declared_trigger(self):
        for expected, state in self.reasons():
            entry = episode(state)
            self.assertEqual(set(entry), EPISODE_KEYS, expected)
            self.assertEqual(entry["fallback_reason"], to_trace_record(state)["fallback_reason"], expected)
            self.assertIn(entry["fallback_trigger"], FALLBACK_TRIGGERS, expected)
            self.assertEqual(entry["user_id"], 7)

    def test_answerability_counts_must_add_up_to_the_clarification_events(self):
        state = finish(self, decisions=[False, True])
        self.assertEqual(episode(state, answerable=0)["noninformative_answer_count"], 1)
        for informative, noninformative in ((0, 0), (1, 1), (2, 0)):
            with self.assertRaises(ValueError, msg=(informative, noninformative)):
                manifest_episode(state, informative_answer_count=informative, noninformative_answer_count=noninformative,
                                 timed_out=False, latency_ms=LATENCY, cost_usd=0.0)

    def test_states_without_a_valid_trace_are_refused(self):
        with self.assertRaises(TraceError):
            episode(start())  # unfinished
        with self.assertRaises(TraceError):
            episode(finish(self, decisions=[False, False, False], config=AgentConfig(exhaustion_policy="rank_available_pool")))
        failed = finish(self, raw={"explain_recommendations": ["nonsense"]})
        self.assertEqual(failed.termination_reason, "failed")
        with self.assertRaises(TraceError):
            episode(failed)
        anonymous = ctl.start_episode("profile", POOL, HISTORY)
        with self.assertRaises(TraceError):
            episode(finish(self, state=anonymous))

    def test_a_fallback_that_is_still_waiting_for_its_explanations_is_not_finished(self):
        waiting = ctl.apply_model_error(start(), "TimeoutError: slow")
        self.assertEqual((waiting.termination_reason, waiting.phase), ("fallback", "explain"))
        with self.assertRaises(TraceError):
            episode(waiting)

    def test_a_recommendation_never_carries_a_fallback_reason_or_trigger(self):
        state = finish(self)
        state.reason = "left over"  # cannot happen through the controller; the entry must not depend on that
        entry = episode(state)
        self.assertEqual((entry["fallback_reason"], entry["fallback_trigger"]), (None, None))

    def test_the_callers_measurements_pass_through_untouched(self):
        state = finish(self)
        entry = manifest_episode(state, informative_answer_count=0, noninformative_answer_count=0, timed_out=True,
                                 latency_ms={"model": 1.5, "retrieval": 0.0, "end_to_end": 2.5}, cost_usd=0.002)
        self.assertEqual((entry["timed_out"], entry["cost_usd"], entry["latency_ms"]),
                         (True, 0.002, {"model": 1.5, "retrieval": 0.0, "end_to_end": 2.5}))


class AgentPolicyTests(unittest.TestCase):
    def test_shape_and_derived_values(self):
        policy = agent_policy(AgentConfig(), timeout_ms=30000)
        self.assertEqual(set(policy), AGENT_KEYS)
        self.assertEqual((policy["max_control_events"], policy["max_serialized_events"]), (8, 10))
        self.assertEqual((policy["fallback_source"], policy["parser_version"], policy["timeout_ms"]),
                         ("frozen_b3_order", PARSER_VERSION, 30000))
        self.assertEqual(policy["fallback_triggers"], list(FALLBACK_TRIGGERS))

    def test_the_explanation_policy_version_names_the_grounding_policy(self):
        for grounding in ("off", "flag", "repair", "reject"):
            version = explanation_policy_version(replace(AgentConfig(), grounding_policy=grounding))
            self.assertTrue(version.endswith("/" + grounding), version)

    def test_a_timeout_must_be_a_positive_integer(self):
        for bad in (0, -1, 1.5, "30000", None, True):
            with self.assertRaises(ValueError, msg=repr(bad)):
                agent_policy(AgentConfig(), timeout_ms=bad)


if __name__ == "__main__":
    unittest.main()
