"""Controller tests.

The model and the retriever here are SOFTWARE TEST DOUBLES: canned answers, no weights, no index.
They show that the controller handles a response correctly (limits, fallbacks, pause/resume,
traces). They do not show that a real model chooses useful actions or emits the format reliably;
that needs Member 6's checkpoint and Member 3's retrieval.

The oracle for "is this episode legal?" is Member 1's shared ``validate_episode``, not code written
alongside the controller.
"""

import copy
import hashlib
import itertools
import json
import random
import re
import unittest
from pathlib import Path

from src.agent import controller as ctl
from src.agent.config import AgentConfig
from src.agent.state import EpisodeState
from src.agent.trace import TraceError, is_contract_termination, telemetry_record, to_trace_record
from src.data.clarification_sim import SIMULATOR_VERSION
from src.data.p2_actions import ActionError, validate_episode
from src.profiles.agent_text import ACTIONS, candidate_line, render_prompt, serialize_action

EXAMPLES = Path(__file__).resolve().parent.parent / "docs" / "p2_sample_examples.jsonl"
POLICIES = ("fallback_b3_order", "rank_available_pool")


def catalog_for(ids):
    return {i: {"movie_id": i, "title": f"Movie {i} (2001)", "genres": ["Drama", "Comedy"], "tags": ["quirky", "slow"]}
            for i in ids}


class Unavailable(Exception):
    """Stands in for a retrieval timeout or error."""


class ScriptedModel:
    """Canned model. ``decisions`` are the Booleans for assess/evaluate in call order (then True);
    ``raw`` maps a task to raw texts used first, so a test can inject bad output."""

    def __init__(self, decisions=(), raw=None):
        self.decisions = iter(decisions)
        self.raw = {task: list(texts) for task, texts in (raw or {}).items()}
        self.calls = []

    def respond(self, task, state):
        self.calls.append(task)
        if self.raw.get(task):
            return self.raw[task].pop(0)
        judged = task in ("assess_sufficiency", "evaluate_candidates")  # only these two take a yes/no
        return valid_output(task, state, decision=next(self.decisions, True) if judged else True)


def valid_output(task, state, *, decision=True, ranking=None):
    if task == "assess_sufficiency":
        return serialize_action({"action": task, "sufficient": decision})
    if task == "evaluate_candidates":
        return serialize_action({"action": task, "pool_sufficient": decision})
    if task == "ask_clarification":
        return serialize_action({"action": task, "question": f"Question number {state.asks + 1}?"})
    if task == "retrieve_candidates":
        return serialize_action({"action": task, "query": f"search {state.retrievals + 1}"})
    if task == "rank_candidates":
        return serialize_action({"action": task, "movie_ids": ranking or sorted(state.pool, reverse=True)})
    shown = (state.ranked or sorted(state.pool))[: state.config.explain_count]  # fuzz may ask before ranking
    return serialize_action({"action": task, "explanations": [{"movie_id": i, "text": f"Because {i}."} for i in shown]})


def default_retrieval():
    counter = itertools.count(1000)

    def retrieve(step):
        return [next(counter) for _ in range(min(step.max_new, 5))]

    return retrieve


def drive(case, state, model, *, retrieval=None, answer="I like heists.", on_state=None, stop_at=None):
    """Run the loop from the controller's docstring. Also proves each transition leaves its input alone."""
    retrieval = retrieval or default_retrieval()
    for _ in range(120):
        step = ctl.next_step(state)
        if step.kind == "done" or step.kind == stop_at:
            return state
        before = copy.deepcopy(state.to_dict())
        if step.kind == "model":
            new = ctl.apply_model_output(state, model.respond(step.task, state))
        elif step.kind == "answer":
            new = ctl.apply_answer(state, answer, "test_double")
        else:
            try:
                ids = retrieval(step)
            except Unavailable as error:
                new = ctl.apply_retrieval_failure(state, str(error))
            else:
                new = ctl.apply_retrieval(state, ids, "test-index-v1")
        case.assertEqual(state.to_dict(), before, "a transition mutated its input state")
        state = new
        if on_state:
            on_state(state)
    raise AssertionError("episode did not finish")


class ControllerCase(unittest.TestCase):
    def start(self, *, pool=None, history=(900,), **config):
        pool = list(pool or range(1, 51))
        return ctl.start_episode("Likes slow comedies.", pool, list(history), config=AgentConfig(**config),
                                 user_id=7, split="validation")

    def check_trace(self, state):
        """Legality against Member 1's shared validator plus independent structural checks."""
        record = to_trace_record(state)
        strict = state.termination_reason == "recommend"
        summary = validate_episode(record["events"], record["initial_candidate_movie_ids"], allow_fallback=not strict)
        events = record["events"]
        added = [m for e in events if e["action"] == "retrieve_candidates" for m in e["added_movie_ids"]]
        self.assertEqual(record["final_candidate_movie_ids"], record["initial_candidate_movie_ids"] + added)
        ranks = [e for e in events if e["action"] == "rank_candidates"]
        self.assertEqual(len(ranks), 1)
        self.assertEqual(ranks[0]["movie_ids"], state.ranked)
        self.assertFalse(set(state.ranked) & set(state.history_movie_ids), "recommended a watched movie")
        self.assertFalse(set(record["final_candidate_movie_ids"]) & set(state.history_movie_ids))
        for event in events:
            self.assertNotIn("injected_movie_ids", event)  # training-only; never valid at inference
        self.assertEqual(json.loads(json.dumps(record)), record)
        self.assertLessEqual(summary["control_events"], state.config.max_control_events)
        return record, summary


class HappyPathTest(ControllerCase):
    def test_direct_recommendation_follows_the_contract_event_order(self):
        model = ScriptedModel()
        state = drive(self, self.start(), model)
        self.assertEqual(state.termination_reason, "recommend")
        self.assertEqual([e["action"] for e in state.events],
                         ["assess_sufficiency", "evaluate_candidates", "rank_candidates", "explain_recommendations"])
        self.assertEqual(model.calls, ["assess_sufficiency", "evaluate_candidates", "rank_candidates",
                                       "explain_recommendations"])
        self.assertEqual(len(state.ranked), 50)
        self.assertEqual(len(state.events[-1]["explanations"]), 10)
        record, summary = self.check_trace(state)
        self.assertNotIn("fallback_reason", record)
        self.assertEqual(summary["control_events"], 2)

    def test_full_model_ranking_is_truncated_to_the_configured_size(self):
        state = drive(self, self.start(pool=range(1, 13), ranking_size=5), ScriptedModel())
        self.assertEqual(state.ranked, [12, 11, 10, 9, 8])
        self.assertEqual(len(state.events[-1]["explanations"]), 5)  # min(10, ranking size)
        rank_decision = [d for d in state.decisions if d["task"] == "rank_candidates"][0]
        self.assertEqual(len(rank_decision["parsed"]["movie_ids"]), 12)  # the model's full ranking is kept
        self.check_trace(state)

    def test_worst_case_path_uses_exactly_the_event_cap(self):
        # assess F, ask, assess F, ask, assess T, evaluate F, retrieve, evaluate T
        state = drive(self, self.start(), ScriptedModel(decisions=[False, False, True, False, True]))
        self.assertEqual(state.termination_reason, "recommend")
        self.assertEqual((state.asks, state.retrievals), (2, 1))
        self.assertEqual(state.control_events, AgentConfig().max_control_events)
        self.assertEqual([e["action"] for e in state.events][:8], [
            "assess_sufficiency", "ask_clarification", "assess_sufficiency", "ask_clarification",
            "assess_sufficiency", "evaluate_candidates", "retrieve_candidates", "evaluate_candidates"])
        self.assertEqual(len(state.pool), 55)
        record, summary = self.check_trace(state)
        self.assertEqual(summary["retrieve_candidates"], 1)
        self.assertEqual(summary["ask_clarification"], 2)
        # The retrieved movies join the ranking (sorted descending puts 1000+ first).
        self.assertEqual(state.ranked[:5], [1004, 1003, 1002, 1001, 1000])


class EveryDecisionPathTest(ControllerCase):
    """All 2^9 yes/no choices the model could make for the two judgments, under both policies."""

    def run_all(self, policy):
        outcomes = set()
        for bits in itertools.product([True, False], repeat=9):
            state = self.start(exhaustion_policy=policy)
            snapshots = []

            def keep(new_state):
                loaded = EpisodeState.from_dict(json.loads(json.dumps(new_state.to_dict())))
                snapshots.append(loaded.to_dict() == new_state.to_dict())

            model = ScriptedModel(decisions=bits)
            state = drive(self, state, model, on_state=keep)
            self.assertTrue(all(snapshots), "an intermediate state failed to round-trip through JSON")
            self.assertLessEqual(state.control_events, 8)
            self.assertLessEqual(len(model.calls), 10)
            self.check_trace(state)
            outcomes.add(state.termination_reason)
            for failure in state.failures:
                if failure["code"] == "budget_exhausted":
                    # With the default budgets the event cap is never the binding limit.
                    self.assertIn(failure["detail"], (ctl.CLARIFICATION_BUDGET, ctl.RETRIEVAL_BUDGET))
        return outcomes

    def test_default_budgets_under_the_contract_fallback_rule(self):
        self.assertEqual(self.run_all("fallback_b3_order"), {"recommend", "fallback"})

    def test_default_budgets_under_the_rank_available_rule(self):
        self.assertEqual(self.run_all("rank_available_pool"), {"recommend", "budget_exhausted_rank"})


class ConflictingExhaustionRulesTest(ControllerCase):
    """Decision A1: two sources disagree on what happens when a budget runs out. Both are
    implemented, neither is the team's decision, and the shared validator cannot tell them apart."""

    def exhausted_clarifications(self, policy):
        return drive(self, self.start(exhaustion_policy=policy), ScriptedModel(decisions=[False] * 3))

    def test_contract_rule_returns_to_the_frozen_b3_order(self):
        state = self.exhausted_clarifications("fallback_b3_order")
        self.assertEqual(state.termination_reason, "fallback")
        self.assertEqual(state.reason, ctl.CLARIFICATION_BUDGET)
        self.assertEqual(state.ranked, state.initial_pool[:50])
        record, _ = self.check_trace(state)
        self.assertEqual((record["fallback_reason"], record["fallback_source"]),
                         (ctl.CLARIFICATION_BUDGET, "frozen_b3_order"))
        self.assertTrue(is_contract_termination(state))
        self.assertEqual(len(state.events[-1]["explanations"]), 10)  # a fallback still explains

    def test_midterm_rule_ranks_the_pool_it_has(self):
        state = self.exhausted_clarifications("rank_available_pool")
        self.assertEqual(state.termination_reason, "budget_exhausted_rank")
        self.assertEqual(state.reason, ctl.CLARIFICATION_BUDGET)
        self.assertEqual(state.ranked, sorted(state.pool, reverse=True)[:50])  # the model's order, not B3's
        record, _ = self.check_trace(state)
        self.assertNotIn("fallback_reason", record)
        self.assertFalse(is_contract_termination(state))  # the proposed v1 trace has no such label
        self.assertFalse(telemetry_record(state)["contract_termination"])

    def test_the_shared_validator_accepts_both_only_in_lenient_mode(self):
        for policy in POLICIES:
            state = self.exhausted_clarifications(policy)
            events, initial = state.events, state.initial_pool
            validate_episode(events, initial, allow_fallback=True)
            with self.assertRaises(ActionError, msg=policy):
                validate_episode(events, initial, allow_fallback=False)

    def test_retrieval_budget_exhaustion_under_each_rule(self):
        decisions = [True, False, False]  # assess T, evaluate F, (retrieve), evaluate F
        fallback = drive(self, self.start(exhaustion_policy="fallback_b3_order"), ScriptedModel(decisions=decisions))
        self.assertEqual((fallback.termination_reason, fallback.reason), ("fallback", ctl.RETRIEVAL_BUDGET))
        # The contract: the fallback ranking is B3's even though the trace records the expansion.
        self.assertEqual(fallback.ranked, fallback.initial_pool)
        self.assertGreater(len(fallback.pool), len(fallback.initial_pool))
        self.check_trace(fallback)

        ranked = drive(self, self.start(exhaustion_policy="rank_available_pool"), ScriptedModel(decisions=decisions))
        self.assertEqual((ranked.termination_reason, ranked.reason), ("budget_exhausted_rank", ctl.RETRIEVAL_BUDGET))
        self.assertTrue(set(ranked.ranked) & set(range(1000, 1005)))  # includes retrieved movies
        self.check_trace(ranked)

    def test_zero_budgets_are_honored(self):
        state = drive(self, self.start(max_clarifications=0), ScriptedModel(decisions=[False]))
        self.assertEqual((state.termination_reason, state.reason), ("fallback", ctl.CLARIFICATION_BUDGET))
        state = drive(self, self.start(max_retrievals=0), ScriptedModel(decisions=[True, False]))
        self.assertEqual((state.termination_reason, state.reason), ("fallback", ctl.RETRIEVAL_BUDGET))

    def test_pool_size_cap_blocks_retrieval(self):
        state = drive(self, self.start(pool=range(1, 6), max_pool_size=5), ScriptedModel(decisions=[True, False]))
        self.assertEqual((state.termination_reason, state.reason), ("fallback", ctl.POOL_FULL))

    def test_event_cap_stops_a_question_that_could_not_be_followed_up(self):
        state = drive(self, self.start(max_control_events=2), ScriptedModel(decisions=[False]))
        self.assertEqual((state.termination_reason, state.reason), ("fallback", ctl.EVENT_CAP))
        self.assertEqual(state.asks, 0)


class InvalidOutputTest(ControllerCase):
    def test_unusable_output_falls_back_by_default_with_an_honest_early_trace(self):
        model = ScriptedModel(raw={"assess_sufficiency": ["Sure, here is my answer!"]})
        state = drive(self, self.start(), model)
        self.assertEqual(state.termination_reason, "fallback")
        self.assertTrue(state.reason.startswith("invalid_action:assess_sufficiency"))
        self.assertEqual([e["action"] for e in state.events], ["rank_candidates", "explain_recommendations"])
        self.assertEqual(state.ranked, state.initial_pool)
        self.assertEqual(model.calls, ["assess_sufficiency", "explain_recommendations"])  # no invented judgments
        self.check_trace(state)

    def test_one_retry_can_recover(self):
        model = ScriptedModel(raw={"assess_sufficiency": ["not json"]})
        state = self.start(max_invalid_actions_per_step=1)
        state = drive(self, state, model)
        self.assertEqual(state.termination_reason, "recommend")
        self.assertEqual(model.calls.count("assess_sufficiency"), 2)
        self.assertEqual([d["valid"] for d in state.decisions[:2]], [False, True])
        self.assertEqual(state.step_invalid_count, 0)
        self.check_trace(state)

    def test_attempt_number_is_visible_to_the_caller(self):
        state = self.start(max_invalid_actions_per_step=2)
        self.assertEqual(ctl.next_step(state).attempt, 1)
        state = ctl.apply_model_output(state, "oops")
        self.assertEqual((ctl.next_step(state).task, ctl.next_step(state).attempt), ("assess_sufficiency", 2))

    def test_retries_run_out_and_then_fall_back(self):
        model = ScriptedModel(raw={"assess_sufficiency": ["x", "y"]})
        state = drive(self, self.start(max_invalid_actions_per_step=1), model)
        self.assertEqual(state.termination_reason, "fallback")
        self.assertEqual(model.calls.count("assess_sufficiency"), 2)
        self.assertEqual(len(state.failures), 2)

    def test_a_valid_but_wrong_action_is_not_accepted(self):
        wrong = serialize_action({"action": "retrieve_candidates", "query": "anything"})
        state = drive(self, self.start(), ScriptedModel(raw={"assess_sufficiency": [wrong]}))
        self.assertEqual(state.termination_reason, "fallback")
        self.assertEqual(state.failures[0]["code"], "unexpected_action")
        self.assertEqual(state.retrievals, 0)  # the stray action did not run

    def test_ranking_with_movies_outside_the_pool_is_rejected(self):
        bad = serialize_action({"action": "rank_candidates", "movie_ids": list(range(2, 52))})  # drops 1, adds 51
        state = drive(self, self.start(), ScriptedModel(raw={"rank_candidates": [bad]}))
        self.assertEqual(state.termination_reason, "fallback")
        self.assertEqual(state.failures[0]["code"], "unbound_ids")
        self.assertEqual(state.ranked, state.initial_pool)
        self.check_trace(state)

    def test_unusable_explanations_end_the_episode_as_failed_without_invented_text(self):
        bad = serialize_action({"action": "explain_recommendations",
                                "explanations": [{"movie_id": 999, "text": "Great."}]})
        state = drive(self, self.start(), ScriptedModel(raw={"explain_recommendations": [bad]}))
        self.assertEqual((state.phase, state.termination_reason), ("done", "failed"))
        self.assertEqual(state.reason, "explanation_invalid:unbound_ids")
        self.assertIsNotNone(state.ranked)  # the ranking is kept for whatever policy the team picks
        self.assertNotIn("explain_recommendations", [e["action"] for e in state.events])
        with self.assertRaises(TraceError):
            to_trace_record(state)
        self.assertEqual(telemetry_record(state)["termination_reason"], "failed")

    def test_repeated_question_is_rejected_unless_the_rule_is_switched_off(self):
        script = {"ask_clarification": [serialize_action({"action": "ask_clarification", "question": "Which mood?"}),
                                        serialize_action({"action": "ask_clarification", "question": "which MOOD"})]}
        state = drive(self, self.start(), ScriptedModel(decisions=[False, False], raw=script))
        self.assertEqual(state.termination_reason, "fallback")
        self.assertTrue(state.reason.endswith("repeated_question"))
        self.assertEqual(state.asks, 1)

        script = {"ask_clarification": [serialize_action({"action": "ask_clarification", "question": "Which mood?"}),
                                        serialize_action({"action": "ask_clarification", "question": "which MOOD"})]}
        state = drive(self, self.start(reject_repeated_questions=False),
                      ScriptedModel(decisions=[False, False, True], raw=script))
        self.assertEqual((state.termination_reason, state.asks), ("recommend", 2))


class RetrievalHandlingTest(ControllerCase):
    def at_retrieval(self, **config):
        state = self.start(history=(900,), **config)
        return drive(self, state, ScriptedModel(decisions=[True, False]), stop_at="retrieval")

    def test_duplicates_pool_members_and_watched_movies_are_dropped_and_counted(self):
        state = self.at_retrieval()
        step = ctl.next_step(state)
        self.assertEqual((step.query, step.max_new), ("search 1", 50))
        state = ctl.apply_retrieval(state, [1000, 1000, 1, 900, 1001], "idx-v9")
        self.assertEqual(state.events[-1], {"action": "retrieve_candidates", "added_movie_ids": [1000, 1001],
                                            "index_version": "idx-v9"})
        log = state.retrieval_log[0]
        self.assertEqual((log["returned"], log["added"], log["dropped_duplicate"], log["dropped_already_in_pool"],
                          log["dropped_watched"], log["dropped_over_capacity"]), (5, 2, 1, 1, 1, 0))
        self.assertEqual(log["query"], "search 1")
        self.assertEqual(state.phase, "evaluate")
        self.assertIsNone(state.pool_ok)  # the old pool judgment no longer applies

    def test_results_beyond_the_pool_cap_are_cut_in_retrieval_order(self):
        state = drive(self, self.start(pool=range(1, 6), max_pool_size=7), ScriptedModel(decisions=[True, False]),
                      stop_at="retrieval")
        self.assertEqual(ctl.next_step(state).max_new, 2)
        state = ctl.apply_retrieval(state, [1000, 1001, 1002, 1003, 1004], "idx")
        self.assertEqual(state.events[-1]["added_movie_ids"], [1000, 1001])
        self.assertEqual(state.retrieval_log[0]["dropped_over_capacity"], 3)
        self.assertEqual(len(state.pool), 7)

    def test_a_retrieval_that_adds_nothing_uses_up_the_attempt(self):
        for policy, expected in (("fallback_b3_order", "fallback"), ("rank_available_pool", "budget_exhausted_rank")):
            with self.subTest(policy):
                state = self.at_retrieval(exhaustion_policy=policy)
                state = ctl.apply_retrieval(state, [1, 900], "idx")  # in the pool / watched
                self.assertEqual(state.retrievals, 1)
                self.assertNotIn("retrieve_candidates", [e["action"] for e in state.events])  # nothing to record
                self.assertEqual((state.termination_reason, state.reason), (expected, ctl.NO_NEW_CANDIDATES))
                self.assertIn("retrieval_empty", [f["code"] for f in state.failures])
                state = drive(self, state, ScriptedModel())
                self.check_trace(state)

    def test_retrieval_error_uses_the_contract_fallback_under_either_policy(self):
        for policy in POLICIES:
            with self.subTest(policy):
                state = self.at_retrieval(exhaustion_policy=policy)
                state = ctl.apply_retrieval_failure(state, "timeout after 5s")
                self.assertEqual(state.termination_reason, "fallback")
                self.assertEqual(state.reason, "retrieval_error:timeout after 5s")
                state = drive(self, state, ScriptedModel())
                self.assertEqual(state.ranked, state.initial_pool)
                self.check_trace(state)

    def test_end_to_end_with_a_failing_retriever(self):
        def broken(step):
            raise Unavailable("index offline")

        state = drive(self, self.start(), ScriptedModel(decisions=[True, False]), retrieval=broken)
        self.assertEqual(state.termination_reason, "fallback")
        self.check_trace(state)

    def test_caller_mistakes_raise_and_leave_the_state_unchanged(self):
        state = self.at_retrieval()
        before = copy.deepcopy(state.to_dict())
        for bad_ids, version in (([1000], ""), (["1000"], "idx"), ([True], "idx"), ([0], "idx"), ("1000", "idx")):
            with self.subTest(ids=bad_ids, version=version), self.assertRaises(ctl.TransitionError):
                ctl.apply_retrieval(state, bad_ids, version)
        self.assertEqual(state.to_dict(), before)

    def test_functions_refuse_the_wrong_phase(self):
        state = self.start()
        for call in (lambda: ctl.apply_answer(state, "a", "s"),
                     lambda: ctl.apply_retrieval(state, [1000], "idx"),
                     lambda: ctl.apply_retrieval_failure(state, "x")):
            with self.assertRaises(ctl.TransitionError):
                call()
        asking = drive(self, state, ScriptedModel(decisions=[False]), stop_at="answer")
        with self.assertRaises(ctl.TransitionError):
            ctl.apply_model_output(asking, "{}")
        with self.assertRaises(ctl.TransitionError):
            ctl.apply_answer(asking, "  ", "s")
        with self.assertRaises(ctl.TransitionError):
            ctl.build_prompt(asking, catalog_for(range(1, 51)))


class PauseAndResumeTest(ControllerCase):
    def test_serialized_state_resumes_to_the_same_result(self):
        straight = drive(self, self.start(), ScriptedModel(decisions=[False, True]))
        paused = drive(self, self.start(), ScriptedModel(decisions=[False, True]), stop_at="answer")
        self.assertEqual(ctl.next_step(paused).question, "Question number 1?")
        revived = EpisodeState.from_dict(json.loads(json.dumps(paused.to_dict())))  # e.g. via Service A
        resumed = drive(self, revived, ScriptedModel(decisions=[True]))
        self.assertEqual(resumed.to_dict()["events"], straight.to_dict()["events"])
        self.assertEqual(resumed.to_dict(), straight.to_dict())

    def test_stateless_calls_rebuild_state_from_replayed_clarifications(self):
        """Three HTTP calls to a stateless Service B. Only the clarification list travels."""
        history, pool = [900], list(range(1, 51))
        answers = []

        def call(decisions):
            state = ctl.start_episode("Likes slow comedies.", pool, history, config=AgentConfig(),
                                      clarifications=answers)
            return drive(self, state, ScriptedModel(decisions=decisions), stop_at="answer")

        first = call([False])
        self.assertEqual((first.asks, first.control_events), (0, 1))
        answers.append({"question": ctl.next_step(first).question, "answer": "Heist films."})

        second = call([False])
        self.assertEqual((second.asks, second.resumed_clarifications, second.control_events), (1, 1, 3))
        answers.append({"question": ctl.next_step(second).question, "answer": "Not too violent."})

        # Two questions already used: a third is impossible however the model answers.
        third = call([False])
        self.assertEqual((third.termination_reason, third.reason), ("fallback", ctl.CLARIFICATION_BUDGET))
        self.assertEqual(third.asks, 2)

        # With enough information, the resumed conversation still has room for its one retrieval.
        resumed = call([True, False, True])
        self.assertEqual(resumed.termination_reason, "recommend")
        self.assertEqual(resumed.control_events, 8)
        self.assertEqual(resumed.retrievals, 1)
        self.assertEqual(resumed.events[0]["action"], "assess_sufficiency")  # earlier events are not invented
        self.assertEqual(resumed.clarifications[0]["answer_source"], "client_supplied")

    def test_episodes_do_not_share_state_when_interleaved(self):
        def run(pool_start, decisions):
            return ctl.start_episode("p", list(range(pool_start, pool_start + 50)), [], config=AgentConfig(),
                                     user_id=pool_start, split="validation"), ScriptedModel(decisions=decisions)

        alone = [drive(self, *run(1, [False, True])), drive(self, *run(500, [True, False, True]))]
        (a, model_a), (b, model_b) = run(1, [False, True]), run(500, [True, False, True])
        retrieve_a, retrieve_b = default_retrieval(), default_retrieval()
        for _ in range(60):  # advance each by one step in turn
            for which in ("a", "b"):
                state, model, retrieval = (a, model_a, retrieve_a) if which == "a" else (b, model_b, retrieve_b)
                step = ctl.next_step(state)
                if step.kind == "done":
                    continue
                if step.kind == "model":
                    state = ctl.apply_model_output(state, model.respond(step.task, state))
                elif step.kind == "answer":
                    state = ctl.apply_answer(state, "I like heists.", "test_double")  # same as drive()'s default
                else:
                    state = ctl.apply_retrieval(state, retrieval(step), "test-index-v1")
                if which == "a":
                    a = state
                else:
                    b = state
        self.assertEqual(a.to_dict(), alone[0].to_dict())
        self.assertEqual(b.to_dict(), alone[1].to_dict())


class PromptTest(ControllerCase):
    def test_prompt_shows_only_catalog_fields_and_never_the_watched_list_or_extras(self):
        catalog = catalog_for(range(1, 51))
        for doc in catalog.values():
            doc["future_rating"] = "SENTINEL_TARGET_RATING"  # a field a careless catalog join might carry
        state = ctl.start_episode("Likes slow comedies.", list(range(1, 51)), [4242, 4243], config=AgentConfig())
        for task in ("assess_sufficiency", "evaluate_candidates", "rank_candidates", "explain_recommendations"):
            self.assertEqual(ctl.next_step(state).task, task)
            prompt = ctl.build_prompt(state, catalog)
            self.assertNotIn("SENTINEL_TARGET_RATING", prompt)
            self.assertNotIn("4242", prompt)
            self.assertNotIn("4243", prompt)
            state = ctl.apply_model_output(state, valid_output(task, state))

    def test_explain_prompt_lists_the_top_ranked_movies_in_rank_order(self):
        catalog = catalog_for(range(1, 51))
        state = ctl.start_episode("p", list(range(1, 15)), [], config=AgentConfig())
        for task in ("assess_sufficiency", "evaluate_candidates", "rank_candidates"):
            state = ctl.apply_model_output(state, valid_output(task, state))
        lines = [line for line in ctl.build_prompt(state, catalog).splitlines() if line.startswith("[")]
        self.assertEqual([int(re.match(r"\[(\d+)\]", line)[1]) for line in lines], list(range(14, 4, -1)))

    def test_missing_catalog_entry_is_reported(self):
        state = self.start(pool=range(1, 6))
        state = ctl.apply_model_output(state, valid_output("assess_sufficiency", state))
        with self.assertRaises(ctl.MissingCandidateError):
            ctl.build_prompt(state, catalog_for([1, 2]))

    def test_candidate_tasks_use_the_shared_renderer(self):
        state = self.start(pool=range(1, 6))
        state = ctl.apply_model_output(state, valid_output("assess_sufficiency", state))
        catalog = catalog_for(range(1, 6))
        expected = render_prompt("evaluate_candidates", state.profile_text, [], [catalog[i] for i in range(1, 6)])
        self.assertEqual(ctl.build_prompt(state, catalog), expected)


class MemberOneExampleReplayTest(unittest.TestCase):
    """Replay Member 1's committed example episode (docs/p2_sample_examples.jsonl) through the controller."""

    LINE = re.compile(r"^\[(\d+)\] (.*?) \| genres: (.*?) \| tags: (.*)$")

    def load(self):
        rows = sorted((json.loads(line) for line in EXAMPLES.read_text(encoding="utf-8").splitlines()),
                      key=lambda row: row["step"])
        catalog = {}
        for row in rows:
            for line in row["prompt"].splitlines():
                match = self.LINE.match(line)
                if match:
                    doc = {"movie_id": int(match[1]), "title": match[2],
                           "genres": [] if match[3] == "unspecified" else match[3].split(", "),
                           "tags": [] if match[4] == "none" else match[4].split(", ")}
                    self.assertEqual(candidate_line(doc), line)
                    catalog[doc["movie_id"]] = doc
        return rows, catalog

    def replay_to_ranking(self):
        rows, catalog = self.load()
        initial = rows[5]["state"]["candidate_movie_ids"]
        state = ctl.start_episode(rows[0]["state"]["profile_text"], initial, [], config=AgentConfig(),
                                  user_id=rows[0]["user_id"], split="train")
        for index, row in enumerate(rows[:8]):
            step = ctl.next_step(state)
            self.assertEqual((step.kind, step.task), ("model", row["task"]), row["sample_id"])
            # Byte-for-byte the prompt Member 1 trained on.
            self.assertEqual(ctl.build_prompt(state, catalog), row["prompt"], row["sample_id"])
            state = ctl.apply_model_output(state, row["completion"])
            self.assertEqual(state.failures, [], row["sample_id"])
            if row["task"] == "ask_clarification":
                turn = rows[index + 1]["state"]["clarifications"][-1]
                self.assertEqual(ctl.next_step(state).question, turn["question"])
                state = ctl.apply_answer(state, turn["answer"], SIMULATOR_VERSION)
            if row["task"] == "retrieve_candidates":
                step = ctl.next_step(state)
                self.assertEqual((step.kind, step.query), ("retrieval", json.loads(row["completion"])["query"]))
                state = ctl.apply_retrieval(state, rows[7]["state"]["candidate_movie_ids"][50:], "member1-feature-index")
        return rows, catalog, state

    def test_first_eight_steps_reproduce_prompts_and_reach_the_event_cap_exactly(self):
        rows, _, state = self.replay_to_ranking()
        self.assertEqual((state.phase, state.control_events, state.asks, state.retrievals), ("rank", 8, 2, 1))
        self.assertEqual(state.pool, rows[7]["state"]["candidate_movie_ids"])
        self.assertEqual(len(state.pool), 100)

    def test_training_rank_prompt_differs_from_what_inference_can_show(self):
        """Decision A6. Member 1's rank samples add held-out items to the pool for supervision, so the
        training prompt has movies the retriever never returned. A strict controller rejects that
        completion, and the explain step then sees a different top ten."""
        rows, catalog, state = self.replay_to_ranking()
        row = rows[8]
        training_ids = row["state"]["candidate_movie_ids"]
        injected = [i for i in training_ids if i not in set(state.pool)]
        self.assertGreater(len(training_ids), len(state.pool))
        self.assertTrue(injected)
        self.assertNotEqual(ctl.build_prompt(state, catalog), row["prompt"])

        rejected = ctl.apply_model_output(state, row["completion"])
        self.assertEqual(rejected.termination_reason, "fallback")
        self.assertEqual([d["error"]["code"] for d in rejected.decisions if not d["valid"]], ["unbound_ids"])

        kept = [i for i in json.loads(row["completion"])["movie_ids"] if i in set(state.pool)]
        self.assertEqual(sorted(kept), sorted(state.pool))
        accepted = ctl.apply_model_output(state, serialize_action({"action": "rank_candidates", "movie_ids": kept}))
        self.assertEqual((accepted.phase, accepted.ranked), ("explain", kept[:50]))

        shown = accepted.ranked[:10]
        explanations = [{"movie_id": i, "text": f"Because {i}."} for i in shown]
        done = ctl.apply_model_output(
            accepted, serialize_action({"action": "explain_recommendations", "explanations": explanations}))
        self.assertEqual(done.termination_reason, "recommend")
        # The whole replayed episode is legal under the strictest reading of the shared validator.
        summary = validate_episode(done.events, done.initial_pool)
        self.assertEqual(summary["control_events"], 8)
        self.assertEqual(summary["final_pool_size"], 100)


class TelemetryAndTraceTest(ControllerCase):
    def test_telemetry_holds_digests_and_config_not_trace_fields(self):
        raw = serialize_action({"action": "assess_sufficiency", "sufficient": True})
        state = drive(self, self.start(), ScriptedModel(raw={"assess_sufficiency": [raw]}))
        telemetry = telemetry_record(state)
        self.assertEqual(telemetry["decisions"][0]["raw_sha256"], hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(telemetry["config"]["label"], "unratified-defaults")
        self.assertTrue(telemetry["contract_termination"])
        self.assertEqual(json.loads(json.dumps(telemetry)), telemetry)
        self.assertNotIn("decisions", to_trace_record(state))  # contract fields stay clean

    def test_fallback_decision_is_labelled_not_passed_off_as_a_model_choice(self):
        state = drive(self, self.start(), ScriptedModel(raw={"assess_sufficiency": ["bad"]}))
        sources = [(d["task"], d["source"]) for d in state.decisions]
        self.assertIn(("rank_candidates", "fallback"), sources)
        self.assertEqual([d["source"] for d in state.decisions if d["task"] == "rank_candidates"], ["fallback"])

    def test_no_trace_before_the_episode_finishes_or_without_identity(self):
        with self.assertRaises(TraceError):
            to_trace_record(self.start())
        anonymous = drive(self, ctl.start_episode("p", list(range(1, 51)), [], config=AgentConfig()), ScriptedModel())
        with self.assertRaises(TraceError):
            to_trace_record(anonymous)


class AdversarialFuzzTest(ControllerCase):
    """Seeded chaos: garbage, wrong actions, random judgments and rankings, flaky retrieval."""

    JUNK = ["", "{", "null", "[]", "hello", '{"action": "assess_sufficiency"}',
            '{"action": "rank_candidates", "movie_ids": [1, 1]}']

    def chaos_output(self, rng, task, state):
        roll = rng.random()
        if roll < 0.12:
            return rng.choice(self.JUNK)
        if roll < 0.2:
            return valid_output(rng.choice([t for t in ACTIONS if t != task]), state)
        if task == "rank_candidates":
            order = list(state.pool)
            rng.shuffle(order)
            if rng.random() < 0.15:
                order = order[1:]  # forget a movie
            return serialize_action({"action": task, "movie_ids": order}) if order else "{}"
        if task == "explain_recommendations" and rng.random() < 0.2:
            return serialize_action({"action": task, "explanations": [{"movie_id": 1, "text": "x"}]})
        if task == "ask_clarification" and state.clarifications and rng.random() < 0.3:
            return serialize_action({"action": task, "question": state.clarifications[0]["question"]})
        return valid_output(task, state, decision=rng.random() < 0.5)

    def test_every_seed_terminates_legally_within_bounds(self):
        seen = set()
        for seed in range(400):
            rng = random.Random(seed)
            policy = POLICIES[seed % 2]
            retries = seed // 2 % 2
            history = [3, 4, 1001, 1002, 1003]
            config = AgentConfig(exhaustion_policy=policy, max_invalid_actions_per_step=retries,
                                 max_pool_size=rng.choice([50, 52, 100]), ranking_size=rng.choice([5, 10, 50]))
            pool = [i for i in range(1, 61) if i not in history][: rng.choice([30, 50])]
            state = ctl.start_episode("p", pool, history, config=config, user_id=seed + 1, split="validation")
            counter = itertools.count(1000)

            def retrieve(step):
                roll = rng.random()
                if roll < 0.1:
                    raise Unavailable("flaky")
                if roll < 0.25:
                    return []
                ids = [next(counter) for _ in range(rng.randint(1, 8))]
                return ids + [1, 1001] + ids[:1]  # also a pool member, a watched movie and a duplicate

            class Chaos:
                def respond(_, task, current):
                    return self.chaos_output(rng, task, current)

            with self.subTest(seed=seed):
                state = drive(self, state, Chaos(), retrieval=retrieve)
                self.assertEqual(state.phase, "done")
                seen.add(state.termination_reason)
                model_calls = [d for d in state.decisions if d["source"] == "model"]
                self.assertLessEqual(len(model_calls), (config.max_control_events + 3) * (retries + 1))
                self.assertLessEqual(state.control_events, config.max_control_events)
                self.assertLessEqual(state.asks, config.max_clarifications)
                self.assertLessEqual(state.retrievals, config.max_retrievals)
                self.assertLessEqual(len(state.pool), max(config.max_pool_size, len(pool)))
                self.assertEqual(state.history_movie_ids, history)
                if state.termination_reason == "failed":
                    with self.assertRaises(TraceError):
                        to_trace_record(state)
                    continue
                self.check_trace(state)
                if state.termination_reason == "recommend":
                    judged = [e for e in state.events if e["action"] in ("assess_sufficiency", "evaluate_candidates")]
                    self.assertTrue(judged[-1]["action"] == "evaluate_candidates" and judged[-1]["pool_sufficient"])
                    self.assertTrue([e for e in judged if e["action"] == "assess_sufficiency"][-1]["sufficient"])
                if state.termination_reason == "fallback":
                    self.assertEqual(state.ranked, state.initial_pool[: config.ranking_size])
        # The fuzz must actually reach every ending, or it proves little.
        self.assertTrue({"recommend", "fallback", "budget_exhausted_rank", "failed"} <= seen, seen)


if __name__ == "__main__":
    unittest.main()
