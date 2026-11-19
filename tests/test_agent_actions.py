import json
import unittest
from pathlib import Path

from src.agent import actions as agent_actions
from src.agent.actions import (
    INVALID_ACTION, REPEATED_QUESTION, UNBOUND_IDS, UNEXPECTED_ACTION, AgentActionError,
    normalize_question, parse_model_output,
)
from src.data import p2_actions
from src.profiles import agent_text
from src.profiles.agent_text import serialize_action

EXAMPLES = Path(__file__).resolve().parent.parent / "docs" / "p2_sample_examples.jsonl"


def raw(**action):
    return serialize_action(action)


class SharedGrammarIsReusedTest(unittest.TestCase):
    def test_parser_module_uses_the_shared_objects_not_copies(self):
        # If these were redefined, training and inference could drift apart silently.
        self.assertIs(agent_actions.parse_action, p2_actions.parse_action)
        self.assertIs(agent_actions.validate_action, p2_actions.validate_action)
        self.assertIs(agent_actions.ActionError, p2_actions.ActionError)

    def test_phase_map_covers_exactly_the_shared_actions(self):
        from src.agent.state import PHASE_TASK

        self.assertEqual(set(PHASE_TASK.values()), set(agent_text.ACTIONS))


class ExpectedActionTest(unittest.TestCase):
    def test_accepts_each_requested_action(self):
        cases = {
            "assess_sufficiency": (raw(action="assess_sufficiency", sufficient=False), {}),
            "ask_clarification": (raw(action="ask_clarification", question="Which mood?"), {}),
            "evaluate_candidates": (raw(action="evaluate_candidates", pool_sufficient=True), {}),
            "retrieve_candidates": (raw(action="retrieve_candidates", query="slow thrillers"), {}),
            "rank_candidates": (raw(action="rank_candidates", movie_ids=[3, 1, 2]), {"pool_ids": [1, 2, 3]}),
            "explain_recommendations": (
                raw(action="explain_recommendations", explanations=[{"movie_id": 3, "text": "t"}]),
                {"explain_ids": [3]},
            ),
        }
        for task, (text, kwargs) in cases.items():
            with self.subTest(task):
                self.assertEqual(parse_model_output(text, task, **kwargs)["action"], task)

    def test_valid_but_wrong_action_is_rejected(self):
        text = raw(action="retrieve_candidates", query="x")
        with self.assertRaises(AgentActionError) as caught:
            parse_model_output(text, "assess_sufficiency")
        self.assertEqual(caught.exception.code, UNEXPECTED_ACTION)

    def test_grammar_violations_use_the_shared_parser(self):
        bad = {
            "prose around json": 'Sure! {"action": "assess_sufficiency", "sufficient": true}',
            "code fence": '```json\n{"action": "assess_sufficiency", "sufficient": true}\n```',
            "duplicate keys": '{"action": "assess_sufficiency", "sufficient": true, "sufficient": false}',
            "non-finite number": '{"action": "rank_candidates", "movie_ids": [NaN]}',
            "extra field": '{"action": "assess_sufficiency", "sufficient": true, "why": "x"}',
            "missing field": '{"action": "assess_sufficiency"}',
            "string boolean": '{"action": "assess_sufficiency", "sufficient": "true"}',
            "unknown action": '{"action": "buy_tickets"}',
            "not an object": "[1, 2]",
            "empty": "",
        }
        for name, text in bad.items():
            with self.subTest(name), self.assertRaises(AgentActionError) as caught:
                parse_model_output(text, "assess_sufficiency")
            self.assertEqual(caught.exception.code, INVALID_ACTION)

    def test_non_text_output_is_a_normal_failure_not_a_crash(self):
        for value in (None, 7, b'{"action": "assess_sufficiency", "sufficient": true}'):
            with self.subTest(value=value), self.assertRaises(AgentActionError):
                parse_model_output(value, "assess_sufficiency")


class IdBindingTest(unittest.TestCase):
    POOL = [10, 20, 30]

    def test_ranking_must_be_a_permutation_of_the_pool_shown(self):
        bad = {
            "missing movie": [10, 20],
            "extra movie": [10, 20, 30, 40],
            "unseen movie replaces one": [10, 20, 99],
            "duplicate": [10, 10, 20, 30],
        }
        for name, ids in bad.items():
            with self.subTest(name), self.assertRaises(AgentActionError) as caught:
                parse_model_output(raw(action="rank_candidates", movie_ids=ids), "rank_candidates", pool_ids=self.POOL)
            self.assertIn(caught.exception.code, (UNBOUND_IDS, INVALID_ACTION))
        # Any order of the pool is fine.
        parse_model_output(raw(action="rank_candidates", movie_ids=[30, 10, 20]), "rank_candidates", pool_ids=self.POOL)

    def test_explanations_must_cover_the_shown_movies_in_order(self):
        def explain(ids):
            return raw(action="explain_recommendations", explanations=[{"movie_id": i, "text": "t"} for i in ids])

        parse_model_output(explain([30, 10]), "explain_recommendations", explain_ids=[30, 10])
        for ids in ([10, 30], [30], [30, 10, 20], [30, 99]):
            with self.subTest(ids=ids), self.assertRaises(AgentActionError) as caught:
                parse_model_output(explain(ids), "explain_recommendations", explain_ids=[30, 10])
            self.assertEqual(caught.exception.code, UNBOUND_IDS)

    def test_binding_arguments_are_required_for_those_tasks(self):
        with self.assertRaises(ValueError):
            parse_model_output(raw(action="rank_candidates", movie_ids=[1]), "rank_candidates")
        with self.assertRaises(ValueError):
            parse_model_output(
                raw(action="explain_recommendations", explanations=[{"movie_id": 1, "text": "t"}]),
                "explain_recommendations",
            )


class RepeatedQuestionTest(unittest.TestCase):
    def test_repeat_is_rejected_ignoring_case_punctuation_and_spacing(self):
        text = raw(action="ask_clarification", question="What  is one of your FAVORITE movies")
        with self.assertRaises(AgentActionError) as caught:
            parse_model_output(text, "ask_clarification", previous_questions=["What is one of your favorite movies?"])
        self.assertEqual(caught.exception.code, REPEATED_QUESTION)

    def test_a_different_question_is_accepted(self):
        text = raw(action="ask_clarification", question="Do you prefer older films?")
        parse_model_output(text, "ask_clarification", previous_questions=["What is one of your favorite movies?"])

    def test_the_rule_can_be_switched_off(self):
        text = raw(action="ask_clarification", question="Same?")
        parse_model_output(text, "ask_clarification", previous_questions=["Same?"], reject_repeated_questions=False)

    def test_normalization(self):
        self.assertEqual(normalize_question("  Hi,   THERE!? "), normalize_question("hi there"))


class MemberOneCompletionsTest(unittest.TestCase):
    """Member 1's ground-truth completions must parse under the controller's strict rules."""

    def test_every_committed_example_completion_is_accepted_in_context(self):
        rows = [json.loads(line) for line in EXAMPLES.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 10)
        for row in rows:
            ids = row["state"]["candidate_movie_ids"]
            kwargs = {}
            if row["task"] == "rank_candidates":
                kwargs["pool_ids"] = ids
            if row["task"] == "explain_recommendations":
                kwargs["explain_ids"] = ids
            with self.subTest(sample=row["sample_id"]):
                action = parse_model_output(row["completion"], row["task"], **kwargs)
                self.assertEqual(action, row["target"])


if __name__ == "__main__":
    unittest.main()
