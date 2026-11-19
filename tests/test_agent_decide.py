"""Tests for the callable entry point (decide/resume) and the runnable example.

The model and retriever are the stand-ins from src/agent/example.py (rules, not a language model; a
made-up catalog, not an index). They prove the calling pattern and the handling of failures.
"""

import copy
import json
import math
import unittest

from src.agent import controller as ctl
from src.agent import example
from src.agent.config import AgentConfig
from src.agent.decide import RequestError, RetrievalResult, advance, decide, resume
from src.agent.grounding import UNSUPPORTED
from src.agent.state import EpisodeState, StateError
from src.agent.trace import telemetry_record

REQUEST = {"profile_text": example.PROFILE, "history_movie_ids": example.HISTORY, "clarifications": []}
CALL = dict(model=example.DemoModel(), retriever=example.DemoRetriever(), catalog=example.CATALOG)


class SpyRetriever(example.DemoRetriever):
    def __init__(self):
        self.searches, self.initial_calls = [], []

    def initial_pool(self, **kwargs):
        self.initial_calls.append(kwargs)
        return super().initial_pool(**kwargs)

    def search(self, **kwargs):
        self.searches.append(kwargs)
        return super().search(**kwargs)


class RecordingModel(example.DemoModel):
    def __init__(self):
        self.prompts = []

    def generate(self, prompt):
        self.prompts.append(prompt)
        return super().generate(prompt)


class ExampleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.results = example.main(show=False)

    def test_first_call_asks_a_question_and_pauses(self):
        first = self.results["first"]
        self.assertEqual(first.response["action"], "ask")
        self.assertTrue(first.response["question"].strip())
        self.assertEqual((first.state.phase, first.state.pending_question), ("await_answer", first.response["question"]))

    def test_recommendation_is_ten_unseen_ranked_items_with_finite_scores_and_explanations(self):
        for key in ("replayed", "resumed"):
            with self.subTest(key):
                response = self.results[key].response
                self.assertEqual(response["action"], "recommend")
                items = response["items"]
                self.assertEqual([i["rank"] for i in items], list(range(1, 11)))
                ids = [i["movie_id"] for i in items]
                self.assertEqual(len(set(ids)), 10)
                self.assertFalse(set(ids) & set(example.HISTORY))
                scores = [i["score"] for i in items]
                self.assertTrue(all(math.isfinite(s) for s in scores))
                self.assertEqual(scores, sorted(scores, reverse=True))
                self.assertTrue(all(i["explanation"].strip() for i in items))

    def test_stateless_replay_and_explicit_state_give_the_same_recommendations(self):
        self.assertEqual(self.results["replayed"].response, self.results["resumed"].response)

    def test_the_example_explanations_pass_the_grounding_screen(self):
        for key in ("replayed", "resumed", "degraded"):
            statuses = {r["status"] for g in self.results[key].state.grounding for r in g["results"]}
            self.assertNotIn(UNSUPPORTED, statuses, key)
            self.assertTrue(statuses, key)

    def test_the_degraded_call_is_explicit_and_returns_the_fallback_order(self):
        degraded = self.results["degraded"]
        state = degraded.state
        self.assertEqual((state.termination_reason, state.reason), ("fallback", "model_error:assess_sufficiency"))
        self.assertEqual(degraded.response["action"], "recommend")
        self.assertEqual([i["movie_id"] for i in degraded.response["items"]], state.initial_pool[:10])
        self.assertEqual(telemetry_record(state)["failures"][0]["code"], "model_error")

    def test_example_output_is_labelled_as_stand_ins(self):
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            example.main(show=True)
        self.assertIn("STAND-INS ONLY", buffer.getvalue())


class DecideTest(unittest.TestCase):
    def test_collaborators_receive_the_documented_arguments(self):
        retriever, model = SpyRetriever(), RecordingModel()
        turns = [{"question": "Which genres?", "answer": "Thriller."}]
        decide({**REQUEST, "clarifications": turns}, model=model, retriever=retriever, catalog=example.CATALOG)
        self.assertEqual(retriever.initial_calls[0]["clarifications"], turns)
        self.assertEqual(retriever.initial_calls[0]["history_movie_ids"], example.HISTORY)
        search = retriever.searches[0]
        self.assertEqual(search["max_new"], 88)  # room left under max_pool_size (100 - 12)
        self.assertTrue(set(example.HISTORY) <= set(search["exclude_movie_ids"]))  # watched movies are excluded
        self.assertEqual(search["exclude_movie_ids"], sorted(search["exclude_movie_ids"]))
        self.assertTrue(search["query"].startswith("movies similar to"))

    def test_the_model_only_ever_sees_prompt_text_not_the_history_list(self):
        model = RecordingModel()
        decide({**REQUEST, "history_movie_ids": [1, 3, 17]}, model=model, retriever=example.DemoRetriever(),
               catalog=example.CATALOG)
        self.assertTrue(model.prompts)
        for prompt in model.prompts:
            self.assertNotIn("history_movie_ids", prompt)
            self.assertNotIn("[17]", prompt)  # a watched movie is not offered as a candidate

    def test_a_model_exception_is_an_ordinary_outcome_not_a_crash(self):
        class Dead(example.DemoModel):
            def generate(self, prompt):
                raise MemoryError("out of memory")

        decision = decide(REQUEST, model=Dead(), retriever=example.DemoRetriever(), catalog=example.CATALOG)
        # The fallback ranking exists, but there is no model to write explanations: report, don't invent.
        self.assertEqual(decision.response["action"], "error")
        self.assertEqual(decision.state.termination_reason, "failed")
        self.assertEqual(decision.state.reason, "explanation_invalid:model_error")
        self.assertIn("MemoryError", decision.state.failures[0]["detail"])

    def test_a_retriever_exception_falls_back_to_the_retrievers_own_order(self):
        class Broken(example.DemoRetriever):
            def search(self, **kwargs):
                raise ConnectionError("index unreachable")

        replay = {**REQUEST, "clarifications": [{"question": "Which genres?", "answer": "Thriller."}]}
        decision = decide(replay, model=example.DemoModel(), retriever=Broken(), catalog=example.CATALOG)
        self.assertEqual(decision.response["action"], "recommend")
        self.assertEqual(decision.state.termination_reason, "fallback")
        self.assertTrue(decision.state.reason.startswith("retrieval_error:ConnectionError"))
        self.assertEqual([i["movie_id"] for i in decision.response["items"]], decision.state.initial_pool[:10])

    def test_a_catalog_gap_is_a_data_bug_and_is_raised(self):
        gappy = {i: doc for i, doc in example.CATALOG.items() if i != 2}
        replay = {**REQUEST, "clarifications": [{"question": "Which genres?", "answer": "Thriller."}]}
        with self.assertRaises(ctl.MissingCandidateError):  # raised once candidates are shown to the model
            decide(replay, model=example.DemoModel(), retriever=example.DemoRetriever(), catalog=gappy)

    def test_a_retriever_returning_a_watched_movie_in_the_initial_pool_is_rejected(self):
        class Leaky(example.DemoRetriever):
            def initial_pool(self, **kwargs):
                return RetrievalResult([1, 2, 4], "idx")  # movie 1 is in the history

        with self.assertRaises(ctl.TransitionError):
            decide(REQUEST, model=example.DemoModel(), retriever=Leaky(), catalog=example.CATALOG)

    def test_malformed_requests_are_reported(self):
        bad = [None, {}, {**REQUEST, "profile_text": 5}, {**REQUEST, "history_movie_ids": "1,2"},
               {**REQUEST, "history_movie_ids": [True]}, {**REQUEST, "clarifications": [{"question": "q"}]},
               {**REQUEST, "clarifications": [{"question": "q", "answer": "  "}]}, {**REQUEST, "clarifications": "x"}]
        for request in bad:
            with self.subTest(request=request), self.assertRaises(RequestError):
                decide(request, **CALL)

    def test_the_request_is_not_modified(self):
        request = copy.deepcopy({**REQUEST, "clarifications": [{"question": "Q?", "answer": "A."}]})
        before = copy.deepcopy(request)
        decide(request, **CALL)
        self.assertEqual(request, before)

    def test_config_is_honoured_and_recorded(self):
        config = AgentConfig(label="my-run", max_clarifications=0)
        decision = decide(REQUEST, config=config, **CALL)
        self.assertEqual(decision.state.config.label, "my-run")
        self.assertEqual(decision.state.reason, ctl.CLARIFICATION_BUDGET)  # no question allowed, so the fallback
        self.assertEqual(decision.state.initial_index_version, example.DemoRetriever.index_version)

    def test_user_id_and_split_reach_the_trace(self):
        decision = decide({**REQUEST, "user_id": 7, "split": "validation"}, **CALL)
        self.assertEqual((decision.state.user_id, decision.state.split), (7, "validation"))


class ResumeTest(unittest.TestCase):
    def paused(self):
        return decide(REQUEST, **CALL).state

    def test_resume_accepts_a_json_round_tripped_state_and_leaves_the_original_alone(self):
        state = self.paused()
        before = copy.deepcopy(state.to_dict())
        decision = resume(json.loads(json.dumps(state.to_dict())), "Thriller.", "user", **CALL)
        self.assertEqual(decision.response["action"], "recommend")
        self.assertEqual(state.to_dict(), before)

    def test_resume_keeps_the_full_event_history(self):
        decision = resume(self.paused(), "Thriller.", "user", **CALL)
        actions = [e["action"] for e in decision.state.events]
        self.assertEqual(actions[:3], ["assess_sufficiency", "ask_clarification", "assess_sufficiency"])
        self.assertEqual(decision.state.clarifications[0]["answer_source"], "user")

    def test_resume_validates_untrusted_state(self):
        data = self.paused().to_dict()
        for name, mutate in {
            "tampered phase": lambda d: d.update(phase="celebrate"),
            "watched movie in pool": lambda d: d.update(history_movie_ids=d["pool"][:1]),
            "extra field": lambda d: d.update(surprise=1),
        }.items():
            broken = copy.deepcopy(data)
            mutate(broken)
            with self.subTest(name), self.assertRaises(StateError):
                resume(broken, "Thriller.", "user", **CALL)

    def test_resume_refuses_a_state_that_is_not_waiting_for_an_answer(self):
        finished = decide({**REQUEST, "clarifications": [{"question": "Q?", "answer": "A."}]}, **CALL).state
        with self.assertRaises(ctl.TransitionError):
            resume(finished, "More.", "user", **CALL)

    def test_budget_carries_across_resume(self):
        config = AgentConfig(max_clarifications=1)

        class AlwaysAsk(example.DemoModel):
            def generate(self, prompt):
                task = example.re.search(r'Output format: \{"action": "(\w+)"', prompt).group(1)
                if task == "assess_sufficiency":
                    return example._action(task, sufficient=False)
                return super().generate(prompt)

        call = dict(model=AlwaysAsk(), retriever=example.DemoRetriever(), catalog=example.CATALOG)
        first = decide(REQUEST, config=config, **call)
        self.assertEqual(first.response["action"], "ask")
        second = resume(first.state, "Thriller.", "user", **call)  # still insufficient, but no question left
        self.assertEqual(second.state.reason, ctl.CLARIFICATION_BUDGET)
        self.assertEqual(second.state.asks, 1)

    def test_advance_on_a_finished_state_just_reports_it(self):
        done = decide({**REQUEST, "clarifications": [{"question": "Q?", "answer": "A."}]}, **CALL)
        again = advance(done.state, **CALL)
        self.assertEqual(again.response, done.response)


if __name__ == "__main__":
    unittest.main()
