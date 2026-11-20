import json
import unittest

from src.agent import controller as ctl
from src.agent.config import AgentConfig, ConfigError
from src.agent.state import STATE_VERSION, EpisodeState, StateError
from tests.test_agent_controller import ScriptedModel, Unavailable, drive


def fresh(**overrides):
    kwargs = dict(profile_text="Likes comedies.", initial_pool=[1, 2, 3, 4, 5], history_movie_ids=[90, 91])
    kwargs.update(overrides)
    return ctl.start_episode(**kwargs)


class ConfigTest(unittest.TestCase):
    def test_defaults_are_the_documented_unratified_proposals(self):
        config = AgentConfig()
        self.assertEqual((config.max_clarifications, config.max_retrievals, config.max_pool_size), (2, 1, 100))
        self.assertEqual((config.max_control_events, config.ranking_size, config.explain_count), (8, 50, 10))
        self.assertEqual(config.label, "unratified-defaults")

    def test_event_cap_equals_worst_case_of_the_default_budgets(self):
        # 3 assessments + 2 questions + 2 pool evaluations + 1 retrieval = 8
        self.assertEqual(AgentConfig().worst_case_control_events, AgentConfig().max_control_events)

    def test_explained_movies_follow_the_contract_and_are_not_a_free_setting(self):
        # The shared validator requires explanations for exactly min(10, ranked), so this is derived.
        self.assertEqual(AgentConfig(ranking_size=50).explain_count, 10)
        self.assertEqual(AgentConfig(ranking_size=5).explain_count, 5)
        with self.assertRaises(TypeError):
            AgentConfig(explain_count=3)

    def test_rejects_invalid_values(self):
        for bad in (
            dict(max_clarifications=-1), dict(max_retrievals=True), dict(max_pool_size=0),
            dict(max_control_events=0), dict(ranking_size=0),
            dict(exhaustion_policy="whatever"), dict(label=" "), dict(reject_repeated_questions="yes"),
            dict(max_invalid_actions_per_step=-1),
        ):
            with self.subTest(bad=bad), self.assertRaises(ConfigError):
                AgentConfig(**bad)

    def test_round_trip_and_strict_loading(self):
        config = AgentConfig(max_clarifications=1, exhaustion_policy="rank_available_pool")
        self.assertEqual(AgentConfig.from_dict(json.loads(json.dumps(config.to_dict()))), config)
        partial = config.to_dict()
        del partial["max_retrievals"]
        with self.assertRaises(ConfigError):  # a stored state must not silently adopt a newer default
            AgentConfig.from_dict(partial)


class StateSerializationTest(unittest.TestCase):
    def test_json_round_trip_is_lossless(self):
        state = fresh(user_id=7, split="validation")
        loaded = EpisodeState.from_dict(json.loads(json.dumps(state.to_dict())))
        self.assertEqual(loaded.to_dict(), state.to_dict())
        self.assertEqual(loaded.schema_version, STATE_VERSION)

    def test_to_dict_does_not_alias_the_state(self):
        state = fresh()
        data = state.to_dict()
        data["pool"].append(999)
        data["config"]["max_retrievals"] = 9
        self.assertEqual(state.pool, [1, 2, 3, 4, 5])
        self.assertEqual(state.config.max_retrievals, 1)

    def test_tampered_state_is_rejected(self):
        base = fresh().to_dict()

        def mutate(**changes):
            data = json.loads(json.dumps(base))
            data.update(changes)
            return data

        cases = {
            "unknown phase": mutate(phase="celebrate"),
            "wrong version": mutate(schema_version="p2-agent-state-v0"),
            "duplicate pool ids": mutate(pool=[1, 1, 2], initial_pool=[1, 1, 2]),
            "boolean as movie id": mutate(pool=[True, 2], initial_pool=[True, 2]),
            "watched movie in pool": mutate(history_movie_ids=[3]),
            "pool does not start with initial pool": mutate(pool=[5, 4, 3, 2, 1]),
            "pool beyond cap": mutate(pool=list(range(1, 200)), initial_pool=[1, 2, 3, 4, 5]),
            "retrievals beyond budget": mutate(retrievals=5),
            "await_answer without question": mutate(phase="await_answer"),
            "question outside await_answer": mutate(pending_question="Hmm?"),
            "explain without ranking": mutate(phase="explain"),
            "done without reason": mutate(phase="done"),
            "ranked outside pool": mutate(ranked=[77], phase="explain"),
            "negative counter": mutate(control_events=-1),
            "extra field": {**base, "surprise": 1},
            "missing field": {key: value for key, value in base.items() if key != "events"},
            "bad clarification": mutate(clarifications=[{"question": "q", "answer": ""}]),
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(StateError):
                EpisodeState.from_dict(data)

    def test_untrusted_dict_is_not_aliased_by_loaded_state(self):
        data = fresh().to_dict()
        state = EpisodeState.from_dict(data)
        data["pool"].append(555)
        data["events"].append({"action": "x"})
        self.assertNotIn(555, state.pool)
        self.assertEqual(state.events, [])


class ResumedStateConsistencyTest(unittest.TestCase):
    """A state that leaves the process can be edited. The counters and records must agree with each
    other, so editing one number (to regain budget, say) is caught. A rewrite that edits every related
    field consistently is not caught here; signing the state would be the service's job."""

    @classmethod
    def setUpClass(cls):
        # asks once, widens once: events, decisions, a retrieval log and a clarification all exist
        done = drive(unittest.TestCase(), fresh(user_id=7, split="validation", initial_pool=list(range(1, 51))),
                     ScriptedModel(decisions=[False, True, False, True]))
        assert done.termination_reason == "recommend" and done.retrievals == 1 and done.asks == 1
        cls.base = json.loads(json.dumps(done.to_dict()))

    def mutate(self, **changes):
        data = json.loads(json.dumps(self.base))
        data.update(changes)
        return data

    def test_the_untouched_state_loads(self):
        self.assertEqual(EpisodeState.from_dict(self.mutate()).to_dict(), self.base)

    def test_edited_counters_are_rejected(self):
        events = self.base["events"]
        without = lambda action: [e for e in events if e["action"] != action]
        cases = {
            "fewer control events than the events show": self.mutate(control_events=0),
            "more control events than the events show": self.mutate(control_events=self.base["control_events"] + 1),
            "question event removed, answer kept": self.mutate(events=without("ask_clarification")),
            "answer removed, question event kept": self.mutate(clarifications=[]),
            "replayed count raised": self.mutate(resumed_clarifications=1),
            "retrieval event removed": self.mutate(events=without("retrieve_candidates")),
            "retrieval count below the log": self.mutate(retrievals=0, retrieval_log=self.base["retrieval_log"]),
            "step retries beyond the allowance": self.mutate(step_invalid_count=5),
            "retrieval count and log both cleared, event kept": self.mutate(retrievals=0, retrieval_log=[]),
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(StateError):
                EpisodeState.from_dict(data)

    def test_malformed_records_are_rejected_not_left_to_fail_later(self):
        decision, failure = self.base["decisions"][0], {"code": "x", "stage": "y", "detail": "z"}
        dropped = lambda record, key: {k: v for k, v in record.items() if k != key}
        cases = {
            "decision without valid": self.mutate(decisions=[dropped(decision, "valid")] + self.base["decisions"][1:]),
            "decision with a wrong index": self.mutate(decisions=[{**decision, "index": 5}] + self.base["decisions"][1:]),
            "decision from an unknown source": self.mutate(decisions=[{**decision, "source": "oracle"}] + self.base["decisions"][1:]),
            "decision for an unknown task": self.mutate(decisions=[{**decision, "task": "dance"}] + self.base["decisions"][1:]),
            "valid decision that carries an error": self.mutate(decisions=[{**decision, "error": {"code": "c", "message": "m"}}] + self.base["decisions"][1:]),
            "invalid decision without an error": self.mutate(decisions=[{**decision, "valid": False, "error": None}] + self.base["decisions"][1:]),
            "error without a code": self.mutate(decisions=[{**decision, "valid": False, "error": {"message": "m"}}] + self.base["decisions"][1:]),
            "failure without a code": self.mutate(failures=[dropped(failure, "code")]),
            "failure with a non-text stage": self.mutate(failures=[{**failure, "stage": 3}]),
            "event without an action": self.mutate(events=[{"sufficient": True}] + self.base["events"]),
            "event with an unknown action": self.mutate(events=[{"action": "dance"}] + self.base["events"]),
        }
        for name, data in cases.items():
            with self.subTest(name), self.assertRaises(StateError):
                EpisodeState.from_dict(data)

    def test_every_state_the_controller_can_reach_survives_a_round_trip(self):
        import itertools
        seen = []

        def check(state):
            seen.append(1)
            self.assertEqual(EpisodeState.from_dict(json.loads(json.dumps(state.to_dict()))).to_dict(), state.to_dict())

        start = lambda **c: fresh(initial_pool=list(range(1, 51)), user_id=7, split="validation", **c)
        for bits in itertools.product([True, False], repeat=9):
            drive(self, start(), ScriptedModel(decisions=bits), on_state=check)
        raw = {"assess_sufficiency": ["nonsense"]}
        drive(self, start(), ScriptedModel(raw=raw), on_state=check)
        drive(self, start(config=AgentConfig(max_invalid_actions_per_step=2)), ScriptedModel(raw=raw), on_state=check)
        drive(self, start(), ScriptedModel(decisions=[True, False]), retrieval=lambda step: [], on_state=check)

        def broken(step):
            raise Unavailable("down")

        drive(self, start(), ScriptedModel(decisions=[True, False]), retrieval=broken, on_state=check)
        drive(self, start(), ScriptedModel(raw={"explain_recommendations": ["nonsense"]}), on_state=check)
        replayed = ctl.start_episode("p", list(range(1, 51)), [90], clarifications=[{"question": "Q?", "answer": "A"}] * 4)
        check(replayed)
        drive(self, replayed, ScriptedModel(), on_state=check)
        self.assertGreater(len(seen), 3000)  # the walk really visited thousands of states


class StartEpisodeTest(unittest.TestCase):
    def test_new_episode_starts_at_assessment_with_empty_counters(self):
        state = fresh()
        self.assertEqual(state.phase, "assess")
        self.assertEqual((state.control_events, state.retrievals, state.asks), (0, 0, 0))
        self.assertEqual(state.pool, state.initial_pool)
        self.assertEqual(ctl.next_step(state).task, "assess_sufficiency")

    def test_initial_pool_must_exclude_watched_history(self):
        with self.assertRaises(ctl.TransitionError):
            fresh(history_movie_ids=[3])

    def test_bad_inputs_raise_instead_of_producing_a_state(self):
        for kwargs in (dict(initial_pool=[]), dict(initial_pool=[1, 1]), dict(initial_pool=[0, 1]),
                       dict(profile_text=None), dict(history_movie_ids=["a"])):
            with self.subTest(kwargs=kwargs), self.assertRaises(ctl.TransitionError):
                fresh(**kwargs)

    def test_replayed_clarifications_count_against_the_budget(self):
        # Service B is stateless: Service A resends earlier turns. They must not reset the budget.
        turns = [{"question": "Q1?", "answer": "A1"}, {"question": "Q2?", "answer": "A2"}]
        state = fresh(clarifications=turns)
        self.assertEqual(state.asks, 2)
        self.assertEqual(state.resumed_clarifications, 2)
        self.assertEqual(state.control_events, 4)  # each earlier question took an assessment + a question
        self.assertEqual(state.events, [])  # earlier events are not fabricated
        self.assertEqual({turn["answer_source"] for turn in state.clarifications}, {"client_supplied"})

    def test_replayed_turns_need_question_and_answer(self):
        with self.assertRaises(ctl.TransitionError):
            fresh(clarifications=[{"question": "Q?", "answer": " "}])


if __name__ == "__main__":
    unittest.main()
