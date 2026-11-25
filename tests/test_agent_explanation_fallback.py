"""Tests for ``explanation_failure_policy`` (decision A4): what happens when the explain step has no usable
model output.

Default ``fail``: the episode ends "failed" (unchanged behaviour). ``template`` (Member 2 accepted this rule
for the first real validation run on 2026-11-24; the setting itself stays off unless the caller sets it): the
model's ranking is kept and the ten explanations are written from each movie's own displayed catalog fields,
openly recorded with Member 2's labels. The model here is a software test double, so these tests show how the
controller handles a failure, not how often a real model fails.
"""

import hashlib
import json
import re
import unittest
from dataclasses import replace

from src.agent import controller as ctl
from src.agent import example
from src.agent.config import AgentConfig, ConfigError
from src.agent.decide import decide
from src.agent.grounding import PASSED_SCREEN, Evidence, check_explanation, repair_text
from src.agent.manifest import (agent_policy, controller_counts, explanation_policy_version, fallback_trigger,
                                manifest_episode)
from src.agent.state import EXPLANATION_SOURCE_FALLBACK, EXPLANATION_SOURCE_MODEL, EpisodeState, StateError
from src.agent.trace import explanation_fallback_reason, explanation_source, telemetry_record, to_trace_record
from src.profiles.agent_text import serialize_action
from tests.test_agent_controller import AutoCatalog, ControllerCase, ScriptedModel, drive

UNBOUND = serialize_action({"action": "explain_recommendations", "explanations": [{"movie_id": 999, "text": "Great."}]})
CUT_OFF = '{"action": "explain_recommendations", "explanations": [{"movie_id": 50, "text": "It is'
FAILURES = {"unbound_ids": UNBOUND, "invalid_action": CUT_OFF, "model_error": None}


class ExplainCase(ControllerCase):
    def at_explain(self, **config):
        state = self.start(**config)
        model = ScriptedModel()
        for _ in range(3):  # assess, evaluate, rank
            step = ctl.next_step(state)
            state = ctl.apply_model_output(state, model.respond(step.task, state), catalog=AutoCatalog())
        self.assertEqual(state.phase, "explain")
        return state

    def fail_explain(self, state, kind, catalog=None):
        catalog = AutoCatalog() if catalog is None else catalog
        if FAILURES[kind] is None:
            return ctl.apply_model_error(state, "TimeoutError: explain call timed out", catalog=catalog)
        return ctl.apply_model_output(state, FAILURES[kind], catalog=catalog)


class DefaultIsOffTest(ExplainCase):
    def test_the_default_is_fail(self):
        self.assertEqual(AgentConfig().explanation_failure_policy, "fail")

    def test_an_unusable_explanation_still_ends_the_episode_as_failed(self):
        for kind in FAILURES:
            with self.subTest(kind):
                state = self.fail_explain(self.at_explain(), kind)
                self.assertEqual((state.phase, state.termination_reason), ("done", "failed"))
                self.assertEqual(state.reason, f"explanation_invalid:{kind}")
                self.assertNotIn("explain_recommendations", [e["action"] for e in state.events])
                self.assertIsNone(explanation_source(state))
                self.assertIsNone(telemetry_record(state)["explanation_source"])
                with self.assertRaises(Exception):
                    to_trace_record(state)

    def test_a_normal_episode_reports_model_explanations(self):
        state = drive(self, self.start(), ScriptedModel())
        self.assertEqual(telemetry_record(state)["explanation_source"], "model")


class TemplateFallbackTest(ExplainCase):
    def test_the_ranking_is_kept_and_ten_bound_texts_come_from_the_catalog(self):
        for kind in FAILURES:
            with self.subTest(kind):
                before = self.at_explain(explanation_failure_policy="template")
                state = self.fail_explain(before, kind)
                self.assertEqual((state.phase, state.termination_reason), ("done", "recommend"))
                self.assertEqual(state.ranked, before.ranked)
                event = state.events[-1]
                self.assertEqual(event["action"], "explain_recommendations")
                self.assertEqual([item["movie_id"] for item in event["explanations"]], before.ranked[:10])
                for item in event["explanations"]:
                    self.assertEqual(item["text"], repair_text(AutoCatalog()[item["movie_id"]]))
                self.assertEqual([e["action"] for e in state.events].count("explain_recommendations"), 1)
                self.check_trace(state)  # still legal under Member 1's shared validator

    def test_the_substitution_is_recorded_not_quiet(self):
        for kind, code in (("unbound_ids", "unbound_ids"), ("invalid_action", "invalid_action"), ("model_error", "model_error")):
            with self.subTest(kind):
                state = self.fail_explain(self.at_explain(explanation_failure_policy="template"), kind)
                attempt, substitute = state.decisions[-2], state.decisions[-1]
                self.assertEqual((attempt["source"], attempt["valid"], attempt["error"]["code"]), ("model", False, code))
                self.assertEqual((substitute["source"], substitute["task"], substitute["valid"], substitute["fallback"]),
                                 ("fallback", "explain_recommendations", True, "catalog_metadata_fallback"))
                self.assertEqual(state.failures[-1]["code"], code)  # the unusable attempt stays on record
                self.assertEqual(explanation_source(state), "catalog_metadata_fallback")
                telemetry = telemetry_record(state)
                self.assertEqual(telemetry["explanation_source"], "catalog_metadata_fallback")
                self.assertEqual(telemetry["config"]["explanation_failure_policy"], "template")
                self.assertEqual(json.loads(json.dumps(telemetry)), telemetry)
                self.assertEqual(EpisodeState.from_dict(json.loads(json.dumps(state.to_dict()))).to_dict(), state.to_dict())

    def test_the_template_text_passes_the_grounding_screen(self):
        state = self.fail_explain(self.at_explain(explanation_failure_policy="template"), "invalid_action")
        shown = {i: AutoCatalog()[i] for i in state.ranked[:10]}
        evidence = Evidence(state.profile_text, (), shown)
        for item in state.events[-1]["explanations"]:
            self.assertEqual(check_explanation(item["movie_id"], item["text"], evidence)["status"], PASSED_SCREEN)

    def test_a_valid_model_explanation_is_untouched_and_reported_as_the_models(self):
        state = drive(self, self.start(explanation_failure_policy="template"), ScriptedModel())
        self.assertEqual(telemetry_record(state)["explanation_source"], "model")
        self.assertTrue(all(d["source"] == "model" for d in state.decisions))
        self.assertEqual(state.events[-1]["explanations"][0]["text"], "It is listed under Drama and Comedy.")

    def test_a_retry_comes_before_the_template(self):
        state = self.at_explain(explanation_failure_policy="template", max_invalid_actions_per_step=1)
        state = self.fail_explain(state, "invalid_action")
        self.assertEqual((state.phase, ctl.next_step(state).attempt), ("explain", 2))
        self.assertNotIn("fallback", [d["source"] for d in state.decisions])
        state = self.fail_explain(state, "invalid_action")
        self.assertEqual(state.phase, "done")
        self.assertEqual([d["source"] for d in state.decisions[-3:]], ["model", "model", "fallback"])

    def test_a_retry_that_works_needs_no_template(self):
        state = self.at_explain(explanation_failure_policy="template", max_invalid_actions_per_step=1)
        state = self.fail_explain(state, "invalid_action")
        good = ScriptedModel().respond("explain_recommendations", state)
        state = ctl.apply_model_output(state, good, catalog=AutoCatalog())
        self.assertEqual(explanation_source(state), "model")

    def test_rejected_grounding_falls_back_to_the_template(self):
        state = self.at_explain(explanation_failure_policy="template", grounding_policy="reject")
        shown = state.ranked[:10]
        texts = ["It is listed under Western."] + ["It is listed under Drama and Comedy."] * 9  # a genre shown nowhere
        raw = serialize_action({"action": "explain_recommendations",
                                "explanations": [{"movie_id": i, "text": t} for i, t in zip(shown, texts)]})
        state = ctl.apply_model_output(state, raw, catalog=AutoCatalog())
        self.assertEqual((state.phase, state.termination_reason), ("done", "recommend"))
        self.assertEqual(state.grounding[-1]["outcome"], "rejected")
        self.assertEqual(explanation_source(state), "catalog_metadata_fallback")
        self.assertEqual(state.events[-1]["explanations"][0]["text"], repair_text(AutoCatalog()[shown[0]]))

    def test_a_contract_fallback_ranking_keeps_its_label_and_still_gets_explanations(self):
        state = self.start(explanation_failure_policy="template")
        bad_assess = serialize_action({"action": "retrieve_candidates", "query": "anything"})
        model = ScriptedModel(raw={"assess_sufficiency": [bad_assess], "explain_recommendations": [CUT_OFF]})
        state = drive(self, state, model)
        self.assertEqual((state.termination_reason, state.fallback_source), ("fallback", "frozen_b3_order"))
        self.assertEqual(state.ranked, state.initial_pool)
        self.assertEqual(explanation_source(state), "catalog_metadata_fallback")
        record, _ = self.check_trace(state)
        self.assertEqual(record["fallback_reason"], state.reason)
        entry = manifest_episode(state, informative_answer_count=0, noninformative_answer_count=0, timed_out=False,
                                 latency_ms={"model": 1.0, "retrieval": 0.0, "end_to_end": 1.0}, cost_usd=0.0)
        self.assertEqual(entry["fallback_trigger"], fallback_trigger(state.reason))
        self.assertEqual(entry["invalid_action_count"], 2)  # the bad assessment and the cut-off explanation


class TemplateUnavailableTest(ExplainCase):
    def test_without_a_catalog_the_episode_fails_as_under_the_default(self):
        state = ctl.apply_model_error(self.at_explain(explanation_failure_policy="template"), "TimeoutError: slow")
        self.assertEqual((state.phase, state.termination_reason), ("done", "failed"))
        self.assertEqual(state.reason, "explanation_invalid:model_error")
        self.assertEqual([f["code"] for f in state.failures], ["model_error", "template_unavailable"])
        self.assertNotIn("fallback", [d["source"] for d in state.decisions])

    def test_a_catalog_missing_a_shown_movie_fails_and_says_which(self):
        before = self.at_explain(explanation_failure_policy="template")
        partial = {i: AutoCatalog()[i] for i in before.ranked[:10] if i != before.ranked[3]}
        state = self.fail_explain(before, "invalid_action", catalog=partial)
        self.assertEqual((state.phase, state.termination_reason), ("done", "failed"))
        self.assertEqual(state.failures[-1]["code"], "template_unavailable")
        self.assertIn(str(before.ranked[3]), state.failures[-1]["detail"])


class ConfigAndPolicyTest(unittest.TestCase):
    def test_only_the_two_documented_values_are_accepted(self):
        for value in ("fail", "template"):
            self.assertEqual(AgentConfig(explanation_failure_policy=value).explanation_failure_policy, value)
        for bad in ("retry", "", None, True):
            with self.assertRaises(ConfigError, msg=repr(bad)):
                AgentConfig(explanation_failure_policy=bad)

    def test_a_config_stored_before_the_field_existed_loads_as_fail(self):
        old = AgentConfig().to_dict()
        del old["explanation_failure_policy"]
        self.assertEqual(AgentConfig.from_dict(old), AgentConfig())

    def test_every_other_field_is_still_required_and_unknown_ones_are_still_refused(self):
        full = AgentConfig().to_dict()
        for name in full:
            if name == "explanation_failure_policy":
                continue
            missing = {k: v for k, v in full.items() if k != name}
            with self.assertRaises(ConfigError, msg=name):
                AgentConfig.from_dict(missing)
        with self.assertRaises(ConfigError):
            AgentConfig.from_dict({**full, "surprise": 1})

    def test_a_state_stored_before_the_field_existed_still_loads(self):
        state = ctl.start_episode("Likes slow comedies.", list(range(1, 51)), [900], user_id=7, split="validation")
        data = json.loads(json.dumps(state.to_dict()))
        del data["config"]["explanation_failure_policy"]
        self.assertEqual(EpisodeState.from_dict(data).config.explanation_failure_policy, "fail")
        del data["config"]["grounding_policy"]
        with self.assertRaises(StateError):
            EpisodeState.from_dict(data)

    def test_the_manifest_names_the_template_rule_only_when_it_is_on(self):
        self.assertEqual(explanation_policy_version(AgentConfig()), "p2-grounding-screen-v1/flag")  # unchanged
        on = replace(AgentConfig(), explanation_failure_policy="template")
        self.assertEqual(explanation_policy_version(on), "p2-grounding-screen-v1/flag/catalog_metadata_fallback")
        self.assertEqual(agent_policy(on, timeout_ms=1000)["explanation_policy_version"],
                         "p2-grounding-screen-v1/flag/catalog_metadata_fallback")


class BrokenExplainModel(example.DemoModel):
    """The demo model, except that the explain call times out or returns cut-off JSON."""

    def __init__(self, mode):
        self.mode = mode

    def generate(self, prompt):
        if re.search(r'Output format: \{"action": "(\w+)"', prompt).group(1) == "explain_recommendations":
            if self.mode == "raise":
                raise TimeoutError("explain call timed out")
            return CUT_OFF
        return super().generate(prompt)


class ThroughDecideTest(unittest.TestCase):
    REQUEST = {"profile_text": example.PROFILE, "history_movie_ids": example.HISTORY,
               "clarifications": [{"question": "Which movie genres do you enjoy most?", "answer": "Thrillers."}]}

    def call(self, mode, **config):
        return decide(self.REQUEST, model=BrokenExplainModel(mode), retriever=example.DemoRetriever(),
                      catalog=example.CATALOG, config=AgentConfig(**config))

    def test_default_returns_the_explicit_error(self):
        for mode in ("raise", "cut_off"):
            with self.subTest(mode):
                response = self.call(mode).response
                self.assertEqual((response["action"], response["code"]), ("error", "failed"))

    def test_template_returns_a_normal_recommendation_with_catalog_text(self):
        for mode in ("raise", "cut_off"):
            with self.subTest(mode):
                decision = self.call(mode, explanation_failure_policy="template")
                response = decision.response
                self.assertEqual(response["action"], "recommend")
                self.assertEqual(len(response["items"]), 10)
                for item in response["items"]:
                    self.assertEqual(item["explanation"], repair_text(example.CATALOG[item["movie_id"]]))
                self.assertEqual(telemetry_record(decision.state)["explanation_source"], "catalog_metadata_fallback")


EXPECTED_REASON = {"unbound_ids": "invalid_action:explain_recommendations:unbound_ids",
                   "invalid_action": "invalid_action:explain_recommendations:invalid_action",
                   "model_error": "model_error:explain_recommendations"}


class MemberTwoContractTest(ExplainCase):
    """Member 2's rule for the first real validation run (docs/phase2_interface_contract.md and
    docs/phase2_checkpoint_selection_handoff.md on origin/member2-phase2, 2026-11-24): label the event,
    give a reason, keep the failed output's evidence, count the call correctly, leave flagged text alone."""

    def test_the_fallback_event_carries_source_and_reason_in_the_trace(self):
        for kind, reason in EXPECTED_REASON.items():
            with self.subTest(kind):
                state = self.fail_explain(self.at_explain(explanation_failure_policy="template"), kind)
                record, _ = self.check_trace(state)
                explain = [event for event in record["events"] if event["action"] == "explain_recommendations"]
                self.assertEqual(len(explain), 1)
                self.assertEqual(explain[0]["source"], EXPLANATION_SOURCE_FALLBACK)
                self.assertEqual(explain[0]["explanation_fallback_reason"], reason)
                self.assertEqual(explanation_fallback_reason(state), reason)
                self.assertEqual(telemetry_record(state)["explanation_fallback_reason"], reason)
                self.assertEqual(state.decisions[-1]["reason"], reason)
                self.assertEqual(record["termination_reason"], "recommend")  # provenance is not a termination

    def test_a_model_written_explanation_is_labelled_model_with_no_reason(self):
        for policy in ("fail", "template"):
            with self.subTest(policy):
                state = drive(self, self.start(explanation_failure_policy=policy), ScriptedModel())
                record, _ = self.check_trace(state)
                explain = next(event for event in record["events"] if event["action"] == "explain_recommendations")
                self.assertEqual(explain["source"], EXPLANATION_SOURCE_MODEL)
                self.assertNotIn("explanation_fallback_reason", explain)
                self.assertIsNone(telemetry_record(state)["explanation_fallback_reason"])

    def test_the_failed_output_is_preserved_as_a_digest_and_an_error(self):
        state = self.fail_explain(self.at_explain(explanation_failure_policy="template"), "invalid_action")
        attempt = next(d for d in state.decisions if d["task"] == "explain_recommendations" and not d["valid"])
        self.assertEqual(attempt["raw_sha256"], hashlib.sha256(CUT_OFF.encode("utf-8")).hexdigest())
        self.assertEqual(attempt["error"]["code"], "invalid_action")
        self.assertTrue(attempt["error"]["message"])
        timed_out = self.fail_explain(self.at_explain(explanation_failure_policy="template"), "model_error")
        attempt = next(d for d in timed_out.decisions if d["task"] == "explain_recommendations" and not d["valid"])
        self.assertIsNone(attempt["raw_sha256"])  # nothing came back to hash
        self.assertEqual(attempt["error"], {"code": "model_error", "message": "TimeoutError: explain call timed out"})

    def test_a_rejected_output_counts_as_invalid_but_a_timeout_counts_only_as_a_model_call(self):
        def counts(kind):
            return controller_counts(self.fail_explain(self.at_explain(explanation_failure_policy="template"), kind))

        after_ranking = controller_counts(self.at_explain(explanation_failure_policy="template"))
        self.assertEqual(after_ranking["invalid_action_count"], 0)
        for kind in ("unbound_ids", "invalid_action"):
            with self.subTest(kind):
                got = counts(kind)
                self.assertEqual(got["model_call_count"], after_ranking["model_call_count"] + 1)  # the attempt counts
                self.assertEqual(got["invalid_action_count"], 1)
        got = counts("model_error")
        self.assertEqual(got["model_call_count"], after_ranking["model_call_count"] + 1)
        self.assertEqual(got["invalid_action_count"], 0)

    def test_the_substitute_text_is_not_a_model_call(self):
        state = self.fail_explain(self.at_explain(explanation_failure_policy="template"), "invalid_action")
        self.assertEqual(sum(1 for d in state.decisions if d["source"] == "model"), controller_counts(state)["model_call_count"])
        self.assertEqual(state.decisions[-1]["source"], "fallback")

    def test_flagged_model_text_is_kept_unmodified_under_the_default_screen(self):
        state = self.at_explain(explanation_failure_policy="template")
        shown = state.ranked[:10]
        texts = ["It is listed under Western."] + ["It is listed under Drama and Comedy."] * 9  # a genre shown nowhere
        raw = serialize_action({"action": "explain_recommendations",
                                "explanations": [{"movie_id": i, "text": t} for i, t in zip(shown, texts)]})
        state = ctl.apply_model_output(state, raw, catalog=AutoCatalog())
        self.assertEqual(state.grounding[-1]["outcome"], "kept")
        self.assertEqual(state.grounding[-1]["results"][0]["status"], "unsupported")  # flagged for the audit
        self.assertEqual([item["text"] for item in state.events[-1]["explanations"]], texts)  # but not changed
        self.assertEqual(state.events[-1]["source"], EXPLANATION_SOURCE_MODEL)
        self.assertEqual(explanation_source(state), EXPLANATION_SOURCE_MODEL)


if __name__ == "__main__":
    unittest.main()
