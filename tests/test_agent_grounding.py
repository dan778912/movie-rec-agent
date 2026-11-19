"""Grounding screen tests.

The screen is deterministic and checks only what is mechanically checkable against the evidence the
model saw. These tests pin down what it catches and, as importantly, what it does not claim: a
passing text is "passed_screen", never "supported". The human audit (Member 2's rubric) is separate.
"""

import hashlib
import json
import re
import time
import unittest
from pathlib import Path

from src.agent import controller as ctl
from src.agent.config import AgentConfig
from src.agent.grounding import (
    INSUFFICIENT_EVIDENCE, NEEDS_REVIEW, NO_CHECKABLE_CLAIM, PASSED_SCREEN, UNSUPPORTED,
    Evidence, check_explanation, check_explanations, repair_text,
)
from src.agent.trace import telemetry_record, to_trace_record
from src.profiles.agent_text import serialize_action

EXAMPLES = Path(__file__).resolve().parent.parent / "docs" / "p2_sample_examples.jsonl"
LINE = re.compile(r"^\[(\d+)\] (.*?) \| genres: (.*?) \| tags: (.*)$")


def load_example():
    """Evidence and explanations from the last step of Member 1's committed example episode."""
    rows = sorted((json.loads(line) for line in EXAMPLES.read_text(encoding="utf-8").splitlines()),
                  key=lambda row: row["step"])
    row = rows[9]
    shown = {}
    for line in row["prompt"].splitlines():
        match = LINE.match(line)
        if match:
            shown[int(match[1])] = {
                "movie_id": int(match[1]), "title": match[2],
                "genres": [] if match[3] == "unspecified" else match[3].split(", "),
                "tags": [] if match[4] == "none" else match[4].split(", ")}
    evidence = Evidence(row["state"]["profile_text"], tuple(c["answer"] for c in row["state"]["clarifications"]), shown)
    return evidence, row["target"]["explanations"]


EVIDENCE, TEMPLATE_EXPLANATIONS = load_example()
SUBJECT = 1343  # Cape Fear (1991): Thriller only, tags include "lawyers"
OTHER = 10


def screen(text, movie_id=SUBJECT, evidence=EVIDENCE):
    return check_explanation(movie_id, text, evidence)


def codes(result):
    return [problem["code"] for problem in result["problems"]]


class PositiveControlTest(unittest.TestCase):
    def test_every_template_explanation_member_one_trained_on_passes(self):
        self.assertEqual(len(TEMPLATE_EXPLANATIONS), 10)
        for item, result in zip(TEMPLATE_EXPLANATIONS, check_explanations(TEMPLATE_EXPLANATIONS, EVIDENCE)):
            with self.subTest(movie=item["movie_id"], text=item["text"]):
                self.assertEqual((result["status"], result["problems"]), (PASSED_SCREEN, []))
                self.assertGreater(sum(result["verified"].values()), 0)

    def test_plain_prose_about_the_movies_own_fields_passes(self):
        doc = EVIDENCE.shown[SUBJECT]
        text = f"{doc['title']} is listed under {' and '.join(doc['genres'])}."
        self.assertEqual(screen(text)["status"], PASSED_SCREEN)

    def test_answers_are_usable_evidence(self):
        result = screen("Like Silence of the Lambs, The (1991), which you rated 5/5, it is listed under Thriller.")
        self.assertEqual((result["status"], result["problems"]), (PASSED_SCREEN, []))  # cited in a clarification answer


class FabricationDetectionTest(unittest.TestCase):
    def setUp(self):
        self.subject = EVIDENCE.shown[SUBJECT]
        for genre in ("Comedy", "Western", "Drama"):  # preconditions the cases below rely on
            self.assertNotIn(genre, self.subject["genres"])
        self.assertNotIn("gunfight", [tag.lower() for tag in self.subject["tags"]])
        self.assertIn("lawyers", [tag.lower() for tag in self.subject["tags"]])

    def expect(self, text, status, code, **kwargs):
        result = screen(text, **kwargs)
        self.assertEqual(result["status"], status, result)
        self.assertIn(code, codes(result), result)
        return result

    def test_wrong_rating_for_a_profile_movie(self):
        self.expect("Like Firm, The (1993), which you rated 5/5, it is listed under Thriller.", UNSUPPORTED, "rating_mismatch")

    def test_rating_attached_to_the_recommendation_itself(self):
        text = f"{self.subject['title']} is rated 4.5/5."
        self.expect(text, UNSUPPORTED, "rating_without_source")

    def test_title_that_was_never_shown(self):
        self.expect("Like Pulp Fiction (1994), it is listed under Thriller.", UNSUPPORTED, "title_not_displayed")

    def test_year_of_a_real_title_does_not_launder_a_fake_title(self):
        # "Beverly Hills Cop IV (1994)" ends in a real year but is not the profile's movie.
        self.expect("Like Beverly Hills Cop IV (1994), it is listed under Thriller.", UNSUPPORTED, "title_not_displayed")

    def test_another_recommendation_presented_as_the_users_history(self):
        other = EVIDENCE.shown[OTHER]["title"]
        self.expect(f"Like {other}, which you rated 4/5, it is listed under Thriller.", UNSUPPORTED, "cites_other_candidate")

    def test_genre_shown_nowhere(self):
        self.expect("It is listed under Thriller and Western.", UNSUPPORTED, "genre_not_displayed")

    def test_genre_shown_only_for_the_users_history_needs_review_not_rejection(self):
        self.expect("It is listed under Thriller and Comedy.", NEEDS_REVIEW, "genre_not_listed_for_movie")

    def test_tag_shown_nowhere(self):
        self.expect('It is tagged "space opera".', UNSUPPORTED, "tag_not_displayed")

    def test_tag_shown_only_for_the_users_history(self):
        self.expect('It is tagged "gunfight".', NEEDS_REVIEW, "tag_not_listed_for_movie")

    def test_decade_nobody_mentioned(self):
        self.expect("It is from the 1980s, the decade you said you enjoy most.", UNSUPPORTED, "decade_not_displayed")

    def test_reference_to_a_clarification_that_never_happened(self):
        no_answers = Evidence(EVIDENCE.profile_text, (), EVIDENCE.shown)
        self.expect("You said you enjoy Thriller, and it is listed under Thriller.", UNSUPPORTED,
                    "answer_reference_without_answer", evidence=no_answers)
        # The same sentence is fine when an answer exists.
        self.assertNotIn("answer_reference_without_answer", codes(screen("You said you enjoy Thriller.")))

    def test_claims_about_fields_the_catalog_does_not_have(self):
        self.expect("Directed by Steven Spielberg and starring a big cast.", INSUFFICIENT_EVIDENCE, "unavailable_field")

    def test_a_word_the_evidence_itself_uses_is_not_flagged_as_unavailable(self):
        evidence = Evidence("No highly rated movies in the available history.", (),
                            {1: {"movie_id": 1, "title": "Heist (2001)", "genres": ["Crime"], "tags": ["plot twist"]}})
        self.assertNotIn("unavailable_field", codes(check_explanation(1, "It has a plot twist.", evidence)))

    def test_generic_praise_has_nothing_to_check(self):
        result = screen("A great pick that you will love!")
        self.assertEqual((result["status"], result["problems"]), (NO_CHECKABLE_CLAIM, []))

    def test_free_text_answer_title_is_recognised_but_other_titles_are_not(self):
        evidence = Evidence("No highly rated movies in the available history.", ("I love Heat (1995) and De Niro.",),
                            {1: {"movie_id": 1, "title": "Casino Royale (2006)", "genres": ["Action"], "tags": []}})
        self.assertEqual(check_explanation(1, "Like Heat (1995), it is listed under Action.", evidence)["status"], PASSED_SCREEN)
        self.assertEqual(check_explanation(1, "Like Casino (1995), it is listed under Action.", evidence)["status"], UNSUPPORTED)

    def test_known_limit_a_partial_title_in_a_free_text_answer_is_accepted(self):
        # Pinned so the leniency is visible: only a trailing run of words plus the year must be found.
        evidence = Evidence("No highly rated movies in the available history.", ("I loved Miracle on 34th Street (1994).",),
                            {1: {"movie_id": 1, "title": "Heist (2001)", "genres": ["Crime"], "tags": []}})
        self.assertEqual(check_explanation(1, "Like Elm Street (1994), it is listed under Crime.", evidence)["status"], PASSED_SCREEN)

    def test_overlong_text_is_reported_not_silently_half_checked(self):
        result = screen("Thriller. " * 400)
        self.assertIn("text_too_long", codes(result))

    def test_the_strongest_finding_wins(self):
        result = screen('It is listed under Thriller and Comedy and Western, tagged "space opera".')
        self.assertEqual(result["status"], UNSUPPORTED)
        self.assertEqual({p["status"] for p in result["problems"]}, {UNSUPPORTED, NEEDS_REVIEW})

    def test_problem_records_are_json_and_bounded(self):
        result = screen("Like " + "x" * 5000 + " (1994) it is Western.")
        self.assertEqual(json.loads(json.dumps(result)), result)
        self.assertTrue(all(len(p["detail"]) <= 160 for p in result["problems"]))


class RobustnessTest(unittest.TestCase):
    def test_pathological_inputs_finish_quickly_without_crashing(self):
        hostile_profile = "Highly rated movies: " + "a (1/5; genres: " * 1500 + "x"
        cases = [
            Evidence(hostile_profile, ("(" * 2000,), EVIDENCE.shown),
            Evidence("", (), EVIDENCE.shown),
            Evidence("No highly rated movies in the available history.", ("",), EVIDENCE.shown),
        ]
        texts = ["(1994) " * 300, "4/5 " * 500, '"' * 1999, "Like " * 400, "‮\u0000" * 500, ""]
        started = time.monotonic()
        for evidence in cases:
            for text in texts:
                result = check_explanation(SUBJECT, text, evidence)
                self.assertIn("status", result)
        self.assertLess(time.monotonic() - started, 5.0)


class RepairTextTest(unittest.TestCase):
    def test_repaired_text_passes_the_screen_for_every_displayed_movie(self):
        for movie_id, doc in EVIDENCE.shown.items():
            with self.subTest(movie=movie_id):
                result = check_explanation(movie_id, repair_text(doc), EVIDENCE)
                self.assertEqual((result["status"], result["problems"]), (PASSED_SCREEN, []))

    def test_repair_uses_only_the_movies_own_fields(self):
        doc = {"movie_id": 1, "title": "Heist (2001)", "genres": ["Crime", "Drama", "Thriller"], "tags": ['say "hi"', "twist", "slow", "x"]}
        self.assertEqual(repair_text(doc), 'It is listed under Crime, Drama and Thriller and is tagged "twist" and "slow".')
        self.assertEqual(repair_text({"title": "T (2000)", "genres": [], "tags": []}), "It has no listed genre or tag.")


# --------------------------------------------------------------------------- integration with the controller
class AutoCatalog:
    def __init__(self, docs):
        self.docs = docs

    def __getitem__(self, movie_id):
        return self.docs[movie_id]

    def __contains__(self, movie_id):
        return movie_id in self.docs


POOL = list(range(1, 13))
DOCS = {i: {"movie_id": i, "title": f"Movie {i} (2001)", "genres": ["Drama", "Comedy"], "tags": ["quirky"]} for i in POOL}
GOOD = "It is listed under Drama and Comedy."
BAD = "It is listed under Western."  # a genre shown nowhere


def episode_at_explanation(**config):
    state = ctl.start_episode("Highly rated movies: Heat (1995) (4/5; genres: Crime; tags: none).", POOL, [],
                              config=AgentConfig(ranking_size=12, **config), user_id=1, split="validation")
    catalog = AutoCatalog(DOCS)
    for task, action in (("assess_sufficiency", {"sufficient": True}), ("evaluate_candidates", {"pool_sufficient": True}),
                         ("rank_candidates", {"movie_ids": sorted(POOL, reverse=True)})):
        state = ctl.apply_model_output(state, serialize_action({"action": task, **action}), catalog=catalog)
    assert state.phase == "explain"
    return state, catalog


def explanation_output(state, texts):
    shown = state.ranked[: state.config.explain_count]
    return serialize_action({"action": "explain_recommendations", "explanations": [
        {"movie_id": movie_id, "text": text} for movie_id, text in zip(shown, texts)]})


class GroundingPolicyTest(unittest.TestCase):
    def finish(self, policy, texts, *, retries=0):
        state, catalog = episode_at_explanation(grounding_policy=policy, max_invalid_actions_per_step=retries)
        return ctl.apply_model_output(state, explanation_output(state, texts), catalog=catalog), catalog

    def mixed(self):
        return [BAD] + [GOOD] * 9

    def test_flag_keeps_the_text_and_records_the_finding(self):
        state, _ = self.finish("flag", self.mixed())
        self.assertEqual(state.termination_reason, "recommend")
        self.assertEqual(state.events[-1]["explanations"][0]["text"], BAD)
        record = state.grounding[-1]
        self.assertEqual((record["policy"], record["outcome"], record["attempt"]), ("flag", "kept", 1))
        self.assertEqual([r["status"] for r in record["results"]], [UNSUPPORTED] + [PASSED_SCREEN] * 9)
        self.assertIn("genre_not_displayed", [p["code"] for p in record["results"][0]["problems"]])

    def test_repair_replaces_only_the_unsupported_explanation_and_the_trace_matches(self):
        state, catalog = self.finish("repair", self.mixed())
        self.assertEqual(state.termination_reason, "recommend")
        texts = [item["text"] for item in state.events[-1]["explanations"]]
        self.assertEqual(texts[1:], [GOOD] * 9)
        self.assertEqual(texts[0], repair_text(DOCS[state.ranked[0]]))
        first = state.grounding[-1]["results"][0]
        self.assertTrue(first["repaired"])
        self.assertEqual(first["original_sha256"], hashlib.sha256(BAD.encode()).hexdigest())
        self.assertEqual(state.grounding[-1]["outcome"], "repaired")
        # What the user sees (the trace text) is the repaired text, and it passes the screen.
        self.assertEqual(to_trace_record(state)["events"][-1]["explanations"][0]["text"], texts[0])
        evidence = Evidence(state.profile_text, (), {i: DOCS[i] for i in state.ranked[:10]})
        self.assertEqual(check_explanation(state.ranked[0], texts[0], evidence)["status"], PASSED_SCREEN)
        self.assertEqual(state.decisions[-1]["parsed"]["explanations"][0]["text"], BAD)  # the model's own output is kept

    def test_reject_ends_the_episode_as_failed_when_no_retry_remains(self):
        state, _ = self.finish("reject", self.mixed())
        self.assertEqual((state.phase, state.termination_reason), ("done", "failed"))
        self.assertEqual(state.reason, "explanation_invalid:ungrounded_explanation")
        self.assertEqual(state.grounding[-1]["outcome"], "rejected")
        self.assertFalse(state.decisions[-1]["valid"])

    def test_reject_with_a_retry_can_recover(self):
        state, catalog = episode_at_explanation(grounding_policy="reject", max_invalid_actions_per_step=1)
        state = ctl.apply_model_output(state, explanation_output(state, self.mixed()), catalog=catalog)
        self.assertEqual((state.phase, ctl.next_step(state).attempt), ("explain", 2))
        state = ctl.apply_model_output(state, explanation_output(state, [GOOD] * 10), catalog=catalog)
        self.assertEqual(state.termination_reason, "recommend")
        self.assertEqual([(g["attempt"], g["outcome"]) for g in state.grounding], [(1, "rejected"), (2, "kept")])

    def test_only_unsupported_findings_trigger_repair_or_rejection(self):
        for policy in ("repair", "reject"):
            with self.subTest(policy):
                review = "It is listed under Crime."  # shown for the user's history, not for this movie
                generic = "A great pick that you will love!"
                unavailable = "Directed by someone famous."
                state, _ = self.finish(policy, [review, generic, unavailable] + [GOOD] * 7)
                self.assertEqual(state.termination_reason, "recommend")
                texts = [item["text"] for item in state.events[-1]["explanations"]]
                self.assertEqual(texts[:3], [review, generic, unavailable])  # never altered
                statuses = [r["status"] for r in state.grounding[-1]["results"]][:3]
                self.assertEqual(statuses, [NEEDS_REVIEW, NO_CHECKABLE_CLAIM, INSUFFICIENT_EVIDENCE])

    def test_off_skips_the_screen_and_needs_no_catalog(self):
        state, _ = episode_at_explanation(grounding_policy="off")
        state = ctl.apply_model_output(state, explanation_output(state, [BAD] * 10))
        self.assertEqual(state.termination_reason, "recommend")
        self.assertEqual((state.grounding[-1]["outcome"], state.grounding[-1]["results"]), ("unchecked", []))

    def test_the_catalog_is_required_otherwise_and_the_state_is_left_alone(self):
        state, _ = episode_at_explanation()
        before = state.to_dict()
        with self.assertRaises(ctl.TransitionError):
            ctl.apply_model_output(state, explanation_output(state, [GOOD] * 10))
        self.assertEqual(state.to_dict(), before)

    def test_a_catalog_gap_is_reported_not_skipped(self):
        state, catalog = episode_at_explanation()
        gappy = AutoCatalog({i: doc for i, doc in DOCS.items() if i != state.ranked[0]})
        with self.assertRaises(ctl.MissingCandidateError):
            ctl.apply_model_output(state, explanation_output(state, [GOOD] * 10), catalog=gappy)

    def test_fallback_episodes_are_screened_too(self):
        state = ctl.start_episode("p", POOL, [], config=AgentConfig(ranking_size=12), user_id=1, split="validation")
        state = ctl.apply_model_output(state, "garbage", catalog=AutoCatalog(DOCS))  # falls back to the B3 order
        self.assertEqual((state.termination_reason, state.phase), ("fallback", "explain"))
        state = ctl.apply_model_output(state, explanation_output(state, self.mixed()), catalog=AutoCatalog(DOCS))
        self.assertEqual(state.grounding[-1]["results"][0]["status"], UNSUPPORTED)
        self.assertEqual(state.termination_reason, "fallback")

    def test_results_reach_telemetry_and_survive_json(self):
        state, _ = self.finish("flag", self.mixed())
        telemetry = telemetry_record(state)
        self.assertEqual(telemetry["grounding"], state.grounding)
        self.assertEqual(json.loads(json.dumps(telemetry)), telemetry)
        restored = ctl.EpisodeState.from_dict(json.loads(json.dumps(state.to_dict())))
        self.assertEqual(restored.grounding, state.grounding)

    def test_the_screen_never_labels_anything_supported(self):
        state, _ = self.finish("flag", [GOOD] * 10)
        statuses = {r["status"] for g in state.grounding for r in g["results"]}
        self.assertEqual(statuses, {PASSED_SCREEN})
        self.assertNotIn("supported", statuses)


if __name__ == "__main__":
    unittest.main()
