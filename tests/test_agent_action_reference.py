"""The action reference (docs/member4_action_reference.md) must not drift from the code.

The page is for Member 6 and anyone else who has to produce or parse model output. If the shared
grammar, the controller's defaults or its reason codes change, these tests fail until the page is
updated, so the page cannot quietly go stale. Every example on the page is run through the real parser.
"""

from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from src.agent.actions import (INVALID_ACTION, REPEATED_QUESTION, UNBOUND_IDS, UNEXPECTED_ACTION,
                               AgentActionError, parse_model_output)
from src.agent.config import AgentConfig
from src.profiles.agent_text import ACTION_FIELDS, ACTIONS

ROOT = Path(__file__).resolve().parents[1]
DOC = (ROOT / "docs" / "member4_action_reference.md").read_text(encoding="utf-8")
COMMITTED = [json.loads(line) for line in
             (ROOT / "docs" / "p2_sample_examples.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
COMPLETIONS = {row["task"]: row["completion"] for row in COMMITTED}

# The one earlier question the repeated-question example refers to (it is Member 1's committed question).
EARLIER_QUESTION = "What is one of your favorite movies?"


def section(number: int) -> str:
    match = re.search(rf"^## {number}\. .*?\n(.*?)(?=^## |\Z)", DOC, re.M | re.S)
    assert match, f"section {number} not found"
    return match.group(1)


def fenced_blocks() -> list[tuple[list[str], str]]:
    """(words of the info string, text) for every fenced block. A longer fence may contain a shorter one."""
    pattern = re.compile(r"^(`{3,})([^\n`]*)\n(.*?)\n\1[ \t]*$", re.M | re.S)
    return [(match.group(2).split(), match.group(3)) for match in pattern.finditer(DOC)]


def ints(text: str) -> list[int]:
    return [int(part) for part in text.split(",")]


class ActionTableTests(unittest.TestCase):
    def test_every_action_has_exactly_one_field_so_the_table_shape_is_valid(self):
        for name in ACTIONS:
            self.assertEqual(len(ACTION_FIELDS[name]), 1, name)

    def test_table_lists_the_shared_actions_and_fields_in_order(self):
        rows = re.findall(r"^\| `(\w+)` \| `(\w+)` \|", section(2), re.M)
        self.assertEqual(rows, [(name, ACTION_FIELDS[name][0]) for name in ACTIONS])


class SettingsTableTests(unittest.TestCase):
    def test_documented_defaults_equal_the_real_defaults(self):
        rows = re.findall(r"^\| `(\w+)` \| `([^`]+)` \|", section(5), re.M)
        config = AgentConfig()
        for name, shown in rows:
            self.assertEqual(shown, str(getattr(config, name)).lower(), name)
        self.assertEqual({name for name, _ in rows}, {
            "max_clarifications", "max_retrievals", "max_pool_size", "max_control_events", "ranking_size",
            "max_invalid_actions_per_step", "exhaustion_policy", "reject_repeated_questions", "grounding_policy"})

    def test_the_event_arithmetic_stated_on_the_page_is_true(self):
        self.assertEqual(AgentConfig().worst_case_control_events, 8)
        self.assertIn("3 assessments, 2 questions,\n2 evaluations and 1 retrieval, which is 8 control events", DOC)

    def test_the_top_ten_rule_stated_on_the_page_is_the_configured_one(self):
        self.assertEqual(AgentConfig().explain_count, 10)
        self.assertIn("exactly the top 10", DOC)


class ReasonCodeTests(unittest.TestCase):
    def test_documented_codes_are_the_real_ones(self):
        rows = re.findall(r"^\| `(\w+)` \| ", section(6), re.M)
        self.assertEqual(set(rows), {INVALID_ACTION, UNEXPECTED_ACTION, UNBOUND_IDS, REPEATED_QUESTION,
                                     "model_error"})


class ValidExampleTests(unittest.TestCase):
    def valid_blocks(self):
        return [(words, text) for words, text in fenced_blocks() if "valid" in words]

    def test_every_action_has_an_example_and_the_real_parser_accepts_each(self):
        seen = set()
        for _, text in self.valid_blocks():
            action = json.loads(text)
            name = action["action"]
            seen.add(name)
            pool = action["movie_ids"] if name == "rank_candidates" else None
            shown = [item["movie_id"] for item in action["explanations"]] if name == "explain_recommendations" else None
            self.assertEqual(parse_model_output(text, name, pool_ids=pool, explain_ids=shown), action, text)
        self.assertEqual(seen, set(ACTIONS))

    def test_verbatim_examples_are_committed_completions_and_labelled_honestly(self):
        committed = {row["completion"] for row in COMMITTED}
        verbatim = [text for words, text in self.valid_blocks() if "verbatim" in words]
        shortened = [text for words, text in self.valid_blocks() if "shortened" in words]
        self.assertEqual(len(verbatim) + len(shortened), len(self.valid_blocks()), "every example is labelled")
        self.assertEqual(len(verbatim), 6)
        for text in verbatim:
            self.assertIn(text, committed)

        self.assertEqual(len(shortened), 2)
        for text in shortened:
            action = json.loads(text)
            real = json.loads(COMPLETIONS[action["action"]])
            field = ACTION_FIELDS[action["action"]][0]
            self.assertLess(len(action[field]), len(real[field]), "a shortened example must be shorter")
            self.assertEqual(action[field], real[field][: len(action[field])], "and a true prefix")


class RejectedExampleTests(unittest.TestCase):
    def rejected_blocks(self):
        for words, text in fenced_blocks():
            tag = next((word for word in words if word.startswith("rejected:")), None)
            if tag:
                yield tag.split(":", 1)[1], dict(word.split("=", 1) for word in words if "=" in word), text

    def test_each_rejected_output_gets_the_stated_code_from_the_real_parser(self):
        codes = set()
        for code, context, text in self.rejected_blocks():
            codes.add(code)
            with self.assertRaises(AgentActionError, msg=text) as caught:
                parse_model_output(
                    text, context["task"],
                    pool_ids=ints(context["pool"]) if "pool" in context else None,
                    explain_ids=ints(context["explain"]) if "explain" in context else None,
                    previous_questions=[EARLIER_QUESTION] if code == REPEATED_QUESTION else (),
                )
            self.assertEqual(caught.exception.code, code, text)
        self.assertEqual(codes, {INVALID_ACTION, UNEXPECTED_ACTION, UNBOUND_IDS, REPEATED_QUESTION})

    def test_the_earlier_question_in_the_repeat_example_is_a_committed_one(self):
        self.assertIn(EARLIER_QUESTION, " ".join(row["completion"] for row in COMMITTED))
        self.assertIn(EARLIER_QUESTION, DOC)

    def test_the_fence_and_prose_cases_are_really_shown(self):
        texts = [text for _, _, text in self.rejected_blocks()]
        self.assertTrue(any(text.startswith("```") for text in texts), "a fenced output is shown")
        self.assertTrue(any(not text.lstrip().startswith(("{", "`")) for text in texts), "a prose-prefixed one too")


if __name__ == "__main__":
    unittest.main()
