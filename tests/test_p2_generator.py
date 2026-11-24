"""Tests for strict P2 generation and action validation."""

from __future__ import annotations

import unittest

from src.reranking.p2_generator import (
    P2ActionGenerator,
    P2GenerationError,
    validate_generated_action,
)


class FakeTokenizer:
    """Tokenizer replacement used without model dependencies."""

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        truncation: bool,
    ) -> dict[str, list[int]]:
        if truncation:
            raise AssertionError(
                "P2 input counting must not truncate"
            )

        count = len(text.split())

        if add_special_tokens:
            count += 1

        return {"input_ids": list(range(count))}


class P2GeneratorTests(unittest.TestCase):
    def test_valid_action_is_accepted(self) -> None:
        action = validate_generated_action(
            '{"action": "assess_sufficiency", '
            '"sufficient": true}',
            expected_action="assess_sufficiency",
        )

        self.assertEqual(
            action,
            {
                "action": "assess_sufficiency",
                "sufficient": True,
            },
        )

    def test_wrong_action_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            P2GenerationError,
            "Expected action",
        ):
            validate_generated_action(
                '{"action": "evaluate_candidates", '
                '"pool_sufficient": true}',
                expected_action="assess_sufficiency",
            )

    def test_surrounding_prose_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            P2GenerationError,
            "invalid JSON",
        ):
            validate_generated_action(
                'Here is the answer: '
                '{"action": "assess_sufficiency", '
                '"sufficient": true}'
            )

    def test_duplicate_json_key_is_rejected(self) -> None:
        with self.assertRaisesRegex(
            P2GenerationError,
            "duplicate JSON key",
        ):
            validate_generated_action(
                '{"action": "assess_sufficiency", '
                '"sufficient": true, "sufficient": false}'
            )

    def test_input_counting_does_not_truncate(self) -> None:
        generator = P2ActionGenerator.__new__(
            P2ActionGenerator
        )
        generator.tokenizer = FakeTokenizer()

        count = generator.count_input_tokens(
            "one two three"
        )

        self.assertEqual(count, 4)

    def test_empty_prompt_is_rejected(self) -> None:
        generator = P2ActionGenerator.__new__(
            P2ActionGenerator
        )
        generator.tokenizer = FakeTokenizer()

        with self.assertRaisesRegex(
            ValueError,
            "nonempty",
        ):
            generator.count_input_tokens("   ")


if __name__ == "__main__":
    unittest.main()