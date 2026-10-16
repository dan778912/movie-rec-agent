"""Hand-calculated fixtures and edge cases for the shared ranking metrics."""

import math
import unittest

from src.evaluation.metrics import hit_rate_at_k, mean_metrics, ndcg_at_k, recall_at_k


METRICS = (recall_at_k, ndcg_at_k, hit_rate_at_k)


class RankingMetricTest(unittest.TestCase):
    def test_hand_calculated_rankings(self):
        # IDCG@3 = 1 + 1/log2(3), because there are two eligible positives.
        cases = (
            ([10, 20, 30], 1.0, 1.0, 1.0),
            ([10, 30, 20], 1.0, 0.9197207891481876, 1.0),
            ([10, 30, 40], 0.5, 0.6131471927654584, 1.0),
            ([30, 40, 50], 0.0, 0.0, 0.0),
        )
        for ranked, expected_recall, expected_ndcg, expected_hit in cases:
            with self.subTest(ranked=ranked):
                self.assertAlmostEqual(recall_at_k(ranked, {10, 20}, 3), expected_recall)
                self.assertAlmostEqual(ndcg_at_k(ranked, {10, 20}, 3), expected_ndcg)
                self.assertEqual(hit_rate_at_k(ranked, {10, 20}, 3), expected_hit)

    def test_candidates_missing_targets_do_not_shrink_ground_truth(self):
        # Retrieval missed 20 and 30: both still count in Recall and ideal DCG.
        self.assertAlmostEqual(recall_at_k([10, 40, 50], {10, 20, 30}, 3), 1 / 3)
        self.assertAlmostEqual(ndcg_at_k([10, 40, 50], {10, 20, 30}, 3), 0.46927872602275644)

    def test_cutoff_smaller_than_truth_can_have_perfect_ndcg(self):
        self.assertAlmostEqual(recall_at_k([10, 20], {10, 20, 30}, 2), 2 / 3)
        self.assertEqual(ndcg_at_k([10, 20], {10, 20, 30}, 2), 1.0)

    def test_short_rankings_have_no_gain_for_missing_positions(self):
        self.assertAlmostEqual(recall_at_k([10], {10, 20}, 3), 0.5)
        self.assertAlmostEqual(ndcg_at_k([10], {10, 20}, 3), 0.6131471927654584)
        self.assertEqual(hit_rate_at_k([10], {10, 20}, 3), 1.0)

    def test_recommendations_outside_cutoff_do_not_receive_credit(self):
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                self.assertEqual(metric([30, 40, 10], {10}, 2), 0.0)

    def test_empty_rankings_score_zero(self):
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                self.assertEqual(metric([], {10, 20}, 10), 0.0)

    def test_empty_truth_is_rejected(self):
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                with self.assertRaisesRegex(ValueError, "positive target"):
                    metric([10], set(), 10)

    def test_truth_uses_distinct_movie_ids(self):
        self.assertEqual(recall_at_k([10], [10, 10, 20], 10), 0.5)
        self.assertAlmostEqual(ndcg_at_k([10], [10, 10, 20], 10), 0.6131471927654584)

    def test_duplicate_rankings_are_rejected_even_after_cutoff(self):
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                with self.assertRaisesRegex(ValueError, "duplicate"):
                    metric([10, 20, 10], {10, 20}, 1)

    def test_invalid_cutoffs_are_rejected(self):
        for metric in METRICS:
            for k in (0, -1, 1.0, "10", True, None):
                with self.subTest(metric=metric.__name__, k=k):
                    with self.assertRaises((TypeError, ValueError)):
                        metric([10], {10}, k)

    def test_invalid_ids_are_rejected_in_both_inputs(self):
        for metric in METRICS:
            for movie_id in (0, -1, 1.0, "10", True, None):
                with self.subTest(metric=metric.__name__, movie_id=movie_id):
                    with self.assertRaises((TypeError, ValueError)):
                        metric([10, movie_id], {10}, 1)
                    with self.assertRaises((TypeError, ValueError)):
                        metric([10], [10, movie_id], 1)

    def test_unordered_rankings_and_malformed_collections_are_rejected(self):
        for metric in METRICS:
            for ranked in ({10, 20}, frozenset({10}), "10", b"10", {10: 1}, None):
                with self.subTest(metric=metric.__name__, ranked=ranked):
                    with self.assertRaises(TypeError):
                        metric(ranked, {10}, 10)
            for truth in ("10", b"10", {10: 1}, None):
                with self.subTest(metric=metric.__name__, truth=truth):
                    with self.assertRaises(TypeError):
                        metric([10], truth, 10)

    def test_generators_and_inputs_are_not_mutated(self):
        ranked = [10, 30, 20]
        relevant = {10, 20}
        for metric in METRICS:
            with self.subTest(metric=metric.__name__):
                expected = metric(ranked, relevant, 3)
                self.assertEqual(metric(iter(ranked), iter(relevant), 3), expected)
                self.assertEqual(ranked, [10, 30, 20])
                self.assertEqual(relevant, {10, 20})

    def test_bounds_and_recall_hit_rate_monotonicity(self):
        ranked = [30, 10, 40, 20]
        truth = {10, 20, 50}
        for metric in METRICS:
            scores = [metric(ranked, truth, k) for k in range(1, 7)]
            self.assertTrue(all(0 <= score <= 1 for score in scores))
            if metric is not ndcg_at_k:
                self.assertEqual(scores, sorted(scores))


class MeanMetricTest(unittest.TestCase):
    def test_macro_average_gives_each_user_equal_weight(self):
        # User 1 has one target; user 2 has three. Macro recall is 2/3,
        # whereas pooling two hits over four total targets would give 1/2.
        rows = [
            {"recall_at_10": recall_at_k([10], {10}, 10)},
            {"recall_at_10": recall_at_k([20], {20, 30, 40}, 10)},
        ]
        self.assertAlmostEqual(mean_metrics(rows)["recall_at_10"], 2 / 3)
        self.assertNotEqual(mean_metrics(rows)["recall_at_10"], 0.5)

    def test_equivalent_key_sets_and_generator_input(self):
        rows = [
            {"recall_at_10": 0.5, "ndcg_at_10": 0.8},
            {"ndcg_at_10": 0.6, "recall_at_10": 1.0},
        ]
        result = mean_metrics(iter(rows))
        self.assertEqual(result["recall_at_10"], 0.75)
        self.assertAlmostEqual(result["ndcg_at_10"], 0.7)
        self.assertEqual(rows[0], {"recall_at_10": 0.5, "ndcg_at_10": 0.8})

    def test_empty_or_inconsistent_rows_are_rejected(self):
        for rows in ([], [{}], [{"recall_at_10": 0.5}, {"ndcg_at_10": 0.5}], [{"": 0.5}], [{1: 0.5}]):
            with self.subTest(rows=rows):
                with self.assertRaises(ValueError):
                    mean_metrics(rows)
        with self.assertRaises(TypeError):
            mean_metrics([0.5])

    def test_invalid_score_values_are_rejected(self):
        for value in (math.nan, math.inf, -math.inf, -0.1, 1.1, "0.5", True, None):
            with self.subTest(value=value):
                with self.assertRaises((TypeError, ValueError)):
                    mean_metrics([{"recall_at_10": value}])


if __name__ == "__main__":
    unittest.main()
