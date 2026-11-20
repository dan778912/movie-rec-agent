# Synthetic prediction preparation checks

These files are artificial test inputs, not real model predictions or project performance results.

The complete split has three evaluable users (1, 2, 3); the frozen selection contains users 1 and 2. User 4 has only an out-of-catalog target and intentionally has no evaluable profile. Full embedding predictions contain four recommendations per user. The configuration requires the first three recommendations for the two selected users. The frozen candidate pool contains exactly those three-item embedding prefixes for all three users. P1 reranks each frozen pool without adding or removing movies.

The selected users preserve the independently calculated scoring example: embedding Recall@2 = 0.5, NDCG@2 = 0.5, HitRate@2 = 1, Recall@3 = 0.75; P1 Recall@2 = 0.75, NDCG@2 = approximately 0.806574, HitRate@2 = 1, Recall@3 = 0.75. User 2's relevant movie 50 is absent from its candidate pool and remains in the recall denominator.

Input rows are deliberately unsorted. User 1 contains extra model metadata and an explanation to check that preparation preserves the original recommendation content. Tests mutate unselected users and discarded tails to verify that filtering cannot hide malformed predictions.
