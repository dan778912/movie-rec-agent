# Synthetic scoring demonstration

These invented movies, users, and predictions check the scorer. Their scores are
not MovieLens results and do not measure any trained model.

The saved subset contains users 1 and 2. Their complete relevant target sets are
`{10, 20}` and `{40, 50}`. Movie 50 stays in user 2's recall denominator even
though the embedding candidate pool never retrieved it. Out-of-catalog positives
999 and 998 are counted in coverage and excluded from ranking relevance. User 3
has only an out-of-catalog target and legitimately has no evaluation profile.

| Metric | Embedding | P1 permutation of the same pools |
| --- | ---: | ---: |
| Recall@2 | 0.5 | 0.75 |
| NDCG@2 | 0.5 | 0.8065735963827292 |
| HitRate@2 | 1.0 | 1.0 |
| Recall@3 | 0.75 | 0.75 |

The averages give each user equal weight. Full-split positive-target catalog
coverage is 4/6; coverage among selected users is 4/5. Prediction user coverage
is 2/2 for both valid files. `invalid_predictions.jsonl` deliberately recommends
watched movie 90 to user 1 and must fail validation without creating results.

The configuration uses three predictions per user so these calculations can be
checked by hand. The real project configuration requires 50. Configuration paths
are relative to the repository root, not this fixture directory.
