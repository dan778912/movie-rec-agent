# Synthetic prediction-validation examples

These nine movie IDs and two users are invented fixtures for tests and a local
validator demonstration. They are **not MovieLens records, model predictions,
an evaluation-user subset, or project experiment results**.

Both users belong to the `validation` split. User 1 has already rated movie 90;
user 2 has already rated movie 80. Their empty `liked_movies` fields deliberately
show why validation must use the complete `history_movie_ids` field.

The examples use three recommendations per user; pass `--required-count 3`.
The project's production prediction length is 50.

| File | Expected use |
|---|---|
| `catalog.csv` | The complete nine-movie synthetic catalog |
| `profiles.jsonl` | Complete synthetic watched histories |
| `users.json` | The exact expected user IDs: 1 and 2 |
| `embedding_valid.jsonl` | Valid embedding output and frozen candidate pool |
| `p1_valid.jsonl` | Valid P1 permutation of each user's embedding candidates |
| `watched_invalid.jsonl` | Invalid: user 1 is recommended previously watched movie 90 |
| `p1_invalid_pool.jsonl` | Invalid: P1 substitutes movie 70 for candidate movie 30 |

The test suite runs the validator on copies in temporary directories and checks
that the prediction inputs are not modified. No actual dataset download,
training, or network access is required.
