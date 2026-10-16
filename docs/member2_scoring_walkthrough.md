# Member 2 — run and understand the scoring command

This step connects the prediction validator, complete held-out targets, and
metric helpers. It uses Python's standard library and the existing environment
on your Mac. No model download or additional package installation is required.

## 1. Activate the project environment

Open Terminal and run this complete block:

```bash
cd "$HOME/Codex Projects/movie-rec-agent"
source .venv/bin/activate
```

Run every subsequent command from that project directory.

## 2. Run the synthetic embedding example

```bash
python -m src.evaluation.evaluate \
  --config tests/fixtures/evaluation_scoring/config.json \
  --predictions tests/fixtures/evaluation_scoring/embedding.jsonl \
  --method embedding \
  --output-dir outputs/reports/scoring_demo/embedding
```

This example contains two users and three recommendations per user. Its
configuration deliberately uses cutoffs 2 and 3 so you can check the arithmetic.
The real experiment configuration still uses 300 users, 50 recommendations,
ranking cutoff 10, and recall cutoffs 20/50.

The printed JSON summary should report `status: "evaluated"`, `user_count: 2`,
`prediction_user_coverage: 1.0`, and these macro metrics:

| Metric key | Expected value |
|---|---:|
| `recall_at_2` | 0.5 |
| `ndcg_at_2` | 0.5 |
| `hit_rate_at_2` | 1.0 |
| `recall_at_3` | 0.75 |

These are **synthetic implementation checks**, not MovieLens recommender
performance. Keep them out of the midterm's real-results table.

## 3. Understand why these scores are correct

| User | Relevant catalog movies | Submitted ranking |
|---|---|---|
| 1 | 10, 20 | 10, 30, 20 |
| 2 | 40, 50 | 60, 40, 70 |

At cutoff 2, each user receives one of their two relevant movies, so each has
Recall@2 = 1/2. User 1's hit is at rank 1 and user 2's hit is at rank 2. Their
NDCG@2 values are approximately 0.613147 and 0.386853, whose mean is 0.5.
Both users have a hit, so mean HitRate@2 is 1.0.

At cutoff 3, user 1 receives both relevant movies, giving recall 1.0. User 2
still receives only movie 40, giving recall 0.5. The equal-weight mean is
(1.0 + 0.5) / 2 = 0.75. Movie 50 was missed by retrieval and still remains in
user 2's recall denominator and ideal NDCG calculation.

There are also two out-of-catalog positive targets: one belongs to user 1 and
one belongs to a third user with no in-catalog positives. The full target file
has catalog-positive coverage 4/6; the selected two users have coverage 4/5.
These are different populations and therefore have separately labeled values
in the summary. Neither fraction is the recommendation Recall@3.

## 4. Check a synthetic P1 reranking

```bash
python -m src.evaluation.evaluate \
  --config tests/fixtures/evaluation_scoring/config.json \
  --predictions tests/fixtures/evaluation_scoring/p1.jsonl \
  --method p1 \
  --candidate-pool tests/fixtures/evaluation_scoring/embedding.jsonl \
  --output-dir outputs/reports/scoring_demo/p1
```

Expected metrics are Recall@2 = 0.75, NDCG@2 approximately 0.806574,
HitRate@2 = 1.0, and Recall@3 = 0.75. P1 improves the top-two order in this
constructed example. It cannot recover movie 50 because that movie is absent
from user 2's frozen embedding pool. Full-pool Recall@3 therefore stays equal.
This illustrates the comparison logic; it does not predict whether real P1
fine-tuning will improve results.

## 5. Inspect the saved files

Each requested output directory contains:

- `summary.json`: macro scores, user count, coverage, full configuration,
  input hashes, Python version, and implementation identity.
- `per_user.csv`: one row per selected user, with target counts and metric scores.
  These rows support the paired comparison in the [bootstrap walkthrough](member2_bootstrap_walkthrough.md).
- `validation.json`: the prediction-integrity report, including expected users,
  valid users, error counts, and P1 pool identity when applicable.

Identical reruns are accepted. If inputs, configuration, implementation, or
recorded provenance change, choose a new result directory to preserve the old
run. Never change the frozen user list merely to improve scores.

Generated results are ignored by Git. Code, small synthetic fixtures,
configuration files, and documentation are the shareable implementation package.

## 6. Run the automated checks

```bash
python -m unittest discover -s tests -v
```

Expect all tests to pass with `OK`. Tests check hand-calculated scores,
complete target denominators, equal user weighting, provenance mismatches,
invalid predictions, and safe output handling. They use tiny temporary data and
do not rebuild the real MovieLens dataset.

## 7. Apply the scorer to your real experiment later

Once Member 5 supplies the actual embedding predictions for the saved 300 users,
run the real-data scoring command in the [handoff](member2_handoff.md#4-score-validated-exports).
Members 3 and 4 supply compatible popularity and BM25 files; Member 6 supplies
P1 output plus the same frozen B3 pool. The scorer already runs prediction
validation before producing any scores.

The proposed experiment contract still needs team review. Paired bootstrap
comparison and automatic tables are implemented; follow the
[bootstrap](member2_bootstrap_walkthrough.md) and
[table](member2_table_walkthrough.md) walkthroughs next. Actual baseline/reranker
predictions are needed for midterm performance evidence; synthetic demonstrations
cannot replace them.
