# Member 3: training data, popularity, and explanation audit

Use the commands below from the repository root. Python standard library only;
`python3` works on this Mac. The team environment specification is Python 3.10+.
The data must already exist at `data/processed/ml25m-global-v1/`; run Member 1's
`python3 -m src.data.prepare` if not. Do not edit the shared processed artifacts.

Frozen Member 3 settings: [configs/member3_v1.json](../../configs/member3_v1.json).
All generated artifacts below are Git-ignored. Configurations, code, and tests
should be committed. Reports record configuration, seed, split-manifest hash,
source-code hashes, Git revision, and whether the worktree had uncommitted work.
A changed runtime does not change the deterministic JSONL output checksum.

## Step 1: give Member 6 practice examples

```bash
# Small readable smoke export: 20 sampled users, at most 20 examples each.
python3 -m src.baselines.training --users 20 --output outputs/predictions/p1-training-smoke.jsonl

# Initial training handoff: 1,000 sampled users, at most 20 examples each.
python3 -m src.baselines.training
```

The full command above creates `outputs/predictions/p1-training.jsonl` and
`p1-training.report.json`. It scans all training ratings and chooses a seeded,
uniform reservoir of users who have eligible targets. Within each sampled user,
it chooses up to 20 targets uniformly without replacement. This bounds compute,
prevents a few prolific raters dominating the export, and avoids taking only the
first users in the file. The report contains user IDs, population exclusions,
exported labels, positive fraction, runtime, and errors. No class balancing is
silently applied. To change scale, version the config before running training;
`--users` is a recorded override for smoke runs. This initial sample is not a
claim that the whole training dataset has been exported.

An example has the following fields (illustrative values):

```json
{
  "example_id": "123:2571:1100000000",
  "user_id": 123,
  "movie_id": 2571,
  "split": "train",
  "profile": "Highly rated movies: ...",
  "candidate_document": "Matrix, The (1999). Genres: ...",
  "label": "relevant",
  "target_rating": 5.0,
  "target_timestamp": 1100000000,
  "negative_source": null,
  "history_count": 20,
  "history_latest_timestamp": 1099999999,
  "liked_movies": [],
  "disliked_movies": []
}
```

**Model input: `profile` and `candidate_document` only. Supervision: `label`.**
Target ratings/timestamps, labels, and provenance must not be added to the model
input. Member 6 can transform these into their chosen SFT/classification format.
The file supports the pointwise relevance task. It does not contain listwise
candidate-pool targets, human explanation labels, or fabricated justification
supervision. Coordinate those additional formats with Member 6 before training.

Construction rules:

- Only `train_ratings.csv` supplies interactions. Validation/test targets are not read.
- Positive >=4.0, explicit negative <=2.5; skip 3/3.5 as targets.
- Five strictly earlier ratings are required by default. Same-timestamp events
  never appear in each other's profiles.
- Earlier 3/3.5 ratings still count as watched history, but are not liked/disliked.
- Candidate movies must be in the shared movie documents and not already rated.
- Profiles use the shared deterministic formatter and up to 10 liked/5 disliked
  examples, ordered by rating, recency, and movie ID.

Metadata caveat: shared movie documents contain static genome tags and user tags
aggregated up to the training cutoff; they are not metadata snapshots at each
training target time. Out-of-catalog history is counted but omitted from profile
text because shared documents exist only for catalog movies. Disclose both
choices with Member 6; use the same supplied metadata across text methods.

## Step 2: build popularity recommendations

```bash
python3 -m src.baselines.popularity
```

Creates `outputs/predictions/b0-validation.jsonl` and `b0-validation.report.json`.
Every validation profile user gets up to 100 recommendations in the shared format:

```json
{"user_id":123,"split":"validation","method":"popularity","recommendations":[{"movie_id":50,"score":1000,"rank":1}]}
```

Score = `catalog.csv`'s `train_positives`. Sort descending; break ties by movie ID
ascending; exclude every `history_movie_ids` item, including those omitted from
the text profile. The score is a count, not a probability. Report short lists when
there are fewer than 100 unseen movies. Test-stage counts remain training-only;
test profiles remove the full training-plus-validation watched history.

Member 2 can supply a fixed subset as a JSON array of unique integer IDs:

```bash
python3 -m src.baselines.popularity --user-ids configs/reranker-validation-users.json --output outputs/predictions/b0-validation-subset.jsonl
```

That subset file is not yet supplied by Member 2. By default all validation users
are scored. Once settings are frozen, `--split test` writes `b0-test.jsonl`.
Do not use test targets for tuning.

There is no B0 model hyperparameter sweep: the score/tie rule is fixed. Export a
single top-100 ordering and let the shared evaluator compute Recall/NDCG at the
required cutoffs. Member 2's validator and evaluator remain authoritative; this
module does not invent a second final metric implementation.

Known limitations: global popularity favors widely rated movies and ignores a
user's taste except for watched-item exclusion. Recommendation quality, coverage,
and specific failure cases must be measured with the shared evaluator. Its code
is currently a stub, so an evaluated result-table row is still pending.

## Step 3: manually check P1 explanations when available

```bash
python3 -m src.baselines.audit prepare --predictions outputs/predictions/p1-validation.jsonl
```

The predictions must follow the shared JSONL format with an `explanation` string
on each recommendation. The seeded reservoir sample selects up to 50 explanations
and attaches the matching canonical user profile and movie document. Missing
explanations are counted; an export with none fails clearly. The output is
`outputs/reports/p1-audit.jsonl` plus a provenance report. This is a review packet,
not a completed faithfulness audit.

Read the evidence and fill `verdict` with `supported`, `unsupported`, or `unclear`.
List the exact unsupported claims in `unsupported_claims`, and record observations
in `reviewer_notes`. Check invented user preferences, unsupported plot/cast facts,
and contradictions. A true fact absent from the supplied context is still
unsupported by that context. If Member 6 supplies additional input, attach it
before judging. Recommendation accuracy and explanation grounding are different.

```bash
python3 -m src.baselines.audit summarize
```

This writes `outputs/reports/p1-audit-summary.json` with verdict counts, unsupported
and unclear fractions, and unsupported examples. It retains a pending status if
any reviews are unfilled; it never treats missing reviews as supported.
P1 predictions do not exist in this repo yet, so no real audit can be completed.

## Reproduce the handoff integrity checks

```bash
python3 -m src.baselines.checks training --input outputs/predictions/p1-training.jsonl --output outputs/reports/p1-training-checks.json
python3 -m src.baselines.checks popularity --input outputs/predictions/b0-validation.jsonl --output outputs/reports/b0-validation-checks.json
```

These compare saved artifacts against source training history and catalog data,
including every exported training record's selected history and exact text, and
every B0 user's expected ranking. They fail on altered labels, future history,
watched recommendations, and missing users. Checking training scans the full
training CSV and may take about a minute. These are Member 3 handoff checks;
Member 2's shared prediction validator, metrics, and result table are still needed.
For a subset B0 export, pass the same `--user-ids` file to the check command.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Tests cover target-time leakage, timestamp ties, label thresholds/exclusions,
seeded sampling, input checksums, popularity ties/watched removal, subset failures,
and the distinction between a pending review and a completed audit.
