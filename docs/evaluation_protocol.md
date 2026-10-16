# Shared evaluation protocol — Phase 1

**Status: draft for team review.** The team has not yet approved or frozen this
contract. Dataset rules describe the existing `ml25m-global-v1` implementation;
prediction length, sampling, failure handling, and reporting rules are proposed
experiment choices.

**Implementation status:** pure metric functions, hand-calculated tests, a strict
prediction-file validator, and provenance-bound user-subset selection are
implemented, together with end-to-end scoring and per-user/summary outputs.
The full official dataset has been built and verified locally; the saved
validation subset contains 300 users. Synthetic fixtures demonstrate valid
embedding/P1 scoring and deliberate validation failures. Paired bootstrap
comparison and automatic CSV/Markdown tables are implemented. Real model
predictions are not yet available. A generated subset is a fixed local
artifact, not evidence that the team has approved this draft contract.

## 1. Phase 1 comparison

| System | Output source | Purpose |
|---|---|---|
| B0 — `popularity` | Positive training-rating counts | Basic recommendation baseline |
| B2 — `bm25` | Lexical matching of canonical text | Keyword retrieval baseline |
| B3 — `embedding` | Semantic matching of canonical text | Retrieval baseline and P1 candidate generator |
| P1 — `p1` | Fine-tuned ordering of B3's frozen candidates | Measure the effect of reranking |

The primary comparison is **B3 versus P1 on the same users and exactly the same
50 candidate movie IDs per user**. An unfavorable or uncertain result is still
a result to report. Explanations receive a separate grounding audit; ranking
metrics cannot establish explanation quality.

## 2. Data, histories, and relevant targets

The authoritative data references are [`DATA_DICTIONARY.md`](../DATA_DICTIONARY.md),
[`configs/data_v1.json`](../configs/data_v1.json), and the
[implemented builder](../src/data/build.py). Processed files belong under
`data/processed/ml25m-global-v1/`.

| Rule | Existing implementation |
|---|---|
| Eligible user | At least 20 training ratings |
| Catalog movie | At least five training ratings from eligible users |
| Positive target | Held-out rating at least 4.0 |
| Validation history | Training interactions |
| Test history | Training plus validation interactions |
| Watched set | Complete `history_movie_ids` from the matching split's profile |
| Evaluable user | Eligible user with at least one in-catalog positive target in that split |
| Catalog for both splits | The same training-defined `catalog.csv` |

The builder selects the configured 90th-percentile timestamp of all ratings for
the test cutoff, then the configured 90th-percentile timestamp of pre-test
ratings for the validation cutoff. Tied timestamps mean the realized sizes need
not be exactly 81/9/10. Record the actual timestamps from `statistics.json`.

- Training: `timestamp < validation_cutoff`.
- Validation: `validation_cutoff <= timestamp < test_cutoff`.
- Test: `timestamp >= test_cutoff`.

Do not move ratings across cutoffs to force every user into every split.
Validation and test can have different evaluable populations. This clarifies the
roadmap's broader wording about “20 ratings” and users on both sides of a
cutoff; it follows the implemented training-only eligibility rule.

For each selected user `u`, define `R_u` as the distinct `movie_id` values with
`in_catalog=1` in that split's positive target CSV. **Retain every such target in
the denominator even if a method failed to retrieve it.** Do not restrict ground
truth to the candidate pool. Report out-of-catalog positives separately.

Every recommender excludes the complete `history_movie_ids`. `liked_movies` and
`disliked_movies` are selected examples and are not complete watched sets.

All text methods use the same canonical movie documents and deterministic
profile text unless a representation change is a separately named experiment.
User tags are pre-validation only. Genome tags are static source metadata:
their construction is not a fully prospective feature. Disclose that temporal
limitation for all methods using them.

## 3. Training and final-test separation

Member 6 defines training-example requirements and owns fine-tuning. Member 3
prepares training examples, checks labels and class balance, and later audits
explanations. This resolves overlapping generator assignments in the roadmap
using the team's clarified Phase 1 split.

Training targets come from the training portion only. Build each training
example's history from interactions preceding that example's target event.
Validation selects models, checkpoints, prompts, pool size, and hyperparameters.
Never use test targets for supervision, selection, or failure-driven tuning.

After settings are frozen, final-test profiles use training plus validation
history. Any refitting on training plus validation must be declared in the final
protocol and consistently applied to compared methods; it cannot use test
targets.

## 4. Fixed user subsets

The fixed **300-user validation subset** is saved in
[`configs/validation_users_300_v1.json`](../configs/validation_users_300_v1.json).
It selects from 4,751 evaluable validation users with seed `20261001`. The
implemented selector reads IDs from `validation_profiles.jsonl`, sorts them,
samples without replacement with a dedicated seeded generator, sorts the selected
IDs, and saves the actual list. The saved IDs are authoritative for every
method in the shared comparison. A seed alone is insufficient for the handoff.

The file records dataset/subset versions, selection algorithm and Python version,
seed, population/sample sizes, source-manifest/catalog/profile/target hashes,
available statistics hash, and the canonical sorted-ID-list hash. Its complete
file hash is recorded in [`configs/evaluation_v1.json`](../configs/evaluation_v1.json).
That configuration describes proposed evaluation settings and actual artifact
provenance. The scorer accepts it with `--config`; the standalone validator uses
its existing explicit file arguments and does not accept `--config`.

Before selection, the generator checks the relevant processed artifacts against
the manifest, verifies profile users equal users with in-catalog positive
targets, and rejects target/history overlap. Users with only out-of-catalog
positives legitimately have no evaluation profile and are not in the sampling
population. The full data builder/verifier has separately checked all ten
processed-file hashes, histories, and temporal boundaries once; normal metric
tests do not need to repeat that full-data scan.

An identical existing subset is accepted, while a different payload is never
silently overwritten. Use a new versioned filename for an approved changed
experiment. Reproducing bytes requires matching input artifacts and the recorded
Python version; cross-version metadata or sampling differences do not justify
resampling. Teammates should consume the saved ID list directly.

For the final comparison, select **500–1,000 test users** from
`test_profiles.jsonl` before viewing final-test scores. Fix the exact count based
on compute and save the same metadata. The test subset remains planned; it has
not been generated in this increment. No final-test model scores are used for
current development.

Every directly compared method uses the same users. Inexpensive baselines may
additionally run on the complete eligible population in a separately labeled
table. Never silently drop failed users or resample after observing scores.

## 5. Prediction contract — proposed; validator implemented

Each method exports UTF-8 JSONL with one object per selected user. These rules
are implemented by the validator; the team must still approve the experiment
contract before freezing official runs. This shortened
example illustrates fields; a Phase 1 record normally contains 50 items:

```json
{"user_id":123,"split":"validation","method":"embedding","recommendations":[{"movie_id":50,"score":0.92,"rank":1},{"movie_id":260,"score":0.87,"rank":2}]}
```

Proposed requirements:

1. `user_id`, `movie_id`, and `rank` are integer IDs/ranks, not booleans or strings.
2. `split` and `method` match the declared run.
3. Every expected user appears once; subset-specific files have no extra users.
4. Each record has exactly 50 distinct, valid, unseen catalog movie IDs.
5. Array order agrees with consecutive ranks `1` through `50`.
6. Scores are finite numbers; booleans, NaN, and infinity are invalid.
7. Rank is authoritative. Scores need not share a scale across methods. A
   listwise reranker may export `score = -rank`, documenting
   `score_kind: "rank_surrogate"` in its run metadata. This is not a probability.
8. Explanation fields are allowed and do not affect ranking metrics.

These describe the **prepared evaluation input**, not a requirement to discard
the producers' original full-population Top 100 exports. The implemented
`src.evaluation.prepare predictions` command validates the entire declared
source population/list length before selecting frozen users and the required
prefix into a new immutable artifact. It preserves all original files and
records source/derived hashes, source label, and the explicit transformation.
Invalid source rows, including unselected users and positions beyond the output
cutoff, block preparation. P1 must already contain its complete configured pool
and is never shortened from a larger candidate set.

`src.evaluation.prepare users` exports the saved IDs as a plain array for
Members 3/4's current loaders. The metadata-rich saved file remains authoritative.
Owner-supplied labels/metadata are recorded as provenance, not independently
verified claims about model execution. See the [integration guide](member2_integration.md)
for reviewed branch interfaces and artifact checksums. The 300-user/50-item
settings remain draft team choices. BM25 used these 300 users for tuning;
current comparisons are validation development evidence, not final-test estimates.

P1 exports the **complete permutation of the frozen B3 pool**. Its movie-ID set
must equal Member 5's frozen 50-item set for each user. Hash and save that pool
before reranking. Matching the number of candidates alone is insufficient.

The implemented validator rejects malformed JSON (including duplicate object
keys and nonfinite JSON constants), missing fields, incorrect types, invalid
IDs, watched movies, duplicates, bad ranks, nonfinite scores, wrong
lengths/splits, missing/extra users, and P1 pool mismatches. Scores may be zero,
negative, or tied; rank remains authoritative. Errors identify the rule and
include the line/user when available. Validation aggregates errors and never
silently cleans data or drops users. Records are returned for downstream use
only when the whole file is valid.

The validator takes an explicit catalog, split-specific profiles, saved expected
user list, prediction file, split, method, and required count (default 50).
Supported methods are `popularity`, `bm25`, `embedding`, and `p1`. P1 requires
`--candidate-pool` pointing to the frozen `embedding` JSONL; that file is also
validated against the same catalog, profiles, users, split, and count. Other
methods reject a candidate-pool argument.

The saved user file is a JSON object containing `split` (`validation` or `test`)
and a nonempty `user_ids` list of unique positive integers. The included two-user
fixture is synthetic. Real runs use the saved 300-user subset described above.
If the user file includes `source_sha256`, the validator requires the actual
catalog/profile hashes to match it, rejecting a mixed-version input context.
This binding complements the data pipeline's full history/provenance checks.

The command prints a JSON validation report to standard output and can also save
it using `--report`. The report includes `status`, `expected_users`,
`users_with_records`, `valid_users`, `prediction_user_coverage`, `missing_users`,
`unexpected_users`, `error_count`, `errors_by_code`, and an `errors` list with
`code`, `message`, `line`, and `user_id`, together with the prediction path and
input SHA-256 hashes. Exit status is 0 for valid predictions, 1 for invalid
prediction contents, and 2 for invalid arguments or unusable input context,
including an invalid frozen embedding pool. Input-context failures print an
error to standard error without a prediction-validation JSON report. Reported coverage describes the
selected users' complete valid prediction records; a report can still be invalid
because of unexpected users or other file-level errors.

See the [evaluation README](../src/evaluation/README.md) for runnable examples.

Invalid or incomplete files block official comparisons. Agree on genuine model
failure handling before inference. One proposed deterministic P1 fallback is the
original B3 order for that user, with the reason and fallback count recorded.
Its scores remain in the comparison. File-integrity problems should be repaired
and rerun, rather than being treated as acceptable dropped users.

## 6. Metric definitions and API

All core metrics use binary relevance. Let `L_u[:K]` be the first `K` ordered
recommendations and `R_u` the complete eligible positive target set.

```text
Recall@K(u) = |set(L_u[:K]) intersect R_u| / |R_u|

DCG@K(u)  = sum(rel_i / log2(i + 1), i = 1 ... K)
IDCG@K(u) = sum(1 / log2(i + 1), i = 1 ... min(K, |R_u|))
NDCG@K(u) = DCG@K(u) / IDCG@K(u)

HitRate@K(u) = 1 if set(L_u[:K]) intersects R_u, otherwise 0
```

Here `i` is a one-based rank and `rel_i` is 1 for a target in `R_u`, otherwise 0.
Missing positions in a short ranking add no gain. Its ideal denominator still
uses `min(K, |R_u|)`, so shortening a list does not shrink its ideal ranking.

Report each metric's arithmetic mean across users (macro averaging). Combining
all hits and targets into one ratio would overweight users with many targets.

The implemented functions in `src/evaluation/metrics.py` are:

- `recall_at_k(ranked_movie_ids, relevant_movie_ids, k)`
- `ndcg_at_k(ranked_movie_ids, relevant_movie_ids, k)`
- `hit_rate_at_k(ranked_movie_ids, relevant_movie_ids, k)`
- `mean_metrics(per_user_results)`

Metric inputs may be iterables. IDs must be positive Python integers, excluding
booleans. Duplicate recommendations anywhere in the supplied ranking are
invalid; relevant IDs are deduplicated. `k` must be a positive integer excluding
booleans. Empty target sets fail because such users are outside the evaluable
population. Empty rankings are valid for the low-level metrics and return zero.
The file validator separately enforces the full prediction length.

`mean_metrics` receives score-only mappings such as
`{"recall_at_10": 0.5, "ndcg_at_10": 0.6, "hit_rate_at_10": 1.0}`. Every row
must have the same nonempty keys and finite numeric values in `[0, 1]`; empty
input is invalid. Keep user IDs and other metadata separate from these mappings.
Each row receives equal weight.

These functions do not load data files. They cannot establish catalog
membership, watched-item exclusion, or complete prediction-user coverage; run
the prediction validator with the matching context before scoring model files.

### Independently calculated fixture

For `R = {10, 20}` and `K = 3`:

| Ordered recommendations | Recall@3 | NDCG@3 | Hit Rate@3 |
|---|---:|---:|---:|
| `[10, 20, 30]` | 1.0 | 1.0 | 1.0 |
| `[10, 30, 20]` | 1.0 | 0.919720789 | 1.0 |
| `[10, 30, 40]` | 0.5 | 0.613147193 | 1.0 |
| `[30, 40, 50]` | 0.0 | 0.0 | 0.0 |

For the second row, `DCG = 1/log2(2) + 1/log2(4) = 1.5` and
`IDCG = 1/log2(2) + 1/log2(3) = 1.630929754`; their ratio is approximately
`0.919720789`.

## 7. Cutoffs and named coverage measures

Primary final-ranking metrics are **Recall@10 and NDCG@10**; Hit Rate@10 is
optional. Candidate retrieval uses **Recall@20 and Recall@50**. Report
Recall@100 only for a declared run returning at least 100 candidates. A
50-candidate file cannot establish a retriever's Recall@100.

For B3 versus P1, candidate Recall@50 comes from the common frozen B3 pool and
is identical. P1 can change the top ten but cannot recover absent movies. Its
reordered top 20 is not a change to the original retrieval pool. B0 top-50 recall
can be reported as ranking coverage if generated; it is not an embedding
candidate-retrieval experiment.

Avoid an ambiguous “Coverage” column:

| Measure | Definition |
|---|---|
| Catalog-positive coverage | In-catalog positive target interactions divided by all positive target interactions, using `in_catalog` |
| Excluded positive targets | Count of positive target interactions with `in_catalog=0`, alongside the total target count |
| Prediction-user coverage | Expected users with complete valid predictions divided by all expected users |
| Candidate Recall@50 | Mean per-user recall of the first 50 candidates against each user's complete eligible target set |

Label catalog coverage as full-split or subset-specific. `statistics.json`
contains full-split values. Recommendation metrics use distinct movie IDs,
whereas catalog coverage counts target interaction rows. Official comparisons
require 100% prediction-user coverage under the declared failure policy.

## 8. Paired bootstrap intervals

`src.evaluation.bootstrap` implements this procedure over two complete scored
report bundles. It checks experiment identity, user sets, report consistency,
and the P1/B3 candidate-pool binding before comparing. Differences always mean
challenger minus baseline; choosing embedding as baseline and P1 as challenger
therefore produces P1 minus B3.

For important differences, beginning with P1 minus B3 in Recall@10/NDCG@10:

1. Match per-user scores by user ID; require identical populations.
2. Calculate one method difference per user.
3. Resample `n` users with replacement from those `n` paired differences.
4. Calculate the resampled mean difference.
5. Repeat 10,000 times with proposed seed `20261002`.
6. Report the observed mean difference and the 2.5th/97.5th percentiles as a 95%
   percentile interval.

The implemented percentile convention sorts bootstrap means and linearly
interpolates at `(B - 1) * p`, where `B` is the resample count and `p` is the
requested percentile fraction. A dedicated `random.Random(seed)` selects user
indices. The same seeded draws are used for every metric. Settings come from
the matching saved run configurations, without silent command-line overrides.

Save seed, resample count, confidence level, percentile convention, and hashes
of both input report bundles. Do not independently resample the two methods.
The command writes `comparison.json` and `per_user_differences.csv`; it preserves
the original scoring outputs. Two-user examples are implementation tests and
cannot establish the real system's effectiveness. An interval spanning zero leaves the
direction uncertain under this protocol. User-resampling uncertainty does not
cover all uncertainty, such as training-seed variability or representation
choices.

## 9. Run metadata and generated results

The implemented `src.evaluation.evaluate` command validates predictions, checks
the configured data/subset hashes, and scores the complete selected population.
It saves `per_user.csv`, `summary.json`, and `validation.json`. Ranking cutoffs
produce Recall/NDCG and optional Hit Rate; retrieval cutoffs produce Recall.
All reported cutoffs must fit within the required submitted ranking length.
Recall@20 describes each submitted ranking's first 20 movies, including P1's
reranked order. Recall@50 measures the complete 50-item pool in this experiment.
Full-split and selected-user catalog-positive coverage are reported separately.

The synthetic scoring fixture has hand-calculated results and does not represent
model performance. `src.evaluation.report` generates tables from validated scored
bundles and optional matching bootstrap bundles. It checks shared experiment
identity and preserves input hashes. Tables list only methods with supplied
results. Synthetic tables are explicitly labelled, and draft protocol status
remains visible. Avoid retyping scores or maintaining separate implementations
in notebooks.

The table generator writes `table.md`, `table.csv`, `differences.csv`, and
`report.json`. Markdown displays six decimal places; numerical output keeps
full precision. Coverage has separately named selected-user, full-split, and
prediction-user values. Optional paired intervals are bound to the submitted
run files and verified before being included. Omitted intervals are not filled
with zeros or guessed values. A table may contain an available subset of the
methods, while explicitly listing missing methods. P1 requires its embedding
counterpart to verify its frozen pool.

Each run records:

- Protocol/configuration version and full configuration.
- Dataset version, source-manifest SHA-256, subset SHA-256, prediction SHA-256,
  and the B3 candidate-pool SHA-256 where applicable. The manifest records
  individual data-artifact hashes.
- Code commit, Python version, and random seeds.
- Split, expected/evaluated users, target counts, named coverage measures,
  candidate count, errors, and fallback counts/reasons.
- Per-user scores, means, and important paired confidence intervals.

Model owners supply model/checkpoint names and revisions, template/prompt
versions, inference settings, training-example version, training time,
inference runtime/latency, hardware, and approximate compute/API cost. These
facts cannot be inferred from the recommendation IDs.

Large data, model artifacts, and predictions stay out of Git under the existing
ignore rules. `outputs/reports/` is also ignored: place a small reviewed midterm
summary under `docs/` and document generated-artifact reproduction and sharing.

## 10. Handoffs and completion gates

| Handoff | Deliverables |
|---|---|
| Implemented locally | Verified processed data, saved 300-user subset/configuration, draft protocol, tested validator/scorer/bootstrap/table generator, synthetic examples, handoff guides |
| Unblock prediction integration | Review/publish the integration package and obtain team approval of the experiment contract |
| Establish baseline results | Actual B0/B2/B3 prediction files, validated scores and coverage, generated comparison table |
| Compare reranking | Verified B3/P1 pools, paired intervals, failure analysis, midterm summary |

The validator enforces watched-item exclusion in model files using complete
profile histories. The first integration handoff remains incomplete until the
integration package is published and the team approves the contract. The saved
subset/configuration and full processed-data verification are complete locally.
The core Phase 1 evaluation tooling is implemented. Baseline/reranker owners
still need to produce actual prediction files; Member 2 must score and review
those results and turn them into honest midterm evidence.

Before freezing the contract, Members 1–6 review the data rules, saved user list,
output length, score convention, P1 pool requirement, and fallback policy. Record
the agreed version in the repository.

Phase 2 remains a required later phase, with roles assigned after Phase 1. P2
starts from B3's pool but may expand it by re-retrieval. Its future protocol must
record initial/expanded pools, clarification simulation, information available
to each control, decision/failure rates, and cost. Simulated clarification cannot
establish real-user satisfaction. P1's unchanged-pool condition does not apply
to P2's final expanded pool.

## Repository references

- [Team roadmap, Sections 4 and 7–11](../Movie_Recommendation_Agent_Roadmap.md)
- [Data dictionary](../DATA_DICTIONARY.md)
- [Data configuration](../configs/data_v1.json)
- [Data builder](../src/data/build.py)
- [Evaluation package instructions](../src/evaluation/README.md)
- [Member 2 teammate handoff](member2_handoff.md)
- [Saved validation-user subset](../configs/validation_users_300_v1.json)
- [Evaluation configuration/provenance](../configs/evaluation_v1.json)
