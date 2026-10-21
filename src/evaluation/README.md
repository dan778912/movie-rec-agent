# Shared evaluation — Member 2

Implemented: Recall@K, binary NDCG@K, Hit Rate@K, equal-weight user averaging,
strict prediction-file validation, reproducible user selection, and end-to-end
scoring with per-user CSV and JSON summary outputs, and paired user bootstrap
comparisons with confidence intervals, automatic CSV/Markdown tables, and
provenance-recorded preparation of teammate exports plus plain user-ID export.
The official MovieLens 25M data has been built and verified locally, and the
saved validation subset contains 300 users. Tests and validator demonstrations
still use tiny synthetic data, so they need no download or GPU. These components
use the Python standard library and need no additional dependencies.

The [draft protocol](../../docs/evaluation_protocol.md) describes the data rules,
proposed prediction contract, and B3/P1 comparison. The contract awaits team
review. The [teammate handoff](../../docs/member2_handoff.md) explains how to use
the real data and saved IDs. The core Phase 1 evaluation tooling is implemented.
Actual B0/B2/B3/P1 validation is complete on the same 300 users. See the
[portable validation archive](../../docs/results/member2_validation_300_v1/README.md)
for complete run bundles, paired comparisons, and the four-method table.

The [integration guide](../../docs/member2_integration.md) explains the actual
reviewed producer interfaces, artifact access, and preparation commands. Members
3/4 normally export all users with 100 recommendations; Member 5 also freezes a
Top 50 file. Preparation validates their entire export, then derives the saved
300-user/50-item input without changing original files. The 300-user/50-item
contract remains a proposed team choice.

## Activate the environment on your Mac

Open Terminal and run:

```bash
cd "$HOME/Codex Projects/movie-rec-agent"
source .venv/bin/activate
```

Run all subsequent commands from the repository root.

## Real data and the fixed 300-user subset

The processed dataset at `data/processed/ml25m-global-v1/` is available on this
Mac and has passed Member 1's full verification: all ten processed-file hashes,
complete histories, and temporal boundaries were checked. Do not rerun the full
build or full rating-file verification before every unit test. A teammate who
does not have the data should follow the one-time build in the
[handoff](../../docs/member2_handoff.md).

Verified full-data counts:

| Quantity | Value |
|---|---:|
| Source ratings | 25,000,095 |
| Training-eligible users | 138,950 |
| Catalog movies | 21,333 |
| Evaluable validation users | 4,751 |
| Validation positive targets, all eligible users | 148,263 |
| Validation positive targets in catalog | 126,029 |
| Full-validation catalog-positive coverage | 85.0037% |

These counts describe data availability, not recommender performance. Some
users have positive targets only outside the catalog and therefore correctly
have no evaluation profile. The full target file retains their positives for
catalog-coverage reporting.

The shared experiment uses the actual IDs saved in
[`configs/validation_users_300_v1.json`](../../configs/validation_users_300_v1.json):
300 users selected without replacement from the 4,751 evaluable validation users,
with seed `20261001`. Never resample after seeing model scores. Every method in
the shared table uses these same users.

To verify reproducibility with matching source files and the recorded Python
version, run:

```bash
python -m src.evaluation.subsets \
  --data-dir data/processed/ml25m-global-v1 \
  --split validation \
  --size 300 \
  --seed 20261001 \
  --output configs/validation_users_300_v1.json
```

The report should contain `status: "created_or_verified"`, `selected_users: 300`,
and `eligible_users: 4751`, plus the saved-file hash. An existing identical file
is accepted without changing it. A different subset/provenance is rejected;
preserve the saved file and investigate rather than deleting or overwriting it.

The generator verifies catalog, profiles, targets, and available statistics
against the source manifest; validates the evaluable user population; and checks
that held-out targets do not overlap histories. It records dataset/subset
versions, source hashes, algorithm, Python version, seed, size, the actual sorted
IDs, and a separate ID-list hash. The complete file hash appears in the command
report and the [experiment configuration](../../configs/evaluation_v1.json).
Because Python version is part of that metadata, byte-identical regeneration is
not promised across Python versions. Teammates should use the committed saved
IDs as authoritative instead of recreating them independently.

`configs/evaluation_v1.json` records proposed settings and actual provenance.
The scorer accepts it with `--config`; the standalone validator continues to use
explicit file arguments. Bootstrap comparison reads the settings embedded in the
two scored runs. The saved data/subset hashes establish identity; team approval
of the protocol is still outstanding.

For Members 3/4's bare-array loaders, use the companion export
[`configs/validation_users_300_export_v1/user_ids.json`](../../configs/validation_users_300_export_v1/user_ids.json).
Its provenance binds it to the original saved subset. Reproduce/check it:

```bash
python -m src.evaluation.prepare users \
  --config configs/evaluation_v1.json \
  --output-dir configs/validation_users_300_export_v1
```

Expected status is `users_exported`, with 300 selected users. Keep the original
metadata-rich saved subset as the scorer/validator input; the array is a producer
compatibility export.

## Run the checks

```bash
python -m unittest discover -s tests -v
```

This includes Member 1's synthetic data-pipeline test and Member 2's metric,
validator, subset, preparation, scoring, paired bootstrap, and reporting tests. It does not download data or
produce model experiment results.
To run only the hand-calculated metric checks:

```bash
python -m unittest discover -s tests -p 'test_metrics.py' -v
```

## Try the prediction validator

The [synthetic fixtures](../../tests/fixtures/evaluation_validation/README.md)
contain two users, a tiny catalog, complete watched histories, and three
recommendations per user. Three is a demonstration length; the validator's
default required count for Phase 1 is 50.

### 1. Validate compatible embedding predictions

```bash
python -m src.evaluation.validate \
  --catalog tests/fixtures/evaluation_validation/catalog.csv \
  --profiles tests/fixtures/evaluation_validation/profiles.jsonl \
  --users tests/fixtures/evaluation_validation/users.json \
  --predictions tests/fixtures/evaluation_validation/embedding_valid.jsonl \
  --split validation \
  --method embedding \
  --required-count 3
```

The JSON report should have `status: "valid"`, `expected_users: 2`,
`valid_users: 2`, `prediction_user_coverage: 1.0` (100%), and `error_count: 0`.
The exit status is 0.

### 2. Validate P1 against the frozen embedding candidates

```bash
python -m src.evaluation.validate \
  --catalog tests/fixtures/evaluation_validation/catalog.csv \
  --profiles tests/fixtures/evaluation_validation/profiles.jsonl \
  --users tests/fixtures/evaluation_validation/users.json \
  --predictions tests/fixtures/evaluation_validation/p1_valid.jsonl \
  --split validation \
  --method p1 \
  --required-count 3 \
  --candidate-pool tests/fixtures/evaluation_validation/embedding_valid.jsonl
```

This should also report two valid users and 100% prediction-user coverage. P1
reorders exactly the same candidate movie IDs per user. The embedding pool is
itself validated against the same context. `p1` requires `--candidate-pool`;
`popularity`, `bm25`, and `embedding` reject that argument.

### 3. Observe a deliberately invalid watched recommendation

```bash
python -m src.evaluation.validate \
  --catalog tests/fixtures/evaluation_validation/catalog.csv \
  --profiles tests/fixtures/evaluation_validation/profiles.jsonl \
  --users tests/fixtures/evaluation_validation/users.json \
  --predictions tests/fixtures/evaluation_validation/watched_invalid.jsonl \
  --split validation \
  --method embedding \
  --required-count 3
```

This is expected to fail: `status: "invalid"`, a `watched_movie` error identifying
line 1, user 1, and movie 90, and exit status 1. This is an intentional fixture
failure, not a setup problem. Replacing P1's prediction file with
`p1_invalid_pool.jsonl` in the earlier command demonstrates an altered-pool error.

## What validation checks

The validator aggregates errors and rejects:

- Malformed JSON, duplicate object keys, and nonfinite JSON constants.
- Missing fields or wrong types, including boolean IDs and ranks.
- Wrong splits/methods, duplicated users, missing users, and unexpected users.
- Unknown catalog IDs, repeated movies, and watched movies.
- Incorrect list lengths, ranks inconsistent with array order, nonfinite scores.
- A P1 movie-ID set different from that user's frozen B3 candidate set.

Supported method identifiers are `popularity`, `bm25`, `embedding`, and `p1`.
IDs and ranks are positive integers. Scores are finite numbers; zero, negative,
and tied scores are allowed. Rank is authoritative. Explanation fields may be
included and do not affect these checks.

Watched-item exclusion uses the matching profile's complete
`history_movie_ids`; `liked_movies` is only a selection of examples. See the
[data dictionary](../../DATA_DICTIONARY.md).

The validator does not clean files, remove bad recommendations, or drop failed
users. The Python API returns records for downstream scoring only when the whole
file is valid. Fix the producing method/export and rerun when a real file fails.

## Inputs, reports, and exit status

`--users` requires a saved JSON object with these required fields:

```json
{"split": "validation", "user_ids": [1, 2]}
```

The split must be `validation` or `test` and match the run. The nonempty ID list
must contain unique positive integers. This two-user fixture is synthetic; real
runs use the saved 300-user subset with provenance metadata. When the user file
includes `source_sha256`, the validator requires the actual catalog and profile
hashes to match those recorded in the subset. Mixing another data version with
the frozen user list is rejected as an unusable context.

The JSON report is printed to standard output. The complete real-data command
below demonstrates saving a copy using `--report`; this option belongs at the
end of the command, rather than being a standalone Terminal command.

Report fields include `status`, `expected_users`, `users_with_records`,
`valid_users`, `prediction_user_coverage`, `missing_users`, `unexpected_users`,
`error_count`, `errors_by_code`, and `errors`. Each error has `code`, `message`,
`line`, and `user_id`; unavailable locations use null. The prediction path and
input SHA-256 hashes identify the checked artifacts. Coverage is valid expected users divided
by expected users. Always check `status` and `error_count`: a complete set of
expected users can coexist with unexpected users or other file-level errors.

| Exit status | Meaning |
|---|---|
| `0` | Valid predictions and context |
| `1` | Prediction contents failed validation |
| `2` | Invalid arguments or unusable context, including a missing catalog, malformed user list, or invalid frozen embedding pool |

An input-context error is printed to standard error without a JSON prediction
validation report: the prediction file could not be assessed using valid
reference data. A P1 prediction with a different movie set from an otherwise
valid frozen pool is a prediction-content error (exit 1).

Validation checks prediction integrity and binds real catalog/profile files to
the frozen subset's hashes. It does not calculate Recall/NDCG or independently
reconstruct histories from raw ratings. Member 1's builder/verifier owns that
complete data provenance check.

## Real Phase 1 validation — completed record and local reproduction

The completed validation reports are archived; a fresh clone does not include
large prediction artifacts merely by fetching model-owner branches. Obtain the
matching files through agreed storage, then use the
[integration guide](../../docs/member2_integration.md) to prepare them. The
following validation-only command requires the prepared local embedding file.

```bash
python -m src.evaluation.validate \
  --catalog data/processed/ml25m-global-v1/catalog.csv \
  --profiles data/processed/ml25m-global-v1/validation_profiles.jsonl \
  --users configs/validation_users_300_v1.json \
  --predictions outputs/predictions/prepared/b3_validation_v1/predictions.jsonl \
  --split validation \
  --method embedding \
  --required-count 50 \
  --report outputs/reports/validation_checks/b3_validation.json
```

The official run should have 300 valid users, 100% prediction-user coverage, and
zero errors. Other methods use `--method popularity`, `--method bm25`, or
`--method p1` and the corresponding prediction-file path. For P1, also append
`--candidate-pool` with the frozen embedding prediction-file path to the complete
command; the earlier synthetic P1 example shows that syntax. Member 5 freezes
that 50-movie pool before Member 6 reranks it. P1 must return the same 50 IDs in
a different order, with no additions or omissions.

The saved test population contains 500 users. Member 6's test predictions
already passed this standalone validator without reading labels. Use the saved
[test inference record](../../docs/results/member2_test_500_v1/README.md) and
frozen selection for further integrity checks. The user subsequently relayed
explicit team approval to run and publish P1/B3 test results; the
[500-user comparison](../../docs/results/member2_p1_b3_test_500_v1/README.md)
is now complete. `evaluate`, `prepare`, and `subsets` can open target labels;
the new archive preserves exact authorized test commands separately from the
validation examples below. Keep the saved IDs/settings unchanged.

## Score complete prediction files

The scorer checks the configuration, data manifest, saved users, complete target
population, and prediction file before calculating scores. It rejects invalid
or missing predictions instead of silently averaging only successful users.

Run this complete, synthetic example now:

```bash
python -m src.evaluation.evaluate \
  --config tests/fixtures/evaluation_scoring/config.json \
  --predictions tests/fixtures/evaluation_scoring/embedding.jsonl \
  --method embedding \
  --output-dir outputs/reports/scoring_demo/embedding
```

Expected macro scores: Recall@2 `0.5`, NDCG@2 `0.5`, HitRate@2 `1.0`, and
Recall@3 `0.75`. The [walkthrough](../../docs/member2_scoring_walkthrough.md)
explains these values and how to check a synthetic P1 reranking. These test
scores must not be presented as real model performance.

The scorer saves three files:

| File | Contents |
|---|---|
| `summary.json` | Macro metrics, user count, named coverage populations, configuration and source identity |
| `per_user.csv` | One sorted row per selected user, with target counts and metric scores |
| `validation.json` | The prediction validator's integrity report |

Reusing an identical result directory is accepted; different results require a
new output directory. The scorer protects data inputs and does not write scored
results for invalid predictions. Generated reports remain Git-ignored.

For a matching locally prepared embedding validation export:

```bash
python -m src.evaluation.evaluate \
  --config configs/evaluation_v1.json \
  --predictions outputs/predictions/prepared/b3_validation_v1/predictions.jsonl \
  --method embedding \
  --output-dir outputs/reports/b3_validation_v1
```

That command requires the actual prepared prediction file. It documents the
completed validation workflow; raw inputs are managed separately from Git.
It uses Recall@10, NDCG@10, HitRate@10, Recall@20, and Recall@50 as configured.
For P1, supply its prediction file, `--method p1`, and the required
`--candidate-pool` pointing to the frozen embedding export. The complete P1
command is in the [handoff](../../docs/member2_handoff.md).

Configuration `data_dir` and `users_file` paths resolve from the repository root;
command-line prediction/output paths follow the current working directory.
Only the selected split's target file is read. The summary preserves draft
configuration status: successful scoring does not approve an experiment design.

Scoring exits with 0 on success, 1 on invalid prediction contents, or 2 for
configuration/data/output errors. Invalid predictions produce a validation
report on standard output and an explanation on standard error, without scored
result files. Correct the problem or choose a new result directory as indicated
by the message.

## Compare two scored runs with paired bootstrap

After scoring the synthetic embedding and P1 examples, run:

```bash
python -m src.evaluation.bootstrap \
  --baseline-dir outputs/reports/scoring_demo/embedding \
  --challenger-dir outputs/reports/scoring_demo/p1 \
  --output-dir outputs/reports/scoring_demo/p1_vs_embedding
```

The command compares every reported metric as **challenger minus baseline**.
It joins by user ID, rejects mismatched experiments or incomplete results, and
checks that P1's frozen embedding pool is the embedding run being compared.
It reads the saved `summary.json`, `per_user.csv`, and `validation.json` from
each run; it does not require access to the original prediction/data paths.
It checks report consistency rather than rerunning the models or rescoring
their original predictions. Preserve the complete scored report bundles.

Bootstrap settings come from both runs' matching configurations: the current
proposal is 10,000 paired user resamples, 95% confidence, seed `20261002`.
Percentiles use linear interpolation at `(number_of_resamples - 1) * p`.
Each resample selects users with replacement and carries both method scores
together. The same seeded index sequence is used for each metric.

Output files are `comparison.json` and `per_user_differences.csv`. The JSON
records metric means, observed differences, intervals, settings, and input
report hashes. The CSV records both methods' scores and their difference for
each user. Identical reruns are accepted; changed comparisons require a new
output directory. Exit 0 means success; exit 2 reports incompatible inputs or
an output error and does not produce a scored comparison.

Synthetic expected Recall@2 difference: `0.25`, with interval `[0.0, 0.5]`.
Synthetic expected NDCG@2 difference: approximately `0.306574`, with interval
`[0.226294, 0.386853]`. These are two-user implementation examples. An interval
containing zero leaves the direction uncertain under this procedure. The
interval covers user-resampling uncertainty, not training-seed or model-selection
uncertainty. The [walkthrough](../../docs/member2_bootstrap_walkthrough.md)
explains the arithmetic and the real-experiment command.

Scoring summaries keep `bootstrap_applied: false`: bootstrap is a separate
comparison artifact and does not rewrite those original run reports.

## Generate a comparison table

After the two synthetic scoring runs and bootstrap comparison exist, run:

```bash
python -m src.evaluation.report \
  --runs outputs/reports/scoring_demo/embedding outputs/reports/scoring_demo/p1 \
  --comparisons outputs/reports/scoring_demo/p1_vs_embedding \
  --kind synthetic \
  --output-dir outputs/reports/scoring_demo/table
```

The generator validates complete scored bundles and rejects mixed datasets,
configurations, users, target counts, or scoring implementations. P1 requires
its corresponding embedding run. Optional comparison bundles must bind to the
supplied runs and pass checks of their saved differences and bootstrap results.
Report-file paths may change when bundles are shared; the recorded hashes and
method identity establish which results belong together.

Output files:

| File | Contents |
|---|---|
| `table.md` | Readable metric means, separate coverage figures, optional paired differences and intervals |
| `table.csv` | One row per supplied method, retaining full numerical precision |
| `differences.csv` | One row per supplied comparison/metric, including interval settings |
| `report.json` | Machine-readable table, missing-method list, experiment identity, and input hashes |

Markdown rounds displayed scores to six decimal places. CSV and JSON preserve
their numerical precision. Methods are ordered as popularity, BM25, embedding,
then P1. A method without a supplied run receives no row or invented score.
`--comparisons` is optional: omitted comparisons produce no invented intervals.
A single supplied baseline is supported; including P1 also requires embedding.

`--kind synthetic` marks test tables explicitly. Real model results use
`--kind experiment`; the known synthetic dataset cannot be labelled as an
experiment. Configuration status remains visible, including
`draft_for_team_review`. Successful table generation does not approve the
protocol or establish model effectiveness.

Outputs are immutable: identical reruns pass, while changed results require a
new output directory. Input reports, comparisons, and data artifacts are
protected from output overwrites. Exit 0 means success; exit 2 reports unusable
inputs or an output error. The generator does not rerun recommendation models.

See the [table walkthrough](../../docs/member2_table_walkthrough.md) for the
expected synthetic table and the corresponding real-validation workflow. The
completed portable archive is linked above.

## Try a metric example

```bash
python -c 'from src.evaluation.metrics import recall_at_k, ndcg_at_k, hit_rate_at_k; ranked = [10, 30, 20]; relevant = {10, 20}; print("Recall@3:", recall_at_k(ranked, relevant, 3)); print("NDCG@3:", ndcg_at_k(ranked, relevant, 3)); print("Hit Rate@3:", hit_rate_at_k(ranked, relevant, 3))'
```

Expected values: Recall@3 = `1.0`, NDCG@3 approximately `0.919720789`, Hit Rate@3 =
`1.0`. A relevant movie at rank 3 lowers NDCG compared with ideal ranks 1 and 2.

## Use the metrics correctly

- Supply IDs in rank order and the complete in-catalog held-out positive target
  set. Keep retrieval misses in that target set.
- IDs and cutoffs are positive integers excluding booleans. Duplicate
  recommendations, invalid cutoffs, and empty target sets fail; relevant IDs
  are deduplicated.
- Empty/short rankings receive zero gain for missing positions. The file
  validator separately enforces full list length.
- Calculate scores per user, then average equally with `mean_metrics`. It accepts
  score-only mappings with the same nonempty keys, such as `recall_at_10` and
  `ndcg_at_10`. Values must be finite numbers in `[0, 1]`; empty populations fail.
  Keep IDs and metadata separate.
- The metric helpers cannot check catalog membership, watched IDs, user coverage,
  or pool equality. Run the validator with matching context before scoring files.

## Current implementation and results status

`src.evaluation.prepare`, `src.evaluation.evaluate`, `src.evaluation.bootstrap`,
`src.evaluation.report`, `src.evaluation.subsets`, and `src.evaluation.validate` are implemented. The
validator's `--report` saves an integrity report; the scorer calculates metrics,
bootstrap compares two runs, and the report command generates shared tables.

The package contains saved validation/test populations and configurations,
synthetic examples, the tested evaluation tools, current handoff documentation,
and portable lightweight result archives. The complete validation evidence is
available without rerunning model inference or accessing original target data.
Bootstrap/table generation can verify copied complete report bundles by their
byte hashes while retaining historical absolute source paths.

P1 validation NDCG@10 is **0.041927**, compared with B3 **0.014578**; the paired
difference is **+0.027349**, 95% interval **[0.018656, 0.036476]**. Observed P1
means remain below popularity/BM25; BM25 was tuned on the same validation users.
Recall@50 is unchanged because P1 preserves the candidate pool. There are
**218/300 pools with no relevant held-out movie**. No untuned-Qwen control was
evaluated, so the comparison does not isolate the effect of fine-tuning.

The [test archive](../../docs/results/member2_test_500_v1/README.md) contains
the original inference manifest and corrected receipt v2. All 500 users and
25,000 recommendations passed the official structural validator with zero
errors and exact B3 movie-ID sets. The source hashes are accounted for by
published commit `281cca11474b3bf680ec13444d57cd2df7420fd8`, including LF/CRLF
conversion for two files. No extra source handoff or rerun is required for that
issue. The model archive identity and label-access declaration remain owner
reported; the adapter archive was not independently hashed by Member 2.

The validation config retains `draft_for_team_review`. The separate inference
freeze did not itself authorize scoring. After the user relayed explicit team
approval to run and publish P1/B3 test results, the
[authorized comparison](../../docs/results/member2_p1_b3_test_500_v1/README.md)
was completed using frozen settings. P1 test NDCG@10 is **0.036421**, versus B3 **0.012429**: difference **+0.023992**, paired 95% interval **[0.016787, 0.031517]**. Both methods have 500 valid users and zero errors; Recall@50 is identical (**0.022180**).
No untuned-Qwen control was evaluated. Historical receipt/freeze JSON describes
the earlier pre-scoring stage.

Remaining work includes broader protocol review, owner explanation/model/cost
records, Phase 2, and the joint final comparison. The current scorer accepts
`popularity`, `bm25`, `embedding`, and `p1`; P2 needs an explicit protocol/scorer
extension. Completed test outcomes must not guide further model selection.
See the [current checklist](../../docs/member2_completion_checklist.md).
