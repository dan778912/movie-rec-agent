# Member 2 handoff — Phase 1

The real MovieLens 25M dataset and a fixed 300-user validation subset have been
built and verified locally. Metric helpers, prediction validation, and subset
selection are implemented, together with end-to-end scoring, per-user outputs,
and summary reports, paired bootstrap comparisons, and automatic Markdown/CSV
tables, plus explicit teammate-export preparation and a plain user-ID export.
Actual B0/B2/B3/P1 validation is complete on the same 300 saved users. The
[portable validation archive](results/member2_validation_300_v1/README.md)
contains the complete reports, paired comparisons, final table, and provenance.
The [scoring walkthrough](member2_scoring_walkthrough.md) remains a
hand-calculated implementation example.

This package contains the evaluation code, saved populations/configurations,
documentation, and small result archives for `member2-evaluation`. The group's
Git/PR process determines merge status. Large raw/processed data, models, and
prediction files remain ignored; small reviewed reports are preserved under
`docs/results/` so teammates can use the findings without regenerating models.

## Current findings and test receipt

P1 validation NDCG@10 is **0.041927**, versus B3 **0.014578**, with paired
difference **+0.027349** and 95% interval **[0.018656, 0.036476]**. P1 remains
below popularity/BM25 on the observed validation means. Recall@50 is unchanged
because the pools are fixed; 218/300 B3 pools contain no relevant target.
These validation findings do not establish final-test performance or isolate
fine-tuning from an untuned reranker control.

The [500-user test archive](results/member2_test_500_v1/README.md) preserves
the confirmed P1/B3 inference handoff and corrected receipt v2. Member 6 has
returned the predictions and manifest: all 500 users and 25,000 recommendations
passed the official structural validator with zero errors, exact candidate
sets, and verified input/file hashes. The source provenance is resolved in
published commit `281cca11474b3bf680ec13444d57cd2df7420fd8`; two manifest hashes
correspond to CRLF versions of committed LF files. No extra source handoff or
inference rerun is needed for that issue.

Test scores remain sealed pending explicit team authorization and the final
model/evaluation freeze. The scoped owner confirmation does not approve the
whole draft protocol or other methods/P2. The test archive contains no test
performance results. See the [current checklist](member2_completion_checklist.md)
for the remaining responsibilities.

## 1. Use the same data and user IDs

Use dataset version `ml25m-global-v1` and these repository-relative inputs:

| Input | Path |
|---|---|
| Movie catalog | `data/processed/ml25m-global-v1/catalog.csv` |
| Validation profiles and full watched histories | `data/processed/ml25m-global-v1/validation_profiles.jsonl` |
| Held-out validation positives | `data/processed/ml25m-global-v1/validation_targets.csv` |
| Data provenance | `data/processed/ml25m-global-v1/manifest.json` |
| Shared 300 users | [configs/validation_users_300_v1.json](../configs/validation_users_300_v1.json) |
| Settings and artifact hashes | [configs/evaluation_v1.json](../configs/evaluation_v1.json) |

The verified data contains 25,000,095 source ratings, 21,333 catalog movies,
and 4,751 evaluable validation users. The saved subset selects 300 of those
users with seed `20261001`. Every selected user has at least 17,862 unseen
catalog movies, enough for the proposed 50-item output.

All ten processed-file checksums and Member 1's history/temporal integrity checks
passed. Full-validation catalog-positive coverage is 126,029 / 148,263 =
85.0037%. This describes which held-out positives the catalog can represent;
it is not recommendation recall or a model result.

On Priyanka's Mac, activate the existing environment from Terminal:

```bash
cd "$HOME/Codex Projects/movie-rec-agent"
source .venv/bin/activate
```

Run subsequent commands from the repository root. A teammate without processed
data first runs this one-time preparation command in their own checkout:

```bash
python -m src.data.prepare
```

Priyanka's existing data is ready to reuse. To check the saved subset against
matching data and the recorded Python version, run:

```bash
python -m src.evaluation.subsets \
  --data-dir data/processed/ml25m-global-v1 \
  --split validation \
  --size 300 \
  --seed 20261001 \
  --output configs/validation_users_300_v1.json
```

Expected: `status: "created_or_verified"`, `selected_users: 300`, and
`eligible_users: 4751`. An identical existing file is preserved. A differing
payload is rejected. The recorded runtime is Python 3.12.10; runtime metadata
can prevent byte-identical regeneration on another Python version. Teammates
should use the saved `user_ids` list directly rather than independently sampling
or replacing it. The validator also binds that list to catalog/profile hashes.

## 2. Agree on the proposed prediction contract

The [evaluation protocol](evaluation_protocol.md) is a **draft for team review**.
Generating the subset has not established team approval. Proposed Phase 1 rules:

- Exactly one JSONL record for each of the 300 selected users, with no extra users.
- Fields: `user_id`, `split`, `method`, and `recommendations`.
- Use `split: "validation"`; methods are `popularity`, `bm25`, `embedding`, or `p1`.
- Each recommendation contains integer `movie_id`, finite numeric `score`, and
  integer `rank`. Export exactly 50 distinct catalog movies per user, excluding
  every movie in the profile's complete `history_movie_ids`.
- Array order follows consecutive ranks 1 through 50. Rank determines order;
  scores may be negative or tied. Document `score = -rank` as a rank surrogate
  if used by a listwise reranker.
- P1 returns exactly the same 50 movie IDs as that user's frozen B3 pool, in
  reranked order. Save the original embedding file and its hash before reranking.

Member 3 exports popularity predictions and prepares training examples to Member
6's requirements. Member 4 exports BM25 predictions. Member 5 exports and freezes
embedding candidates. Member 6 owns fine-tuning and exports P1 predictions.
Training supervision must come from training time; validation/test targets used
for evaluation must not become training labels.

## 3. Obtain, prepare, and validate actual exports

The completed validation run used Members 3/4's reproduced baselines, Member
5's authoritative frozen MiniLM-L6 pool, and Member 6's supplied P1 predictions.
The owners' implementations and small reports live on their branches. Large
prediction files are Git-ignored; fetching alone does not download them. For a
new run, obtain the exact files and owner metadata through agreed storage, then
follow the [integration guide](member2_integration.md).

The preparation command validates every source user and recommendation before
selecting the frozen users and prefix. It preserves original files and records
the transformation. The companion
[plain user-ID array](../configs/validation_users_300_export_v1/user_ids.json)
works with Members 3/4's existing array-only `--user-ids` loaders. The original
metadata-rich user file remains the evaluation input.

For P1, confirm the authoritative frozen embedding model/pool with Members 5/6.
The reviewed Member 5 report retains original L6 as frozen; exploratory Qwen and
chunked runs do not automatically replace it. BM25's 300 tuning IDs equal our
shared subset, so describe current scores as validation-development evidence.

Validate Member 5's embedding output:

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

Validate Member 6's P1 export against that frozen embedding pool:

```bash
python -m src.evaluation.validate \
  --catalog data/processed/ml25m-global-v1/catalog.csv \
  --profiles data/processed/ml25m-global-v1/validation_profiles.jsonl \
  --users configs/validation_users_300_v1.json \
  --predictions outputs/predictions/prepared/p1_validation_v1/predictions.jsonl \
  --split validation \
  --method p1 \
  --required-count 50 \
  --candidate-pool outputs/predictions/prepared/b3_validation_v1/predictions.jsonl \
  --report outputs/reports/validation_checks/p1_validation.json
```

Expected for each: `status: "valid"`, 300 valid users, prediction-user coverage
1.0, and zero errors. Exit 0 means valid; exit 1 means invalid predictions;
exit 2 means invalid arguments or reference data. Correct rejected exports rather
than dropping failed users. The evaluation configuration currently records
settings/provenance and is accepted by the scorer. The standalone validator does
not accept a `--config` argument.

## 4. Score validated exports

The scorer runs validation itself and then loads the correct target data.
The following validation commands document the workflow used for the completed
experiment. A new checkout needs the matching local prediction artifacts:

```bash
python -m src.evaluation.evaluate \
  --config configs/evaluation_v1.json \
  --predictions outputs/predictions/prepared/b3_validation_v1/predictions.jsonl \
  --method embedding \
  --output-dir outputs/reports/b3_validation_v1
```

For a prepared P1 export and its exact frozen embedding pool:

```bash
python -m src.evaluation.evaluate \
  --config configs/evaluation_v1.json \
  --predictions outputs/predictions/prepared/p1_validation_v1/predictions.jsonl \
  --method p1 \
  --candidate-pool outputs/predictions/prepared/b3_validation_v1/predictions.jsonl \
  --output-dir outputs/reports/p1_validation_v1
```

Each directory receives `summary.json`, `per_user.csv`, and `validation.json`.
The summary contains Recall@10, NDCG@10, HitRate@10, Recall@20/50, and full-split
versus selected-user catalog-positive coverage. It records the input hashes and
configuration status. Repeating identical inputs is accepted; changed results
require a new output directory. Use the per-user CSV for paired comparisons.

The actual P1 validation run scored the original received file directly; its
exact command is preserved in the validation archive. The generic prepared-file
example above is also supported. New inputs require new versioned output
directories. Synthetic walkthroughs establish implementation behavior only.

## 5. Compare matching scored runs

To reproduce the saved paired comparison from complete validation report bundles:

```bash
python -m src.evaluation.bootstrap \
  --baseline-dir outputs/reports/b3_validation_v1 \
  --challenger-dir outputs/reports/p1_validation_v1 \
  --output-dir outputs/reports/p1_vs_b3_validation_v1
```

The difference is P1 minus B3. The command requires the same users, data,
configuration, truth counts, metric definitions, and scoring implementation. It
also verifies that the P1 pool hash matches the B3 prediction hash. It reads the
three report files per run and can compare bundles without original data paths.
Keep the full bundles together; it checks report consistency without rerunning
models or rescoring original predictions.

The command uses the matching configurations' proposed 10,000 paired user
resamples, 95% confidence, and seed `20261002`. It saves `comparison.json` and
`per_user_differences.csv`. Linear-interpolated percentile intervals describe
user-resampling uncertainty. An interval spanning zero leaves the difference's
direction uncertain under this procedure.

The [bootstrap walkthrough](member2_bootstrap_walkthrough.md) contains a runnable
synthetic comparison and hand-calculated expected values. Synthetic intervals
cannot replace actual model evidence in the midterm report.

## 6. Generate the shared results table

The completed four-method table and both comparisons are archived. For a new
table using locally generated bundles, supply their corresponding directories:

```bash
python -m src.evaluation.report \
  --runs outputs/reports/b0_validation_v1 outputs/reports/b2_validation_v1 outputs/reports/b3_validation_v1 outputs/reports/p1_validation_v1 \
  --comparisons outputs/reports/b3_vs_b2_validation_v1 outputs/reports/p1_vs_b3_validation_v1 \
  --kind experiment \
  --output-dir outputs/reports/midterm_validation_with_p1_v1
```

Only supply actual scored methods. If some are unavailable, omit their run
directories; the generator lists missing methods without inventing scores.
`--comparisons` is optional until a paired comparison is available. Including
P1 requires its embedding counterpart, even when no comparison is supplied.

The generator saves `table.md`, `table.csv`, `differences.csv`, and `report.json`.
It checks run identity and verifies supplied paired-comparison data before
including intervals. Markdown displays six decimal places; CSV/JSON preserve
numerical precision. Selected-user catalog coverage, full-split catalog coverage,
and prediction-user coverage have separate labels. Draft protocol status stays
visible. The [table walkthrough](member2_table_walkthrough.md) contains a runnable
synthetic example and explains each output.

## 7. Verify the package and complete the integration

```bash
python -m unittest discover -s tests -v
```

Expected: all tests pass with `OK`. The tests use synthetic fixtures and do not
download data. Passing them establishes implementation checks, not model
performance.

The core Phase 1 tooling and the four-method validation analysis are complete.
The [portable archive](results/member2_validation_300_v1/README.md) supplies
reviewable midterm evidence. Remaining work is team review of the findings and
full protocol, model/explanation records from their owners, later P2 evaluation,
and the authorized frozen final-test comparison. See the
[completion checklist](member2_completion_checklist.md).

Full 50-item pool equality means B3 and P1 have identical Recall@50 by
construction; their top-ten ranking metrics can differ. Keep final-test scores
out of development decisions. Phase 2 evaluation remains later work.
