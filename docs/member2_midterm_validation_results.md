# Member 2: completed midterm baseline evaluation

> Historical baseline-only snapshot from October 20, 2026. P1 validation is now complete: use [the current P1 findings](member2_p1_validation_results.md) and [tracked four-method table](results/member2_validation_300_v1/midterm_validation_with_p1_v1/table.md). Any pending P1 steps below describe the earlier run, not current work. Test prediction validation is also complete; test scores remain sealed.

Completed locally on October 20, 2026. These are real MovieLens validation results, not synthetic fixtures.

## What is complete

1. Created a separate detached integration checkout from Member 2's published commit. The original Member 2 branch and main were not changed.
2. Imported exact published Member 3 and Member 4 source/configuration files; recorded their commits and per-file hashes.
3. Verified canonical catalog, movie documents, validation profiles, validation targets, and the data manifest. Verified all eight checksummed files in Member 5's validation package.
4. Ran the combined existing test suite: **218 tests passed** in 3.761 seconds.
5. Reproduced popularity and BM25 Top-100 predictions for the saved 300 users, using the frozen settings without retuning.
6. Validated the full embedding source covering 4,751 users, then selected the saved 300 users. Validated the complete Top-100 popularity/BM25 sources, then retained their Top-50 prefixes.
7. Scored all three methods with the same evaluator, users, ground truth, and 50-item output contract; generated the results table and paired BM25/embedding comparison.

All prepared predictions have exactly 300 users and 50 unique unseen catalog movies per user. All shared prediction checks passed.

## Results

Dataset `ml25m-global-v1`; validation split; user selection seed `20261001`; binary relevance >=4.0; macro averages over users. Recall denominators contain all in-catalog relevant held-out movies, including retrieval misses.

| Method | Recall@10 | NDCG@10 | Recall@50 | Users |
|---|---:|---:|---:|---:|
| Popularity (B0) | 0.024239 | 0.070249 | 0.084906 | 300 |
| BM25 (B2) | 0.023407 | 0.058131 | 0.058474 | 300 |
| Frozen MiniLM embeddings (B3) | 0.005026 | 0.014578 | 0.018921 | 300 |

Values are fractions, not percentages. Popularity has the highest observed means. A popularity-versus-BM25 uncertainty comparison was not run, so this table alone does not establish statistical superiority between them.

Embedding minus BM25 NDCG@10 is **-0.043553**, with paired 95% percentile interval **[-0.059626, -0.028922]**. The comparison uses 10,000 user resamples and seed `20261002`.

## Interpretation and limitations

- The frozen embedding retriever underperforms the simple baselines on these 300 validation users. This is an observed result, not evidence that semantic retrieval or fine-tuning generally fails.
- **218 of 300 embedding candidate pools contain no relevant held-out movie.** Reordering those pools cannot create a relevant recommendation for those users under this evaluation's labels.
- For the same users, ground truth, and exact frozen pools, P1 Recall@10 cannot exceed B3 Recall@50, **0.018921**. This bound is below both popularity and BM25 Recall@10 here. P1 can still improve over B3's current ordering and may improve NDCG@10; the completed P1 evaluation now measures those gains; see the current P1 findings linked above.
- BM25 selected its settings on these same 300 validation users. Report these scores as development results; they are not an independent final-test estimate.
- The configuration still says `draft_for_team_review`. Running the pipeline does not establish team approval. Keep that status visible until the team confirms the contract.
- Full-validation catalog-positive coverage is **85.00%**; selected-user coverage is **86.68%**. Prediction-user coverage is 100%. These quantities have different denominators.
- No test targets or real test metrics were used. No new model training or embedding inference was run.

## Historical baseline-only report paragraph

We implemented a shared evaluator and compared popularity, BM25, and frozen MiniLM embedding retrieval on the same 300 validation users. Popularity achieved NDCG@10 of 0.070249, BM25 0.058131, and embeddings 0.014578. Embeddings also recovered fewer held-out relevant movies in the 50-item candidate pool. This identifies candidate retrieval as an important limitation: a reranker can reorder available movies but cannot recover missing ones. These are preliminary development results; BM25 settings were selected on these validation users. We will next evaluate the fine-tuned P1 model over the exact frozen embedding pools and assess whether it improves ranking and explanation quality.

## Historical next steps at the baseline-only stage

1. Review the protocol and these preliminary findings with the team before labeling the comparison agreed/final.
2. Receive actual fine-tuned P1 validation predictions from Member 6, with model/configuration identity, candidate-pool identity, and available runtime information.
3. Validate P1's exact 50-item pool agreement, score P1, compute P1-minus-B3 paired intervals, and generate a new version of the table.
4. Incorporate the results into the team's two-slide presentation and evaluation/results section of its report. Member 3 owns the explanation audit; Member 6 supplies explanations/model evidence.

The prepared 300-user B3 pool is `outputs/predictions/prepared/b3_validation_v1/predictions.jsonl`. It is ready for Member 6 to consume, along with the canonical profiles/movie documents already supplied by Member 5. The shared-results ZIP includes this pool and the fixed user list, but no ratings, profile histories, movie-document data, or model weights.

## Historical local execution paths and provenance

The portable validation bundles and key execution provenance are now tracked in [the results archive](results/member2_validation_300_v1/README.md). The paths below identify the original run.

- Main table: `outputs/reports/midterm_validation_v1/table.md` and `table.csv`.
- Per-method reports: `outputs/reports/b0_validation_v1`, `b2_validation_v1`, and `b3_validation_v1`.
- Paired comparison: `outputs/reports/b3_vs_b2_validation_v1`.
- Exact import commits/hashes and package verification: `outputs/reports/workflow_midterm_v1/integration_setup/integration_sources.json`.
- Exact executed commands, return codes, and logs: `outputs/reports/workflow_midterm_v1/`.
- Shared data and Python environment are reused through symlinks to the original checkout. The new checkout is separate; imported files are local and uncommitted. Producer reports record actual source hashes, and the integration manifest identifies the source commits.

To open this checkout in Terminal:

```bash
cd "$HOME/Codex Projects/movie-rec-agent-midterm-evaluation"
source .venv/bin/activate
```

## Commands executed

These commands document the completed run. Saved evaluation bundles are immutable; to rerun after changing inputs, use new output directory names throughout rather than overwriting these results. Full paths in historical JSON records identify this local run; portable summary/per-user bundles can be read on another machine.

```bash
# generate_popularity
.venv/bin/python -m src.baselines.popularity --split validation --user-ids configs/validation_users_300_export_v1/user_ids.json --output outputs/predictions/incoming/b0_validation_300_top100.jsonl

# build_bm25_index
.venv/bin/python -m src.retrieval.run_bm25 index --config configs/bm25_frozen.json --output data/processed/bm25/index.json

# generate_bm25
.venv/bin/python -m src.retrieval.run_bm25 predict --split validation --config configs/bm25_frozen.json --index data/processed/bm25/index.json --user-ids configs/validation_users_300_export_v1/user_ids.json --output outputs/predictions/incoming/b2_validation_300_top100.jsonl --report-dir outputs/reports/producer_b2_validation_300_v1

# prepare_b0
.venv/bin/python -m src.evaluation.prepare predictions --config configs/evaluation_v1.json --source outputs/predictions/incoming/b0_validation_300_top100.jsonl --method popularity --source-count 100 --source-users selected --source-label member3-popularity-f6ddddb-reproduced --expected-source-sha256 320ee74711bcaeb309543921d044ed11522bb7ff4d7385d59c2da0b13f427795 --source-metadata outputs/predictions/incoming/b0_validation_300_top100.report.json --output-dir outputs/predictions/prepared/b0_validation_v1

# score_b0
.venv/bin/python -m src.evaluation.evaluate --config configs/evaluation_v1.json --predictions outputs/predictions/prepared/b0_validation_v1/predictions.jsonl --method popularity --output-dir outputs/reports/b0_validation_v1

# prepare_b2
.venv/bin/python -m src.evaluation.prepare predictions --config configs/evaluation_v1.json --source outputs/predictions/incoming/b2_validation_300_top100.jsonl --method bm25 --source-count 100 --source-users selected --source-label member4-bm25-5799b0d-frozen-reproduced --expected-source-sha256 549d20ad02a7bb19fe62814c14022bc5e80602b28cb01adbee9fa7171838e6e2 --source-metadata outputs/reports/producer_b2_validation_300_v1/summary.json --output-dir outputs/predictions/prepared/b2_validation_v1

# score_b2
.venv/bin/python -m src.evaluation.evaluate --config configs/evaluation_v1.json --predictions outputs/predictions/prepared/b2_validation_v1/predictions.jsonl --method bm25 --output-dir outputs/reports/b2_validation_v1

# prepare_b3
.venv/bin/python -m src.evaluation.prepare predictions --config configs/evaluation_v1.json --source outputs/predictions/incoming/member5-validation/validation_candidates_top50.jsonl --method embedding --source-count 50 --source-users all --source-label member5-original-data-frozen-minilm-l6-v1 --expected-source-sha256 0f61c4d87a73062400bfeea26371efaf33a25caf1cfd342e250de729be3706df --source-metadata outputs/predictions/incoming/member5-validation/validation_run.json --output-dir outputs/predictions/prepared/b3_validation_v1

# score_b3
.venv/bin/python -m src.evaluation.evaluate --config configs/evaluation_v1.json --predictions outputs/predictions/prepared/b3_validation_v1/predictions.jsonl --method embedding --output-dir outputs/reports/b3_validation_v1

# compare_embedding_vs_bm25
.venv/bin/python -m src.evaluation.bootstrap --baseline-dir outputs/reports/b2_validation_v1 --challenger-dir outputs/reports/b3_validation_v1 --output-dir outputs/reports/b3_vs_b2_validation_v1

# make_baseline_table
.venv/bin/python -m src.evaluation.report --runs outputs/reports/b0_validation_v1 outputs/reports/b2_validation_v1 outputs/reports/b3_validation_v1 --comparisons outputs/reports/b3_vs_b2_validation_v1 --kind experiment --output-dir outputs/reports/midterm_validation_v1
```
