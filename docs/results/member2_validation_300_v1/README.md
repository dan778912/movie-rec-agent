# Member 2: completed 300-user validation experiment

These are real MovieLens development results, archived from the completed runs without changing their bytes. Start with the [four-method table](midterm_validation_with_p1_v1/table.md), [P1 summary](p1_validation_v1/summary.json), and [P1-versus-B3 comparison](p1_vs_b3_validation_v1/comparison.json).

## Findings

| Method | Recall@10 | NDCG@10 | Recall@50 |
|---|---:|---:|---:|
| Popularity (B0) | 0.024239 | 0.070249 | 0.084906 |
| BM25 (B2) | 0.023407 | 0.058131 | 0.058474 |
| Frozen embeddings (B3) | 0.005026 | 0.014578 | 0.018921 |
| P1 LoRA reranker | 0.012708 | 0.041927 | 0.018921 |

Values are fractions and macro-averaged over the identical 300 saved validation users. P1-minus-B3 NDCG@10 is +0.027349, with paired 95% percentile interval [0.018656, 0.036476], 10,000 resamples, seed 20261002. P1 improves B3's order but its observed scores remain below popularity and BM25. Recall@50 is unchanged because P1 preserves the pool. The B3 pool contains no positive held-out movie for 218 of 300 users.

These are validation findings. BM25 was selected on these same users. There is no evaluated untuned-Qwen control, so the difference does not isolate the contribution of fine-tuning. The saved configuration retains `draft_for_team_review`; archiving results does not change protocol approval or authorize test scoring.

## Complete portable bundles

Each of the four run directories includes `summary.json`, `per_user.csv`, and `validation.json`. Each comparison includes `comparison.json` and `per_user_differences.csv`. The table directory includes Markdown, CSV, differences, and its full report. These complete bundles can be loaded on another machine: original absolute paths identify the historical execution and are not rewritten. Source-byte hashes, rather than those historical locations, bind the reports together.

The provenance directory includes exact execution commands, the baseline import commits/source hashes, the recorded 218-test integration result, the original P1 input receipt, and prediction-preparation records. Member 3/4 source code remains on its owners' branches; it is not copied into Member 2's implementation by this publication.

## Rebuild the table without raw data or predictions

From the repository root, with Python 3.10+:

```bash
python -m src.evaluation.report \
  --runs docs/results/member2_validation_300_v1/b0_validation_v1 docs/results/member2_validation_300_v1/b2_validation_v1 docs/results/member2_validation_300_v1/b3_validation_v1 docs/results/member2_validation_300_v1/p1_validation_v1 \
  --comparisons docs/results/member2_validation_300_v1/b3_vs_b2_validation_v1 docs/results/member2_validation_300_v1/p1_vs_b3_validation_v1 \
  --kind experiment \
  --output-dir outputs/reports/reproduced_validation_table_v1
```

The command reads only saved report bundles. To independently recalculate the P1-versus-B3 bootstrap, use:

```bash
python -m src.evaluation.bootstrap \
  --baseline-dir docs/results/member2_validation_300_v1/b3_validation_v1 \
  --challenger-dir docs/results/member2_validation_300_v1/p1_validation_v1 \
  --output-dir outputs/reports/reproduced_p1_vs_b3_validation_v1
```

Use new output directories if different bytes already exist. Rerunning the original scorers requires canonical data and the exact owner prediction files, supplied through team storage; those files are identified by hashes in the archived summaries and are not included here.

## Integrity

From this directory, run `shasum -a 256 -c SHA256SUMS.txt`. The archive manifest lists every preserved source file and its original integration-checkout path. No test labels, test performance metrics, raw ratings, profiles, movie documents, model weights, or raw prediction exports are included.
