# Member 2: completed P1 validation and B3 comparison

The official evaluation of Member 6's P1 validation predictions is complete. The [tracked report archive](results/member2_validation_300_v1/README.md) includes the complete per-method reports, paired comparisons, four-method table, and execution provenance.

## Integrity and population

- P1 prediction SHA-256: `b70ccd8ea374695c3bed24cd908caee65bffdfe96de7e30b693de990091c3ad2`, matching the supplied checksum.
- Prepared B3 validation pool SHA-256: `c388c0fd10bb895c1e0710ec10fa96a7d87785939f2498a5bf23533d18d000a9`.
- Exactly 300 frozen validation users, 50 valid unseen catalog movies per user, correct split/method/ranks, finite scores, and exact B3 candidate-set preservation.
- Zero validation errors or missing users. Input rankings and scores were not altered before evaluation.

## Results

| Method | Recall@10 | NDCG@10 | Recall@50 |
|---|---:|---:|---:|
| Popularity (B0) | 0.024239 | 0.070249 | 0.084906 |
| BM25 (B2) | 0.023407 | 0.058131 | 0.058474 |
| Frozen embeddings (B3) | 0.005026 | 0.014578 | 0.018921 |
| P1 reranker | 0.012708 | 0.041927 | 0.018921 |

Metrics are fractions and macro-averaged over the same users. Ground truth includes all in-catalog held-out positive targets, including retrieval misses; a positive rating is at least 4.0.

| P1 minus B3 | Difference | Paired 95% interval |
|---|---:|---|
| Recall@10 | +0.007682 | [0.004188, 0.011711] |
| NDCG@10 | +0.027349 | [0.018656, 0.036476] |
| HitRate@10 | +0.106667 | [0.066667, 0.146667] |
| Recall@50 | 0.000000 | [0.000000, 0.000000] |

The percentile intervals use 10,000 paired-user resamples and seed 20261002. Full precision and Recall@20 are in the [comparison JSON](results/member2_validation_300_v1/p1_vs_b3_validation_v1/comparison.json).

## Interpretation and limits

P1 improves the frozen B3 order on these validation users. Its observed scores remain below popularity and BM25. Its identical Recall@50 follows from preserving the exact candidate set. For 218 of 300 users, the B3 pool contains no positive held-out movie, so reordering alone cannot produce a positive recommendation.

BM25 was selected on these same validation users. These findings are development evidence, not final test results. An untuned-Qwen control was not evaluated, so this comparison does not isolate the effect of fine-tuning. The saved validation configuration retains `draft_for_team_review`; the scoped P1/B3 test-inference confirmation does not rewrite that historical configuration or establish full-project approval.

## Model and test records

The original validation prediction attachment did not include model/run metadata or explanations. Later, Member 6 identified the validated Qwen2.5-1.5B LoRA adapter and confirmed unchanged ranking settings for test inference. Those owner declarations, the received test manifest, and the completed structural checks are archived in the [500-user test record](results/member2_test_500_v1/README.md). Adapter identity is owner reported; the model archive has not been independently hashed by Member 2. Scores are ranking scores, not calibrated probabilities.

Member 3 owns the explanation-grounding audit, using explanation-enabled examples supplied by Member 6. Member 2's ranking evaluation does not establish explanation faithfulness.

## Reproduction and remaining work

The [archive instructions](results/member2_validation_300_v1/README.md#rebuild-the-table-without-raw-data-or-predictions) rebuild the comparison and table from portable saved reports without opening labels or requiring model inference. Original scoring and preparation commands are retained under the archive's provenance directory.

Member 2's Phase 1 implementation and validation comparison are complete. The report/presentation should use these four-method findings rather than the earlier baseline-only snapshot. Final test scoring and the P1/P2 comparison remain for the team's frozen final-evaluation stage; test scores remain sealed while model selection continues.
