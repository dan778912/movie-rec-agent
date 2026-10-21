# Recommendation experiment results

Dataset: `ml25m-global-v1`. Evaluation: `eval-test-v1`. Split: `test`. Users: 500.

Configuration status: `frozen_for_p1_b3_test_inference`. A generated table does not establish team approval of this configuration.

## Mean recommendation metrics

Scores are macro means over the same users. Values below are rounded to six decimal places; CSV/JSON retain full precision.

| System | hit_rate_at_10 | ndcg_at_10 | recall_at_10 | recall_at_20 | recall_at_50 |
|---|---:|---:|---:|---:|---:|
| B3 (embedding) | 0.094000 | 0.012429 | 0.006732 | 0.010011 | 0.022180 |
| P1 (fine-tuned reranker) | 0.184000 | 0.036421 | 0.013464 | 0.016612 | 0.022180 |

Recall at the full candidate-pool size measures candidate coverage. P1 reorders its frozen B3 pool, so it cannot improve recall at that full size.

## Data and prediction coverage

Catalog coverage is in-catalog positive targets divided by all positive targets. Full-split coverage includes users outside the selected subset; prediction-user coverage describes submitted records.

| System | Full-split catalog coverage | Selected-user catalog coverage | Full-split positive / in-catalog / outside | Selected positive / in-catalog / outside | Prediction-user coverage |
|---|---:|---:|---|---|---:|
| B3 (embedding) | 0.677552 | 0.702830 | 120035 / 81330 / 38705 | 15264 / 10728 / 4536 | 1.000000 |
| P1 (fine-tuned reranker) | 0.677552 | 0.702830 | 120035 / 81330 / 38705 | 15264 / 10728 / 4536 | 1.000000 |

## Supplied paired bootstrap comparisons

Differences mean challenger minus baseline. Intervals are paired-user percentile intervals; no p-value or significance decision is produced.

| Baseline | Challenger | Metric | Difference | Interval | Confidence | Resamples | Seed |
|---|---|---|---:|---|---:|---:|---:|
| embedding | p1 | hit_rate_at_10 | +0.090000 | [0.060000, 0.118050] | 0.950000 | 10000 | 20261002 |
| embedding | p1 | ndcg_at_10 | +0.023992 | [0.016787, 0.031517] | 0.950000 | 10000 | 20261002 |
| embedding | p1 | recall_at_10 | +0.006733 | [0.000571, 0.012117] | 0.950000 | 10000 | 20261002 |
| embedding | p1 | recall_at_20 | +0.006601 | [0.000043, 0.013029] | 0.950000 | 10000 | 20261002 |
| embedding | p1 | recall_at_50 | +0.000000 | [0.000000, 0.000000] | 0.950000 | 10000 | 20261002 |

Methods without supplied results: popularity, bm25.
Missing methods have no placeholder scores. These tables report supplied results only.
