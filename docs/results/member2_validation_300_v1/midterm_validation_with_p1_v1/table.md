# Recommendation experiment results

Dataset: `ml25m-global-v1`. Evaluation: `eval-v1`. Split: `validation`. Users: 300.

Configuration status: `draft_for_team_review`. A generated table does not establish team approval of this configuration.

## Mean recommendation metrics

Scores are macro means over the same users. Values below are rounded to six decimal places; CSV/JSON retain full precision.

| System | hit_rate_at_10 | ndcg_at_10 | recall_at_10 | recall_at_20 | recall_at_50 |
|---|---:|---:|---:|---:|---:|
| B0 (popularity) | 0.296667 | 0.070249 | 0.024239 | 0.042872 | 0.084906 |
| B2 (BM25) | 0.260000 | 0.058131 | 0.023407 | 0.032248 | 0.058474 |
| B3 (embedding) | 0.110000 | 0.014578 | 0.005026 | 0.008987 | 0.018921 |
| P1 (fine-tuned reranker) | 0.216667 | 0.041927 | 0.012708 | 0.015212 | 0.018921 |

Recall at the full candidate-pool size measures candidate coverage. P1 reorders its frozen B3 pool, so it cannot improve recall at that full size.

## Data and prediction coverage

Catalog coverage is in-catalog positive targets divided by all positive targets. Full-split coverage includes users outside the selected subset; prediction-user coverage describes submitted records.

| System | Full-split catalog coverage | Selected-user catalog coverage | Full-split positive / in-catalog / outside | Selected positive / in-catalog / outside | Prediction-user coverage |
|---|---:|---:|---|---|---:|
| B0 (popularity) | 0.850037 | 0.866809 | 148263 / 126029 / 22234 | 9355 / 8109 / 1246 | 1.000000 |
| B2 (BM25) | 0.850037 | 0.866809 | 148263 / 126029 / 22234 | 9355 / 8109 / 1246 | 1.000000 |
| B3 (embedding) | 0.850037 | 0.866809 | 148263 / 126029 / 22234 | 9355 / 8109 / 1246 | 1.000000 |
| P1 (fine-tuned reranker) | 0.850037 | 0.866809 | 148263 / 126029 / 22234 | 9355 / 8109 / 1246 | 1.000000 |

## Supplied paired bootstrap comparisons

Differences mean challenger minus baseline. Intervals are paired-user percentile intervals; no p-value or significance decision is produced.

| Baseline | Challenger | Metric | Difference | Interval | Confidence | Resamples | Seed |
|---|---|---|---:|---|---:|---:|---:|
| bm25 | embedding | hit_rate_at_10 | -0.150000 | [-0.200000, -0.103333] | 0.950000 | 10000 | 20261002 |
| bm25 | embedding | ndcg_at_10 | -0.043553 | [-0.059626, -0.028922] | 0.950000 | 10000 | 20261002 |
| bm25 | embedding | recall_at_10 | -0.018381 | [-0.030182, -0.009094] | 0.950000 | 10000 | 20261002 |
| bm25 | embedding | recall_at_20 | -0.023262 | [-0.035490, -0.013287] | 0.950000 | 10000 | 20261002 |
| bm25 | embedding | recall_at_50 | -0.039553 | [-0.053169, -0.027610] | 0.950000 | 10000 | 20261002 |
| embedding | p1 | hit_rate_at_10 | +0.106667 | [0.066667, 0.146667] | 0.950000 | 10000 | 20261002 |
| embedding | p1 | ndcg_at_10 | +0.027349 | [0.018656, 0.036476] | 0.950000 | 10000 | 20261002 |
| embedding | p1 | recall_at_10 | +0.007682 | [0.004188, 0.011711] | 0.950000 | 10000 | 20261002 |
| embedding | p1 | recall_at_20 | +0.006226 | [0.002399, 0.010294] | 0.950000 | 10000 | 20261002 |
| embedding | p1 | recall_at_50 | +0.000000 | [0.000000, 0.000000] | 0.950000 | 10000 | 20261002 |

Methods without supplied results: none.
Missing methods have no placeholder scores. These tables report supplied results only.
