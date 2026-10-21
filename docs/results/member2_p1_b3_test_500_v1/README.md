# Authorized P1 versus B3 test results: 500 users

Completed October 21, 2026 after the user explicitly relayed the team's approval to **run and publish P1/B3 test scoring now**. The [authorization record](provenance/scoring_authorization.json) records that decision and its scope. This is the frozen P1/B3 comparison, not the later full baseline/P2 comparison.

Start with the [generated test table](p1_b3_test_table_500_v1/table.md), [P1 metric summary](p1_test_500_v1/summary.json), and [paired comparison](p1_vs_b3_test_500_v1/comparison.json).

## Results

| Metric | Frozen B3 order | P1 reranking | P1 minus B3 | Paired 95% interval |
|---|---:|---:|---:|---|
| NDCG@10 | 0.012429 | 0.036421 | +0.023992 | [0.016787, 0.031517] |
| Recall@10 | 0.006732 | 0.013464 | +0.006733 | [0.000571, 0.012117] |
| HitRate@10 | 0.094000 | 0.184000 | +0.090000 | [0.060000, 0.118050] |
| Recall@20 | 0.010011 | 0.016612 | +0.006601 | [0.000043, 0.013029] |
| Recall@50 | 0.022180 | 0.022180 | +0.000000 | [0.000000, 0.000000] |

All values are fractions and macro-averaged over the identical 500 saved test users. A positive target is a rating of at least 4.0. Recall denominators include all distinct in-catalog positive targets, including those absent from the retrieved pool. Confidence intervals use 10,000 paired-user percentile bootstrap resamples, seed 20261002, confidence 95%; differences mean P1 minus B3.

## Takeaway and limits

P1 improves ordering within the frozen B3 candidate pools on this test subset. NDCG@10 increases from 0.012429 to 0.036421, with improvement +0.023992 and interval [0.016787, 0.031517]. Top-10 hit users increase from 47/500 to 92/500.

Recall@50 is identical for every user because the 50 movie IDs are unchanged. **380 of 500 users (76%) have no relevant held-out movie in their B3 pool**, leaving no relevant movie for P1 to promote. Retrieval remains a limitation.

Popularity and BM25 have not been scored on this test subset in this run; P2 was not evaluated here. The results do not establish superiority to those systems or isolate fine-tuning from an untuned-Qwen control. The [300-user validation findings](../member2_validation_300_v1/README.md) are a separate experiment with different users/temporal targets; do not mix their means into this test table.

## Integrity and frozen settings

- Both official validators report zero errors, exactly 500 users, 50 recommendations per user, and no missing/extra users.
- The P1 candidate-pool hash equals the B3 prediction hash, and the exact per-user movie-ID sets are preserved.
- Both original prediction files were copied byte-for-byte into the scoring checkout, without resampling, reranking, truncation, or reserialization.
- Config: [evaluation_test_frozen_v1.json](../../../configs/evaluation_test_frozen_v1.json), SHA-256 `a28d28ce9636d5e1220ba0af2c3a8e8b4e79b60b0444eb6b1186fe81febfa44b`.
- Saved users: [test_users_500_v1.json](../../../configs/test_users_500_v1.json), count 500, selection seed 20261003; its SHA-256 remains `0abfc86581184ce7953c2feefc3dadae8a7c43017a71c81326cb1f22daa93daf`.
- B3 predictions SHA-256: `44ec13348849458fa58746e126369afc5fa0c2cf4bbd867bc732b83f412b93a1`.
- P1 predictions SHA-256: `15d5d332f61224824f339bdd73275522dec71a93ae05d0a460f9bb7fb1412634`.
- The scorer opened and verified the authorized test targets against the canonical manifest: SHA-256 `ba2c10aaa7bef7dfb9d944a16466314bd47f12ca4bc99147fc38bfcd8fb94f93`.
- Scoring implementation commit: `df164278516cd4664cc772b6518ba6d77478d7c9`, clean working tree at both scoring runs.

The configuration and saved selection retain their historical administrative status fields and original bytes. The separate authorization record permits this scoring run without rewriting frozen inputs. Earlier [pre-scoring receipt/freeze records](../member2_test_500_v1/README.md) remain historical evidence; their no-label-access statements describe those earlier steps, not this authorized scoring run.

The unchanged model identity and input manifest are in the [test inference record](../member2_test_500_v1/p1_test_inference_manifest_500_v1.json). Model archive identity, inference no-label-access declaration, and GPU task runtime remain owner reported. The evaluator's command runtimes are evaluation overhead, not model inference latency or cost.

## Portable report bundles

Each method directory contains `summary.json`, `per_user.csv`, and `validation.json`. The paired comparison contains its JSON and per-user differences; the table includes all four generated files. These original report bytes are preserved. Historical absolute source paths identify execution inputs; hashes bind relocated bundles.

From the repository root, rebuild the test table from the tracked reports without opening targets or running inference:

```bash
python -m src.evaluation.report \
  --runs docs/results/member2_p1_b3_test_500_v1/b3_test_500_v1 docs/results/member2_p1_b3_test_500_v1/p1_test_500_v1 \
  --comparisons docs/results/member2_p1_b3_test_500_v1/p1_vs_b3_test_500_v1 \
  --kind experiment \
  --output-dir outputs/reports/reproduced_p1_b3_test_table_500_v1
```

Use a new output directory for changed inputs. Exact scoring, comparison, and table commands with exit codes and elapsed times are in [the execution record](provenance/commands.json). To reproduce scoring, obtain the same two prediction files from team storage and reproduce the canonical dataset; no raw prediction or target files are committed here.

## Coverage and later work

Full-test catalog-positive coverage is 0.677552; selected-user catalog-positive coverage is 0.702830; prediction-user coverage is 1.0. These use different denominators and are not recommendation recall.

The authorized P1/B3 evaluation and publication are complete. Later baseline/P2 comparisons remain separate work requiring their frozen outputs and an agreed protocol. P2 evaluation needs explicit method/protocol support; the current evaluator accepts popularity, BM25, embedding, and P1. This authorization does not approve tuning models from test outcomes. Record this test-result release chronology in any later P2 experiment.

From this archive directory, run `shasum -a 256 -c SHA256SUMS.txt` to verify the tracked records. The archive manifest identifies every preserved report and its source hash.
