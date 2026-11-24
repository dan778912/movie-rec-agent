# Member 2: untuned Qwen validation control

These are scored results for the corrected `qwen_base` export and its frozen B3
candidate pool on the same 300 saved validation users. The scoring configuration
still says `draft_for_phase2_team_review`; these are validation findings, not
final test results or a selected P2 checkpoint.

| Method | NDCG@10 | Recall@10 | HitRate@10 | Recall@50 |
|---|---:|---:|---:|---:|
| B3 ordered pool | 0.014577808 | 0.005026270 | 0.110000000 | 0.018920688 |
| Untuned Qwen2.5-1.5B-Instruct reranker | 0.019771560 | 0.007733067 | 0.150000000 | 0.018920688 |

Both prediction files passed the authoritative validator for all 300 users with
zero errors. The Qwen output is a permutation of each user's exact frozen B3
50-item pool. The identical Recall@50 follows from that fixed pool. Macro means
were recomputed from the archived `per_user.csv` files.

The [paired comparison](qwen_base_vs_b3/comparison.json) gives Qwen minus B3
NDCG@10 `+0.005194`, with a paired 95% percentile interval
`[-0.000675, 0.011195]` from 10,000 user resamples (seed `20261002`). This
interval includes zero.

## Provenance

| Input | SHA-256 |
|---|---|
| Ordered B3 `validation_candidates_300.jsonl` | `c388c0fd10bb895c1e0710ec10fa96a7d87785939f2498a5bf23533d18d000a9` |
| Corrected `qwen_base_validation_predictions_300_v1.jsonl` | `397d21f290bd688fb5bf3830518bf5f59be9654793ccac0809b471b30798087f` |
| Scoring config `configs/evaluation_phase2_validation_draft_v1.json` | `8d61a89d16457b141579e28b6ba57b49ebd73ce3d9f27f2c849ada22eb44b9b7` |

Member 6 supplied the [corrected untuned export](https://discord.com/channels/1544775300351660062/1544775300817092680/1573144980153966814)
and the [ordered B3 pool](https://discord.com/channels/1544775300351660062/1544775300817092680/1573148692008009769).
Their downloaded bytes matched the posted SHA-256 values. Member 6 identified
the base model as `Qwen/Qwen2.5-1.5B-Instruct`, revision
`989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, without a LoRA adapter. The
inference manifest has not been independently archived here.

The report bundles contain `summary.json`, `per_user.csv`, and `validation.json`;
the comparison contains `comparison.json` and `per_user_differences.csv`.
They preserve the evaluator output bytes, including historical absolute paths.
The summaries record the canonical data hashes, evaluator source hashes, and
clean Git commit `9032fd13c62b7b18d3c53a2e3c685c53f4f49b55`. Raw
prediction files and MovieLens data remain outside Git.
`SHA256SUMS.txt` records the bytes of every file in this archive other than
itself.

## Reproduce the scoring

Place the two exact input files at the paths below and prepare the canonical
`data/processed/ml25m-global-v1/` dataset. From the repository root:

```powershell
python -m src.evaluation.evaluate --config configs/evaluation_phase2_validation_draft_v1.json --predictions outputs/predictions/phase2_validation/validation_candidates_300.jsonl --method embedding --output-dir outputs/reports/phase2_validation/b3_reproduced
python -m src.evaluation.evaluate --config configs/evaluation_phase2_validation_draft_v1.json --predictions outputs/predictions/phase2_validation/qwen_base_validation_predictions_300_v1.jsonl --method qwen_base --candidate-pool outputs/predictions/phase2_validation/validation_candidates_300.jsonl --output-dir outputs/reports/phase2_validation/qwen_base_reproduced
python -m src.evaluation.bootstrap --baseline-dir outputs/reports/phase2_validation/b3_reproduced --challenger-dir outputs/reports/phase2_validation/qwen_base_reproduced --output-dir outputs/reports/phase2_validation/qwen_base_vs_b3_reproduced
```

The original P1 validation prediction JSONL is still needed to rescore P1 with
this exact config and evaluator. The [historical P1 report](../member2_validation_300_v1/p1_validation_v1/summary.json)
uses an earlier evaluation version, so the strict paired comparison tool
correctly rejects a direct comparison of the archived bundles. Do not report a
P1-versus-untuned confidence interval as an authoritative current result until
the original P1 predictions are recovered and rescored.
