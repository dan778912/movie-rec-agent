# Phase 2 B0/B2 test prediction receipt

**Status: label-blind prediction export, not a scored test result.** On 2026-11-06, Member 2 regenerated B0 popularity and B2 BM25 predictions for the saved 500 test users using the frozen Phase 1 producer branches. The files are Git-ignored local inputs for the eventual joint table. No test target or test rating file was passed to either producer or to the structural validator, and no B0/B2 test metric was computed in this step. Preserve the final Phase 2 scoring freeze before preparing top-50 records or reading test labels.

| Item | B0 popularity | B2 BM25 |
|---|---|---|
| Producer commit | `f6ddddb754023fe1130cef6f7f116414d7ca1886` on `origin/member-3-training-data-popularity` | `5799b0d71c2de357b87a631fa3a5fd1ddecee056` on `origin/member-4-bm25` |
| Frozen producer config | `configs/member3_v1.json`, `popularity_k=100` | `configs/bm25_frozen.json`, `k1=0.6`, `b=0.25`, full canonical text, `top_k=100` |
| Source prediction JSONL | `outputs/predictions/phase2_baselines/b0_test_500.jsonl` | `outputs/predictions/phase2_baselines/b2_test_500.jsonl` |
| Prediction SHA-256 | `5a7c4566f784ce5ac1ebb91b8aba1728409232afe0d235eaf2688301de8ac9e4` | `914a75284229acbedf1730320a2bd16360c15601d306019cba7f4a66f71bc6c7` |
| Structural result | 500/500 users, 100 ranks each, zero errors | 500/500 users, 100 ranks each, zero errors |

Both producers used `data/processed/ml25m-global-v1/` with manifest SHA-256 `f2e137ff374a8cfc02348cb42d04d9a04c9a1769685a7eccc21933383a500b5a`, catalog SHA-256 `bf39972971caf93923c01e89b2a2c32775435b8f00a087735b9d613d5488f45b`, and test profiles SHA-256 `e2a04d1df890b052a6d0c6baba73044f9738f1575b6679b33a233ed3f686001f`. The selected user file is `configs/test_users_500_export_v1/user_ids.json` (500 IDs); the authoritative validator used `configs/test_users_500_v1.json` (SHA-256 `0abfc86581184ce7953c2feefc3dadae8a7c43017a71c81326cb1f22daa93daf`). The B2 index SHA-256 is `77b787ae0db1b2351d2895330f76921411d16d925d071d71cf3be1f91dc73e95`.

The local evidence is in `outputs/predictions/phase2_baselines/` and `outputs/reports/phase2_baselines/`: B0's producer report, B2's producer summary/per-user report, and both authoritative structural validation reports. These large files are deliberately not committed. Regenerate them with the branch producer commands in [Member 3's baseline README](https://github.com/dan778912/movie-rec-agent/blob/f6ddddb/src/baselines/README.md) and [Member 4's BM25 README](https://github.com/dan778912/movie-rec-agent/blob/5799b0d/src/retrieval/README.md), supplying the saved 500-user ID array via `--user-ids` and the canonical processed data via `--data`. The B2 prediction command also needs the index built from its frozen config. Verify the regenerated SHA-256 values above; if any differ, version the new run separately.

The Phase 2 draft scorer requires 50 predictions per user, while both frozen baseline producers export 100. After the team freezes the shared Phase 2 evaluation config, use `src.evaluation.prepare predictions` to retain their top 50 under that config, then score the prepared files with the same evaluator revision and user/target set as B3, P1, and P2. Keep these original top-100 exports and hashes as the source receipt. Do not insert B0/B2 numbers into the final table until compatible scored bundles exist.
