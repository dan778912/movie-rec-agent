# Member 2: completed 500-user test-inference handoff

The handoff is confirmed, Member 6's inference is complete, and the returned P1 predictions passed the official structural validator with zero errors. **Test performance metrics remain uncomputed.** The [tracked test record](results/member2_test_500_v1/README.md) is the current source for confirmations, hashes, the received manifest, and the corrected receipt.

## Fixed inputs and output contract

- Dataset: `ml25m-global-v1`.
- Saved population: 500 users, seed 20261003, selected from 3,779 canonical test-profile IDs without opening targets or ratings.
- [User metadata](../configs/test_users_500_v1.json) and [plain ID array](../configs/test_users_500_export_v1/user_ids.json) contain the exact IDs. Consume these files without resampling.
- Plain-ID SHA-256: `7d0fced3b4e76ddbaf315883a249380c9966cd9b1c8971d355d69f413aa05b49`.
- Prepared B3-pool SHA-256: `44ec13348849458fa58746e126369afc5fa0c2cf4bbd867bc732b83f412b93a1`.
- Every output record has one requested user, `split: test`, `method: p1`, 50 unique unseen catalog movie IDs, ranks 1–50 in array order, finite scores, and exactly the supplied user's B3 movie-ID set.

Member 6 already received and verified the inference-input ZIP containing matching profiles, catalog documents, candidates, and saved IDs. No additional input upload or inference rerun is needed. The [archived prediction schema](results/member2_test_500_v1/preparation/prediction_schema.json) describes the record structure; the official validator also checks cross-record, watched-item, and exact candidate-set rules.

## Confirmations and received predictions

Member 5 confirmed the final B3 pool, as relayed by the user. Member 6 accepted the 500-user compute budget, identified the validated adapter, and confirmed the saved inputs and unchanged ranking settings. See the [freeze summary](member2_test_freeze_confirmation.md).

Returned file: `p1_test_predictions_500_v1.jsonl`, SHA-256 `15d5d332f61224824f339bdd73275522dec71a93ae05d0a460f9bb7fb1412634`. All 500 users and 25,000 recommendations passed the [official validation report](results/member2_test_500_v1/prediction_validation/validation.json). Both attachment checksums and all four inference-input hashes match. No labels or test quality metrics were used in these checks.

The two initially questioned source hashes match committed files exactly after LF-to-CRLF line-ending conversion. The [corrected provenance review](results/member2_test_500_v1/prediction_validation/code_provenance_review.json) resolves the issue; the earlier receipt ZIP v1 is superseded.

## Configuration and remaining evaluation

Use [evaluation_test_frozen_v1.json](../configs/evaluation_test_frozen_v1.json) for the later shared test scoring. The [original prepared configuration](../configs/evaluation_test_v1.json), saved metadata, and export provenance retain historical pending labels to preserve their recorded hashes. The later confirmation supersedes those labels for P1/B3 inference only.

The roadmap schedules final testing after the applicable model and evaluation decisions are frozen. P1 inference completion does not authorize opening targets or tuning P2 against test results. Member 2 will run the final comparison across the relevant frozen baselines, P1, and P2 when the team authorizes that stage.
