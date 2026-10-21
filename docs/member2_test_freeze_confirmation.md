# Confirmed P1/B3 test-inference settings and current status

Member 5 confirmed the existing B3 test pool, as relayed by the user. Member 6 accepted the 500-user budget, verified the handoff ZIP, and identified the validated P1 adapter and unchanged ranking settings. Member 6 has now completed inference; Member 2 verified all 500 returned users and 25,000 recommendations with zero structural errors. The user later relayed explicit team approval to run and publish P1/B3 test results; the [500-user comparison](results/member2_p1_b3_test_500_v1/README.md) is now complete. Preserved receipt/freeze JSON describes the earlier pre-scoring stage.

## Preserved settings

- Test users: 500, seed 20261003, the exact saved IDs.
- Recommendations: 50 per user, preserving the supplied B3 movie-ID set exactly.
- Base model: `Qwen/Qwen2.5-1.5B-Instruct`.
- Base revision: `775b11afaf83e0dc75bd5abaf90133e47b3ec082`.
- Adapter archive SHA-256 reported by Member 6: `6d4a86f80f4f6f7bbe6f1b9c5df221f3847f12fce4afb99ea08afc4f1a32e392`.
- Maximum sequence length: 1024. Inference batch size: 2.
- Prompts and ranking method: unchanged from validation, as confirmed by Member 6.

The adapter archive was not supplied for independent hashing. The owner-reported no-label-access statement and GPU task runtime are recorded in the original manifest; they are not independently measured inference access or latency/cost evidence.

The [machine-readable owner confirmation](results/member2_test_500_v1/freeze_confirmation.json), [original inference manifest](results/member2_test_500_v1/p1_test_inference_manifest_500_v1.json), and [corrected validation receipt](results/member2_test_500_v1/prediction_validation/receipt.json) preserve the experiment record. All five code/configuration files in the manifest are accounted for by committed sources; two match after LF-to-CRLF conversion. No extra source handoff is needed.

## Configuration identity and scope

[evaluation_test_frozen_v1.json](../configs/evaluation_test_frozen_v1.json) differs from the [original prepared version](../configs/evaluation_test_v1.json) only in administrative `status` and `subset_status` fields. Saved IDs, metadata, input pools, profiles, documents, and metric settings are unchanged.

The historical preparation metadata and confirmation record retain their original bytes, including pending labels or the then-future inference step. The confirmation supersedes preparation-time labels for the P1/B3 inference scope, and the received prediction receipt records completion. It did not itself authorize scoring or certify other baseline/P2 decisions. The later explicit authorization and scoring are recorded in the separate P1/B3 comparison archive.

No test targets or ratings were opened during earlier selection, handoff, prediction validation, or initial receipt publication. Targets were later accessed for expressly authorized P1/B3 scoring using unchanged frozen settings. The broader joint comparison and P2 remain separate work under their applicable protocol. See the [inference-record index](results/member2_test_500_v1/README.md) for preserved history and the [P1/B3 test archive](results/member2_p1_b3_test_500_v1/README.md) for scored results.
