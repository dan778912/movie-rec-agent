# Member 2: frozen 500-user test record

Current status: P1 inference is complete and its prediction file passed structural validation. **Test performance scores remain sealed.** This record does not authorize final scoring or certify that all baseline/P2 decisions are frozen.

## Authoritative current records

- [Frozen scoring settings](../../../configs/evaluation_test_frozen_v1.json): the confirmed P1/B3 inference configuration, preserving the agreed metric settings.
- [Saved user selection](../../../configs/test_users_500_v1.json) and [plain IDs](../../../configs/test_users_500_export_v1/user_ids.json): 500 users, seed 20261003; consume the saved IDs without resampling.
- [Owner freeze confirmation](freeze_confirmation.json): scope and model/settings identifiers reported by the owners.
- [Official prediction validator report](prediction_validation/validation.json): 500 users, 25,000 recommendations, zero errors, exact frozen B3 candidate sets.
- [Corrected receipt](prediction_validation/receipt.json) and [code provenance review v2](prediction_validation/code_provenance_review.json).
- [Original Member 6 inference manifest](p1_test_inference_manifest_500_v1.json), preserved byte-for-byte.

The preserved receipt README describes the contents of the original receipt ZIP v2. In this tracked archive, the inference manifest is at the archive root rather than inside `prediction_validation/`; the draft chat reply is not included. The report and receipt bytes remain unchanged.

## Verified handoff identities

| Artifact | SHA-256 |
|---|---|
| Plain saved 500-user IDs | `7d0fced3b4e76ddbaf315883a249380c9966cd9b1c8971d355d69f413aa05b49` |
| Prepared frozen B3 pool | `44ec13348849458fa58746e126369afc5fa0c2cf4bbd867bc732b83f412b93a1` |
| P1 test predictions | `15d5d332f61224824f339bdd73275522dec71a93ae05d0a460f9bb7fb1412634` |
| Original inference manifest | `2b60261393bbddca8ff81b7cd0fb3cae6547c6d4705e384caebe1b4146fa15cd` |
| Previously shared inference-input ZIP | `2c45072e5794cb3edba0451245a61e5515f09693ff1a1dbef77a6daac795259b` |

Both supplied checksums and all four actual inference-input hashes were verified by Member 2. The model identifiers match the earlier owner declaration, but Member 2 was not supplied the adapter archive for independent hashing. Member 6's no-test-label-access statement and GPU runtime are owner reported; task runtime is not independently measured latency or monetary cost.

## Resolved source-code question

All five source/configuration files listed in the inference manifest are accounted for by Member 6's published commit `281cca11474b3bf680ec13444d57cd2df7420fd8`. The scorer and reranker hashes match exactly after converting committed LF line endings to CRLF. The other three files match directly. No further code handoff or inference rerun is needed for this issue. The earlier missing-source concern and receipt ZIP v1 are superseded by this corrected v2 record.

## Historical preparation records

The preparation subdirectory, original `configs/evaluation_test_v1.json`, saved selection metadata, and user-export provenance preserve their preparation-time pending labels and original bytes. The later freeze confirmation supersedes those labels for the P1/B3 inference scope. Its historical next-step text was written before predictions arrived; the current status is the completed validation described above. Do not edit those hashed historical files to update status.

No test targets or test ratings were opened during selection, handoff preparation, prediction validation, or publication. The expected target hash in the user metadata was copied from the canonical manifest; it was not verified by opening labels. The test prediction file and inference-input data remain in team storage rather than Git.

## Remaining work

Use validation results for ongoing development. After the team freezes all relevant final settings and authorizes the roadmap's final evaluation stage, Member 2 will score the available final systems (popularity, BM25, embeddings, P1, and P2) on the identical saved users, compute the appropriate paired intervals, and generate the final table. No test outcome should guide model selection.

From this directory, run `shasum -a 256 -c SHA256SUMS.txt` to verify archived records. This reads only this documentation archive, not the test targets or test ratings.
