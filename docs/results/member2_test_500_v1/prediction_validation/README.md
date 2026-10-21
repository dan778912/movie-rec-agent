# P1 test inference receipt: corrected version 2

Reviewed October 21, 2026. This version supersedes the earlier receipt's source-documentation concern. No test targets or test ratings were read, and no test ranking metrics were computed.

## Verified by Member 2

- Both supplied file checksums match exactly.
- All 500 frozen users appear exactly once, with 50 recommendations each (25,000 total).
- Method `p1`, split `test`, ordered ranks 1–50, finite scores, and each user's exact frozen B3 movie-ID set all pass the official validator with zero errors.
- All four inference-input hashes match the saved handoff.
- All five source/configuration files listed in the manifest are accounted for in published commit `281cca11474b3bf680ec13444d57cd2df7420fd8` on `member-6-reranking`.

## Correction: source files are already committed

The raw committed hashes for `qwen_scorer.py` and `reranker.py` differ from the manifest because of LF versus CRLF line endings. Converting the committed LF files to CRLF reproduces BOTH manifest hashes exactly. Configuration, prompts, and the CLI file match directly without conversion.

The earlier request for another source snapshot was unnecessary. No additional source-file handoff or inference rerun is required for this issue. The detailed hashes and exact comparison are preserved in `code_provenance_review.json`.

The manifest cites an earlier base commit and reports local changes relative to that base; its two modified source files are accounted for by the current published commit with the line-ending conversion above.

## Scope and next step

Checkpoint identifiers and batch/sequence-length settings match the previous owner freeze declaration. The adapter archive was not supplied for independent hashing. Member 6's statement that inference did not access test labels remains owner reported.

Test scores stay sealed pending explicit team authorization and the applicable final model/evaluation freeze. Once authorized, Member 2 can score P1 and B3 on the identical frozen 500 users and calculate their paired comparison using the frozen settings. This receipt contains no test performance results.

## Included files

- `validation.json`: unchanged official structural validation report.
- `receipt.json`: corrected receipt.
- `code_provenance_review.json`: resolved line-ending comparison.
- `p1_test_inference_manifest_500_v1.json`: original Member 6 manifest, unchanged.
- `reply_to_member6.txt`: suggested reply; no message has been sent.
