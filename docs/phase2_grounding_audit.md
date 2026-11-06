# Phase 2 explanation grounding audit — Member 2

**Status:** preparation method and reviewer rubric. No P2 outputs have been reviewed, and no grounding rate is claimed.

## Sample and worksheet

After the P2 trace contract and validation protocol are accepted, fix the sample size and seed on validation, before looking at final test explanations. The default is **30 user episodes** and seed `20261002`. The selection takes the first 30 users by SHA-256 of `seed|split|user_id`; it does not read targets, scores, action paths, or explanation text. Every top-ten explanation for each selected user enters the worksheet, including fallback episodes if selected. A different number of episodes is a separate, declared audit, not a replacement for this sample.

Run `python -m src.evaluation.p2_grounding_audit` with the canonical catalog **and matching versioned `movie_documents.jsonl`**, matching split profiles, frozen user-list JSON, frozen B3 50-item predictions, P2 50-item predictions, and complete P2 trace. For example:

```powershell
python -m src.evaluation.p2_grounding_audit `
  --catalog data/processed/ml25m-global-v1/catalog.csv `
  --movie-documents data/processed/ml25m-global-v1/movie_documents.jsonl `
  --profiles data/processed/ml25m-global-v1/validation_profiles.jsonl `
  --users configs/validation_users_300_v1.json `
  --split validation `
  --candidate-pool PATH_TO_FROZEN_B3_VALIDATION.jsonl `
  --predictions PATH_TO_P2_VALIDATION.jsonl `
  --trace PATH_TO_P2_VALIDATION_TRACE.jsonl `
  --sample-size 30 --seed 20261002 `
  --output-dir outputs/reports/p2-grounding-audit-validation-v1
```

The command requires complete valid P2 records and trace coverage before it writes anything. It checks that the movie-document IDs exactly cover the catalog and that sampled document titles and genres match their catalog rows. It creates an immutable `manifest.json` with input SHA-256 values, including the movie documents, selected IDs, selection rule, and a hash of `review.csv`. Repeating the command with identical inputs is harmless; changed content requires a new output directory. It deliberately never opens targets or calculates ranking metrics.

The CSV has one row per selected explanation. It copies the exact trace explanation; candidate title and genres from the versioned catalog; the exact movie-document text, genres, ordered genome tags, and ordered user tags; starting **visible** profile text; ordered clarification turns and stated answer sources; and initial/retrieved/final candidate-pool membership. The worksheet also includes a digest of each sampled movie-document record. It does **not** treat the answer-source string as proof that an answer used pre-cutoff history. The reviewer fills `claims_reviewed`, `catalog_or_document_evidence`, `profile_or_answer_evidence`, `unsupported_or_unverifiable_claims`, `grounding_label`, `reviewer`, `review_notes`, and `adjudication` in a **copy** of the CSV. Preserve the untouched package and its hashes.

## Reviewer rubric

Read each explanation as a set of checkable claims. Record all concrete movie, genre, tag, rating, preference, or clarification claims in `claims_reviewed`. Check movie identity and title/genre claims against the exact catalog row; check tag claims against the versioned movie-document tags and text. A tag is evidence that the supplied retrieval document contained that descriptor, not proof of an objective fact about the film. Check claims about the user's likes or dislikes against the visible starting profile and clarification answers that preceded the final recommendation. The full watched-ID list is an exclusion control, not evidence of a preference or rating. The trace proves that an answer string was logged, not that its claimed source or timing is correct; consult the simulator provenance separately for that question.

Use one `grounding_label` per explanation:

| Label | Rule |
|---|---|
| `supported` | Every concrete claim has a cited supporting field in the supplied evidence; no contradiction or unavailable factual claim remains. |
| `unsupported` | At least one concrete claim contradicts the supplied evidence, names the wrong movie, or asserts a user preference absent from the supplied profile and answers. Quote the offending claim and evidence gap. |
| `insufficient_evidence` | No claim is disproved, but a claim needs a field absent from the supplied catalog and movie documents. For example, a plot fact absent from their tags or text cannot be checked. Do not consult outside movie knowledge and silently promote it to supported. |
| `no_checkable_claim` | The text is generic praise without a concrete movie or user-preference claim. This may be fluent but does not satisfy the intended grounded-justification goal. |

If both contradicted and unverifiable claims occur, use `unsupported` and list both in the notes. A syntactically empty explanation is a validator failure, so the worksheet should never contain one. If the producer used retrieval metadata beyond the canonical movie documents, require a versioned, hashed snapshot and make a new worksheet version that includes it; do not silently use mutable external sources.

Have a second reviewer independently label a declared subset or all rows. Resolve disagreements with evidence citations, preserve both initial judgments and adjudication, and report the process. Count explanations by final label and report the denominator as all sampled explanations. Report the user sample size, seed, source hashes, any nonreviewed rows and reasons, and a separate claim-level count if individual claims are tabulated. The sample is uniform over user episodes, so each selected user contributes the same ten explanations in the production run; it is descriptive of this frozen output, not a guarantee of behavior in live conversations. Report unsupported, insufficient-evidence, and generic cases separately rather than folding them into a single favorable rate.

## Scope of the result

This tool verifies structure and prepares evidence; it cannot determine semantic truth, temporal provenance of a simulated answer, whether model actions occurred, or real-user satisfaction. Audit those separately under the [Phase 2 protocol](phase2_evaluation_protocol.md). Do not choose a new prompt, checkpoint, simulator, or retrieval rule from final-test audit findings.
