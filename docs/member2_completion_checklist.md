# Member 2 — current completion and remaining work

## Complete: evaluation implementation and validation

- Ranking metrics, strict validation, saved users, explicit export preparation,
  scoring, paired-user bootstrap intervals, and Markdown/CSV tables.
- The saved 300-user validation population (seed `20261001`) and exact 50-item
  prediction contract, with input hashes and per-user metrics.
- Actual popularity (B0), BM25 (B2), frozen embeddings (B3), and P1 evaluation on
  the same users, complete comparison bundles, and a four-method table.
- P1-minus-B3 paired intervals: 10,000 resamples, seed `20261002`, confidence 95%.
- Portable reports and execution provenance in the
  [validation archive](results/member2_validation_300_v1/README.md).

P1 NDCG@10 is **0.041927**, versus B3 **0.014578**: difference **+0.027349**,
95% interval **[0.018656, 0.036476]**. P1 remains below the observed popularity
and BM25 means. Recall@50 is unchanged, as expected for identical candidate
sets. **218/300 pools contain no relevant held-out movie**. BM25 was tuned on
these validation users, so these are development findings.

The original integration check passed 218 tests, including the imported
Member 3/4 packages. Those imported implementations remain their owners' code;
this Member 2 package archives the exact source commits and hashes without
adding the owners' packages to this branch.

## Complete: test preparation and prediction integrity

- Saved **500 test users**, seed `20261003`, from the frozen test-profile IDs
  without reading test targets or test ratings.
- Preserved original prepared config and the separate later P1/B3 inference-only
  frozen config, with their original verified bytes.
- Member 5's B3 pool confirmation and Member 6's compute/model/settings
  confirmation, recorded within their stated scope.
- Received Member 6's final-test predictions and original inference manifest;
  verified both supplied SHA-256 checksums and all four input hashes.
- Official structural validation passed **500 users / 25,000 recommendations**,
  zero errors, exact B3 movie-ID sets, ordered ranks, finite scores, and complete
  coverage.
- Corrected receipt v2 resolves source provenance at published commit
  `281cca11474b3bf680ec13444d57cd2df7420fd8`; the two previously flagged hashes
  match the committed code after LF-to-CRLF conversion. No extra source handoff
  or inference rerun is needed for this issue.
- The [test archive](results/member2_test_500_v1/README.md) preserves the
  preparation/freeze records, original manifest, and current corrected receipt.

**Test performance has not been computed.** The receipt does not authorize
scoring, certify all baseline/P2 decisions, or approve the full draft protocol.
Historical pending labels in hashed preparation artifacts are retained and
explained by the later scoped confirmation record.

## Remaining coordination and course deliverables

1. Use the archived validation table and limitations in the midterm presentation
   and report. Share the archive link with Members 3–6 and the report presenter.
2. Review the [draft protocol](evaluation_protocol.md) and findings with the
   group. Successful scoring and the P1/B3 inference-only freeze do not amount to
   full team approval. Record final experiment decisions before test scoring.
3. Model owners retain the training/checkpoint/prompt and runtime records;
   Member 3 performs the explanation audit once suitable outputs are available.
   Member 2's ranking metrics do not establish explanation quality. The test
   manifest's hardware/runtime and no-label-access declaration are owner
   reported; the adapter archive was not independently hashed by Member 2.
4. Follow the roadmap's later Phase 2 plan. Its development roles and final
   evaluation controls need team agreement; no new role assignments are implied
   by this checklist.
5. After explicit team authorization and the applicable final model/evaluation
   freeze, run the joint final comparison on the same 500 users using all
   required available predictions (popularity, BM25, embeddings, P1, and P2),
   paired intervals, coverage, failure analysis, and the final results table.
   Missing outputs remain dependencies on their owners. Keep test scores sealed
   while development/model selection continues.

No additional P1 inference or metric implementation is required for the files
already received. Preserve original predictions and hashes. Do not silently
drop users, alter rankings, resample the population, or tune from test results.

The instructor's midterm presentation is two slides and two minutes, followed by
one minute of questions. The midterm report is four pages, single-spaced, 10 pt;
it covers objectives, motivation, related work, method/novelty, hypotheses,
experiments/evaluation, preliminary findings, progress, and updated milestones.
Use the course system for submission instructions and report deadlines.
