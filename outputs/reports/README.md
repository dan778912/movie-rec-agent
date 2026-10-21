# Evaluation reports

Generated result tables, plots, and run reports belong here and are ignored by Git. Commit small report summaries elsewhere when needed for review. See [Sections 7, 8, and 12 of the roadmap](../../Movie_Recommendation_Agent_Roadmap.md).

## Published Member 2 evidence

The local generation area remains ignored. Reviewed lightweight records are
preserved in tracked directories:

- [300-user validation archive](../../docs/results/member2_validation_300_v1/README.md):
  complete popularity/BM25/B3/P1 scored bundles, two paired comparisons, the
  four-method table, exact commands, and source provenance.
- [500-user test archive](../../docs/results/member2_test_500_v1/README.md):
  label-blind preparation, scoped inference confirmation, original Member 6
  manifest, official structural validation, and corrected receipt v2.
- [Authorized 500-user P1/B3 test comparison](../../docs/results/member2_p1_b3_test_500_v1/README.md):
  both scored bundles, paired intervals, table, authorization, and exact commands.

The corrected receipt accounts for source-file LF/CRLF differences; the old
receipt v1 is superseded. Test predictions passed integrity checks for 500 users
and 25,000 recommendations with zero errors. That receipt describes the earlier
label-blind stage. The user later relayed explicit team approval to run and
publish P1/B3 test results; the completed comparison is archived separately
above. Preserved receipt/freeze JSON retains its historical sealed/pending
statements. This completes P1/B3 scoring, not Phase 2 or the broader joint final
comparison.

Preserve original report/configuration bytes and their hashes. Complete
validation bundles can be copied and compared without rewriting historical
absolute paths. Raw data, model weights, and prediction files stay in separate
agreed storage rather than this small report archive.
