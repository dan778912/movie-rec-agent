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

The corrected receipt accounts for source-file LF/CRLF differences; the old
receipt v1 is superseded. Test predictions passed integrity checks for 500 users
and 25,000 recommendations with zero errors. **No test performance results are
published or computed here; scoring remains sealed pending team authorization
and the applicable final model/evaluation freeze.**

Preserve original report/configuration bytes and their hashes. Complete
validation bundles can be copied and compared without rewriting historical
absolute paths. Raw data, model weights, and prediction files stay in separate
agreed storage rather than this small report archive.
