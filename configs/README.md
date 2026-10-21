# Experiment configurations

Store versioned, reviewable settings here. `data_v1.json` freezes Member 1's split, catalog, and text rules. Other members should add their own validation-selected configurations without changing a frozen version in place. See [Sections 4, 7, and 8 of the roadmap](../Movie_Recommendation_Agent_Roadmap.md).

Member 2's `validation_users_300_v1.json` contains the actual selected validation
IDs and source identity. `evaluation_v1.json` binds that file and data manifest
to the proposed ranking, candidate-pool, and bootstrap settings. The scorer
accepts it with `--config`; downstream comparison/table commands consume saved
report bundles. The protocol remains a draft for team review.

`validation_users_300_export_v1/user_ids.json` is the plain-array companion for
producer loaders. Its provenance binds it to the metadata-rich saved subset;
recreate/check it with `python -m src.evaluation.prepare users --config configs/evaluation_v1.json --output-dir configs/validation_users_300_export_v1`.

## Saved test selection and scoped inference freeze

- `test_users_500_v1.json`: the exact 500-user selection from the frozen test
  profile population, seed `20261003`, created without opening target labels.
- `test_users_500_export_v1/user_ids.json`: the identical selection as a plain
  array for producer loaders; companion provenance binds it to the metadata file.
- `evaluation_test_v1.json`: the original historical prepared configuration,
  retaining its pending status and original hash.
- `evaluation_test_frozen_v1.json`: the later P1/B3 inference-only frozen copy;
  only administrative `status` and `subset_status` differ from the prepared copy.

Use these saved files directly without regenerating or resampling the test
selection. General subset/preparation commands can open targets. Preserve
verified bytes and consult the
[test inference archive](../docs/results/member2_test_500_v1/README.md) for the
confirmation superseding preparation-time pending labels. The scoped inference
freeze did not itself authorize scoring. The user later relayed explicit team
approval to run and publish P1/B3 test results; the
[completed comparison](../docs/results/member2_p1_b3_test_500_v1/README.md)
uses `evaluation_test_frozen_v1.json` unchanged. This does not approve the full
draft protocol or freeze P2/other baseline decisions.

The complete [300-user validation archive](../docs/results/member2_validation_300_v1/README.md)
uses the unchanged `evaluation_v1.json`, whose status remains
`draft_for_team_review`. P1/B3 test scoring is complete under separately recorded
authorization. Sealed/pending statements in preserved receipt/preparation JSON
describe their earlier pre-scoring stage.
