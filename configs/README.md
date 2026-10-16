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
