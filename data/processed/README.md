# Processed data

Member 1's versioned artifacts are generated in `ml25m-global-v1/` by `python -m src.data.prepare`. Generated data is ignored by Git. Share the manifest, checksums, and reproducible command rather than committing large files. See [Sections 4, 5, and 8 of the roadmap](../../Movie_Recommendation_Agent_Roadmap.md) and the root [data dictionary](../../DATA_DICTIONARY.md).

Phase 2 artifacts (P2 training samples, episodes, catalog seed, and quality report) are generated in `p2-v1/` by `python -m src.data.p2_samples`, `python -m src.data.catalog_seed`, and `python -m src.data.p2_checks`. See [src/data/README.md](../../src/data/README.md#phase-2-p2-training-samples-catalog-seed-and-checks).
