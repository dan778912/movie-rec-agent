# Movie Recommendation Agent

This folder implements the **Member 1** data workload from the [team roadmap](Movie_Recommendation_Agent_Roadmap.md): a frozen global temporal split, catalog, canonical movie documents, user profiles, targets, and data checks. **Member 2** adds tested ranking metrics, strict prediction validation, fixed users, teammate-export preparation, scoring, paired bootstrap comparisons, CSV/Markdown tables, synthetic examples, and a draft evaluation protocol. These components use only the Python 3.10+ standard library. Windows needs the built-in `curl.exe` for TLS downloads.

## Team repository layout

The repository follows Section 8 of the roadmap. Each folder has a README describing its handoff point. The current code implements Member 1's data and profile work plus Member 2's metrics, prediction validation, fixed-user selection, scoring, paired bootstrap comparisons, and comparison-table generation. Actual performance results await the model owners' prediction files. Other folders are ready for their owners to add code, configurations, tests, and reports.

See the [Member 2 evaluation instructions](src/evaluation/README.md) for runnable scoring and validation examples; the [Member 2 handoff](docs/member2_handoff.md) for teammate integration; and the [draft evaluation protocol](docs/evaluation_protocol.md) for proposed shared experiment rules. The scorer checks prediction integrity and data identity before calculating per-user metrics, macro means, and catalog-positive coverage. The saved [300-user validation subset](configs/validation_users_300_v1.json) identifies the shared experiment population. The protocol still awaits team review, and real prediction files are needed to establish baseline performance.

The [integration guide](docs/member2_integration.md) covers the reviewed Member
3/4/5 interfaces. It prepares their full-population Top 100/50 exports as explicit,
provenance-recorded subset files and exports a plain user-ID array for existing
producer loaders. Original prediction files are preserved, and the entire source
file is validated before any user selection or prefix operation.

| Owner | Folder | Work |
|---|---|---|
| Member 1 | `src/data/`, `src/profiles/`, `data/` | Frozen dataset and text representations |
| Member 2 | `src/evaluation/` | Shared metrics, validation, subsets, scoring, paired intervals, tables |
| Member 3 | `src/baselines/` | Popularity baseline and P1 training-data support |
| Member 4 | `src/retrieval/` | BM25 retrieval |
| Member 5 | `src/retrieval/` | Embedding retrieval and candidate pools |
| Member 6 | `src/reranking/` | P1 and P2 fine-tuning and inference |
| Phase 2 team | `src/agent/`, `src/app/` | Agent loop and application; assignment follows Phase 1 |

Shared experiment settings belong in `configs/`; model artifact references in `models/`; standard prediction JSONL files in `outputs/predictions/`; generated tables and reports in `outputs/reports/`; unit and smoke tests in `tests/`. Large datasets, model weights, and generated outputs are Git-ignored. Commit code, small configuration files, documentation, and tests. To collaborate, clone the repository, create a branch for your task, and open a pull request into `main` after your checks pass.

The GitHub repository is private. Its owner must invite each teammate as a collaborator before they can clone or push. Once invited, a teammate can start with:

```powershell
git clone https://github.com/dan778912/movie-rec-agent.git
cd movie-rec-agent
git switch -c member-4-bm25  # replace with your task's branch name
```

Members should commit their own code and tests, then open a pull request for review. The shared processed data is reproduced with the command below; it is not stored in GitHub. Frontend, service, and deployment details will be added by the Phase 2 team after the modeling gate.

If processed data is absent, run the complete pipeline once from this folder:

```powershell
python -m src.data.prepare
```

The command downloads the official MovieLens 25M archive and publisher MD5, verifies the archive, extracts the unmodified source CSVs into `data/raw/ml-25m/`, builds `data/processed/ml25m-global-v1/`, verifies output SHA-256 hashes and profile time boundaries, then prints statistics. This full build has passed locally; it need not run before each evaluation test. A rerun should yield identical output hashes. To rebuild from an already extracted dataset when needed:

```powershell
python -m src.data.build --raw data/raw/ml-25m
python -m src.data.verify
```

Run the available tests, including the synthetic data fixture and hand-calculated ranking metrics:

```powershell
python -m unittest discover -s tests -v
```

The raw dataset and derived artifacts are ignored by Git. **Do not redistribute the raw files or commit them.** The [GroupLens dataset terms](https://files.grouplens.org/datasets/movielens/ml-25m-README.html) allow research use with conditions and prohibit redistribution without separate permission. Cite the MovieLens data source in publications.

## Fixed validation users

The shared subset contains 300 of the 4,751 evaluable validation users, selected with seed `20261001`. Use the actual IDs in [configs/validation_users_300_v1.json](configs/validation_users_300_v1.json), rather than independently resampling. [configs/evaluation_v1.json](configs/evaluation_v1.json) records proposed settings and actual data/subset hashes. The scorer accepts this file with `--config`; the standalone validator continues to use explicit input arguments.

To reproduce or check the subset using matching processed data and the recorded Python version:

```bash
python -m src.evaluation.subsets \
  --data-dir data/processed/ml25m-global-v1 \
  --split validation \
  --size 300 \
  --seed 20261001 \
  --output configs/validation_users_300_v1.json
```

The command checks selected data files against the manifest and verifies the profile/positive-target population. It accepts identical existing bytes and refuses to overwrite a different frozen subset. The saved IDs remain authoritative across Python versions; a different recorded Python version can prevent byte-identical regeneration. Producing or verifying a subset does not constitute team approval of the draft protocol. No real recommender predictions or experiment scores have been produced by Member 2 at this stage.

## Try end-to-end scoring

Run this complete synthetic example from the repository root:

```bash
python -m src.evaluation.evaluate \
  --config tests/fixtures/evaluation_scoring/config.json \
  --predictions tests/fixtures/evaluation_scoring/embedding.jsonl \
  --method embedding \
  --output-dir outputs/reports/scoring_demo/embedding
```

It saves `summary.json`, `per_user.csv`, and `validation.json`. Expected synthetic
macro scores are Recall@2 = 0.5, NDCG@2 = 0.5, HitRate@2 = 1.0, and Recall@3 =
0.75. These are hand-checkable test results, not MovieLens model performance.
The [evaluation README](src/evaluation/README.md) explains the outputs and the
real 300-user scoring command to use when predictions are available.

After running both synthetic scoring examples in the evaluation instructions,
compare their saved user scores:

```bash
python -m src.evaluation.bootstrap \
  --baseline-dir outputs/reports/scoring_demo/embedding \
  --challenger-dir outputs/reports/scoring_demo/p1 \
  --output-dir outputs/reports/scoring_demo/p1_vs_embedding
```

The command saves `comparison.json` and `per_user_differences.csv`. Differences
are challenger minus baseline. It verifies matching experiments and pairs users
by ID, then applies the saved bootstrap settings. See the
[bootstrap walkthrough](docs/member2_bootstrap_walkthrough.md) for expected
synthetic values and how to interpret the intervals. Two-user demonstrations
check implementation and do not establish real model performance.

Generate a clearly labelled table from those example results:

```bash
python -m src.evaluation.report \
  --runs outputs/reports/scoring_demo/embedding outputs/reports/scoring_demo/p1 \
  --comparisons outputs/reports/scoring_demo/p1_vs_embedding \
  --kind synthetic \
  --output-dir outputs/reports/scoring_demo/table
```

It saves `table.md`, `table.csv`, `differences.csv`, and `report.json`. Tables
contain only the supplied validated methods; missing methods are listed without
inventing scores. The [table walkthrough](docs/member2_table_walkthrough.md)
explains the outputs and the real-experiment command.

## Frozen decisions

The versioned settings are in [configs/data_v1.json](configs/data_v1.json). Every source rating is ordered by timestamp globally. The test boundary is the timestamp at zero-based index `floor((N−1)×0.90)` of the sorted source ratings. Ratings at that timestamp go into test, and earlier ratings go into the pretest portion. The validation boundary uses the same 90% rule inside pretest. Ratings at that timestamp go into validation; earlier ratings go into training. Timestamp ties stay wholly in the later split. This gives a single, leakage-safe cutoff for all users; split sizes may differ slightly from exact 81/9/10 percentages.

A user is eligible if they have at least 20 **training** ratings. This criterion uses no future activity. The catalog contains movies with at least five training ratings by eligible users. Validation and test profile files contain only eligible users with at least one positive held-out movie *in that catalog*; all positive targets, including out-of-catalog ones, remain in target files for coverage reporting. A positive is a rating of at least 4.0. The exact user selection and catalog coverage are in `statistics.json`.

Movie documents and profiles are deterministic. Movie documents use the MovieLens title (including year), genres, eight highest-relevance genome tags, and up to five most frequent cleaned user tags observed **before the validation cutoff**. Ties break by tag ID or tag text. The genome is treated as static catalog metadata, as permitted by the roadmap; its relevance scores were created by GroupLens using user-contributed content and are therefore *transductive metadata*. All text-based methods must use this same document and disclose this choice. A stricter experiment can omit genome tags for every method.

Profiles use only available ratings: training for validation, training plus validation for test. The text includes up to ten liked movies (rating at least 4, sorted by rating, recency, ID), with their genres and tags, plus up to five explicitly low rated movies (at most 2.5, sorted by rating, recency, ID). `history_movie_ids` includes *all* rated movies, even movies outside the candidate catalog, so retrieval methods can remove watched items. Live application profiles should use the same formatter in `src/profiles/text.py`.

For validation model selection, use the validation profiles/targets and training split. After settings are frozen, use test profiles/targets and training plus validation history. Do not use test targets for tuning.

## Files

See [DATA_DICTIONARY.md](DATA_DICTIONARY.md) for fields. The processed folder contains:

- `train_ratings.csv`, `validation_ratings.csv`, `test_ratings.csv`: all interactions from users meeting the training-only eligibility rule.
- `catalog.csv`: eligible movie IDs and training counts.
- `movie_documents.jsonl`: canonical movie text, one catalog movie per line.
- `validation_profiles.jsonl`, `test_profiles.jsonl`: canonical profile text, selected examples, and the complete watched movie ID list.
- `validation_targets.csv`, `test_targets.csv`: positive held-out ratings with an `in_catalog` flag.
- `statistics.json`: cutoffs, user/item/interaction counts, positives, sparsity, and positive-target catalog coverage.
- `manifest.json`: version, configuration, raw SHA-256 hashes, and SHA-256 hashes for every other output file.

The profiles and target files are sorted by user ID. All CSV output uses UTF-8 and LF line endings. JSONL uses UTF-8, sorted keys, and compact separators. No current date is embedded in the generated files, so reruns are byte identical with the same source and config.

## Completed v1 build

The full official archive passed its MD5 check (`6b51fb2759a8657d3bfcbfc42b592ada`). The build read 25,000,095 source ratings. It selected 138,950 users based on training history and a 21,333 movie catalog. The validation cutoff is **2016-08-18 12:40:46 UTC** and the test cutoff is **2018-01-04 02:44:50 UTC**. After user filtering, the training, validation, and test files contain 20,246,363, 377,431, and 328,542 ratings respectively. Training catalog sparsity is 99.318%.

Validation has 148,263 positive targets, of which 126,029 (85.00%) are in the catalog; 4,751 users have at least one in-catalog positive target. Test has 120,035 positive targets, of which 81,330 (67.76%) are in the catalog; 3,779 users have at least one in-catalog positive target. The lower test catalog coverage matters: final results must report it alongside recommendation metrics. These counts are generated in `statistics.json`, and all ten processed files have recorded SHA-256 checksums in `manifest.json`.

## Source

Official download: [MovieLens 25M](https://grouplens.org/datasets/movielens/25m/). The publisher describes 25,000,095 ratings, 1,093,360 tag applications, 62,423 movies, and 162,541 users in the [dataset README](https://files.grouplens.org/datasets/movielens/ml-25m-README.html). The pipeline records the actual source count and verifies the archive against the publisher's MD5. Output files use SHA-256 checksums.
