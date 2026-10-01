# MovieLens 25M data and representation pipeline

This folder implements the **Member 1** workload from the [team roadmap](Movie_Recommendation_Agent_Roadmap.md): a frozen global temporal split, catalog, canonical movie documents, user profiles, targets, and data checks. It uses Python 3.10+ standard library only. Windows needs the built-in `curl.exe` for TLS downloads.

From this folder, run the complete pipeline:

```powershell
python -m src.data.prepare
```

The command downloads the official MovieLens 25M archive and publisher MD5, verifies the archive, extracts the unmodified source CSVs into `data/raw/ml-25m/`, builds `data/processed/ml25m-global-v1/`, verifies output SHA-256 hashes and profile time boundaries, then prints statistics. Rerunning it should yield identical output hashes. To use an already extracted dataset:

```powershell
python -m src.data.build --raw data/raw/ml-25m
python -m src.data.verify
```

Run the synthetic fixture test:

```powershell
python -m unittest discover -s tests -v
```

The raw dataset and derived artifacts are ignored by Git. **Do not redistribute the raw files or commit them.** The [GroupLens dataset terms](https://files.grouplens.org/datasets/movielens/ml-25m-README.html) allow research use with conditions and prohibit redistribution without separate permission. Cite the MovieLens data source in publications.

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
