# Data dictionary — `ml25m-global-v1`

All IDs are MovieLens IDs. Times are integer Unix seconds in UTC. Ratings are 0.5–5.0 stars. `validation` means the middle global time slice; `test` means the latest slice.

| File | Fields | Meaning |
|---|---|---|
| `train_ratings.csv`, `validation_ratings.csv`, `test_ratings.csv` | `user_id,movie_id,rating,timestamp` | Ratings from users with at least 20 training interactions. Files are ordered by user ID, time, movie ID. |
| `catalog.csv` | `movie_id,title,genres,train_ratings,train_positives` | Candidate movies with at least five eligible-user training ratings; genres are pipe-separated as in source. Counts use training only. |
| `movie_documents.jsonl` | `movie_id,title,genres,genome_tags,user_tags,text` | One canonical retrieval/reranking document per catalog movie. Genres and tags are arrays; `text` is the stable shared input. |
| `validation_profiles.jsonl`, `test_profiles.jsonl` | `user_id,split,text,history_movie_ids,history_count,history_latest_timestamp,liked_movies,disliked_movies` | One profile per evaluable user with at least one in-catalog positive target. History contains only events before the relevant cutoff. Selected movie objects have `movie_id,title,rating,timestamp,genres,tags`. |
| `validation_targets.csv`, `test_targets.csv` | `user_id,movie_id,rating,timestamp,in_catalog` | Every held-out positive of an eligible user. `in_catalog` is 1 if recommendable and 0 otherwise. Evaluate recommendation recall against `in_catalog=1`, and report excluded positives using the full file. |
| `statistics.json` | `source_ratings,eligible_users,catalog_movies,train_catalog_interactions,train_catalog_density,train_catalog_sparsity,splits,evaluation,validation_cutoff_unix,validation_cutoff_utc,test_cutoff_unix,test_cutoff_utc` | Split totals refer to eligible users. `evaluation` includes target coverage and number of users with at least one in-catalog positive. Density is training interactions within the catalog divided by eligible users × catalog movies. Sparsity is 1 − density. |
| `manifest.json` | `version,config,raw_sha256,file_sha256` | Immutable input/output identity for reproducibility. `file_sha256` excludes the manifest itself. |

For each profile, `history_movie_ids` is the complete watched set used by every recommender to exclude seen items. `liked_movies` and `disliked_movies` are only the selected examples rendered into `text`; neither is an exhaustive history. The `text` field is deterministic and intentionally has no LLM-written content.

`genome_tags` are static source metadata. `user_tags` are derived only from pre-validation tags. Genome construction itself used contributed content before MovieLens 25M was published, so it cannot be claimed as a fully prospective feature. That limitation applies equally to every text method using this document.
