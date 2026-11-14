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

## Phase 2 artifacts — `p2-v1`

Generated in `data/processed/p2-v1/` by Member 1's Phase 2 commands (see `src/data/README.md`). Settings: `configs/p2_samples_v1.json`.

| File | Fields | Meaning |
|---|---|---|
| `p2_samples_train.jsonl`, `p2_samples_validation.jsonl` | `schema_version,prompt_version,sample_id,episode_id,step,task,split,user_id,state,prompt,target,completion,provenance` | One supervised agent decision per line. `task` is one of `assess_sufficiency`, `ask_clarification`, `evaluate_candidates`, `retrieve_candidates`, `rank_candidates`, `explain_recommendations`. Train on `prompt` → `completion`. `target` is the parsed action; `completion` is its canonical JSON text. `state` holds `profile_text`, `clarifications` (`question`,`answer`), and `candidate_movie_ids` (or `null`) so the prompt can be re-rendered with `src/profiles/agent_text.py`. `provenance` (anchor/boundary timestamps, profile level, label basis) is audit metadata and must **not** enter the model input. |
| `p2_episodes_train.jsonl`, `p2_episodes_validation.jsonl` | `schema_version,episode_id,split,user_id,anchor_timestamp,boundary_timestamp,level,sparse_k,pool_regime,history_count,history_latest_timestamp,visible_movie_ids,profile_text,future_positive_movie_ids,future_positive_timestamps,future_negative_movie_ids,initial_candidate_movie_ids,events,final_candidate_movie_ids,injected_movie_ids,termination_reason,forced,summary,sample_ids` | One simulated episode per line, using the `p2-trace-v1` event vocabulary (extra audit fields on `ask_clarification` and `rank_candidates`). History is every rating with timestamp `< anchor_timestamp`; held-out labels come from `[anchor_timestamp, boundary_timestamp)`. Pool sizes and `split: "train"` differ from the 50-item evaluation trace, so these files are training/test fixtures, not evaluator input. |
| `p2_samples.report.json` | config, versions, input hashes, per-split counts, label balance, prompt lengths, output hashes, provenance, limitations | Generation receipt. |
| `catalog_seed.jsonl` | `movie_id,title,display_title,year,genres,genome_tags,user_tags,display_tags,text,rating_avg,rating_count,train_positives,popularity_rank,imdb_id,tmdb_id` | One record per catalog movie. `text` equals `movie_documents.jsonl`. `rating_avg`/`rating_count`/`train_positives` use `train_ratings.csv` only. `display_tags` is the three-genome-plus-two-user tag slice shown in profiles and agent prompts. `imdb_id`/`tmdb_id` come from `links.csv` for inspection only. |
| `catalog_seed.report.json` | versions, input hashes, counts, metadata gaps, output hash | Seed receipt. |
| `p2_quality_report.json` | `base`, `samples`, `catalog_seed`, `artifact_sha256`, `error_count`, `errors_by_kind`, `error_examples` | Data-quality and temporal-boundary check results. `error_count` must be 0. |

`train` samples anchor inside the training period and read training ratings only. `validation` samples anchor at the validation cutoff: their canonical profile equals `validation_profiles.jsonl` text and their labels come from `validation_ratings.csv`. They are intended for checkpoint selection. The 300 frozen validation evaluation users and 500 frozen test users are excluded from both splits, and train/validation users are disjoint.
