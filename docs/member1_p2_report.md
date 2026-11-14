# Member 1 — Phase 2 data handoff: P2 samples, catalog seed, quality report

**Status:** generated and checked on 2026-11-14 with zero check errors. Ready for review by Members 2–6. The label thresholds, simulator, and prompt template are **proposed v1** and should be ratified together with the [Phase 2 interface contract](https://github.com/dan778912/movie-rec-agent/blob/member2-phase2/docs/phase2_interface_contract.md) before P2 test inference.

| Deliverable | Location (Git-ignored data) | Code |
|---|---|---|
| P2 training samples covering six action types | `data/processed/p2-v1/p2_samples_{train,validation}.jsonl` | `src/data/p2_samples.py` |
| Simulated episodes in `p2-trace-v1` event vocabulary | `data/processed/p2-v1/p2_episodes_{train,validation}.jsonl` | same |
| Data-quality and temporal-boundary report | `data/processed/p2-v1/p2_quality_report.json` plus this document | `src/data/p2_checks.py` |
| Movie catalog seed | `data/processed/p2-v1/catalog_seed.jsonl` | `src/data/catalog_seed.py` |
| Shared prompt renderer, action grammar, and simulator | — | `src/profiles/agent_text.py`, `src/data/p2_actions.py`, `src/data/clarification_sim.py` |

Reproduce from the repository root after `python -m src.data.prepare`. The raw CSVs must be extracted in `data/raw/ml-25m/`.

```bash
python -m src.data.p2_samples && python -m src.data.catalog_seed && python -m src.data.p2_checks
```

The whole run takes about 4 minutes on a laptop and uses the standard library only. A [full example episode](p2_sample_examples.jsonl) (ten samples, all six tasks) is committed for review.

## 1. How samples are built

Each **episode** fixes a user, an anchor time `t`, and a visible profile:

- **History:** every rating with timestamp `< t`.
- **Held-out preferences:** the user's ratings in `[t, boundary)`. They supervise the decisions but never appear in a prompt.
- **`train` split:** `t` is a timestamp boundary inside the training period and `boundary` is the validation cutoff. Only `train_ratings.csv` is read.
- **`validation` split:** `t` is the validation cutoff and `boundary` is the test cutoff. History equals the frozen validation profile history, and labels come from `validation_ratings.csv`. This split is for checkpoint selection.
- **Profile levels:**
  - *canonical:* exactly the frozen formatter over the whole history. It matches the offline P2 start state.
  - *sparse-k:* the formatter over the user's last k ∈ {1, 2, 3, 5, 8} catalog ratings. It models a live app user who has rated only a few movies.

The episode walks the agent loop and emits one sample per decision. Its control events stay within the contract's cap: at most 3 assessments, 2 questions, 2 pool evaluations, and 1 retrieval, for 8 in total. Each episode then ends with one ranking and one explanation event. The sample label rules are:

| Task | Target | Label rule (uses held-out preferences only to choose the label) |
|---|---|---|
| `assess_sufficiency` | `sufficient: bool` | `false` when the best answer the simulator could give would raise the cosine between the preference vector and the held-out positives' centroid by at least 0.04. Questions must remain under the cap. |
| `ask_clarification` | `question` | A template question for the intent with the largest gain: liked example, favorite genre, themes, dislikes, or era. |
| `evaluate_candidates` | `pool_sufficient: bool` | `false` when a fresh retrieval with the current information would beat the current pool by at least 0.05 cosine against the held-out centroid. The pool is scored on its best 10 items, and retrievals must remain under the cap. |
| `retrieve_candidates` | `query` | Restates displayed preferences: up to 4 liked titles, 3 genres, 5 themes, a stated decade, and genres to avoid. |
| `rank_candidates` | full permutation | Held-out positives first (rating ↓, earlier ↑). Unlabeled movies follow in pool order, and held-out explicit dislikes (≤ 2.5) go last. |
| `explain_recommendations` | top-10 `{movie_id, text}` | A template that cites the overlap between the candidate's displayed genres/tags and a displayed liked movie or a clarification answer. When nothing overlaps, it uses an honest no-overlap sentence instead. |

**Preference vector.** The vector is IDF-weighted over genres, genome tags, user tags, and decade. Visible likes count +1, visible dislikes −0.5, and revealed facts add their own features.

**Candidate pools.** Pools come from a dependency-free inverted-index retriever: cosine with BM25-style length normalization and a popularity prior. They are either:

- *profile:* retrieved from the starting profile, as the B3 analogue (65% of episodes);
- *off-target:* the most popular movies from the user's two weakest genres, simulating a retrieval miss (35%).

Pool size is 20 and re-retrieval adds 20. For ranking samples only, up to 3 held-out positives and 2 held-out dislikes are inserted at random positions. They are listed in `provenance.label_basis.injected_movie_ids`.

**Simulator `pre_cutoff_history_sim_v1`.**

- Answers come from history before `t` only. The simulator raises an error if a later rating is passed in.
- It prefers facts absent from the visible text, and repeating an intent reveals that intent's next fact.
- Unrecognized questions get `I'm not sure how to answer that.` Exhausted intents get a declared noninformative answer.
- Question → intent uses fixed keyword rules. Every template round-trips in the unit tests.
- **Proposal:** Member 4's offline P2 runner could use these same functions, so training and evaluation answers come from one frozen implementation.

## 2. Results

| | train | validation |
|---|---:|---|
| Users / episodes | 2,000 / 3,979 | 400 / 400 |
| Samples | 23,466 | 2,102 |
| `assess_sufficiency` (insufficient / sufficient) | 6,748 (2,769 / 3,979) | 580 (180 / 400) |
| `ask_clarification` | 2,769 | 180 |
| `evaluate_candidates` (insufficient / sufficient) | 4,985 (1,006 / 3,979) | 471 (71 / 400) |
| `retrieve_candidates` | 1,006 | 71 |
| `rank_candidates` / `explain_recommendations` | 3,979 / 3,979 | 400 / 400 |
| First assessment says "ask": canonical / sparse | 22% / 77% | 21% / 75% |
| Clarifying intents: genre / liked example / themes / era | 1,440 / 815 / 346 / 168 | 100 / 50 / 30 / 0 |
| Answers revealing a fact absent from visible text | 2,061 of 2,769 (74%) | 137 of 180 (76%) |
| Episodes ending at the question cap / retrievals ending at the retrieval cap | 218 / 318 | 16 / 21 |
| Prompt characters, p50 / p95 / max (rank task) | 5,173 / 8,030 / 9,082 | 5,370 / 8,327 / 9,350 |

The label rules produce the intended behavior. A sparse profile is about 3.5× more likely to trigger a question than a canonical one. Retrieval fires mostly on off-target pools (555 of 1,361) and occasionally on profile pools once clarification has changed the preferences (451 of 2,618). The `dislikes` intent is never the best question under the vector proxy, because negative evidence rarely moves the cosine toward the positives. The simulator still answers it if the model asks.

The cap-forced decisions — 218 episodes (5%) that still wanted a third question, and 318 of 1,006 retrievals (32%) — are labelled "proceed" to match the loop cap. They carry `forced_by_question_cap` / `forced_by_retrieval_cap` in `provenance.label_basis`, so Member 6 can down-weight or drop them.

## 3. Data-quality and temporal-boundary checks

`p2_checks` recomputes everything from source files rather than trusting the generator. Result: **`error_count: 0`**.

| Check | Result |
|---|---|
| Frozen `ml25m-global-v1` checksums and profile cutoffs (`src.data.verify`) | 10/10 hashes; 4,751 validation and 3,779 test profiles verified |
| Split time ranges | train max 2016-08-18 12:40:42 < validation cutoff 12:40:46 ≤ validation min. Validation max < test cutoff 2018-01-04 02:44:50 ≤ test min. |
| **Replay from boundary-truncated data** | All 4,379 episodes were rebuilt from source ratings cut at `t` (history) and `boundary` (labels). Each reproduced its stored samples and episode **byte for byte**, so no rating outside those windows influenced any sample. |
| History, future, and visible items | History count and latest timestamp match, and all are `< t`. Every label lies in `[t, boundary)`, and every visible movie is in the history. Validation canonical profiles equal the frozen profile text. |
| Simulated answers | 2,949 answers replayed from pre-anchor history. All source movies are watched before `t`. No held-out positive title appears in profile or answer text. |
| Candidates | Every initial, added, and final candidate is in the catalog and unwatched. Injected items are genuine held-out ratings. |
| Grammar and order | Every target parses strictly, and every completion is its canonical serialization. Every prompt re-renders exactly. Every episode passes the `p2-trace-v1` ordering rules with ≤ 8 control events. |
| Explanation grounding (independent parser) | 43,790 / 43,790 explanations cite only displayed titles, ratings, genres, tags, and answers. |
| Population | Train and validation users are disjoint. All 782 distinct frozen evaluation users (300 validation + 500 test, 18 in both) are excluded. The test split is never opened by the generator, simulator, or seed builder (checked by a unit test). |
| Catalog seed | 21,333 rows = catalog. Text equals `movie_documents.jsonl`. `rating_count` and `train_positives` equal `catalog.csv` training counts. |

**Restored artifact.** `train_ratings.csv` and the raw `ratings.csv` / `genome-scores.csv` were missing locally. They were restored from the MD5-verified archive and rebuilt, and all ten processed files matched the frozen manifest hashes exactly.

### Data-quality findings for the team

1. **Test profiles often show out-of-catalog liked movies.** 1,172 of 3,779 test profiles (31%) list at least one liked movie outside the candidate catalog, against 174 of 4,751 validation profiles (3.7%). This follows the frozen rule that profiles include all history. However, Member 3's P1 training profiles render **catalog movies only**, so the P1/P2 training and test profile distributions differ more at test than at validation. P2 samples use the frozen rule, so canonical training profiles match evaluation profiles. **Member 6 should not mix the two profile styles unknowingly.**
2. **Catalog metadata gaps.** Of 21,333 catalog movies:
   - 8,443 lack genome tags and 4,054 lack user tags; 3,166 have no displayable tag at all;
   - 111 have no genre and 9 have no parseable year;
   - 20 title strings are duplicated (distinct MovieLens IDs);
   - 32 have no TMDB ID.

   Explanations for those movies fall back to genre-only or no-overlap text.
3. **40 user tags contain literal double quotes** (for example `"damn dirty apes"`). Explanations and simulator answers never cite them, because quoted citations would be ambiguous. The checker caught this before the final run.

## 4. Limitations

- Clarification answers are simulated from pre-anchor ratings, not real user replies. They support a controlled offline claim only.
- Decision labels use a genre/tag/decade vector proxy for held-out taste. The 0.04 / 0.05 thresholds were set by inspecting label balance on a 150-user training-period smoke run, not on validation or test outcomes.
- Training pools come from the simulator's feature index, not Member 3's vector index or the frozen B3 pools. Pools hold 20–40 items (45 with injections) rather than 50.
- Off-target pools are a simulated retrieval miss. Ranking pools contain injected held-out items, so the share of positives is higher than in real B3 pools. Without injection, an initial 20-item pool contains on average 0.28 of the user's first ten held-out positives.
- Explanations are deterministic templates — grounded but plain.
- Like Phase 1, genome tags are static transductive metadata. User tags and catalog membership come from the whole pre-validation period, so a `train` anchor can see metadata dated between `t` and the validation cutoff. The same limitation applies to Member 3's P1 samples.

## 5. Handoff notes

- **Member 6 (P1 → P2 fine-tuning):**
  - Train on `prompt` → `completion`. Keep `provenance` out of the input.
  - The validation file is for checkpoint selection; it excludes the 300 evaluation users.
  - Prompts reach about 9.4k characters (~2.5k tokens) on rank tasks, so raise `max_length` above P1's 1024 or filter by task.
  - Filter by `task` to rebalance. Consider dropping the cap-forced labels.
- **Member 4 (agent):**
  - Render inference prompts with `src.profiles.agent_text.render_prompt` (`p2-prompt-v1`) so they match training exactly.
  - Parse model output with `src.data.p2_actions.parse_action`. `validate_episode` implements the contract's ordering rules.
  - The episode files are ready-made traces for parser and loop tests.
  - The offline runner can answer questions with `clarification_sim.build_fact_pool` + `answer_question`, passing ratings from **before** the split cutoff (training for validation; training + validation for test).
- **Member 3 (retrieval / Service B):** the `catalog_seed.jsonl` `text` field is the canonical document to embed, and `movie_id` is the key. The training simulator's `retrieve_candidates` query is free text, so the vector index should embed queries with the same encoder as documents.
- **Member 5 (Service A):** `catalog_seed.jsonl` contains every `movies` column (`genres` is a list; join with `|` for the current schema). Its precomputed training-only `rating_avg`/`rating_count` equal what `seed_catalog.py` computes, which removes the 20M-row scan. `display_title` (for example "The Matrix") and `popularity_rank` are added for search and display.
- **Member 2 (evaluation):** the evidence for checklist item A3 is `p2_quality_report.json`, together with `p2_samples.report.json` and `catalog_seed.report.json`, which hold input hashes, config, code commit, counts, and output SHA-256 values. `configs/validation_users_300_v1.json` and `configs/test_users_500_v1.json` are byte-identical copies from `member2-phase2`, used only for exclusion.

### Output SHA-256

| File | SHA-256 |
|---|---|
| `p2_samples_train.jsonl` | `04bd696861dcaa8df78626f4bc1ef67c37f37dd115b279cd85736278e94dcb2b` |
| `p2_samples_validation.jsonl` | `63583a11c05be03303dc622afb996f6204ba590056464b105b7ff677c3715501` |
| `p2_episodes_train.jsonl` | `093b9848c5e7dc54da245b26557350efde9fac502c71c8a17f7446978122d614` |
| `p2_episodes_validation.jsonl` | `241e3e5dae9acf56d2da2ad9462aa229996e5fb7be38368476c602935447c8a2` |
| `catalog_seed.jsonl` | `3efd7d4199027843c37d8fcd98d69e15977b553c05c3f32a9ca8a65d2e1b07dc` |
