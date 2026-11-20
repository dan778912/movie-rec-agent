# Member 1 — Phase 2 data handoff: P2 samples, catalog seed, quality report

**Status:** generated and checked on 2026-11-14 with zero check errors. The method follows the project proposal and midterm report §3.4. The items the team still needs to confirm are in [member1_p2_team_decisions.md](member1_p2_team_decisions.md).

| Deliverable | Location (Git-ignored data) | Code |
|---|---|---|
| P2 training samples covering six action types | `data/processed/p2-v1/p2_samples_{train,dev}.jsonl` | `src/data/p2_samples.py` |
| Simulated episodes in `p2-trace-v1` event vocabulary | `data/processed/p2-v1/p2_episodes_{train,dev}.jsonl` | same |
| Data-quality and temporal-boundary report | `data/processed/p2-v1/p2_quality_report.json` plus this document | `src/data/p2_checks.py` |
| Movie catalog seed | `data/processed/p2-v1/catalog_seed.jsonl` | `src/data/catalog_seed.py` |
| Shared prompt renderer, action grammar, and simulator | — | `src/profiles/agent_text.py`, `src/data/p2_actions.py`, `src/data/clarification_sim.py` |

Reproduce from the repository root after `python -m src.data.prepare`. The raw CSVs must be extracted in `data/raw/ml-25m/`.

```bash
python -m src.data.p2_samples && python -m src.data.catalog_seed && python -m src.data.p2_checks
```

The run takes about 15 minutes on a laptop: 7 for generation, 7.5 for the full replay check. It uses the standard library only. A [full example episode](p2_sample_examples.jsonl) (ten samples, all six tasks) is committed for review.

## 1. Consistency with the proposal and midterm report

| Commitment | Source | Implementation |
|---|---|---|
| The agent decides for itself whether the information is sufficient and asks a short clarifying question only when it is not | Proposal; midterm §1 | `assess_sufficiency` and `ask_clarification` samples. Labels come from held-out outcomes, so no fixed rule such as a rating count decides. |
| Cap the number of follow-up questions | Proposal risk 1 | At most **2** clarification turns. |
| Narrow to a shortlist, then rank and explain | Proposal | `evaluate_candidates`, `rank_candidates`, `explain_recommendations`. |
| Fall back to a larger shortlist when narrowing misses good movies | Proposal risk 2; midterm §3.4 | At most **1** wider retrieval, expanding the pool from **50 to at most 100** candidates. |
| Action supervision compares sampled action sequences on held-out training-period interactions, choosing higher **NDCG@10** and **fewer actions on ties** | Midterm §3.4 | Implemented exactly; see §2. |
| Template questions about preferences masked from earlier history; answers come only from that hidden history | Midterm §3.4; roadmap §6 | Masked (sparse) profiles plus the template-driven simulator `pre_cutoff_history_sim_v1`. |
| Validation/test targets never supply answers or action labels | Midterm §3.4 | Both splits (`train`, and a user-disjoint `dev` for checkpoint selection) read **training ratings only**. The generator never opens validation or test rating files. |
| Explanations use supplied history and metadata and are audited for support | Proposal; midterm §3.4, H4 | Templates cite only displayed items. An independent grounding check covers all 43,740 explanations. Member 2's two-reviewer audit is still planned. |
| Train P2 on train-only examples | Midterm §5 (Nov 4–10) | Same as above. |
| Simulation cannot establish real-user satisfaction | Midterm §3.4 | Stated as a limitation (§5). |

## 2. How samples are built

Each **episode** fixes a user, an anchor time `t` inside the training period, and a visible profile:

- **History:** every rating with timestamp `< t`.
- **Held-out interactions:** the user's training ratings in `[t, validation cutoff)`. They choose the labels but never appear in a prompt.
- **Canonical profile:** exactly the frozen formatter over the whole history; this is the offline P2 start state.
- **Masked (sparse-k) profile:** the formatter over only the last k ∈ {1, 2, 3, 5, 8} catalog ratings. The rest of the history stays hidden, matching the midterm's masked preferences and the live app's few-rating user.

**Action labels (midterm §3.4).**

- *Enumeration:* for each episode, the builder lists every action sequence within the budget — 0, 1, or 2 template questions over the intents liked example, favorite genre, themes, dislikes, and era (answers come from the simulator), followed by an optional wider retrieval. That is up to 62 sequences.
- *Scoring:* each sequence produces a final pool, which a preference-vector ranker orders. Its NDCG@10 is computed against all held-out in-catalog positives (median 14 per episode).
- *Choice:* the episode follows the sequence with the highest NDCG@10, and ties go to fewer actions.
- *What each sample records:*
  - every `assess_sufficiency` label is "ask" while the chosen sequence still has a question left;
  - `ask_clarification` targets the chosen intent;
  - `evaluate_candidates` / `retrieve_candidates` follow the chosen retrieval decision;
  - once the single retrieval is used, the pool is labelled sufficient — the budget ends the loop, which matches the midterm's fallback "exhausted budgets fall back to ranking available candidates".

**Ranker and pools.** The ranker uses IDF-weighted genre, genome-tag, user-tag, and decade vectors. Visible likes count +1, visible dislikes −0.5, and revealed facts add their own features. The initial 50-candidate pool is either:

- *profile:* retrieved from the starting profile, as the B3 analogue (64% of episodes);
- *off-target:* the most popular movies from the user's two weakest genres, simulating the retrieval miss in midterm §4.3 (36%).

The wider retrieval executes the model's query text and adds 50 candidates.

**Other targets.**

- `rank_candidates`: held-out positives first (rating ↓, earlier ↑). Unlabeled movies follow in pool order, and held-out explicit dislikes (≤ 2.5) go last. This matches the roadmap's relevant (≥ 4) / not relevant (≤ 2.5) rule. For ranking supervision, up to 3 held-out positives and 2 held-out dislikes are inserted into the final pool and recorded in `injected_movie_ids`.
- `explain_recommendations`: template text for the top 10. It cites shared genres/tags between the candidate and a displayed liked movie, or a clarification answer. When nothing overlaps, it says so honestly instead.

**Simulator `pre_cutoff_history_sim_v1`.**

- Answers come only from history before `t`. It raises an error on later ratings.
- It prefers facts the visible text does not show, and repeating an intent reveals the next fact.
- Unrecognized or exhausted questions get declared noninformative answers.

## 3. Results

| | train | dev |
|---|---:|---:|
| Users / episodes | 2,000 / 3,974 | 400 / 400 |
| Samples | 22,860 | 2,224 |
| `assess_sufficiency` (ask / proceed) | 6,718 (2,744 / 3,974) | 647 (247 / 400) |
| `ask_clarification` | 2,744 | 247 |
| `evaluate_candidates` (retrieve / proceed) | 4,712 (738 / 3,974) | 465 (65 / 400) |
| `retrieve_candidates` | 738 | 65 |
| `rank_candidates` / `explain_recommendations` | 3,974 / 3,974 | 400 / 400 |
| Chosen questions per episode: 0 / 1 / 2 | 2,358 / 488 / 1,128 | 251 / 51 / 98 |
| Episodes that ask: canonical / masked profile | 35% / 47% | 33% / 42% |
| Episodes that retrieve: off-target / profile pool | 33% / 11% | 26% / 10% |
| Intents asked: liked example / genre / themes / era / dislikes | 979 / 712 / 613 / 243 / 197 | 79 / 73 / 59 / 16 / 20 |
| Answers revealing a fact absent from visible text | 2,098 of 2,744 (76%) | 179 of 247 (72%) |
| NDCG@10 of the no-action sequence is 0 / chosen sequence is 0 | 73% / 51% | 75% / 54% |
| Mean NDCG@10 gain of the chosen sequence over no action (proxy ranker) | 0.063 | 0.059 |
| Prompt characters, p50 / p95 / max (rank task, 50–105 candidates) | 9,099 / 16,462 / 18,290 | 8,982 / 16,262 / 18,075 |

The rule produces the intended behaviour:

- Masked profiles ask more often than full profiles.
- Retrieval fires about three times as often on off-target pools as on profile pools.

Because held-out positives are sparse, most comparisons tie. Ties go to "proceed", so "proceed" labels outnumber "ask" and "retrieve". That is the midterm rule working as written, and the same sparsity appears in evaluation: 76% of test pools contain no known positive (midterm §4.3).

## 4. Data-quality and temporal-boundary checks

`p2_checks` recomputes everything from source files rather than trusting the generator. Result: **`error_count: 0`**.

| Check | Result |
|---|---|
| Frozen `ml25m-global-v1` checksums and profile cutoffs (`src.data.verify`) | 10/10 hashes; 4,751 validation and 3,779 test profiles verified |
| Split time ranges | train max 2016-08-18 12:40:42 < validation cutoff 12:40:46 ≤ validation min. Validation max < test cutoff 2018-01-04 02:44:50 ≤ test min. |
| **Replay from boundary-truncated data** | All 4,374 episodes were rebuilt from training ratings cut at `t` (history) and at the validation cutoff (labels). Each reproduced its stored samples and episode **byte for byte**, so no rating outside those windows influenced any sample. |
| Labels from the training period only | Every episode's label window ends at the validation cutoff. The generator reads no validation or test rating file (checked by a unit test). |
| History, future, and visible items | History count and latest timestamp match, and all are `< t`. Labels lie in `[t, cutoff)`, and visible movies are in the history. |
| Formatter parity | The P2 canonical formatter reproduces the frozen text of 500 sampled validation profiles exactly, from training ratings alone. |
| Simulated answers | 2,991 answers replayed from pre-anchor history. All source movies are watched before `t`. No held-out positive title appears in profile or answer text (whole-title matching). |
| Candidates | Every initial, added, and final candidate is in the catalog and unwatched. Injected items are genuine held-out ratings. |
| Grammar, order, and labels | Every target parses strictly, and every completion is its canonical serialization. Every prompt re-renders exactly. Every episode passes the `p2-trace-v1` ordering rules with ≤ 8 control events. The asked intents and the retrieval choice equal the plan's chosen sequence, and ties never add actions. |
| Explanation grounding (independent parser) | 43,740 / 43,740 explanations cite only displayed titles, ratings, genres, tags, and answers. |
| Population | Train and dev users are disjoint. All 782 distinct frozen evaluation users (300 validation + 500 test, 18 in both) are excluded. |
| Catalog seed | 21,333 rows = catalog. Text equals `movie_documents.jsonl`. `rating_count` and `train_positives` equal `catalog.csv` training counts. |

**Restored artifact.** `train_ratings.csv` and the raw `ratings.csv` / `genome-scores.csv` were missing locally. They were restored from the MD5-verified archive and rebuilt, and all ten processed files matched the frozen manifest hashes.

### Data-quality findings for the team

1. **Test profiles show newer, out-of-catalog liked movies — a train/test shift.**
   - *Size:* 1,172 of 3,779 test profiles (31%) list at least one liked movie outside the catalog, against 174 of 4,751 validation profiles (3.7%).
   - *Cause:* 2,412 of the 2,520 such entries were rated after the validation cutoff. They are newer movies that the training-defined catalog cannot contain.
   - *Training data:* only 0.1% of canonical P2 training profiles show one, and Member 3's P1 profiles exclude them by construction.
   - *Status:* the formatter matches evaluation; the test distribution does not. See decision D3.
2. **Catalog metadata gaps.** Of 21,333 catalog movies:
   - 8,443 lack genome tags and 4,054 lack user tags; 3,166 have no displayable tag at all;
   - 111 have no genre and 9 have no parseable year;
   - 20 title strings are duplicated (distinct MovieLens IDs);
   - 32 have no TMDB ID.

   Explanations for those movies fall back to genre-only or no-overlap text.
3. **40 user tags contain literal double quotes** (for example `"damn dirty apes"`). Explanations and simulator answers never cite them. The checker caught this.

## 5. Limitations

- Clarification answers are simulated from earlier ratings, not real replies, so they support only a controlled offline claim (midterm §3.4).
- NDCG@10 in the action labels comes from a lightweight preference-vector ranker, not the P2 model. The labels show which actions helped that ranker reach held-out positives. Held-out positives are sparse, so "proceed" dominates through ties.
- Training pools come from the simulator's feature index, not Member 3's vector index or the frozen B3 pools. Off-target pools simulate a retrieval miss. Ranking pools contain injected held-out items.
- Prompts with 50–105 candidates reach about 18k characters (~4.6k tokens), well above P1's 1,024-token limit.
- Explanations are deterministic templates — grounded but plain.
- As in Phase 1, genome tags are static transductive metadata. User tags and catalog membership come from the whole pre-validation period, so an early anchor can see metadata dated after `t` (midterm §3.1).

## 6. Handoff notes

- **Member 6 (P1 → P2 fine-tuning):**
  - Train on `prompt` → `completion`. Keep `provenance` out of the input.
  - Use `p2_samples_dev.jsonl` for checkpoint selection.
  - `max_length` must be about 5k tokens, or rank/evaluate prompts must be truncated or chunked (decision D2).
  - "Proceed" labels dominate; consider task-level reweighting.
- **Member 4 (agent):**
  - Render inference prompts with `src.profiles.agent_text.render_prompt` (`p2-prompt-v1`) and parse output with `src.data.p2_actions.parse_action`.
  - `validate_episode` implements the contract's ordering rules, and the episode files are ready-made traces for tests.
  - The offline runner can answer questions with `clarification_sim.build_fact_pool` + `answer_question`, passing only ratings before the split cutoff.
- **Member 3 (retrieval / Service B):** the `catalog_seed.jsonl` `text` field is the document to embed. Retrieval queries are free text, so embed them with the same encoder.
- **Member 5 (Service A):** `catalog_seed.jsonl` contains every `movies` column (`genres` is a list; join with `|`), with the same training-only `rating_avg`/`rating_count` that `seed_catalog.py` computes. It adds `display_title` and `popularity_rank`.
- **Member 2 (evaluation):** the evidence for checklist A3 is `p2_quality_report.json`, together with `p2_samples.report.json` and `catalog_seed.report.json` (input hashes, config, commit, counts, output hashes). The frozen user lists in `configs/` are byte-identical copies used only for exclusion.

### Output SHA-256

| File | SHA-256 |
|---|---|
| `p2_samples_train.jsonl` | `2302345d954ead9c4830680a7d1de15b9f2ae5e74186de011567cc2f9b3d8a75` |
| `p2_samples_dev.jsonl` | `dfeea7215fbf4ad30a0a09c2fb95c210ef704e713eaca433141361c534ade454` |
| `p2_episodes_train.jsonl` | `99b44acb8b843b757f23c16de405e71e02a5b3aaa356c55fbb644965ba5c9aad` |
| `p2_episodes_dev.jsonl` | `750df33095cc0c716862f21959caafa639f3ebb544b745865ebc26b8416739ea` |
| `catalog_seed.jsonl` | `3efd7d4199027843c37d8fcd98d69e15977b553c05c3f32a9ca8a65d2e1b07dc` |
