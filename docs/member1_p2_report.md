# Member 1 — Phase 2 data handoff: P2 samples, catalog seed, quality report

**Status:** `p2-samples-v2`, generated and checked on 2026-11-22 with zero check errors from commit `94a9740`. The method follows the project proposal, midterm report §3.4 and H2, and Member 2's proposed Phase 2 protocol. `p2-v2` supersedes `p2-v1` (2026-11-14), which was never handed off. v2 adds 25%/50% masked profiles and per-movie scoring samples. The field guide for the handoff ZIP is [member1_p2_handoff.md](member1_p2_handoff.md).

| Deliverable | Location (Git-ignored data) | Code |
|---|---|---|
| P2 training samples covering six action types | `data/processed/p2-v2/p2_samples_{train,dev}.jsonl` | `src/data/p2_samples.py` |
| Per-movie scoring samples (P1 prompt plus clarifications) for ranking | `data/processed/p2-v2/p2_score_samples_{train,dev}.jsonl` | same |
| Simulated episodes in `p2-trace-v1` event vocabulary | `data/processed/p2-v2/p2_episodes_{train,dev}.jsonl` | same |
| Data-quality and temporal-boundary report | `data/processed/p2-v2/p2_quality_report.json` plus this document | `src/data/p2_checks.py` |
| Movie catalog seed | `data/processed/p2-v2/catalog_seed.jsonl` | `src/data/catalog_seed.py` |
| Handoff ZIP with checksums | `data/processed/handoff/member1_p2_v2_handoff.zip` | `src/data/p2_handoff.py` |
| Shared prompt renderers, action grammar, and simulator | — | `src/profiles/agent_text.py`, `src/data/p2_actions.py`, `src/data/clarification_sim.py` |

Reproduce from the repository root after `python -m src.data.prepare`. The raw CSVs must be extracted in `data/raw/ml-25m/`.

```bash
python -m src.data.p2_samples && python -m src.data.catalog_seed && python -m src.data.p2_checks && python -m src.data.p2_handoff
```

The run takes about 20 minutes on a laptop: 10 for generation, 10 for the full replay check. It uses the standard library only. Committed for review:

- a [full example episode](p2_sample_examples.jsonl): a 50%-masked profile with two questions and a wider retrieval, ten samples covering all six tasks;
- [per-movie scoring examples](p2_score_sample_examples.jsonl), with and without clarifications.

## 1. Consistency with the proposal and midterm report

| Commitment | Source | Implementation |
|---|---|---|
| The agent decides for itself whether the information is sufficient and asks a short clarifying question only when it is not | Proposal; midterm §1 | `assess_sufficiency` and `ask_clarification` samples. Labels come from held-out outcomes, so no fixed rule such as a rating count decides. |
| Cap the number of follow-up questions | Proposal risk 1 | At most **2** clarification turns. |
| Narrow to a shortlist, then rank and explain | Proposal | `evaluate_candidates`, `rank_candidates`, `explain_recommendations`. |
| Fall back to a larger shortlist when narrowing misses good movies | Proposal risk 2; midterm §3.4 | At most **1** wider retrieval, expanding the pool from **50 to at most 100** candidates. |
| Action supervision compares sampled action sequences on held-out training-period interactions, choosing higher **NDCG@10** and **fewer actions on ties** | Midterm §3.4 | Implemented exactly; see §2. |
| Template questions about preferences masked from earlier history; answers come only from that hidden history | Midterm §3.4; roadmap §6 | Masked and sparse profiles plus the template-driven simulator `pre_cutoff_history_sim_v1`. |
| Profile masks at 25% and 50% | Midterm H2 | `masked_25` / `masked_50` episodes hide 25% / 50% of the canonical profile's liked and disliked entries. |
| P2 starts from the unchanged canonical profile | Member 2 protocol | `canonical` episodes (40%) use the frozen formatter exactly (parity check below). |
| "LoRA learns action/question outputs"; "clarification updates scoring on fixed candidates" | Midterm §3.4 | Listwise `rank_candidates` over the fixed pool after clarifications, which is the format the team built. Per-movie scoring samples (P1 prompt plus clarifications) are kept as a backup. |
| Validation/test targets never supply answers or action labels | Midterm §3.4 | Both splits (`train`, and a user-disjoint `dev` for loss monitoring) read **training ratings only**. The generator never opens validation or test rating files. |
| Select models on validation | Midterm §5; Member 2 protocol | Final checkpoint choice on the 300 validation users; `dev` is for screening. |
| Explanations use supplied history and metadata and are audited for support | Proposal; midterm §3.4, H4 | Templates cite only displayed items. An independent grounding check covers all 43,740 explanations. Member 2's two-reviewer audit is still planned. |
| Train P2 on train-only examples | Midterm §5 (Nov 4–10) | Same as above. |
| Simulation cannot establish real-user satisfaction | Midterm §3.4 | Stated as a limitation (§5). |

## 2. How samples are built

Each **episode** fixes a user, an anchor time `t` inside the training period, and a visible profile:

- **History:** every rating with timestamp `< t`.
- **Held-out interactions:** the user's training ratings in `[t, validation cutoff)`. They choose the labels but never appear in a prompt.
- **Profile level** (drawn per episode):
  - `canonical` (40%): exactly the frozen formatter over the whole history; the offline P2 start state.
  - `masked_25` / `masked_50` (20% each): the canonical profile with 25% / 50% of its liked entries and of its disliked entries hidden. The hidden entries are chosen by hash and rounded half up, so 10 liked → 3 / 5 hidden and 5 disliked → 1 / 3. They are not replaced, so clarification can reveal them (midterm H2).
  - `sparse` (20%): the formatter over only the last k ∈ {1, 2, 3, 5, 8} catalog ratings, like a new app user.

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
- **Per-movie scoring** (`p2_score_samples_*`): for each episode, the first 10 held-out positives and first 5 held-out explicit dislikes. The P1 ranking prompt is used verbatim, with `profile` = the episode's profile text plus `Clarifications:` Q/A lines after its chosen questions. Labels are `relevant` (≥ 4) or `not_relevant` (≤ 2.5), with 3/3.5 excluded, as in P1. Episodes without questions give prompts byte-identical to P1's, so the P1 adapter and scorer carry over.

**Simulator `pre_cutoff_history_sim_v1`.**

- Answers come only from history before `t`. It raises an error on later ratings.
- It prefers facts the visible text does not show, and repeating an intent reveals the next fact.
- Unrecognized or exhausted questions get declared noninformative answers.

## 3. Results

| | train | dev |
|---|---:|---:|
| Users / episodes | 2,000 / 3,974 | 400 / 400 |
| Action samples | 22,802 | 2,178 |
| `assess_sufficiency` (ask / proceed) | 6,675 (2,701 / 3,974) | 628 (228 / 400) |
| `ask_clarification` | 2,701 | 228 |
| `evaluate_candidates` (retrieve / proceed) | 4,726 (752 / 3,974) | 461 (61 / 400) |
| `retrieve_candidates` | 752 | 61 |
| `rank_candidates` / `explain_recommendations` | 3,974 / 3,974 | 400 / 400 |
| Per-movie scoring samples (relevant / not_relevant) | 43,297 (33,213 / 10,084) | 4,259 (3,225 / 1,034) |
| Scoring samples after ≥ 1 clarification | 19,913 | 1,674 |
| Episodes: canonical / masked_25 / masked_50 / sparse | 1,610 / 786 / 807 / 771 | 160 / 93 / 71 / 76 |
| Chosen questions per episode: 0 / 1 / 2 | 2,385 / 477 / 1,112 | 265 / 42 / 93 |
| Episodes that ask: canonical / masked_25 / masked_50 / sparse | 36% / 38% / 44% / 47% | 34% / 31% / 27% / 43% |
| Episodes that retrieve: off-target / profile pool | 34% / 11% | 23% / 10% |
| Intents asked: liked example / genre / themes / era / dislikes | 1,013 / 705 / 585 / 227 / 171 | 97 / 59 / 45 / 10 / 17 |
| Answers revealing a fact absent from visible text | 1,948 of 2,701 (72%) | 166 of 228 (73%) |
| NDCG@10 of the no-action sequence is 0 / chosen sequence is 0 | 73% / 52% | 77% / 59% |
| Mean NDCG@10 gain of the chosen sequence over no action (proxy ranker) | 0.059 | 0.053 |
| Prompt characters, p50 / p95 / max: rank task (50–105 candidates) | 9,168 / 16,546 / 18,290 | 9,062 / 16,330 / 18,075 |
| Prompt characters, p50 / p95 / max: per-movie scoring | 1,447 / 2,183 / 2,512 | 1,437 / 2,188 / 2,468 |

The rule produces the intended behaviour:

- In train, ask rates rise as more of the profile is hidden: 36% canonical, 38% at 25% masked, 44% at 50% masked, and 47% for sparse profiles.
- Dev has only 71–160 episodes per level, so its per-level rates are noisy.
- Retrieval fires about three times as often on off-target pools as on profile pools.

Because held-out positives are sparse, most comparisons tie. Ties go to "proceed", so "proceed" labels outnumber "ask" and "retrieve". That is the midterm rule working as written. The same sparsity appears in evaluation: 76% of test pools contain no known positive (midterm §4.3).

## 4. Data-quality and temporal-boundary checks

`p2_checks` recomputes everything from source files rather than trusting the generator. Result: **`error_count: 0`**.

| Check | Result |
|---|---|
| Frozen `ml25m-global-v1` checksums and profile cutoffs (`src.data.verify`) | 10/10 hashes; 4,751 validation and 3,779 test profiles verified |
| Split time ranges | train max 2016-08-18 12:40:42 < validation cutoff 12:40:46 ≤ validation min. Validation max < test cutoff 2018-01-04 02:44:50 ≤ test min. |
| **Replay from boundary-truncated data** | All 4,374 episodes were rebuilt from training ratings cut at `t` (history) and at the validation cutoff (labels). Each reproduced its stored action samples, scoring samples, and episode **byte for byte**, so no rating outside those windows influenced any sample. |
| Labels from the training period only | Every episode's label window ends at the validation cutoff. The generator reads no validation or test rating file (checked by a unit test). |
| History, future, and visible items | History count and latest timestamp match, and all are `< t`. Labels lie in `[t, cutoff)`, and visible movies are in the history. |
| Formatter parity | The P2 canonical formatter reproduces the frozen text of 500 sampled validation profiles exactly, from training ratings alone. |
| Masked profiles | Recomputed independently from the canonical profile. Every masked entry belongs to it, the hidden counts equal 25% / 50% of liked and of disliked entries (rounded half up), and unmasked levels hide nothing. 8,583 entries are hidden in total. |
| Simulated answers | 2,929 answers replayed from pre-anchor history. All source movies are watched before `t`. No held-out positive title appears in profile or answer text (whole-title matching). |
| Candidates | Every initial, added, and final candidate is in the catalog and unwatched. Injected items are genuine held-out ratings. |
| Grammar, order, and labels | Every target parses strictly, and every completion is its canonical serialization. Every prompt re-renders exactly. Every episode passes the `p2-trace-v1` ordering rules with ≤ 8 control events. The asked intents and the retrieval choice equal the plan's chosen sequence, and ties never add actions. |
| Per-movie scoring samples | 47,556 / 47,556 checked. Each movie is a held-out rating in `[t, cutoff)`, in the catalog and unwatched. The label matches the rating (≥ 4 / ≤ 2.5, never 3–3.5). The selection is exactly the first 10 positives and first 5 dislikes. `profile` equals the replayed profile plus replayed answers. `candidate_document` equals `movie_documents.jsonl`. The prompt re-renders exactly. |
| Explanation grounding (independent parser) | 43,740 / 43,740 explanations cite only displayed titles, ratings, genres, tags, and answers. |
| Population | Train and dev users are disjoint. All 782 distinct frozen evaluation users (300 validation + 500 test, 18 in both) are excluded. |
| Catalog seed | 21,333 rows = catalog. Text equals `movie_documents.jsonl`. `rating_count` and `train_positives` equal `catalog.csv` training counts. |

**Restored artifact.** `train_ratings.csv` and the raw `ratings.csv` / `genome-scores.csv` were missing locally. They were restored from the MD5-verified archive and rebuilt, and all ten processed files matched the frozen manifest hashes.

### Data-quality findings for the team

1. **Test profiles show newer, out-of-catalog liked movies — a train/test shift.**
   - *Size:* 1,172 of 3,779 test profiles (31%) list at least one liked movie outside the catalog, against 174 of 4,751 validation profiles (3.7%).
   - *Cause:* 2,412 of the 2,520 such entries were rated after the validation cutoff. They are newer movies that the training-defined catalog cannot contain.
   - *Training data:* only 5 of 1,610 canonical P2 training profiles (0.3%) show any out-of-catalog entry, and Member 3's P1 profiles exclude them by construction.
   - *Status:* the formatter matches evaluation; the test distribution does not.
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
- Action prompts with 50–105 candidates reach about 18k characters, measured at up to 6,458 Qwen tokens including the completion. That is well above P1's 1,024-token limit and above the proposed 5,120. Per-movie scoring records stay under 800 tokens.
- Nearly every rank sample contains injected held-out movies: only 3 train and 0 dev rank samples are retriever-only.
- Per-movie scoring labels, like P1's, use explicitly rated movies only, not unrated pool members. About three quarters are `relevant`.
- Explanations are deterministic templates — grounded but plain.
- As in Phase 1, genome tags are static transductive metadata. User tags and catalog membership come from the whole pre-validation period, so an early anchor can see metadata dated after `t` (midterm §3.1).

## 6. Handoff notes

The handoff ZIP and its field guide ([member1_p2_handoff.md](member1_p2_handoff.md)) answer Member 6's request: files, examples, input vs. output fields, check results, and checksums.

- **Member 6 (P1 → P2 fine-tuning):**
  - Train action samples on `prompt` → `completion`. Train scoring samples on the P1 path (`profile`, `candidate_document`, `label`). Keep `provenance` out of the input.
  - Use `dev` as the eval set for loss and parse validity. The final checkpoint choice uses the 300 validation users.
  - `max_length` 6,656 for training, with no truncation; serve with 6,144 input and 1,024 output.
  - Rank listwise; the per-movie files are an optional backup.
  - "Proceed" labels dominate; consider task-level reweighting.
- **Member 4 (agent):**
  - Render prompts with `render_prompt` (`p2-prompt-v1`) or `render_score_prompt` (`p2-score-prompt-v1`). Parse with `parse_action`.
  - `validate_episode` implements the contract's ordering rules, and the episode files are ready-made traces for tests.
  - The offline runner answers questions with `clarification_sim.build_fact_pool` + `answer_question`, passing only ratings before the split cutoff.
  - Invalid-output proposal: D9.
- **Member 3 (retrieval / Service B):** embed the `catalog_seed.jsonl` `text` field. Retrieval queries are free text, so embed them with the same encoder.
- **Member 5 (Service A):** `catalog_seed.jsonl` contains every `movies` column (`genres` is a list; join with `|`), with the same training-only `rating_avg`/`rating_count` that `seed_catalog.py` computes. It adds `display_title` and `popularity_rank`.
- **Member 2 (evaluation):** the evidence for checklist A3 is `p2_quality_report.json`, together with `p2_samples.report.json` and `catalog_seed.report.json` (input hashes, config, commit, counts, output hashes). The frozen user lists in `configs/` are byte-identical copies used only for exclusion.

**Provenance note.** `p2_samples.report.json` records commit `94a9740` with `worktree_dirty: true`. During generation, only `src/data/README.md` and the new packager `src/data/p2_handoff.py` changed; neither is imported by the generator. `git diff 94a9740` is empty for every generator, check, simulator, and prompt module and for the config.

### Output SHA-256

| File | SHA-256 |
|---|---|
| `p2_samples_train.jsonl` | `4f7f859cee4b555d2d3796aded3f9e196229d2687523c2efa576da1f6ce5a8b1` |
| `p2_samples_dev.jsonl` | `e471065cc3ea220cd2316484b546f0ad286bdb1020fe4ef1985f4f04ac5a14f8` |
| `p2_score_samples_train.jsonl` | `8319917c3c1e623d6156270d98b7617684fa400bcdf964e9561bad8b3ad9afde` |
| `p2_score_samples_dev.jsonl` | `4cb9acb52c2493ac13a9c7f40f6c81f4c96f76f0f233978bfb756e1e6d8c1551` |
| `p2_episodes_train.jsonl` | `31478ad040409fe117a4667b9241a1e0a25933ebd539e1be6c19413b8150faaa` |
| `p2_episodes_dev.jsonl` | `af11576e285a140727a929c413c5846357ff03fbb0621376cf145b2b91239bc5` |
| `catalog_seed.jsonl` | `3efd7d4199027843c37d8fcd98d69e15977b553c05c3f32a9ca8a65d2e1b07dc` (unchanged from v1) |
