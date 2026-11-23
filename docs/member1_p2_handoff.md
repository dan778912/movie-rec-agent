# Member 1 P2 data handoff (`p2-samples-v2`)

For Member 6 (P1 → P2 fine-tuning), with notes for Members 2–5. The data files are Git-ignored and travel as `member1_p2_v2_handoff.zip`. This page is the ZIP's `README.md`.

**Verify after unzipping:** `shasum -a 256 -c SHA256SUMS.txt` must print `OK` for every file.

## What is in the package

| Path | Use |
|---|---|
| `data/p2_samples_train.jsonl` / `data/p2_samples_dev.jsonl` | **Action samples:** one agent decision per line, covering all six actions |
| `data/p2_score_samples_train.jsonl` / `data/p2_score_samples_dev.jsonl` | **Per-movie scoring samples:** P1 format plus clarifications, for ranking by scoring (decision D6) |
| `data/p2_episodes_train.jsonl` / `data/p2_episodes_dev.jsonl` | Whole simulated episodes in the `p2-trace-v1` event vocabulary. For Member 4's tests and audit; **not** training input. |
| `data/catalog_seed.jsonl` | One row per catalog movie, for Service A (Member 5) and the vector index (Member 3) |
| `reports/p2_quality_report.json` | Data-quality and leakage checks. `error_count` is 0. |
| `reports/p2_samples.report.json` | Generation receipt: config, input hashes, code commit, counts, label balance, prompt lengths |
| `reports/catalog_seed.report.json` | Seed receipt |
| `examples/p2_sample_examples.jsonl` | One complete episode: all six actions in order |
| `examples/p2_score_sample_examples.jsonl` | Per-movie scoring examples, with and without clarifications |
| `docs/` | Full method report and the team decision record |
| `SHA256SUMS.txt` | Checksums of everything above |

## Splits

- **`train`** and **`dev`** both anchor inside the training period and use **training ratings only**. Validation and test targets never supply answers or labels.
- **`dev` users are disjoint from `train` users.** The 300 validation and 500 test evaluation users are excluded from both.
- **Use `dev` as the eval set during training:** loss, action accuracy, and the share of outputs that parse. Do not take a random fraction of `train` (`validation_fraction` in the P1 config), because that would put one user's episodes on both sides.
- **The final checkpoint choice is separate.** It uses full-system NDCG@10 on Member 2's 300 validation users (decision D7).

## Counts

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
| … of which follow at least one clarification | 19,913 | 1,674 |
| Episodes by profile: canonical / 25% masked / 50% masked / sparse | 1,610 / 786 / 807 / 771 | 160 / 93 / 71 / 76 |

Prompt characters, p50 / p95 / max (train):

| Task | Characters |
|---|---|
| `assess_sufficiency`, `ask_clarification` | 1,426 / 2,140 / 2,469 |
| `evaluate_candidates` | 8,571 / 15,850 / 17,549 |
| `retrieve_candidates` | 8,477 / 9,478 / 10,058 |
| `rank_candidates` | 9,168 / 16,546 / 18,290 |
| `explain_recommendations` | 2,941 / 3,713 / 4,235 |
| `score_candidate` | 1,447 / 2,183 / 2,512 |

## Fields: model input vs. expected output

### Action samples (`p2_samples_*.jsonl`)

| Field | Role |
|---|---|
| `prompt` | **Model input.** Rendered by `src/profiles/agent_text.py::render_prompt` (`p2-prompt-v1`): system text, user profile, clarifications so far, candidate pool when the task needs one, task instruction, output format. |
| `completion` | **Expected output.** One JSON object, `action` first, e.g. `{"action": "assess_sufficiency", "sufficient": false}`. Train on `prompt` → `completion` (+ EOS), loss on the completion only. |
| `task` | Action name; use for per-task metrics or reweighting. Not model input. |
| `target` | `completion` parsed to an object (same content). Not model input. |
| `state` | `profile_text`, `clarifications`, `candidate_movie_ids`: the pieces `prompt` was rendered from. Not model input. |
| `sample_id`, `episode_id`, `step`, `split`, `user_id`, `schema_version`, `prompt_version` | Identifiers and versions. Not model input. |
| `provenance` | Audit only. It contains **held-out information** (label basis, future ratings, NDCG values). **Never put it in the input.** |

Output formats by task:

| `task` | `completion` |
|---|---|
| `assess_sufficiency` | `{"action": "assess_sufficiency", "sufficient": true\|false}` |
| `ask_clarification` | `{"action": "ask_clarification", "question": "<question>"}` |
| `evaluate_candidates` | `{"action": "evaluate_candidates", "pool_sufficient": true\|false}` |
| `retrieve_candidates` | `{"action": "retrieve_candidates", "query": "<search query>"}` |
| `rank_candidates` | `{"action": "rank_candidates", "movie_ids": [<every candidate ID once>]}` (listwise; see D6) |
| `explain_recommendations` | `{"action": "explain_recommendations", "explanations": [{"movie_id": <id>, "text": "<justification>"}, ...]}` for the top 10 in order |

Parse model output with `src/data/p2_actions.py::parse_action`, then `validate_action`. Both are strict: duplicate keys, NaN, extra fields, unknown actions, and duplicate or foreign IDs all raise `ActionError`.

### Per-movie scoring samples (`p2_score_samples_*.jsonl`)

| Field | Role |
|---|---|
| `profile` | **Model input.** The profile text, followed by `\n\nClarifications:\nQ: …\nA: …` when the episode asked questions. With no clarifications it equals the P1 profile text. |
| `candidate_document` | **Model input.** The canonical movie document, identical to `movie_documents.jsonl` `text`. |
| `prompt` | **Model input**, already rendered. It is the P1 `build_ranking_prompt(profile, candidate_document)` text exactly. |
| `label` / `completion` | **Expected output:** `relevant` (held-out rating ≥ 4) or `not_relevant` (≤ 2.5). Ratings 3/3.5 are excluded, as in P1. |
| `example_id`, `user_id`, `movie_id`, `split` | Identifiers. With `profile`, `candidate_document`, and `label` these are the fields the P1 loader requires. The P1 loader also requires `split == "train"`, so load `dev` with an eval-split variant. |
| `provenance` | Audit only. It contains the **held-out rating and its time**. **Never put it in the input.** |

Score with the P1 rule, the relevant-minus-not_relevant log-probability, then order the pool by score.

## Notes for training

- **Prompt length (D2).** Action prompts that list a 50–100 movie pool reach about 18k characters, roughly 4.6k tokens by estimate. Measure them with the pinned Qwen tokenizer.
  - The P1 trainer keeps the **last** `max_length` tokens. At 1,024 tokens it would delete the instructions and profile from those prompts.
  - **Proposed:** `max_length` 6,144 for action samples, and fail or report instead of truncating silently.
  - Scoring prompts are short.
- **Pick one ranking format (D6).** Recommended: train the per-movie scoring file, which keeps the P1 adapter's ranking behaviour, and drop the listwise `rank_candidates` samples. The listwise samples remain for comparison.
- **Label balance (D4).** "Proceed" outnumbers "ask"/"retrieve" because the midterm rule sends ties to fewer actions. Reweight by task or label if needed; do not relabel.
- **Profiles (D8).** Episodes mix four levels: canonical 40%, 25%-masked 20%, 50%-masked 20%, and sparse (last 1–8 ratings) 20%. These match the primary evaluation, midterm H2, and live app users. `provenance.level` gives the level per sample.

## Notes for other members

- **Member 4 (agent):**
  - Render prompts with `render_prompt` / `render_score_prompt` and parse with `parse_action`.
  - The offline simulator is `src/data/clarification_sim.py` (`pre_cutoff_history_sim_v1`, D1). Pass it only ratings before the split cutoff.
  - Invalid-output handling proposal: D9.
- **Member 3 (retrieval / Service B):** embed `catalog_seed.jsonl` `text`. Retrieval queries are free text.
- **Member 5 (Service A):** `catalog_seed.jsonl` has every `movies` column (`genres` is a list), with training-only `rating_avg`/`rating_count`.
- **Member 2 (evaluation):** checklist A3 evidence is `reports/p2_quality_report.json` plus the two receipts.

## Reproduce

From the repository root, after `python -m src.data.prepare` and with the raw CSVs extracted in `data/raw/ml-25m/`:

```bash
python -m src.data.p2_samples && python -m src.data.catalog_seed && python -m src.data.p2_checks && python -m src.data.p2_handoff
```

The run takes about 15–20 minutes, uses the standard library only, and is deterministic. Output hashes must equal `SHA256SUMS.txt`.
