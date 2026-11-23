# P2 training data — decision record (Member 1)

**Status (2026-11-23):** P2 data is `p2-samples-v2`. Updated after reading the P2 branches that Members 3, 4 and 6 pushed on 2026-11-20 and the team chat to 2026-11-22. Prompt lengths now use real Qwen token counts (D2). D6 is withdrawn because the team built the listwise path. New items are D11 (rank-step injection) and D12 (budget exhaustion). It follows the project proposal, midterm report §3.4/§5 (H2), and Member 2's proposed Phase 2 protocol on `origin/member2-phase2`. Items marked *decided* are Member 1's own data choices. Items marked *proposed* need the named owners to confirm. All must be frozen before P2 test inference. Method and results: [member1_p2_report.md](member1_p2_report.md). Handoff package: [member1_p2_handoff.md](member1_p2_handoff.md).

## What teammates' branches already settle

Checked on 2026-11-23 after `git fetch --all`.

| Branch (last commit) | What it settles for P2 data | Status there |
|---|---|---|
| `member2-phase2` (`9032fd1`, Nov 18): Phase 2 protocol, interface contract, checkpoint-selection handoff | Primary metric is macro NDCG@10 (P2 minus P1, paired bootstrap) on the saved 300 validation users for decisions and 500 test users for the final result. P2 starts from the **unchanged canonical profile** and the **exact frozen B3 top-50**. One frozen, versioned simulator. Fallback ranking = frozen B3 order. At most 8 control events plus rank and explain. Checkpoint rule: shortlist on Member 1's dev, select by NDCG@10 on the 300 users, ties to the earlier step. | "Proposed, not frozen or team approved" |
| `member-6-p2-training` (`024122f`, Nov 20): P2 continued-LoRA trainer | Reads `p2_samples_{train,dev}.jsonl` (all six actions, listwise rank). `max_length` 5,120 with `allow_truncation: false`. Any over-length record **stops the run**; nothing is cut. Shortlists 3 checkpoints by `eval_loss`. Base revision `775b11af`. Its loader accepts Member 1's v2 files unchanged: 22,802 / 2,178 records, users disjoint (checked 2026-11-23). | Code; config version `p2-qwen-lora-v1` |
| `member-6-untuned-qwen-control` (Nov 13) and chat (Nov 19) | Untuned control re-exported as `qwen_base` at revision `989aa79`. That revision has the same `model.safetensors`, `config.json` and `tokenizer.json` as `775b11af`; only `tokenizer_config.json` differs. | Done |
| `member4-p2-agent` (`824c67a`, Nov 20): controller, `decide`/`resume`, action reference | Built on Member 1's `321c284`, uses `render_prompt` and `parse_action` unchanged. Strict parsing (A9). Rankings must be a permutation of the shown pool (A6). Limits 2 questions / 1 retrieval / 100 pool / 8 control events. Member 4 notes 8 is exactly the worst case of 2 + 1 (A2). Exhaustion rule open (A1, D12 below). Offline runner unowned (A12). | "Draft for team discussion" |
| `member-3-service-b-retrieval` (`f66d683`, Nov 20): Service B, retrieval, offline producer | Checkpoint = HF base + one effective PEFT adapter + tokenizer files. `model.generate(prompt) -> str`: raw `p2-prompt-v1` text, no chat template, greedy, one JSON action out. Provisional limits 6,144 input / 2,048 output tokens. Wraps Member 1's simulator. Includes an offline prediction/trace/manifest producer. | "Development proposals, not team-approved" |
| `member5-service-a`, `member5-scalability` (Nov 21–22) | Service A calls `POST /decide`; expects `ask` or `recommend`. | Provisional |

## Settled by the proposal, midterm report, and Member 2's protocol

| Topic | Decision | Source |
|---|---|---|
| Action labels | Compare every action sequence within the budget on NDCG@10 against held-out training-period interactions; choose the highest, preferring fewer actions on ties. No hand-set thresholds. | Midterm §3.4 |
| Budgets | At most 2 clarification turns and 1 wider retrieval | Proposal risk 1; midterm §3.4 |
| Pool sizes | Initial pool of 50, one expansion to at most 100 | Midterm §3.4 |
| Label sources | Training ratings only. Validation/test targets never supply answers or labels. | Midterm §3.4 |
| Clarification questions | Templates about preferences masked from earlier history; answers only from that hidden history | Midterm §3.4; roadmap §6 |
| Budget exhaustion | Proceed to ranking the available candidates | Midterm §3.4 |
| Evaluation start state | Unchanged canonical profile and exact frozen B3 pool; do not change the frozen profile rule | Member 2 protocol |
| Claims | Simulated answers cannot establish real-user satisfaction | Midterm §3.4; Member 2 protocol |

## Decisions

### D1. Clarification simulator — *proposed* (Members 2, 4)

`pre_cutoff_history_sim_v1` already meets every simulator requirement in Member 2's protocol. It:

- answers only from history before the cutoff;
- prefers facts absent from the visible profile text;
- returns declared noninformative answers;
- records source movie IDs for audit.

Member 2's contract example already uses this `answer_source` name.

**Proposal:** freeze `src/data/clarification_sim.py` at Git blob `be5011ece7fdaa6de22edeb874ae950dd539282d`. The file is byte-identical on `member-1-samples-p2`, `member4-p2-agent` and `member-3-service-b-retrieval`, so training and evaluation answers already match. The midterm's "genre comparison" questions are covered by the favorite-genre and dislike intents.

### D2. Prompt length — *proposed* (Members 3, 4, 6)

**Measured, not estimated.** Token counts below use the pinned tokenizer (`Qwen/Qwen2.5-1.5B-Instruct` at `775b11af`, `tokenizer.json` identical at `989aa79`). Each total is counted the way Member 6's trainer counts it: prompt + completion + `<|im_end|>`.

| Task (train) | Records | Total p50 / p95 / max | Max prompt | Max completion | Over 5,120 | Over 6,144 |
|---|---:|---|---:|---:|---:|---:|
| `rank_candidates` | 3,974 | 3,248 / 5,906 / 6,458 | 5,735 | 747 | 745 | 27 |
| `evaluate_candidates` | 4,726 | 2,723 / 5,037 / 5,547 | 5,532 | 15 | 153 | 0 |
| `retrieve_candidates` | 752 | 2,744 / 3,047 / 3,193 | 3,102 | 126 | 0 | 0 |
| `explain_recommendations` | 3,974 | 1,423 / 1,735 / 2,082 | 1,296 | 905 | 0 | 0 |
| `assess_sufficiency`, `ask_clarification` | 9,376 | ≈ 420 / 660 / 771 | 754 | 35 | 0 | 0 |
| `score_candidate` (per-movie file) | 43,297 | 433 / 675 / 795 | 793 | 4 | 0 | 0 |

Dev: 75 records exceed 5,120 (60 rank, 15 evaluate), and 4 rank records exceed 6,144 (max 6,242).

- **5,120 does not fit.** Member 6's trainer stops when any record is too long, so the current config would fail on 898 train and 75 dev records. Evaluate prompts on 100-movie pools alone reach 5,532 tokens.
- **Proposal:**
  - **Training:** `max_length` **6,656**. The longest record is 6,458, leaving 198 tokens. Keep `allow_truncation: false`.
  - **Serving (Service B):** input **6,144**, output **1,024**. Inference pools are at most 100 with nothing injected. The longest training prompt, 5,735, includes up to 105 candidates. The longest completion is an explanation at 905. Member 3's 2,048 output is also safe.
  - Member 6 should rerun `--length-check-only` to confirm, and add the GPU memory smoke test.

### D3. Test-time profile shift — *decided: report as a limitation* (Members 2, 3, 6)

| Profiles | With ≥ 1 out-of-catalog liked movie | Out-of-catalog share of liked entries |
|---|---:|---:|
| Frozen validation evaluation profiles | 174 / 4,751 (3.7%) | 0.5% |
| Frozen test evaluation profiles | 1,172 / 3,779 (31.0%) | 6.7% |
| P2 training samples, canonical (any visible entry) | 5 / 1,610 (0.3%) | ≈ 0% |

- **Cause:** 2,412 of the 2,520 out-of-catalog liked entries in test profiles were rated after the validation cutoff.
- **Not an option:** Member 2's protocol forbids changing the frozen profile rule.
- **Remedy:** report as a limitation, consistent with the 67.76% test catalog coverage in midterm §4.3.

### D4. Class balance — *decided: no relabelling* (Member 6)

Most sequence comparisons tie, and ties go to "proceed" (midterm rule). Member 6 may reweight by task or label. Counts are in the report.

### D5. Final-report wording — *proposed* (all)

- **Describe labels as:** "NDCG@10 comparison of action sequences under a lightweight preference-vector ranker".
- **Say:** answers are simulated.
- **Report:**
  - question frequency;
  - the share of answers revealing a new fact;
  - retrieval frequency.
- **Do not claim:** real-user satisfaction or question quality.

### D6. Ranking format — *decided: listwise* (Member 1; withdrawn recommendation)

On 2026-11-22 Member 1 recommended per-movie scoring for the rank step. The team has since built the listwise path end to end:

- Member 3's Service B exposes `generate(prompt) -> str`, returning one JSON action.
- Member 4's controller accepts a `rank_candidates` permutation of the shown pool.
- Member 6's trainer uses the listwise samples.

Switching now would need a scoring endpoint and controller changes for an unmeasured gain, so Member 1 withdraws the recommendation. **Listwise `rank_candidates` is the ranking format.**

The per-movie files (`p2_score_samples_*`) stay in the package as an optional backup and ablation. They are worth revisiting only if the real checkpoint's rank-step parse validity is poor. Nobody needs to train on them.

### D7. Checkpoint selection — *agreed in principle* (Members 1, 2, 6; Members 3, 4 to confirm)

**Correction:** an earlier version of this page said checkpoint selection uses Member 1's `dev` split. The midterm says "select models on validation", and Member 2's protocol reserves the 300 validation users for decisions.

Member 2's handoff (Nov 18) and Member 6's proposal (Nov 20) state the same rule, and Member 1 agrees:

1. Shortlist the **3** checkpoints with the lowest completion loss on the 400-user `dev` split. Report strict JSON parse rates per action separately.
2. Select among the shortlist by macro NDCG@10 on the fixed 300 validation users with Member 2's evaluator. An exact tie goes to the earlier training step.
3. Never use test.

Freeze the rule before any 300-user result is read. **Caveat:** every dev rank sample contains injected items (D11), so dev loss on the rank step does not measure retriever-only pools. Member 6's planned retriever-only check covers that gap.

### D8. Training profile mix — *decided* (Member 1)

Episodes now mix four profile levels:

| Level | Share | What it shows | Evaluation condition it matches |
|---|---:|---|---|
| canonical | 40% | Frozen canonical profile | Primary evaluation (Member 2) |
| masked_25 | 20% | Canonical profile with 25% of entries hidden | Midterm H2 |
| masked_50 | 20% | Canonical profile with 50% of entries hidden | Midterm H2 |
| sparse | 20% | Last 1–8 ratings | New app user (live demo) |

In masked profiles, 25% / 50% of liked and of disliked entries are hidden, not replaced. The simulator can reveal them.

### D9. Invalid model output — *implemented by Member 4* (Members 2, 4)

Member 4's controller already does what Member 1 proposed:

- strict `parse_action` / `validate_action`, with no silent repair (A9);
- no retry under greedy decoding;
- the contract fallback to frozen B3 order, with the failure recorded.

Member 2 confirms the contract side.

### D10. Training pools vs. B3 pools — *open* (Members 3, 6)

- **The mismatch:** training pools come from Member 1's dependency-free feature index. Evaluation starts from frozen MiniLM B3 pools.
- **Option:** regenerate training pools with the B3 encoder for training users. This needs `sentence-transformers` and Member 3's index.
- **Until then:** a stated limitation.

### D11. Rank-step injection — *facts confirmed; change proposed* (Members 1, 4, 6)

This answers Member 4's questions (A6) and Member 6's three requests. Counts are from the v2 files.

| | train | dev |
|---|---:|---:|
| Rank samples with injected items | 3,971 / 3,974 | 400 / 400 |
| **Retriever-only rank samples** | **3** | **0** |
| Rank-target positives that were injected | 11,817 / 20,959 (56%) | 1,195 / 1,994 (60%) |
| Episodes with no retrieved positive (injection is the only signal) | 1,669 (42%) | 194 (49%) |

- **Purpose.** Simulated pools often hold no held-out positive. Injection guarantees every rank sample has something to put first and something to put last. Member 4's guess is right.
- **Training period only: yes.** Injected movies are the user's held-out ratings in `[anchor, boundary)`. The boundary sits inside the training period, and `p2_checks` verifies every one is a training rating. No validation or test target is used.
- **Unmarked: yes, by content.** Injected movies go to random positions and are displayed exactly like retrieved ones. Two indirect signals remain:
  - the pool size: 51–55 or 101–105 movies, where inference shows 50 or 100;
  - injected movies come from the user's own future, not from the retriever, so they can look different from their neighbours.
- **Retriever-only examples: no.** There are 3 in train and none in dev. Dev rank loss therefore never sees an inference-like pool.
- **Explanations.** The explain step mostly covers injected positives (they are ranked first). Its templates use only the movie's metadata and the visible profile and clarifications, never ratings, so they apply unchanged to retrieved movies.

**Proposed change (`p2-samples-v3`, not yet generated):**

1. Injected movies **replace** random unlabeled pool members instead of being added, so pool sizes match inference (50 or 100). This removes the size signal.
2. In about half of the episodes that already have at least one retrieved positive, do not inject. That gives about 1,150 train and 100 dev retriever-only rank samples, marked by an empty `injected_movie_ids`.
3. Member 6 reports dev rank loss separately for the retriever-only subset.

### D12. Budget exhaustion — *Member 1's position; Member 2 decides* (Members 1, 2, 4)

This is Member 4's A1. Member 1 supports the midterm rule: **the model ranks the available pool**. The trace is labelled distinctly (Member 4's `budget_exhausted_rank`), and B3-order fallback is kept for errors only. Reasons:

- It is what the submitted midterm says.
- With a B3 fallback, exhausted users would score on B3's order and dilute what P2's headline measures.
- The training labels already proceed at the budget: 1,112 train `assess_sufficiency` samples say "sufficient" after 2 questions, and all 752 `evaluate_candidates` samples after the retrieval say the pool is sufficient. A trained model should rarely reach exhaustion. Member 4's 177 / 515 came from a scripted stand-in.

Either way, report the exhaustion count. Member 2 owns the contract label.

## Decision record

| Item | Decision | Who / date |
|---|---|---|
| D1 simulator freeze | Proposed: freeze `pre_cutoff_history_sim_v1`, blob `be5011ec` (identical on Members 1, 3, 4 branches) | Member 1, 2026-11-23; awaiting Members 2, 4 |
| D2 prompt length | Proposed: train `max_length` 6,656; serve 6,144 in / 1,024 out; no truncation (measured with the real tokenizer) | Member 1, 2026-11-23; awaiting Members 3, 6 |
| D3 profile shift | Report as limitation | Member 1, 2026-11-22 (consistent with Member 2 protocol) |
| D4 class balance | No relabelling; reweighting optional | Member 1, 2026-11-22 |
| D5 report wording | Proposed wording above | Awaiting team |
| D6 ranking format | Listwise; per-movie files optional backup | Member 1, 2026-11-23 (matches Members 3, 4, 6 code) |
| D7 checkpoint selection | Dev top-3 by loss, then NDCG@10 on 300 validation users, tie to earlier step | Members 1, 2, 6 agree; awaiting Members 3, 4 |
| D8 profile mix | 40 / 20 / 20 / 20 canonical / masked_25 / masked_50 / sparse | Member 1, 2026-11-22 |
| D9 invalid output | No retry, contract fallback (implemented by Member 4) | Awaiting Member 2 |
| D10 training pools | Open | Members 3, 6 |
| D11 rank-step injection | Facts confirmed; v3 change proposed (replace not add; retriever-only share) | Member 1, 2026-11-23; awaiting Members 4, 6 |
| D12 budget exhaustion | Member 1 supports "rank available pool", labelled; B3 fallback for errors only | Awaiting Member 2 |
