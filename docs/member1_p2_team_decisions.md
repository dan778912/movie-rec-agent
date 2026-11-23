# P2 training data — decision record (Member 1)

**Status (2026-11-22):** P2 data is `p2-samples-v2`. It follows the project proposal, midterm report §3.4/§5 (H2), and Member 2's proposed Phase 2 protocol on `origin/member2-phase2`. Items marked *decided* are Member 1's own data choices. Items marked *proposed* need the named owners to confirm. All must be frozen before P2 test inference. Method and results: [member1_p2_report.md](member1_p2_report.md). Handoff package: [member1_p2_handoff.md](member1_p2_handoff.md).

## What teammates' branches already settle

Checked on 2026-11-22 after `git fetch --all`.

| Branch (last commit) | What it settles for P2 data | Status there |
|---|---|---|
| `member2-phase2` (Nov 6) — Phase 2 protocol, interface contract, validation config draft | Primary metric is macro NDCG@10 (P2 minus P1, paired bootstrap) on the saved 300 validation users for decisions and 500 test users for the final result. P2 starts from the **unchanged canonical profile** and the **exact frozen B3 top-50**. One frozen, versioned simulator that answers only from pre-cutoff history absent from the visible profile. Fallback ranking = frozen B3 order. At most 8 control events plus rank and explain. | "Proposed, not frozen or team approved" |
| `member-6-untuned-qwen-control` (Nov 13) | Untuned Qwen control done (300 users, all valid). The trainer reads `prompt`/`completion`, saves a PEFT adapter to `<output>/adapter`, uses `max_length` 1024, and **cuts long prompts from the left**. The P1 loader needs `example_id, user_id, movie_id, split, profile, candidate_document, label`. | Code; P2 loader not written yet |
| `member5-service-a` (Nov 12) | Service A calls `POST /decide` with `profile_text`, `history_movie_ids`, `clarifications`; expects `ask` or `recommend`. | Explicitly provisional; mock Service B |
| Member 3 (Service B, vector index) | Nothing for Phase 2 on any branch. No checkpoint layout, Service B I/O, or max input length published. | Missing |
| Member 4 (agent, action schema) | Nothing for Phase 2 on any branch. The only action schema in the repository is Member 2's `p2-trace-v1` plus Member 1's `p2-prompt-v1` renderer and `p2_actions` parser. | Missing |

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

**Proposal:** freeze it at the commit named in `p2_samples.report.json`. Member 4's offline runner imports `src.data.clarification_sim`, so training and evaluation answers match. The midterm's "genre comparison" questions are covered by the favorite-genre and dislike intents.

### D2. Prompt length — *proposed* (Members 4, 6)

Action prompts with a 50–100 candidate pool (`evaluate_candidates`, `retrieve_candidates`, listwise `rank_candidates`) reach about 18k characters.

- **The problem:** Member 6's trainer keeps only the last `max_length` tokens. At 1,024 tokens it would delete the instructions, profile, and clarifications from about 40% of action samples.
- **Proposal:**
  - set `max_length` to **6,144** for P2 action samples (Qwen2.5-1.5B supports 32k);
  - make the trainer **fail or report** on truncation instead of cutting silently;
  - Member 6 measures exact token counts with the pinned tokenizer (it is not installed in Member 1's environment).
- **Short samples:** per-movie scoring samples (D6) are short and fit within P1's 1,024.

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

### D6. Ranking by per-movie scoring — *recommended; Members 4, 6 confirm*

The midterm says "LoRA learns action/question outputs" and "clarification updates scoring on fixed candidates". A 1.5B model writing a 50–100 ID permutation risks invalid output, and invalid output falls back to B3 order.

- **New file:** `p2_score_samples_{split}.jsonl` supports P1-style per-movie scoring.
- **Prompt:** the **P1 prompt verbatim**, with clarification turns appended to the profile block. Byte-identical to P1 when there are none.
- **Labels:** `relevant` (held-out rating ≥ 4) / `not_relevant` (≤ 2.5).
- **Compatibility:** the fields match Member 6's P1 loader, and the relevant-minus-not_relevant scorer applies unchanged.
- **Recommended agent path:**
  1. score the final pool;
  2. order by score;
  3. emit the top 50 as `rank_candidates` (the contract allows a subset of the final pool).
- **Listwise samples:** listwise `rank_candidates` samples remain available. Train on one ranking format, not both.

### D7. Checkpoint selection — *proposed* (Members 2, 6)

**Correction:** an earlier version of this page said checkpoint selection uses Member 1's `dev` split. The midterm says "select models on validation", and Member 2's protocol reserves the 300 validation users for decisions.

**Proposal:**

- Use `dev` (training-period users, disjoint from train) for per-checkpoint loss, action accuracy, and parse validity.
- Choose the final checkpoint among the few best by full-system NDCG@10 on the 300 validation users with Member 2's evaluator.
- Never use test.

### D8. Training profile mix — *decided* (Member 1)

Episodes now mix four profile levels:

| Level | Share | What it shows | Evaluation condition it matches |
|---|---:|---|---|
| canonical | 40% | Frozen canonical profile | Primary evaluation (Member 2) |
| masked_25 | 20% | Canonical profile with 25% of entries hidden | Midterm H2 |
| masked_50 | 20% | Canonical profile with 50% of entries hidden | Midterm H2 |
| sparse | 20% | Last 1–8 ratings | New app user (live demo) |

In masked profiles, 25% / 50% of liked and of disliked entries are hidden, not replaced. The simulator can reveal them.

### D9. Invalid model output — *proposed* (Member 4 owns)

**Proposal:**

- Parse with `p2_actions.parse_action`, then `validate_action`.
- On failure under greedy decoding, do not retry, since the same output would repeat.
- Instead, use the contract fallback: `termination_reason: "fallback"`, a `fallback_reason`, `fallback_source: "frozen_b3_order"`, and the raw output or its digest in the manifest.
- With D6, the ranking step cannot produce an invalid permutation.

### D10. Training pools vs. B3 pools — *open* (Members 3, 6)

- **The mismatch:** training pools come from Member 1's dependency-free feature index. Evaluation starts from frozen MiniLM B3 pools.
- **Option:** regenerate training pools with the B3 encoder for training users. This needs `sentence-transformers` and Member 3's index.
- **Until then:** a stated limitation.

## Decision record

| Item | Decision | Who / date |
|---|---|---|
| D1 simulator freeze | Proposed: freeze `pre_cutoff_history_sim_v1` | Member 1, 2026-11-22; awaiting Members 2, 4 |
| D2 prompt length | Proposed: `max_length` 6,144; no silent truncation | Member 1, 2026-11-22; awaiting Members 4, 6 |
| D3 profile shift | Report as limitation | Member 1, 2026-11-22 (consistent with Member 2 protocol) |
| D4 class balance | No relabelling; reweighting optional | Member 1, 2026-11-22 |
| D5 report wording | Proposed wording above | Awaiting team |
| D6 ranking method | Recommended: per-movie scoring; data provided | Member 1, 2026-11-22; awaiting Members 4, 6 |
| D7 checkpoint selection | Proposed: dev for screening, 300 validation users for final pick | Awaiting Members 2, 6 |
| D8 profile mix | 40 / 20 / 20 / 20 canonical / masked_25 / masked_50 / sparse | Member 1, 2026-11-22 |
| D9 invalid output | Proposed: no retry, contract fallback | Awaiting Member 4 |
| D10 training pools | Open | Members 3, 6 |
