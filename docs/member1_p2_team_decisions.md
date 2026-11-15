# P2 training data — alignment record and open items (Member 1)

**Status:** the sample design follows the project proposal and midterm report §3.4. Settled items are recorded below with their source. The open items need team confirmation before P2 validation runs and must be frozen before any P2 test inference. Full method and results: [member1_p2_report.md](member1_p2_report.md).

## Settled by the proposal and midterm report

| Topic | Decision | Source |
|---|---|---|
| Action labels | Compare every action sequence within the budget on NDCG@10 against held-out training-period interactions; choose the highest, preferring fewer actions on ties. No hand-set thresholds. | Midterm §3.4 |
| Budgets | At most 2 clarification turns and 1 wider retrieval | Proposal risk 1; midterm §3.4 |
| Pool sizes | Initial pool of 50, one expansion to at most 100 | Midterm §3.4 |
| Label sources | Training ratings only. Validation/test targets never supply answers or labels. Checkpoint selection uses a user-disjoint `dev` split from the training period. | Midterm §3.4, §5 |
| Clarification questions | Templates about preferences masked from earlier history; answers only from that hidden history | Midterm §3.4; roadmap §6 |
| Budget exhaustion | Proceed to ranking the available candidates | Midterm §3.4 |
| Claims | Simulated answers cannot establish real-user satisfaction | Midterm §3.4 |

**Earlier draft replaced.** An earlier draft labelled actions with cosine-gain thresholds (0.04 / 0.05) and took dev labels from validation ratings. Both conflicted with midterm §3.4 and were removed before any data was shared.

## Open items for the team

### D1. Freeze the clarification simulator (Members 2, 4)

`pre_cutoff_history_sim_v1` answers five intents — liked example, favorite genres, themes, dislikes, era — from ratings before the cutoff only. It covers the proposal's example question (light vs. serious) through the "moods or themes" intent. The Phase 2 protocol requires one frozen simulator for offline evaluation. **Recommend:** Member 4's offline runner uses this module, so training and evaluation answers match. Freeze it before P2 validation scoring.

### D2. Prompt length vs. `max_length` (Members 4, 6)

- **The problem:** a 50→100 candidate pool makes rank and evaluate prompts reach about 18k characters (~4.6k tokens). P1 trained at 1,024 tokens.
- **Options:**
  - (a) Raise `max_length` to about 5k tokens for P2.
  - (b) Have Member 4's agent show the pool in chunks, which also requires matching training prompts.
  - (c) Shorten candidate lines.
- **Note:** Members 4 and 6 must use the same `p2-prompt-v1` rendering (`src/profiles/agent_text.py`); any change needs regenerated samples and a new version string.

### D3. Test-time profile shift (Members 3, 6, 2)

| Profiles | With ≥ 1 out-of-catalog liked movie | Out-of-catalog share of liked entries |
|---|---:|---:|
| Frozen validation evaluation profiles | 174 / 4,751 (3.7%) | 0.5% |
| Frozen test evaluation profiles | 1,172 / 3,779 (31.0%) | 6.7% |
| P2 training samples, canonical | 0.1% | ≈ 0% |
| Member 3's P1 training samples | 0% by construction | 0% |

- **Cause:** 2,412 of the 2,520 out-of-catalog liked entries in test profiles were rated after the validation cutoff. They are newer movies that the training-defined catalog cannot contain.
- **Who is affected:** P1 and P2 both see this shift at test. It is consistent with the catalog-coverage limitation already in midterm §4.3: 67.76% test catalog-positive coverage.
- **Options:**
  - Report it as a limitation (recommended).
  - Optionally run a validation-only diagnostic: re-score with out-of-catalog likes removed.
- **Not an option:** changing the frozen profile rule, which would invalidate the published P1/B3 comparison.

### D4. Class balance (Member 6)

Because held-out positives are sparse, most sequence comparisons tie, and ties go to "proceed". Training labels are:

- `assess_sufficiency`: 2,744 "ask" vs 3,974 "proceed";
- `evaluate_candidates`: 738 "retrieve" vs 3,974 "proceed".

This is the midterm rule as written. Member 6 may reweight by task or label; the samples should not be relabelled.

### D5. Final-report wording (all)

The final report should describe the labels as "NDCG@10 comparison of action sequences under a lightweight preference-vector ranker", note that answers are simulated, and report:

- question frequency;
- the share of answers revealing a new fact (76% in training);
- retrieval frequency.

It should not claim real-user satisfaction or question quality.

## Decision record

| Item | Decision | Who / date |
|---|---|---|
| D1 simulator freeze | Pending | |
| D2 prompt length | Pending | |
| D3 profile shift handling | Pending | |
| D4 class balance | Pending | |
| D5 report wording | Pending | |
