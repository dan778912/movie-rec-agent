# Member 4 agent module — open decisions and how the code handles them

**Status: working draft by Member 4. Nothing here is team-approved.** It records where the sources
disagree or say nothing, what `src/agent/` does in the meantime, and which setting changes the
behaviour. It follows the format of [member1_p2_team_decisions.md](member1_p2_team_decisions.md).

Written 2026-11-19 against these exact sources; re-check them before relying on this page:

| Source | Ref |
|---|---|
| This branch's base (Member 1's samples, shared prompt text and action grammar) | `member-1-samples-p2` at `321c28443bca3e502ebcf393a079ba6527109027` |
| Proposed interface contract, protocol, evaluator | `origin/member2-phase2` at `9032fd13c62b7b18d3c53a2e3c685c53f4f49b55` (`docs/phase2_interface_contract.md`: "proposed v1") |
| Service A and its provisional `/decide` client | `origin/member5-service-a` at `853512100a7d4175ac8b3eff49f794d2a6e277db` |
| Submitted midterm report, section on limits | "Invalid actions or exhausted budgets fall back to ranking available candidates." |

**What "default" means below:** the value the code uses if nobody decides. Each default is a
proposal taken from a named source, chosen to be easy to change. It is not a recommendation to the
team and not an approved setting. Configuration lives in `AgentConfig` (`src/agent/config.py`) and
every trace carries the configuration it ran with (`label: "unratified-defaults"`).

## Register

| ID | Question | What the sources say | What the code does now | Setting | Who needs to decide |
|---|---|---|---|---|---|
| **A1** | What happens when a budget runs out and the latest judgment is still "no"? | **Conflict**, see below. | Both rules implemented. Default `fallback_b3_order`. | `exhaustion_policy` | Members 2 and 1, then the whole team (the final report must describe what runs) |
| A2 | Operational limits: questions, retrievals, pool size, event cap | Member 1 trained at 2 questions, 1 retrieval, 50 → at most 100. The proposed trace allows 8 control events. These agree: 8 is exactly the worst case of 2 + 1 (3 assessments + 2 questions + 2 pool checks + 1 retrieval). The protocol assigns the online loop cap and timeouts to Member 4. None is ratified. | 2 / 1 / 100 / 8. A question or retrieval is refused when its follow-up events would not fit under the cap. | `max_clarifications`, `max_retrievals`, `max_pool_size`, `max_control_events` | Members 1, 2, 3, 4. Freeze on validation data before any test inference. |
| A3 | A model output that cannot be used | The contract allows the frozen fallback for it but ratifies no numeric validity threshold. With greedy decoding a retry repeats the same prompt. | No retry; the first unusable output at a step triggers the B3 fallback (any policy). The raw output's SHA-256 and the reason are recorded. | `max_invalid_actions_per_step` | Members 2, 4, 6 (after seeing real checkpoint output) |
| A4 | Explanations unusable after a good ranking | No rule. The contract requires explanations in every episode, including fallbacks, and says the manifest states the explanation policy. | The episode ends `failed`, the ranking is kept, no text is invented, and no trace is produced. The caller must return an explicit failure. | none yet | Members 2 and 4 (decide with the grounding work) |
| A5 | Ranking length | The prompt asks the model to rank the whole pool (50–100). The contract's rank event is the 50 prediction IDs. | The model's output must be a permutation of the pool shown; the top `ranking_size` are kept. The full ranking's digest is in telemetry. Explained movies are `min(10, ranking_size)` because the shared validator requires exactly that. | `ranking_size` | Member 2 to confirm |
| A6 | Rank step: training prompt versus what inference can show | Member 1's rank samples inject held-out movies into the pool as supervision (the committed example shows 105 candidates where the retriever returned 100) and the target ranks them. At inference no such movies exist. | Strict: rejects a ranking that is not a permutation of the pool shown. `injected_movie_ids` is never emitted. A test replays Member 1's example and pins this. | none | Members 1 and 6: is the rank-step fine-tuning target matched to inference? Member 2 |
| A7 | Retrieval interface and ownership | The model emits a free-text `query`. Member 3's interface does not exist yet. Unspecified: who removes duplicates, watched movies and over-cap results, and who updates the dynamic profile from clarification answers. | The caller receives `query` and `max_new`. The controller filters defensively (pool members, watched, duplicates, over-cap) and logs counts per retrieval. Answers go into the prompt's Clarifications block; `profile_text` is never edited. | none | Members 3 and 4 |
| A8 | Repeated questions | The prompt says not to repeat. The protocol assigns the handling to Member 4. | A question matching an earlier one (ignoring case, punctuation, spacing) is an unusable output. | `reject_repeated_questions` | Member 4 proposes; Member 2 to acknowledge |
| A9 | Strict versus lenient JSON | Training completions are one bare JSON object. No real P2 checkpoint output has been seen. | Strict (Member 1's `parse_action`): no code fences, no prose. Nothing is repaired silently. | none | Revisit with the measured invalid-output rate from the real checkpoint |
| A10 | Live pause and resume | Service B is stateless. Service A resends `clarifications` (`{question, answer}`) on every call. The contract proposes envelope fields (`contract_version`, `request_id`, `decision_trace`) but names no error codes. | State is rebuilt from the replayed clarifications. Each counts as 2 control events against the cap, so the budget is not reset. Earlier events are not invented. Replayed answers get `answer_source: "client_supplied"`. A live trace is assembled across calls by Service A. | none | Members 3, 5, 2 |
| A11 | Prompt length for 50–105 candidates (Member 1's D2) | About 4.6k tokens versus P1's 1,024. Options: longer training length, chunking, shorter lines. | Renders the full pool with the shared `p2-prompt-v1`. No chunking. | none | Members 4 and 6 (a changed renderer needs a new `PROMPT_VERSION` and regenerated samples) |
| A12 | Who writes the offline runner (loops over the saved users, calls the controller and the answer simulator, writes prediction/trace/manifest files)? | Member 1 suggests it is Member 4's. The October work split does not list it. | Not written. | n/a | Members 2, 3, 4 |
| A13 | Termination labels outside the contract | The proposed contract defines only `recommend` and `fallback`. | `budget_exhausted_rank` (under `rank_available_pool`) and `failed` are this module's own labels. See the A1 evidence. | n/a | Member 2, if A1 picks rank-available |
| A14 | Validator coverage | Member 2's trace validator is on an unmerged branch. | The in-repo oracle is Member 1's `validate_episode`. A one-off cross-check against Member 2's validator is recorded below. | n/a | Add a permanent test once the branches merge |
| A15 | Grounding checks | Not part of this increment. Member 1 has a checker for its template explanations; Member 2 has a human audit rubric. | Not implemented. | n/a | Members 2 and 4 |

## A1 in detail: the two fallback rules

| | Midterm report (submitted) and Member 1's reading | Proposed v1 contract (Member 2) |
|---|---|---|
| Trigger | "Invalid actions or exhausted budgets" | A `false` judgment the loop cannot resolve, or a model/retrieval error. The producer "may" use the fallback. |
| What is ranked | The available candidates. Member 1 reads this as the model ranking the current pool: "proceed to ranking the available candidates". | The frozen B3 order, exactly, even if the trace recorded an earlier expansion. |
| Trace label | Not specified | `termination_reason: "fallback"` plus `fallback_reason` and `fallback_source: "frozen_b3_order"` |
| A `recommend` trace | Not specified | Requires the latest sufficiency and pool judgments to be `true` |
| Explanations | Not specified | Still required for the top ten |

The two are not obviously the same thing. For an *invalid action* they can agree if "available
candidates" means the B3 candidates, and the controller handles that case the contract's way. They
disagree for an *exhausted budget*: ranking the current pool by the model versus returning B3's
order. Under the contract's way, users whose budget runs out get B3's performance by construction.

**What the code does.** `exhaustion_policy` selects the rule for exhausted budgets only:

* `fallback_b3_order` (default): the contract's rule.
* `rank_available_pool`: the model ranks the pool it has, and the trace is labelled
  `budget_exhausted_rank`.

Errors (a retrieval failure) and unusable outputs always use the contract fallback under either
setting, because "rank what you have" needs a working model and a trusted pool.

**Evidence, so the choice is informed.** Two checks, both on scripted test doubles (they exercise
the controller, not a model):

1. *Member 1's `validate_episode`* accepts both kinds of trace only in its lenient mode
   (`allow_fallback=True`) and rejects both in strict mode. It cannot tell the rules apart. Pinned
   by `ConflictingExhaustionRulesTest`.
2. *Member 2's real validator* (`src/evaluation/validate.py` at `9032fd1`, run from an extracted
   copy outside this repository on 2026-11-19). 515 scripted users: every one of the 512 yes/no
   paths through the two judgments, plus an unusable first output, an empty retrieval and a
   retrieval error. Results:
   * `fallback_b3_order`: **all 515 users accepted** (336 `recommend`, 179 `fallback`).
   * `rank_available_pool`: **177 users rejected, and exactly those**: the ones labelled
     `budget_exhausted_rank`, with the message "termination_reason must be 'recommend' or
     'fallback'". The other 338 were accepted.

   So choosing rank-available needs Member 2 to extend the evaluator. This cross-check is not a test
   in the repository because it needs code from another unmerged branch (A14).

**Questions for the decision.** Is Member 2's evaluator the authority, so the report is reworded to
match the contract? Or is the submitted wording the intent, so the contract and evaluator change?
Under the contract, exhausted-budget users are scored on B3's order; under rank-available they are
scored on the model's. That affects what P2's headline numbers mean, so it should be decided before
validation runs, not after seeing results.

## Things this increment deliberately does not do

* No model call, retrieval call, timeout or HTTP. The controller is a state machine that a caller
  drives; the real adapters wait on Members 3 and 6.
* No grounding check, no offline runner, no prompt chunking (A15, A12, A11).
* It does not claim that any real model emits valid actions, or that clarification or retrieval
  improves recommendations. Nothing here measures either.
