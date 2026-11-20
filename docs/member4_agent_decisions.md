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
| A3 | A model output that cannot be used, or a failed model call | The contract allows the frozen fallback for it but ratifies no numeric validity threshold. With greedy decoding a retry repeats the same prompt. Timeouts belong to whoever owns the model client. | No retry; the first unusable output, or an exception from the model call, at a step triggers the B3 fallback (any policy). The raw output's SHA-256 (none for an exception) and the reason are recorded. | `max_invalid_actions_per_step` | Members 2, 4, 6 (after seeing real checkpoint output) |
| A4 | What to do with an explanation the grounding screen marks `unsupported`, or with no usable explanation at all | No rule. The contract requires explanations in every episode, including fallbacks, and says the manifest states the explanation policy. The audit rubric is human. | `grounding_policy`: `flag` (default: keep the text, record the problems), `repair` (replace only that explanation with a text built from the movie's own displayed fields), `reject` (treat as unusable: retries, then `failed`), `off`. Unusable explanations end the episode `failed`: the ranking is kept, no text is invented, no trace is produced, and the caller returns an explicit error. | `grounding_policy` | Members 2 and 4 |
| A5 | Ranking length | The prompt asks the model to rank the whole pool (50–100). The contract's rank event is the 50 prediction IDs. | The model's output must be a permutation of the pool shown; the top `ranking_size` are kept. The full ranking's digest is in telemetry. Explained movies are `min(10, ranking_size)` because the shared validator requires exactly that. | `ranking_size` | Member 2 to confirm |
| A6 | Rank step: does training on pools with injected held-out items transfer to retriever-only pools at inference? | Member 1's report documents the injection as deliberate supervision (up to 3 held-out positives and 2 dislikes added to the final pool; the committed example has 105 candidates where the retriever returned 100, and all 3 injected positives land in the explained top ten). At inference no such movies exist. Member 6 has replied (relayed 2026-11-19): inference compatibility is intended and will be tested before checkpoint selection; Member 1's confirmation is pending ([details](#a6-in-detail-what-is-known-after-member-6s-reply)). Whether the mismatch costs anything is still unmeasured. | Strict: a ranking must be a permutation of the pool shown, and `injected_movie_ids` is never emitted. A test replays Member 1's example and pins this. A question, not a defect claim. | none | Member 1 (confirmation pending; Member 6 has replied); see [member4_team_review_note.md](member4_team_review_note.md) |
| A7 | Retrieval interface and ownership | The model emits a free-text `query`. Member 3's interface does not exist yet. Unspecified: who removes duplicates, watched movies and over-cap results, and who updates the dynamic profile from clarification answers. | The caller receives `query` and `max_new`. The controller filters defensively (pool members, watched, duplicates, over-cap) and logs counts per retrieval. Answers go into the prompt's Clarifications block; `profile_text` is never edited. | none | Members 3 and 4 |
| A8 | Repeated questions | The prompt says not to repeat. The protocol assigns the handling to Member 4. | A question matching an earlier one (ignoring case, punctuation, spacing) is an unusable output. | `reject_repeated_questions` | Member 4 proposes; Member 2 to acknowledge |
| A9 | Strict versus lenient JSON | Training completions are one bare JSON object. No real P2 checkpoint output has been seen. | Strict (Member 1's `parse_action`): no code fences, no prose. Nothing is repaired silently. | none | Revisit with the measured invalid-output rate from the real checkpoint |
| A10 | Live pause and resume | Service B is stateless. Service A resends `clarifications` (`{question, answer}`) on every call. The contract proposes envelope fields (`contract_version`, `request_id`, `decision_trace`) but names no error codes. | Two ways to continue are offered. `decide` rebuilds the state from the replayed clarifications: each counts as 2 control events against the cap, so the budget is not reset, and earlier events are not invented (replayed answers get `answer_source: "client_supplied"`). `resume` continues from a stored state and keeps its full event history. Both give the same recommendations in the example. A live trace is assembled across calls by Service A. | none | Members 3, 5, 2 |
| A11 | Prompt length for 50–105 candidates (Member 1's D2) | About 4.6k tokens versus P1's 1,024. Options: longer training length, chunking, shorter lines. | Renders the full pool with the shared `p2-prompt-v1`. No chunking. Measured on the real 300 validation pools: see [the rehearsal](#real-data-rehearsal-stand-in-model). | none | Members 4 and 6 (a changed renderer needs a new `PROMPT_VERSION` and regenerated samples) |
| A12 | Who writes the offline runner (loops over the saved users, calls the controller and the answer simulator, writes prediction/trace/manifest files)? | Member 1 suggests it is Member 4's. The October work split does not list it. | Not written, and not started until ownership is confirmed. The online `decide` entry point is a different thing and is included. | n/a | Members 2, 3, 4 |
| A13 | Termination labels outside the contract | The proposed contract defines only `recommend` and `fallback`. | `budget_exhausted_rank` (under `rank_available_pool`) and `failed` are this module's own labels. See the A1 evidence. | n/a | Member 2, if A1 picks rank-available |
| A14 | Validator coverage | Member 2's trace validator is on an unmerged branch. | The in-repo oracle is Member 1's `validate_episode`. A one-off cross-check against Member 2's validator at `9032fd1` is recorded below and was re-run after the grounding work with the same result. | n/a | Add a permanent test once the branches merge |
| A15 | Which claims and evidence count as grounded? | The work split names a "grounding check" without criteria. Member 2 has a human audit rubric; Member 1 has a checker for its own template explanations. | A deterministic screen, `src/agent/grounding.py`, of what is mechanically checkable (see below). It never says "supported": that is the human audit's word. | `grounding_policy` | Members 2 and 4: are these the agreed criteria? |
| A16 | Adapter shapes between the controller and Service B | Member 3's interfaces do not exist yet. | `ModelClient.generate(prompt)`, `Retriever.initial_pool(...)` / `search(...)`, a `catalog` mapping, and `decide` / `resume` returning a Service-A-shaped response plus the explicit state. Proposed in `src/agent/decide.py` and shown running in `src/agent/example.py`. Timeouts belong to the client. | n/a | Member 3 to accept or change |
| A17 | Which fallback trigger codes do we declare, and what do the controller-derived manifest counts mean? | Member 2's proposed `p2-run-manifest-v1` wants per-user counts and a fallback trigger taken from a list the producer freezes. It does not say which codes. Its test fixture uses three examples: `timeout`, `invalid_action`, `loop_cap`. | `src/agent/manifest.py` proposes 11 codes, definitions of `invalid_action_count` and `loop_cap_hit`, and parser and explanation-policy version strings. Latency, cost, timeouts and answerability come from the caller. Details below. | constants in `manifest.py` | Member 2: accept or change the list and definitions |

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

## A6 in detail: what is known after Member 6's reply

**Member 6's reply** (relayed by Priyanka on 2026-11-19; not independently confirmed by me): the P2
checkpoint should be compatible with retriever-only pools at inference. The injected held-out items are
understood as a training-only way to give the ranking step useful supervision, not an inference
requirement. Member 6's code will not assume exactly 105 candidates, and the checkpoint will be tested on
ordinary 100-item retriever-only pools before it is selected. Member 6 also asked Member 1 to confirm that
(a) injected items come only from training-period data, (b) the model input does not mark which
candidates were injected, and (c) the training or validation handoff includes some retriever-only
examples, so the mismatch can be measured directly.

**What the repository shows on those three points.** This is my reading of Member 1's code at `321c284`,
offered as evidence for Member 1 to check, not as Member 1's confirmation.

| Point | What I found | Where |
|---|---|---|
| (a) provenance | Users are drawn from `train_ratings.csv` (training-period data), excluding the frozen validation and test user lists. Injected movies are that user's own later ratings (positives rated 4 or more, dislikes rated 2.5 or less) after the episode's anchor, and the shared checks require each one to be a held-out rating. | `src/data/p2_samples.py` (user draw, `positives`/`negatives`, injection), `src/data/p2_checks.py` (`injection` check), `configs/p2_samples_v1.json` (`exclude_user_lists`) |
| (b) marking | In the one committed rank example (`train:9:1`), the ids appear only in `provenance.label_basis.injected_movie_ids`, not in the `prompt`, `completion`, `state` or `target`. The five injected movies use the same line format as the rest and sit at scattered prompt positions (lines 14, 19, 63, 92, 115 of 120), consistent with random insertion. I checked one example, not the full training file, which is not in the repository. | `docs/p2_sample_examples.jsonl` record 9 |
| (c) retriever-only examples | Not present as far as I can see. The `inject_*` settings are global and `build()` applies them whatever the split, so dev rank samples would also contain injected items. If so, a dev run would not by itself measure the retriever-only case. Member 1 should confirm this. | `src/data/p2_samples.py` (no split condition around the injection), `configs/p2_samples_v1.json` |

**What this means for the controller.** Nothing changes. The controller already requires a ranking to be
a permutation of the pool it showed, and it never assumes a pool size. That matches Member 6's
description of inference. If the held-out items are not marked in the prompt, there is no marker the
model could rely on at training time that disappears at inference. A subtler cue is still possible, for
example injected movies being less similar to the profile than retrieved ones. That is a hypothesis, not a
finding, and it is exactly what a retriever-only comparison would test.

**Still open:** Member 1's confirmation of (a) to (c); whether retriever-only examples will be added to the
dev handoff; the result of Member 6's retriever-only test. I have not changed the A6 setting: there is none,
and the controller stays strict.

## A15 in detail: the grounding screen

The explain prompt says "cite only titles, ratings, genres, tags, and answers shown above". The screen
checks that mechanically against exactly what the model saw (the profile text, the clarification
answers, and the catalog lines of the ten movies listed). Its statuses reuse the audit rubric's words
where one exists, so the results line up with the human review in
[phase2_grounding_audit.md](https://github.com/dan778912/movie-rec-agent/blob/member2-phase2/docs/phase2_grounding_audit.md).

| Status | Meaning | Examples |
|---|---|---|
| `unsupported` | A title, rating, genre, tag, decade or reference to a clarification that was never shown, or that contradicts the profile | a title not in the profile, answers or listed movies; `rated 5/5` where the profile says 4/5; a genre or tag shown nowhere; "you said ..." when nothing was asked; another recommendation presented as the user's history |
| `insufficient_evidence` | A claim about a field the catalog data does not have | "directed by ...", cast, plot, box office (unless the displayed tags use the word) |
| `needs_review` | Possible misattribution | a genre or tag shown for the user's history but not for this movie |
| `no_checkable_claim` | Nothing in the text can be checked | "A great pick you will love!" |
| `passed_screen` | Nothing detectable beyond the evidence. **Not** "supported" | Member 1's template explanations |

`grounding_policy` acts on `unsupported` only. The other statuses are recorded for the audit and never
change the text. Every result is stored in the state (`grounding`) and in telemetry, never in the
contract trace, which keeps only the displayed text. Under `repair` the trace holds the repaired
text, so the trace and what a user sees agree, and the model's own output stays in `decisions`.

**What this does not do**, deliberately: it cannot tell whether prose is true or sensible, whether a
simulated answer used only pre-cutoff history, or whether decisions were autonomous. Titles in
free-text answers are matched leniently (a trailing run of words plus the year), so a partial title can
pass; a test pins that limit. Words inside a cited title (the "Mystery" in "Mystery Men (1999)", or
quotation marks in a title) are names, not claims, so cited titles are left out of the genre and tag
scans; a title written without its "(YYYY)" is not recognised as a title. The positive control is that all ten of
Member 1's committed template explanations pass; the negative controls are the fabrications in the table above.

## A17 in detail: what the controller supplies for the run manifest

`src/agent/manifest.py` computes the manifest values that the controller's state can show and takes the
rest as explicit arguments. It is not the offline runner (A12): no I/O, no loop over users, no files.

| Manifest value | Supplied by | How |
|---|---|---|
| `policy.agent` object | the controller (`agent_policy`) | parser and explanation-policy version strings, the event caps from the config, `frozen_b3_order`, the trigger list. `timeout_ms` is the caller's. |
| `fallback_reason` | the controller | the exact trace text (`state.reason`) |
| `fallback_trigger` | the controller (`fallback_trigger`) | a stable code derived from that text; unknown text raises |
| `model_call_count` | the controller | model attempts in the state, including failed calls |
| `invalid_action_count` | the controller | outputs that were parsed and rejected; a failed model call is not one |
| `retrieval_call_count` | the controller | calls made, including ones that failed or added nothing |
| `loop_cap_hit` | the controller | a question, retrieval, pool-size or event limit stopped a "no" judgment |
| `timed_out`, `latency_ms`, `cost_usd` | the caller | only the caller can time or price a call |
| informative and noninformative answer counts | the caller | the controller cannot judge an answer; the two must add up to the clarification events |

**Proposed trigger codes (11):** `invalid_action`, `unexpected_action`, `unbound_ids`, `repeated_question`
(an unusable model output); `model_error` (the call failed, timeouts included); `retrieval_error`,
`retrieval_returned_no_new_candidates`; `clarification_budget_exhausted`, `retrieval_budget_exhausted`,
`pool_size_cap_reached`, `control_event_cap_reached` (a limit stopped a "no" judgment).

**Evidence (one-off, scripts not committed).** I ran 526 scripted episodes (190 fallbacks, 336 recommendations;
all 11 triggers occur) through the episode checks of Member 2's validator at `9032fd1`, and all were accepted.
The same run rejected five deliberate corruptions: a trigger missing from the frozen list, a fallback reason
that differs from the trace, answerability counts off by one, a retrieval count below the successful
retrievals, and a recommendation with no model call. Separately, my `policy.agent` block passed that
validator's full manifest check and failed when `max_control_events` was changed to 9, when `parser_version`
was removed, and when a trigger was listed twice. That shows the shapes agree, not that the definitions are
the right ones.

**Questions for Member 2.** (1) Do you want a coarse list like your fixture's `timeout`, `invalid_action`,
`loop_cap`, or these finer codes? Timeouts are currently `model_error`, because the controller sees only that
the call failed. (2) Should `invalid_action_count` exclude failed model calls, as proposed? (3) Is the
`loop_cap_hit` definition what you meant by the loop cap?

**Limit.** Counts come from the state. A state rebuilt from replayed clarifications (`decide`) does not know
about model calls made while earlier turns were served, so whole-episode counts need a state carried through
`resume`, or sums across calls.

## Real-data rehearsal (stand-in model)

On 2026-11-19, after Member 6 shared the original ordered B3 validation pool (`validation_candidates_300.jsonl`,
SHA-256 `c388c0fd10bb895c1e0710ec10fa96a7d87785939f2498a5bf23533d18d000a9`, the hash in Member 2's checkpoint-selection
handoff), I ran the controller on all 300 validation users with a rule-based **stand-in** model: the real B3 pools, the real
profiles and watched histories from `data/processed/ml25m-global-v1`, and catalog documents built from
`movie_documents.jsonl` with Member 1's `display_tags`. The stand-in judged every pool sufficient, asked one question for
a third of the users, widened the pool for a fifth (with arbitrary catalog movies, not a real index), ranked the pool in its
own order, and wrote an explanation from each movie's displayed fields. It says nothing about any model's quality.
The scripts are scratch and are not committed; this is not the offline runner (A12).

* **Structure.** All 300 pools started cleanly (none contains a watched movie, none repeats a movie) and every pool movie has a
  catalog document. All 300 episodes finished as `recommend`. Member 2's validate-only command
  (`python -m src.evaluation.validate`, branch `member2-phase2` at `9032fd1`) reported `status: valid`, 0 errors, 300 of 300
  users, with that exact B3 file as the candidate pool.
* **Catalog adapter.** The controller's catalog documents need `tags`, not the `genome_tags` and `user_tags` stored in
  `movie_documents.jsonl`. They must be joined with `display_tags` (three genome tags then two user tags), or the prompt
  will not match the training text. Whoever builds the service's catalog (Member 3) needs this (A16).
* **Prompt sizes (A11).** Measured on the first 60 users, in characters, with tokens estimated at 3.9 characters per
  token (the ratio implied by Member 1's "about 18k characters, about 4.6k tokens"; no tokenizer was run):

| Prompt | Median chars | Max chars | About tokens (median / max) |
|---|---|---|---|
| assess (no candidates) | 1,951 | 2,211 | 500 / 566 |
| evaluate, 50 candidates | 7,980 | 9,349 | 2,046 / 2,397 |
| rank, 50 candidates | 7,972 | 9,341 | 2,044 / 2,395 |
| rank, 100 candidates (simulated) | 14,183 | 15,687 | 3,636 / 4,022 |
| explain, top 10 | 3,285 | 3,688 | 842 / 945 |

  The 100-candidate row adds 50 arbitrary unwatched catalog movies to the B3 pool, so a real widened pool will differ a little.
  Even before any widening, the evaluate and rank prompts are about twice P1's 1,024-token limit.

## Review fixes (2026-11-19)

A review pass of the branch found four correctness problems, all confirmed by running the code and fixed
with tests:

* **Genre words inside titles** were scanned as genre claims, so a correct explanation of "Mystery Men
  (1999)" came out `unsupported` (see A15). Cited titles are now excluded from the genre and tag scans.
* **Replayed turns past the event cap.** Six replayed clarifications used to start a normal episode with
  14 control events against a cap of 8. If the replayed turns leave no room for one assessment and one pool
  evaluation, `start_episode` now treats the episode as budget-exhausted before any model call: under the
  default policy that is the B3-order fallback with reason `control_event_cap_reached`. Three replayed turns
  (6 events used, 2 needed) still run normally (A10).
* **Request validation.** `decide` now rejects, as `RequestError`, a repeated history id, a `user_id` that is
  not a positive integer, an empty or non-string `split`, and a non-string `answer_source`. These used to
  surface as internal `TransitionError`s (A16).
* **First-retrieval failure.** An exception from `Retriever.initial_pool` now raises `InitialRetrievalError`
  (the original is `__cause__`) instead of escaping raw. There is no fallback, because the B3 order is what
  failed to arrive; the service should answer with its own error and tell it apart from a bad request (A16).

Three lower-severity findings were not changed: resumed state is still only structurally validated (the
budget counters in an untrusted state can be edited by whoever holds it), the grounding screen still
rebuilds its evidence strings for each of the ten explanations, and `state.py` still keeps its own small
copies of two helpers that Member 1's `p2_actions.py` has.

## Things this increment deliberately does not do

* No model call, retrieval call, timeout or HTTP. The controller is a state machine that a caller
  drives; the real adapters wait on Members 3 and 6.
* No offline runner, no prompt chunking (A12, A11), and the grounding screen is only a screen (A15).
* It does not claim that any real model emits valid actions, or that clarification or retrieval
  improves recommendations. Nothing here measures either.
