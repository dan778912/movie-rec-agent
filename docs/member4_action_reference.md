# P2 action reference: what the model writes and what the parser does

**Status: current, not finalized.** Written 2026-11-19 by Member 4 in answer to Member 6's request for the
structured-action format, action names, required fields, valid examples and invalid-output handling.

This page is a reference to the grammar that already exists, not a new schema. Field names and types come
from Member 1's shared code (`member-1-samples-p2` at `321c284`: `src/profiles/agent_text.py` and
`src/data/p2_actions.py`). The checks after parsing come from `src/agent/` on `member4-p2-agent`. Nothing
here is team-approved, and if Member 1's grammar changes this page must change with it. A test
(`tests/test_agent_action_reference.py`) fails if the action table, the defaults, the reason codes or the
examples below stop matching the code.

## 1. Output rules

The model's reply is exactly one JSON object and nothing else.

- No text before or after it, and no markdown code fences. Such output is rejected, not repaired (A9).
- Field names exactly as in section 2: none missing, none extra.
- No duplicate keys, and no `NaN` or `Infinity`.
- Booleans are `true` or `false`, not strings. Movie IDs are positive integers, not strings or booleans.
- The `action` value must be the one this step asks for. A different valid action is rejected.

## 2. Actions

Every object also carries `"action": "<name>"`. Each action has exactly one other field.

| action | field | what it holds, and what is checked after parsing |
|---|---|---|
| `assess_sufficiency` | `sufficient` | Boolean: is there enough information about this user to recommend? |
| `ask_clarification` | `question` | Nonempty string. It must not repeat an earlier question (case and punctuation are ignored). |
| `evaluate_candidates` | `pool_sufficient` | Boolean judgment of the whole candidate pool shown. |
| `retrieve_candidates` | `query` | Nonempty string, sent to retrieval as the search request. |
| `rank_candidates` | `movie_ids` | Distinct positive integers, best first. Must be an exact permutation of the pool shown: every shown ID once, no others. |
| `explain_recommendations` | `explanations` | A list of `{"movie_id": int, "text": nonempty string}` covering exactly the top 10 ranked movies, in rank order. |

## 3. What the model does not write

The trace has extra fields that the controller adds from what it knows. The model must not emit them: they
would be extra fields, so the output would be rejected.

| trace event | added by the controller | source |
|---|---|---|
| `evaluate_candidates` | `candidate_movie_ids` (the current pool) | controller state |
| `ask_clarification` | `answer`, `answer_source` | the user or the simulator |
| `retrieve_candidates` | `added_movie_ids`, `index_version` | the retriever |

The trace keeps the first 50 IDs of the model's ranking. The training records' `injected_movie_ids` is
training provenance and is never part of a completion or a trace (see A6).

## 4. Step order, and where the controller overrides the model

The controller asks for one task at a time; the model never picks the next step by naming another action.

```text
assess -> [not sufficient, question allowed: ask -> (answer arrives) -> assess]
       -> sufficient: evaluate -> [pool not sufficient, retrieval allowed: retrieve -> (results arrive) -> evaluate]
       -> rank -> explain
```

With the default limits, the controller overrides the model in two places, and both end in the fallback:

- `sufficient: false` when no question is allowed (2 already asked, or the event cap would be exceeded).
- `pool_sufficient: false` when no retrieval is allowed (1 already used, pool already at the cap, or the
  event cap would be exceeded).

The fallback is the frozen B3 order (first 50), labelled `fallback`, and the model still writes the
explanations for the top 10. A model that answers `false` at those points therefore loses its own ranking
for that user. This is the proposed v1 contract rule; the other reading is open (A1).

The limits fit the event cap exactly: 2 questions plus 1 retrieval is 3 assessments, 2 questions,
2 evaluations and 1 retrieval, which is 8 control events.

## 5. Settings (defaults are unratified)

All are fields of `AgentConfig` (`src/agent/config.py`) and are passed to `decide`. None is frozen by the team.

| setting | default | meaning |
|---|---|---|
| `max_clarifications` | `2` | Questions per episode. |
| `max_retrievals` | `1` | Re-retrievals per episode. |
| `max_pool_size` | `100` | Pool size after retrieval; extra retrieved IDs are dropped. |
| `max_control_events` | `8` | The contract's cap on control events. |
| `ranking_size` | `50` | Length of the ranking kept in the trace. |
| `max_invalid_actions_per_step` | `0` | Extra tries after an unusable output. `0` means fall back at once. |
| `exhaustion_policy` | `fallback_b3_order` | The other value, `rank_available_pool`, writes a label the proposed validator rejects (A1). |
| `reject_repeated_questions` | `true` | Whether a repeated question counts as unusable. |
| `grounding_policy` | `flag` | `off`, `flag`, `repair` or `reject`. It acts only on explanations that cite evidence not shown. |
| `explanation_failure_policy` | `fail` | What to do when the explain step has no usable output. `fail` ends the episode as `failed`; `template` writes the ten explanations from each movie's own genres and tags and records that it did (A4). Off by default and unratified. |

## 6. When the output is unusable

Reason codes recorded for every failed step:

| code | meaning |
|---|---|
| `invalid_action` | Not valid JSON, or it breaks section 1 or 2: prose, fences, extra or missing fields, wrong types. |
| `unexpected_action` | A valid action, but not the one this step asked for. |
| `unbound_ids` | Ranking or explanation IDs differ from what was shown. |
| `repeated_question` | The question repeats an earlier one. |
| `model_error` | The model call itself failed (exception or timeout). Nothing was parsed. |

What happens next, with `max_invalid_actions_per_step` at its default of `0`:

- At the assess, ask, evaluate, retrieve or rank step, the episode takes the fallback above. The reason is
  recorded as `invalid_action:<step>:<code>`, or `model_error:<step>` when the call failed.
- At the explain step, by default (`explanation_failure_policy` = `fail`) the episode ends as `failed`. The
  ranking is kept, no trace is produced, and the service answers `{"action": "error", ...}`. The controller
  does not invent explanation text. With `template`, the ranking is kept and the ten explanations are built
  from each movie's own genres and tags instead; the trace event looks like any other explanation event, and
  telemetry says `explanation_source: "template_fallback"` (otherwise `"model"`).
- With a value above `0`, the same step is asked again with the same prompt, up to that many times. No
  error message is added to the prompt.

Each attempt is logged (the step, the code, the message and a SHA-256 of the raw output), so
structured-output validity can be counted as the share of model steps that were valid.

## 7. Valid examples

The first six are copied from Member 1's committed completions (`docs/p2_sample_examples.jsonl`). The last
two are cut short: real rankings list the whole pool and real explanation lists have 10 entries.

```json valid verbatim
{"action": "assess_sufficiency", "sufficient": false}
```

```json valid verbatim
{"action": "assess_sufficiency", "sufficient": true}
```

```json valid verbatim
{"action": "ask_clarification", "question": "Is there a film you loved that I should know about?"}
```

```json valid verbatim
{"action": "evaluate_candidates", "pool_sufficient": false}
```

```json valid verbatim
{"action": "evaluate_candidates", "pool_sufficient": true}
```

```json valid verbatim
{"action": "retrieve_candidates", "query": "Movies similar to Silence of the Lambs, The (1991), Miracle on 34th Street (1994), Beverly Hills Cop III (1994), Firm, The (1993). Genres: Thriller, Crime, Drama. Themes: oscar (best music - original score), tom clancy, lawyers, gunfight, sequel."}
```

```json valid shortened
{"action": "rank_candidates", "movie_ids": [10, 356, 168, 590, 464]}
```

```json valid shortened
{"action": "explain_recommendations", "explanations": [{"movie_id": 10, "text": "Like Beverly Hills Cop III (1994), which you rated 4/5, it is listed under Action and Thriller."}, {"movie_id": 356, "text": "Like Firm, The (1993), which you rated 4/5, it is listed under Drama and is tagged \"oscar (best music - original score)\"."}]}
```

## 8. Rejected outputs

Each block shows an output and the code it gets. The line before the block says which step it was for.

Prose around the object (`invalid_action`):

```text rejected:invalid_action task=assess_sufficiency
Here is my answer: {"action": "assess_sufficiency", "sufficient": true}
```

Wrapped in a markdown fence (`invalid_action`):

````text rejected:invalid_action task=assess_sufficiency
```json
{"action": "assess_sufficiency", "sufficient": true}
```
````

An extra field (`invalid_action`):

```text rejected:invalid_action task=assess_sufficiency
{"action": "assess_sufficiency", "sufficient": true, "reason": "enough"}
```

A string where a Boolean belongs (`invalid_action`):

```text rejected:invalid_action task=assess_sufficiency
{"action": "assess_sufficiency", "sufficient": "true"}
```

A duplicate key (`invalid_action`):

```text rejected:invalid_action task=assess_sufficiency
{"action": "assess_sufficiency", "sufficient": true, "sufficient": false}
```

A trace-only field written by the model (`invalid_action`):

```text rejected:invalid_action task=evaluate_candidates
{"action": "evaluate_candidates", "pool_sufficient": true, "candidate_movie_ids": [1, 2, 3]}
```

A valid action for the wrong step, here an assessment when the pool evaluation was asked for (`unexpected_action`):

```text rejected:unexpected_action task=evaluate_candidates
{"action": "assess_sufficiency", "sufficient": true}
```

A ranking that leaves out a shown movie, with a pool of 10, 356 and 168 (`unbound_ids`):

```text rejected:unbound_ids task=rank_candidates pool=10,356,168
{"action": "rank_candidates", "movie_ids": [10, 356]}
```

A ranking that adds a movie that was not shown (`unbound_ids`):

```text rejected:unbound_ids task=rank_candidates pool=10,356,168
{"action": "rank_candidates", "movie_ids": [10, 356, 168, 999]}
```

Explanations in the wrong order, when the top 10 shown were 10 then 356 (`unbound_ids`):

```text rejected:unbound_ids task=explain_recommendations explain=10,356
{"action": "explain_recommendations", "explanations": [{"movie_id": 356, "text": "x"}, {"movie_id": 10, "text": "y"}]}
```

A question that repeats the earlier "What is one of your favorite movies?" (`repeated_question`):

```text rejected:repeated_question task=ask_clarification
{"action": "ask_clarification", "question": "what is one of your favorite movies"}
```

## 9. Open points that affect training and checkpoint selection

- **A9, strict parsing.** A checkpoint that wraps its JSON in fences or prose will send those users to the
  fallback. If that happens often, whether to relax parsing is a team decision, made before validation.
- **A3, retries.** The default is no retry. Raising it is one setting, but the evaluator requires every
  invalid attempt to be recorded either way.
- **A5 and A6, pools.** At inference the pool is whatever the retriever returned (B3's 50, plus up to 50
  more), with no injected movies. The model is asked to rank the whole pool, and the trace keeps 50.
- **A2, limits.** The counts in section 5 are Member 1's training design and the proposed contract, not a
  frozen team policy. Changing one changes which prompts occur at inference.

To ask for a change to anything here, edit the decision in
[member4_agent_decisions.md](member4_agent_decisions.md) or tell Member 4; the controller's behaviour is set
by the settings above, not by code changes, wherever the table says so.
