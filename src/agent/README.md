# Agent loop — Phase 2, Member 4

The controller for P2's bounded clarification and wider-retrieval loop: it tracks one
recommendation episode, decides which action the model is asked for next, checks every model
output, enforces the limits, screens the explanations against the evidence, and records a trace.
It is written to be called by Service B (Member 3); see
[Sections 6, 10, and 11 of the roadmap](../../Movie_Recommendation_Agent_Roadmap.md).

**Every limit and policy here is unratified.** The defaults are proposals from named sources, and
the places where sources conflict are listed in
[docs/member4_agent_decisions.md](../../docs/member4_agent_decisions.md). The two questions that
need the team are summarised in [docs/member4_team_review_note.md](../../docs/member4_team_review_note.md).
Read those before treating any behaviour as the team's decision.

For what the model must write and what the parser does with bad output (action names, fields,
examples, reason codes, settings), see
[docs/member4_action_reference.md](../../docs/member4_action_reference.md). A test keeps it in step
with the code.

## What is reused, not redefined

| Reused from | What |
|---|---|
| `src/profiles/agent_text.py` (Member 1) | The six actions, `p2-prompt-v1` prompt text, `render_prompt`, `serialize_action` |
| `src/data/p2_actions.py` (Member 1) | `parse_action`, `validate_action`, `validate_episode` (the oracle in the tests) |
| `src/data/p2_checks.py` (Member 1) | `GENRES`, the 19 MovieLens genre names |
| `src/profiles/text.py` | `user_profile`, which the example uses to build profile text |

Training samples, inference prompts and validation must share these, so this module imports them
and a test checks it has not made copies. If the prompt text changes, `PROMPT_VERSION` changes
there, not here.

## Layout

| File | Role |
|---|---|
| `config.py` | `AgentConfig`: budgets and policies, stored inside each state |
| `state.py` | `EpisodeState`: everything an episode remembers; JSON round trip with validation on load |
| `actions.py` | `parse_model_output`: the shared parser plus "right action for this step" and ID binding |
| `grounding.py` | Deterministic screen of explanations against the evidence the model was shown |
| `controller.py` | The state machine: `start_episode`, `next_step`, `build_prompt`, `apply_model_output`, `apply_model_error`, `apply_answer`, `apply_retrieval`, `apply_retrieval_failure` |
| `decide.py` | The entry point for Service B: `decide`, `resume`, `advance`, plus the proposed `ModelClient` / `Retriever` shapes |
| `manifest.py` | The run-manifest fields the controller can supply: `agent_policy`, `fallback_trigger`, `controller_counts`, `manifest_episode` (A17; not the offline runner) |
| `trace.py` | `to_trace_record` (the proposed `p2-trace-v1`) and `telemetry_record` (digests, config, retrieval log, grounding results) |
| `example.py` | A runnable demonstration with stand-ins, `python -m src.agent.example` |

## How Member 3 calls it

```python
from src.agent.decide import decide, resume, RetrievalResult

decision = decide(
    {"profile_text": ..., "history_movie_ids": [...], "clarifications": [{"question": ..., "answer": ...}]},
    model=my_model_client,      # .generate(prompt) -> str
    retriever=my_retriever,     # .initial_pool(...) and .search(...) -> RetrievalResult
    catalog=my_catalog,         # movie_id -> {"movie_id", "title", "genres", "tags"}
)
decision.response   # {"action": "ask", "question": ...}
                    # {"action": "recommend", "items": [{movie_id, score, rank, explanation} x 10]}
                    # {"action": "error", "code": "failed", "reason": ...}   (proposed; see below)
decision.state      # explicit EpisodeState; .to_dict() is plain JSON
```

State is explicit in both directions. There are two ways to continue after an "ask":

* **Stateless**: Service A stores the question and answer and sends them back in `clarifications`
  on the next `decide` call. Service B keeps nothing between calls. The replayed turns count
  against the budget, so a conversation cannot be given fresh questions.
* **Explicit state**: keep `decision.state.to_dict()` and call `resume(state_dict, answer, source, ...)`.
  The dict is validated as untrusted input and the full event history is kept.

Both gave the same recommendations in the example.

What Member 3 supplies, and what happens when it goes wrong (all proposed, none agreed):

| Supplied | Contract | If it fails |
|---|---|---|
| `model.generate(prompt)` | Returns the model's raw text. Owns its timeout. | An exception is a recorded failed call, handled like unusable output: the B3 fallback, or an explicit `error` if it was the explanation step |
| `retriever.initial_pool(...)` | Ordered B3-style candidates, no watched movie. May use the clarification answers; the controller never edits `profile_text`. | A watched movie in the pool is rejected (`TransitionError`) |
| `retriever.search(query, exclude_movie_ids, max_new)` | Free-text query from the model; returns ids and an index version | An exception falls back to the retriever's own order. Duplicates, watched and over-cap ids are dropped and counted |
| `catalog[movie_id]` | Documents for every movie shown to the model | A missing entry raises `MissingCandidateError` (a data bug) |

The score on each item is a rank surrogate (1.0 for rank 1, falling linearly), not a probability.
Note `error` is not in Service A's current client, which knows only `ask` and `recommend`.

Failures are ordinary outcomes, not exceptions: malformed output, the wrong action for a step, a
ranking that is not a permutation of the pool shown, a repeated question, a model or retriever
exception, an empty retrieval. Each is recorded and ends in the declared fallback. `TransitionError`
is reserved for caller mistakes, such as answering when no question is pending, and `RequestError`
for a malformed request.

## Grounding

The explain prompt says to cite only what was shown. `grounding.py` checks that mechanically: every
title, rating, genre, quoted tag, decade and "you said" must trace to the profile, the answers or a
listed movie. `grounding_policy` chooses what happens to an `unsupported` explanation (`flag` keeps and
records it, `repair` swaps in a text built from the movie's own fields, `reject` treats it as
unusable, `off` skips the check). A text that passes is `passed_screen`, never "supported": truth and
sense are for the human audit. Details and limits: the A15 section of the decisions page.

## Status

| Layer | State |
|---|---|
| Episode state, transitions, parser integration, trace conversion | Implemented |
| Grounding screen and its policies | Implemented as a v0; the criteria are a proposal (A15) |
| `decide` / `resume` entry point and the runnable example | Implemented against stand-ins |
| Tests with test doubles (scripted model and retriever) | Passing: `tests/test_agent_*.py` |
| Real model call, decoding settings, timeouts | Not started; needs Member 6's P2 checkpoint and Member 3's model wrapper |
| Real retrieval | Not integrated; needs Member 3's interface |
| Service B integration, full application | Not integrated |
| Offline evaluation runner | **Not started; ownership is unconfirmed (A12)** |
| Prompt chunking for long pools | Not started (A11) |
| Whether clarification or wider retrieval improves recommendations | Not measured |

The tests use scripted stand-ins, so they show that the controller handles a response correctly.
They do not show that a real model chooses useful actions or writes valid JSON.

## Run it and the tests

From the repository root:

```bash
python -m src.agent.example
```

```bash
python -m unittest tests.test_agent_state tests.test_agent_actions tests.test_agent_controller tests.test_agent_grounding tests.test_agent_decide tests.test_agent_action_reference tests.test_agent_manifest -v
```

The tests need only the standard library. They read Member 1's committed
`docs/p2_sample_examples.jsonl` to replay its example episode through the controller, comparing every
prompt byte for byte with the one the model was trained on, and to check the grounding screen against
the template explanations Member 1 trained on.
