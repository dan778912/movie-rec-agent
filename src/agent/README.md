# Agent loop — Phase 2, Member 4

The controller for P2's bounded clarification and wider-retrieval loop: it tracks one
recommendation episode, decides which action the model is asked for next, checks every model
output, enforces the limits, and records a trace. It is written to be called by Service B
(Member 3); see [Sections 6, 10, and 11 of the roadmap](../../Movie_Recommendation_Agent_Roadmap.md).

**Every limit and policy here is unratified.** The defaults are proposals from named sources, and
the places where sources conflict are listed in
[docs/member4_agent_decisions.md](../../docs/member4_agent_decisions.md). Read that page before
treating any behaviour as the team's decision.

## What is reused, not redefined

| Reused from | What |
|---|---|
| `src/profiles/agent_text.py` (Member 1) | The six actions, `p2-prompt-v1` prompt text and `render_prompt`, `serialize_action` |
| `src/data/p2_actions.py` (Member 1) | `parse_action`, `validate_action`, `validate_episode` (the oracle in the tests) |

Training samples, inference prompts and validation must share these, so this module imports them
and a test checks it has not made copies. If the prompt text changes, `PROMPT_VERSION` changes
there, not here.

## Layout

| File | Role |
|---|---|
| `config.py` | `AgentConfig`: budgets and policies, stored inside each state |
| `state.py` | `EpisodeState`: everything an episode remembers; JSON round trip with validation on load |
| `actions.py` | `parse_model_output`: the shared parser plus "right action for this step" and ID binding |
| `controller.py` | `start_episode`, `next_step`, `build_prompt`, `apply_model_output`, `apply_answer`, `apply_retrieval`, `apply_retrieval_failure` |
| `trace.py` | `to_trace_record` (the proposed `p2-trace-v1`) and `telemetry_record` (digests, config, retrieval log) |

## How it works

The controller does no input or output. It is a state machine: each function takes a state and
returns a **new** state, and never changes its input. The caller does the model and retrieval
calls and feeds the results back, so the same code works offline, inside a stateless service, and
with test doubles.

```python
from src.agent import controller as ctl
from src.agent.config import AgentConfig

state = ctl.start_episode(profile_text, initial_pool, watched_ids, config=AgentConfig(),
                          clarifications=replayed_turns)   # replayed_turns: [] on the first call
while True:
    step = ctl.next_step(state)
    if step.kind == "model":        # step.task names which of the six actions to request
        raw = my_model(ctl.build_prompt(state, catalog))
        state = ctl.apply_model_output(state, raw)
    elif step.kind == "answer":     # a question is pending: a live call returns step.question to the user
        state = ctl.apply_answer(state, user_or_simulator_answer, "answer_source_name")
    elif step.kind == "retrieval":  # search for step.query, adding at most step.max_new movies
        state = ctl.apply_retrieval(state, found_movie_ids, "index_version")
    else:                           # "done"
        break
```

Because every limit is in the state, a conversation that pauses for an answer keeps its budget. A
stateless Service B can either return `state.to_dict()` for the caller to send back, or rebuild the
state from the replayed `clarifications` (they count against the budget).

Failures are ordinary outcomes, not exceptions: malformed output, the wrong action for a step, a
ranking that is not a permutation of the pool shown, a repeated question, an empty or failed
retrieval. Each is recorded and ends in the declared fallback. `TransitionError` is reserved for
caller mistakes, such as answering when no question is pending.

## Status

| Layer | State |
|---|---|
| Episode state, transitions, parser integration, trace conversion | Implemented |
| Tests with test doubles (scripted model and retriever) | Passing: `tests/test_agent_*.py` |
| Real model call, decoding settings, timeouts | Not started; needs Member 6's P2 checkpoint and Member 3's model wrapper |
| Real retrieval | Not integrated; needs Member 3's interface |
| Service B integration, full application | Not integrated |
| Grounding checks, offline runner, prompt chunking | Not started (decisions A15, A12, A11) |
| Whether clarification or wider retrieval improves recommendations | Not measured |

The tests use scripted stand-ins, so they show that the controller handles a response correctly.
They do not show that a real model chooses useful actions or writes valid JSON.

## Run the tests

From the repository root:

```bash
python -m unittest tests.test_agent_state tests.test_agent_actions tests.test_agent_controller -v
```

The tests need only the standard library. They read Member 1's committed
`docs/p2_sample_examples.jsonl` to replay its example episode through the controller and compare
every prompt byte for byte with the one the model was trained on.
