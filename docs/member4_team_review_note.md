# Two questions from the Member 4 controller — for Members 1, 2 and 6

**Status:** a draft from Member 4 (Priyanka) for team discussion, 2026-11-19. Nothing here is decided and
neither question is a complaint about anyone's work. Both come from making the controller agree with
the sources as they stand. Details and the other open items are in
[member4_agent_decisions.md](member4_agent_decisions.md) (A1 and A6).

Sources: Member 1 `member-1-samples-p2` at `321c284`; Member 2 `member2-phase2` at `9032fd1`
(`docs/phase2_interface_contract.md`, "proposed v1"); the submitted midterm report.

## 1. When a budget runs out, which rule applies? (Members 2 and 1, then everyone)

| | Submitted midterm and Member 1's decision record | Proposed v1 contract |
|---|---|---|
| Exhausted budget or unusable action | "fall back to ranking available candidates"; read as: the model ranks the pool it has | "may" fall back to the exact frozen B3 order, labelled `fallback` |
| `recommend` trace | not specified | needs the latest sufficiency and pool judgments to be `true` |
| Trace labels allowed | not specified | `recommend`, `fallback` only |

The controller implements both (`exhaustion_policy`), with the contract rule as the default only
because it is the one the proposed validator accepts. I ran 515 scripted episodes through Member 2's
validator at `9032fd1`: the contract rule was accepted for all 515; the other rule was rejected for
exactly the 177 episodes where the budget ran out, on the label alone ("termination_reason must be
'recommend' or 'fallback'"). The two rules also change what P2's headline number means, since
exhausted-budget users are scored on B3's order under one and on the model's under the other.

**Questions.** Member 2: is the "may fall back" rule meant to cover exhausted budgets, or only errors?
Members 2 and 1: which rule do we want, so the final report describes what actually runs? If it is
"rank the pool", the evaluator needs a labelled variant. Either way it should be settled before any
validation run, not after seeing results.

## 2. Rank-step training pools versus inference pools (Members 1 and 6)

**What the data does.** Member 1's report says that for ranking supervision up to 3 held-out positives
and 2 held-out dislikes are inserted into the final pool and recorded in `injected_movie_ids`
(config `inject_positives: 3`, `inject_negatives: 2`). This is documented and deliberate. In the
committed example (`train:9:1`) the rank prompt has 105 candidates where the retriever returned 100.
The 3 injected positives sit at positions 1, 3 and 4 of the target ranking, the 2 dislikes at 104 and
105, and all 3 injected positives are among the 10 movies the explanation step then covers.

**What inference can show.** At inference the pool contains only what the retriever returned, so there
are no injected movies. The controller therefore accepts a ranking only if it is a permutation of the
pool it showed. That is a safety rule (it keeps recommendations to retrieved, unwatched, in-catalog
movies); it is not a verdict on the training design. Replaying the committed example shows the training
completion for the rank step is rejected for exactly this reason (`tests/test_agent_controller.py`,
`test_training_rank_prompt_differs_from_what_inference_can_show`).

**Questions, not conclusions.** I do not know whether this helps or hurts the model at inference.

1. *Member 1:* what is the injection meant to achieve? (My guess: guarantee positives to rank in the
   many pools that have none. Is that right?) What share of rank-step positives are injected rather
   than retrieved?
2. *Members 1 and 6:* is compatibility with retriever-only pools at inference intended to follow from
   the augmentation, or does it need checking? For example, could the dev checkpoint-selection run
   score the rank step on retriever-only pools, or would a matched variant without injection help?
3. *Member 1:* the explanation step is trained on the first 10 of the injected-containing order, and
   inference shows retriever-only movies. Does that change what the explanation templates can cite?
4. *Member 6:* the rank-step completion lists the injected movies too. Is a completion-only loss over
   such targets expected to transfer to pools that do not contain them?

**What the controller does meanwhile.** It stays strict. If the team decides inference rankings may
include something other than the shown pool, that is a one-line, configurable change on my side, but
it should be a team decision because the evaluator also requires final movies to come from the pool.

## What I am not asking

No change to the frozen P1/B3 results, the data, or the training samples is requested here. I am also
not taking on the offline evaluation runner; who owns it is a separate open question (A12).
