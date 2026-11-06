# Phase 2 evaluation protocol — proposed for team review

**Status: proposed, not frozen or team approved.** This extends the [Phase 1
protocol](evaluation_protocol.md) for the agentic P2 system. The structural
artifact contract is in [Phase 2 interface contract](phase2_interface_contract.md).
Passing a validator does not approve the experimental design or establish that
the model made the logged decisions autonomously.

## Question and comparison

The primary question is whether the complete P2 system improves final
recommendation ranking over the one-shot P1 system on the **same saved users**.
The primary outcome is macro NDCG@10, with P2 minus P1 as the paired difference.
Recall@10 and HitRate@10 are secondary final-ranking outcomes. Report B3, P1,
and P2 together so the reader can distinguish the original retrieval order,
one-shot reranking, and the agentic system. B0 popularity and B2 BM25 belong in
the broader shared results table when their matching outputs are available.

P2 starts from each user's **exact frozen B3 top-50 order**. It can ask a
clarifying question and add candidates through re-retrieval before ranking 50
items. P1 uses the same initial B3 movie-ID set and cannot expand it. Thus
P2-versus-P1 tests the *combined* P2 checkpoint, clarification policy, answer
simulation, retrieval policy, and ranking path. A positive difference alone
does not isolate which component caused it. The P2 checkpoint derives from P1;
record both checkpoint identities, training changes, prompts, retrieval index,
and inference settings. If feasible, run separately named clarification-only
and re-retrieval-only ablations with a fixed checkpoint and otherwise matched
inputs. Do not describe them as the primary P2 result.

## Frozen data and chronology

| Item | Rule |
|---|---|
| Dataset and catalog | `ml25m-global-v1`; the training-defined catalog and global temporal cutoffs from Phase 1. No test interaction or test tag enters a profile, index, training sample, clarification answer, or model-selection decision. The existing limitation of static genome metadata remains disclosed. |
| Targets | All distinct in-catalog positive movies in the corresponding held-out split, rating at least 4.0. Retrieval misses remain in the recall denominator. |
| History and exclusion | Validation uses training history; test uses training plus validation history. Exclude every `history_movie_ids` movie, including items absent from the shortened textual profile. |
| Development users | The saved 300 IDs in `configs/validation_users_300_v1.json`; use for decisions and error analysis. |
| Test users | The saved 500 IDs in `configs/test_users_500_v1.json`; do not resample or remove failures. |
| Starting pool | The per-user frozen B3 50-item prediction file, including its order and SHA-256. P2's trace must reproduce this list exactly. |
| P1 control | The existing frozen P1 50-item predictions on these 500 users, if P2 receives the same initial profile, catalog, and B3 pool. If any input representation changes, rerun a separately versioned matched P1 control before making a controlled P1/P2 claim. Preserve the historical P1 result either way. |

The [published 500-user P1/B3 comparison](results/member2_p1_b3_test_500_v1/README.md)
is historical evidence: P1 minus B3 NDCG@10 is +0.023992, with a paired 95%
interval [0.016787, 0.031517]. Its prediction files, user set, settings,
results, and hashes must not be rewritten. Test targets were opened for that
authorized scoring in October 2026. Record this chronology; subsequent P2
checkpoint, prompt, simulator, retrieval, threshold, and fallback choices must
come from training/validation and be committed before P2 test inference, not
from the observed P1/B3 test outcomes. Preserve owner declarations and source
hashes so this boundary can be audited.

The frozen 500-user B3 prediction JSONL and P1 prediction JSONL are **not
tracked in Git**. The [test archive](results/member2_p1_b3_test_500_v1/README.md)
records their SHA-256 values (`44ec13348849458fa58746e126369afc5fa0c2cf4bbd867bc732b83f412b93a1`
for B3 and `15d5d332f61224824f339bdd73275522dec71a93ae05d0a460f9bb7fb1412634`
for P1). Obtain these original files from team storage and verify their hashes
before P2 trace validation or scoring. Archived aggregate reports alone cannot
check an individual user's starting pool or rescore a new paired comparison.
The Phase 2 bootstrap/table code binds compared bundles to the same evaluator
implementation SHA. Re-score the **original hash-verified B3 and P1 prediction
JSONLs** with the Phase 2 evaluator and its frozen configuration, then compare
those fresh bundles with P2. This is re-scoring, not model re-inference or a
revision to the published historical P1/B3 archive; preserve the old reports.

## Clarification simulation

MovieLens has no user answers to conversational questions. Freeze a
**versioned deterministic answer simulator** before test inference, and report
its limitations. Both P1 and P2 receive the **unchanged Phase 1 canonical
starting profile** and frozen B3 pool. Do not mask or regenerate that starting
profile while reusing the historical B3/P1 predictions. A P2 clarification may
reveal only a fact derived from that user's *available pre-cutoff history* that
was absent from the canonical profile's visible text (for example, a rating
outside its selected examples). The simulator must never read validation
targets for a validation run or test targets for a test run. The complete
watched IDs may be supplied to
enforce exclusion, but they are not a license to reveal future ratings. Keep
the source interaction ID/timestamp or a stable source digest, question match
rule, answer template, and simulator version for audit outside the public
free-text answer. If no eligible answer fits the question, return a declared
noninformative answer; never search held-out targets to make the answer useful.

Define and freeze on validation: how eligible omitted history facts are selected,
which question intents the simulator recognizes, how it chooses among matching
facts, its exact answer and nonanswer templates, and how repeated or unsafe
questions are handled. Verify that the same initial profile and pool were used
for the P1 control and P2. Report the number of questions, answerable questions,
noninformative answers, and questions revealing a genuinely new preference
fact. The structural evaluator can check that an answer and source label exist;
it cannot prove the answer is temporally valid or that the model did not already
see the fact. Audit these claims separately. Simulated answers support only a
controlled offline workflow claim, not real-user satisfaction.

## Run policy and validity

The [Phase 2 validation scoring draft](../configs/evaluation_phase2_validation_draft_v1.json)
and [test scoring draft](../configs/evaluation_phase2_test_draft_v1.json) retain
the saved populations and metric settings for review. Their status is draft;
approve and freeze a new shared configuration before official P2 test scoring.
All B3, P1, and P2 prediction files in a paired table must be scored under that
same final configuration and evaluator implementation.

Freeze one run manifest for each split before its inference: dataset/subset and
B3 hashes; checkpoint and tokenizer revisions; training sample and prompt
versions; retrieval/index version and catalog hash; simulator version; sampling
and generation settings/seeds; agent action grammar and loop cap; timeouts;
fallback policy; code commit; hardware and runtime measurement method. An
identical declared policy applies to all selected users. The owner exports the
standard 50-rank prediction JSONL plus one structured trace JSONL record per
user, following the [contract](phase2_interface_contract.md). P2 final 50 may
be a subset of a larger final candidate pool. Its trace begins with the exact
B3 50 and records each added candidate in order. Reject invalid catalog IDs,
watched items, duplicate IDs, malformed actions, missing users, invalid ranks,
and ungrounded pool membership. A syntactically valid trace is necessary but
does not prove that logged events represent actual model calls.

Create a separately versioned Phase 2 evaluation configuration carrying the
same frozen data, user, relevance, metric, and bootstrap settings as the
matching Phase 1 test run, plus the reviewed P2 trace and fallback policy.
Do not edit `configs/evaluation_test_frozen_v1.json`, whose status and hashes
were scoped to P1/B3 inference.

Freeze fallback triggers, explanation policy, and failure taxonomy on
validation **before** final test. Under the v1 trace contract, a fallback's
final ranking is the frozen B3 order, recorded as
`fallback_source: "frozen_b3_order"`; record the truthful partial action trace
and `fallback_reason` without inventing decisions that did not occur.
Every selected user must still have a complete final record and trace. Record
the failure and fallback reason, source ranking used, and whether the result is
model generated or fallback; never silently drop a user. If a required record
or input hash is missing, repair the export or rerun inference rather than
scoring an incomplete file. The official table uses all saved users under the
predeclared system policy. Report fallback and error rates alongside it; a
success-only slice is diagnostic and cannot replace the official estimate.

The trace's eight-**control**-event cap, plus one rank and one explanation
event, is a structural export rule (at most ten serialized events). Member 4 must also
define an online loop cap, repeated-question handling, validation of model
actions, retrieval and inference timeouts, and a terminal response on failure.
The operational cap and policy need not be encoded as eight user-facing turns.

## Metrics and uncertainty

Score the 50-item final ranking with the existing binary, macro-averaged
Recall@10, NDCG@10, HitRate@10, Recall@20, and Recall@50 definitions. Use the
same complete target set and 500 users as P1/B3. Distinguish these measures:

| Measure | Definition and interpretation |
|---|---|
| Final ranking Recall@50 | Relevant targets appearing in P2's submitted top 50. It can rise when re-retrieval supplies relevant movies, but ranking may omit other pool members. |
| Initial B3 pool Recall@50 | Relevant targets among the exact frozen B3 50; shared starting opportunity. |
| Expanded-pool recall | Relevant targets anywhere in the final candidate pool divided by all eligible positive targets, per user then macro averaged. Also report final-pool size distribution. This is not Recall@50 if the pool exceeds 50. |
| Added-candidate yield | Added movies that are relevant divided by the number of added movies, with the count and denominator explicit. This is a diagnostic requiring target access only during authorized scoring. |
| Decision/failure rate | Users with each action, re-retrieval, clarification, invalid action, timeout, loop termination, or fallback divided by all selected users; retain counts. |
| Latency/cost | Sum Service B processing time across every call in an offline episode, excluding simulator/human waiting time; separately report online request latency for `ask` and `recommend`. Include stage timings, p50/p95, sample count, hardware, concurrency, index version, model tokens/calls, and approximate compute/API cost. Exclude evaluator command runtime from model latency. |

For the primary P2-minus-P1 NDCG@10 and important secondary differences, match
per-user scores by ID and use the existing **10,000 paired-user percentile
bootstrap resamples**, seed `20261002`, confidence 95%, with the same
interpolation convention as Phase 1. Report the observed difference, interval,
and exact input hashes. An interval including zero leaves direction uncertain
under this procedure. The interval describes user variation for these fixed
systems; it does not incorporate training-seed, simulator, or index uncertainty.
Do not select among multiple P2 variants by reading the 500-user test metrics.

## Explanation and failure audit

Every displayed recommendation needs a justification tied to supplied catalog
metadata and available user preferences. Keep generated text and structured
evidence references when available. On validation, define an audit rubric and
sample rule; then apply them to a saved, reproducible sample of test outputs.
For each explanation, check factual claims against the versioned catalog,
pre-cutoff profile/clarification facts, and movie ID; record unsupported claims,
wrong movie or preference references, empty explanations, and disagreement
resolution. Report numerator and denominator, examples, and reviewer method.
Ranking accuracy is not evidence of explanation grounding.

Failure analysis should compare P1 and P2 per user and group by: initial B3
pool had/ lacked a positive, re-retrieval attempted/succeeded, clarification
asked/answered/noninformative, final pool size, fallback/timeout, and
explanation-grounding result. Show representative successes and failures with
temporal/provenance checks. Do not choose a different P2 setting from test
failures.

## Completion gates and dependencies

1. **Contract reviewed:** Members 2–6 agree on the offline schema, answer
   simulator, source and version hashes, failure/fallback policy, latency
   definitions, and Service A/B request/response. Record the accepted version.
2. **Evaluator ready:** Synthetic positive and negative tests cover exact B3
   start, ordered expansion, watched/catalog exclusion, complete user coverage,
   action sequence and cap, final ranking membership, malformed JSON, and
   missing or mismatched traces. Published Phase 1 artifacts still validate.
3. **Validation ready:** Member 1's P2 samples and catalog seed; Member 3's
   versioned retrieval index and Service B; Member 4's agent/actions/traces;
   Member 6's loadable P2 checkpoint and generation settings; Member 5's
   Service A/DB/frontend integration. Freeze settings from validation results.
4. **Final comparison ready:** Receive the original hash-verified B3/P1
   prediction files and re-score them under the same Phase 2 evaluator commit
   and configuration used for P2; preserve the historical P1/B3 reports.
   Receive P2 predictions, traces, run manifest, checkpoint/index identifiers,
   and latency/error records. Produce a paired P1/P2 result and a B3/P1/P2
   table on exactly 500 users with coverage and interval; add B0/B2 only when
   their matching outputs exist. Report an unfavorable or uncertain result as
   such. A positive effect is not a completion requirement.
5. **End-to-end acceptance:** Demonstrate a real Service B checkpoint/index
   through Service A and the React UI: enter ratings, ask and answer a
   clarification if triggered, expand a candidate pool if triggered, exclude
   watched items, return ranked recommendations with grounded explanations,
   persist the conversation, and replay a trace. Record measured latency,
   failures, setup/redeployment steps, and the limits of simulated answers.

No final P2 performance values are claimed in this proposed document.
