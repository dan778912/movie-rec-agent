# Phase 2 interfaces and offline trace — proposed v1

**Status: proposed for Members 2–6 to review.** `p2-trace-v1` below is the
intended offline artifact shape and evaluator input. The Service A/B HTTP
extension is a proposed integration target; the current
`origin/member5-service-a` `/decide` client is explicitly provisional and
supports only the three legacy request fields and `ask`/`recommend` responses.
The [experiment protocol](phase2_evaluation_protocol.md) defines the controlled
comparison, simulator, and acceptance rules. Structural checks cannot prove
temporal provenance, autonomous decisions, or explanation truthfulness.

## Ownership and data flow

| Boundary | Owner and responsibility |
|---|---|
| Frontend → Service A | Member 5 accepts ratings, clarification answers, and displays recommendations; Service A persists sessions, watched history, answers, recommendations, and trace references in PostgreSQL. |
| Service A → Service B | Member 5 sends the current profile and conversation state over internal HTTP; Member 3 hosts stateless model inference. The request contains enough state to reconstruct a decision without Service B reading PostgreSQL. |
| Inside Service B | Member 3 provides retrieval/index and checkpoint loading; Member 4 parses validated model actions, enforces loop limits, invokes retrieval, and emits structured events; Member 6 supplies the selected checkpoint and loading settings. |
| Offline producer → Member 2 | Members 3/4/6 export a 50-rank P2 prediction file, a complete `p2-trace-v1` file, and a run manifest. Member 2 validates the artifacts and scores them against the frozen population. |

The live HTTP exchange may contain zero or more `ask` turns before the final
`recommend` turn, within the agreed online loop cap. The offline trace is
**one complete per-user episode** ending
in a 50-item ranking, assembled from all decisions in that episode. Do not
mistake Service A's current short `ask`/`recommend` database summaries for the
complete agent trace; they do not capture B3 starts, expansions, model actions,
index versions, or timings.

## Offline prediction JSONL

One UTF-8 JSON object per selected user, matching the Phase 1 ranking shape.
Examples below are expanded for reading; write each complete object on one
line in the actual JSONL files:

```json
{
  "user_id": 123,
  "split": "validation",
  "method": "p2",
  "recommendations": [
    {"movie_id": 104, "score": 0.9, "rank": 1},
    {"movie_id": 101, "score": 0.8, "rank": 2},
    {"movie_id": 102, "score": 0.7, "rank": 3}
  ]
}
```

The example is a **three-item synthetic illustration**; the official run
requires exactly **50** recommendations per user. `user_id`, `movie_id`, and
`rank` are positive integers excluding booleans. `score` is finite numeric
data, not necessarily a probability; rank is authoritative and must be
consecutive in array order. Movies are distinct, in the versioned catalog,
unwatched in the split-specific complete history, and members of that user's
trace `final_candidate_movie_ids`. Every saved user appears exactly once; no
other user may appear. `split` is `validation` or `test`, and `method` is `p2`.
Optional item-level `explanation` text may be exported, but the mandatory
top-ten explanation record is in the trace. If both are present, the owner
must check that the displayed text and trace text agree.

P2 **does not** have P1's exact-final-set constraint. P2's 50 ranked movie IDs
may be any distinct subset of its final candidate pool. If no retrieval occurs,
the final pool remains the original B3 50 and the submitted 50 are a permutation
of it. Candidate-pool recall across a larger pool is a separately calculated
diagnostic, not the submitted ranking's Recall@50.

## Offline trace JSONL: `p2-trace-v1`

The evaluator receives a separate `--trace` JSONL and the frozen B3
`--candidate-pool` prediction JSONL. There is exactly one trace record per
expected user, matching the prediction `user_id` and `split`:

For the 500-user test run, the original B3 JSONL must be handed off from team
storage and match SHA-256
`44ec13348849458fa58746e126369afc5fa0c2cf4bbd867bc732b83f412b93a1`.
The original P1 JSONL for the paired comparison must match
`15d5d332f61224824f339bdd73275522dec71a93ae05d0a460f9bb7fb1412634`.
These prediction files are not tracked in Git; [archived summaries](results/member2_p1_b3_test_500_v1/README.md)
do not substitute for per-user pool validation. Re-score the original
hash-verified B3/P1 files with the same Phase 2 evaluator implementation as
P2 before generating new paired intervals or joint tables. Keep the old
published reports intact.

```json
{
  "schema_version": "p2-trace-v1",
  "user_id": 123,
  "split": "validation",
  "method": "p2",
  "initial_candidate_movie_ids": [101, 102, 103],
  "events": [
    {"action": "assess_sufficiency", "sufficient": false},
    {
      "action": "ask_clarification",
      "question": "Which genre would you prefer?",
      "answer": "I prefer comedy.",
      "answer_source": "pre_cutoff_history_sim_v1"
    },
    {"action": "assess_sufficiency", "sufficient": true},
    {
      "action": "evaluate_candidates",
      "candidate_movie_ids": [101, 102, 103],
      "pool_sufficient": false
    },
    {
      "action": "retrieve_candidates",
      "added_movie_ids": [104],
      "index_version": "catalog-index-v1"
    },
    {
      "action": "evaluate_candidates",
      "candidate_movie_ids": [101, 102, 103, 104],
      "pool_sufficient": true
    },
    {"action": "rank_candidates", "movie_ids": [104, 101, 102]},
    {
      "action": "explain_recommendations",
      "explanations": [
        {"movie_id": 104, "text": "Its catalog genre matches your stated comedy preference."},
        {"movie_id": 101, "text": "Its listed genres match your earlier likes."},
        {"movie_id": 102, "text": "It shares a catalog tag with a movie you liked."}
      ]
    }
  ],
  "final_candidate_movie_ids": [101, 102, 103, 104],
  "termination_reason": "recommend"
}
```

This example is synthetic and uses a required prediction count of three; it
does not claim those IDs or explanation facts are valid in the real catalog.
The production record has 50 initial B3 IDs and the explanation event covers
the first ten ranked movies. The initial list must equal the frozen B3 list
**in order**, not only as a set. The final list is that initial list followed
by each `retrieve_candidates.added_movie_ids` list in event order. Added IDs
must be fresh, valid catalog movies absent from complete watched history.
The lists preserve provenance for every ranked ID.

### Event vocabulary and order

| `action` | Required fields | Meaning |
|---|---|---|
| `assess_sufficiency` | `sufficient`: Boolean | P2's judgment of whether available preference information suffices. At least one is required for `recommend`; a truthful early-failure trace may omit it. |
| `ask_clarification` | Nonempty `question`, `answer`, `answer_source` strings | One simulated question/answer. Allowed after the latest sufficiency assessment says `false`; another assessment can follow. `answer_source` names the frozen simulator, not a claim of a real human answer. |
| `evaluate_candidates` | `candidate_movie_ids` equal to the **entire current pool in order**; `pool_sufficient`: Boolean | P2's judgment of the current pool after a `sufficient:true` assessment. At least one is required for `recommend`; a truthful early-failure trace may omit it. The exact list binds the judgment to the pool version being evaluated. |
| `retrieve_candidates` | Nonempty `added_movie_ids` of fresh catalog/unwatched IDs; nonempty `index_version` | Allowed after the latest candidate evaluation says `false`. A changed pool requires a later evaluation before ranking. |
| `rank_candidates` | `movie_ids` exactly equal to the final prediction's 50 movie IDs in order | Exactly one required. All clarification, assessment, evaluation, and retrieval events occur before it. |
| `explain_recommendations` | `explanations`: ordered objects with positive `movie_id` and nonempty `text` | Exactly the first `min(10, required_count)` ranked IDs in order. Required after ranking. The validator checks binding and presence, not truthfulness. |

There are at most **eight control events** (`assess_sufficiency`,
`ask_clarification`, `evaluate_candidates`, `retrieve_candidates`) in an
offline episode, plus exactly one `rank_candidates` and one
`explain_recommendations` event: at most ten serialized events. Multiple
assessments, questions, evaluations, and retrievals can occur within the
control-event cap. Any new
assessment or clarification invalidates the prior pool judgment, so a normal
recommendation needs a subsequent current-pool evaluation. Retrieval also
invalidates that judgment. A successful
`termination_reason: "recommend"` requires the latest sufficiency and pool
assessments to be `true` before ranking. If the loop cannot resolve a `false`
decision or a model/retrieval error occurs, the producer may use the
predeclared v1 fallback: `termination_reason: "fallback"`, a nonempty
`fallback_reason`, and `fallback_source: "frozen_b3_order"`. Its complete
50-item prediction must match the frozen B3 **order exactly**, even if the
trace records an earlier expansion. It still supplies top-ten explanations
and remains in the official score. An early-failure trace may omit sufficiency
and candidate assessments; record only decisions that actually happened.
Do not fabricate assessments to make a timeout appear successful. The run
manifest states fallback triggers and explanation policy. Never encode a
timeout by omitting the user or quietly replacing their record.

The validator rejects malformed/nonfinite JSON, duplicate keys, missing/extra
users, wrong method or split, invalid actions/fields/types/order, invalid or
repeated IDs, watched or out-of-catalog movies, pool mismatches, incorrect
ranking/explanation bindings, and control-event-cap violations. It hashes the
prediction, trace, and frozen B3 files in its report. It does not verify that
model calls actually occurred, that a simulated answer used only pre-cutoff
history, or that explanation prose is grounded; these require provenance and
audit under the protocol.

## Run manifest and evidence

Each owner run supplies a machine-readable manifest or report identifying:

- Dataset, catalog, profile, frozen users, B3 predictions, P2 predictions,
  and trace files by SHA-256; split; selected user count; run ID; code commit.
- P1 and P2 checkpoint IDs/revisions, tokenizer, training sample version,
  prompt/template version, decoding parameters, score kind (including
  `rank_surrogate` if scores are `-rank`), seeds, and checkpoint-loading
  example. Record any changed architecture or prompt separately.
- Retrieval index version/hash, ingestion source/catalog version, embedding
  model, filtering and expansion settings; simulator code/version and its
  pre-cutoff answer source; action parser, loop cap, timeout, and fallback policy.
- Per-user terminal status, fallback/error cause, model/retrieval call count,
  answerability outcome, and stage/end-to-end timings. Include raw model action
  output or a stable digest plus parsed action and validation result so the
  decision path can be audited. Aggregate p50/p95 and
  counts with hardware, concurrency, and approximate compute/API cost.

Hashes and owner assertions should be labeled as such; Member 2 independently
checks the artifacts it receives and does not infer checkpoint identity from
movie IDs. Large predictions, traces, data, and weights can remain in agreed
team storage; preserve immutable hashes and exact reproduction commands in Git.

## Service A ↔ Service B HTTP contract to ratify

`origin/member5-service-a` currently calls `POST /decide` with
`profile_text`, full `history_movie_ids`, and a `clarifications` array of
`{question, answer}`. It expects either `{action:"ask", question}` or
`{action:"recommend", items:[{movie_id,score,rank,explanation}]}`. Preserve
these fields while integrating the real Service B. The proposed versioned
extension adds `contract_version`, a correlation `request_id`, and a structured
`decision_trace`/`trace_id` in the response; Service A persists them with its
conversation turn. The team should settle exact envelope names and error codes
with Members 3–5 before deployment. Keep the user-facing response small: a
nonempty question for `ask`, or **ten** distinct, unseen ranked catalog items
with finite scores and nonempty grounded explanations for `recommend`.

Service B owns checkpoint/index loading and internal re-retrieval; Service A
does not need a third user-facing action for `retrieve_candidates`. Service A
must send prior clarification turns back on subsequent calls because Service B
is stateless, enforce a request timeout, and persist the full decision trace
or a stable reference to it. Invalid responses or timeouts need explicit
recorded failures. The existing mock's fixed minimum-history question rule is
a test double; it is not evidence of autonomous P2 sufficiency judgment.

The offline export may contain a 50-item ranked list so Member 2 can compute
Recall@50. The live UI may show only the first ten items. Both outputs must
come from the same declared ranking policy; the shorter UI response must not
be mistaken for an incomplete offline prediction.
