# Phase 2 producer run receipt — proposed v1

**Status: proposed for Members 2–6 to review.** This specifies the
`p2-run-manifest-v1` handoff receipt. The validator checks file identity and
arithmetic without parsing or scoring held-out labels. The existing
[`p2-trace-v1` contract](phase2_interface_contract.md) defines the episode
events and recommendation format; this receipt does not replace that trace.
The full synthetic receipt is built in
[`tests/test_p2_run_manifest.py`](../tests/test_p2_run_manifest.py).

## Freeze and receipt

Before inference, the producer records the `policy` object and its
`policy_sha256` in an immutable, dated team record. Compute the digest from the
UTF-8 bytes of compact JSON with sorted keys, `allow_nan=False`, and one final
LF (`\n`). The producer then adds `output_sha256`, `episodes`, and `aggregate`
to the receipt after inference. The internal digest check only detects
inconsistency in the submitted file; it cannot prove when the policy was
frozen. Keep the external dated record or commit for that evidence.

The top-level JSON object has the required fields below and may also include
the optional `owner_notes` field:

| Field | Meaning |
|---|---|
| `schema_version` | Literal `p2-run-manifest-v1`. |
| `run_id` | Unique 3–128 character ID using letters, digits, dots, underscores, or hyphens. |
| `policy` | Pre-inference declaration described below. |
| `policy_sha256` | Lowercase SHA-256 of canonical `policy` JSON bytes. |
| `output_sha256` | Exact `predictions` and `trace` SHA-256 digests. |
| `episodes` | One telemetry object for **every** frozen user, including fallbacks. |
| `aggregate` | Counts, costs, and latency percentiles recomputed from `episodes`. |
| `owner_notes` | Optional nonempty text for additional provenance. |

All JSON is strict UTF-8: duplicate keys and nonfinite numbers are rejected.
All SHA-256 digests are lowercase 64-digit hexadecimal strings. Additional
fields within the required objects are rejected under this v1 schema, so a
producer can catch misspellings before submitting a run.

## Policy fields

| Object | Required fields and checks |
|---|---|
| Root `policy` | `dataset_version`, `split`, and `user_count` match the evaluation config and frozen population; `code_commit` is a full lowercase 40- or 64-digit Git ID; `code_dirty` is Boolean. It also contains `source_sha256`, `model`, `retrieval`, `simulator`, `agent`, and `runtime`. |
| `source_sha256` | Exact bytes of `config`, `dataset_manifest`, `catalog`, split `profiles`, split `targets`, frozen `users`, and ordered B3 `candidate_pool`. The target file is **hashed after the freeze but its labels are not parsed**. The config, dataset manifest, and frozen user list must also agree on their embedded hashes and versions. |
| `model` | Nonempty `p1_checkpoint_id`, `p1_checkpoint_revision`, `p2_checkpoint_id`, `p2_checkpoint_revision`, `tokenizer_id`, `training_sample_version`, and `prompt_version`. `decoding` records finite `temperature` from 0 to 2, `top_p` above 0 through 1, positive `max_new_tokens`, nonnegative integer `seed`, and nonempty `score_kind` (for example `rank_surrogate` if scores equal `-rank`). |
| `retrieval` | Nonempty `index_version`, `embedding_model_id`; lowercase `index_manifest_sha256`; `ingestion_catalog_sha256` equal to the frozen catalog hash; `max_candidate_pool_size` at least the B3 pool size; `watched_exclusion: true`. Each successful retrieval event must carry this `index_version`. |
| `simulator` | Nonempty `version`, `answer_source`, and `pre_cutoff_source`; lowercase `code_sha256`. Each clarification event's `answer_source` must match the declared value. The simulator owner must separately demonstrate that answers use only eligible pre-cutoff history. |
| `agent` | Nonempty `parser_version` and `explanation_policy_version`; `max_control_events: 8`, `max_serialized_events: 10`, positive `timeout_ms`, `fallback_source: "frozen_b3_order"`, and a nonempty unique list of `fallback_triggers` reason codes. |
| `runtime` | Nonempty `hardware` and `measurement_version`, positive integer `concurrency`, and `latency_scope: "offline_episode_excludes_simulator_wait"`. Online `ask` and `recommend` request latency needs a separate report. |

The receipt checks the *format* of checkpoint, index-manifest, and simulator
identifiers and digests. It cannot verify their contents without the named
artifacts or an owner loading/ingestion demonstration. The controlled P1
prediction file and its evaluator bundle are verified separately by the
paired-comparison workflow.

## Per-user episode and aggregate

Each `episodes` object has exactly:

- `user_id`, `termination_reason` (`recommend` or `fallback`),
  `fallback_reason` (null on success, otherwise exactly the trace text), and
  `fallback_trigger` (null on success, otherwise a code from the frozen policy).
- Nonnegative integer `model_call_count`, `retrieval_call_count`,
  `informative_answer_count`, `noninformative_answer_count`, and
  `invalid_action_count`. A normal recommendation has at least one model call.
  Retrieval calls can exceed successful retrieval events when a call fails,
  but cannot be fewer. The two answerability counts sum to the number of
  clarification events. Their informative/noninformative classification is
  the producer's assertion and needs simulator audit.
- Boolean `timed_out` and `loop_cap_hit`; finite nonnegative `cost_usd`.
- `latency_ms` with finite nonnegative `model`, `retrieval`, and `end_to_end`
  values. Model plus retrieval time cannot exceed episode end-to-end time.
  The values sum Service B work across calls and exclude simulator wait time.

The `aggregate` object includes `user_count`, `fallback_user_count`,
`fallback_by_trigger`, `invalid_action_user_count`, `timeout_user_count`,
`loop_cap_user_count`, the five episode counters summed over users,
`cost_usd` summed over users, and `latency_ms` p50/p95 for each of `model`,
`retrieval`, and `end_to_end`. The percentile convention is linear
interpolation at `(n-1) × p`. The validator recalculates every aggregate and
requires exact integer agreement and close numeric agreement. A diagnostic
report still needs to inspect representative raw actions, explanation claims,
and actual timing instrumentation.

## Validate the handoff

Run this command from the repository root with the actual immutable files:

```powershell
python -m src.evaluation.p2_run_manifest `
  --manifest outputs/predictions/phase2_validation/run_manifest.json `
  --config configs/evaluation_phase2_validation_draft_v1.json `
  --candidate-pool outputs/predictions/phase2_validation/b3.jsonl `
  --predictions outputs/predictions/phase2_validation/p2.jsonl `
  --trace outputs/predictions/phase2_validation/p2_trace.jsonl
```

Exit code 0 emits a JSON report with verified source hashes, full user count,
and recomputed aggregate. A malformed manifest or source mismatch exits 2;
an invalid prediction/trace exits 1 with the existing validation report.
Keep the emitted report with the run receipt and the separately audited
checkpoint, index, simulator, latency, and action evidence.
