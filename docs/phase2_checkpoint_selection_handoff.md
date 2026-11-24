# Phase 2 checkpoint selection and validation handoff — decision draft

**Updated 2026-11-24:** the corrected untuned Qwen control and original ordered B3 pool have arrived and were scored on all 300 validation users. Member 1 has published a hash-verified v2 sample handoff. The P2 checkpoint, end-to-end export, and original P1 validation prediction JSONL are still outstanding. The checkpoint and runtime policy must be frozen before P2 inference. The [Phase 2 protocol](phase2_evaluation_protocol.md), [interface contract](phase2_interface_contract.md), and [runbook](phase2_member2_runbook.md) remain the detailed sources.

## New handoffs and decisions since 2026-11-18

- Member 1's [v2 handoff ZIP](https://github.com/dan778912/movie-rec-agent/blob/member-1-samples-p2/data/processed/handoff/member1_p2_v2_handoff.zip) has SHA-256 `18af124d1f42c1b57460db6ee0902c062aec17d5b7038c90f115553e08be32e2`. Independent verification passed its ZIP hash and all 14 internal checksum entries. It contains 22,802 train and 2,178 dev action samples, 3,974 train and 400 dev episodes, plus quality evidence. Its 400 dev users are disjoint from the saved 300 validation and 500 test users. The quality report records zero errors and 4,374/4,374 replay boundaries truncated to the pre-cutoff history. Member 3 still needs to establish the online index's temporal boundary separately.
- Most v2 rank samples use 101–105 candidate IDs because they inject held-out training-period preferences, while the proposed Service B cap is 100. Member 6 and Member 3 should smoke-test an ordinary 100-candidate inference case and keep this training-versus-serving difference explicit.
- Member 1 confirmed the proposed two-stage checkpoint rule: shortlist the three checkpoints with lowest completion loss on the 400-user dev set, report strict action parse rates, then choose among the shortlist by macro NDCG@10 on the frozen 300-user end-to-end validation; break an exact tie by earlier training step. This replaces Member 1's earlier note that dev alone selects the final checkpoint. Member 6 needs to publish the complete shortlist; the team needs to freeze the shared inference policy before inspecting the 300-user P2 results.
- The [untuned control report](results/member2_qwen_base_validation_300_v1/README.md) scores the hash-verified original B3 pool and corrected Qwen export: B3 NDCG@10 `0.014578`, Recall@10 `0.005026`, HitRate@10 `0.11`; untuned Qwen `0.019772`, `0.007733`, `0.15`. Both pass 300/300 validation with zero errors. The original P1 prediction JSONL is still needed for the strict current-version paired CI against Qwen.
- Member 4's agent defaults are two questions, one retrieval, at most 100 candidates, eight control events, zero retries, and grounding flagged for audit. The current Member 2 validator accepts B3-order `fallback` on budget exhaustion. If the team chooses to rank the current pool instead, the agent, trace schema, evaluator, manifest, and tests must explicitly name and accept that outcome before inference. Invalid explanations currently cause incomplete exports; the Member 2 policy below specifies a labelled deterministic explanation fallback for the producer to implement. Use GPU smoke runs to set the offline timeout. No numeric quality, validity, or latency pass/fail threshold has been frozen.

## Member 2 evaluator policy for the first P2 validation run

**Member 2 confirmation, 2026-11-24; the producer settings and measured timeout still need a shared freeze before inference.** Use the top-three dev-loss shortlist and 300-user macro NDCG@10 selection rule above. Keep the frozen simulator `pre_cutoff_history_sim_v1`, prompt `p2-prompt-v1`, greedy decoding, no silent truncation, and the Member 6 recommended 6,144 input/1,024 output token limits. Member 3 owns the offline inference/export runner; Member 2 validates, scores, bootstraps, and analyzes its outputs.

- Limit each episode to two clarification questions, one retrieval expansion, at most 100 candidates and eight control events, followed by one rank and one top-ten explanation event. Keep zero extra model retries. Count every invalid output, model call, fallback, and timeout. The official export must include every one of the 300 users.
- For an unresolved decision, exhausted budget, invalid action before explanation, retrieval error, or earlier model timeout, use `termination_reason: "fallback"` and return the original B3 50 **in their exact rank order**. The trace must name `fallback_source: "frozen_b3_order"` and its reason. This is the current evaluator contract. Ranking the current pool on exhaustion would be a separately versioned contract change, not an alternate label for this run.
- If the final ranking is valid but the explanation model call fails, times out, or produces unusable output, retain that ranking and generate all ten explanations deterministically from each movie's displayed catalog fields. Use no inferred user preferences in this fallback text. Mark the `explain_recommendations` event with `source: "catalog_metadata_fallback"` and `explanation_fallback_reason`; preserve the failed raw output or its SHA-256 and parser error in the decision log. Count a rejected model output in `invalid_action_count`; a model-call timeout counts as a model call and timeout, not as an invalid output. The current trace validator accepts these explicit provenance fields and still checks the ten explanation/movie bindings. A valid model explanation uses `source: "model"`.
- For parseable explanations flagged as unsupported by the deterministic grounding screen, retain the original model text and flag it for the human grounding audit. Do not silently repair or reject it. Report grounding failures separately from ranking quality.
- The 25-second offline episode deadline is not approved. Member 3 and Member 4 should measure real GPU calls and full episodes, including a two-question/one-expansion case, then record separate per-call and whole-episode limits with headroom before the 300-user export. Report observed p50/p95 latency, timeouts, invalid-output rate, fallback rate, and approximate cost. There is no numeric pass/fail threshold for those measurements or NDCG@10; the structural gate is 300/300 valid prediction and trace records. Freeze any numeric test acceptance gate on validation evidence before opening the 500-user test outputs.

The inventory below incorporates the received files; the local regeneration note remains a dated 2026-11-18 observation.

## Two populations and a proposed selection rule

| Population | Current source | Intended use |
|---|---|---|
| 400 user-disjoint training-period `dev` episodes | Member 1's [v2 handoff ZIP](https://github.com/dan778912/movie-rec-agent/blob/member-1-samples-p2/data/processed/handoff/member1_p2_v2_handoff.zip) | Screen training checkpoints for loadability, strict action parsing, and completion loss. These labels and episode NDCG values come from a proxy ranker and are not end-to-end P2 evaluation results. |
| Frozen 300 validation users | `configs/validation_users_300_v1.json` and `configs/evaluation_phase2_validation_draft_v1.json` | Run shortlisted checkpoint(s) end to end on identical profiles, ordered B3 pools, simulator, index, prompt, decoding, fallback policy, and selected users. The proposed primary system metric is macro NDCG@10 on all 300 final rankings, including fallback episodes. |
| Frozen 500 test users | `configs/test_users_500_v1.json` | One final evaluation after model and policy selection; never choose a checkpoint or threshold using these targets. |

**Proposed resolution for team review:** Member 6 shortlists the three checkpoints with lowest completion loss on the 400-user dev set, reports strict action parse rates, and freezes the shortlist before the 300-user system runs. Among structurally valid end-to-end runs on the same 300 users, select the checkpoint with the highest macro NDCG@10; break an exact tie by the earlier training step. Report Recall@10 and HitRate@10 as secondary metrics, and the paired P2-minus-P1 NDCG@10 confidence interval separately. The interval is evidence about the chosen system, not a checkpoint-selection threshold. Use the existing 10,000 paired-user percentile resamples, seed `20261002`, and 95% confidence for the P1 comparison.

Member 1's November 23 clarification resolves the earlier conflict with its [decision record](https://github.com/dan778912/movie-rec-agent/blob/member-1-samples-p2/docs/member1_p2_team_decisions.md): the dev set screens candidates, while the saved 300 validation users select the final checkpoint. The team still needs to record the frozen rule and the shared simulator, prompt, and online action policy before inspecting P2 system-validation outcomes.

## Current input inventory

| Required input | Status on this workspace and GitHub | Owner or source |
|---|---|---|
| Canonical `ml25m-global-v1` catalog, validation profiles, targets, and manifest | Present locally under `data/processed/ml25m-global-v1/`; Git-ignored. Rebuild and verify with `python -m src.data.prepare` on another machine. | Member 1 data pipeline |
| Frozen 300 IDs and P2 validation scoring config | Tracked on `member2-phase2`; config status is `draft_for_phase2_team_review`. | Member 2 |
| 400-user P2 dev samples and episodes | Published in Member 1's v2 ZIP on `member-1-samples-p2`; ZIP and all internal hashes independently verified. | Member 1 to Member 6 |
| Exact ordered B3 50-item prediction JSONL for the 300 users | Received from Member 6, stored locally under ignored `outputs/predictions/phase2_validation/`, hash `c388c0fd10bb895c1e0710ec10fa96a7d87785939f2498a5bf23533d18d000a9`; scored 300/300 with zero errors. Raw file is not in Git. | Member 6 / team storage |
| Original P1 50-item validation prediction JSONL | Absent locally and from Git. Original SHA-256: `b70ccd8ea374695c3bed24cd908caee65bffdfe96de7e30b693de990091c3ad2`. A saved score is not a prediction file. | Member 6 / team storage |
| Untuned Qwen control prediction JSONL | Corrected export received from Member 6 and stored locally under ignored `outputs/predictions/phase2_validation/`; hash `397d21f290bd688fb5bf3830518bf5f59be9654793ccac0809b471b30798087f`. Scored 300/300 with zero errors; [results archived](results/member2_qwen_base_validation_300_v1/README.md). Raw file is not in Git. | Member 6 |
| P2 validation predictions, `p2-trace-v1` traces, and `p2-run-manifest-v1` | No real export yet. Each must cover all 300 saved users, name the checkpoint, prompt, simulator, index, code, and runtime policy, and be delivered as immutable files. | Members 3, 4, and 6 |

Local input verification on 2026-11-18 passed `src.data.verify` for all ten
canonical processed-file hashes. An isolated regeneration from Member 1 commit
`321c284` produced 8,974 of 22,860 expected training samples in about twelve
minutes on this Windows/Python 3.11 host. That run was stopped before completing
the train or dev split; its partial file is marked incomplete outside the repo,
and the standard `data/processed/p2-v1/` path remains absent. This is a runtime
observation, not a data-quality failure or a verified P2 sample handoff.

For the historical 500-user comparison, the original B3 and P1 test JSONLs are also outside Git; the [test receipt](results/member2_test_500_v1/README.md) records their hashes. Do not substitute an archived score or a regenerated pool without creating a separately identified run.

## Validation and acceptance

Run the following from the repository root with actual immutable handoff paths and a new output directory. The complete command sequence is in the [runbook](phase2_member2_runbook.md).

```powershell
python -m src.evaluation.evaluate --config configs/evaluation_phase2_validation_draft_v1.json --predictions <p2.jsonl> --method p2 --candidate-pool <b3.jsonl> --trace <p2_trace.jsonl> --output-dir <new-p2-report-dir>
python -m src.evaluation.p2_run_manifest --manifest <run_manifest.json> --config configs/evaluation_phase2_validation_draft_v1.json --candidate-pool <b3.jsonl> --predictions <p2.jsonl> --trace <p2_trace.jsonl>
```

The implemented **export gate** is a structurally valid prediction and trace for every selected user: exactly 50 distinct catalog/unwatched final movie IDs per user, an exact ordered B3 top-50 start, only fresh valid additions, no missing/extra/duplicate users, at most eight control events plus exactly one ranking and one top-ten explanation event, valid event order, and a declared B3-order fallback when needed. An invalid raw model action can be handled by the frozen fallback policy; record it in per-user telemetry. A valid final export does not establish a zero raw invalid-action rate or prove that model decisions were autonomous.

The run manifest must report checkpoint and index identities, action/error/fallback counts, model and retrieval calls, Service B processing latency p50/p95, hardware and concurrency, and approximate cost. **No numeric raw-action validity, latency, or quality threshold has been ratified.** The team must freeze any such thresholds, plus online timeout and question/retrieval limits, on development data before the final test. Member 1's training design uses at most two questions, one wider retrieval, and 50→at most 100 candidates; Member 2's trace validator enforces the broader eight-control-event structural cap. Members 1, 3, 4, and 6 must reconcile the operational policy with the trace before a production export is called final.

## Handoff requests

| Member | Concrete request |
|---|---|
| 1 | Hand the verified v2 ZIP to Member 6; provide any additional source-generation evidence needed for final data-quality review. |
| 5 | Retrieve the original 500-user B3 test prediction JSONL from team storage if controlled test reruns are planned. The 300-user B3 pool has arrived. |
| 6 | Retrieve the original 300-user P1 prediction JSONL (SHA-256 `b70ccd8ea374695c3bed24cd908caee65bffdfe96de7e30b693de990091c3ad2`) and the 500-user P1 test JSONL; provide the P2 checkpoint candidates, training step/revision and loading evidence, generation settings, strict parse rates, and three-checkpoint dev-loss shortlist. The corrected untuned export has arrived. |
| 3 and 4 | Freeze the real Service B/index and agent/simulator/prompt/action policy, then export one P2 prediction, trace, and run-manifest record per validation user with measured calls, failures, and latency. |
| 1, 2, 3, 4, and 6 | Review and record the checkpoint-selection rule, operational caps, fallback/timeout settings, and any numeric acceptance thresholds before final-test inference. |
