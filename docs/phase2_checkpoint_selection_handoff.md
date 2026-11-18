# Phase 2 checkpoint selection and validation handoff — decision draft

**Status (2026-11-18): proposed for team review, not a frozen selection rule or a completed P2 run.** This handoff separates Member 1's training-period development samples from Member 2's saved system-validation users. It records the files and decisions needed before Member 6 can name a selected P2 checkpoint. The [Phase 2 protocol](phase2_evaluation_protocol.md), [interface contract](phase2_interface_contract.md), and [runbook](phase2_member2_runbook.md) remain the detailed sources.

## Two populations and a proposed selection rule

| Population | Current source | Intended use |
|---|---|---|
| 400 user-disjoint training-period `dev` episodes | Member 1's [`p2_samples_dev.jsonl` and `p2_episodes_dev.jsonl`](https://github.com/dan778912/movie-rec-agent/blob/member-1-samples-p2/docs/member1_p2_report.md), generated locally and Git-ignored | Screen training checkpoints for loadability, strict action parsing, and a predeclared training metric such as completion loss. These labels and episode NDCG values come from a proxy ranker and are not end-to-end P2 evaluation results. |
| Frozen 300 validation users | `configs/validation_users_300_v1.json` and `configs/evaluation_phase2_validation_draft_v1.json` | Run shortlisted checkpoint(s) end to end on identical profiles, ordered B3 pools, simulator, index, prompt, decoding, fallback policy, and selected users. The proposed primary system metric is macro NDCG@10 on all 300 final rankings, including fallback episodes. |
| Frozen 500 test users | `configs/test_users_500_v1.json` | One final evaluation after model and policy selection; never choose a checkpoint or threshold using these targets. |

**Proposed resolution for team review:** Member 6 predeclares how the 400-user dev set screens checkpoints and freezes a short list before the 300-user system runs. Among structurally valid end-to-end runs on the same 300 users, select the checkpoint with the highest macro NDCG@10; break an exact tie by the earlier training step. Report Recall@10 and HitRate@10 as secondary metrics, and the paired P2-minus-P1 NDCG@10 confidence interval separately. The interval is evidence about the chosen system, not a checkpoint-selection threshold. Use the existing 10,000 paired-user percentile resamples, seed `20261002`, and 95% confidence for the P1 comparison.

This rule is **not yet approved**. Member 1's [decision record](https://github.com/dan778912/movie-rec-agent/blob/member-1-samples-p2/docs/member1_p2_team_decisions.md) says its 400-user dev split selects checkpoints; the Member 2 draft says the saved 300 validation users are used for model decisions. Members 1, 2, and 6 must agree whether the 300-user run may select the final deployed checkpoint, the stage-one screening metric and shortlist size, and the tie rule **before** reading the system-validation outcomes. Member 4 also needs the simulator, prompt, and online action policy to be fixed for a fair run.

## Current input inventory

| Required input | Status on this workspace and GitHub | Owner or source |
|---|---|---|
| Canonical `ml25m-global-v1` catalog, validation profiles, targets, and manifest | Present locally under `data/processed/ml25m-global-v1/`; Git-ignored. Rebuild and verify with `python -m src.data.prepare` on another machine. | Member 1 data pipeline |
| Frozen 300 IDs and P2 validation scoring config | Tracked on `member2-phase2`; config status is `draft_for_phase2_team_review`. | Member 2 |
| 400-user P2 dev samples and episodes | Generator, config, examples, and quality report are on `member-1-samples-p2`; complete generated JSONLs are Git-ignored and absent locally. Regenerate or transfer the hash-verified outputs. | Member 1 to Member 6 |
| Exact ordered B3 50-item prediction JSONL for the 300 users | Absent locally and from Git. Original prepared-file SHA-256: `c388c0fd10bb895c1e0710ec10fa96a7d87785939f2498a5bf23533d18d000a9`. The archive records the score but cannot reconstruct each ordered pool. Re-encoding on a new machine may produce a different pool and would require a separately versioned matched P1 control. | Member 5's original B3 export / team storage |
| Original P1 50-item validation prediction JSONL | Absent locally and from Git. Original SHA-256: `b70ccd8ea374695c3bed24cd908caee65bffdfe96de7e30b693de990091c3ad2`. A saved score is not a prediction file. | Member 6 / team storage |
| Untuned Qwen control prediction JSONL | Absent locally and from Git. The evaluator now accepts the distinct `qwen_base` method, requires the exact B3 candidate set, and can compute a paired P1-minus-base interval once a real 300-user export exists. Member 6 must explicitly export `method: "qwen_base"` and pin the base-model revision. | Member 6 |
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
| 1 | Provide or regenerate the hash-verified P2 train/dev sample and episode JSONLs and quality report; confirm the simulator version and the 400-dev role in checkpoint screening. |
| 5 | Retrieve the original 300-user ordered B3 prediction JSONL with the SHA-256 above, plus the original 500-user B3 test JSONL from team storage. |
| 6 | Retrieve the original 300-user P1 prediction JSONL and the 500-user P1 test JSONL; export the 300-user untuned control with `method: "qwen_base"` and a pinned base-model revision; provide the P2 checkpoint candidates, training step/revision and loading evidence, generation settings, and the agreed stage-one dev metric/shortlist. |
| 3 and 4 | Freeze the real Service B/index and agent/simulator/prompt/action policy, then export one P2 prediction, trace, and run-manifest record per validation user with measured calls, failures, and latency. |
| 1, 2, 3, 4, and 6 | Review and record the checkpoint-selection rule, operational caps, fallback/timeout settings, and any numeric acceptance thresholds before final-test inference. |
