# Member 2 Phase 2 results status

**Updated 2026-11-24: P2 comparison pending.** The [Phase 2 protocol](phase2_evaluation_protocol.md),
[interface contract](phase2_interface_contract.md), [evaluator runbook](phase2_member2_runbook.md),
and [failure-analysis plan](phase2_failure_analysis_plan.md) are prepared for team review.
No P2 prediction, execution trace, checkpoint, or real Service B result has been handed off
in the inspected branches. This page records the evidence available now; it is not the
final B3/P1/P2 result table.

## Completed 300-user validation control

The corrected untuned `qwen_base` prediction export and exact ordered B3
validation pool were received from Member 6, verified against their posted
SHA-256 hashes, and scored under one `eval-p2-draft-v1` configuration and clean
evaluator revision. Both files passed 300/300 structural validation with zero
errors. The [portable scored bundles and commands](results/member2_qwen_base_validation_300_v1/README.md)
record complete provenance.

| Metric | B3 | Untuned Qwen | Qwen minus B3 | Paired 95% interval |
|---|---:|---:|---:|---|
| NDCG@10 | 0.014578 | 0.019772 | +0.005194 | [-0.000675, 0.011195] |
| Recall@10 | 0.005026 | 0.007733 | +0.002707 | [-0.000792, 0.006608] |
| HitRate@10 | 0.110000 | 0.150000 | +0.040000 | [0.003333, 0.076667] |

The original 300-user P1 validation prediction JSONL is still needed to
rescore P1 under this same evaluator and publish the authoritative
P1-versus-untuned paired interval. The historical P1 bundle uses an earlier
evaluation version and cannot be passed directly to the strict comparator.

Member 1's v2 P2 sample ZIP has also arrived and passed its ZIP and internal
checksums. The [checkpoint and validation handoff](phase2_checkpoint_selection_handoff.md)
records its counts, dev/validation roles, and current producer requests. No
selected P2 checkpoint or real P2 predictions and traces have arrived yet.

The following are **training/dev action-label counts**, parsed from the
complete v2 sample JSONLs; each saved completion action matched its target.
They are not observed action frequencies from a running P2 agent.

| Action | Train samples | Dev samples |
|---|---:|---:|
| Assess information sufficiency | 6,675 | 628 |
| Ask clarification | 2,701 | 228 |
| Evaluate candidates | 4,726 | 461 |
| Widen / retrieve candidates | 752 | 61 |
| Rank candidates | 3,974 | 400 |
| Explain recommendations | 3,974 | 400 |
| **Total** | **22,802** | **2,178** |

One train/dev episode yields one Rank label; Ask can appear zero, one, or two
times. Actual validation action counts, invalid-action rate, and mean control
events per episode need Member 3's real P2 execution traces.

## Canonical data verified locally

On November 6, 2026, `python -m src.data.prepare` downloaded the official
MovieLens 25M archive, verified its publisher MD5, and built the processed
data. A Windows LF serialization fix in `src/data/build.py` was then applied;
`python -m src.data.build --raw data/raw/ml-25m` and
`python -m src.data.verify` reproduced the frozen manifest SHA-256
`f2e137ff374a8cfc02348cb42d04d9a04c9a1769685a7eccc21933383a500b5a`.
The catalog, test profiles, and test targets also match their archived hashes,
all ten processed-file checksums pass, and the saved 500-user test context
validates. The large raw and processed files remain Git-ignored local inputs.

## Historical test evidence

The [authorized October 2026 P1/B3 archive](results/member2_p1_b3_test_500_v1/README.md)
contains the following 500-user results. These are from the earlier `eval-test-v1`
configuration and scorer. They are useful context, but their saved scored bundles
cannot be combined directly with a new P2 scored bundle because the evaluator
implementation and shared configuration must match.

| Metric | B3 | P1 | P1 minus B3 | Paired 95% interval | P2 |
|---|---:|---:|---:|---|---|
| NDCG@10 | 0.012429 | 0.036421 | +0.023992 | [0.016787, 0.031517] | Pending |
| Recall@10 | 0.006732 | 0.013464 | +0.006733 | [0.000571, 0.012117] | Pending |
| HitRate@10 | 0.094000 | 0.184000 | +0.090000 | [0.060000, 0.118050] | Pending |
| Recall@20 | 0.010011 | 0.016612 | +0.006601 | [0.000043, 0.013029] | Pending |
| Recall@50 | 0.022180 | 0.022180 | +0.000000 | [0.000000, 0.000000] | Pending |

The archive reports that 380 of 500 users had no relevant held-out movie in
the initial B3 top-50 pool. That is the historical retrieval limitation P2's
re-retrieval may address. It is not evidence that P2 actually improves it.

## Additional baseline inputs prepared

The frozen popularity (B0) and BM25 (B2) producers were run on the same saved
500 test users, without reading test labels. Both top-100 prediction files
passed the authoritative structural validator with 500/500 users and zero
errors. Their hashes, producer commits, and local file paths are in the
[B0/B2 prediction receipt](phase2_baseline_prediction_receipt.md). They remain
**unscored** until the team freezes the shared Phase 2 test configuration;
their top 50 can then be prepared and evaluated under the same revision as
B3/P1/P2. This completes the label-blind B0/B2 input generation step but not
the joint results table.

## Inputs required for the final comparison

1. Obtain the original frozen 500-user B3 and P1 prediction JSONL files from
   team storage and verify SHA-256 `44ec13348849458fa58746e126369afc5fa0c2cf4bbd867bc732b83f412b93a1`
   and `15d5d332f61224824f339bdd73275522dec71a93ae05d0a460f9bb7fb1412634`, respectively.
2. Build the canonical processed dataset and catalog locally with
   `python -m src.data.prepare`, then verify its manifest against the frozen
   source hashes. Obtain the 500-user P2 prediction JSONL, complete
   `p2-trace-v1` JSONL, run manifest, selected
   checkpoint and index identities, simulator provenance, and latency/error logs.
3. Confirm the P1 control used the same canonical initial profile, catalog, and
   exact B3 pool as P2. If it did not, run and version a new matched P1 control.
4. Approve and freeze a Phase 2 test configuration and experiment settings
   **before** P2 test inference. Re-score the original B3/P1 predictions and
   P2 predictions with one evaluator revision and that same configuration;
   then run the paired comparison, diagnostics, table, and failure audit using
   the [runbook](phase2_member2_runbook.md).

The final report should replace `Pending` with measured values and a P2-minus-P1
paired interval, identify the complete input hashes and fallback count, and
preserve the historical P1/B3 archive unchanged. The [acceptance checklist](phase2_acceptance_checklist.md)
and [demo plan](phase2_demo_plan.md) track the separate live-system evidence.
