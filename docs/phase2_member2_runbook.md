# Member 2 Phase 2 runbook

**Status:** validation workflow ready for synthetic checks; real P2 inputs and the team-reviewed final-test freeze are pending. Read the [experiment protocol](phase2_evaluation_protocol.md), [interface contract](phase2_interface_contract.md), and [artifact reproduction guide](phase2_artifact_reproduction.md) before accepting a producer export.

## Artifacts to obtain

1. Reproduce the canonical `ml25m-global-v1` processed data with `python -m src.data.prepare` from the repository root. This downloads and verifies the official MovieLens 25M archive, builds the catalog/profiles/targets, and runs `src.data.verify`. The large files are Git-ignored. Check the resulting manifest and frozen hashes before scoring. Member 1 must separately supply the **Phase 2** sample/data-quality report and live catalog seed.
2. The original 500-user B3 prediction JSONL and P1 prediction JSONL from team storage. Their historical SHA-256 digests are respectively `44ec13348849458fa58746e126369afc5fa0c2cf4bbd867bc732b83f412b93a1` and `15d5d332f61224824f339bdd73275522dec71a93ae05d0a460f9bb7fb1412634`. The archived scored reports cannot replace these files for P2 pool validation or same-revision rescoring.
3. P2's prediction JSONL, `p2-trace-v1` JSONL, run manifest, checkpoint/index identifiers, simulator source/version, and latency/error records from Members 3, 4, and 6. A producer's file name is not evidence of its contents; verify its hash and validate every user.

The saved 500-user popularity (B0) and BM25 (B2) prediction inputs have now
been generated without reading test labels. See their
[receipt](phase2_baseline_prediction_receipt.md) for hashes and producer
commits. They are top-100 source exports; prepare their top 50 only after the
shared Phase 2 scoring configuration is frozen.

Keep original inputs immutable. Store large local inputs under Git-ignored `data/` or `outputs/predictions/`, and use a new versioned directory for each derived report. Do not edit published Phase 1 archives or frozen configs.

## Validate on the saved development users

These commands illustrate the 300-user validation path. Replace the three `outputs/predictions/phase2_validation/` file names with the actual immutable handoff paths. Run from the repository root. The draft scoring config preserves the saved user/data identities; it does not freeze P2 inference settings.

```powershell
python -m src.evaluation.validate `
  --catalog data/processed/ml25m-global-v1/catalog.csv `
  --profiles data/processed/ml25m-global-v1/validation_profiles.jsonl `
  --users configs/validation_users_300_v1.json `
  --predictions outputs/predictions/phase2_validation/p2.jsonl `
  --candidate-pool outputs/predictions/phase2_validation/b3.jsonl `
  --trace outputs/predictions/phase2_validation/p2_trace.jsonl `
  --split validation --method p2 --required-count 50
```

For `p2`, the validator requires the complete original B3 file and one trace per frozen user. It checks ordered initial-pool identity, valid expansion, final-ranking membership, action sequence/cap, fallback policy, and top-ten explanation binding. It cannot establish temporal safety of simulator answers or semantic explanation grounding; audit those separately. A structurally invalid run is not scored.

Score B3, P1, and P2 under **one** evaluator revision and **one** shared scoring config. The P1/B3 files must be the originals, not the older scored reports:

```powershell
python -m src.evaluation.evaluate --config configs/evaluation_phase2_validation_draft_v1.json --predictions outputs/predictions/phase2_validation/b3.jsonl --method embedding --output-dir outputs/reports/phase2_validation/b3
python -m src.evaluation.evaluate --config configs/evaluation_phase2_validation_draft_v1.json --predictions outputs/predictions/phase2_validation/p1.jsonl --method p1 --candidate-pool outputs/predictions/phase2_validation/b3.jsonl --output-dir outputs/reports/phase2_validation/p1
python -m src.evaluation.evaluate --config configs/evaluation_phase2_validation_draft_v1.json --predictions outputs/predictions/phase2_validation/p2.jsonl --method p2 --candidate-pool outputs/predictions/phase2_validation/b3.jsonl --trace outputs/predictions/phase2_validation/p2_trace.jsonl --output-dir outputs/reports/phase2_validation/p2
python -m src.evaluation.bootstrap --baseline-dir outputs/reports/phase2_validation/p1 --challenger-dir outputs/reports/phase2_validation/p2 --output-dir outputs/reports/phase2_validation/p1_vs_p2
python -m src.evaluation.report --runs outputs/reports/phase2_validation/b3 outputs/reports/phase2_validation/p1 outputs/reports/phase2_validation/p2 --comparisons outputs/reports/phase2_validation/p1_vs_p2 --kind experiment --output-dir outputs/reports/phase2_validation/table
python -m src.evaluation.p2_diagnostics --config configs/evaluation_phase2_validation_draft_v1.json --candidate-pool outputs/predictions/phase2_validation/b3.jsonl --predictions outputs/predictions/phase2_validation/p2.jsonl --trace outputs/predictions/phase2_validation/p2_trace.jsonl --output-dir outputs/reports/phase2_validation/p2_diagnostics

python -m src.evaluation.p2_run_manifest --manifest outputs/predictions/phase2_validation/run_manifest.json --config configs/evaluation_phase2_validation_draft_v1.json --candidate-pool outputs/predictions/phase2_validation/b3.jsonl --predictions outputs/predictions/phase2_validation/p2.jsonl --trace outputs/predictions/phase2_validation/p2_trace.jsonl

python -m src.evaluation.phase2_failure_analysis --p1-dir outputs/reports/phase2_validation/p1 --p2-dir outputs/reports/phase2_validation/p2 --diagnostics-dir outputs/reports/phase2_validation/p2_diagnostics --analysis-spec configs/phase2_failure_analysis_spec_draft_v1.json --output-dir outputs/reports/phase2_validation/failure_analysis

python -m src.evaluation.p2_grounding_audit --catalog data/processed/ml25m-global-v1/catalog.csv --movie-documents data/processed/ml25m-global-v1/movie_documents.jsonl --profiles data/processed/ml25m-global-v1/validation_profiles.jsonl --users configs/validation_users_300_v1.json --split validation --candidate-pool outputs/predictions/phase2_validation/b3.jsonl --predictions outputs/predictions/phase2_validation/p2.jsonl --trace outputs/predictions/phase2_validation/p2_trace.jsonl --sample-size 30 --seed 20261002 --output-dir outputs/reports/phase2_validation/grounding_audit
```

The table reports submitted ranking metrics; diagnostics separately reports initial and expanded-pool recall, added-candidate yield, action/fallback rates, and pool sizes. The [failure-analysis plan](phase2_failure_analysis_plan.md) specifies how to interpret these and select qualitative cases. The report writer refuses to overwrite a different existing bundle; choose a new versioned output directory for a changed run.

The [run-manifest check](phase2_run_manifest.md) compares the producer's declared policy, source hashes,
per-user telemetry, and output hashes with the validated export; it cannot
prove the checkpoint actually ran or that the policy was frozen before
inference. The failure-analysis command computes paired and subgroup summaries
from compatible saved bundles, using the validation-only draft threshold. The
[grounding audit](phase2_grounding_audit.md) creates a label-blind worksheet;
human reviewers must judge explanation claims against its versioned evidence.
Use a separately approved frozen analysis spec for test; the CLI rejects the
draft status there.

## Before final-test scoring

- Complete the [acceptance checklist](phase2_acceptance_checklist.md) for experiment source identity, the simulator, checkpoint/index/run settings, and the producer's validation evidence. Record team review and the exact source commit and artifact hashes.
- Freeze a new Phase 2 test scoring configuration from the [draft](../configs/evaluation_phase2_test_draft_v1.json) and use it for **all** final B3/P1/P2 rescoring. Its current draft status is not an approved P2 test freeze. Preserve the historical Phase 1 results and source files unchanged.
- Confirm P2 starts from the same canonical profile and exact B3 pool as the P1 control. If the starting input differs, regenerate a separately versioned matched P1 control before making a controlled comparison.
- Do not choose or revise P2 settings from the already published P1/B3 test scores. Once the approved P2 outputs arrive, run each complete method on all 500 saved users, then report the paired interval, coverage, diagnostics, failures, explanation audit, and measured runtime. Missing B0/B2 test inputs remain missing in the joint table; do not copy validation scores into it.

The [demo plan](phase2_demo_plan.md) is the independent end-to-end check after a real Service B is integrated with Service A. A mock result cannot complete it.
