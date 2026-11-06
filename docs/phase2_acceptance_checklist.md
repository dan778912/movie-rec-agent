# Phase 2 acceptance checklist — Member 2

**Status:** working checklist; record evidence beside each item before marking it complete. The team-provided Phase 2 responsibility split supersedes the old role assignments in the [Phase 1 roadmap](../Movie_Recommendation_Agent_Roadmap.md). A passing Phase 1 test or a working mock UI is not evidence that P2 passes.

Use the [Phase 2 evaluation protocol](phase2_evaluation_protocol.md), [interface contract](phase2_interface_contract.md), [failure-analysis plan](phase2_failure_analysis_plan.md), and [demo plan](phase2_demo_plan.md) together. Freeze the agreed versions and any numeric thresholds on validation data before P2 final-test scoring. Mark a failed or unavailable check honestly; completion does **not** require P2 to outperform P1.

## Evidence available now

| Evidence | Current status and limit |
|---|---|
| [P1/B3 500-user test archive](results/member2_p1_b3_test_500_v1/README.md) | Scored and published after the recorded team authorization: archived scored bundles, paired intervals, source hashes, and 100% prediction-user coverage. This is a P1/B3 result only; the original JSONL files are managed separately. |
| [Original prediction receipt](results/member2_test_500_v1/README.md) | Records source hashes, but the frozen B3 and P1 prediction JSONL files are in team storage, not Git. P2 trace validation needs the original ordered B3 JSONL. The updated evaluator also needs both files to re-score B3/P1 for a compatible P2 comparison. |
| [Saved test users](../configs/test_users_500_v1.json) and [P1/B3 frozen inference settings](../configs/evaluation_test_frozen_v1.json) | The population is reusable. The old freeze explicitly covers P1/B3; it does not by itself freeze P2's simulation, action policy, expanded pool, or joint report. |
| [Phase 2 evaluator](../src/evaluation/README.md) | The P2 trace validator, scorer, paired comparison, table support, and separate pool/action diagnostics are implemented and covered by synthetic tests. They still need a team-reviewed contract and real P2 export before the final run. |
| [`origin/member5-service-a` app branch](https://github.com/dan778912/movie-rec-agent/tree/member5-service-a/src/app) | PostgreSQL schema, Service A, and React UI exist on that branch. Service A defaults to a deterministic `MockServiceBClient`; its HTTP `/decide` shape and stored action payload are provisional. `docker-compose.yml` runs development PostgreSQL only. |

## What can be checked now

| Work | When it can be verified | Gates |
|---|---|---|
| Draft/review the P2 protocol and interface; inspect frozen population, P1/B3 archive, and provisional Service A shape | Now, from tracked files and team review | A1–A2, B1 design |
| Run P2 trace/prediction validation, scoring, diagnostics, and adversarial synthetic fixtures | Implemented and tested locally; actual P2 outputs and the original frozen B3/P1 JSONL are needed for final-run validation and comparison | B2–B4 implementation |
| Audit P2 sample leakage and seed catalog | After Member 1 supplies the versioned files/report | A3 |
| Validate real P2 decisions, re-retrieval, checkpoint loading, and stage latency | After Members 3, 4, and 6 supply Service B, index, agent, checkpoint, and traces | A4, C1–C5, C8 |
| Run final B0/B2/B3/P1/P2 test table, intervals, and failure analysis | After the frozen P2 export and matching baseline reports are available; preserve the already published P1/B3 record | B3–B5, D1–D2 |
| Execute independent UI demo, persistence, and deployment checks | After Member 5's branch is integrated with real Service B and full trace/version support | C6–C7, D3–D4 |

## Gate A — frozen experiment and source identity

- [ ] **A1 — Protocol approved.** Record the agreed protocol revision, target data/version hashes, the existing 500 test user IDs, ranking length, P2 starting B3 pool, permitted expansion, clarification-answer simulation, missing/invalid prediction policy, metric definitions, bootstrap settings, and predeclared decisions/thresholds. Owner: Member 2 with Members 1, 3, 4, and 6. Evidence: reviewed protocol and configuration commit.
- [ ] **A2 — No test-guided tuning.** Separate validation decisions and P2 checkpoint selection from the previously viewed P1/B3 test results. Record the chronology and any unavoidable exposure; do not change model/prompt/retrieval settings in response to final-test labels. Owner: Members 2, 3, 4, and 6. Evidence: dated freeze and run manifest.
- [ ] **A3 — Data handoff checked.** Member 1's six P2 sample types, catalog seed, data-quality report, and temporal boundaries have counts, provenance, and checks for future-rating/target leakage. Index contents and user profiles follow the approved time boundary. Owner: Member 1; verified by Member 2/3. Evidence: versioned data manifest and check report.
- [ ] **A4 — Model/runtime identity fixed.** Record P1 and P2 checkpoint IDs/hashes, training configuration, prompt/action schema versions, random seeds, decoding parameters, hardware, and index version. Both checkpoints load from saved artifacts. Owner: Member 6 with Member 3. Evidence: loading run and manifest.
- [ ] **A5 — Original B3/P1 predictions available.** Obtain both 500-user prediction JSONL files from team storage and verify their bytes against the [recorded receipt](results/member2_test_500_v1/README.md). P2 inference and trace validation need the ordered B3 pool. The current paired-comparison/table tools require matching evaluator source hashes and settings, so re-score B3/P1 with the updated evaluator and shared frozen configuration before joining them to P2. Preserve the historical scored reports unchanged. A new matched P1 inference also requires its checkpoint and run inputs. Owner: Member 2 with artifact owners. Evidence: checksum verification, new scored bundles, and source paths in the private run manifest.

## Gate B — evaluator and trace integrity

- [ ] **B1 — Contract implemented.** Each offline user has the exact ordered frozen B3 starting pool, ordered `assess_sufficiency`, `evaluate_candidates`, and `rank_candidates` events as applicable, a clarification question and simulated answer when used, each retrieval request and returned pool, an explanation event covering the final top ten, a final 50-item ranking, and failure/fallback reasons. The export bundle records run timing and source/model versions in its manifest; retrieval events identify their index version. Owner: Members 2–4 and 6. Evidence: contract example, producer export, and parser test.
- [ ] **B2 — Strict P2 validation.** The evaluator rejects missing/extra/duplicate users, invalid ranks/scores, duplicate/unknown/watched final movies, impossible pool provenance, absent top-ten explanation, more than eight **control** events (plus one rank and one explanation event), or broken action order. `ask_clarification` follows an insufficient `assess_sufficiency`; `retrieve_candidates` follows `evaluate_candidates` with `pool_sufficient: false`; final IDs come from the initial or expanded pool. P2's final 50 may be a subset of a larger pool. Owner: Member 2. Evidence: positive and deliberately invalid synthetic tests plus validation reports.
- [ ] **B3 — Complete scoring and diagnostics.** A declared run scores every saved user under one frozen failure policy. The core scorer reports final Recall@10/NDCG@10 and Recall@50 with separately named catalog-positive and prediction-user coverage. `src.evaluation.p2_diagnostics` reports initial B3 pool recall, expanded-pool recall, added-candidate yield, action/fallback rates, and pool sizes in separate immutable `summary.json` and `per_user.csv` files. Both retain failed/fallback users in the denominator as specified. Owner: Member 2. Evidence: scorer/diagnostics tests and both report bundles.
- [ ] **B4 — Paired uncertainty.** B3, P1, and P2 are compared on exactly the same saved users and relevant targets. Re-score the original P1/B3 predictions under the updated evaluator and shared config; the archived P1/B3 score bundles have earlier evaluator hashes and cannot be directly paired with new P2 scores. Use the original P1 control only if P2 receives the same initial profile, catalog, and B3 pool; otherwise produce a separately versioned matched P1 inference before making the controlled claim. P2-minus-P1 differences use a paired user bootstrap with the frozen resampling settings; report point differences and 95% intervals, including intervals that cross zero. Owner: Member 2. Evidence: comparison JSON, per-user differences, and input hashes.
- [ ] **B5 — Joint table and provenance.** Generate the B0/B2/B3/P1/P2 table from compatible validated scored bundles for the same final-test population and evaluator/config revision. List absent runs as missing, never fill cells from the separate 300-user validation experiment or invent values. Preserve the original published P1/B3 archive and label the new compatible re-scoring separately. Record code commits, original prediction hashes, data manifest, configurations, and commands. Owner: Member 2 with baseline owners. Evidence: generated table and portable archive.

## Gate C — agent, retrieval, and service behavior

- [ ] **C1 — Model-driven controls.** A trace distinguishes P2's model-issued `assess_sufficiency` and `evaluate_candidates` decisions from application guards and fallbacks. Demonstrate both a clarification and a direct recommendation; for expansion, show `pool_sufficient: false` followed by a broader retrieval request and `rank_candidates` over the resulting pool. A fixed UI rating minimum or a hard-coded ask rule does not satisfy this gate. Owner: Members 4 and 6. Evidence: trace examples and action tests.
- [ ] **C2 — Bounded loop and safe failures.** The action parser rejects malformed/unknown actions; the agent enforces the frozen question/retrieval/total-step caps and deterministic terminal behavior on timeout, empty pool, repeated action, or model/service failure. Record every fallback and reason. Owner: Members 3 and 4. Evidence: cap/error tests and traces.
- [ ] **C3 — Online retrieval.** The vector index is built from the approved catalog, has a version and reproducible ingest instructions, expands the candidate pool when requested, excludes the complete watched set at every retrieval and final ranking, and reports candidate recall plus retrieval latency. Owner: Member 3. Evidence: index manifest, retrieval tests, and latency report.
- [ ] **C4 — Grounded justifications.** Each displayed explanation is auditable against supplied movie metadata and known user preferences/clarification answers. Audit a declared sample for unsupported claims; report the sample rule, counts, failures, and examples. Ranking scores alone do not pass this gate. Owner: Members 4 and 2. Evidence: grounding checks and manual audit record.
- [ ] **C5 — Real Service B.** A separately runnable FastAPI Service B loads Member 6's P2 checkpoint, invokes Member 4's agent and Member 3's versioned retrieval, and satisfies the reviewed Service A request/response contract. Its health/version evidence names model, index, and code versions. Owner: Member 3 with Members 4 and 6. Evidence: integration test and deployment logs.
- [ ] **C6 — Real Service A integration.** Service A uses `SERVICE_B_MODE=http`, handles clarification turns and errors through the agreed contract, persists full relevant agent traces or durable trace references with recommendations in PostgreSQL, and exposes enough evidence to inspect a demo run. Its profile uses the shared formatter; no fixed rating-count rule decides P2's action. Owner: Member 5 with Member 3. Evidence: HTTP integration tests and database/trace inspection. Member 5's current branch only stores `ask`/`recommend` summaries and its mock is not this evidence.
- [ ] **C7 — Frontend and end-to-end behavior.** Through the React UI, a new user can rate a few catalog movies, answer a generated question when asked, receive ten distinct unseen recommendations with grounded explanations and a truthful model/version label, and start another profile without restarting services. Owner: Members 5 and 2. Evidence: independent tester's run record and screenshots/trace IDs.
- [ ] **C8 — Measured latency and cost.** Report service-level and end-to-end latency distributions, retrieval/model time, hardware, timeout/failure counts, and approximate model/API cost using an identified measurement workload. Do not treat evaluator runtime as inference latency. Owner: Members 2 and 3. Evidence: measurement script/log and summary.

## Gate D — final analysis, deployment, and handoff

- [ ] **D1 — Failure analysis.** Use the [failure-analysis plan](phase2_failure_analysis_plan.md) and P2 diagnostics bundle to report retrieval misses, clarification decision frequency, re-retrieval frequency, invalid actions, capped loops, fallback rates, watched/grounding violations, and representative successes/failures. State the limits of simulated clarification and the P1 control. Owner: Member 2 with owners of traces. Evidence: report, diagnostics hashes, and sampled trace IDs.
- [ ] **D2 — Claims bounded by evidence.** Explain whether P2 improved or worsened P1, with paired intervals and the observed control condition. Do not claim real-user satisfaction from simulated answers, causal value of each P2 capability without an ablation, or deployed P2 behavior from mock output. Owner: Member 2. Evidence: final report, presentation, and reviewed claims.
- [ ] **D3 — Reproducible application.** Pin dependencies, data/index/checkpoint versions, migration and seed steps, Service A/B startup, configuration, health checks, and redeployment procedure. Docker Compose, CI/CD, Terraform/AWS, and an instructor-accessible deployment meet the roadmap or have a documented instructor-approved scope. Owner: Members 3 and 5 with Member 2 verification. Evidence: independent setup/redeploy log and accessible URL.
- [ ] **D4 — Independent acceptance and demo.** A teammate who did not implement the UI executes the [demo plan](phase2_demo_plan.md), including ambiguous and clear profiles and a second session, records actual outputs/traces/latency, and can identify the real P2 checkpoint. Owner: Member 2. Evidence: completed demo run sheet; known deviations remain visible.

## Final decision record

Fill this only after the checks above have actual evidence.

| Field | Value |
|---|---|
| Protocol/config revision and reviewer decisions | Pending |
| Final-test run IDs and source hashes | Pending |
| P2-minus-P1 primary metric and paired interval | Pending |
| Prediction/trace coverage and fallback count | Pending |
| Grounding audit and latency report | Pending |
| Application URL, commit, model/index versions | Pending |
| Independent tester, date, trace IDs, outcome | Pending |
| Unmet checks, limitations, or approved exceptions | Pending |
