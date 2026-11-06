# Phase 2 demonstration plan — real P2 pipeline

**Status:** rehearsal plan; no live P2 result is recorded here. Run this only after the [acceptance checklist](phase2_acceptance_checklist.md) and [interface contract](phase2_interface_contract.md) have been reviewed. The [Phase 2 protocol](phase2_evaluation_protocol.md) governs offline claims; this demo shows application behavior and cannot replace the controlled B3/P1/P2 comparison.

The roadmap's minimum acceptance test asks an independent teammate to enter ratings, see a clarification when P2 asks, receive ten unseen recommendations with grounded explanations and a fine-tuned model label, and repeat with a different profile. The additional re-retrieval proof below comes from the agent trace, because the current UI has no pool-inspection view.

## 1. Ready-to-run prerequisites

| Check | Required evidence before booking the demo |
|---|---|
| Integrated source | Record the exact commits for Service A/frontend (Member 5), Service B/retrieval (Member 3), agent (Member 4), and selected P2 checkpoint (Member 6). The current `member5-service-a` branch is not yet integrated into this branch. |
| Real model path | Service B loads the selected P2 checkpoint, agent module, and versioned vector index; its health/version record names all three. Service A is configured with `SERVICE_B_MODE=http` and the actual `SERVICE_B_BASE_URL`. An `ask` from `MockServiceBClient` is not P2 evidence. |
| Reviewed HTTP contract | Service B accepts Service A's profile text, complete watched IDs, and prior question/answer pairs; responses and full traces follow the reviewed [interface contract](phase2_interface_contract.md). Service A persists or exports enough trace data to verify internal model decisions, retrieval requests, loop counts, and versions. |
| Catalog and persistence | PostgreSQL migrations are applied; Member 1's approved catalog seed is loaded; movie search works. The live profile formatter and watched history are verified. A new user need not be a MovieLens user. |
| Deployment | Record the frontend and API URLs, deployment commit, setup/redeploy instructions, and who can access them. Member 5's existing `src/app/docker-compose.yml` starts development PostgreSQL only; it is not a full application deployment. |
| Demo fixtures | On validation data or a separate smoke environment, choose and save **actual catalog movie IDs and ratings** for an ambiguous-profile case, a clear-profile case, and a pool-expansion case. Record the clarification answer verbatim. Confirm they exercise the intended branches with the frozen checkpoint/index before the presentation. Do not select cases using held-out test labels. |
| Measurement | Agree on a trace/run ID, clock source, latency measurement method, and location for UI screenshots, HTTP responses, Service B traces, and Service A database records. Do not show simulated or hand-written model outputs as live output. |

Service A currently offers `POST /sessions`, `POST /sessions/{id}/turn`, `POST /sessions/{id}/clarifications`, `GET /sessions/{id}/recommendations`, and `GET /users/{id}/history` on [`origin/member5-service-a`](https://github.com/dan778912/movie-rec-agent/blob/member5-service-a/src/app/service_a/main.py). Its provisional [`HttpServiceBClient`](https://github.com/dan778912/movie-rec-agent/blob/member5-service-a/src/app/service_a/service_b_client.py) posts to `/decide`. The current database trace stores only `ask`/`recommend` summaries, and the current frontend has no P2 version label. The integration must carry and expose the agreed full trace and model/index identity before this plan can certify P2.

## 2. Preflight on the deployed build

1. An independent tester opens the application URL in a fresh browser session and records the displayed P2 model/version label. Confirm it against Service B's health/version record and the deployment manifest, which also identify source commit and index version. If the label is missing, record the failure; do not infer the model from the UI appearance.
2. Confirm Service A and Service B are independently running, PostgreSQL is reachable, and catalog search returns the saved fixture IDs/titles. Capture service health, index version, checkpoint ID, and configuration including the loop caps. Confirm `SERVICE_B_MODE=http` in the running Service A process.
3. Start tracing and timing before the first submission. Record the system clock/timezone, whether the model was cold or warm, hardware, timeout setting, and any retries. A warm-up may be performed separately and disclosed; do not silently discard a slow first request.
4. Keep the frozen fixture card visible to the tester. The presenter does not edit code, files, or database rows during the demo. If behavior diverges from rehearsal, record the actual path and investigate it; do not substitute mock or pre-recorded output.

## 3. Live script

| Step | Tester action | Evidence to capture and acceptance check |
|---|---|---|
| 1. Ambiguous profile | Search the catalog and rate the predeclared small, ambiguous set. Start the chat. | Save rated movie IDs, ratings, session ID, request/response time, and Service B trace ID. P2 may see any number of ratings; the UI has no fixed minimum that forces a question. |
| 2. Clarification | If P2 asks, read the generated question and enter the fixture answer. Continue until the agent terminates. | Capture the question, answer, model-issued `assess_sufficiency` event, action order, and turn count. `ask` must follow an insufficient assessment. The test case is chosen to exercise this branch; if P2 does not ask, mark that planned check unmet rather than claiming it occurred. Confirm Member 4's online loop cap was respected. |
| 3. Recommendations | Inspect the final ten recommendations and explanations. | Check exactly ten distinct catalog IDs, none in the complete rated/watch history; ranks 1–10; no blank explanation. Sample claims against entered preferences, clarification answer, and supplied MovieLens metadata. Save the full response and trace, including any fallback reason. |
| 4. Clear profile | Without restarting services, start another session using the predeclared clear ratings. | Capture a direct `recommend` decision when the chosen case produces one, with no fabricated clarification. Save a separate session/trace ID, ten unseen recommendations, and latency. If a question occurs, record it and mark the direct-path check unmet. |
| 5. Wider retrieval | Run the predeclared pool-expansion case (it may be either session above or a third session). | In the trace, show the live ordered initial B3 pool and index version, `evaluate_candidates` with `pool_sufficient: false`, a wider retrieval request, returned expanded pool, `rank_candidates` and the final recommendations' provenance, and watched exclusion. The top ten must have an explanation event. A different final order alone does not demonstrate re-retrieval. |
| 6. Stored history | Refresh/reopen the UI and read the session/history views. | Check persisted ratings, answered clarification, recommendations, and trace references remain associated with the right user/session. Start a different preference profile without restarting either service. |

The current Service A wire flow for a tester who needs to reproduce a UI problem is `POST /sessions` with `ratings: [{movie_id, title, rating}]`, then `POST /sessions/{id}/turn`; if the response action is `ask`, send `POST /sessions/{id}/clarifications` with `{"answer":"..."}`. Finally read `GET /sessions/{id}/recommendations` and `GET /users/{user_id}/history`. Use actual catalog IDs recorded on the fixture card. The agreed integrated API may add trace/version fields or endpoints; save the exact OpenAPI/schema version used in the run.

## 4. Cross-check against offline evaluation

- Show the generated final-test table and P2-minus-P1 paired comparison from the [protocol](phase2_evaluation_protocol.md): same saved 500 users, complete predictions, initial B3 pool for both, declared clarification simulation, and P2 pool-expansion rules. Recover both frozen B3 and P1 prediction JSONL files from team storage and verify their hashes against the [receipt](results/member2_test_500_v1/README.md). Re-score them under the updated evaluator and common config before comparing with P2; preserve the [historical P1/B3 archive](results/member2_p1_b3_test_500_v1/README.md) unchanged. Include B0/B2 only when their actual matching test runs exist.
- Show the [failure analysis](phase2_failure_analysis_plan.md): clarification and re-retrieval decision frequencies, retrieval misses, invalid/capped actions, fallbacks, explanation grounding, runtime/latency, and representative trace IDs. Explain that simulated answers do not measure real-user satisfaction.
- If P2 is worse than P1 or an interval spans zero, report the observed values and limits. The demonstration still tests the real pipeline; it does not manufacture an improvement claim.

## 5. Completed run sheet (fill from actual run)

| Field | Actual value or evidence link |
|---|---|
| Date/time, timezone, independent tester | Pending |
| Frontend, Service A, Service B URLs and source commits | Pending |
| Checkpoint/prompt/schema/index versions and hashes | Pending |
| Deployment mode; `SERVICE_B_MODE`; hardware | Pending |
| Fixture card with exact movie IDs/ratings and answer | Pending |
| Session/user IDs and trace IDs for ambiguous, clear, expansion cases | Pending |
| Actual action sequences, question count, retrieval count, cap/fallback events | Pending |
| Ten-item, watched-exclusion, and grounding audit result | Pending |
| Request, retrieval, model, and end-to-end latency record | Pending |
| Stored-history/restart-free second profile result | Pending |
| Offline table/comparison/failure report references | Pending |
| Deviations, open issues, and final acceptance decision | Pending |

Keep screenshots and trace exports tied to these IDs. Redact private session data before sharing public artifacts; preserve the unredacted record in agreed team storage if needed for audit.
