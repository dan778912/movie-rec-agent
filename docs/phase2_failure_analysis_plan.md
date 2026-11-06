# Phase 2 failure analysis plan — Member 2

**Status:** prespecified analysis plan; P2 outputs and findings are pending. Do not fill in final-test observations while choosing a checkpoint, prompt, simulator, retrieval rule, or threshold.

## Inputs and unit of analysis

Use the same frozen users, targets, catalog, and cutoffs for B3, P1, and P2. Join the authoritative per-user evaluator outputs to each P2 prediction and execution trace by `(split, user_id)`. Preserve SHA-256 hashes of the original predictions, traces, B3 initial pools, expanded pools or index snapshot, checkpoint, and evaluation configuration. Missing or invalid users are run failures, not observations to drop.

The original 500-user B3 and P1 prediction JSONLs are held in team storage rather than Git. Retrieve them, verify their archived hashes, and re-score them with the same Phase 2 evaluator and shared scoring configuration used for P2. The historical scored reports remain unchanged; their older evaluator source hashes prevent a direct paired comparison with a newly scored P2 bundle.

For development, use the fixed validation population. Run the final test analysis only after the P2 experiment settings and artifacts are frozen. Published P1/B3 test scores must not select or tune P2.

The [draft analysis specification](../configs/phase2_failure_analysis_spec_draft_v1.json)
sets the starting-profile short/long boundary at **1,539 Unicode characters**.
This is the floor of the median length (1,539.5) among the 300 saved validation
profiles and divides those profiles 150/150. It was selected from profile text
only, without validation or test targets or any P2 outputs. The spec remains a
draft until the team records the pre-test freeze; the analysis command rejects
a test run using a draft spec.

## Quantitative breakdown

Report the full paired distribution of P2 minus P1 Recall@10 and NDCG@10: mean, median, better/worse/tie counts, and the prespecified paired-user bootstrap interval. Show absolute B3, P1, and P2 means from the shared evaluator. Keep all users in each denominator.

For each user, record whether any eligible positive appears in the initial B3 top-50 pool, whether a re-retrieval occurred, whether the final expanded pool contains a new eligible positive, whether a positive reaches the P2 submitted final top 50 and top 10, and whether the agent asked a question. Expanded-pool recall uses every movie in the final candidate pool, which may exceed 50; final ranking Recall@50 uses only the submitted 50. These fields separate retrieval opportunity from ranking performance. Report counts and means by the following *predeclared* groups; groups can overlap and their denominators must be explicit:

| Group | Question answered |
|---|---|
| Initial B3 pool contains a positive / contains none | Could P1 have ranked a relevant movie? |
| P2 re-retrieved / did not re-retrieve | How often was expansion used, and with what outcome? |
| Expansion added a positive / added none | Did a larger pool improve candidate coverage? |
| P2 asked / did not ask | How often did clarification occur? |
| Short / long starting profile, using thresholds fixed on validation | Where are sufficiency decisions concentrated? |
| P2 better / tied / worse than P1 on NDCG@10 | Which path explains the per-user change? |

Also report candidate-pool sizes, final ranking Recall@50, expanded-pool recall, action counts, loop-limit hits, fallbacks, invalid outputs, user coverage, and p50/p95 Service B and end-to-end latency. Keep the full-split and selected-user catalog-positive coverage distinct from prediction-user coverage.

## Failure labels and examples

Inspect a fixed, reproducibly selected sample on validation and a separately frozen descriptive sample on test. Record all applicable labels per case; counts may therefore overlap:

1. **Initial retrieval miss:** no eligible target in B3's initial pool.
2. **Expansion miss:** re-retrieval ran but added no eligible target, or a relevant expanded-pool item was omitted from the final 50.
3. **Ranking miss:** a relevant movie was in the final candidate pool but outside the final top 10.
4. **Clarification error:** the agent asked an unanswerable or irrelevant question, proceeded despite an incomplete-profile case under a rubric fixed on validation, repeated a question, or exhausted the loop limit. MovieLens has no native sufficiency label; record the rubric and state whether the judgment uses a simulator rather than a human response.
5. **Grounding error:** an explanation makes a claim unsupported by the supplied profile or movie metadata. Record the claim and the evidence actually available to the model.
6. **Service failure:** malformed action, missing trace, timeout, retrieval/index mismatch, checkpoint mismatch, or fallback. Preserve the error and trace identifier.

For each selected case include the user ID, profile/input hash, B3/P1/P2 ranks of relevant movies, initial and final pool membership, ordered actions, explanation excerpt and grounding evidence, latency, label(s), and a short causal interpretation. Do not infer that an action caused a score change merely because the two co-occur; use the controlled comparisons and any prespecified ablations to support that claim.

## Interpretation rules

- A P1-to-P2 difference measures the combined P2 checkpoint and agent-enabled pipeline unless a same-checkpoint loop ablation is also run. Describe that limitation explicitly.
- Clarification gains based on simulated answers support an offline simulation claim, not a claim about actual user satisfaction.
- If a paired interval includes zero, report the direction as uncertain under the specified user-resampling procedure.
- Report unfavorable and null findings, including cases where P2 improves one subgroup but harms the overall mean.

## Report completion record

When outputs arrive, add a dated results report with links to immutable input manifests, exact commands, table/interval artifacts, subgroup counts, example-selection seed, and the labeled example records. Record any deviations from this plan before inspecting final-test P2 scores.
