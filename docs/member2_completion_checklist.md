# Member 2 — Phase 1 completion and integration checklist

## Implemented locally

- Ranking metrics: Recall@K, binary NDCG@K, HitRate@K, and equal-weight user means.
- Strict prediction validation: valid unseen movie IDs, complete expected users,
  contiguous ranks, finite scores, and exact P1/B3 candidate-set agreement.
- A provenance-bound, saved 300-user validation subset and experiment config.
- Full-source prediction validation followed by explicit subset/prefix
  preparation, with immutable outputs and transformation provenance.
- A plain-array user-ID export for Members 3/4's existing loaders.
- End-to-end scoring with per-user CSV, summary, coverage, and input identity.
- Paired user bootstrap intervals for matched experiments.
- Automatic Markdown/CSV tables with verified paired comparisons and provenance.
- Synthetic fixtures, hand-calculated checks, automated tests, and Mac instructions.

The official MovieLens dataset has been prepared and verified on Priyanka's Mac.
All tools use the Python standard library. Existing data and the virtual
environment can be reused throughout this work.

## Still required for the group handoff

1. Review the [draft protocol](evaluation_protocol.md) with the group: saved user
   list, exact 50-item output, rank/score semantics, P1 pool identity, and failure
   handling. The draft is not approved merely because the tools run.
2. Publish the reviewed implementation through `member2-evaluation` and the
   group's normal pull-request process. Check Git/GitHub for its current
   publication state; passing local checks alone does not publish the package.
3. Give Members 3–6 the [handoff guide](member2_handoff.md), prediction contract,
   fixed user file, and complete commands. Large data, model weights, and generated
   outputs stay out of Git; teammates reproduce data with Member 1's pipeline.

## Real result dependencies

| Owner | Required input |
|---|---|
| Member 3 | Popularity predictions; training examples following Member 6's specifications |
| Member 4 | BM25 predictions |
| Member 5 | Embedding predictions and immutable 50-item pools |
| Member 6 | Fine-tuned P1 rankings over those exact pools |
| Member 2 | Validate, score, compare, generate the shared table, review limitations |

Every method in the same table must use the same frozen users, catalog, profiles,
targets, and evaluation settings. Never train using the validation/test labels
reserved for evaluation. Do not silently drop users after inference failures.

The reviewed branches contain code and small reports; large prediction exports
must still be obtained through team storage. Follow the
[integration guide](member2_integration.md) to verify and prepare them. Confirm
the frozen embedding model and pool with Members 5/6. Preserve the original
exports and owner run metadata alongside the prepared artifact provenance.

Once the corresponding files are available locally, run the sequence:
full-source validation/preparation → scoring → paired comparison → table generation.
An initial real-baseline table is useful while fine-tuning is still in progress;
include only available methods and report the remaining work honestly. BM25 was
tuned on the same 300 validation users; do not present these as untouched test results.

## Midterm evidence to prepare from real results

- The actual methods evaluated, frozen population size, data version, and metrics.
- Recall@10/NDCG@10 and retrieval recall, with clearly named coverage values.
- P1 minus B3 differences and paired intervals when both are available.
- Retrieval misses and the fixed-catalog limitation; P1 cannot add absent movies.
- Model details, training/inference cost and latency, fallbacks, and explanation
  audit findings from the responsible members where available.
- What is implemented, what has been measured, and revised remaining milestones.

The instructor's midterm presentation is two slides and two minutes, followed by
one minute of questions. The midterm report is four pages, single-spaced, 10 pt;
it covers objectives, motivation, related work, method/novelty, hypotheses,
experiments/evaluation, preliminary findings if available, progress, and updated
milestones. Confirm submission dates in the course system; the supplied brief
does not specify them.

Synthetic tables demonstrate implementation correctness and cannot serve as
real performance findings. Phase 2 agent evaluation and final-test evaluation
remain later work; their protocols must be frozen before interpreting scores.
