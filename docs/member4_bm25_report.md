# Member 4 — BM25 (B2) baseline

Implementation, validation tuning, indexing, prediction exports, and the retrieval report are complete on `member-4-bm25`. The shared evaluator is still a README scaffold in this checkout, so acceptance into Member 2's authoritative comparison table remains an integration dependency. A provisional B2 table row is generated with the validation report.

## Deliverables

| Member 4 item | Artifact |
|---|---|
| BM25 over canonical movie documents | [Scoring/index module](../src/retrieval/bm25.py) |
| Watched-item removal and standard B2 exports | [Command module](../src/retrieval/run_bm25.py); `outputs/predictions/bm25_validation.jsonl` and `bm25_test.jsonl` |
| Reproducible indexing/retrieval commands | [Retrieval README](../src/retrieval/README.md) |
| Validation parameter and field tuning | [Predeclared grid](../configs/bm25_grid.json); [all trial results](bm25_tuning_results.json) |
| Frozen BM25 configuration | [bm25_frozen.json](../configs/bm25_frozen.json) |
| Candidate recall and latency | [Validation evidence](bm25_validation_summary.json); results below |
| Failure analysis | Concrete validation cases below; `outputs/reports/bm25-validation/qualitative_examples.json` |
| Verification | [Tests](../tests/test_bm25.py); real-file audit described below |

Large data, indexes, and prediction outputs are available locally and deliberately ignored by Git. A fresh clone recreates them using the documented commands. The small JSON evidence snapshots linked here are committed for review.

## Data and selection protocol

The official MovieLens 25M archive passed publisher MD5 verification. Member 1's existing pipeline reproduced `ml25m-global-v1`, verified all ten output checksums and profile history boundaries, and produced 21,333 catalog movies. Movie documents use training-derived user tags and static tag-genome metadata; the latter has the same transductive limitation documented by Member 1. No ratings, validation targets, or test targets are used to build BM25 statistics.

The predeclared sweep uses 300 validation users sampled with seed **20261001**. Their exact IDs are frozen in the configuration and the tuning output. It compares three `k1` settings, two `b` settings, and three document-field conditions, for **18 trials**. Every trial uses identical canonical profile text and the same users, targets, catalog, watched exclusions, and 100-candidate export size. Only validation targets are loaded. Mean binary **NDCG@10** is the selection objective; exact ties use grid order.

The selected configuration is **canonical `text`, `k1=0.6`, `b=0.25`**. The tokenizer removes pure numbers and fixed stopwords; it does not stem, expand synonyms, or interpret preference polarity. Full scoring and tokenization details are in the retrieval README.

Best NDCG@10 trial for each predeclared field condition on the 300 tuning users:

| Movie-document fields | k1 | b | Recall@10 | Candidate Recall@50 | NDCG@10 |
|---|---:|---:|---:|---:|---:|
| Canonical text — selected | 0.6 | 0.25 | 0.023407 | 0.058474 | 0.058131 |
| Title + genres | 1.2 | 0.75 | 0.008585 | 0.016017 | 0.019018 |
| Genres + genome tags + user tags | 0.6 | 0.25 | 0.023792 | 0.060971 | 0.053838 |

The title/genre-only condition loses substantial lexical matching information in these diagnostics. The tag-only condition has slightly higher candidate recall at its best-NDCG setting, but canonical text wins the registered ranking objective. These are validation selection results; no statistical significance is claimed. The full sweep took **34.28 seconds** on local CPU and incurred no paid compute/API cost.

## Full-validation results

The frozen configuration was applied to **all 4,751 validation profiles**, including the 300 tuning users. These results describe validation performance, not an independent final-test estimate. Metrics are macro averages over users; relevance is a held-out rating of at least 4.0, restricted to the frozen catalog.

| Method | Recall@10 | Candidate Recall@20 | Candidate Recall@50 | Candidate Recall@100 | NDCG@10 | Users |
|---|---:|---:|---:|---:|---:|---:|
| BM25 (B2) | 0.024875 | 0.038178 | 0.065834 | 0.098856 | 0.058992 | 4,751 |

All users received 100 unseen, unique catalog recommendations: **475,100 recommendations**, **100% user coverage**, no invalid predictions, no missing users, no short candidate pools, and no zero-match queries. Across these top-100 lists, recommendation catalog coverage was **54.88%**. Coverage is distinct from accuracy: 3,406 users had no relevant top-10 item, and 1,906 had no relevant top-100 item.

Catalog loss is reported separately. Across the entire validation split there are **148,263 positive target rows**, of which **126,029** are in the catalog and **22,234** are excluded (**85.00%** positive-target catalog coverage). Of the excluded rows, 157 belong to users with no in-catalog positive, who consequently have no evaluation profile. The metric denominator contains in-catalog target IDs for each evaluated user; excluded targets are not silently counted as retrieved or hidden from the coverage report.

The provisional results-table row lives at `outputs/reports/bm25-validation/results.md`. Member 2 should regenerate the cross-method table and any paired confidence intervals from the saved standard prediction files using the shared evaluator once it is implemented.

## Retrieval latency and index

Measured on macOS 26.5.1, ARM, Python 3.11.5, local CPU. The final validation benchmark ran after the test-export process completed. Three warmup queries precede one measured query per validation user. Times include query tokenization, BM25 scoring, complete watched-item filtering, and top-100 selection. They exclude index loading/weight preparation, file I/O, and diagnostics.

| Measurement | Value |
|---|---:|
| Mean retrieval latency | 5.823 ms/user |
| Median retrieval latency | 5.830 ms/user |
| p95 retrieval latency | 6.327 ms/user |
| Maximum retrieval latency | 19.077 ms/user |
| Index build + save | 0.354 seconds |
| Index load | 0.057 seconds |
| Weight preparation | 0.056 seconds |
| Validation export and summary, excluding qualitative-example writing | 29.93 seconds |
| Serialized index size | 3,288,361 bytes |
| Vocabulary size | 26,060 terms |

There is no model training or LLM inference in B2. Latency is a local measurement; CPU load and hardware can change it. [Index build evidence](bm25_index_summary.json) and the validation summary retain unrounded timings and hashes.

## Short failure analysis

Cases are selected deterministically from the three lowest and three highest validation NDCG scores, breaking ties by user ID. [Selected case evidence](bm25_qualitative_examples.json) is included for inspection. These examples illustrate behavior rather than estimate failure prevalence. An item absent from held-out positives is **unobserved as relevant**, not necessarily disliked.

**Literal matching succeeds for clear franchises and shared tags.** User **51760** likes *The Avengers* and *Spider-Man 2*. BM25 ranks the held-out positive *Captain America: Civil War* first, matching `superhero`, `superheroes`, `super hero`, `comic`, and `robert downey jr`. User **110019** likes *The Godfather: Part II*; *The Godfather* is a held-out positive at rank 1, supported by shared title, mafia, filmmaker, and actor terms. Both users have NDCG@10 = 1.0; this does not imply all ten recommendations are positive.

**BM25 cannot interpret negative preference statements.** User **101** rates *Donnie Darko* at **0.5/5**, so it appears in the canonical profile's low-rated section. BM25 still adds positive contributions for `donnie` and `darko` and places *S. Darko: A Donnie Darko Tale* at **rank 4**. That recommendation is not a held-out positive. This example exposes a limitation of keyword scoring over a profile containing both liked and disliked titles; changing profile treatment would need an explicitly comparable representation experiment.

**A broad history can bury the user's later interests.** For the same user **101**, crime/war/award vocabulary dominates the top results while held-out positives such as *You've Got Mail* and *Notting Hill* are absent from the top 100, despite sharing `comedy`, `romance`, and `romantic`. This user has Recall@100 = 0. User **91** similarly receives many science-fiction/war matches but misses the held-out positives *Möbius* and *Battle for Sevastopol* at 100. Exact topical overlap alone does not identify which part of a mixed history matters next.

**Lexical limitations remain.** BM25 cannot bridge unseen paraphrases or distinguish the intent of a tag, and sparse metadata provides fewer matching opportunities. Pure numeric years and sequel numbers are removed consistently from query and catalog text, which can also remove useful distinctions. The field sweep establishes that dropping tags performs poorly here; it does not isolate each of these other mechanisms. Candidate Recall@100 of 0.098856 leaves substantial room for Member 5's semantic retriever, to be compared on the same users by the shared evaluator.

## Verification and final-test handoff

All **10 repository tests passed**, including hand-calculated BM25 and metric examples, deterministic tie/zero-match behavior, watched filtering, Unicode tokenization, index save/reload, checksum rejection, and an end-to-end tune/index/predict fixture that has **no test-target file**. A separate audit reread both real JSONL outputs and checked exact user coverage, 100 unique in-catalog unseen IDs per user, ranks 1–100, descending scores, and agreement with the saved prediction hashes. A second full-validation export was **byte identical**.

Frozen B2 predictions were also generated for **all 3,779 test profiles**, yielding **377,900 recommendations**, with zero invalid/missing/short outputs. No test targets were loaded by the B2 command and no test relevance metrics were computed. Test prediction must use the validation-selected frozen configuration. Member 2 can subset these files or rerun prediction with `--user-ids` for the team's fixed population without any custom schema conversion.

Prediction SHA-256 values:

```text
validation  27940f3735fe35c5699d6124abf989c0ec10244c6d1e6c4adf5b150e5c27613d
test        778b880a88cf672675892699ff29a368f050c5aa779947df3b9645aebc42185e
```

The evidence JSON includes the input manifest/version, input/output SHA-256 hashes, configuration, random seed, base Git revision, and exact source-code hashes. The commands reproduce prediction bytes; runtime measurements naturally vary. Shared-evaluator acceptance and the final shared comparison table are the remaining team handoff, not an unreported successful check.
