# Member 5 implementation plan and checklist

Goal: implement and run B3 semantic retrieval on Member 1's frozen data, compare two pretrained encoders on validation only, and deliver reproducible top-50 candidate pools.

Architecture: Sentence Transformers encodes the existing canonical `text` without rewriting it. A persisted normalized float32 matrix plus ordered MovieLens IDs forms an exact local cosine index (NumPy exhaustive search, no approximate-index recall loss). Retrieval excludes the full `history_movie_ids`, orders by score descending then movie ID ascending, and emits the repository's standard JSONL records. Cached vectors are keyed by input hash, model revision, sequence length, and runtime versions.

Spec: `Movie_Recommendation_Agent_Roadmap.md` sections 4–9 (Member 5), and `DATA_DICTIONARY.md`.

## Decisions made before looking at experiment results

- Python >=3.10; isolated environment; no changes to Member 1's pipeline.
- Compare sentence-transformers/all-MiniLM-L6-v2 and all-MiniLM-L12-v2, pinned to immutable Hugging Face revisions. Both use their model's 256-token limit, normalized vectors, and identical canonical inputs. Record profile truncation.
- Evaluate all 4,751 validation profiles. Primary selection: mean in-catalog Recall@50; ties: NDCG@10, then model name. Top-50 remains the contract for Member 6; report recall at 20/50/100 to quantify pool-size tradeoffs.
- No test-target reading for model selection. Once selected, export validation and test candidate pools for all users, avoiding arbitrary subsets owned by Member 2. Do not calculate final test metrics in this task.
- Metrics in `src/retrieval/diagnostics.py` are explicitly provisional local diagnostics, not a replacement for Member 2's authoritative evaluator. Formal BM25 comparison awaits Member 4's standard predictions.
- Save model configuration, data hashes, code snapshot identity, timings, output hashes, environment versions, counts, and failures. Fail rather than silently emit undersized pools or stale cache results.

## Tasks

- [x] 1. Inspect existing data and design; create feature branch and isolated environment.
- [ ] 2. Implement tested exact index, validation, watched filtering, deterministic ordering, and cache integrity.
- [ ] 3. Implement pinned-model encoding and reproducible CLI, standard predictions, frozen top-50 exports, and artifact metadata.
- [ ] 4. Implement separately labeled diagnostic metrics and candidate-file validation with hand-calculated tests.
- [ ] 5. Run both models on full validation data; compare recall@20/50/100, recall@10, NDCG@10, truncation and runtime; choose model by predeclared rule.
- [ ] 6. Freeze selected configuration; generate all validation and test candidate pools; validate every record and hashes.
- [ ] 7. Document reproducible commands, model sources, results, limitations, and handoff; run full tests and independent review.

## Test and execution sequence

Write failing tests first for ranking/filtering/ties, insufficient candidates, malformed vectors and IDs, cache invalidation/corruption, metric denominators and duplicate predictions. Implement and run `python -m unittest discover -s tests -v` after each component. Run real inference and confirm output counts, valid ranks, catalog membership, seen-item exclusion, top-50/top-100 prefix equivalence, and repeat cached-run prediction hashes.

## Review focus

1. Incorrect vector-to-movie mapping, cache reuse after text/config changes.
2. Seen-item leakage and insufficient unseen candidates.
3. Silent tokenizer truncation, hardware-dependent floating-point order.
4. Missing users / duplicate recommendations or mismatched evaluation populations.
5. Accidental test-label tuning or representing provisional metrics as team results.

## Pending team dependencies

- Member 2: authoritative evaluator, agreed reranker subsets, confidence intervals and official table.
- Member 4: actual BM25 predictions for same-user comparison.
- Member 6: consumer acceptance and P1 comparison using the exact exported pools.

## Execution ledger

- User explicitly requested plan followed by execution of all currently feasible Member 5 work; proceed inline.
- Branch `member5-embedding` created in clean existing checkout; project and Git metadata writes authorized. No push or merge requested.
- Member 1 pipeline previously completed: all ten artifact checksums and 4,751 validation / 3,779 test profiles verified under Python 3.12.
- Formal team model/pool approval remains external; selection in this run is a reproducible Member 5 proposal with versioned candidate artifacts.
