# Retrieval — Members 4 and 5

Member 4's B2 implementation is in `bm25.py` and `run_bm25.py`. Member 5 can add embedding retrieval separately in this folder. Both consume Member 1's canonical documents, profiles, and complete watched-ID lists. See [the roadmap](../../Movie_Recommendation_Agent_Roadmap.md) and [the Member 4 report](../../docs/member4_bm25_report.md).

## Reproduce B2

Run from the repository root using Python 3.10+. BM25 and the data pipeline require only the standard library. Raw data, indexes, prediction files, and generated reports are Git-ignored; the frozen configuration and small evidence summaries are committed.

Prepare the official MovieLens 25M dataset if the processed folder is absent:

```bash
python -m src.data.prepare
```

On macOS, if Python cannot locate its CA certificates, the tested download command is `SSL_CERT_FILE=/etc/ssl/cert.pem python3 -m src.data.prepare`. Certificate verification remains enabled.

Use the committed validation-selected configuration to rebuild the index and export predictions:

```bash
python -m src.retrieval.run_bm25 index
python -m src.retrieval.run_bm25 predict --split validation --output outputs/predictions/bm25_validation.jsonl --report-dir outputs/reports/bm25-validation
python -m src.retrieval.run_bm25 predict --split test --output outputs/predictions/bm25_test.jsonl --report-dir outputs/reports/bm25-test
```

The test command generates predictions and operational statistics only. It never reads test targets or calculates test relevance metrics. Member 2 should evaluate the frozen predictions once the final comparison is ready.

To reproduce model selection on validation only:

```bash
python -m src.retrieval.run_bm25 tune
```

This reads [configs/bm25_grid.json](../../configs/bm25_grid.json), evaluates 18 trials, and writes `configs/bm25_frozen.json` and `outputs/reports/bm25-tuning/tuning.json`. It samples 300 validation users with seed 20261001, saves the exact IDs, maximizes mean binary NDCG@10, and resolves exact ties by grid order. The sweep compares `k1 ∈ {0.6, 1.2, 1.8}`, `b ∈ {0.25, 0.75}`, and three explicit document-field conditions: full canonical text; title plus genres; genres plus both tag fields. Field conditions are declared representation ablations; the canonical user-profile text stays identical for every trial. After retuning, rerun `index` before exporting predictions.

Both `tune` and `predict` accept `--user-ids path/to/users.json`, a JSON array of unique user IDs supplied by Member 2. Supplying it to `tune` replaces local sampling. Supplying it to `predict` exports exactly that population and rejects IDs without profiles. The supplied subset must match the chosen split. By default, prediction covers every profile in that split; no user is skipped because of a low score or metric.

Additional paths are configurable through `--data`, `--config`, `--index`, and output flags; see each command's `--help`.

## Scoring and safeguards

BM25 uses `log(1 + (N - df + 0.5)/(df + 0.5))` IDF and `tf*(k1+1)/(tf + k1*(1-b+b*dl/avgdl))` term saturation. Query terms have binary frequency, so repetition does not multiply their contribution. An inverted index stores term counts; scoring weights are prepared once per parameter pair. JSON index persistence avoids executable deserialization.

The fixed tokenizer case-folds Unicode words, splits punctuation/hyphens, and removes English function words, shared template labels, and pure numbers, including ratings and years. It applies no stemming, synonyms, embeddings, or polarity handling. Stopwords and tokenizer version live in `bm25.py` and are independent of validation/test labels.

All recommendations exclude the complete `history_movie_ids`, including low-rated movies and watched IDs outside the catalog. Ties break by ascending movie ID. Empty or unmatched queries return unseen movies at score zero in ID order. If fewer than 100 unseen catalog movies exist, all available movies are returned and the short pool is reported. No popularity fallback or fabricated similarity score is used.

Dataset checksums, catalog/index consistency, configuration identity, and every prediction's IDs, ranks, scores, watched exclusions, and counts are checked before success is reported. Duplicate/missing subset users and incompatible data fail clearly. Index statistics use catalog documents only. No evaluation ratings are indexed. Shared genome tags retain Member 1's documented transductive-metadata limitation.

## Outputs and Member 2 handoff

Predictions use the roadmap's JSONL schema with `method: "bm25"`, one record per user, and 100 ranked candidates. The top 10/20/50 are prefixes of the same list. Indexing produces `data/processed/bm25/index.json` and its `.report.json` with build time and index hash.

Each prediction report records configuration, input/output/code hashes, Git revision, CPU/Python environment, candidate size, user and catalog coverage, latency, and invalid/short/zero-match counts. Latency includes tokenization, scoring, filtering, and top-k selection; index loading/preparation and file I/O are reported separately. It uses three warmup queries, then one measured query per user in a single CPU process.

Validation additionally produces:

- `summary.json`: macro Recall@10/20/50/100, NDCG@10, and coverage, including excluded positives across the entire validation split.
- `per_user_metrics.csv`: inspectable user-level diagnostics.
- `results.md`: a provisional B2 results-table row.
- `qualitative_examples.json`: the three lowest and three highest NDCG examples, query-term overlap, and missed relevant candidates.

These local diagnostics support tuning and failure analysis. **Member 2's shared evaluator is still a scaffold in this checkout; shared validation, confidence intervals, and the authoritative cross-method table remain that owner's integration step.** Feed it the exported JSONL without conversion. Do not use these validation diagnostics as final test claims. Full-validation results include the 300 tuning users.

Run the hand-calculated scoring/metric tests and synthetic end-to-end workflow tests:

```bash
python -m unittest discover -s tests -v
```
