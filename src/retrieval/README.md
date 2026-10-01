# Retrieval — Members 4 and 5

Member 4 owns BM25 (B2). Member 5's implementation below supplies B3 and immutable candidate pools for Member 6. Both consume Member 1's canonical text unchanged.

## Member 5 setup

Use **Python 3.12** (verified with 3.12.14). The locked NumPy, SciPy and NetworkX require Python >=3.12; Python 3.10/3.11 cannot install this lock. Newer interpreters/platforms have not been tested. From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-embedding.txt
python -m unittest discover -s tests -v
```

`requirements-embedding.txt` locks the environment used for these experiments; Member 1's standard-library-only `requirements.txt` is unchanged. Model weights download from Hugging Face on first use. No API key, paid inference, or training is required. For an alternate cache location set `HF_HOME`. Do not commit raw data, vectors, or model weights.

Member 1's data can be generated with `python -m src.data.prepare`. The commands below assume the standard processed-data path; pass the actual generated folder to `--data` if it is elsewhere.

## Reproduce validation experiments

```bash
python -m src.retrieval.run_embedding \
  --data data/processed/ml25m-global-v1 \
  --config configs/embedding-all-MiniLM-L6-v2.json \
  --split validation \
  --cache models/embedding-cache \
  --output outputs/predictions/embedding-L6-validation

python -m src.retrieval.diagnostics \
  --data data/processed/ml25m-global-v1 \
  --predictions outputs/predictions/embedding-L6-validation/predictions.jsonl \
  --output outputs/reports/embedding-L6-validation.json
```

Repeat with `configs/embedding-all-MiniLM-L12-v2.json` and a different output directory. The CLI reserves the output name with a process lock before encoding. A concurrent run using that name fails immediately. Each run writes to a private sibling staging directory and atomically publishes it only after all files validate. Any pre-existing output directory, including an empty/partial one, is refused. Failed runs remove only their own staging directory; a forcibly killed process may leave an orphan staging directory. Reuse the same cache directory for repeat runs and test export. A per-entry process lock covers cache lookup, encoding, and vector/metadata publication; concurrent cold starts wait (up to 600 seconds) and reuse the first completed result. A matching corrupt cache fails visibly; remove only that inactive cache entry before rebuilding. Persistent `.lock` files are normal and must not be deleted while jobs run; OS locks are released when a process exits. This coordination targets local filesystems, not distributed/network filesystem locks.

The two immutable model revisions are stored in the config files. Both models use their documented 256-token input limit. `run.json` reports how many canonical texts are truncated, so this loss of later history is visible. Model selection is predeclared in `docs/member5-plan.md`: validation Recall@50, then NDCG@10, then model name. Top-50 is the handoff contract even when reporting recall at 20 and 100.

## Outputs and index

Each output directory contains:

- `predictions.jsonl`: top 100 unseen catalog movies for each input user.
- `candidates_top50.jsonl`: the exact first 50 entries of those predictions.
- `index.npz`: normalized float32 movie matrix `vectors` and aligned `movie_ids`; together these form an exhaustive local nearest-neighbor index.
- `run.json`: frozen config, model revision, input/output hashes, code snapshot identity, environment, cache paths/hits, truncation counts, timings, and error count.

Each JSONL record uses `user_id`, `split`, `method: "embedding"`, and `recommendations` entries with `movie_id`, `score`, and one-based `rank`. Member 6 can consume the file directly and join on `movie_id` to Member 1's `movie_documents.jsonl` for movie text. Scores are cosine similarities, not probabilities. Never independently regenerate a different candidate set for P1.

The index uses exact cosine search, not ANN. It excludes every movie in the complete `history_movie_ids`. Equal scores are ordered by ascending MovieLens ID. If too few unseen movies exist to satisfy the requested count, the run fails instead of silently padding or duplicating. To reload:

```python
from pathlib import Path
from src.retrieval.embedding import ExactIndex
index = ExactIndex.load(Path('path/to/index.npz'))
# query must come from the same pinned encoder/settings.
recommendations = index.search(query, set(history_movie_ids), k=50)
```

Use `run.json` hashes to verify artifacts before sharing. Caches are local accelerators, not portable absolute-path dependencies: another machine can rebuild them from the pinned configuration and data.

## Evaluation and test policy

`src.retrieval.diagnostics` is deliberately validation-only and explicitly marks its output as **provisional Member 5 diagnostics**. It computes macro-average recall and binary NDCG against in-catalog positive targets; catalog coverage counts all targets, including users with only excluded positives. It validates user coverage, candidate membership, watched exclusion, ranks and duplicates. Member 2's evaluator remains authoritative for official results and confidence intervals.

After choosing configuration using validation only, use the same retrieval command with `--split test` and the selected config to generate test candidates. This reads test profiles, never test targets. Every profile receives candidates; Member 2 can later select the shared reranker subset without rerunning retrieval.

Retrieval latency in `run.json` measures search/filter/sort only; model loading and batch encoding are separately timed. It is not online end-to-end latency. Cached timing must not be compared with cold encoding timing as if both were first runs. Exact float scores can differ across hardware/runtime versions; frozen candidate files plus their hashes are the comparison artifacts.

## Team handoff dependencies

- Member 2: run shared evaluator, agree subset IDs, and produce official table/paired confidence intervals.
- Member 4: supply same-split standard BM25 predictions; no BM25 comparison is claimed before they exist.
- Member 6: confirm direct ingestion of top-50 JSONL and compare P1 to B3 on identical pools/users.

Model sources: [L6 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2), [L12 model card](https://huggingface.co/sentence-transformers/all-MiniLM-L12-v2), [Sentence Transformers API](https://sbert.net/docs/package_reference/sentence_transformer/model.html).

## This local checkout

The current desktop checkout links `.venv`, `data/processed/ml25m-global-v1`, `models/embedding-cache`, `models/hf-cache`, and `outputs/predictions/member5` to artifacts in the current Codex workspace. These ignored local links are convenience only and are not part of Git. To reuse downloaded weights here, set `export HF_HOME="$PWD/models/hf-cache"` before the commands above. Other machines should create their own environment and artifacts using the same commands.
