# Member 5: long-input embedding experiments and validation results

Updated: 2026-10-09. These are **validation-only exploratory results**. The original frozen L6 configuration, predictions, and Member 6 candidate pools have **not** been replaced. No new test-set predictions or test metrics were produced for these experiments.

Qwen has the highest **Recall@50** among these three runs, but its **NDCG@10 decreases**. Relative to frozen L6, Qwen improves Recall@50 by **10.26%** (an absolute **0.2794 percentage-point** increase) while reducing NDCG@10 by **26.09%**. This is a candidate-retrieval improvement, not an across-the-board ranking improvement or evidence of better reranker results.

## Comparison and interpretation

All values below are fractions, not percentages. Full-precision values are in the [comparison CSV](experiments/member5-long-input/comparison.csv).

| Metric | Frozen L6, 256 tokens | L6 chunked profiles | Qwen full profiles |
|---|---:|---:|---:|
| Recall@10 | 0.00792632 | **0.00934303** | 0.00756962 |
| Recall@20 | 0.01359461 | 0.01507793 | **0.01537926** |
| Recall@50 | 0.02723213 | 0.02791435 | **0.03002625** |
| Recall@100 | 0.04222446 | **0.04712402** | 0.04712237 |
| NDCG@10 | 0.02043394 | **0.02199689** | 0.01510291 |
| NDCG@20 | 0.01960041 | **0.02144965** | 0.01659324 |
| NDCG@50 | 0.02199697 | **0.02368618** | 0.02078345 |
| NDCG@100 | 0.02663806 | **0.02931097** | 0.02657459 |

The comparison rule recorded in the [experiment plan](member5-long-input-plan.md) prioritizes Recall@50, followed by NDCG@10. Qwen is the numerical Recall@50 winner under that rule, including a 7.57% relative improvement over chunked L6. However, these exploratory results do not automatically promote a new frozen model.

Qwen and chunked L6 have almost identical Recall@100; their tiny difference is not a meaningful win/loss claim. Chunked L6 has higher NDCG at every reported cutoff. Qwen's improved Top 50 coverage is worth testing with Member 6, but aggregate retrieval metrics cannot establish whether reranking improves the final recommendations. No paired confidence intervals or statistical significance tests were computed.

Absolute Recall@50 remains low. A value of 0.0300 means about 3% of each user's in-catalog relevant targets are recovered on average; it does **not** mean 3% of the 50 recommendations are correct. Popularity/BM25 results under the same evaluation contract are still needed to contextualize performance.

## Data and evaluation contract

- Data version: `ml25m-global-v1`.
- Catalog: **21,333 movies**; validation profiles: **4,751 users**.
- Positive targets: **148,263**; out-of-catalog positives: **22,234**.
- Catalog positive coverage: **0.8500367590025832**.
- Canonical movie/profile text is preserved. Every movie in the complete `history_movie_ids` is excluded from retrieval.
- Each user receives Top 100 predictions and the exact Top 50 prefix; scores descend, with ascending MovieLens ID for equal scores.
- Recall and binary NDCG are macro-averaged over the diagnostic evaluator's eligible users using in-catalog positive targets. These remain **provisional Member 5 diagnostics**, not the official Member 2 evaluator.

The actual downloaded movie documents, validation profiles, validation targets, and manifest match the hashes recorded for both L6 comparisons. Exact hashes are retained in [Qwen run metadata](experiments/member5-long-input/discovery-12846755/run.json) and [validation metrics](experiments/member5-long-input/discovery-12846755/validation-metrics.json).

The backup contains validation data only. The supplied job script and verified retrieval/diagnostic source use the validation split. This local verification did not read test labels; it is not a system-level audit of all remote activity.

## Encoding configurations

| Setting | Frozen L6 | Chunked L6 | Discovery Qwen |
|---|---|---|---|
| Model | all-MiniLM-L6-v2 | all-MiniLM-L6-v2 | Qwen3-Embedding-0.6B |
| Inference device / dtype | CPU / float32 | CPU / float32 | CUDA / float16 |
| Batch size | 64 | 64 | 4 |
| Input limit | 256 | 254 content tokens per chunk + CLS/SEP | 2048, with overflow rejection |
| Profile handling | Truncate | Normalize each chunk; content-token-weighted mean; normalize again | Full profile with fixed query instruction |
| Movie handling | Original 256-token encoding | Original movie vectors reused | Full movie text, no instruction |

Pinned revisions:

- L6: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Qwen: `97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3`.

The Qwen query prefix is exactly the following JSON string, including its newline and final space:

```json
"Instruct: Given a user's movie rating history, retrieve unseen movies that match the user's preferences.\nQuery: "
```

Chunked L6 retains all **2,225,255 profile content tokens** across **9,991 nonoverlapping chunks**, with at most three chunks per profile. It does not decode and re-tokenize chunks. Its saved movie index SHA-256 is identical to the frozen L6 index. One movie document remains truncated under the unchanged L6 movie policy. Chunking changes aggregation as well as retained context, so its improvement is not an isolated length effect.

Qwen uses left padding and the pinned model's last-token pooling. The recorded maximum input lengths are **241 tokens for movies** and **660 tokens for profiles including the instruction**, with **zero truncation**. Index/cache vectors are normalized float32 even though inference uses float16. The downloaded metadata and source were checked; tokenization was not rerun during artifact verification.

Qwen changes model, instruction, context handling, device, and inference precision together. This comparison measures that complete configuration, not the causal effect of longer context alone. The successful school run uses [the Discovery configuration](../configs/embedding-Qwen3-0.6B-discovery.json), not the earlier local MPS configuration.

## Execution provenance and cost

Discovery job **12846755** ran on **one NVIDIA A40**, with **46,068 MiB** total device memory reported by `nvidia-smi`. The submitted job requested four CPUs and 32G host memory. These are allocation requests, not measured peak usage. See the [executed job script](experiments/member5-long-input/discovery-12846755/executed-job.sbatch) and [log excerpt](experiments/member5-long-input/discovery-12846755/job-log-summary.txt).

The runtime recorded Python 3.12.8, NumPy 2.5.3, sentence-transformers 5.7.0, tokenizers 0.23.2, torch 2.14.1, and transformers 5.18.0. Qwen's movie and profile vector caches both missed; this does not establish whether model weights were already cached.

| Stage / measurement | Frozen L6, historical CPU run | Chunked L6, CPU | Qwen, A40 |
|---|---:|---:|---:|
| Movie encoding or vector-cache access, s | 61.86 | 0.07 (cache hit) | 160.97 |
| Profile encoding, s | 64.82 | 188.83 | 84.76 |
| Runner total, s | 166.53 | 211.71 | 472.70 |
| Process peak RSS, GiB | Not recorded | 0.84 | 2.22 |

For Qwen, model loading took **200.97 s**, the two encoding stages **245.73 s** combined, and retrieval **20.91 s**. Loading accounts for approximately 42.5% of runner time; the available record does not separate downloads, storage waits, and other loading overhead. Other data handling accounts for the remainder of total time.

Retrieval latency was **4.40 ms mean**, **4.35 ms p50**, and **4.77 ms p95**, excluding query encoding. These are not online end-to-end recommendation latencies. Devices, cache states, and execution conditions differ, so this table is not a controlled model-only speed ranking.

Qwen process peak RSS was **2,383,683,584 bytes**. RSS is not GPU VRAM usage. `nvidia-smi` was captured before inference, so its initial 0 MiB usage does not measure inference memory; peak GPU memory was not recorded. The reported API inference charge is zero, which does not imply zero cluster resource cost.

User-supplied Slurm accounting reports `COMPLETED`, exit code `0:0`, **494 s** job elapsed, and Python steps of **479 s** and **4 s**. The script identifies the two steps as retrieval and diagnostics. The log spans 2026-10-08 16:35:16 to 16:43:20 PDT; Slurm and runner timing boundaries differ. The backup does not contain a separate `sacct` export, so that accounting remains attributed to the supplied terminal output.

The local MPS attempts were stopped and did not produce the reported Qwen result. Their partial caches were not mixed into the successful CUDA run. Local exploratory overhead is not included in Discovery's timing.

### Source identity

The Qwen run records no Git commit because the transferred execution directory did not contain `.git`. The log's `fatal: not a git repository` message is from provenance collection, not a failed inference job. All five recorded source/dependency hashes match the actual downloaded files and the implementation supplied in this branch:

```text
source_fingerprint: d17162eff57055322a9065ff420183d960c42ac3430b5b9fe601d48d1ac158c7
```

The full per-file hashes are in `run.json`. The earlier upload manifest identifies a Git starting point plus uncommitted source; that starting commit alone is not an execution version. The publication commit records the supplied code, not a claim that the school job ran from that later commit.

Chunked L6 ran before explicit dtype metadata was added for Qwen. Its [run metadata](experiments/member5-long-input/l6-chunked/run.json) and [exact execution-source snapshot](experiments/member5-long-input/l6-chunked/execution-source/) are retained separately. The original [L6 baseline reference](experiments/member5-long-input/baseline-reference.json) is also retained. Cross-machine byte-identical re-encoding is not promised; frozen files and hashes identify the measured results.

## Downloaded artifact verification

The local archive and extracted backup were read without modification. Verification covered:

- Actual input/manifest hashes, all three output hashes, and all five execution-source/dependency hashes.
- Equality of the job's saved config, Discovery config, and `run.json.config`.
- All **475,100 Top 100 entries** and **237,550 Top 50 entries**: schema, user coverage, catalog membership, deduplication, complete watched-history exclusion, ranks, sort order, and exact prefix relationship.
- Recalculation from saved predictions and validation targets using the hash-matched Member 5 diagnostic implementation: the complete metrics result matches the school JSON exactly. This is not an independently implemented official evaluator.
- Successful index loading: **21,333 x 1,024**, normalized float32, with movie IDs/order matching the documents.

See the [machine-readable verification record](experiments/member5-long-input/discovery-12846755/backup-verification.json). Actual query-to-index score replay was not performed because the archive omits the profile-vector cache; no model was restarted to reconstruct it.

The old upload manifest lists 40 files. The backup contains 34 matching files with no hash mismatches. Its missing data dictionary and five test files were recovered separately from local files whose hashes match that manifest. They are identified as local supplements, not as original server downloads. The current publication tests are run separately from this historical backup audit.

## Artifact locations and access

Small run manifests, metrics, the comparison table, verification evidence, and an execution-log excerpt are included under [`docs/experiments/member5-long-input/`](experiments/member5-long-input/). Full predictions, candidate files, index, data, and model/vector caches are not ordinary Git source files in this PR.

The verified archive is:

```text
qwen-validation-12846755-backup.tar.gz
SHA-256: 04c864de6141cb355d6746f4711de8f2213ea8e8a62c7fc75a3ff66854c03892
```

Current locations:

- Discovery: `/scratch1/xianhanh/movie-rec-transfer/qwen-validation-12846755-backup.tar.gz`.
- Owner's verified local copy: `~/Desktop/movie-rec-results/qwen-validation-12846755-backup.tar.gz`.
- Predictions inside the archive: `outputs/qwen-validation-12846755/`.

**No public GitHub Release asset or team-wide download URL has been published.** The server path requires authorized Discovery access and file permissions; it is not a public link. Teammates without access should obtain the checksum-identified archive from Member 5 through the team's agreed storage. Scratch storage should not be the only retained copy.

For an account authorized to read that archive, set the actual Discovery SSH host and username before downloading:

```bash
export DISCOVERY_HOST='your-authorized-discovery-ssh-host'
export DISCOVERY_USER='your-authorized-username'
mkdir -p movie-rec-results
scp "${DISCOVERY_USER}@${DISCOVERY_HOST}:/scratch1/xianhanh/movie-rec-transfer/qwen-validation-12846755-backup.tar.gz" movie-rec-results/
cd movie-rec-results
printf '%s\n' '04c864de6141cb355d6746f4711de8f2213ea8e8a62c7fc75a3ff66854c03892  qwen-validation-12846755-backup.tar.gz' | shasum -a 256 -c -
mkdir qwen-validation-12846755-backup
tar -xzf qwen-validation-12846755-backup.tar.gz -C qwen-validation-12846755-backup
```

The archive's data is a validation-only subset retaining the original manifest, not a complete dataset replica. It includes the inputs needed for the validation commands below. The original full log, actual job script, configuration, and source snapshot are retained in the archive.

## Reproduction commands

These are documented commands, **not a request to restart the stopped local experiments**. Use Python 3.12 and a compatible CUDA environment for the Discovery config. The archived run used Python 3.12.8. Model weights may download on first use; configure `HF_HOME` for the machine rather than relying on another user's absolute cache paths.

From the repository root:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-embedding.txt
python -m unittest discover -s tests -v

# Point this at the verified validation-only backup or the matching prepared dataset.
export M5_DATA='/absolute/path/to/qwen-validation-12846755-backup/data/processed/ml25m-global-v1'
export HF_HOME="$PWD/models/hf-cache"

python -m src.retrieval.run_embedding \
  --data "$M5_DATA" --config configs/embedding-L6-chunked.json \
  --split validation --cache models/long-input-cache \
  --output outputs/predictions/L6-chunked-validation-reproduction

# Run this command only on an allocated, compatible CUDA device.
python -m src.retrieval.run_embedding \
  --data "$M5_DATA" --config configs/embedding-Qwen3-0.6B-discovery.json \
  --split validation --cache models/qwen-cuda-cache \
  --output outputs/predictions/Qwen3-validation-reproduction

python -m src.retrieval.diagnostics \
  --data "$M5_DATA" \
  --predictions outputs/predictions/Qwen3-validation-reproduction/predictions.jsonl \
  --output outputs/reports/Qwen3-validation-reproduction.json
```

Use fresh output directory names: existing outputs are refused. To reproduce the original L6 baseline, substitute `configs/embedding-all-MiniLM-L6-v2.json` and a new output directory. To evaluate saved predictions without inference, run only the diagnostics command, pointing `--predictions` at the extracted archive's file. The CPU and MPS Qwen configs are alternatives, not the configuration that produced the school result.

The [archived Slurm script](experiments/member5-long-input/discovery-12846755/executed-job.sbatch) shows the actual execution sequence; adapt its account, storage paths, and environment setup before submission on another account. Do not edit frozen candidate files to make a reranker comparison.

## Remaining team work

Member 2 still needs to provide formal evaluator confirmation, a shared reranker user subset, and paired uncertainty estimates. Member 4's same-condition BM25 outputs are needed for that baseline comparison. Member 6 must confirm ingestion and evaluate B3/P1 on identical frozen pools. Publication of this report makes the experiment reviewable; it does not claim those integration tasks are complete or authorize replacing the frozen L6 delivery.
