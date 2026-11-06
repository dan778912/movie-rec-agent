# Where Member 2's evaluation inputs come from

This page distinguishes files reproducible from this checkout from frozen
prediction handoffs that are Git-ignored. Run all commands from the repository
root unless the linked producer instructions say otherwise.

## Canonical MovieLens data: reproducible here

The root [README](../README.md) and [data pipeline](../src/data/README.md)
document the complete build:

```powershell
python -m src.data.prepare
```

It downloads the official MovieLens 25M archive and publisher checksum,
extracts the source CSVs into `data/raw/ml-25m/`, builds
`data/processed/ml25m-global-v1/`, and runs the repository verifier. If the
source CSVs already exist, use `python -m src.data.build --raw data/raw/ml-25m`
followed by `python -m src.data.verify`. Both directories are Git-ignored.
The resulting `manifest.json` and its file hashes must match the frozen
selection/scoring configuration before an official comparison.
This checkout now writes LF for `statistics.json` and `manifest.json` on
Windows, too; the reproduced manifest SHA-256 is
`f2e137ff374a8cfc02348cb42d04d9a04c9a1769685a7eccc21933383a500b5a`.

## Frozen B3: original export preferred

The earlier scorer read
`outputs/predictions/prepared/b3_test_500_v1/predictions.jsonl`, with SHA-256
`44ec13348849458fa58746e126369afc5fa0c2cf4bbd867bc732b83f412b93a1`.
The [preparation record](results/member2_test_500_v1/preparation/preparation.json)
identifies its source as Member 5's full 3,779-user L6 test candidate file,
`test_candidates_top50.jsonl`, with SHA-256
`45cd1b45d07bcf2c453a092261155889fd4029a39c9bc47aeccb7cb997653786`.
The original full and selected files are not tracked in Git.

If those files cannot be retrieved, the producer code and pinned configuration
are on `origin/member5-long-input-english-only`. Its
[retrieval README](https://github.com/dan778912/movie-rec-agent/blob/cd80dbf/src/retrieval/README.md)
documents Python 3.12, `requirements-embedding.txt`, and
`src.retrieval.run_embedding`; use `--split test` with
`configs/embedding-all-MiniLM-L6-v2.json`. Its output
`candidates_top50.jsonl` can be narrowed to the saved 500 users with
`src.evaluation.prepare predictions --method embedding --source-count 50
--source-users all` using [the preparation guide](member2_integration.md).
Compare both resulting hashes to the frozen values above. A newly generated
candidate file with a different hash is a **new run**, not a substitute for
the original B3 starting pool in the historical P1 comparison.

## Frozen P1: original model export required for exact replay

The [execution record](results/member2_p1_b3_test_500_v1/provenance/commands.json)
read `outputs/predictions/incoming/member6-p1-test-500-v1/p1_test_predictions_500_v1.jsonl`,
SHA-256 `15d5d332f61224824f339bdd73275522dec71a93ae05d0a460f9bb7fb1412634`.
Its [inference manifest](results/member2_test_500_v1/p1_test_inference_manifest_500_v1.json)
names the Qwen LoRA adapter archive SHA-256
`6d4a86f80f4f6f7bbe6f1b9c5df221f3847f12fce4afb99ea08afc4f1a32e392`,
base revision, source commit, inputs, 17 inference chunks, and A40 runtime.
The adapter weights and original prediction JSONL are not in Git. Member 6's
`origin/member-6-reranking` branch contains `src.reranking.run_reranker` for a
new run with `--adapter`, frozen profiles, movie documents, and B3 candidates.
The smoke keyword scorer is not the P1 checkpoint. A rerun must be separately
versioned and cannot claim to be the frozen P1 control unless its inputs and
output identity are verified.

## P2: produced after the reviewed freeze

Members 3, 4, and 6 must produce the final 50-rank P2 prediction JSONL,
complete `p2-trace-v1` JSONL, checkpoint/index versions, simulator record,
and timing/error manifest. The [interface contract](phase2_interface_contract.md)
defines their format and the [runbook](phase2_member2_runbook.md) shows how to
validate, score, compare, and diagnose them. No fetched branch currently
contains a completed P2 checkpoint, agent export, or Service B run.
