# Member 2 — prepare teammate exports and complete the handoff

The evaluation commands are implemented. This guide bridges the interfaces in
the teammates' reviewed branches without changing their original predictions or
their model code. The shared evaluation protocol remains a draft for team review.

## Reviewed branch interfaces

| Owner | Reviewed commit | Existing export | Preparation for the current shared evaluation |
|---|---|---|---|
| Member 3 | `99272fb` | Popularity, normally all validation profiles and Top 100; `--user-ids` accepts a JSON array | Select saved users, retain Top 50; use the plain-array export when requesting a subset |
| Member 4 | `5799b0d` | BM25, normally all 4,751 validation users and Top 100; `--user-ids` accepts a JSON array | Select saved users and retain Top 50; preserve the original Top 100 file |
| Member 5 | `cd80dbf` | Embedding Top 100 plus exact `candidates_top50.jsonl`, covering all 4,751 validation users | Select saved users from the authoritative frozen Top 50 artifact |

BM25's frozen 300 tuning IDs match our saved 300 IDs. Its catalog, document,
validation-profile, and target hashes match the prepared data. Member 5's frozen
baseline metadata also matches the data. Report shared validation scores as
development results: the BM25 configuration was tuned on these users. Keep the
final-test comparison separate.

These commits publish code and small reports. Large prediction exports are
Git-ignored and are not supplied merely by fetching branches. Obtain them and
their run metadata through the team's agreed storage. No real prediction files
from those owners have been evaluated in Priyanka's checkout during this work.

## What preparation does

`src.evaluation.prepare predictions` requires an explicit source population:

- `--source-users all`: exactly every evaluable user in the split's profile file.
- `--source-users selected`: exactly the frozen user subset.

`--source-count` declares the number of recommendations in every source record.
The adapter validates the **entire source file** before selecting users or
truncating rankings. Wrong/duplicate/missing users, watched or unknown movies,
incorrect ranks, and invalid scores fail even in discarded users or ranking
positions. Preparation does not repair bad predictions.

After source validation, it selects the saved users, keeps the required prefix,
preserves recommendation order/scores/ranks and other record fields, serializes
deterministically, and validates the derived file against the shared contract.
The original file is unchanged. Input/output hashes and the explicit operation
are recorded in a separate preparation report.

`--source-label` records the owner-confirmed model/run identity. It is a label,
not independent verification of that model. Optional `--source-metadata` records
an owner-supplied JSON metadata object and its hash; its field meanings are not
automatically audited. Optional `--expected-source-sha256` verifies the source
file bytes against a checksum obtained from the owner or published run record.

## 1. Export the real saved IDs for Members 3 and 4

Activate the existing Mac environment:

```bash
cd "$HOME/Codex Projects/movie-rec-agent"
source .venv/bin/activate
```

Run:

```bash
python -m src.evaluation.prepare users \
  --config configs/evaluation_v1.json \
  --output-dir configs/validation_users_300_export_v1
```

This creates `user_ids.json` as a plain JSON array and
`user_ids.provenance.json` binding that export to the original metadata-rich
saved subset. Expected status is `users_exported`, with 300 selected users.
The original `configs/validation_users_300_v1.json` remains the authoritative
evaluation input. Share the exported array with Members 3 and 4; their current
`--user-ids` loaders can read it directly.

Teammates can use the committed array directly. Regenerating in another checkout
can change recorded absolute source paths; use a new output directory in that
case and compare the array hash instead of overwriting the original provenance.

## 2. Run a small adapter demonstration now

The preparation fixture has three evaluable source users, a saved two-user
subset, and four embedding recommendations per source user. The output keeps
three recommendations for each selected user. These small values illustrate
the real 4,751-to-300 and 100-to-50 operations.

```bash
python -m src.evaluation.prepare predictions \
  --config tests/fixtures/evaluation_preparation/config.json \
  --source tests/fixtures/evaluation_preparation/full_embedding.jsonl \
  --method embedding \
  --source-count 4 \
  --source-users all \
  --source-label synthetic-embedding-demo \
  --output-dir outputs/predictions/preparation_demo/embedding
```

Expected status is `prepared`, with `source_user_count: 3`, `selected_users: 2`,
`source_count: 4`, and `output_count: 3`. All records and scores in this example
are synthetic implementation checks.

The example does not supply an owner-published expected checksum, so
`expected_source_sha256_verified: false` is expected. The report still records
the actual source and derived hashes; the optional expected-checksum argument
adds comparison against a separately supplied identity.

Output files are `predictions.jsonl`, `preparation.json`,
`source_validation.json`, and `derived_validation.json`. P1 preparation also
writes a selected `candidate_pool.jsonl`. Existing identical outputs are
accepted; different outputs require a new directory. Inputs and data directories
are protected from overwrites. Invalid predictions exit 1; input/config/output
errors exit 2. Successful preparation exits 0.

## 3. Prepare actual popularity and BM25 exports when received

Save owner-provided files under a separate incoming directory. The paths below
are suggested local handoff locations, not claims that the files already exist.

```bash
python -m src.evaluation.prepare predictions \
  --config configs/evaluation_v1.json \
  --source outputs/predictions/incoming/b0-validation.jsonl \
  --method popularity \
  --source-count 100 \
  --source-users all \
  --source-label member3-popularity-v1 \
  --output-dir outputs/predictions/prepared/b0_validation_v1
```

```bash
python -m src.evaluation.prepare predictions \
  --config configs/evaluation_v1.json \
  --source outputs/predictions/incoming/bm25_validation.jsonl \
  --method bm25 \
  --source-count 100 \
  --source-users all \
  --source-label bm25-frozen-v1 \
  --output-dir outputs/predictions/prepared/b2_validation_v1
```

For a producer export already limited to exactly the saved users, use
`--source-users selected`. Keep `--source-count` equal to its actual full list
length. Member 4's reviewed code requires Top 100 exports; it does not need a
model/configuration change merely to derive the evaluation's Top 50 prefix.

The reviewed BM25 validation report publishes source SHA-256
`27940f3735fe35c5699d6124abf989c0ec10244c6d1e6c4adf5b150e5c27613d`.
For that exact artifact, append `--expected-source-sha256` followed by the hash
to the complete preparation command. Investigate a mismatch with the owner;
another run may be a different artifact and needs its own identity.

## 4. Prepare the authoritative frozen embedding pool

Confirm the baseline and artifact with Members 5 and 6. The reviewed report
keeps original MiniLM-L6 as its frozen delivery; Qwen/chunked experiments do not
automatically replace it. The branch name `english-only` is not evidence of an
English-only catalog: the inspected code and recorded data use the full catalog.

For the owner-confirmed frozen L6 Top 50 artifact:

```bash
python -m src.evaluation.prepare predictions \
  --config configs/evaluation_v1.json \
  --source outputs/predictions/incoming/frozen-l6/candidates_top50.jsonl \
  --method embedding \
  --source-count 50 \
  --source-users all \
  --source-label frozen-minilm-l6-v1 \
  --expected-source-sha256 0f61c4d87a73062400bfeea26371efaf33a25caf1cfd342e250de729be3706df \
  --output-dir outputs/predictions/prepared/b3_validation_v1
```

That checksum is recorded in Member 5's frozen baseline reference. It applies
to that exact file, not an exploratory model's file. Obtain the matching run
record as well. Use `--source-metadata` with its actual local JSON path if desired.

Member 5's Qwen archive is documented on an owner-local/Discovery path, without a
published team-wide download URL. Do not assume that artifact or the original
frozen pool is accessible from this Mac merely because its report is in Git.

## 5. Prepare P1 without changing its candidate set

P1 source lists must already contain the configured complete 50-item pool.
Top 100 P1 output is rejected rather than truncated. The source candidate file
must cover the same declared population as the P1 file.

For a P1 export already generated for the shared 300 users:

```bash
python -m src.evaluation.prepare predictions \
  --config configs/evaluation_v1.json \
  --source outputs/predictions/incoming/p1_validation_300.jsonl \
  --method p1 \
  --source-count 50 \
  --source-users selected \
  --source-label member6-p1-v1 \
  --candidate-pool outputs/predictions/prepared/b3_validation_v1/predictions.jsonl \
  --output-dir outputs/predictions/prepared/p1_validation_v1
```

For an all-profile P1 export, use `--source-users all` and its corresponding
all-profile frozen Top 50 candidate file. Both full files are checked before
the selected records are written. The prepared candidate artifact uses the same
canonical serialization as prepared embedding predictions, allowing exact
file-hash binding in the paired comparison.

## 6. Score prepared files and publish the implementation

Example once the real prepared B3 file exists:

```bash
python -m src.evaluation.evaluate \
  --config configs/evaluation_v1.json \
  --predictions outputs/predictions/prepared/b3_validation_v1/predictions.jsonl \
  --method embedding \
  --output-dir outputs/reports/b3_validation_v1
```

Use the corresponding prepared files/methods for B0, B2, and P1. P1 scoring
requires its prepared embedding pool. Then follow the
[bootstrap](member2_bootstrap_walkthrough.md) and
[table](member2_table_walkthrough.md) commands with the saved scored directories.
Keep preparation reports alongside scored bundles when sharing the complete
experiment record; a prediction hash binds them to the scored input.

Run the full automated checks before committing:

```bash
python -m unittest discover -s tests -v
```

Publish code, tests, documentation, and small configurations through
`member2-evaluation` and the group's review process. Include the exported plain
user-ID array and its provenance. Large source/derived prediction files and
generated experiment reports remain ignored. Publication does not approve the
draft protocol or establish real model results.
