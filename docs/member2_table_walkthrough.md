# Member 2 — generate the shared results table

This step finishes the core Phase 1 evaluation tools. It turns saved scoring
and bootstrap reports into a readable Markdown table, CSV files, and a
machine-readable provenance record. The demonstration remains synthetic; real
project performance awaits your teammates' predictions.

## 1. Activate the environment and check the existing reports

Open Terminal and run:

```bash
cd "$HOME/Codex Projects/movie-rec-agent"
source .venv/bin/activate
```

Your previous successful commands created:

- `outputs/reports/scoring_demo/embedding/`
- `outputs/reports/scoring_demo/p1/`
- `outputs/reports/scoring_demo/p1_vs_embedding/`

Reuse these directories. If missing, follow the
[scoring](member2_scoring_walkthrough.md) and
[bootstrap](member2_bootstrap_walkthrough.md) walkthroughs first.

## 2. Generate the synthetic comparison table

Copy the complete command into Terminal:

```bash
python -m src.evaluation.report \
  --runs outputs/reports/scoring_demo/embedding outputs/reports/scoring_demo/p1 \
  --comparisons outputs/reports/scoring_demo/p1_vs_embedding \
  --kind synthetic \
  --output-dir outputs/reports/scoring_demo/table
```

`--runs` supplies the saved scoring directories. `--comparisons` supplies the
paired-bootstrap directory. `--kind synthetic` ensures the resulting table is
explicitly marked as a test example. `--output-dir` selects a new destination
for the generated files.

The printed JSON should report `status: "table_generated"`, two runs, and one
comparison. Methods without supplied reports are listed as missing; they
receive no invented row or score. Popularity and BM25 are therefore absent from
this demonstration. That is expected.

## 3. Check the metric values

The table should contain the following values, allowing for displayed rounding:

| Method | Recall@2 | NDCG@2 | HitRate@2 | Recall@3 |
|---|---:|---:|---:|---:|
| Embedding | 0.500000 | 0.500000 | 1.000000 | 0.750000 |
| P1 | 0.750000 | 0.806574 | 1.000000 | 0.750000 |

The paired-difference section uses challenger minus baseline, here P1 minus
embedding:

| Metric | Difference | 95% interval |
|---|---:|---|
| Recall@2 | +0.250000 | [0.000000, 0.500000] |
| NDCG@2 | +0.306574 | [0.226294, 0.386853] |
| HitRate@2 | 0.000000 | [0.000000, 0.000000] |
| Recall@3 | 0.000000 | [0.000000, 0.000000] |

Coverage is shown separately: selected-user catalog-positive coverage is 4/5 =
0.8, full-split catalog-positive coverage is 4/6, and prediction-user coverage
is 1.0. These measurements have different denominators. All scoring metrics
average the selected users equally.

The real experiment still uses the saved 300 users and cutoffs 10/20/50.
Two-user synthetic values are implementation checks, not midterm evidence of
model performance.

## 4. Inspect the four output files

The output directory contains:

- `table.md`: readable results, coverage, paired intervals, experiment identity,
  missing-method notes, and synthetic/draft labels.
- `table.csv`: one row per supplied method, with metric values and named coverage
  fields. Numerical precision is preserved.
- `differences.csv`: one row per supplied comparison/metric, containing the two
  means, difference, interval, and bootstrap settings.
- `report.json`: a machine-readable record of the table and input report hashes.

Markdown displays six decimal places. CSV/JSON retain full numerical precision.
Use generated values rather than retyping independent calculations in a
notebook. Identical reruns are accepted; changed inputs require a new output
directory to preserve prior results.

The generator checks that all runs share data, configuration, users, target
counts, metric definitions, and scoring implementation. It requires P1's frozen
pool to match the supplied embedding run. Saved comparison files must correspond
to the exact runs supplied to the table command and match the paired calculation.
Relocated report bundles can be used without reopening original data/model files.

## 5. Run the automated tests

```bash
python -m unittest discover -s tests -v
```

Expect all tests to pass with `OK`. The suite checks output values, mismatch
rejection, method ordering, labels, paired intervals, and safe file handling.
It uses tiny synthetic data and does not download or train any model.

## 6. Generate the real midterm table when predictions arrive

Once all four real methods have been scored and P1/B3 compared, use:

```bash
python -m src.evaluation.report \
  --runs outputs/reports/b0_validation_v1 outputs/reports/b2_validation_v1 outputs/reports/b3_validation_v1 outputs/reports/p1_validation_v1 \
  --comparisons outputs/reports/p1_vs_b3_validation_v1 \
  --kind experiment \
  --output-dir outputs/reports/midterm_validation_v1
```

This command requires actual model outputs. If only baselines are ready, supply
only their scored directories and omit `--comparisons` until a compatible paired
comparison exists. A single available baseline is supported. Any table containing
P1 must include its embedding counterpart.

The generator includes only available methods, lists missing ones, and preserves
the protocol's draft status. It does not infer training time, latency, cost,
fallback counts, or explanation quality; the responsible model owners supply
those separately.

## 7. Complete the handoff

Your core Member 2 evaluation tooling is implemented locally. Remaining project
work is to review/share that package, obtain team agreement on the draft rules,
receive real predictions, generate and interpret their results, and prepare the
midterm evidence. See the [handoff](member2_handoff.md) and
[completion checklist](member2_completion_checklist.md).

Generated output directories remain Git-ignored. Share implementation code,
small fixtures, frozen configuration files, and guides through the reviewed
branch. A small reviewed real-results excerpt can later be placed under `docs/`
for the midterm; synthetic results must remain visibly labelled.
