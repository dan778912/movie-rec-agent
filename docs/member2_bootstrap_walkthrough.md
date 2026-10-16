# Member 2 — paired bootstrap comparison

This step compares two methods using the same evaluation users. It reports the
average score difference and a percentile bootstrap interval. Python's standard
library is sufficient; your existing Mac environment is ready.

## 1. Confirm the scoring outputs are available

Your earlier terminal output showed 105 passing tests and successful synthetic
embedding/P1 scoring. Those commands created these result directories:

- `outputs/reports/scoring_demo/embedding/`
- `outputs/reports/scoring_demo/p1/`

Each contains `summary.json`, `per_user.csv`, and `validation.json`. The bootstrap
command reads all three files per run. If these directories are absent, run the
two scoring commands in the [previous walkthrough](member2_scoring_walkthrough.md).

Activate the environment from Terminal:

```bash
cd "$HOME/Codex Projects/movie-rec-agent"
source .venv/bin/activate
```

## 2. Run the paired comparison

```bash
python -m src.evaluation.bootstrap \
  --baseline-dir outputs/reports/scoring_demo/embedding \
  --challenger-dir outputs/reports/scoring_demo/p1 \
  --output-dir outputs/reports/scoring_demo/p1_vs_embedding
```

Here, embedding is the baseline and P1 is the challenger. Every difference means
**P1 minus embedding**. A positive difference favors P1 for that metric; a
negative difference favors embedding.

The command checks that both runs used the same evaluation configuration, data,
users, target counts, metrics, and scoring implementation. It also checks that
P1 reranked the exact embedding export used in this comparison. File order does
not determine pairing: records are matched by user ID.

Expected synthetic results, rounded for readability:

| Metric | Baseline mean | Challenger mean | Difference | 95% interval |
|---|---:|---:|---:|---|
| Recall@2 | 0.50 | 0.75 | +0.25 | [0.00, 0.50] |
| NDCG@2 | 0.50 | 0.806574 | +0.306574 | [0.226294, 0.386853] |
| HitRate@2 | 1.00 | 1.00 | 0.00 | [0.00, 0.00] |
| Recall@3 | 0.75 | 0.75 | 0.00 | [0.00, 0.00] |

The JSON uses metric keys such as `recall_at_2` and `ndcg_at_2`, with fields
`baseline_mean`, `challenger_mean`, `mean_difference`, `ci_lower`, and `ci_upper`.
It also records the comparison direction, user count, bootstrap settings,
and hashes of the report files it compared.

These are **two-user synthetic implementation checks**. They cannot establish
real model effectiveness, even when an illustrative interval lies above zero.
Real project conclusions require the actual frozen-population results.

## 3. Understand the resampling

For Recall@2, user 1 improves from 0.5 to 1.0, a difference of +0.5. User 2
stays at 0.5, a difference of 0. The observed mean difference is therefore
(0.5 + 0) / 2 = 0.25.

Each bootstrap resample draws two user IDs with replacement. Possible draws
include `[1, 1]`, `[1, 2]`, `[2, 1]`, and `[2, 2]`; their mean differences are
0.5, 0.25, 0.25, and 0. Both methods' scores move together whenever a user is
drawn. This preserves the paired comparison.

The current saved settings repeat this process 10,000 times with seed
`20261002`. The lower and upper endpoints are the 2.5th and 97.5th percentiles
of the bootstrap mean differences. Percentiles use linear interpolation at
`(number_of_resamples - 1) * p` after sorting.

An interval containing zero leaves the improvement direction uncertain under
this procedure. The interval describes variation from resampling users. It does
not include all uncertainty from retraining, model selection, or other design
choices. A zero-width interval occurs when every observed user has the same
paired difference; it does not establish certainty beyond this sample.

## 4. Inspect the two saved files

The output directory contains:

- `comparison.json`: metric means, differences, confidence intervals, settings,
  provenance, and comparison-input report hashes.
- `per_user_differences.csv`: baseline/challenger scores and their differences
  for each paired user, sorted by user ID.

Original scoring summaries continue to say `bootstrap_applied: false`: those
reports describe individual scoring runs. The new comparison is a separate
artifact. It does not rewrite earlier reports.

Repeating an identical comparison is accepted. Changed inputs, settings, or
recorded implementation require a new output directory. Preserve complete report
bundles when sharing results; this command checks their internal consistency
without reopening original model predictions or the large data files.

## 5. Run the automated checks

```bash
python -m unittest discover -s tests -v
```

Expect all tests to pass with `OK`. The added tests check paired arithmetic,
reproducibility, percentile calculation, mismatched users/experiments,
inconsistent reports, and output preservation. They use tiny temporary data.

## 6. Use actual results when available

After scoring the real B3 and P1 exports with the shared 300-user configuration,
compare those saved result directories:

```bash
python -m src.evaluation.bootstrap \
  --baseline-dir outputs/reports/b3_validation_v1 \
  --challenger-dir outputs/reports/p1_validation_v1 \
  --output-dir outputs/reports/p1_vs_b3_validation_v1
```

This command awaits real model predictions and scored outputs. It will compare
the real cutoffs 10/20/50 using the same pairing and bootstrap procedure.
Automatic table generation is now implemented: continue with the
[table walkthrough](member2_table_walkthrough.md). The team still needs actual
baseline/reranker results to populate its midterm evidence.
