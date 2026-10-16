# Tests

Run the standard-library suite from the repository root:

```bash
python -m unittest discover -s tests -v
```

`test_data_pipeline.py` covers Member 1's deterministic split, profile boundaries,
and repeatability. Member 2's tests cover metric arithmetic, prediction integrity,
reproducible subsets, full-source export preparation, end-to-end scoring, paired bootstrap intervals, and table
generation. They use tiny synthetic fixtures and temporary directories; they do
not download MovieLens or run recommendation models.

The [scoring fixture](fixtures/evaluation_scoring/README.md) has hand-calculated
expected values. Synthetic test scores and generated example tables do not
represent real model performance. See the [evaluation instructions](../src/evaluation/README.md)
for runnable examples and the [roadmap](../Movie_Recommendation_Agent_Roadmap.md)
for integration responsibilities.
