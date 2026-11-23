# Tests

Place data, evaluation, retrieval, reranking, and application tests here. The current `test_data_pipeline.py` covers Member 1's deterministic split, profile boundary, and repeatability. Run `python -m unittest discover -s tests -v` from the repository root. See [Sections 8 and 9 of the roadmap](../Movie_Recommendation_Agent_Roadmap.md).

`test_p2_samples.py` builds a synthetic MovieLens fixture end to end and checks Member 1's Phase 2 artifacts: all six action types, 25%/50% masked profiles, P1-format per-movie scoring samples, temporal boundaries, deterministic replay, tamper detection, simulator intent round trips, the action grammar and episode rules, explanation grounding, and the catalog seed.
