# Member 6: P1 reranking and LoRA fine-tuning

This folder implements the Phase 1 P1 one-shot reranker.

P1 receives Member 5's frozen embedding candidate pools, scores each candidate
using a Qwen2.5-1.5B-Instruct model fine-tuned with LoRA, and returns the same
movie IDs in a new order. It also generates a short explanation for each
recommendation using only the supplied user profile and canonical MovieLens
movie document.

P1 does not ask clarifying questions, trigger re-retrieval, or implement the P2
agent loop.

## Inputs

The reranking pipeline uses:

- Member 3's `p1-training.jsonl` for fine-tuning.
- Member 1's `validation_profiles.jsonl` for user profile text.
- Member 1's `movie_documents.jsonl` for canonical movie text.
- Member 5's frozen `candidates_top50.jsonl` candidate pools.
- Member 2's fixed validation-user subset and prediction validator.

Member 3's training examples use only the `profile` and
`candidate_document` fields as model input. The `label` field supplies the
target. Ratings, timestamps, IDs, and provenance fields are not added to the
model prompt.

## Files

- `config.py`: validates the versioned LoRA configuration.
- `training_data.py`: loads Member 3's pointwise relevance examples.
- `prompts.py`: contains the ranking and explanation prompts.
- `train_qwen.py`: fine-tunes Qwen with LoRA.
- `inference_data.py`: loads profiles, movie documents, and candidate pools.
- `qwen_scorer.py`: loads the saved adapter for scoring and explanation generation.
- `reranker.py`: reorders candidates without changing the candidate movie set.
- `run_reranker.py`: command-line inference entry point.

The keyword-overlap scorer in `reranker.py` is only for pipeline smoke tests.
It is not the official P1 model and must not be reported as a P1 experiment.

## Lightweight tests

The unit tests use only the Python standard library and synthetic data:

```bash
py -3.12 -m unittest discover -s tests -p "test_reranking.py" -v