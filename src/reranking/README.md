# Member 6: Qwen reranking and LoRA fine-tuning

This folder contains Member 6's Phase 1 and Phase 2 model work.

## Phase 1: one-shot reranking

P1 receives Member 5's frozen embedding candidate pools, scores each candidate
using Qwen2.5-1.5B-Instruct fine-tuned with LoRA, and returns the same movie IDs
in a new order. P1 does not ask clarification questions or trigger wider
retrieval.

P1 uses:

- Member 3's `p1-training.jsonl` for fine-tuning.
- Member 1's canonical profile and movie-document files.
- Member 5's frozen top-50 embedding candidate pools.
- Member 2's fixed evaluation users and validator.

Only `profile` and `candidate_document` are model inputs for P1 training.
`label` is the target. Audit metadata is not placed in the prompt.

## Phase 2: agentic continued fine-tuning

P2 starts from the trained P1 LoRA adapter and continues fine-tuning it on
Member 1's six structured agent tasks:

1. `assess_sufficiency`
2. `ask_clarification`
3. `evaluate_candidates`
4. `retrieve_candidates`
5. `rank_candidates`
6. `explain_recommendations`

P2 training uses only each sample's `prompt` as input and `completion` as the
target. Hidden state, provenance, ratings, and action-label evidence are not
added to the model input.

The P2 model must return exactly one JSON object with no Markdown or surrounding
prose. The model is trained and served with the same `p2-prompt-v1` format used
by the agent controller.

## P2 inputs

The generated data files are Git-ignored and must be obtained from Member 1 or
reproduced from the frozen MovieLens data:

- `p2_samples_train.jsonl`
- `p2_samples_dev.jsonl`
- `p2_episodes_train.jsonl`
- `p2_episodes_dev.jsonl`
- `p2_quality_report.json`
- `catalog_seed.jsonl`
- Associated reports and checksums

Training and development users must be disjoint. Validation and test targets
must never be used as P2 training labels.

## P2 token-length policy

P2 provisionally uses a maximum sequence length of 6,656 tokens because ranking
and candidate-evaluation prompts are much longer than P1's 1,024-token inputs.

Silent truncation is prohibited. Before training, the code measures every
prompt and completion with the real Qwen tokenizer. Training stops if any
record exceeds the configured limit.

Run the tokenizer audit without loading the model:

```bash
python -m src.reranking.train_p2_qwen \
  --training-data /path/to/p2_samples_train.jsonl \
  --development-data /path/to/p2_samples_dev.jsonl \
  --initial-adapter /path/to/p1/adapter \
  --output-dir /new/path/to/p2-length-check \
  --length-check-only
```

Use a new, empty output directory for each run.

## P2 smoke test

Run a small GPU smoke test before full training:

```bash
python -m src.reranking.train_p2_qwen \
  --training-data /path/to/p2_samples_train.jsonl \
  --development-data /path/to/p2_samples_dev.jsonl \
  --initial-adapter /path/to/p1/adapter \
  --config configs/p2_qwen_lora_v1.json \
  --output-dir /new/path/to/p2-smoke \
  --max-train-examples 100 \
  --max-dev-examples 50
```

A smoke test confirms that the P1 adapter can be loaded as trainable, the
6,656-token configuration fits the selected GPU, and a P2 adapter can be saved
and reloaded. Smoke-test metrics are not final experiment results.

## Full P2 training

After the length and GPU smoke checks pass:

```bash
python -m src.reranking.train_p2_qwen \
  --training-data /path/to/p2_samples_train.jsonl \
  --development-data /path/to/p2_samples_dev.jsonl \
  --initial-adapter /path/to/p1/adapter \
  --config configs/p2_qwen_lora_v1.json \
  --output-dir /new/path/to/p2-full-v1
```

The output directory contains:

- `adapter/`: the selected P2 LoRA adapter.
- `checkpoints/`: retained training checkpoints.
- `token_lengths.json`: real tokenizer-length statistics.
- `checkpoint_candidates.json`: checkpoints ordered by dev loss.
- `run.json`: configuration, hashes, task counts, runtime, hardware, and
  training metrics.

The current proposed checkpoint rule shortlists the three retained checkpoints
with the lowest completion loss on Member 1's 400-user training-period dev
split. The shortlisted checkpoints are then run end to end on the fixed
300-user system-validation set. The highest macro NDCG@10 wins, with an exact
tie broken by the earlier training step. This rule must be approved before
system-validation results are inspected.

## Source files

### P1

- `config.py`: validates the P1 LoRA configuration.
- `training_data.py`: loads P1 relevance examples.
- `prompts.py`: defines P1 ranking and explanation prompts.
- `train_qwen.py`: trains the P1 LoRA adapter.
- `inference_data.py`: loads P1 inference files.
- `qwen_scorer.py`: scores P1 candidates.
- `reranker.py`: reorders candidate pools.
- `run_reranker.py`: runs P1 inference.

### P2

- `p2_config.py`: validates the P2 continued-training configuration.
- `p2_training_data.py`: validates Member 1's P2 JSONL handoff.
- `train_p2_qwen.py`: audits lengths and continues training the P1 adapter.
- `configs/p2_qwen_lora_v1.json`: versioned P2 training settings.

## Tests

The unit tests use synthetic data and do not download or load Qwen:

```bash
python -m unittest discover -s tests -v
```

The keyword-overlap scorer in `reranker.py` is only for pipeline smoke tests.
It is not an official P1 or P2 model.