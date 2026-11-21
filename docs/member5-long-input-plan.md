# Member 5 long-input validation protocol (2026-10-08)

User approved two additional validation experiments against frozen L6, not a replacement of its artifacts. Implementation proceeds in the existing retrieval flow on a separate branch.

## Fixed design before observing new metrics

- Baseline: existing L6 / 256-token artifacts; verify hashes and recalculate validation diagnostics. Do not replace selected config, handoff, or test predictions.
- L6 chunked profiles: tokenize canonical text without special tokens; split into nonoverlapping chunks of at most 254 content tokens plus two BERT special tokens. Encode each chunk with the pinned original L6, normalize each chunk, take a content-token-count-weighted mean, then L2 normalize. Do not decode/re-tokenize chunks. Movie encoding stays identical to baseline. Report both original length and zero dropped profile tokens. This changes aggregation as well as truncation.
- Qwen3-Embedding-0.6B: revision 97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3. Encode movies without prompt and profiles with `Instruct: Given a user's movie rating history, retrieve unseen movies that match the user's preferences.\nQuery: `. Use canonical text verbatim after this prefix. Left padding, model-provided pooling, float32 CPU and normalized embeddings. Measure token lengths including prompt before encoding; reject over-limit records instead of silently truncating. Start with 2048 tokens and raise only if required by input lengths, before observing metrics. This is a model+prompt+length comparison, not an isolated proof of truncation causality.
- Same 21,333 documents, 4,751 validation profiles, full watched history exclusion, exact index and top100/top50 format. No test labels or new test predictions.
- Primary descriptive comparison Recall@50, then NDCG@10; also Recall/NDCG@10/20/50/100, catalog coverage, cold encode/load/search/wall times and process peak RSS. No significance claim without paired intervals. Batch size may differ for memory, recorded explicitly. No automatic promotion based on this exploratory run.
- Cache identities include encoding strategy, query prompt and length policy for new encodings; L6 movie identity remains the original one. Original caches are not overwritten. Outputs are new directories in this task's outputs folder; fresh caches/model weights are in work.

## Implementation and execution

- [x] Write failing behavioral tests for token coverage, weighted normalized aggregation, role-specific prompts, full-input rejection, and cache identity separation.
- [x] Implement long-input encoder and integrate strategy-aware cache/measurement into the existing runner without changing default encoding.
- [x] Run entire suite and a real-model smoke test; measure Qwen CPU resource cost before full run.
- [x] Execute L6 chunking and Qwen full validation with fixed configs in fresh output directories.
- [x] Independently validate predictions, original artifact preservation, diagnostics and report metrics/costs without test labels.
- [x] Independent code review, resolve substantiated issues, and prepare reproducible code/config/report.

## Execution decisions and evidence

- New branch `member5-long-input-experiments` preserves the original branch. Tests were added before implementation, initially failed, then passed. A real-tokenizer smoke caught the removed Transformers `prepare_for_model` API; an integration regression reproduced it before switching to pinned BERT CLS/content/SEP tensors.
- Ruling: run Qwen on MPS with float16 if its resource smoke passes. The sandbox hid MPS; outside it MPS is available and physical memory is 16 GiB. The first CPU smoke automatically loaded native bfloat16 (4 movies: 5.01 s; 4 profiles: 60.21 s). We then explicitly fixed dtype: CPU float32 measured 0.93 s / 4.59 s, MPS float16 measured 1.23 s / 1.96 s for the same samples. These tiny batches include warm-up and cannot reliably predict full throughput. MPS was retained after the successful compatibility/resource check; the CPU config remains reproducible. This decision is based solely on resource cost, not labels or scores. Both device and dtype enter the cache identity/config; do not compare GPU and CPU wall times as a model-only speed claim. CPU config remains available as a slower alternative.
- L6 execution completed on CPU float32 before Qwen inference; exact executed source is archived beside its run manifest. No retrieval source is edited during a full experiment. New Qwen dtype identity is added before its full run. Baseline hashes and validation diagnostics were rechecked.

- Before the full Qwen run: 23/23 tests passed, including real BERT tokenizer tail retention, prompt application and overflow rejection. Independent read-only review found no blocking defect. Process RSS is not total MPS/GPU memory; no GPU memory peak is claimed.

- Qwen attempt 1 completed all movie vectors with batch 8 (~1467 s encode progress), but full-profile batches remained ~8–9 s and system swap usage was ~26 GiB. This is evidence of system pressure, not proof that this process caused all swap. Interrupted before any Qwen metrics; preserved completed movie cache and exact attempt log/source, then resumed in a fresh process with batch 4. Model/text/prompt/dtype unchanged. Movie vectors retain batch-8 provenance; new profile vectors use batch 4. Report successful-run cache times separately from the initial movie encoding and abandoned profile work.

- Attempt 2 (batch 4) was interrupted after 32 batches to investigate continued slowdown. A 120-profile resource probe showed stable active MPS tensor allocation (1,199,702,528 B) and driver allocation (2,290,745,344 B); empty_cache did not reduce driver allocation. This does not establish a leak. No speculative cache-clearing code was added. Final attempt resumes the original encoder with batch 4 and the same completed movie cache. Preserve both interrupted logs and account for their cost separately.

## Final execution and publication update (2026-10-09)

The local MPS runs were stopped at the user's request and are not the reported Qwen result. The user completed Discovery job 12846755 on an NVIDIA A40 using CUDA float16, batch 4, and fresh movie/profile vector caches. Its downloaded archive was verified and validation metrics were recomputed from predictions without re-encoding. See `member5-long-input-results.md` for the final results, evidence, reproducibility commands, and remaining limitations. The earlier CPU/MPS settings above document the initial plan and local attempts, not the final Discovery hardware.

The user subsequently authorized committing, pushing a feature branch, and opening a PR on 2026-10-09. This supersedes the earlier no-push instruction; no merge or replacement of frozen artifacts is authorized or performed.
