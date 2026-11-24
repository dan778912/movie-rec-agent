# Member 3 — team handoff

Branch: `member-3-service-b-retrieval`. This is implementation and integration
evidence; real P2 results still require the model, GPU execution, and team policy.
Member 3 has accepted ownership of the 300-user inference/export runner using
Priyanka's controller. Member 2 retains authoritative scoring and comparisons.

| Recipient | What is ready to hand off | What is still needed |
|---|---|---|
| Member 1 | Your v2 ZIP passed all 14 supplied checksums. Seed IDs/text/prompt metadata match the current 21,333-movie index, so no rebuild is needed. Your simulator source is byte-identical to ours. | Simulator confirmation received; remaining full-policy freeze is coordinated with the team. |
| Member 2 | Offline 50-item prediction producer, complete `p2-trace-v1` traces, `p2-run-manifest-v1` receipt validated on synthetic exports; real retrieval timing and tests. | Ratify checkpoint selection, prompt/simulator policy, budgets, fallback/grounding behavior, and runtime limits before official inference. |
| Member 4 | Your agent is integrated unchanged with real retrieval and a Qwen/LoRA process host. `/v1/decide` + `/v1/resume` preserve complete live episodes. | Joint smoke test on the actual P2 checkpoint and agreement on the proposed policy defaults. |
| Member 5 | Existing `/decide` client compatibility passes. Optional signed continuation API returns full live traces. Original 300-user B3 JSONL received, hash-matched, and validated. | Adopt the v1 trace storage/continuation flow when agreed and run the full Service A/database/B integration. |
| Member 6 | Hugging Face base + single effective PEFT adapter/tokenizer loader, pinned base revision, greedy raw `p2-prompt-v1` inputs, no silent truncation; tiny real Qwen/LoRA load test passes. | Actual P2 artifact and loading settings, including whether P1 weights are merged into the effective adapter/base; confirm input/output budgets and provide GPU execution access. |

See `docs/member3_service_b_handoff.md` for exact commands, checkpoint directory
layout, request formats, trace storage, measurements, and remaining limitations.
The template now uses 6,144 input / 1,024 output tokens, following Member 1's
serving recommendation; Member 6 confirmation is still pending. The 25-second
HTTP deadline, retries, grounding/fallback rules, and other agent defaults are
not approved experiment limits. The v2 data handoff reports training samples up
to 6,458 total tokens, exceeding Member 6's current 5,120-token training config;
its proposed training budget is 6,656. Member 6 must resolve training lengths.

On a model timeout, the failed action stays recorded. Background recovery can reload
the same checkpoint for later live requests within a bounded restart budget. The
service does not substitute a mock model, alter a policy, or retry the failed
recommendation automatically. Official exports with unresolved failed episodes
remain incomplete and are not presented as valid P2 results.

The pushed f66d683 implementation passed GitHub CI, including its CPU Docker build
and container HTTP smoke. Current handoff/config updates still require their own
post-push CI run. GPU latency and trained-P2 quality are not measured locally.
No teammates have been messaged by this implementation.

Verified local inputs:
- Seed: `data/processed/handoff/member1-p2-v2/catalog_seed.jsonl`.
- Frozen B3 pools: `outputs/predictions/incoming/member3-validation/b3.jsonl`.
- Receipt: `docs/member3_handoff_receipt_v2.json` (hashes and prompt-parity evidence).

The existing index remains bound to canonical `movie_documents.jsonl`; the receipt
proves that its per-movie embedded text and prompt metadata equal the seed. Its
immutable manifest is preserved rather than relabelled as a new index build.
