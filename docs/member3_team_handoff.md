# Member 3 — team handoff

Branch: `member-3-service-b-retrieval`. This is implementation and integration
evidence; real P2 results still require the model, GPU execution, and team policy.
Member 3 has accepted ownership of the 300-user inference/export runner using
Priyanka's controller. Member 2 retains authoritative scoring and comparisons.

| Recipient | What is ready to hand off | What is still needed |
|---|---|---|
| Member 1 | Your v2 ZIP passed all 14 supplied checksums. Seed IDs/text/prompt metadata match the current 21,333-movie index, so no rebuild is needed. Your simulator source is byte-identical to ours. | Simulator confirmation received; remaining full-policy freeze is coordinated with the team. |
| Member 2 | Your November 24 first-validation policy is now reflected in the runtime template. Explanation-template sources/reasons are traced and counted separately; complete synthetic exports pass your validator. | Freeze checkpoint/index identity and GPU-measured deadlines before the 300-user export; authoritative scoring remains yours. |
| Member 4 | Your `211a274` explanation-fallback implementation is integrated unchanged. Malformed output, wrong IDs and timeouts retain the valid ranking, with ten labelled catalog explanations. | Raw P2 output/parser checks when the actual checkpoint is available; no new interface implementation needed. |
| Member 5 | Existing `/decide` client compatibility passes. Optional signed continuation API returns full live traces. Original 300-user B3 JSONL received, hash-matched, and validated. | Adopt the v1 trace storage/continuation flow when agreed and run the full Service A/database/B integration. |
| Member 6 | Your tokenizer audit and 6,144 / 1,024 serving limits are acknowledged. Single effective continued-P1 adapter loading is supported. | Actual P2 checkpoint shortlist, hashes/identity, loading example, training provenance, and raw action outputs. |

See `docs/member3_service_b_handoff.md` for exact commands, checkpoint directory
layout, request formats, trace storage, measurements, and remaining limitations.
The template uses 6,144 input / 1,024 output tokens, confirmed by Member 6's real
tokenizer audit. Training uses 6,656 total tokens. Andrew confirmed budgets,
zero retries, B3-order fallback, flagged grounding, and labelled template
explanations after explanation failure. The 20-second model / 25-second episode
or request deadlines remain provisional until the GPU smoke measurements.
Enable `agent.explanation_failure_policy: "template"` in an existing local real
configuration; upstream's default is intentionally still `fail` for old callers.

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
