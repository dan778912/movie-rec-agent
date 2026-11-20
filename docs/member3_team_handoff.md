# Member 3 — team handoff

Branch: `member-3-service-b-retrieval`. This is implementation and integration
evidence; real P2 results still require the model, original pools, and team policy.

| Recipient | What is ready to hand off | What is still needed |
|---|---|---|
| Member 1 | Simulator now uses your full metadata loader and canonical-document checks; raw hashes verified against the dataset manifest. 62,423 metadata movies remain separate from 21,333 retrievable movies. | Confirm the intended frozen simulator policy and supply the P2 data/seed artifacts to their consumers. |
| Member 2 | Offline 50-item prediction producer, complete `p2-trace-v1` traces, `p2-run-manifest-v1` receipt validated on synthetic exports; real retrieval timing and tests. | Ratify checkpoint selection, prompt/simulator policy, budgets, fallback/grounding behavior, and runtime limits before official inference. |
| Member 4 | Your agent is integrated unchanged with real retrieval and a Qwen/LoRA process host. `/v1/decide` + `/v1/resume` preserve complete live episodes. | Joint smoke test on the actual P2 checkpoint and agreement on the proposed policy defaults. |
| Member 5 | Existing `/decide` client compatibility passes. Optional signed continuation API returns full live traces. Startup/config, Docker files, HTTP smoke CLI, health checks, and CI are included. | Adopt the v1 trace storage/continuation flow when agreed; share original ordered B3 JSONLs and run the full Service A/database/B integration. |
| Member 6 | Hugging Face base + single effective PEFT adapter/tokenizer loader, pinned base revision, greedy raw `p2-prompt-v1` inputs, no silent truncation; tiny real Qwen/LoRA load test passes. | Actual P2 artifact and loading settings, including whether P1 weights are merged into the effective adapter/base; confirm input/output budgets and provide GPU execution access. |

See `docs/member3_service_b_handoff.md` for exact commands, checkpoint directory
layout, request formats, trace storage, measurements, and remaining limitations.
The template's 6,144 input / 2,048 output tokens, 25-second HTTP deadline, and agent
defaults are development proposals, not team-approved experiment settings.

On a model timeout, the failed action stays recorded. Background recovery can reload
the same checkpoint for later live requests within a bounded restart budget. The
service does not substitute a mock model, alter a policy, or retry the failed
recommendation automatically. Official exports with unresolved failed episodes
remain incomplete and are not presented as valid P2 results.

Docker/Compose are covered by a new post-push CI workflow but remain unverified until
that workflow or a Docker host runs them. GPU latency and trained-P2 quality are
not measured locally. No teammates have been messaged by this implementation.
