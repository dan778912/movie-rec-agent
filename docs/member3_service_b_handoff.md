# Member 3 — retrieval and Service B handoff

Branch: `member-3-service-b-retrieval`. Python **3.12**.
Run commands from `/Users/karunya/Documents/movie-rec-agent` (or your clone's root).

This implements the Member 3 Phase 2 integration. It provides real Chroma retrieval,
a Qwen/PEFT checkpoint host, Member 4 agent integration, a Service A-compatible API,
an offline producer, measurements, and container configuration. It does not establish
that P2 improves recommendation quality. No P2 weights are present locally, and
this Mac has no CUDA GPU. The original 300-user B3 JSONL has been received and
verified. No official P2 run was performed.

## What each section does

| Component | Purpose |
|---|---|
| `src/retrieval/online.py` | Encodes the catalog with pinned MiniLM, creates a versioned Chroma collection, builds queries from the current profile and answers, excludes watched/existing IDs, and returns expansion candidates. |
| `src/app/service_b/model.py` | Loads the base model and LoRA adapter once in a subprocess, hashes checkpoint files, rejects oversized prompts, and terminates inference on timeout. |
| `src/agent/` | Member 4's unchanged controller: prompts, strict action parsing, bounded loops, explanations, grounding screen, and trace events. |
| `src/app/service_b/main.py` | FastAPI HTTP boundary called by Service A. Returns a question or ten recommendations and logs trace segments and timings. |
| `src/app/service_b/offline.py` | Runs complete saved-user episodes using frozen B3 starting pools and pre-cutoff simulated answers. Produces predictions, traces, telemetry, and an independently validated receipt. |
| `src/app/service_b/benchmark.py` | Measures real retrieval latency, without pretending to measure P2 inference or recommendation quality. |
| Dockerfile and Compose files | Package Service B and Chroma separately; supply an optional NVIDIA GPU override. |

## Start locally now

The implementation environment `.venv-service-b` was created locally and is ignored
by Git. On a new machine create it with Python 3.12 and install dependencies:

```bash
python3.12 -m venv .venv-service-b
.venv-service-b/bin/python -m pip install -r requirements-service-b-model.txt
.venv-service-b/bin/python -m uvicorn src.app.service_b.main:app --port 8001 --log-config configs/service_b_logging.json
```

The default configuration is **explicit mock mode** using Member 4's demonstration
stand-ins, not the trained model. `/health/ready` and `X-Service-B-Mode` identify it.
The lighter `requirements-service-b.txt` is sufficient for mock/API/retrieval tests.
Model loading tests and real encoding need the model requirements file.

Example request (Service A's existing wire shape):

```bash
curl http://localhost:8001/decide -H 'Content-Type: application/json' -d '{"profile_text":"I like thrillers","history_movie_ids":[1,3],"clarifications":[]}'
```

Send any returned question and its answer back in `clarifications` on the next
request. The service keeps no per-user memory. Service A already has `SERVICE_B_MODE=http`
and `SERVICE_B_BASE_URL=http://localhost:8001` settings. Its current client timeout
is 30 seconds; development Service B's request deadline is 25 seconds. These are
development settings, not approved evaluation thresholds.

HTTP errors: malformed payloads use 422; busy inference uses 429; unavailable
resources, initial search failure or an agent failure use 503; a request deadline
uses 504. Member 4's proposed `action:error` becomes HTTP 503 so Service A does not
mistake it for recommendations. Once the model subprocess times out it is stopped
and readiness fails during recovery. Service B attempts a bounded background reload
of the same checkpoint for future requests (two restarts per process by default).
It never retries a failed action invisibly. A changed checkpoint or exhausted restart
budget requires operator intervention. Only one request is admitted;
HTTP cancellation does not free admission while underlying work is still active.
Remote Chroma calls use a small transport adapter for pinned Chroma 1.3.5 to set
HTTP phase timeouts (upstream uses no timeout). `CHROMA_REQUEST_TIMEOUT_SECONDS`
defaults to five seconds; the service request deadline still bounds the HTTP
response. This settings adapter applies after client initialization and uses one
private httpx-session attribute, which must be retested when upgrading Chroma.
Client initialization itself is not covered; external startup supervision should
restart a service that cannot finish connecting to Chroma.

### Complete live trace API (optional Service A extension)

The existing `/decide` response stays unchanged. To retain a complete live episode,
set `SERVICE_B_STATE_SECRET` to a stable secret of at least 32 bytes, then use:

1. `POST /v1/decide` with the same profile/history request and no previous clarifications.
2. Persist its `trace_id`, `decision_trace`, and opaque `continuation` in Service A.
3. When the user answers, `POST /v1/resume` with `{continuation, answer}`.
4. Replace the stored trace with the returned full trace. Recommendations end with
   `continuation:null`; failures return the recorded trace inside the 503 detail.

These responses preserve `action`, `question`, and `items`, adding `contract_version`,
`request_id`, `mode`, `trace_id`, `decision_trace`, `telemetry`, and `continuation`.
The state is signed, expires after one hour, and is bound to checkpoint, model
settings, prompt version, index, and agent policy. Tampered, expired, or incompatible
state is rejected. It is signed rather than encrypted and belongs in Service A's
protected session storage. Service B keeps no conversation database. A stable
secret must be configured across service replicas and restarts; no secret is
committed. Service A adoption is still a Member 5 integration step.

Live traces use `service-b-live-episode-v1`, not the offline evaluator schema.
Only explicit continuation preserves all original decisions; legacy replay remains
compatible and logs individual request segments. Live readiness now also checks
the loaded index's availability and movie count.

## Real index and checkpoint setup

The local catalog has already been indexed into `data/indexes/chroma` with receipt
`data/indexes/movie-index-v1.json`. Both are generated, ignored artifacts. To build
a new index on another machine:

```bash
HF_HOME=data/indexes/huggingface .venv-service-b/bin/python -m src.retrieval.online --manifest data/indexes/movie-index-v1.json
```

An existing receipt or same-content collection is refused rather than overwritten.
Use an unused receipt for a new catalog/encoding version. Ingestion checks CSV/document
ID agreement, records normalized-vector hashes, and publishes a ready collection
only after complete ingestion. Startup verifies receipt metadata, document hashes,
encoding settings, and movie count. The embedding model is
`sentence-transformers/all-MiniLM-L6-v2` at revision
`1110a243fdf4706b3f48f1d95db1a4f5529b4d41`, using 256 tokens.
Collection search is cosine **approximate** nearest-neighbor search, not the old
NumPy exact search. Accordingly it is never used to regenerate official B3 starts.
Live query text places the latest clarification answers before the canonical profile
so the embedding model's 256-token limit does not discard new preferences.
Equal returned distances are ordered by movie ID. Filters run inside Chroma and
are checked again before returning IDs. Exhausted eligible catalogs return fewer
IDs explicitly; a live response with fewer than ten eligible items fails explicitly.

Member 1's `p2-samples-v2` handoff has now been received and verified: all 14
SHA256SUMS entries passed. Its seed has the same 21,333 IDs, embedded `text`, and
prompt metadata as the canonical documents already indexed. The current index
therefore remains valid without rebuilding or changing its immutable manifest.
The seed is available locally at
`data/processed/handoff/member1-p2-v2/catalog_seed.jsonl`; provenance and parity
checks are recorded in `docs/member3_handoff_receipt_v2.json`.

Copy `configs/service_b_real_v1.example.json` to ignored
`configs/service_b_local.json` and set actual checkpoint paths and limits:

```bash
SERVICE_B_CONFIG=configs/service_b_local.json HF_HOME=data/indexes/huggingface .venv-service-b/bin/python -m uvicorn src.app.service_b.main:app --port 8001 --log-config configs/service_b_logging.json
```

Member 6 checkpoint folder requirements:

```text
models/p2-selected/
  adapter_config.json
  adapter_model.safetensors       # or adapter_model.bin
  tokenizer_config.json
  tokenizer.json                 # and any tokenizer-specific supporting files
```

The adapter's `base_model_name_or_path` must match configuration. Pin the base
revision; the loader currently supports Member 6's Qwen2.5 architecture and PEFT
format. It uses raw `p2-prompt-v1` prompts with no added chat-template wrapper,
greedy generation, and no input truncation. The example now uses 6,144 input /
1,024 output tokens, following Member 1's serving recommendation. Member 6 must
still confirm compatibility with the delivered P2 checkpoint. The v2 handoff
reports a longest training record of 6,458 total tokens and proposes a 6,656-token
training budget; Member 6's currently published 5,120-token training setting would
reject over-length examples. Training length and serving limits are separate.
P1's 1,024-token pointwise scoring limit is not
silently reused for P2. P1 remains a separate upstream ranking/scoring pipeline;
this service does not label a P1 adapter as a trained P2 agent.

## Offline producer and approval gate

Member 3 has accepted ownership of running the fixed 300-user validation pipeline
with Priyanka's controller and handing predictions, traces, and measured run evidence
to Member 2. Member 2 retains the protocol, authoritative scoring, and comparisons.
No official inference has been run by this handoff update.

The exporter uses the same agent/model/retrieval components. Every offline episode
starts with the exact original ordered B3 top 50; expansions append fresh valid
movies. It carries explicit agent state through simulated answers so its trace is
the complete episode. Live stateless replay, by comparison, logs per-request
segments; these are not complete evaluator traces. Service A still needs the
adoption of the versioned trace/state extension above to persist a complete live episode.

Before official inference obtain:

- For validation, the original ordered B3 file is already verified and staged at
  `outputs/predictions/incoming/member3-validation/b3.jsonl` (SHA-256
  `c388c0fd10bb895c1e0710ec10fa96a7d87785939f2498a5bf23533d18d000a9`).
  It passed the authoritative validator for all 300 frozen users. Test pools still
  need their separate handoff and authorization.
- Member 6's P2 adapter/tokenizer, P1 identity, training-sample version, and loading settings.
- A team-approved policy: simulator, prompt length, budgets, invalid-output/grounding
  handling, fallback rule, timeouts, and checkpoint-selection procedure.
- Test-specific authorization before any test inference.

Andrew's first-validation policy (recorded 2026-11-24) confirms two questions,
one expansion, 100 pool items, eight control events, zero extra invalid-output
retries, exact-B3-order fallback for exhaustion/earlier errors, and grounding
`flag`. The real configuration template now explicitly selects
`explanation_failure_policy: "template"`. Upstream's default stays `fail` for
backward compatibility; merge the new field into an existing ignored
`configs/service_b_local.json` on the GPU laptop without overwriting its local
paths or hardware settings. The complete run freeze still needs actual checkpoint
and index identities plus GPU-measured per-call/episode deadlines.
Member 1 has confirmed `pre_cutoff_history_sim_v1`; its source bytes match the v2
handoff receipt. All 22,802 train and 2,178 dev action prompts were reproduced
byte-for-byte by our current `p2-prompt-v1` renderer. The newer upstream renderer
file adds optional per-movie scoring helpers, but action-prompt parity is intact.
Member 6 has also confirmed the 6,144 input / 1,024 output token settings after a
real-tokenizer audit. The checkpoint and measured timeouts still need a run freeze.
`grounding_policy:flag` records unsupported explanations without repairing them;
it is not a guarantee of factual correctness. The competing `rank_available_pool`
budget policy is incompatible with the present official export contract. For live
requests the controller's `frozen_b3_order` fallback label refers to that request's
initial online retrieval order, not to a saved benchmark B3 pool.

Prepare an immutable freeze bundle with the actual files:

```bash
.venv-service-b/bin/python -m src.app.service_b.offline prepare \
  --config configs/evaluation_phase2_validation_draft_v1.json \
  --candidate-pool /path/to/original-b3-validation.jsonl \
  --service-config configs/service_b_local.json \
  --p1-checkpoint-id ACTUAL_P1_ID --p1-checkpoint-revision ACTUAL_P1_REVISION \
  --training-sample-version ACTUAL_SAMPLE_VERSION --hardware ACTUAL_GPU_HOST \
  --output outputs/predictions/p2-freeze-v1.json
```

Have the team record approval of that exact bundle in team storage. The run accepts
a JSON receipt containing `approved:true`, `approved_by`, `approved_at`, `split`,
and `freeze_sha256` (the exact bundle's hash). This is an explicit owner assertion,
not cryptographic proof that the team approved or when it approved. Keep the dated
team discussion as independent evidence. Source/config/checkpoint hashes bind the
bundle; changing any requires a new freeze.

```bash
.venv-service-b/bin/python -m src.app.service_b.offline run \
  --config configs/evaluation_phase2_validation_draft_v1.json \
  --candidate-pool /path/to/original-b3-validation.jsonl \
  --service-config configs/service_b_local.json \
  --freeze outputs/predictions/p2-freeze-v1.json \
  --approval-record /path/to/team-approval.json \
  --hourly-rate ACTUAL_USD_PER_HOUR --run-id p2-validation-v1 \
  --output-dir outputs/predictions/p2-validation-v1
```

For a nonofficial validation smoke, add `--smoke 3`; no approval receipt is needed,
and no official run manifest is published. Test-split smoke is blocked. Select
appropriate offline deadlines in a separate configuration before freezing.

Outputs: `p2.jsonl`, `p2_trace.jsonl`, `telemetry.jsonl`, `failures.jsonl`,
`explanation_fallback_summary.json`,
`status.json`, and, only for a complete validated official export,
`run_manifest.json` and `validation.json`. An untraceable failure still makes the
run incomplete; successful assessments are never fabricated. Input hashes and source snapshots are
rechecked. Simulator history is training-only for validation, training plus
validation for test, with cutoff and full-watched-set checks. Target bytes are
hashed for provenance; their labels are never parsed by this producer.

### Explanation failure under the first-validation policy

Priyanka's agent update at `211a27448addb616c354b1649372302c250194fc`
implements the opt-in rule accepted by Member 2 at
`bab49dc0f790d91d33d58743acb737537967a7e0`:

- If a final ranking is valid but explanation output is unusable (including a
  timeout), retain that ranking and generate all ten explanations from each
  movie's displayed catalog fields, with no inferred user preference.
- The explanation event carries `source: "catalog_metadata_fallback"` and
  `explanation_fallback_reason`. A model explanation carries `source: "model"`.
- Keep the original ranking termination: `recommend` remains `recommend`, and
  a B3-order ranking remains `fallback`. Explanation substitution is not a B3
  ranking fallback.
- Keep rejected-output SHA-256 and parser error in controller telemetry. Per-call
  audit additionally preserves complete exception type/message, raw-output digest,
  prompt digest, attempted call count, and measured latency.
- Parseable but unsupported model text is retained under `grounding_policy:flag`;
  it is not replaced by the template fallback.

`explanation_fallback_summary.json` records counts, reasons, and per-user sources
separately from ranking fallbacks. Its SHA-256 and the telemetry SHA-256 are bound
in the standard manifest's `owner_notes`; no incompatible keys were added to the
strict `p2-run-manifest-v1` episode/aggregate schema. Send these companion files
with the three main exports so Member 2 can audit explanation provenance.
If catalog evidence is missing or explicit `explanation_failure_policy:fail` is
selected for a different experiment, unrepairable episodes remain failures.

Old source/config freeze bundles no longer match this code; prepare a new bundle
after committing the changes and measuring the target GPU. The template's
20-second model call and 25-second request/episode settings remain development
values and are not approved export deadlines. Priyanka's controller calls and
trace fields were integrated without rewriting her implementation.

Simulator metadata now calls Member 1's exact `load_metadata` and `check_documents`
helpers. Raw metadata hashes, training-cutoff tag aggregation, and canonical
document parity are verified before answers are generated. It includes all
62,423 MovieLens movies, including 41,090 outside the retrieval catalog. These
movies can support answers about watched history but cannot become recommendations.
The real metadata reconstruction passed locally; regression tests specifically
cover an out-of-catalog liked movie and reject changed raw metadata. Set
`simulator_raw_dir` in the real runtime configuration if raw files live elsewhere.

The receipt uses Member 4's rank surrogate `1 - zero_based_rank / ranking_count`,
not a calibrated probability. Cost is hourly rate times measured episode work;
it excludes loading, idle reservation, and simulator time. A zero rate is a user
declaration of zero marginal cost, not an inferred price. Member 2 scores the
validated exports with the existing evaluator; this producer does not pick a
winning checkpoint or independently compute recommendation metrics.

## Tests and measured evidence

```bash
.venv-service-b/bin/python -m unittest discover -s tests -q
HF_HOME=data/indexes/huggingface .venv-service-b/bin/python -m src.app.service_b.benchmark \
  --manifest data/indexes/movie-index-v1.json \
  --output outputs/reports/member3-service-b/retrieval-latency-new.json
```

Tests cover real persistent Chroma with synthetic vectors, exclusion and expansion,
HTTP question/recommendation flow and cancellation, a real tiny locally generated
Qwen+LoRA checkpoint, process termination, complete simulated agent episodes,
and the full export/receipt path against Member 2's validator. Scripted models and
tiny random weights test plumbing, not trained-model recommendation quality.
Final regression result: **214 tests passed**. A separate local Uvicorn HTTP smoke
passed the ask/answer/recommendation flow. A real Chroma 1.3.5 native HTTP-server
smoke passed ingestion, query filtering, and the bounded-client configuration.
Additional tests cover signed full-episode continuation, trace-bearing failures,
bounded checkpoint recovery, index health, exact Member 5 HTTP-client compatibility,
and full-universe simulator metadata. The legacy and v1 endpoints also passed a
real local HTTP smoke with explicitly labelled demonstration resources.
Explanation-policy regressions additionally cover malformed/wrong-ID outputs,
timeouts, preserved expanded-pool ranking, separate B3/explanation fallback counts,
unchanged flagged text, full error/digest evidence, live HTTP behavior, and a
complete synthetic manifest accepted by Member 2's unchanged validator.

The real-catalog retrieval benchmark used 20 canonical validation profiles, one
warmup, concurrency one, on this ARM Mac CPU. Initial search p50/p95 was
**32.37 / 33.47 ms**; expansion was **24.48 / 25.69 ms**. Resource startup was
**2.69 seconds** with cached model files. These are small-sample retrieval-only
measurements, not GPU inference or end-to-end P2 latency. The generated JSON with
all timings and source hashes is under `outputs/reports/member3-service-b/` locally.

## Containers and team integration

```bash
docker compose -f compose.service-b.yml up --build
docker compose -f compose.service-b.yml --profile tools run --rm ingest
```

Compose starts mock Service B on 8001 and Chroma on host 8002; the container network
uses Chroma port 8000. The Compose index lives in a separate volume from the local
embedded index. Ingest there and use its `compose-movie-index-v1.json` receipt in
the real configuration. For real NVIDIA inference add
`-f compose.service-b.gpu.yml` and set `SERVICE_B_CONFIG=configs/service_b_local.json`.
The base image installs CPU PyTorch; the GPU override builds CUDA 12.8 PyTorch
and requests GPU access. Its host needs a compatible NVIDIA driver and Container
Toolkit. On a CPU-only host explicitly select CPU in the runtime configuration for
small smoke runs. PyTorch wheel sources follow the official
[versioned installation commands](https://pytorch.org/get-started/previous-versions/).
From Service A in another container, use a shared network and the service DNS name
or a host-reachable URL; `localhost` there is not Service B. Persist structured logs
with the application run. HTTP correlation IDs are response headers; the JSON
envelope remains compatible with the existing Service A client.

Docker is not installed on this machine, so Compose/Docker files were inspected
but not built or run locally. `.github/workflows/service-b.yml` runs the regression
suite, validates Compose, builds the CPU image, and smoke-tests its legacy and v1
HTTP endpoints after a branch push or relevant PR. The f66d683 implementation
passed that workflow, including the CPU image and container smoke:
[verified run](https://github.com/dan778912/movie-rec-agent/actions/runs/45112474963).
The current configuration/handoff changes have not yet had their own GitHub CI
run. No cloud deployment, GPU timings, or real P2 scoring is claimed.

## Imported teammate sources

These integration dependencies were copied without altering their implementation:

- Member 4 agent and its shared Member 1 helpers/tests: `63b40aad0bf3605b40737dbb24d8f1b3c2ba4868`.
  Explanation-policy agent files, action reference and decision notes were updated
  from `211a27448addb616c354b1649372302c250194fc`.
- Member 2 evaluator, synthetic fixtures, receipt tests, contracts and frozen-user configs:
  `9032fd13c62b7b18d3c53a2e3c685c53f4f49b55`.
  Policy/interface documents were refreshed from
  `bab49dc0f790d91d33d58743acb737537967a7e0`; evaluator implementation is unchanged.
- Member 1 configuration and handoff docs: `321c28443bca3e502ebcf393a079ba6527109027`.
- MiniLM settings referenced from Member 5's selected embedding config:
  `cd80dbf67e5505e8b1a26173e89830686face813`.
- Service A HTTP contract inspected at `853512100a7d4175ac8b3eff49f794d2a6e277db`.
  Its unchanged HTTP client is preserved as a compatibility-test fixture.

The evaluator and agent are vendored integration dependencies, not new Member 3
authorship. Future merges should preserve their owners' changes. Chroma APIs follow
the official [collection configuration](https://docs.trychroma.com/docs/collections/configure)
and [Python client](https://docs.trychroma.com/reference/python/client) documentation.
