# Member 5 Phase 2 status: PostgreSQL, Service A, frontend

Last updated: 2026-11-12. Branch: `member5-service-a` (branched off `main`,
not merged). Tag: `service-a-v1.0`. This file is the narrative status doc;
`src/app/README.md` has the operational "how to run it" steps.

## Scope

Per `worksplit_p2.md` (the team's final Phase 2 division of responsibilities),
Member 5 owns three deliverables, all under `src/app/`:

1. PostgreSQL schema, migrations, and the data-access layer (`src/app/db/`)
2. Service A, the FastAPI business service (`src/app/service_a/`)
3. The React + TypeScript conversational frontend (`src/app/frontend/`)

## What exists and is verified working

All three pieces are built and have been run end to end together locally:
rate movies -> agent turn (ask or recommend) -> answer a clarifying question
-> recommendation with justification -> history. 16 backend tests pass
(`python -m unittest discover -s tests`, no Docker/Postgres needed - they
use in-memory SQLite). The frontend builds and lints clean
(`npm run build`, `npm run lint` in `src/app/frontend/`).

- **Database**: 8 tables (`movies`, `users`, `sessions`, `watched_items`,
  `clarification_exchanges`, `recommendations`, `recommendation_items`,
  `agent_action_traces`), 3 Alembic migrations, a `repository.py`
  data-access layer that Service A's routes call exclusively (no raw
  SQL/ORM in route handlers). `movies` is seeded from Member 1's frozen
  `movie_documents.jsonl` plus a real aggregate (average rating, rating
  count) computed from Member 1's `train_ratings.csv` (training split
  only, ~20.2M rows) - genuine MovieLens data, not synthetic.
- **Service A**: FastAPI routes for session creation, turn advancement,
  clarification answers, recommendation lookup, history, and title search.
  Profile text is built with `src/profiles/text.py`'s `user_profile()` -
  the exact same formatter Member 1's offline pipeline uses - so online and
  offline profile text stay comparable.
- **Frontend**: Vite + React 19 + TypeScript. Three views (rate, chat,
  history). Went through several visual-design iterations at the user's
  request, landing on an editorial list layout (large bold clickable
  titles, a meta column, a tag, click-to-expand detail) styled after
  a24films.com, replacing an earlier poster-color-block version that was
  explicitly rolled back.

## What's still placeholder, not real

- **There is no real Service B.** Service A talks to it through a
  `ServiceBClient` protocol (`src/app/service_a/service_b_client.py`).
  `MockServiceBClient` is the only implementation wired up today: it asks
  one canned clarifying question if watched history is thin, otherwise
  returns catalog movies in plain `movie_id` order with a templated
  "explanation" string. None of it is real retrieval, ranking, or
  generation. `HttpServiceBClient` exists and POSTs the same contract to
  `SERVICE_B_BASE_URL/decide`, ready to swap in via `SERVICE_B_MODE=http`
  the moment a real Service B exists - but nothing currently implements
  that endpoint.
- **`AgentActionTrace.action`/`.payload`** (`src/app/db/models.py`) is an
  intentionally loose string + JSON blob. No team-agreed agent action
  contract exists yet (that is Member 4's deliverable under
  `worksplit_p2.md`); this schema is a placeholder that can absorb the
  real contract later without a migration that drops data.
- **No cast/crew, poster images, or streaming-platform data anywhere.**
  MovieLens 25M does not include any of these. By explicit user decision,
  nothing here calls an external source (e.g. TMDB) to fill that gap -
  average rating, rating count, and genres are the only descriptive movie
  metadata beyond title that actually exists in this stack.
- **Search is plain SQL `ILIKE` substring matching**, not fuzzy/typo-
  tolerant. Closing that gap would mean a Postgres trigram index or
  routing through Member 3's eventual embedding retrieval - neither is
  done.
- No CI (`.github/workflows` doesn't exist). Ownership of this is actually
  unclear: the old pre-Phase-2 roadmap assigned it to Member 4, but the
  final `worksplit_p2.md` split doesn't mention it at all.
- No deployment (Docker/Terraform/AWS vs. a local-only demo is still an
  open question for the TA, per earlier team discussion - not resolved by
  anything in this branch).

## Push / branch state

- `member5-service-a` and tag `service-a-v1.0` are pushed to
  `origin` (github.com/dan778912/movie-rec-agent). `main` has **not**
  changed - it is still just the initial repo scaffold.
- No pull request opened for this branch yet. The user wants a code
  reviewer to look at it first.
- This work was developed in a separate git worktree isolated from the
  original Phase 1 embedding-experiments checkout, specifically so neither
  piece of work could interfere with the other's uncommitted state. That
  detail doesn't matter for anyone working from a fresh clone - it's just
  a local development-environment note.

## Team-wide state as of 2026-11-12 (important: none of this is on `main`)

While this work was in progress, four other branches appeared on `origin`,
all still unmerged:

| Branch | What it contains |
|---|---|
| `member2-evaluation` | The authoritative evaluator, plus a **published result**: on an authorized 500-user test subset, P1 (fine-tuned Qwen reranker) beats B3 (embedding retrieval) - hit_rate@10 +0.090, NDCG@10 +0.024, recall@10 +0.0067, recall@20 +0.0066, all with paired-bootstrap 95% CIs excluding zero (recall@50 ties, expected since P1 reorders B3's frozen pool rather than changing its membership). This is Phase 1's core question ("does LLM reranking help over embedding retrieval?") answered **yes**, for this subset. |
| `member-6-reranking` | A real P1 reranker: Qwen2.5-1.5B-Instruct fine-tuned with LoRA, with a scoring/explanation inference pipeline (`qwen_scorer.py`, `run_reranker.py`). This is a genuine trained model, not a placeholder. |
| `member-4-bm25` | BM25 keyword retrieval (B2) implemented and validated. |
| `member-3-training-data-popularity` | Popularity baseline (B0), P1 training-data regeneration, and handoff notes to Member 6. **Note:** this branch's content is Phase-1-shaped work (finishing the original six modeling workloads), not the Phase 2 "online retrieval module + Service B" role `worksplit_p2.md` assigns to Member 3. That Phase 2 work does not appear to exist on any branch yet. |

**Implication for this Service A work:** the ingredients a real Service B
would need (a trained P1 reranker, an evaluator, BM25) now exist as
separate Phase 1 pipeline code - but no one has built the actual Service B
HTTP service that would combine them, host the checkpoint, and run the
agent loop. `MockServiceBClient` is still the only thing Service A can
call today. Whoever picks up Member 3's Service B work next has real
ingredients to integrate, not a blank slate - but integrating them is not
done and is not something this branch attempted.

## Suggested next steps for whoever picks this up

1. Get a code reviewer to look at `member5-service-a` before opening a PR.
2. Watch for Member 3 starting actual Service B work (combining the
   checkpoint above, Member 4's agent module, and retrieval) - that's the
   point `SERVICE_B_MODE=http` becomes real instead of theoretical.
3. Nudge on the still-unowned CI question and the agent action contract
   (Member 4) - both are blockers for turning the placeholders in this
   branch into real integration points.
4. If Phase 1 branches (`member2-evaluation`, `member-6-reranking`,
   `member-4-bm25`, `member-3-training-data-popularity`) merge to `main`
   before this branch does, rebase rather than assuming `main` is still
   just the scaffold described above.

## Frontend refresh reviewed for branch publication (2026-11-12)

The working tree now includes a simplified first screen, responsive editorial
recommendation lists, keyboard-accessible movie search, and explicit mock-mode
labeling. Rating and clarification drafts survive navigation. Mutating requests
were moved out of mount effects to avoid repeating turns under StrictMode or
when revisiting the conversation. Search rejects stale responses and offers
retry; requests have a 45-second deadline and clarification recovery checks
session status before attempting another write.

14 frontend regression tests and 4 Chromium browser tests pass, along with the
production build and lint. Browser checks use an isolated SQLite fixture and
mock Service B, not the real model or production PostgreSQL. Desktop and mobile
screenshots were inspected. Full details and known recovery limitations are in
[the frontend review handoff](member5-frontend-review.md).

Independent review found one P2 issue: failed session creation could block an
existing conversation. The fix separates creation errors from conversation
recovery state and adds unit and browser regression coverage. The user authorized
committing and pushing this refresh to `member5-service-a`. The `service-a-v1.0`
tag still identifies the earlier baseline; this refresh does not move that tag,
merge into `main`, or open a new PR.
