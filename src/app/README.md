# Application — Member 5 (Service A, PostgreSQL, frontend)

Per `worksplit_p2.md` (the team's final Phase 2 split), Member 5 owns:

- PostgreSQL schema, migrations, and the data-access layer (`db/`)
- Service A, the FastAPI business service (`service_a/`)
- The React + TypeScript conversational frontend (`frontend/`)

Service B — the GPU model service with the checkpoint (Member 6), agent
module (Member 4), and retrieval (Member 3) — does not exist yet. Service A
is built against a `MockServiceBClient` so it is runnable and testable today;
see "Mock vs. real Service B" below for the swap point.

**Status and full narrative:** `docs/member5-phase2-status.md` (what's real
vs. placeholder here, what's on the other members' unmerged branches as of
its last-updated date, and what's still open). This README stays focused on
how to run things; that doc covers why things are the way they are.

## Layout

```
src/app/
├── db/                  SQLAlchemy models, Alembic migrations, repository.py (DAL), seed_catalog.py
├── service_a/           FastAPI app: routes, schemas, profile building, Service B client
├── frontend/             React + TypeScript chat UI (Vite)
├── docker-compose.yml    Local dev Postgres only — not a deployment decision
└── .env.example           Copy to .env for Service A / Alembic
```

## Run it locally

### 1. Postgres (dev only)

```bash
cd src/app
docker compose up -d
cp .env.example .env   # edit if you changed the docker-compose credentials
```

### 2. Python environment

Needs Python >=3.12 (same requirement as the Member 5 embedding work), kept
in its own `requirements-app.txt` so it doesn't collide with
`requirements-embedding.txt`.

```bash
python3.12 -m venv .venv
./.venv/bin/pip install -r requirements-app.txt
```

### 3. Migrate and seed the catalog

Run from the repo root (`alembic.ini` lives there):

```bash
export DATABASE_URL=postgresql+psycopg://movierec:movierec@localhost:5432/movierec
./.venv/bin/alembic upgrade head
./.venv/bin/python -m src.app.db.seed_catalog   # ~60-90s, reads train_ratings.csv (20M rows)
```

`seed_catalog.py` expects `data/processed/ml25m-global-v1/movie_documents.jsonl`
and `train_ratings.csv` to exist (Member 1's frozen output, or a symlink to it —
`data/processed/` is git-ignored, so each teammate points it at their own
copy/symlink). It also aggregates a real average rating and rating count per
movie from `train_ratings.csv` (training split only, ~20.2M rows), which is
why the first run takes roughly a minute.

### 4. Service A

```bash
./.venv/bin/uvicorn src.app.service_a.main:app --reload --port 8000
```

Env vars (see `.env.example`): `DATABASE_URL`, `SERVICE_B_MODE` (`mock` by
default, `http` once Member 3's Service B exists), `SERVICE_B_BASE_URL`,
`FRONTEND_ORIGIN` (CORS, defaults to the Vite dev server origin).

### 5. Frontend

```bash
cd src/app/frontend
npm install
cp .env.example .env   # VITE_API_BASE_URL, defaults to http://localhost:8000
npm run dev
```

## API (Service A)

| Method & path | Purpose |
|---|---|
| `GET /movies/search?q=` | Title search (search-as-you-type picker), optional `limit` |
| `POST /sessions` | Start a session from imported ratings |
| `GET /sessions/{id}` | Session status |
| `POST /sessions/{id}/turn` | Advance one agent turn — returns an `ask` or `recommend` action |
| `POST /sessions/{id}/clarifications` | Submit an answer; immediately advances to the next turn |
| `GET /sessions/{id}/recommendations` | Latest recommendation for a session |
| `GET /users/{id}/history` | Past sessions + their latest recommendation |

Request/response shapes: `service_a/schemas.py`.

## Mock vs. real Service B

`service_a/service_b_client.py` defines the contract Service A codes
against (`ServiceBClient.decide(profile_text, history_movie_ids,
clarifications) -> {"action": "ask", "question": ...} | {"action":
"recommend", "items": [...]}`). `MockServiceBClient` is a deterministic
stand-in (asks once if watched history looks thin, otherwise returns
candidates in plain catalog order — **not** a real retrieval/ranking
result). `HttpServiceBClient` POSTs the same contract to
`SERVICE_B_BASE_URL/decide`; switch to it with `SERVICE_B_MODE=http` once
Member 3 ships the real service. Share this file with Member 3 so the real
Service B matches the shape Service A already calls.

## Provisional / not yet team-agreed

- `AgentActionTrace.action` / `.payload` (`db/models.py`) have no fixed shape
  yet — Member 4 owns the real agent action contract.
- `MockServiceBClient`'s candidate ordering and ask/recommend rule are
  placeholders, not retrieval or agent logic.

## Tests

```bash
python -m unittest discover -s tests -v
```

`tests/test_app_db.py` and `tests/test_service_a.py` run against in-memory
SQLite — no Docker/Postgres needed to run the suite.
