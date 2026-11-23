# Service A scalability tests

Run only against a disposable local database. These workloads create users,
sessions, and recommendations. They do not evaluate recommendation quality.

## Workloads and interpretation

- `MixedUser`: search versus full-session tasks are selected with weights 5:2.
  This is a task ratio, not a 5:2 ratio of individual HTTP requests.
- `SearchUser`: title search only, using five fixed query terms.
- `SessionUser`: create a fresh session with one rating, advance the turn,
  answer the mock's clarification, read recommendations, and read history.

Every HTTP request is followed by an independent 0.5-1.5 second think time.
A virtual user is not an in-flight request. With mean think time of one second,
20 users can issue approximately 20 requests/second before service latency;
the historical note of 38 requests/second at 20 users is not explained by
these stated pacing rules. The historical Locust script/logs are unavailable.
Do not silently reinterpret those numbers as measurements from this suite.

## Setup

Use the application environment from `requirements-app.txt` and a separate
load-generator environment with `requirements-load.txt`. Start a dedicated
PostgreSQL 16 container and seed the frozen 21,333-row movie catalog. Use the
same catalog and runtime for both variants. The load tests need no labels,
model downloads, or GPU.

Example isolated database (local-only example credentials):

```bash
docker run -d --name movie-rec-scalability -p 127.0.0.1:55432:5432 \
  -e POSTGRES_USER=bench -e POSTGRES_PASSWORD=bench-local-only \
  -e POSTGRES_DB=movie_rec_bench postgres:16
```

Seed the isolated schema and catalog with:

```bash
export BENCH_DATABASE_URL=postgresql://bench:bench-local-only@127.0.0.1:55432/movie_rec_bench
/path/to/app-python -m scripts.scalability.seed --documents /path/to/movie_documents.jsonl
```

The seed command uses `Base.metadata.create_all` for schema-equivalent load fixtures. The recorded
run used `create_all`, so it is not migration validation. Each matrix run clears
only the dedicated database's session/user tables before starting, while keeping
the catalog. Never point this runner at an existing development or production DB.

```bash
export BENCH_DATABASE_URL=postgresql://bench:bench-local-only@127.0.0.1:55432/movie_rec_bench
/path/to/app-python scripts/scalability/run_matrix.py \
  --output /absolute/path/to/new-results-directory \
  --locust /absolute/path/to/load-env/bin/locust \
  --app-python /absolute/path/to/app-env/bin/python \
  --workload MixedUser --limit 12 --workers 1
```

The runner uses port 8027, 20/50/100/150/300 users, a spawn rate of 100 users/s,
and 60 seconds per stage including ramp-up, followed by up to six seconds for
user shutdown. It samples PostgreSQL activity every 0.5 seconds, saves each
stage's CSV/HTML/Locust log and the server log, and probes recovery between
stages. A failed recovery probe stops further escalation. `--limit 0` disables
admission protection for a baseline; it is not the recommended serving setting.
Use a fresh output directory for every run. Compare success throughput and
failure rates, not just the aggregate request rate, because 503 responses also
count as requests. Locust percentile buckets are approximate and include failed
requests; use endpoint rows rather than calling the aggregate a model latency.

For a raw command without the matrix runner:

```bash
locust -f tests/load/locustfile.py MixedUser --headless \
  --host http://127.0.0.1:8027 --users 100 --spawn-rate 100 \
  --run-time 60s --stop-timeout 6 --csv /tmp/mixed-100 \
  --csv-full-history --html /tmp/mixed-100.html
```

## Serving safeguards

`SERVICE_A_MAX_IN_FLIGHT` defaults to 12 per worker. Extra HTTP requests receive
503 with `Retry-After: 1` before they enter synchronous database dependencies.
Slots cover the full ASGI call, including response/dependency cleanup; exceptions
and cancellations return the slot. This is admission control, not a global
rate limiter, fairness policy, queue, or guarantee of minimum throughput.

The pool remains SQLAlchemy's default five connections plus ten overflow.
Increasing workers multiplies both admission and possible connection use:
four workers can use up to 60 database connections, plus monitoring and other
clients. Keep the total below PostgreSQL's connection budget and account for
other replicas before deployment. Do not size workers solely from this local run.

Before a Service B call, Service A converts the read context to plain values and
closes the read transaction, returning its connection. It then reuses the Session
for persistence. This does not add mutation idempotency, same-session locking,
or atomicity across the existing multi-commit write sequence.

Mock scalability, a slow HTTP fixture, and real-model throughput are distinct.
A healthy mock run does not establish real Service B latency or GPU capacity.
