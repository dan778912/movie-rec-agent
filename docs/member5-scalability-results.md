# Member 5: Service A scalability and overload protection

## Scope and provenance

These are local, validation-only engineering measurements with an in-process mock
Service B. They do not measure recommendation quality, model inference, GPU
capacity, or production capacity. No held-out labels or model jobs were used.

The historical Notion summary refers to commit `8535121`, but its original
Locust script and logs are unavailable. Its reported 38 requests/s at 20 users
cannot be reconciled with a one-second mean wait after every request. This run
implements the supplied per-request pacing explicitly; it does not reproduce or
silently replace that historical table.

The new baseline is main commit
`737f83d8b2e439ea27d9c8a956458114e7f37195`. Its application source matches
`member5-service-a` commit `6fe8ab32e35da0dabd08dde5cd2477395f33511f`.
The changes under test were on branch `member5-scalability`. Each fixed-run
manifest records application source hashes because the source was uncommitted
at measurement time. These hashes identify the measured implementation.

## Method

- Host: Apple M2 (user-reported), macOS arm64; client, app and Docker share the host.
- PostgreSQL 16 in a dedicated local container, with a 21,333-movie frozen catalog.
  Schema fixtures use `create_all`, so this is not a migration test. Rating
  aggregates and tags use schema defaults; movie IDs, titles, genres and text
  come from the canonical movie documents.
- SQLAlchemy pool: five persistent connections plus ten overflow, per worker.
- Locust 2.47.0; 20, 50, 100, 150 and 300 virtual users; 100 users/s spawn rate.
  Each stage lasts 60 seconds including ramp-up, plus up to six seconds of drain.
- Each HTTP request is followed by a uniform 0.5-1.5 second wait. Client timeout
  is five seconds. A virtual user is not an in-flight HTTP request.
- Mixed workload chooses search versus a full session with task weights 5:2.
  A full session creates a session, submits a turn, answers the mock's question,
  reads recommendations and reads history. The weights are not request counts.
- Separate search-only and session-only workloads expose endpoint costs. A
  four-worker mixed comparison is limited to 150 and 300 users.
- Each matrix starts with empty user/session tables, retains the movie catalog,
  and saves CSV, HTML, server logs and PostgreSQL activity sampled every 0.5 s.
  A failed five-second recovery probe stops further escalation.

These are single trials on a shared laptop, not repeated confidence-interval
estimates. Stages include startup bursts and tables grow between stages. Some
baseline work overlapped development checks. Aggregate percentile buckets are
approximate and include failures; they are not successful-only model latency.

## Implementation and evidence

`AdmissionLimitMiddleware` bounds active HTTP requests to 12 per worker, before
synchronous dependencies consume threads or connections. Excess requests receive
503 and `Retry-After: 1`. Slots cover response delivery and dependency cleanup,
and are returned on exceptions and cancellation. The pool size is unchanged.

Before calling Service B, `_advance_turn` converts the read context to plain
values and closes its read transaction, returning the connection. It reuses the
Session for persistence after the call. This does not introduce idempotency,
same-session locking, or atomicity across existing multi-commit writes.

The unprotected baseline exhausted its 15-connection pool at 100 users. Logs
contain SQLAlchemy QueuePool timeout errors; telemetry shows transactions idle
while connections remain occupied. Its recovery request timed out. Baseline
150 and 300 were therefore not run. These observations demonstrate pool
saturation; a thread-pool deadlock remains a hypothesis, not a proven diagnosis.

The fixed mixed run recovered successfully at every level through 300 users.
The two changes were tested together; this is not a performance ablation that
isolates their individual contributions. Rejected requests must not be counted
as successful throughput. The changes
support bounded overload handling under the tested conditions; they do not
justify the claim that a production deadlock has been permanently fixed.

## Measured results

Successful throughput excludes every failed request. Percentiles include all
requests, including 503 responses; no successful-only latency claim is made.

### Mixed workload: before and after

| Scenario | Users | Success req/s | 503 / requests | Other failures | p95 (ms) | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| baseline-mixed | 20 | 18.49 | 0/1197 | 0 | 80 | 200 |
| baseline-mixed | 50 | 46.76 | 0/2981 | 0 | 170 | 200 |
| baseline-mixed | 100 | 0.35 | 0/1052 | 1029 | 5000 | Timed out |
| fixed-mixed | 20 | 18.94 | 8/1203 | 0 | 63 | 200 |
| fixed-mixed | 50 | 46.59 | 65/2990 | 0 | 74 | 200 |
| fixed-mixed | 100 | 88.20 | 264/5999 | 0 | 83 | 200 |
| fixed-mixed | 150 | 124.18 | 859/8863 | 0 | 98 | 200 |
| fixed-mixed | 300 | 190.43 | 4609/16932 | 0 | 100 | 200 |

### Isolated workloads

| Scenario | Users | Success req/s | 503 / requests | Other failures | p95 (ms) | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed-search | 20 | 18.95 | 14/1145 | 0 | 170 | 200 |
| fixed-search | 50 | 48.70 | 42/2956 | 0 | 22 | 200 |
| fixed-search | 100 | 96.64 | 133/5918 | 0 | 22 | 200 |
| fixed-search | 150 | 144.91 | 167/8840 | 0 | 24 | 200 |
| fixed-search | 300 | 274.64 | 1112/17554 | 0 | 36 | 200 |
| fixed-session | 20 | 19.20 | 8/1238 | 0 | 62 | 200 |
| fixed-session | 50 | 46.75 | 41/3009 | 0 | 53 | 200 |
| fixed-session | 100 | 89.92 | 306/6115 | 0 | 87 | 200 |
| fixed-session | 150 | 134.07 | 460/9078 | 0 | 61 | 200 |
| fixed-session | 300 | 210.72 | 4010/17703 | 0 | 77 | 200 |

### Four-worker comparison

| Scenario | Users | Success req/s | 503 / requests | Other failures | p95 (ms) | Recovery |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fixed-mixed-4workers | 150 | 123.24 | 596/8519 | 0 | 330 | 200 |
| fixed-mixed-4workers | 300 | 203.05 | 3019/16238 | 0 | 370 | 200 |

Baseline 150/300 were skipped after the 100-user recovery failure. All 17 fixed
stages returned 200 on recovery; all recorded fixed-run request failures were
503 rejections, with no other failure categories recorded. The baseline's 1,029
failures were transport failures reported by Locust as HTTP 0; the server log
independently contains SQLAlchemy QueuePool timeout errors.

At 300 mixed users, four workers improved successful throughput by only 6.6%
(190.43 to 203.05 req/s), while aggregate p95 increased from 100 to 370 ms. The
503 fraction fell from 27.2% to 18.6%. At 150 users, success throughput did not
improve (124.18 versus 123.24 req/s). This does not establish linear horizontal
scaling; the measurements do not isolate CPU, DB, or load-generator contention.

The baseline's maximum observed idle transaction age was 40.704 seconds at 100
users, with 15 occupied connections. The fixed mixed run's maximum idle
transaction age across all levels was 1.127 seconds. At 300 users, sampled total
application connections peaked at 12 with one worker and 48 with four workers.
These are sampled maxima, not an exhaustive trace of every transaction.

### Endpoint example and connection-lifetime probe

At 100 users in the session-only run, p95 was 100 ms for POST turn, 190 ms for
POST clarifications, and 53 ms for GET recommendations. These are mock-backed
HTTP timings, and include failures. See endpoint CSV rows for full counts.

A separate real HTTP fixture waits 250 ms before returning an ask response.
Immediately before that HTTP wait, the baseline had one checked-out connection
and one idle-in-transaction backend; the fixed handler had zero checked-out
connections and one idle backend. Observed HTTP durations were 282 and 261 ms.
This checks release of the read transaction independently from load shedding;
it is not a load test of real Service B. See `connection-probe.json`.

### Verification

- Full Python unittest discovery: **43 passed**, including five new regressions.
- The connection-lifetime regressions failed against the original behavior
  before the production fix was applied.
- New admission tests cover overload before downstream execution, recovery,
  cancellation, and exception cleanup.
- All 17 fixed load stages and the PostgreSQL/HTTP connection probe completed.
- `git diff --check` and a Han-character scan of changed/new text files passed.
- No frontend changes were made; browser tests were not repeated for this change.

A defensible description is: "Added per-worker overload protection and shortened
DB connection lifetime; validated recovery through 300 concurrent simulated
users using reproducible mock-backed load tests." Do not say "fixed a deadlock",
"zero failures at 300 users", or "fourfold scaling" from these results.

## Reproduction and artifacts

See [load-test instructions](../tests/load/README.md) for isolated database setup,
seeding and exact workload commands. Run `SearchUser` and `SessionUser` separately;
use `--workers 4 --levels 150,300` for the horizontal-scaling comparison.

The [checked-in evidence bundle](benchmarks/scalability-2026-11-22/README.md)
contains original endpoint statistics and failure CSVs, run configurations,
recovery probes, source hashes, test logs and the aggregate tables.

The full raw local results are saved under:

```text
/Users/hxh/Documents/Codex/2026-10-01/woxi/outputs/scalability-2026-11-22/
```

Each scenario contains `environment.json`, `stages.json`, `service-a.log`,
`pg-stat-activity.jsonl`, and per-level Locust CSV, HTML and log files. The baseline
source hashes are in `baseline-source.json`; full passing test output is in
`unit-tests.txt`. Full server logs, HTML reports and database telemetry remain
local; the compact evidence bundle above is available in GitHub.

Generate the cross-scenario table after all matrices finish:

```bash
python scripts/scalability/summarize.py /absolute/path/to/scalability-2026-11-22
```

## Remaining limitations

Real Service B latency, downstream timeouts under load, GPU contention and
production data distribution still require integration testing. More workers
multiply pool capacity: four workers can open up to 60 application connections,
plus observers and other clients. Deployment settings must respect the total
PostgreSQL connection budget. There is no guarantee of fair admission, a global
rate limit, automatic retry, or completion of workflows interrupted by a 503.
