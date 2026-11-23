# Service A scalability evidence (2026-11-22)

Read the [experiment report](../../member5-scalability-results.md) and
[reproduction instructions](../../../tests/load/README.md) first.

This is a compact evidence bundle from the original run, not a new benchmark.
CSV line endings are normalized from CRLF to LF for Git; data values are unchanged.
It contains:

- `summary.csv` / `summary.json`: cross-scenario aggregates.
- Each scenario's `*-users_stats.csv`: original Locust endpoint and aggregate
  statistics, including request counts, failures, rates and percentile buckets.
- `*-users_failures.csv`: original failure categories and counts.
- `environment.json`: measured application source hashes and runtime settings
  (the original baseline uses the separate `baseline-source.json`).
- `stages.json`: recovery probes and load-generator exit codes. A zero generator
  exit code does not mean zero HTTP failures.
- `connection-probe.json`: checked-out connections around the slow HTTP fixture.
- `host.json` and `catalog-provenance.json`: host/runtime and input provenance.
- `unit-tests.txt`: original test run; `prepush-unit-tests.txt`: publication check.
- `SHA256SUMS.json`: hashes of evidence files, excluding this README and itself.

To verify the resume-related throughput number, open
`fixed-mixed/300-users_stats.csv`, select `Aggregated`, and subtract
`Failures/s` from `Requests/s`: 261.656207 - 71.224513 = 190.431694 successful
requests/s. Of 16,932 requests, 4,609 were rejected with HTTP 503 (27.2%).

For search latency, `fixed-search/100-users_stats.csv` reports 22 ms at the
95th percentile, with 133 failures out of 5,918 requests. Percentiles include
failures, so this is not a successful-only latency measurement.

These are mock-backed, single-trial, same-host results. No real-model latency,
production capacity, zero-failure capacity, or linear scaling is established.

Full HTML reports, server logs and sampled PostgreSQL telemetry remain in the
local archive referenced in the experiment report. Consequently, the compact
bundle supports direct audit of throughput, latency, failures and recovery,
but not independent recomputation of telemetry-derived maxima. Local absolute
paths in original manifests are provenance, not portable reproduction paths.

The original application source hashes can be compared with this branch's
files. The manifests' `dirty: true` records that measurements preceded commit;
it does not mean the benchmark ran on an unidentified source version.
