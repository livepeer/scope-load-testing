# Metrics Reporting to Daydream v1/metrics — Implementation Plan

## Goal

Report all load test results to the Daydream `POST /v1/metrics` endpoint as `network_events`, tagged as load testing traffic so they can be filtered in ClickHouse. Use the same `DAYDREAM_API_KEY` already configured.

## Context

- **API endpoint:** `POST https://api.daydream.monster/v1/metrics` (PR livepeer/pipelines#2691)
- **Auth:** `Authorization: Bearer <DAYDREAM_API_KEY>` (server stamps `user_id`, `api_key_name`)
- **Batch:** 1-500 events per request, max 1MiB, rate limit 30 req/min
- **Kafka topic:** `network_events` (same topic Scope uses)
- **Fail-loud:** 5xx on Kafka errors, client retries with backoff
- **Test traffic tagging:** `client_source: "scope-loadtest"` in event data (matching Scope's `client_source: "scope"` pattern from `kafka_publisher.py:80`)

## How test traffic is tagged

Scope uses `client_source: "scope"` in all events (`kafka_publisher.py:80`). We use `client_source: "scope-loadtest"` so ClickHouse queries can filter:
```sql
-- Production traffic only
WHERE data.client_source = 'scope'

-- Load test traffic only  
WHERE data.client_source = 'scope-loadtest'

-- Both
WHERE data.client_source IN ('scope', 'scope-loadtest')
```

## Event types emitted per test run

Each load test run produces 3 events:

1. **`loadtest_run_started`** — emitted when a test scenario begins
2. **`loadtest_run_heartbeat`** — emitted with timing/quality data after the stream phase
3. **`loadtest_run_completed`** — emitted when the run finishes (pass or fail)

## Event data schema

All events include these common fields in `data`:

```json
{
  "type": "loadtest_run_completed",
  "client_source": "scope-loadtest",
  "timestamp": "1717200000000",
  "session_id": "lt-uuid-per-run",
  "scenario": "longlive_v2v_1m",
  "pipeline": "longlive",
  "mode": "v2v",
  "duration_class": "short",
  "orchestrator_id": "auto"
}
```

**`loadtest_run_completed` adds:**
```json
{
  "passed": true,
  "connect_s": 13.2,
  "first_frame_s": 0.5,
  "stream_duration_s": 60.0,
  "total_s": 73.7,
  "cold_start": false,
  "frames_validated": 6,
  "frames_black": 0,
  "frames_corrupt": 0,
  "prompt_pool": "nature",
  "error_category": null,
  "error_message": null
}
```

## File structure

```
src/loadtest/
├── metrics_reporter.py   # NEW — HTTP client for /v1/metrics with retry/backoff
├── metrics.py            # MODIFY — add MetricsReporter integration alongside Prometheus
├── sdk_executor.py       # MODIFY — emit start/heartbeat events during run
├── scheduler.py          # MODIFY — wire MetricsReporter, emit completed events
├── config.py             # MODIFY — add metrics_url config field
tests/
├── test_metrics_reporter.py  # NEW — unit tests for reporter
├── test_e2e_metrics.py       # NEW — e2e test against preview endpoint
```

---

## Tasks

### Task 1: MetricsReporter client

**Files:** Create `src/loadtest/metrics_reporter.py`, `tests/test_metrics_reporter.py`

Build an async HTTP client that:
- Batches events in a buffer (max 500 per flush)
- Flushes on `flush()` call (not background — keep it simple)
- Retries 502/503/504 with exponential backoff (1s, 2s, 4s, max 60s)
- Drops batch on 400 (malformed, won't fix by retrying)
- Logs and pauses on 401 (bad key)
- Waits 60s on 429 (rate limited)
- Every event gets `client_source: "scope-loadtest"` injected
- Uses `app: "scope-loadtest"`, `host: hostname`

**Tests:**
- Mock httpx responses for 200, 400, 401, 429, 502
- Verify `client_source: "scope-loadtest"` is injected into all events
- Verify batch size limit (500 max)
- Verify retry logic on 502 (re-enqueue, backoff)
- Verify drop on 400

### Task 2: Wire MetricsReporter into results flow

**Files:** Modify `src/loadtest/scheduler.py`, `src/loadtest/cli.py`

- Create `MetricsReporter` in scheduler if `DAYDREAM_API_KEY` is set
- After each run, build events from `RunResult` and call `reporter.enqueue()` + `reporter.flush()`
- Same for CLI `run` command
- Config: `METRICS_URL` env var, defaults to `https://api.daydream.monster/v1/metrics`

### Task 3: Build events from RunResult

**Files:** Add method to `src/loadtest/metrics_reporter.py`

`build_events(result: RunResult, prompt_pool: str) -> list[dict]` that produces the 3 event types:
- `loadtest_run_started`: session_id, scenario, pipeline, mode, timestamp
- `loadtest_run_heartbeat`: timing data, frame validation counts
- `loadtest_run_completed`: pass/fail, error info, full timing breakdown

### Task 4: Config update

**Files:** Modify `src/loadtest/config.py`, `config/default.yaml`, `.env.example`

- Add `metrics_url` to config (default: `https://api.daydream.monster/v1/metrics`)
- Add `METRICS_URL` to `.env.example`

### Task 5: E2E test against preview endpoint

**Files:** Create `tests/test_e2e_metrics.py`

- Skip if `DAYDREAM_API_KEY` not set (CI-friendly)
- Send a test event batch to the metrics endpoint
- Verify 200 response with `{"status": "accepted", "accepted": N}`
- Verify `client_source: "scope-loadtest"` is in the payload
- If ClickHouse MCP is available, query to verify event landed

### Task 6: ClickHouse verification (manual + documented)

- Document the ClickHouse query to verify load test events
- Add a CLI command or script to check: `loadtest verify-metrics`

---

## Verification checklist

- [ ] `client_source: "scope-loadtest"` in every event (filterable in ClickHouse)
- [ ] Events land in `network_events` Kafka topic
- [ ] Events visible in ClickHouse with correct schema
- [ ] Unit tests pass (mock HTTP, retry logic, event building)
- [ ] E2e test passes against live endpoint
- [ ] Scheduler emits events for every run (pass and fail)
- [ ] CLI `run` command emits events
- [ ] No impact on existing Prometheus metrics (additive, not replacing)
