# CLAUDE.md — Scope Load Testing Harness

## Project Overview

Automated load testing harness for Daydream Scope cloud inference on the Livepeer network. Lightweight Python service (no ML deps) that drives streams via the Daydream SDK service, validates output, and reports metrics.

```
loadtest CLI → Daydream SDK (sdk.daydream.monster) → Livepeer Orchestrator → Scope Runner (GPU)
                                                                              ↕ trickle events channel
  metrics → Prometheus push gateway → Grafana                                 ↕ telemetry (PR 1040)
  events  → POST /v1/metrics → Kafka → ClickHouse (network_events)
```

## Current State (2026-06-09)

### What's deployed and working

- **GCP VM** `loadtest-staging-1` (us-central1-a, IP: 104.197.233.56) — managed by Pulumi
- **Docker stack**: harness + pushgateway + prometheus + promtail + SDK
- **Scheduler**: runs 16 scenarios continuously (9 runs/day at 20% budget)
- **Metrics**: Prometheus push gateway (:9091) + Daydream `/v1/metrics` (staging)
- **Grafana**: dashboard at `eu-metrics-monitoring.livepeer.live` (import `dashboards/grafana/scope-loadtest.json`)

### Open issues / blockers

1. **Trickle telemetry not flowing yet** — `trickle_reader.py` is merged but needs orchestrators with PR daydreamlive/scope#1040 (`orch-prod-scope-1`, `orch-prod-scope-vast-1`). These are behind Cloudflare and unreachable on port 8935 from the GCP VM. Pulumi team needs to configure networking (origin IPs, firewall rules, or add to discovery JSON). The code is ready — just needs connectivity.

2. **v1/metrics → ClickHouse gap** — PR livepeer/pipelines#2691 deployed to staging (`pipelines-api-staging.fly.dev`), events accepted (HTTP 200), but staging Kafka doesn't have ClickPipes to ClickHouse. Production API (`pipelines-api.fly.dev`) doesn't have the route yet (PR not merged to main). Need PR 2691 merged to production, or ClickPipes configured for staging Kafka.

3. **Per-segment payment bug** — `WARNING No running event loop; per-segment payments not started` fires before every stream. The `livepeer-python-gateway` `start_scope()` runs in a worker thread without an event loop. Payments may not be sending, which could explain mid-stream drops at seq ~156 on orch-prod-1.

4. **Signer outages** — intermittent 503 from `signer.daydream.live` causes stream start failures. The harness handles this (classifies as `orchestrator` error) but it inflates failure rate.

### Recent test results (staging orchestrators)

| Scenario | Status | Connect | First Frame |
|----------|--------|---------|-------------|
| longlive_t2v_1m | PASS | 8-107s (warm/cold) | 0.4-1.6s |
| longlive_v2v_1m | PASS | 7-13s (warm) | 0.1-0.6s |
| longlive_v2v_5m | PASS | 13s | 0.2s |
| longlive_v2v_15m | PASS | 13s | 0.5s |
| ltx2_t2v_1m | PASS | 109s (cold) | 1.1s |

## Development Commands

```bash
# Local dev
pip install -e ".[dev]"
pytest                                     # 135+ unit tests
pytest tests/test_e2e_metrics.py -v        # e2e (needs DAYDREAM_API_KEY + METRICS_API_KEY)
loadtest scenarios                         # list all 16 scenarios
loadtest run --scenario longlive_v2v_1m    # single test run
loadtest schedule                          # start scheduler daemon
loadtest datasets                          # show dataset health
loadtest coverage                          # today's coverage
loadtest baselines                         # 7-day rolling baselines

# Docker
docker compose up -d                       # start scheduler + pushgateway
docker compose run --rm harness run --scenario longlive_v2v_1m
docker compose logs -f harness

# On the VM
gcloud compute ssh loadtest-staging-1-261b5c6 --zone=us-central1-a
cd /opt/scope-load-testing
sudo docker compose run --rm harness run --scenario longlive_t2v_1m
sudo docker logs scope-load-testing-harness-1 --tail 50
sudo docker logs sdk --tail 50
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DAYDREAM_API_KEY` | Yes | - | Daydream API key (for SDK stream start) |
| `SDK_URL` | No | `https://sdk.daydream.monster` | SDK service URL (VM uses `http://sdk:8000`) |
| `METRICS_URL` | No | `https://api.daydream.monster/v1/metrics` | Pipelines API metrics endpoint |
| `METRICS_API_KEY` | No | Falls back to `DAYDREAM_API_KEY` | Pipelines API key (separate auth system, `sk_ccY5...`) |
| `PUSHGATEWAY_URL` | No | `http://pushgateway:9091` | Prometheus push gateway |

### Two auth systems

| System | Keys | Used by |
|--------|------|---------|
| Daydream API (`api.daydream.live`) | `sk_fXw...`, `sk_MVU...` | SDK streams, signer |
| Pipelines API (`api.daydream.monster`) | `sk_ccY5...` (created at `app.daydream.monster`) | `/v1/metrics` endpoint |

## Architecture

### How the harness works (SDK path)

1. `POST sdk/stream/start` → gets `stream_id`, `events_url`, channel URLs
2. Starts `TrickleEventsReader` on `events_url` (background, reads telemetry)
3. Publishes input frames (v2v at 10fps, t2v keepalive at 30s interval)
4. Waits for first output frame via `GET sdk/stream/{id}/frame`
5. Monitoring loop: capture frames, validate quality, switch prompts, check stalls
6. Stops stream, collects trickle metrics, reports results

### Key modules

| Module | Purpose |
|--------|---------|
| `sdk_executor.py` | Full test lifecycle via SDK (connect→publish→monitor→stop) |
| `sdk_client.py` | Typed async HTTP client for SDK service |
| `trickle_reader.py` | Background reader for trickle events channel (telemetry from runner) |
| `metrics_reporter.py` | Reports events to `/v1/metrics` with `client_source: "scope-loadtest"` |
| `metrics.py` | Prometheus metrics (push gateway, `pushadd_to_gateway` for accumulation) |
| `scheduler.py` | Continuous scheduler (budget, round-robin scenarios, drift detection) |
| `scenarios.py` | Scenario matrix expansion from config |
| `datasets.py` | Random prompt pool rotation, synthetic video frame generation |
| `validators.py` | Frame quality (black/corrupt/size), prompt sensitivity (pixel diff) |
| `regression.py` | 7-day rolling baselines, drift detection (>20% = alert) |
| `coverage.py` | Daily per-orchestrator coverage tracking (JSON, pruned to 30 days) |
| `results.py` | RunResult, error taxonomy (network/orchestrator/runner/protocol) |
| `config.py` | YAML config loading, validation |
| `discovery.py` | Orchestrator discovery, health tracking, blacklist |

### Trickle events channel (PR daydreamlive/scope#1040)

When connected to orchestrators with PR 1040 deployed, the runner sends telemetry via trickle:
```
Runner → publish_event("stream_heartbeat") → TrickleEventsSink → trickle events channel
  → {"type": "telemetry", "event": {id, type: "stream_trace", data: {type: "stream_heartbeat", ...}}}
    → TrickleEventsReader reads HTTP segments at {events_url}/{seq}
      → TrickleMetrics (runner_ready, media_stats, telemetry_events)
```

**Orchestrators with PR 1040:** `orch-prod-scope-1.daydream.live`, `orch-prod-scope-vast-1.daydream.live`
**Without PR 1040:** `orch-prod-1.daydream.live`, `orch-prod-2.daydream.live` (only lifecycle events, no telemetry)

### Metrics reporting to ClickHouse

Events are sent to `POST /v1/metrics` (PR livepeer/pipelines#2691) with `client_source: "scope-loadtest"`:

```sql
-- ClickHouse query for load test events
SELECT * FROM network_events.network_events
WHERE JSONExtractString(data, 'client_source') = 'scope-loadtest'
ORDER BY timestamp DESC LIMIT 10
```

Event types: `loadtest_run_started`, `loadtest_run_completed` (includes timings, pass/fail, trickle data)

## Scenarios (16 total, config-driven)

Generated from `config/default.yaml` matrix. Adding a pipeline = one YAML entry, no code.

| Pipeline | Modes | Durations |
|----------|-------|-----------|
| longlive | t2v, v2v, i2v | 1m, 5m, 15m |
| ltx2 | t2v, i2v | 1m, 5m |
| longlive+rife | v2v | 5m, 15m |
| depth+longlive+rife | v2v | 5m |

## Datasets

11 prompt pools (220 unique prompts, 0 duplicates): nature, urban, abstract, stress, people, animals, weather, scifi, fantasy, food, motion. Each run picks a random pool and shuffles.

3 synthetic video styles for v2v/i2v: gradient, noise, blocks. Randomly selected per run.

Dataset management: `config/datasets/manifest.yaml` tracks all assets. Use the `enrich-datasets` skill (`.claude/skills/enrich-datasets.md`) to add more via storyboard MCP.

## VM Infrastructure

**VM:** `loadtest-staging-1-261b5c6` (us-central1-a) — managed by Pulumi (`simple-infra/staging` stack)

**Docker compose override:** `/opt/scope-load-testing/docker-compose.override.yml` — sets SDK config, orchestrator URLs, metrics keys. Pulumi manages this file.

**SDK container:** runs local `sdk.daydream.monster` equivalent, configured with:
- `LV2V_ORCH_URLS` — which orchestrators to use
- `SIGNER_URL` — payment signer
- `DISCOVERY_URL` — orchestrator discovery JSON

**Key files on VM:**
- `/opt/scope-load-testing/` — repo checkout
- `/opt/scope-load-testing/data/` — persistent (coverage.json, baselines.json, failures/)
- `/opt/scope-load-testing/docker-compose.override.yml` — Pulumi-managed env config

## Style Guidelines

- Python 3.12+, type hints, async-first (httpx.AsyncClient)
- No ML dependencies — httpx, prometheus_client, pyyaml, Pillow, numpy, click
- Config is YAML, no hardcoded values
- All SDK interactions through `sdk_client.py`
- All Scope API interactions through `scope_client.py`
- Errors classified by taxonomy before reporting
- `client_source: "scope-loadtest"` on ALL events (ClickHouse filtering)
- Adding a pipeline = config only, no code

## Related repos and PRs

| Repo | PR | Status | What |
|------|----|--------|------|
| `daydreamlive/scope` | #1040 | Deployed to scope-1/vast-1 | Trickle telemetry from runner |
| `livepeer/pipelines` | #2691 | Deployed to staging, NOT production | `/v1/metrics` endpoint |
| `livepeer/simple-infra` | #36 | Merged | VM provisioning via Pulumi |
