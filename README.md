# Scope Load Testing

Automated load testing harness for [Daydream Scope](https://github.com/daydreamlive/scope) cloud inference on the Livepeer orchestrator network.

## What It Does

This harness continuously validates that Scope's cloud inference pipeline works correctly across all Livepeer orchestrators. It:

- **Runs real inference sessions** through Scope's HTTP API — connect to cloud, load pipelines, stream video, validate output frames
- **Covers all major pipelines** — longlive, ltx2, chained graphs (longlive+rife, depth+longlive+rife)
- **Tests all modes** — text-to-video (t2v), video-to-video (v2v), image-to-video (i2v) at short (1m), mid (5m), and long (15m) durations
- **Rotates fairly across orchestrators** — discovers all available orchestrators, distributes test load evenly, tracks coverage
- **Detects regressions** — rolling 7-day baselines for latency and FPS, alerts on >20% degradation
- **Reports to Grafana** — real-time dashboards via Prometheus push gateway

## Architecture

```
┌─────────────────────── Docker Compose ───────────────────────┐
│                                                               │
│  scope-1 (daydream-scope :8001) ─┐                           │
│  scope-2 (daydream-scope :8002) ─┤── Livepeer Orchestrators  │
│                                   │                           │
│  loadtest-harness ────────────────┘                           │
│    │                                                          │
│    └── pushgateway (:9091) ──── Grafana                       │
└───────────────────────────────────────────────────────────────┘
```

The harness is a lightweight Python service (~6 dependencies, no ML/GPU needed) that drives unmodified Scope instances via their HTTP API. Each Scope instance connects to a Livepeer orchestrator for remote GPU inference.

## Quick Start

```bash
# 1. Clone
git clone https://github.com/livepeer/scope-load-testing.git
cd scope-load-testing

# 2. Configure
cp .env.example .env
# Edit .env with your Livepeer and Scope credentials

# 3. Generate test videos (requires opencv-python)
pip install opencv-python
python scripts/generate_test_videos.py

# 4. Start
docker compose up -d
```

The scheduler starts automatically, discovers orchestrators, and begins running test scenarios.

## CLI

```bash
# Run a single scenario manually
python -m loadtest.cli run --scenario longlive_t2v_5m --scope-url http://localhost:8001

# Start the scheduler daemon
python -m loadtest.cli schedule

# List discovered orchestrators
python -m loadtest.cli discover

# Show today's test coverage
python -m loadtest.cli coverage

# Show performance baselines
python -m loadtest.cli baselines
```

## Configuration

All configuration is in `config/default.yaml`:

```yaml
budget:
  daily_percent: 20        # % of 24hrs each orchestrator is under test
  max_run_duration_mins: 30 # hard cap per run
  min_run_gap_mins: 15     # cooldown between batches

scenarios:
  - pipeline: longlive
    modes: [t2v, v2v, i2v]
    durations: [1, 5, 15]
    prompts_pool: nature
    # ...
```

**Adding a new pipeline** requires only a config change — add an entry to the `scenarios` list. No code modifications needed.

## How It Works

### Scheduler

1. Discovers all healthy Livepeer orchestrators (refreshes every 4 hours)
2. Calculates a daily test budget per orchestrator (default: 20% of 24hrs = ~10 runs/day)
3. Assigns scenarios using a test-debt priority queue — orchestrators with the least coverage get tested first
4. Runs scenarios concurrently across available Scope instances (1 per orchestrator at a time)
5. After 5 consecutive failures, an orchestrator is blacklisted for the day

### Executor

Each test run follows this lifecycle:

1. **Connect** — `POST /api/v1/cloud/connect` → poll until connected (timeout: 120s)
2. **Load** — `POST /api/v1/pipeline/load` → poll until loaded (timeout: 300s)
3. **Stream** — `POST /api/v1/session/start` → monitoring loop:
   - Capture frames, validate they're not black/corrupt
   - Switch prompts, verify output changes (pixel diff)
   - Track FPS and VRAM usage
   - Detect stalls (fps=0 for >10s)
4. **Cleanup** — stop session, disconnect, capture logs on failure
5. **Report** — classify errors, push metrics, update baselines

### Validation

- **Frame quality** — JPEG decode, dimension check, black frame detection (pixel variance)
- **Prompt sensitivity** — mean pixel difference before/after prompt switch must exceed threshold
- **VRAM leak** — compare first-quarter vs last-quarter VRAM in mid/long sessions
- **Stall detection** — fps_out=0 for >10s triggers failure

### Error Taxonomy

Every failure is classified as: `network` (timeout, disconnect), `orchestrator` (502/503, capacity), `runner` (OOM, CUDA, pipeline crash), or `protocol` (bad response). Logs are captured on failure for post-mortem.

### Regression Detection

- 7-day rolling P50/P95 baselines per scenario
- Flags >20% degradation in first-frame latency, FPS, or pipeline load time
- Cold start frequency tracking per orchestrator

## Grafana Dashboard

Import `dashboards/grafana/scope-loadtest.json` into Grafana. Six panels:

1. **Overview** — total runs, pass rate, orchestrator coverage
2. **Per-Orchestrator** — table with success rate, connect time, FPS, budget progress
3. **Per-Pipeline** — load time, first-frame latency, steady FPS by pipeline/mode
4. **Latency Trends** — 7-day P50/P95 with drift overlay
5. **Error Breakdown** — failures by category over time
6. **Budget & Coverage** — daily budget consumed per orchestrator

## Development

```bash
# Install with dev dependencies
pip install -e ".[dev]"

# Run tests
pytest                           # all tests
pytest --ignore=tests/test_executor.py  # fast tests only (~1s)

# Build Docker image
docker build -f Dockerfile.harness -t scope-loadtest-harness .
```

## Project Structure

```
src/loadtest/
├── cli.py           # Click CLI (run, schedule, discover, coverage, baselines)
├── config.py        # YAML config loading and validation
├── scenarios.py     # Scenario matrix expansion, session body builder
├── scope_client.py  # Typed async HTTP client for Scope API
├── executor.py      # Full test lifecycle (connect→load→stream→cleanup)
├── scheduler.py     # Budget planning, fair rotation, execution loop
├── discovery.py     # Orchestrator discovery and health tracking
├── coverage.py      # Per-orchestrator daily coverage persistence
├── metrics.py       # Prometheus metric definitions and push
├── validators.py    # Frame quality and prompt sensitivity (Pillow)
├── results.py       # RunResult, error taxonomy, log capture
└── regression.py    # Rolling baselines and drift detection
```
