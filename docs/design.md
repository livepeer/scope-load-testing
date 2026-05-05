# Scope Cloud Inference Load Testing — Design Spec

**Date:** 2026-05-05
**Status:** Approved

## 1. Purpose

Build a load testing harness that continuously validates the health, performance, and correctness of Scope cloud inference across the Livepeer orchestrator network. The system must:

- Cover major functional paths: longlive, ltx2, t2v, v2v, i2v
- Use diversified graph configs (single pipeline, chained, full chain) and prompt sets
- Measure success rate, startup time, prompt-to-first-frame latency, steady-state FPS
- Cover short (1 min), mid (5 min), and long (15 min) stream durations
- Run on a configurable schedule, consuming at most N% of each orchestrator's daily capacity
- Complete each run within 30 minutes, then release all resources
- Progressively rotate through all orchestrators with fair distribution
- Report to Grafana in real time, with regression detection and alerting

## 2. Architecture

**Approach:** HTTP-only external harness (no Scope/ML dependencies) driving Scope instances via their REST API, packaged in docker-compose for portability and scaling.

```
┌──────────────────────────────── Host (GCP VM or any machine) ─────────────────────────────┐
│                                                                                            │
│  docker-compose.yml                                                                        │
│  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐                       │
│  │  scope-1          │   │  scope-2          │   │  scope-3          │  (configurable)     │
│  │  daydream-scope   │   │  daydream-scope   │   │  daydream-scope   │                      │
│  │  :8001            │   │  :8002            │   │  :8003            │                      │
│  └────────┬──────────┘   └────────┬──────────┘   └────────┬──────────┘                     │
│           │ HTTP API              │ HTTP API              │ HTTP API                        │
│  ┌────────┴───────────────────────┴───────────────────────┴──────────┐                     │
│  │                         loadtest-harness                           │                     │
│  │  ┌────────────┐ ┌──────────────┐ ┌───────────┐ ┌───────────────┐ │                     │
│  │  │ Scheduler  │ │ Discovery &  │ │ Scenario  │ │ Metrics &     │ │                     │
│  │  │ (budget +  │ │ Coverage     │ │ Executor  │ │ Grafana Push  │ │                     │
│  │  │  rotation) │ │ Tracker      │ │           │ │               │ │                     │
│  │  └────────────┘ └──────────────┘ └───────────┘ └───────────────┘ │                     │
│  └────────────────────────────┬──────────────────────────────────────┘                     │
│  ┌────────────────────────────┴──────────────────────────────────────┐                     │
│  │                     Prometheus Push Gateway :9091                   │                     │
│  └────────────────────────────────────────────────────────────────────┘                     │
└────────────────────────────────────────────────────────────────────────────────────────────┘
                                        │ push metrics
                                        ▼
                                 ┌─────────────┐
                                 │   Grafana    │
                                 └─────────────┘
```

**Key design decisions:**

- **Scope instances are unmodified `daydreamlive/scope` containers** with `CUDA_VISIBLE_DEVICES=""` (no local GPU). They connect to Livepeer for remote inference, exactly like production.
- **Harness is a separate lightweight container** — depends only on `httpx`, `prometheus_client`, `pyyaml`, `Pillow`, `scikit-image`. Zero ML dependencies.
- **Each Scope instance connects to one orchestrator at a time** — clean 1:1 mapping. After scenarios complete, it disconnects and gets reassigned.
- **All config is YAML** — scenarios, prompts, schedules, thresholds. No hardcoded values in code.
- **Persistent data volume** (`data/`) — coverage, baselines, history, failure logs survive restarts.

## 3. Scheduler & Traffic Budget Engine

### 3.1 Budget Model

```yaml
# config/default.yaml
budget:
  daily_percent: 20          # each orchestrator gets 20% of 24hrs = 4.8hrs/day
  max_run_duration_mins: 30  # hard cap per run
  min_run_gap_mins: 15       # cooldown between runs on same orchestrator
  schedule_start: "00:00"    # daily window start (UTC)
  schedule_end: "23:59"      # daily window end (UTC)
```

- `daily_percent` is configurable — tuning this single value scales test traffic up or down.
- Per orchestrator: `runs_needed = (24h * daily_percent) / max_run_duration_mins`. At 20%, that's ~10 runs per orchestrator per day.
- Runs are distributed evenly across the schedule window and interleaved across orchestrators so that at any given time only N are under test (where N = number of Scope instances).

### 3.2 Scheduler Algorithm

1. On startup (and daily at `schedule_start`), call Discovery to get all healthy orchestrators.
2. Calculate the run plan for the day:
   - Each orchestrator gets `runs_needed` slots spread across the window.
   - Orchestrators are interleaved — at most N under test simultaneously.
   - Each slot is assigned a scenario from the pool, rotating to cover all scenarios per orchestrator.
3. Execute runs by assigning available Scope instances to the next pending slot.
4. Track completion: if a run fails or is skipped (orchestrator went unhealthy), reschedule in a later slot.
5. At end of day, log coverage report.

### 3.3 Fair Rotation

The scheduler maintains a priority queue sorted by "test debt" — how far behind each orchestrator is versus its budget. Orchestrators with the most remaining test time get scheduled first. This naturally handles:

- New orchestrators discovered mid-day get caught up quickly.
- Temporarily unhealthy orchestrators get extra runs once they recover.
- No orchestrator is systematically favored or starved.

### 3.4 Coverage Persistence

State is persisted to `data/coverage.json` so it survives restarts:

```json
{
  "2026-05-05": {
    "O-abc123": {
      "runs_completed": 7,
      "runs_planned": 10,
      "scenarios_covered": ["longlive_t2v_short", "ltx2_i2v_mid"],
      "failures": 1,
      "failure_categories": {"runner": 1}
    }
  }
}
```

## 4. Orchestrator Discovery & Health

### 4.1 Discovery Flow

1. Query the Livepeer discovery endpoint to enumerate available orchestrators.
2. Filter to orchestrators that support Scope's model/pipeline requirements.
3. Health check each candidate: attempt a lightweight cloud connect + disconnect via Scope API (30s timeout per orchestrator).
4. Return healthy orchestrators with metadata.

### 4.2 Orchestrator Record

```python
@dataclass
class Orchestrator:
    id: str                       # unique identifier
    address: str                  # connection address
    region: str | None            # geographic region if available
    first_seen: datetime          # when discovery first found it
    last_healthy: datetime        # last successful health check
    last_tested: datetime | None  # last completed test run
    consecutive_failures: int     # health check failures in a row
    status: str                   # healthy | unhealthy | blacklisted
```

### 4.3 Health Check Cadence

- Full discovery runs at scheduler startup and every 4 hours.
- Between discoveries, the scheduler uses the cached list.
- If a run fails with `network` or `orchestrator` error, that orchestrator gets an immediate re-health-check before the next scheduled run.
- After 5 consecutive failures, the orchestrator is blacklisted for the rest of the day (still reported in coverage as "unreachable").
- Blacklist resets at `schedule_start` next day.

## 5. Scenario Definitions

### 5.1 Scenario Format

Each scenario is a YAML file in `config/scenarios/`:

```yaml
name: longlive_v2v_mid
pipeline: longlive
mode: v2v
duration_mins: 5
graph:
  nodes:
    - id: input
      type: source
      source_mode: video_file
      source_name: /data/videos/gradient_512x512_30s.mp4
    - id: longlive
      type: pipeline
      pipeline_id: longlive
    - id: output
      type: sink
    - id: record
      type: record
  edges:
    - from: input
      from_port: video
      to_node: longlive
      to_port: video
      kind: stream
    - from: longlive
      from_port: video
      to_node: output
      to_port: video
      kind: stream
    - from: longlive
      from_port: video
      to_node: record
      to_port: video
      kind: stream
prompts:
  pool: nature
  switch_interval_s: 30
parameters:
  noise_scale: 0.7
  width: 512
  height: 512
validation:
  min_fps: 6
  max_first_frame_s: 60
  frame_check_interval_s: 30
  check_recording: true
```

### 5.2 Scenario Matrix

| Scenario | Pipeline | Mode | Duration | Graph Type |
|----------|----------|------|----------|------------|
| `longlive_t2v_short` | longlive | t2v | 1 min | single |
| `longlive_t2v_mid` | longlive | t2v | 5 min | single |
| `longlive_t2v_long` | longlive | t2v | 15 min | single |
| `longlive_v2v_short` | longlive | v2v | 1 min | single |
| `longlive_v2v_mid` | longlive | v2v | 5 min | single |
| `longlive_v2v_long` | longlive | v2v | 15 min | single |
| `longlive_i2v_short` | longlive | i2v | 1 min | single |
| `longlive_i2v_mid` | longlive | i2v | 5 min | single |
| `ltx2_t2v_short` | ltx2 | t2v | 1 min | single |
| `ltx2_t2v_mid` | ltx2 | t2v | 5 min | single |
| `ltx2_i2v_short` | ltx2 | i2v | 1 min | single |
| `ltx2_i2v_mid` | ltx2 | i2v | 5 min | single |
| `chain_longlive_rife_mid` | longlive+rife | v2v | 5 min | chained |
| `chain_depth_longlive_rife_mid` | depth+longlive+rife | v2v | 5 min | full chain |
| `chain_longlive_rife_long` | longlive+rife | v2v | 15 min | chained |

### 5.3 Prompt Datasets

Prompt pools are YAML files in `config/prompts/`:

```yaml
# config/prompts/nature.yaml
prompts:
  - "a serene mountain lake at sunrise with mist"
  - "ocean waves crashing on a rocky coastline"
  - "a dense forest with sunlight filtering through trees"
  - "a vast desert with rolling sand dunes at golden hour"
  - "a waterfall cascading into a tropical pool"
  # ... 20-30 prompts per pool
```

Pools: `nature.yaml`, `urban.yaml`, `abstract.yaml`, `stress.yaml` (edge cases: very long prompts, special characters, minimal prompts, empty string).

## 6. Executor Lifecycle

Each run follows this sequence:

```
executor.run(scope_url, orchestrator_id, scenario, prompt_set)
│
├─ 1. CONNECT PHASE (timed)
│  ├─ POST /api/v1/cloud/connect {app_id, orchestrator targeting}
│  ├─ Poll GET /api/v1/cloud/status until connected=true (timeout: 120s)
│  └─ Record: connect_duration_s, cold_or_warm_start
│
├─ 2. LOAD PHASE (timed)
│  ├─ POST /api/v1/pipeline/load {pipeline_ids from scenario}
│  ├─ Poll GET /api/v1/pipeline/status until status=loaded (timeout: 300s)
│  └─ Record: load_duration_s
│
├─ 3. STREAM PHASE (timed, runs for scenario.duration_mins)
│  ├─ POST /api/v1/session/start {graph or single pipeline config}
│  ├─ Record: time to first frame (poll /session/metrics until frames_out > 0)
│  ├─ Start recording: POST /recordings/headless/start?node_id=record
│  │
│  ├─ MONITORING LOOP (every frame_check_interval_s):
│  │  ├─ GET /session/metrics → record fps_in, fps_out, vram
│  │  ├─ GET /session/frame → validate not black, correct dimensions
│  │  ├─ If prompt switch due: POST /session/parameters with next prompt
│  │  │   └─ Capture frame before + after switch → SSIM comparison
│  │  └─ Check for stalls: fps_out == 0 for > 10s → flag failure
│  │
│  ├─ At end of duration: stop recording, download, validate MP4
│  └─ POST /api/v1/session/stop
│
├─ 4. CLEANUP PHASE
│  ├─ POST /api/v1/cloud/disconnect
│  ├─ Verify cloud status shows disconnected
│  └─ If any failure: GET /api/v1/logs/tail?lines=100 → store in data/failures/
│
└─ 5. REPORT
   ├─ Classify result: pass / fail (with error taxonomy)
   ├─ Push all metrics to Prometheus with labels
   └─ Return structured result to scheduler
```

**Hard timeout:** A watchdog enforces `max_run_duration_mins`. If any phase exceeds it, the executor force-stops the session, captures logs, and reports a timeout failure. No run ever exceeds 30 minutes.

**Prompt switching:** The executor picks prompts from the configured pool and rotates them at `switch_interval_s`. This tests that the model responds to prompt changes (SSIM check), measures prompt-to-effect latency, and catches crashes on prompt transitions.

## 7. Validation & Quality Checks

### 7.1 Frame Validation

Runs during the monitoring loop at `frame_check_interval_s`:

1. **Decode check** — valid JPEG, no decode errors.
2. **Dimension check** — matches expected width/height from scenario.
3. **Black frame check** — pixel standard deviation > threshold (e.g., std > 5).
4. **Corrupt check** — reasonable color distribution, no single-color artifacts.

### 7.2 Prompt Sensitivity

On each prompt switch:

1. Capture frame just before the switch.
2. Wait ~10s for the model to respond.
3. Capture frame after.
4. Compute SSIM between the two frames.
5. If SSIM > 0.85 → frames are too similar → model isn't responding to prompts → fail.

### 7.3 Recording Validation

After each recording download:

1. File size > 0.
2. Valid MP4 container (can open with OpenCV or ffprobe).
3. Duration within tolerance (expected +/- 5s).
4. Frame count > 0.

### 7.4 VRAM Leak Detection

During mid and long sessions (>= 5 min), compare first-quarter average VRAM with last-quarter average. Growth > 200MB → potential leak, flagged in metrics.

### 7.5 Sequential Load/Unload

Test loading longlive, unloading, then loading ltx2 on the same runner. Catches VRAM leaks or cleanup failures between sessions.

## 8. Regression Detection

### 8.1 Rolling Baseline

The harness maintains `data/baselines.json` with 7-day rolling statistics per scenario type:

```json
{
  "longlive_t2v": {
    "first_frame_p50": 12.3,
    "first_frame_p95": 25.1,
    "steady_fps_p50": 9.2,
    "pipeline_load_p50": 18.5,
    "updated_at": "2026-05-04T00:00:00Z",
    "sample_count": 70
  }
}
```

### 8.2 Drift Detection

After each run, compare against baseline:

- **First-frame latency drift:** `(current - baseline_p50) / baseline_p50 > 0.20` → flag.
- **FPS drift:** `(baseline_fps_p50 - current) / baseline_fps_p50 > 0.20` → flag.
- **Pipeline load drift:** same 20% threshold.

Drift flags are pushed as Prometheus metrics and trigger Grafana alerts.

### 8.3 Model Consistency

For a specific "reference scenario" (fixed prompt, fixed seed), capture a reference frame and store it. On subsequent runs, compare SSIM/PSNR against the stored reference. SSIM < 0.7 flags a potential model version mismatch or weight corruption. Runs once per orchestrator per day.

### 8.4 Cold Start vs Warm Start Tracking

Tag each session as cold or warm based on whether the runner was already provisioned. Track cold start frequency per orchestrator — a sudden spike means machines aren't staying warm.

### 8.5 Orchestrator Routing Fairness

If the same model is requested from the same gateway repeatedly, track whether the same runner is always selected or if requests are round-robined. Detects sticky routing bugs.

## 9. Error Taxonomy

Every failure is classified as one of:

| Category | Examples |
|----------|----------|
| **network** | Connection timeout, DNS failure, unexpected disconnect |
| **orchestrator** | Capacity rejection, routing error, provisioning failure, health check failure |
| **runner** | OOM, CUDA error, pipeline crash, model load failure, corrupt output |
| **protocol** | Malformed response, unexpected message type, API error codes |

On failure, the harness captures `GET /api/v1/logs/tail?lines=100` and stores it in `data/failures/{date}/{orchestrator_id}_{scenario}_{timestamp}.log`. Logs for passing sessions are not stored. Failure logs are rotated (keep last 7 days).

## 10. Metrics & Grafana

### 10.1 Prometheus Metrics

All metrics carry labels: `orchestrator_id`, `pipeline`, `mode`, `scenario`, `duration_class`.

**Histograms (latency distributions):**
- `connect_duration_seconds` — cloud connect time
- `pipeline_load_seconds` — pipeline load time
- `first_frame_seconds` — prompt to first output frame
- `prompt_switch_latency_seconds` — time for output to change after prompt update

**Gauges (per-session snapshots):**
- `stream_fps_out` — output FPS during session
- `stream_fps_in` — input FPS
- `vram_allocated_mb` — GPU memory usage on runner
- `vram_usage_percent` — vram_allocated / vram_total
- `frames_to_cloud` / `frames_from_cloud` — frame delivery counters

**Counters (cumulative):**
- `runs_total{result=pass|fail}` — total runs attempted
- `failures_total{category=network|orchestrator|runner|protocol}` — failures by taxonomy
- `frames_validated_total{result=valid|black|corrupt|wrong_size}` — frame checks
- `recordings_validated_total{result=valid|corrupt|wrong_duration}` — recording checks
- `prompt_sensitivity_checks_total{result=pass|fail}` — SSIM checks

**Gauges (budget & regression):**
- `budget_runs_planned` / `budget_runs_completed` — per orchestrator per day
- `budget_percent_consumed` — daily completion rate
- `orchestrator_coverage_percent` — tested / total
- `baseline_p50_first_frame_seconds` / `baseline_p95_first_frame_seconds` — rolling baselines
- `baseline_drift_percent` — current vs baseline deviation

### 10.2 Push Mechanism

Metrics are pushed to Prometheus push gateway after each run completes. Batched per run, not streamed continuously.

### 10.3 Grafana Dashboards

Seven panels, exported as JSON in `dashboards/grafana/scope-loadtest.json`:

| Panel | Content | Alert Condition |
|-------|---------|-----------------|
| **Overview** | Total runs today, pass rate gauge, active sessions, orchestrator count | Pass rate < 70% |
| **Per-Orchestrator** | Table: success rate, avg connect time, avg FPS, runs completed vs planned | Any O with 0 runs in last 4 hours |
| **Per-Pipeline** | Grouped bars: load time, first-frame latency, steady FPS by pipeline + mode | First-frame P95 > 2x baseline |
| **Latency Trends** | 7-day line chart: P50/P95 first-frame latency, with baseline band | Drift > 20% from 7-day avg |
| **Error Breakdown** | Stacked bar by error taxonomy over time | Runner errors > 10% of runs |
| **Budget & Coverage** | Per-O progress bars: % budget consumed, coverage heat map | Any O under 50% budget at EOD |
| **Quality** | Frame validation pass rate, SSIM distribution, recording validation rate | Black frame rate > 5% |

## 11. Docker Compose & Containers

### 11.1 Compose File

```yaml
services:
  harness:
    build: .
    image: scope-loadtest-harness
    depends_on: [pushgateway]
    volumes:
      - ./config:/app/config
      - ./videos:/data/videos
      - ./data:/app/data
    environment:
      - SCOPE_INSTANCES=scope-1:8001,scope-2:8002
      - GRAFANA_PUSH_URL=http://pushgateway:9091
      - LIVEPEER_DISCOVERY_URL=${LIVEPEER_DISCOVERY_URL}
      - LIVEPEER_TOKEN=${LIVEPEER_TOKEN}
      - SCOPE_CLOUD_APP_ID=${SCOPE_CLOUD_APP_ID}

  scope-1:
    image: daydreamlive/scope:${SCOPE_IMAGE_TAG:-latest}
    command: ["uv", "run", "daydream-scope", "--host", "0.0.0.0", "--port", "8001"]
    environment:
      - CUDA_VISIBLE_DEVICES=
    volumes:
      - ./videos:/data/videos
    ports: ["8001:8001"]

  scope-2:
    image: daydreamlive/scope:${SCOPE_IMAGE_TAG:-latest}
    command: ["uv", "run", "daydream-scope", "--host", "0.0.0.0", "--port", "8002"]
    environment:
      - CUDA_VISIBLE_DEVICES=
    volumes:
      - ./videos:/data/videos
    ports: ["8002:8002"]

  pushgateway:
    image: prom/pushgateway:latest
    ports: ["9091:9091"]
```

### 11.2 Harness Dockerfile

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY src/ /app/src/
COPY config/ /app/config/
ENTRYPOINT ["python", "-m", "loadtest.cli"]
CMD ["schedule"]
```

### 11.3 CLI Commands

```bash
python -m loadtest.cli schedule                              # start the scheduler daemon
python -m loadtest.cli run --scenario longlive_t2v_short \
                           --orchestrator O-abc123           # single manual run
python -m loadtest.cli discover                              # list orchestrators + health
python -m loadtest.cli coverage                              # today's coverage report
python -m loadtest.cli baselines                             # current baseline metrics
```

### 11.4 Scaling

To add a third Scope instance: add `scope-3` to docker-compose.yml and append to `SCOPE_INSTANCES`. The scheduler automatically distributes work across all available instances.

### 11.5 Portability

To deploy on a new machine:
1. Clone the repo.
2. Create `.env` with tokens and discovery URL.
3. `docker compose up -d`.

No GPU needed. No Python environment to manage. Everything runs in containers.

## 12. Repo Structure

```
scope-load-testing/
├── CLAUDE.md
├── docker-compose.yml
├── Dockerfile.harness
├── pyproject.toml
├── .env.example
├── src/
│   └── loadtest/
│       ├── __init__.py
│       ├── cli.py               # CLI entrypoint (run, schedule, discover, coverage, baselines)
│       ├── config.py            # Load YAML config, validate
│       ├── scheduler.py         # Budget calculation, run timing, orchestrator rotation
│       ├── discovery.py         # Livepeer orchestrator discovery + health check
│       ├── coverage.py          # Track which orchestrators tested, persist state
│       ├── executor.py          # Drive a single test scenario against a Scope instance
│       ├── scenarios.py         # Load scenario YAML, build session start bodies
│       ├── scope_client.py      # Async HTTP client for Scope API (typed)
│       ├── metrics.py           # Prometheus metric definitions + push logic
│       ├── validators.py        # Frame quality, SSIM, recording validation
│       ├── results.py           # Result collection, error taxonomy, log capture
│       └── regression.py        # Baseline comparison, drift detection
├── config/
│   ├── default.yaml             # Budget, thresholds, global settings
│   ├── scenarios/               # One YAML per test scenario
│   │   ├── longlive_t2v_short.yaml
│   │   ├── longlive_t2v_mid.yaml
│   │   ├── longlive_t2v_long.yaml
│   │   ├── longlive_v2v_short.yaml
│   │   ├── longlive_v2v_mid.yaml
│   │   ├── longlive_v2v_long.yaml
│   │   ├── longlive_i2v_short.yaml
│   │   ├── longlive_i2v_mid.yaml
│   │   ├── ltx2_t2v_short.yaml
│   │   ├── ltx2_t2v_mid.yaml
│   │   ├── ltx2_i2v_short.yaml
│   │   ├── ltx2_i2v_mid.yaml
│   │   ├── chain_longlive_rife_mid.yaml
│   │   ├── chain_depth_longlive_rife_mid.yaml
│   │   └── chain_longlive_rife_long.yaml
│   └── prompts/
│       ├── nature.yaml
│       ├── urban.yaml
│       ├── abstract.yaml
│       └── stress.yaml
├── videos/
│   ├── solid_red_512x512_30s.mp4
│   ├── solid_green_512x512_30s.mp4
│   ├── gradient_512x512_30s.mp4
│   └── scene_change_512x512_60s.mp4
├── dashboards/
│   └── grafana/
│       └── scope-loadtest.json
├── data/                        # gitignored, persistent volume
│   ├── coverage.json
│   ├── baselines.json
│   ├── history.json
│   └── failures/
├── docs/
│   ├── design.md
│   └── docker-optimization-plan.md
└── tests/
    └── test_executor.py
```

## 13. Success Criteria

| Category | Metric | Pass | Fail |
|----------|--------|------|------|
| **Connect** | Cloud connect time | < 120s | Timeout or rejection |
| **Load** | Pipeline load time | Status "loaded" < 300s | Timeout or error |
| **First frame** | Prompt to first frame | < 60s from stream start | No frame after 60s |
| **Stability** | Stream FPS | fps_out > 0 for full duration | fps_out = 0 for > 10s |
| **Shutdown** | Clean stop | Session stops, recording downloadable | Hang or error |
| **Latency** | P50/P95/P99 prompt-to-first-frame | Track; alert on regression | > 20% degradation vs 7-day avg |
| **Quality** | Frame not black/corrupt | Pixel variance > threshold | Black or zero-variance frame |
| **Quality** | Frame dimensions | Match requested resolution | Mismatch |
| **Quality** | Prompt sensitivity | Different prompts → different frames (SSIM < 0.85) | Identical output |
| **Resources** | VRAM usage | < 90% of total | OOM or > 90% |
| **Resources** | VRAM leak | End VRAM within 200MB of start | Monotonic growth |
| **Recording** | Valid output | Non-zero MP4, duration matches expected +/- 5s | Corrupt or wrong duration |
| **Network** | Connection stability | No unexpected disconnects | Mid-session disconnect |
| **Network** | Frame delivery | frames_from_cloud > 0 and growing | Stuck at 0 |
| **Regression** | Baseline drift | Within 20% of 7-day rolling avg | > 20% deviation |
| **Consistency** | Model output | Reference frame SSIM > 0.7 for same prompt+seed | SSIM < 0.7 |
| **Cold start** | Frequency tracking | Informational — track per orchestrator | Sudden spike = alert |
| **Routing** | Fairness | Requests distributed across runners | Always same runner |

## 14. Dependencies

**Harness container (Python 3.12):**
- `httpx` — async HTTP client for Scope API
- `prometheus_client` — metric definitions + push gateway
- `pyyaml` — config and scenario loading
- `Pillow` — JPEG decode, frame dimension/variance checks
- `scikit-image` — SSIM computation for prompt sensitivity and model consistency
- `click` — CLI framework

**No ML dependencies.** No torch, no CUDA, no numpy beyond what Pillow/scikit-image pull in.
