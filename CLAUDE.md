# CLAUDE.md — Scope Load Testing Harness

## Project Overview

This repo contains a load testing harness for Daydream Scope cloud inference on the Livepeer network. The harness is a lightweight Python service (no ML dependencies) that drives Scope local instances via HTTP API to test end-to-end cloud inference through Livepeer orchestrators.

**Architecture:** The harness runs in Docker containers alongside one or more Scope instances. Each Scope instance connects to a Livepeer orchestrator for remote GPU inference. The harness drives test scenarios via Scope's HTTP API and pushes metrics to Grafana via Prometheus.

```
loadtest-harness (lightweight Python, httpx only)
  → Scope instance(s) (daydream-scope, no local GPU)
    → Livepeer orchestrator(s) (remote GPU inference)
      → metrics → Prometheus push gateway → Grafana
```

## Development Commands

```bash
docker compose up -d                    # Start full stack (harness + scope + prometheus)
docker compose up -d --scale scope=3    # Start with 3 Scope instances
docker compose down                     # Stop everything
docker compose logs -f harness          # Follow harness logs
docker compose logs -f scope-1          # Follow specific scope instance

# Local dev (without Docker)
pip install -e ".[dev]"                 # Install harness with dev deps
pytest                                  # Run harness unit tests
python -m loadtest.cli run              # Run a single test cycle
python -m loadtest.cli schedule         # Start the scheduler daemon
```

## Scope HTTP API Reference

All endpoints are relative to a Scope instance base URL (e.g., `http://scope-1:8000`).

### Core Lifecycle

| Operation | Method | Path | Body |
|-----------|--------|------|------|
| Health | GET | `/health` | - |
| Cloud connect | POST | `/api/v1/cloud/connect` | `{"app_id": "...", "api_key": "...", "user_id": "..."}` |
| Cloud status | GET | `/api/v1/cloud/status` | - |
| Cloud disconnect | POST | `/api/v1/cloud/disconnect` | - |
| Resolve workflow | POST | `/api/v1/workflow/resolve` | `{"nodes": [...]}` |
| Load pipeline(s) | POST | `/api/v1/pipeline/load` | `{"pipeline_ids": ["longlive"]}` |
| Pipeline status | GET | `/api/v1/pipeline/status` | - |
| Start session | POST | `/api/v1/session/start` | See session body formats below |
| Session metrics | GET | `/api/v1/session/metrics` | - |
| Capture frame | GET | `/api/v1/session/frame?sink_node_id=output&quality=85` | - (returns JPEG) |
| Update parameters | POST | `/api/v1/session/parameters` | `{"prompts": [...], "noise_scale": 0.5}` |
| Get parameters | GET | `/api/v1/session/parameters` | - |
| Stop session | POST | `/api/v1/session/stop` | - |
| Get logs | GET | `/api/v1/logs/tail?lines=50` | - |

### Recording Endpoints

| Operation | Method | Path |
|-----------|--------|------|
| Start recording | POST | `/api/v1/recordings/headless/start?node_id=<id>` |
| Stop recording | POST | `/api/v1/recordings/headless/stop?node_id=<id>` |
| Download recording | GET | `/api/v1/recordings/headless?node_id=<id>` (returns MP4) |

### Session Start Body Formats

**Text-to-video (t2v) — single pipeline:**
```json
{
  "pipeline_id": "longlive",
  "input_mode": "text",
  "prompts": [{"text": "a forest in winter", "weight": 100}]
}
```

**Video-to-video (v2v) — single pipeline:**
```json
{
  "pipeline_id": "longlive",
  "input_mode": "video",
  "input_source": {
    "enabled": true,
    "source_type": "video_file",
    "source_name": "/data/videos/test.mp4"
  }
}
```

**Graph mode (multi-pipeline):**
```json
{
  "input_mode": "video",
  "graph": {
    "nodes": [
      {"id": "input", "type": "source", "source_mode": "video_file", "source_name": "/data/videos/test.mp4"},
      {"id": "depth", "type": "pipeline", "pipeline_id": "video-depth-anything"},
      {"id": "longlive", "type": "pipeline", "pipeline_id": "longlive"},
      {"id": "rife", "type": "pipeline", "pipeline_id": "rife"},
      {"id": "output", "type": "sink"},
      {"id": "record", "type": "record"}
    ],
    "edges": [
      {"from": "input", "from_port": "video", "to_node": "depth", "to_port": "video", "kind": "stream"},
      {"from": "depth", "from_port": "video", "to_node": "longlive", "to_port": "vace_input_frames", "kind": "stream"},
      {"from": "input", "from_port": "video", "to_node": "longlive", "to_port": "video", "kind": "stream"},
      {"from": "longlive", "from_port": "video", "to_node": "rife", "to_port": "video", "kind": "stream"},
      {"from": "rife", "from_port": "video", "to_node": "output", "to_port": "video", "kind": "stream"},
      {"from": "rife", "from_port": "video", "to_node": "record", "to_port": "video", "kind": "stream"}
    ]
  }
}
```

**Critical:** `input_mode: "video"` is required for video file sources. Without it, frames don't flow.

### Session Metrics Response Shape

```json
{
  "sessions": {
    "headless": {
      "fps_in": 30.0,
      "fps_out": 10.5,
      "pipeline_fps": 8.2,
      "frames_in": 150,
      "frames_out": 50,
      "elapsed_seconds": 15.0,
      "headless": true
    }
  },
  "gpu": {
    "vram_allocated_mb": 8192.5,
    "vram_reserved_mb": 10240.0,
    "vram_total_mb": 81920.0
  }
}
```

### Cloud Status Response Shape

```json
{
  "connected": true,
  "connecting": false,
  "error": null,
  "webrtc_connected": true,
  "app_id": "daydream/scope-livepeer--prod/ws",
  "connection_id": "abc123",
  "credentials_configured": true,
  "stats": {
    "frames_to_cloud": 1000,
    "frames_from_cloud": 950,
    "cloud_fps_in": 30.0,
    "cloud_fps_out": 10.0
  },
  "last_close_code": null,
  "last_close_reason": null
}
```

## Scope Pipeline Reference

### Top Pipelines (by usage)

| Pipeline | Mode | VRAM | Description |
|----------|------|------|-------------|
| `longlive` | t2v, v2v, i2v | ~20GB | Autoregressive video diffusion (Wan2.1 1.3B base) |
| `ltx2` | t2v, i2v | varies | LTX-2 video generation (installed as plugin `scope-ltx-2`) |
| `video-depth-anything` | v2v | ~1GB | Depth estimation preprocessor |
| `rife` | v2v | ~0.5GB | Frame interpolation (doubles FPS) |

### Common Graph Configurations

1. **longlive** (standalone) — 80.8% of sessions
2. **video-depth → longlive → rife** — 84.4% success rate
3. **longlive → rife** — 84.9% success rate
4. **ltx2** (standalone, plugin) — 44.5% success rate

### Pipeline Modes

- **t2v (text-to-video):** `input_mode: "text"`, prompts only, no video source
- **v2v (video-to-video):** `input_mode: "video"`, video source + prompts, `noise_scale` controls blend
- **i2v (image-to-video):** `input_mode: "video"`, single-frame image as video source, pipeline generates motion

### Model Artifacts (for reference)

```
longlive:
  ├─ daydreamlive/Wan2.1-T2V-1.3B          (~2.5GB)
  ├─ daydreamlive/WanVideo_comfy            (~1.5GB)  UMT5 encoder + VACE
  ├─ daydreamlive/Autoencoders              (~200MB)
  └─ daydreamlive/LongLive-1.3B            (~2.5GB)

video-depth-anything:
  └─ daydreamlive/Video-Depth-Anything-Small (~50MB)

rife:
  └─ daydreamlive/RIFE                      (~30MB)
```

## Livepeer Cloud Mode

### How Scope Connects to Livepeer

Scope's local client connects to Livepeer via:
1. `POST /api/v1/cloud/connect` with `app_id` pointing to the Livepeer gateway
2. Scope internally establishes a WebSocket to the Livepeer orchestrator
3. The orchestrator provisions a runner (GPU machine) with the Scope cloud image
4. Media flows: local Scope → trickle channels → remote runner → trickle channels → local Scope
5. All API calls (pipeline load, parameters, etc.) are proxied over the WS to the remote runner

### Orchestrator Discovery

Livepeer orchestrators are discoverable via the Livepeer network. The load test harness should:
1. Query the discovery endpoint to enumerate available orchestrators
2. Track which orchestrators have been tested (coverage map)
3. Rotate through orchestrators fairly — equal test time budget per orchestrator
4. Skip orchestrators that fail health checks or are unreachable

### Key Environment Variables for Cloud Connect

| Variable | Purpose |
|----------|---------|
| `LIVEPEER_TOKEN` | Base64-encoded JSON with signer/discovery URLs |
| `SCOPE_CLOUD_APP_ID` | Cloud app identifier |
| `SCOPE_CLOUD_API_KEY` | Cloud API key |

## Load Test Design Constraints

### Traffic Budget Model

- **daily_percent:** Configurable (default 20%) — percentage of 24hrs each orchestrator is under test
- **max_run_duration_mins:** 30 — hard cap per run, then release all resources
- **Fair distribution:** Every reachable, healthy orchestrator gets equal test time
- **Progressive coverage:** Don't test all orchestrators at once; rotate so the network can still serve real jobs
- **Example:** 20% of 24hrs = 4.8hrs per orchestrator/day. With 30min runs = ~10 runs/orchestrator/day

### Test Scenarios

Cover all combinations of:
- **Pipelines:** longlive, ltx2
- **Modes:** t2v, v2v, i2v
- **Duration:** short (1 min), mid (5 min), long (15 min)
- **Graphs:** single pipeline, chained (longlive + rife), full chain (depth + longlive + rife)
- **Prompts:** diversified from a prompt dataset (YAML), varied per scenario

### Success Criteria

| Category | Metric | Pass | Fail |
|----------|--------|------|------|
| **Connect** | Cloud connect time | < 120s | Timeout or rejection |
| **Load** | Pipeline load time | Status "loaded" < 300s | Timeout or error |
| **First frame** | Prompt to first frame | < 60s from stream start | No frame after 60s |
| **Stability** | Stream FPS | fps_out > 0 for full duration | fps_out = 0 for > 10s |
| **Shutdown** | Clean stop | Session stops, recording downloadable | Hang or error |
| **Latency** | P50/P95/P99 prompt-to-first-frame | Track and alert on regression | > 20% degradation vs 7-day avg |
| **Quality** | Frame not black/corrupt | Pixel variance > threshold | Black or zero-variance frame |
| **Quality** | Frame dimensions | Match requested resolution | Mismatch |
| **Quality** | Prompt sensitivity | Different prompts produce different frames (SSIM < threshold) | Identical output for different prompts |
| **Resources** | VRAM usage | < 90% of total | OOM or > 90% |
| **Resources** | VRAM leak | End VRAM ≈ start VRAM (within tolerance) | Monotonic VRAM growth |
| **Recording** | Valid output | Non-zero MP4, duration matches expected | Zero-size, corrupt, or wrong duration |
| **Network** | Connection stability | No unexpected disconnects during session | Disconnect mid-session |
| **Network** | Frame delivery | frames_from_cloud > 0 and growing | Stuck at 0 or stalled |

### Regression Detection

- Compare each run's P50 latency and FPS against a 7-day rolling average
- Auto-flag if > 20% degradation
- For identical prompts + seeds, compare SSIM/PSNR of reference frames across runs
- Track cold-start vs warm-start frequency per orchestrator

### Error Taxonomy

Classify every failure as one of:
- **network:** timeout, disconnect, DNS failure
- **orchestrator:** capacity rejection, routing error, provisioning failure
- **runner:** OOM, CUDA error, pipeline crash, model load failure
- **protocol:** malformed response, unexpected message type

On failure, capture `GET /api/v1/logs/tail?lines=100` and store alongside the result for post-mortem. Don't store logs for passing sessions.

## Repo Structure (Target)

```
scope-load-testing/
├── CLAUDE.md                     # This file
├── docker-compose.yml            # Full stack: harness + scope + prometheus + pushgateway
├── Dockerfile.harness            # Lightweight Python image for the harness
├── pyproject.toml                # Harness Python package (httpx, prometheus_client, pyyaml)
├── src/
│   └── loadtest/
│       ├── __init__.py
│       ├── cli.py                # CLI entrypoint (run, schedule, discover)
│       ├── config.py             # Load YAML config, validate
│       ├── scheduler.py          # Daily budget calculation, run timing, orchestrator rotation
│       ├── discovery.py          # Livepeer orchestrator discovery + health check
│       ├── coverage.py           # Track which orchestrators have been tested
│       ├── executor.py           # Drive a single test scenario against a Scope instance
│       ├── scenarios.py          # Scenario definitions (graph configs, modes, durations)
│       ├── scope_client.py       # HTTP client for Scope API (typed, async)
│       ├── metrics.py            # Prometheus metric definitions + push logic
│       ├── validators.py         # Frame quality, SSIM, recording validation
│       ├── results.py            # Result collection, error taxonomy, log capture
│       └── regression.py         # Baseline comparison, drift detection
├── config/
│   ├── default.yaml              # Default test configuration
│   ├── scenarios/                # Test scenario definitions
│   │   ├── longlive_t2v.yaml
│   │   ├── longlive_v2v.yaml
│   │   ├── longlive_i2v.yaml
│   │   ├── ltx2_t2v.yaml
│   │   ├── ltx2_i2v.yaml
│   │   ├── chain_longlive_rife.yaml
│   │   └── chain_depth_longlive_rife.yaml
│   └── prompts/                  # Prompt datasets
│       ├── nature.yaml
│       ├── urban.yaml
│       ├── abstract.yaml
│       └── stress.yaml           # Edge-case prompts (long, special chars, empty)
├── videos/                       # Test input videos (small, committed to repo)
│   ├── solid_red_512x512_30s.mp4
│   ├── solid_green_512x512_30s.mp4
│   ├── gradient_512x512_30s.mp4
│   └── scene_change_512x512_60s.mp4
├── dashboards/
│   └── grafana/
│       └── scope-loadtest.json   # Grafana dashboard definition (importable)
├── docs/
│   ├── design.md                 # Full design spec
│   └── docker-optimization-plan.md
└── tests/
    └── test_executor.py          # Harness unit tests (mock Scope API)
```

## Style Guidelines

- Python 3.12+, type hints everywhere
- Async-first (httpx.AsyncClient, asyncio)
- No ML dependencies in the harness — only httpx, prometheus_client, pyyaml, Pillow (for frame validation), scikit-image (for SSIM)
- Config is YAML, code reads config — no hardcoded values for thresholds, URLs, or timing
- All Scope API interactions go through `scope_client.py` — no raw HTTP calls elsewhere
- Errors are classified by taxonomy (network/orchestrator/runner/protocol) before reporting
- Every metric pushed to Prometheus must have labels: `orchestrator_id`, `pipeline`, `mode`, `scenario`

## Test Video Generation

If test videos need to be regenerated:
```python
import cv2
import numpy as np

def create_test_video(path, color, width=512, height=512, fps=30, duration_s=30):
    w = cv2.VideoWriter(path, cv2.VideoWriter.fourcc(*'mp4v'), fps, (width, height))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    for _ in range(fps * duration_s):
        w.write(frame)
    w.release()
```

## Grafana Dashboard Panels (Target)

1. **Overview:** Total runs, pass rate, active sessions, orchestrator count
2. **Per-orchestrator:** Success rate, avg latency, FPS, coverage completeness
3. **Per-pipeline:** Load time, first-frame latency, steady-state FPS, error rate
4. **Regression:** 7-day P50/P95 latency trend, FPS trend, baseline drift alerts
5. **Errors:** Failure taxonomy breakdown (network/orchestrator/runner/protocol)
6. **Budget:** Daily test budget consumed vs planned, per orchestrator
7. **Quality:** Frame validation pass rate, SSIM scores, recording validation
