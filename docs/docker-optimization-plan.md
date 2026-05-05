# Docker Cold Start Optimization Plan

## Problem

fal.ai cold starts take ~3-4 minutes from machine allocation to first rendered frame. Docker image pull and setup account for ~50% of that time. The remaining time is spent on Python dependency resolution, torch/CUDA initialization, model downloads, and pipeline loading.

## Goal

Reduce cold-start-to-first-frame from ~3-4 minutes to ~30-60 seconds through a phased approach targeting the image build, process startup, and pipeline initialization layers.

## Top Pipeline Usage (driving optimization priorities)

| Rank | Pipeline(s) | Sessions | Success | Unique Users | Total Hours |
|------|-------------|----------|---------|--------------|-------------|
| 1 | longlive | 307 | 80.8% | 46 | 36.2 hrs |
| 2 | video-depth + longlive + rife | 192 | 84.4% | 55 | 24.2 hrs |
| 3 | longlive + rife | 212 | 84.9% | 32 | 23.8 hrs |
| 4 | ltx2 | 173 | 44.5% | 32 | 11.5 hrs |

All top 3 configs include **longlive** (requires ~6.7GB models, ~20GB VRAM). Optimizations that target longlive cover >85% of sessions.

## Model Dependency Map

```
longlive:
  ├─ daydreamlive/Wan2.1-T2V-1.3B          (~2.5GB) ← shared base
  ├─ daydreamlive/WanVideo_comfy            (~1.5GB) ← UMT5 encoder + VACE
  ├─ daydreamlive/Autoencoders              (~200MB) ← lightvae, tae, lighttae
  └─ daydreamlive/LongLive-1.3B            (~2.5GB) ← longlive_base.pt + lora.pt

video-depth-anything:
  └─ daydreamlive/Video-Depth-Anything-Small (~50MB)

rife:
  └─ daydreamlive/RIFE                      (~30MB)

ltx2 (plugin):
  └─ defined in scope-ltx-2 plugin artifacts
```

---

## Phase 1: Docker Build & Dependency Installation

**Target: 25-48s saved. Eliminates runtime dependency resolution entirely.**

### Task 1.1: Add `uv sync --frozen` to Dockerfile.cloud

**Estimated savings: 15-25s**

Currently `Dockerfile.cloud` only runs `uv python install` (line 42) but never pre-installs Python packages. Every cold start pays for `uv run` to resolve and install 169 packages (~4GB) including torch, flash-attn, triton, and 12 nvidia-* packages.

**Current** (`Dockerfile.cloud:39-46`):
```dockerfile
COPY pyproject.toml uv.lock README.md .python-version LICENSE.md patches.pth .
RUN uv python install && chmod -R a+rX /root/.local/share/uv
COPY src/ /app/src/
```

**Change to**:
```dockerfile
COPY pyproject.toml uv.lock README.md .python-version LICENSE.md patches.pth .
RUN uv python install && chmod -R a+rX /root/.local/share/uv
COPY src/ /app/src/
RUN uv sync --frozen --extra livepeer --extra kafka
```

**Files**: `Dockerfile.cloud`

**Testing**:
- Build image locally: `docker build -f Dockerfile.cloud -t scope-test .`
- Verify venv exists: `docker run scope-test ls /app/.venv/bin/livepeer-runner`
- Verify torch is installed: `docker run scope-test /app/.venv/bin/python -c "import torch; print(torch.__version__)"`
- Time comparison: run `uv run daydream-scope --help` in old vs new image, measure startup

### Task 1.2: Split fal overlay for better layer caching

**Estimated savings: 2-3s per code-only deploy**

The fal overlay in `livepeer_fal_app.py` copies `pyproject.toml`, `uv.lock`, AND `src/` in one shot. Any code change in `src/` invalidates the layer that also has lockfile data, forcing a full dependency reinstall in the overlay.

**Current** (`livepeer_fal_app.py:265-269`):
```python
dockerfile_str = f"""
FROM {DOCKER_IMAGE}
WORKDIR /app
COPY pyproject.toml uv.lock README.md patches.pth /app/
COPY src/ /app/src/
"""
```

**Change to**:
```python
dockerfile_str = f"""
FROM {DOCKER_IMAGE}
WORKDIR /app
COPY pyproject.toml uv.lock README.md patches.pth /app/
RUN uv sync --frozen --extra livepeer --extra kafka
COPY src/ /app/src/
"""
```

**Files**: `src/scope/cloud/livepeer_fal_app.py`

**Testing**:
- Deploy to fal preview with a code-only change (modify a comment in `src/`)
- Verify the deploy skips the `uv sync` layer (cached) and only rebuilds the `COPY src/` layer
- Compare deploy time vs current

### Task 1.3: Remove `build-essential` from runtime image

**Estimated savings: 3-5s (200-400MB image reduction)**

`build-essential` (gcc, g++, make, dpkg-dev, libc6-dev) is only needed if compiling C extensions from source. With `uv sync` at build time (Task 1.1), all compilation happens in the build layer. At runtime, only the compiled `.so` files are needed.

**Current** (`Dockerfile.cloud:13-28`):
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  git \
  build-essential \
  software-properties-common \
  ...
```

**Change to** (multi-stage or just remove):
```dockerfile
RUN apt-get update && apt-get install -y --no-install-recommends \
  curl \
  git \
  libgl1 \
  libglib2.0-0 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  libgomp1 \
  && rm -rf /var/lib/apt/lists/*
```

Also remove `software-properties-common` (only needed for `add-apt-repository`, not used) and `python3-dev` (only needed for building C extensions).

**Files**: `Dockerfile.cloud`

**Testing**:
- Build image, verify size reduction: `docker images scope-test`
- Run full pipeline load test to ensure no missing `.so` dependencies
- `docker run scope-test ldd /app/.venv/lib/python3.12/site-packages/torch/lib/libtorch.so` — verify all shared libs resolve

---

## Phase 2: Process Startup

**Target: 12-21s saved. Reduces time from process start to uvicorn ready.**

### Task 2.1: Direct venv invocation, skip `uv run`

**Estimated savings: 5-8s**

`_build_runner_command()` uses `uv run --extra livepeer --extra kafka livepeer-runner`. Even with deps pre-installed (Phase 1), `uv run` still reads `uv.lock` (3519 lines), checks venv consistency, resolves extras, and verifies all 169 packages before exec'ing the actual Python process.

**Current** (`livepeer_fal_app.py:302-316`):
```python
def _build_runner_command() -> list[str]:
    return [
        "uv", "run", "--extra", "livepeer", "--extra", "kafka",
        "livepeer-runner", "--host", RUNNER_HOST, "--port", str(RUNNER_PORT),
    ]
```

**Change to**:
```python
def _build_runner_command() -> list[str]:
    return [
        "/app/.venv/bin/livepeer-runner",
        "--host", RUNNER_HOST, "--port", str(RUNNER_PORT),
    ]
```

**Prerequisite**: Task 1.1 (deps must be pre-installed in the venv).

**Files**: `src/scope/cloud/livepeer_fal_app.py`

**Testing**:
- Deploy to fal preview, connect via WS, verify runner responds normally
- Compare runner startup time in logs (look for `Livepeer runner ready at` timestamp delta)
- Verify that `livepeer-runner` entrypoint script exists in the venv and has correct shebang

### Task 2.2: Defer `import torch` from module level to lifespan

**Estimated savings: 5-8s**

The import chain `livepeer_app.py:43` → `scope.server.app:88` → `pipeline_manager.py:11` → `import torch` triggers torch loading (~3-5s) at **process start**, before uvicorn even begins binding its socket.

The comment at `app.py:373` says "Lazy imports to avoid loading torch at CLI startup" — but `app.py:88` defeats this by importing `PipelineManager` at module level (outside `TYPE_CHECKING`).

**Import chain today**:
```
livepeer_app.py:43  import scope.server.app
  → app.py:88   from .pipeline_manager import PipelineManager
    → pipeline_manager.py:11  import torch           ← 3-5s blocks here
livepeer_app.py:46  from scope.server.frame_processor import FrameProcessor
  → frame_processor.py:10  import torch               ← already cached, 0s
```

**Fix**: Move `from .pipeline_manager import PipelineManager` at `app.py:88` to be lazy (inside functions that use it, or only in `TYPE_CHECKING`). The runtime import already happens inside `lifespan()` at line 377.

Similarly in `livepeer_app.py:46`, defer `from scope.server.frame_processor import FrameProcessor` to first use inside the WS handler.

**Files**: `src/scope/server/app.py`, `src/scope/cloud/livepeer_app.py`

**Testing**:
- Run `time python -c "import scope.server.app"` before and after — should drop by 3-5s
- Run full integration test: deploy to fal preview, connect, load pipeline, stream
- Verify no `ImportError` at runtime for any endpoint that uses `PipelineManager`

### Task 2.3: Skip plugin update checks in cloud

**Estimated savings: 5-15s (CPU contention relief)**

`_prewarm_plugin_update_cache()` (app.py:426) fires as a background async task during lifespan. It calls `list_plugins_sync()` → `_check_plugin_update()` per plugin → spawns `uv pip compile ... --upgrade-package {name}` subprocess (manager.py:1072-1098) with 60s timeout.

While this doesn't block the ready signal directly, it competes for CPU and disk I/O during the critical startup window when torch is importing, CUDA is initializing, and uvicorn is binding.

The plugin update check is a desktop feature — cloud images pin exact versions at build time.

**Fix**: Add environment variable gate:
```python
# In app.py lifespan, after line 426:
if not os.getenv("SCOPE_SKIP_PLUGIN_UPDATE_CHECK"):
    asyncio.create_task(_prewarm_plugin_update_cache())
```

Set `SCOPE_SKIP_PLUGIN_UPDATE_CHECK=1` in `livepeer_fal_app.py` env allowlist and defaults.

**Files**: `src/scope/server/app.py`, `src/scope/cloud/livepeer_fal_app.py`

**Testing**:
- Set env var, start server, check logs for absence of "Plugin update check cache warmed"
- Verify plugin list endpoint still works (returns plugins without `update_available` field)
- Monitor CPU usage during startup: `top -p $(pgrep -f livepeer-runner)` — should show less CPU spike

### Task 2.4: Reduce runner readiness poll interval

**Estimated savings: 0.5-2s**

The `setup()` method polls `/docs` every 1 second (`time.sleep(1)` at line 475). The runner typically starts in 1-3s, so up to 1s is wasted on poll timing.

**Current** (`livepeer_fal_app.py:466-475`):
```python
while time.time() - start < RUNNER_STARTUP_TIMEOUT_SECONDS:
    ...
    if _runner_is_ready():
        return
    time.sleep(1)
```

**Change to**:
```python
    time.sleep(0.2)
```

**Files**: `src/scope/cloud/livepeer_fal_app.py`

**Testing**:
- Deploy to fal preview, check `setup()` timing in logs
- Verify no false-positive readiness (runner reported ready before actually accepting WS)

### Task 2.5: Remove `nvidia-smi` from setup()

**Estimated savings: 1-2s**

`setup()` runs `nvidia-smi` synchronously (livepeer_fal_app.py:401-411) and prints the full output. This is diagnostic — GPU availability is validated implicitly when torch initializes CUDA.

**Fix**: Remove the `nvidia-smi` block entirely, or run it in a background thread.

**Files**: `src/scope/cloud/livepeer_fal_app.py`

**Testing**:
- Deploy to fal preview, verify GPU is still detected by torch
- Check startup logs — `torch.cuda.is_available()` should still report True

---

## Phase 3: Model Downloads & Pipeline Pre-warming

**Target: 50-165s saved on first session. Eliminates model download and pipeline load from user-visible latency.**

### Task 3.1: Pre-download top pipeline models into Docker image

**Estimated savings: 30-120s on first session**

On a true cold start (new machine, empty `/data` volume), all models must be downloaded from HuggingFace. For longlive alone that's ~6.7GB. Video-depth-anything adds ~50MB, RIFE adds ~30MB.

**Option A — Bake into image** (recommended for top 3 pipelines):
```dockerfile
# After uv sync in Dockerfile.cloud
ENV DAYDREAM_SCOPE_MODELS_DIR=/app/models
RUN uv run download_models --pipeline longlive && \
    uv run download_models --pipeline video-depth-anything && \
    uv run download_models --pipeline rife
```

Image grows from ~4GB to ~11GB but eliminates all model downloads.

**Option B — Pre-download in setup()** (lower image size, still pays once per machine):
```python
def setup(self):
    # Pre-download models to /data/models if not cached
    for pipeline in ["longlive", "video-depth-anything", "rife"]:
        subprocess.run(["/app/.venv/bin/python", "-m", "scope.server.download_models",
                       "--pipeline", pipeline], env=runner_env)
    # Then start the runner
```

**Trade-off**: Option A adds ~7GB to image (increases pull time by ~10-15s on fal's internal network) but eliminates 30-120s of HuggingFace download. Net win on first cold start: 20-105s. Option B keeps the image small but doesn't help on the very first cold start per machine.

**Files**: `Dockerfile.cloud` (Option A) or `src/scope/cloud/livepeer_fal_app.py` (Option B)

**Testing**:
- Option A: Build image, verify model files exist: `docker run scope-test ls /app/models/Wan2.1-T2V-1.3B/`
- Option A: Start server with `DAYDREAM_SCOPE_MODELS_DIR=/app/models`, load longlive — should skip download
- Option B: Deploy to fal, connect first session, check logs for download activity
- Both: Verify models are byte-identical to HuggingFace originals (checksum spot-check)

### Task 3.2: Pre-warm longlive pipeline at startup

**Estimated savings: 15-30s on first session**

The `PIPELINE` env var mechanism already exists (`app.py:237,421-422`). When set, `prewarm_pipeline()` fires as a background async task during lifespan and loads the pipeline into GPU memory before the first user connects.

**Fix**: Set `PIPELINE=longlive` in the runner environment:
```python
# In livepeer_fal_app.py setup(), add to runner_env:
runner_env.setdefault("PIPELINE", "longlive")
```

The pipeline loads in the background while the runner reports "ready" to the fal wrapper. By the time the first WS session triggers a pipeline load, it's already done or nearly done.

The longlive pipeline `__init__` sequence:
1. Diffusion model (CausalWanModel) + VACE wrapper (~10-15s)
2. LoRA application (~1-2s)
3. Text encoder (UMT5, ~1.5GB) (~3-5s)
4. VAE (~2-3s)
5. Move to GPU + optional FP8 quantization (~2-5s)

Total: ~18-30s moved from user-visible to background startup.

**Risk**: GPU memory (~20GB) is consumed immediately. On H100 (80GB VRAM) this leaves ~60GB for concurrent operations — more than sufficient.

**Files**: `src/scope/cloud/livepeer_fal_app.py`

**Testing**:
- Deploy to fal preview with `PIPELINE=longlive`
- Connect via WS, request pipeline load — should return near-instantly
- Check `GET /api/v1/pipeline/status` before connecting — should show `"status": "loaded"`
- Monitor VRAM: first session should NOT show the "Loading diffusion model..." stage

### Task 3.3: Dummy inference warmup after pre-warm

**Estimated savings: 5-15s (first-frame perceived latency)**

After pipeline load, the first inference frame is 2-5x slower than steady state due to:
- CUDA kernel compilation and autotuning on first forward pass
- GPU memory allocation pattern establishment
- KV-cache initialization for longlive

**Fix**: Extend `prewarm_pipeline()` to run a single dummy forward pass after loading:
```python
async def prewarm_pipeline(pipeline_id: str):
    try:
        await asyncio.wait_for(
            pipeline_manager.load_pipelines([(pipeline_id, pipeline_id, None)]),
            timeout=300,
        )
        # Warm CUDA kernels with a dummy inference
        pipeline = pipeline_manager.get_pipeline(pipeline_id)
        if pipeline and hasattr(pipeline, '__call__'):
            # Run one forward pass with dummy input to trigger CUDA compilation
            import torch
            dummy = torch.zeros(1, 3, 320, 576, device='cuda', dtype=torch.bfloat16)
            pipeline(video=dummy, prompts=[{"text": "warmup", "weight": 1.0}])
    except Exception as e:
        logger.error(f"Error pre-warming pipeline {pipeline_id}: {e}")
```

**Files**: `src/scope/server/app.py`

**Testing**:
- Deploy with `PIPELINE=longlive`, connect, start stream
- Measure first-frame latency vs steady-state — should be within 2x (not 5x)
- Check logs for warmup completion before first WS session
- Verify no artifacts in first frame (dummy input shouldn't corrupt pipeline state)

### Task 3.4: Parallel pipeline loading for chained graphs

**Estimated savings: 5-10s for multi-pipeline configs**

Top config #2 (`video-depth → longlive → rife`) and #3 (`longlive → rife`) load multiple pipelines. Currently the graph executor loads them sequentially.

Since video-depth-anything, longlive, and rife have independent model weights and no shared state until graph wiring, they can load concurrently.

**Fix**: In the pipeline load path for graph-based sessions, submit all pipeline loads to a thread pool concurrently:
```python
# Instead of sequential:
for pipeline_id in unique_ids:
    await load_pipeline(pipeline_id)

# Use concurrent loading:
await asyncio.gather(*[load_pipeline(pid) for pid in unique_ids])
```

**Files**: `src/scope/server/pipeline_manager.py` (the `load_pipelines` method)

**Testing**:
- Load a multi-pipeline workflow (video-depth + longlive + rife) via HTTP API
- Compare total load time vs sequential: should be ~max(individual times) not sum
- Verify no race conditions: check that all pipelines report `"loaded"` in status
- Memory check: ensure concurrent loads don't exceed H100 VRAM (20 + 1 + 0.5 = ~21.5GB)

---

## Phase 4: Image Size Reduction (Advanced)

**Target: 5-10s saved on image pull. Higher risk, requires thorough testing.**

### Task 4.1: Evaluate slim base image

**Estimated savings: 5-10s**

`nvidia/cuda:12.8.0-cudnn-runtime-ubuntu24.04` is ~3.5GB compressed. Since torch installs its own CUDA runtime libraries via pip (nvidia-cublas-cu12, nvidia-cuda-runtime-cu12, nvidia-cudnn-cu12, etc. — 12 packages), the base image's CUDA + cuDNN are largely redundant.

**Investigation needed**:
- Build with `ubuntu:24.04` (~80MB) as base
- Set `LD_LIBRARY_PATH` to include venv nvidia package paths
- Run all 4 top pipelines end-to-end
- Verify flash-attn and sageattention find their CUDA deps

**Risk**: High. CUDA library resolution is fragile. Some libraries may depend on system-installed CUDA paths. Needs exhaustive testing across all pipelines.

**Files**: `Dockerfile.cloud`

**Testing**:
- Build slim image, attempt `python -c "import torch; print(torch.cuda.is_available())"`
- Load longlive pipeline, run inference, compare output quality
- Run flash-attn and sageattention tests
- Check for any missing `.so` with `ldd` on all torch extension modules

### Task 4.2: Pre-package bundled plugin as wheel

**Estimated savings: 1-3s**

The bundled plugin `scope-ltx-2` is installed via `git+https://` at build time (Dockerfile.cloud:51), which clones the full git repo. Packaging it as a wheel and copying it in would be faster and more deterministic.

**Files**: `Dockerfile.cloud`

**Testing**:
- Build wheel from scope-ltx-2 repo, copy into image, install with `uv pip install`
- Verify plugin appears in `GET /api/v1/plugins`
- Load ltx2 pipeline, verify it works

---

## Phase 5: Lazy Component Loading (Pipeline-Level)

**Target: 3-5s saved on pipeline load. Lower priority, pipeline-specific.**

### Task 5.1: Lazy-load text encoder on first prompt

**Estimated savings: 3-5s**

LongLive's `__init__` loads the UMT5 text encoder (~1.5GB, ~3-5s) unconditionally. In video-only mode, it's unused until the user provides a text prompt.

**Fix**: Defer text encoder loading to first `__call__` that includes a text prompt.

**Files**: `src/scope/core/pipelines/longlive/pipeline.py`

**Testing**:
- Load longlive in video mode, verify pipeline_loaded event timing is 3-5s faster
- Switch to text mode, verify text encoder loads on first prompt
- Verify no latency spike when switching from video → text mode is acceptable
- Check that the text encoder is properly placed on the correct device/dtype

### Task 5.2: Skip OSC/DMX server init in cloud

**Estimated savings: <1s direct, prevents potential UDP bind issues**

The lifespan unconditionally starts an OSC server and creates a DMX server. Neither is useful on cloud — there's no local network for hardware control of a fal machine.

**Fix**: Gate behind cloud detection:
```python
if not os.getenv("FAL_JOB_ID"):
    osc_server = OSCServer(osc_host, osc_port)
    ...
```

**Files**: `src/scope/server/app.py`

**Testing**:
- Set `FAL_JOB_ID=test`, start server, verify no OSC/DMX in logs
- Verify OSC/DMX still works in desktop mode (no env var set)

---

## Summary

| Phase | Tasks | Total Savings | Risk | Dependencies |
|-------|-------|---------------|------|--------------|
| **Phase 1**: Docker Build | 1.1, 1.2, 1.3 | **20-33s** | Low | None |
| **Phase 2**: Process Startup | 2.1, 2.2, 2.3, 2.4, 2.5 | **12-25s** | Low-Med | Phase 1 for Task 2.1 |
| **Phase 3**: Models & Pre-warm | 3.1, 3.2, 3.3, 3.4 | **55-175s** | Low-Med | Phase 1 |
| **Phase 4**: Image Size | 4.1, 4.2 | **6-13s** | High | Phase 1 |
| **Phase 5**: Lazy Loading | 5.1, 5.2 | **3-6s** | Low | None |
| | | **Total: 96-252s** | | |

### Recommended execution order

1. **Phase 1** (all tasks) — foundation, unblocks Phase 2 and 3
2. **Phase 2** (Tasks 2.1, 2.3, 2.2) — quick wins, low risk
3. **Phase 3** (Tasks 3.2, 3.1, 3.3) — biggest user-visible impact
4. **Phase 5** (Task 5.1) — pipeline-level refinement
5. **Phase 4** (Task 4.1) — only if further reduction needed, high testing cost

### Measuring success

Track these metrics per deploy:
- **Image pull + extract time**: `docker pull` duration on fal machines
- **setup() duration**: time from `Starting Livepeer runner wrapper setup` to `Livepeer runner ready`
- **First pipeline load**: `pipeline_loaded` Kafka event `load_duration_ms` for first session
- **First frame latency**: time from `stream_started` to first frame output in `session/metrics`
- **Total cold-start-to-first-frame**: end-to-end from machine allocation to first rendered frame
