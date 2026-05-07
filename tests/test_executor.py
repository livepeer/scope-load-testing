import io
import pytest
import respx
from PIL import Image

from loadtest.config import LoadTestConfig
from loadtest.executor import Executor
from loadtest.scenarios import Scenario


def _make_jpeg(w: int, h: int, color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def _make_scenario(**overrides) -> Scenario:
    defaults = dict(
        name="longlive_t2v_1m",
        pipeline="longlive",
        mode="t2v",
        duration_mins=1,
        graph=None,
        prompts_pool="nature",
        parameters={"width": 512, "height": 512},
    )
    defaults.update(overrides)
    return Scenario(**defaults)


BASE = "http://scope-1:8001"


# --- Connect phase ---


@respx.mock
@pytest.mark.asyncio
async def test_connect_phase_success():
    respx.post(f"{BASE}/api/v1/cloud/connect").respond(
        json={"connected": False, "connecting": True}
    )
    respx.get(f"{BASE}/api/v1/cloud/status").respond(
        json={"connected": True, "connecting": False, "webrtc_connected": True}
    )

    executor = Executor(LoadTestConfig())
    duration = await executor._connect_phase(BASE, app_id="test-app")
    assert duration > 0


@respx.mock
@pytest.mark.asyncio
async def test_connect_phase_timeout():
    respx.post(f"{BASE}/api/v1/cloud/connect").respond(
        json={"connected": False, "connecting": True}
    )
    respx.get(f"{BASE}/api/v1/cloud/status").respond(
        json={"connected": False, "connecting": True, "error": None}
    )

    config = LoadTestConfig()
    config.thresholds.connect_timeout_s = 2
    executor = Executor(config)

    with pytest.raises(TimeoutError):
        await executor._connect_phase(BASE, app_id="test-app")


@respx.mock
@pytest.mark.asyncio
async def test_connect_phase_error():
    respx.post(f"{BASE}/api/v1/cloud/connect").respond(
        json={"connected": False, "connecting": True}
    )
    respx.get(f"{BASE}/api/v1/cloud/status").respond(
        json={"connected": False, "connecting": False, "error": "auth failed"}
    )

    executor = Executor(LoadTestConfig())
    with pytest.raises(RuntimeError, match="auth failed"):
        await executor._connect_phase(BASE, app_id="test-app")


# --- Load phase ---


@respx.mock
@pytest.mark.asyncio
async def test_load_phase_success():
    respx.post(f"{BASE}/api/v1/pipeline/load").respond(json={"status": "loading"})
    respx.get(f"{BASE}/api/v1/pipeline/status").respond(
        json={"status": "loaded", "pipeline_ids": ["longlive"]}
    )

    executor = Executor(LoadTestConfig())
    duration = await executor._load_phase(BASE, ["longlive"])
    assert duration > 0


@respx.mock
@pytest.mark.asyncio
async def test_load_phase_error():
    respx.post(f"{BASE}/api/v1/pipeline/load").respond(json={"status": "loading"})
    respx.get(f"{BASE}/api/v1/pipeline/status").respond(
        json={"status": "error", "error_message": "model not found"}
    )

    executor = Executor(LoadTestConfig())
    with pytest.raises(RuntimeError, match="model not found"):
        await executor._load_phase(BASE, ["longlive"])


@respx.mock
@pytest.mark.asyncio
async def test_load_phase_timeout():
    respx.post(f"{BASE}/api/v1/pipeline/load").respond(json={"status": "loading"})
    respx.get(f"{BASE}/api/v1/pipeline/status").respond(
        json={"status": "loading"}
    )

    config = LoadTestConfig()
    config.thresholds.pipeline_load_timeout_s = 2
    executor = Executor(config)

    with pytest.raises(TimeoutError):
        await executor._load_phase(BASE, ["longlive"])


# --- Full run ---


@respx.mock
@pytest.mark.asyncio
async def test_full_run_t2v_pass(tmp_path):
    # Connect
    respx.post(f"{BASE}/api/v1/cloud/connect").respond(
        json={"connecting": True}
    )
    respx.get(f"{BASE}/api/v1/cloud/status").respond(
        json={"connected": True, "connecting": False, "webrtc_connected": True}
    )
    # Load
    respx.post(f"{BASE}/api/v1/pipeline/load").respond(json={"status": "loading"})
    respx.get(f"{BASE}/api/v1/pipeline/status").respond(json={"status": "loaded"})
    # Session
    respx.post(f"{BASE}/api/v1/session/start").respond(json={"status": "ok"})
    respx.get(f"{BASE}/api/v1/session/metrics").respond(json={
        "sessions": {"headless": {"fps_out": 10.0, "frames_out": 50, "fps_in": 30.0}},
        "gpu": {"vram_allocated_mb": 8000, "vram_total_mb": 81920},
    })
    # Frame (valid, non-black)
    respx.get(f"{BASE}/api/v1/session/frame").respond(
        content=_make_jpeg(512, 512, (128, 64, 200)),
        headers={"content-type": "image/jpeg"},
    )
    # Parameters
    respx.post(f"{BASE}/api/v1/session/parameters").respond(json={"status": "ok"})
    # Stop + disconnect
    respx.post(f"{BASE}/api/v1/session/stop").respond(json={"status": "ok"})
    respx.post(f"{BASE}/api/v1/cloud/disconnect").respond(json={"connected": False})

    config = LoadTestConfig()
    config.thresholds.frame_check_interval_s = 5
    config.thresholds.prompt_switch_interval_s = 10
    scenario = _make_scenario(duration_mins=1)

    executor = Executor(config, data_dir=tmp_path)
    result = await executor.run(
        scope_url=BASE,
        orchestrator_id="O-test",
        scenario=scenario,
        prompts=["prompt A", "prompt B"],
        app_id="test-app",
    )

    assert result.passed is True
    assert result.timings.connect_s is not None
    assert result.timings.connect_s > 0
    assert result.timings.pipeline_load_s is not None
    assert result.timings.first_frame_s is not None
    assert result.timings.stream_duration_s is not None
    assert result.timings.total_s is not None
    assert len(result.fps_samples) > 0
    assert result.frames_validated > 0
    assert result.orchestrator_id == "O-test"
    assert result.scenario == "longlive_t2v_1m"
    assert result.labels["pipeline"] == "longlive"


@respx.mock
@pytest.mark.asyncio
async def test_full_run_connect_failure(tmp_path):
    """Run fails at connect phase — result captures error, no crash."""
    respx.post(f"{BASE}/api/v1/cloud/connect").respond(
        json={"connecting": True}
    )
    respx.get(f"{BASE}/api/v1/cloud/status").respond(
        json={"connected": False, "connecting": True}
    )
    # Cleanup calls should not crash even if session wasn't started
    respx.post(f"{BASE}/api/v1/session/stop").respond(json={"status": "ok"})
    respx.post(f"{BASE}/api/v1/cloud/disconnect").respond(json={"connected": False})
    respx.get(f"{BASE}/api/v1/logs/tail").respond(json={"logs": ["error line"]})

    config = LoadTestConfig()
    config.thresholds.connect_timeout_s = 2
    config.budget.max_run_duration_mins = 1
    scenario = _make_scenario()

    executor = Executor(config, data_dir=tmp_path)
    result = await executor.run(
        scope_url=BASE,
        orchestrator_id="O-fail",
        scenario=scenario,
        prompts=["test"],
        app_id="test-app",
    )

    assert result.passed is False
    assert result.error_category is not None
    assert result.error_message is not None
    assert result.timings.total_s is not None


@respx.mock
@pytest.mark.asyncio
async def test_full_run_stall_detection(tmp_path):
    """Stream stalls (fps_out=0 for too long) → failure."""
    # Connect
    respx.post(f"{BASE}/api/v1/cloud/connect").respond(json={"connecting": True})
    respx.get(f"{BASE}/api/v1/cloud/status").respond(
        json={"connected": True, "connecting": False, "webrtc_connected": True}
    )
    # Load
    respx.post(f"{BASE}/api/v1/pipeline/load").respond(json={"status": "loading"})
    respx.get(f"{BASE}/api/v1/pipeline/status").respond(json={"status": "loaded"})
    # Session starts but metrics show frames flowing initially then stalling
    respx.post(f"{BASE}/api/v1/session/start").respond(json={"status": "ok"})

    call_count = 0

    def metrics_side_effect(request):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            # First calls: frames flowing (for first_frame detection)
            return respx.MockResponse(json={
                "sessions": {"headless": {"fps_out": 10.0, "frames_out": 50}},
                "gpu": {"vram_allocated_mb": 8000, "vram_total_mb": 81920},
            })
        else:
            # After: stalled
            return respx.MockResponse(json={
                "sessions": {"headless": {"fps_out": 0, "frames_out": 50}},
                "gpu": {"vram_allocated_mb": 8000, "vram_total_mb": 81920},
            })

    respx.get(f"{BASE}/api/v1/session/metrics").mock(side_effect=metrics_side_effect)
    respx.get(f"{BASE}/api/v1/session/frame").respond(
        content=_make_jpeg(512, 512, (128, 64, 200)),
        headers={"content-type": "image/jpeg"},
    )
    respx.post(f"{BASE}/api/v1/session/stop").respond(json={"status": "ok"})
    respx.post(f"{BASE}/api/v1/cloud/disconnect").respond(json={"connected": False})
    respx.get(f"{BASE}/api/v1/logs/tail").respond(json={"logs": ["stall log"]})

    config = LoadTestConfig()
    config.thresholds.stall_timeout_s = 3
    config.thresholds.frame_check_interval_s = 2
    config.budget.max_run_duration_mins = 1

    executor = Executor(config, data_dir=tmp_path)
    result = await executor.run(
        scope_url=BASE,
        orchestrator_id="O-stall",
        scenario=_make_scenario(duration_mins=1),
        prompts=["test"],
        app_id="test-app",
    )

    assert result.passed is False
    assert "stall" in (result.error_message or "").lower()
