import pytest
import httpx
import respx
from loadtest.scope_client import ScopeClient


@pytest.fixture
def base_url():
    return "http://scope-1:8001"


@respx.mock
@pytest.mark.asyncio
async def test_health(base_url):
    respx.get(f"{base_url}/health").respond(json={"status": "healthy"})
    async with ScopeClient(base_url) as c:
        r = await c.health()
    assert r["status"] == "healthy"


@respx.mock
@pytest.mark.asyncio
async def test_cloud_connect(base_url):
    respx.post(f"{base_url}/api/v1/cloud/connect").respond(
        json={"connected": False, "connecting": True, "webrtc_connected": False}
    )
    async with ScopeClient(base_url) as c:
        r = await c.cloud_connect(app_id="test-app")
    assert r["connecting"] is True


@respx.mock
@pytest.mark.asyncio
async def test_cloud_status(base_url):
    respx.get(f"{base_url}/api/v1/cloud/status").respond(
        json={"connected": True, "connecting": False, "webrtc_connected": True, "connection_id": "abc"}
    )
    async with ScopeClient(base_url) as c:
        r = await c.cloud_status()
    assert r["connected"] is True
    assert r["connection_id"] == "abc"


@respx.mock
@pytest.mark.asyncio
async def test_cloud_disconnect(base_url):
    respx.post(f"{base_url}/api/v1/cloud/disconnect").respond(
        json={"connected": False}
    )
    async with ScopeClient(base_url) as c:
        r = await c.cloud_disconnect()
    assert r["connected"] is False


@respx.mock
@pytest.mark.asyncio
async def test_pipeline_load(base_url):
    respx.post(f"{base_url}/api/v1/pipeline/load").respond(json={"status": "loading"})
    async with ScopeClient(base_url) as c:
        r = await c.pipeline_load(["longlive"])
    assert r["status"] == "loading"


@respx.mock
@pytest.mark.asyncio
async def test_pipeline_status(base_url):
    respx.get(f"{base_url}/api/v1/pipeline/status").respond(
        json={"status": "loaded", "pipeline_ids": ["longlive"]}
    )
    async with ScopeClient(base_url) as c:
        r = await c.pipeline_status()
    assert r["status"] == "loaded"


@respx.mock
@pytest.mark.asyncio
async def test_session_start(base_url):
    respx.post(f"{base_url}/api/v1/session/start").respond(
        json={"status": "ok", "graph": True, "sink_node_ids": ["output"]}
    )
    async with ScopeClient(base_url) as c:
        r = await c.session_start({"pipeline_id": "longlive", "input_mode": "text"})
    assert r["status"] == "ok"


@respx.mock
@pytest.mark.asyncio
async def test_session_stop(base_url):
    respx.post(f"{base_url}/api/v1/session/stop").respond(json={"status": "ok"})
    async with ScopeClient(base_url) as c:
        r = await c.session_stop()
    assert r["status"] == "ok"


@respx.mock
@pytest.mark.asyncio
async def test_session_metrics(base_url):
    respx.get(f"{base_url}/api/v1/session/metrics").respond(json={
        "sessions": {"headless": {"fps_out": 10.0, "frames_out": 50, "fps_in": 30.0}},
        "gpu": {"vram_allocated_mb": 8000, "vram_total_mb": 81920},
    })
    async with ScopeClient(base_url) as c:
        r = await c.session_metrics()
    assert r["sessions"]["headless"]["fps_out"] == 10.0
    assert r["gpu"]["vram_allocated_mb"] == 8000


@respx.mock
@pytest.mark.asyncio
async def test_session_parameters(base_url):
    respx.post(f"{base_url}/api/v1/session/parameters").respond(json={"status": "ok"})
    async with ScopeClient(base_url) as c:
        r = await c.session_parameters({"prompts": [{"text": "test", "weight": 100}]})
    assert r["status"] == "ok"


@respx.mock
@pytest.mark.asyncio
async def test_capture_frame(base_url):
    jpeg_bytes = b"\xff\xd8\xff\xe0fake_jpeg"
    respx.get(f"{base_url}/api/v1/session/frame").respond(
        content=jpeg_bytes, headers={"content-type": "image/jpeg"}
    )
    async with ScopeClient(base_url) as c:
        data = await c.capture_frame(sink_node_id="output")
    assert data == jpeg_bytes


@respx.mock
@pytest.mark.asyncio
async def test_capture_frame_no_sink(base_url):
    respx.get(f"{base_url}/api/v1/session/frame").respond(
        content=b"\xff\xd8data", headers={"content-type": "image/jpeg"}
    )
    async with ScopeClient(base_url) as c:
        data = await c.capture_frame()
    assert len(data) > 0


@respx.mock
@pytest.mark.asyncio
async def test_get_logs(base_url):
    respx.get(f"{base_url}/api/v1/logs/tail").respond(json={"logs": ["line1", "line2"]})
    async with ScopeClient(base_url) as c:
        r = await c.get_logs(lines=50)
    assert len(r["logs"]) == 2


@respx.mock
@pytest.mark.asyncio
async def test_client_raises_on_http_error(base_url):
    respx.get(f"{base_url}/health").respond(status_code=500)
    async with ScopeClient(base_url) as c:
        with pytest.raises(httpx.HTTPStatusError):
            await c.health()


@respx.mock
@pytest.mark.asyncio
async def test_client_not_entered_raises():
    c = ScopeClient("http://fake")
    with pytest.raises(RuntimeError, match="not entered"):
        await c.health()
