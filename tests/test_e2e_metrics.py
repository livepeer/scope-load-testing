"""E2E test for Daydream /v1/metrics endpoint.

Skipped unless DAYDREAM_API_KEY is set. Configure via env vars:

    DAYDREAM_API_KEY  — required, Daydream API key
    METRICS_URL       — optional, defaults to api.daydream.monster
    METRICS_API_KEY   — optional, separate key for metrics endpoint
                        (useful for preview/staging environments)

Run:
    DAYDREAM_API_KEY=sk_... pytest tests/test_e2e_metrics.py -v
    METRICS_URL=https://pipelines-api-staging.fly.dev DAYDREAM_API_KEY=sk_... pytest tests/test_e2e_metrics.py -v
"""

import os
import time
import uuid

import pytest
import httpx

METRICS_URL = os.environ.get("METRICS_URL", "https://api.daydream.monster/v1/metrics")
API_KEY = os.environ.get("METRICS_API_KEY") or os.environ.get("DAYDREAM_API_KEY", "")

skip_no_key = pytest.mark.skipif(not API_KEY, reason="DAYDREAM_API_KEY not set")


async def _probe_endpoint() -> bool:
    """Check if the metrics endpoint is reachable and not returning 502."""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                METRICS_URL,
                json={"app": "probe", "events": [{"type": "probe", "data": {}}]},
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
            )
            # 200, 400, 401, 429 = endpoint exists. 502 = not deployed.
            return resp.status_code != 502
    except Exception:
        return False


@skip_no_key
@pytest.mark.asyncio
async def test_metrics_endpoint_accepts_loadtest_events():
    """Send a test batch and verify 200 accepted."""
    if not await _probe_endpoint():
        pytest.xfail("v1/metrics endpoint not deployed (502)")

    event_id = f"e2e-test-{uuid.uuid4().hex[:8]}"
    ts = str(int(time.time() * 1000))

    payload = {
        "app": "scope-loadtest",
        "host": "e2e-test",
        "events": [
            {
                "id": event_id,
                "type": "loadtest_e2e_ping",
                "timestamp": ts,
                "data": {
                    "client_source": "scope-loadtest",
                    "e2e_test": True,
                    "scenario": "e2e_verification",
                },
            },
        ],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            METRICS_URL,
            json=payload,
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        )

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    body = resp.json()
    assert body.get("status") == "accepted"
    assert body.get("accepted") == 1


@skip_no_key
@pytest.mark.asyncio
async def test_metrics_endpoint_rejects_no_auth():
    """Verify 401 without auth header."""
    if not await _probe_endpoint():
        pytest.xfail("v1/metrics endpoint not deployed (502)")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            METRICS_URL,
            json={"app": "test", "events": [{"type": "t", "data": {}}]},
            headers={"Content-Type": "application/json"},
        )
    assert resp.status_code == 401


@skip_no_key
@pytest.mark.asyncio
async def test_metrics_endpoint_rejects_empty_events():
    """Verify 400 for empty events array."""
    if not await _probe_endpoint():
        pytest.xfail("v1/metrics endpoint not deployed (502)")

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            METRICS_URL,
            json={"app": "test", "events": []},
            headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
        )
    assert resp.status_code == 400


@skip_no_key
@pytest.mark.asyncio
async def test_client_source_is_scope_loadtest():
    """Verify our events carry the correct client_source tag."""
    from loadtest.metrics_reporter import build_run_events, CLIENT_SOURCE
    from loadtest.results import RunResult, PhaseTimings

    result = RunResult(
        scenario="e2e_test",
        orchestrator_id="auto",
        passed=True,
        timings=PhaseTimings(connect_s=1.0, first_frame_s=0.5, total_s=2.0),
        labels={"pipeline": "longlive", "mode": "t2v", "duration_class": "short"},
    )
    events = build_run_events(result, prompt_pool="nature")

    for event in events:
        assert event["data"]["client_source"] == CLIENT_SOURCE
        assert event["data"]["client_source"] == "scope-loadtest"


@skip_no_key
@pytest.mark.asyncio
async def test_full_reporter_flow():
    """Test MetricsReporter enqueue + flush against live endpoint."""
    if not await _probe_endpoint():
        pytest.xfail("v1/metrics endpoint not deployed (502)")

    from loadtest.metrics_reporter import MetricsReporter, build_run_events
    from loadtest.results import RunResult, PhaseTimings

    result = RunResult(
        scenario="e2e_reporter_test",
        orchestrator_id="auto",
        passed=True,
        timings=PhaseTimings(connect_s=5.0, first_frame_s=1.0, stream_duration_s=10.0, total_s=16.0),
        frames_validated=3,
        labels={"pipeline": "longlive", "mode": "t2v", "duration_class": "short"},
    )

    reporter = MetricsReporter(api_key=API_KEY, metrics_url=METRICS_URL)
    events = build_run_events(result, prompt_pool="e2e-test")
    reporter.enqueue_many(events)

    accepted = await reporter.flush()
    assert accepted == len(events), f"Expected {len(events)} accepted, got {accepted}"
    assert reporter.pending == 0
