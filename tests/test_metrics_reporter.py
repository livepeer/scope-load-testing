import pytest
import respx
import httpx
from loadtest.metrics_reporter import (
    MetricsReporter,
    build_run_events,
    CLIENT_SOURCE,
    MAX_BATCH_SIZE,
)
from loadtest.results import RunResult, PhaseTimings, ErrorCategory


def _make_result(passed=True, **kwargs) -> RunResult:
    defaults = dict(
        scenario="longlive_v2v_1m",
        orchestrator_id="auto",
        passed=passed,
        timings=PhaseTimings(connect_s=13.0, first_frame_s=0.5, stream_duration_s=60.0, total_s=73.5),
        frames_validated=6,
        labels={"pipeline": "longlive", "mode": "v2v", "duration_class": "short"},
    )
    defaults.update(kwargs)
    return RunResult(**defaults)


# --- build_run_events ---


def test_build_events_pass():
    result = _make_result(passed=True)
    events = build_run_events(result, prompt_pool="nature")

    assert len(events) == 2
    assert events[0]["type"] == "loadtest_run_started"
    assert events[1]["type"] == "loadtest_run_completed"

    # All events have client_source
    for e in events:
        assert e["data"]["client_source"] == CLIENT_SOURCE

    # Completed event has timing data
    completed = events[1]["data"]
    assert completed["passed"] is True
    assert completed["connect_s"] == 13.0
    assert completed["first_frame_s"] == 0.5
    assert completed["scenario"] == "longlive_v2v_1m"
    assert completed["prompt_pool"] == "nature"
    assert "error_category" not in completed


def test_build_events_fail():
    result = _make_result(
        passed=False,
        error_category=ErrorCategory.NETWORK,
        error_message="timeout",
    )
    events = build_run_events(result)

    completed = events[1]["data"]
    assert completed["passed"] is False
    assert completed["error_category"] == "network"
    assert completed["error_message"] == "timeout"


def test_build_events_session_id_consistent():
    result = _make_result()
    events = build_run_events(result)

    # Both events share the same session_id
    assert events[0]["data"]["session_id"] == events[1]["data"]["session_id"]
    assert events[0]["data"]["session_id"].startswith("lt-")


def test_build_events_all_have_client_source():
    result = _make_result()
    events = build_run_events(result)
    for e in events:
        assert e["data"]["client_source"] == "scope-loadtest"


# --- MetricsReporter ---


def test_enqueue_injects_client_source():
    reporter = MetricsReporter(api_key="test", metrics_url="http://fake")
    reporter.enqueue({"type": "test", "data": {"foo": 1}})
    assert reporter.pending == 1


@respx.mock
@pytest.mark.asyncio
async def test_flush_200():
    url = "http://fake/v1/metrics"
    respx.post(url).respond(json={"status": "accepted", "accepted": 2})

    reporter = MetricsReporter(api_key="test", metrics_url=url)
    reporter.enqueue({"type": "t1", "data": {"a": 1}})
    reporter.enqueue({"type": "t2", "data": {"b": 2}})

    accepted = await reporter.flush()
    assert accepted == 2
    assert reporter.pending == 0


@respx.mock
@pytest.mark.asyncio
async def test_flush_400_drops_batch():
    url = "http://fake/v1/metrics"
    respx.post(url).respond(status_code=400, json={"error": "bad"})

    reporter = MetricsReporter(api_key="test", metrics_url=url)
    reporter.enqueue({"type": "t1", "data": {}})

    accepted = await reporter.flush()
    assert accepted == 0
    assert reporter.pending == 0  # dropped, not re-enqueued


@respx.mock
@pytest.mark.asyncio
async def test_flush_401_reenqueues():
    url = "http://fake/v1/metrics"
    respx.post(url).respond(status_code=401, json={"error": "unauthorized"})

    reporter = MetricsReporter(api_key="test", metrics_url=url)
    reporter.enqueue({"type": "t1", "data": {}})

    accepted = await reporter.flush()
    assert accepted == 0
    assert reporter.pending == 1  # kept for retry with new key


@respx.mock
@pytest.mark.asyncio
async def test_flush_429_reenqueues():
    url = "http://fake/v1/metrics"
    respx.post(url).respond(status_code=429)

    reporter = MetricsReporter(api_key="test", metrics_url=url)
    reporter.enqueue({"type": "t1", "data": {}})

    accepted = await reporter.flush()
    assert accepted == 0
    assert reporter.pending == 1


@respx.mock
@pytest.mark.asyncio
async def test_flush_502_reenqueues():
    url = "http://fake/v1/metrics"
    respx.post(url).respond(status_code=502)

    reporter = MetricsReporter(api_key="test", metrics_url=url)
    reporter.enqueue({"type": "t1", "data": {}})

    accepted = await reporter.flush()
    assert accepted == 0
    assert reporter.pending == 1


@respx.mock
@pytest.mark.asyncio
async def test_flush_network_error_reenqueues():
    url = "http://fake/v1/metrics"
    respx.post(url).mock(side_effect=httpx.ConnectError("refused"))

    reporter = MetricsReporter(api_key="test", metrics_url=url)
    reporter.enqueue({"type": "t1", "data": {}})

    accepted = await reporter.flush()
    assert accepted == 0
    assert reporter.pending == 1


@respx.mock
@pytest.mark.asyncio
async def test_flush_empty_buffer():
    reporter = MetricsReporter(api_key="test", metrics_url="http://fake")
    accepted = await reporter.flush()
    assert accepted == 0


@respx.mock
@pytest.mark.asyncio
async def test_flush_batch_size_limit():
    url = "http://fake/v1/metrics"
    respx.post(url).respond(json={"status": "accepted", "accepted": MAX_BATCH_SIZE})

    reporter = MetricsReporter(api_key="test", metrics_url=url)
    for i in range(MAX_BATCH_SIZE + 10):
        reporter.enqueue({"type": "t", "data": {"i": i}})

    assert reporter.pending == MAX_BATCH_SIZE + 10

    await reporter.flush()
    # First batch of 500 flushed, 10 remain
    assert reporter.pending == 10


@respx.mock
@pytest.mark.asyncio
async def test_flush_sends_auth_header():
    url = "http://fake/v1/metrics"
    route = respx.post(url).respond(json={"status": "accepted", "accepted": 1})

    reporter = MetricsReporter(api_key="sk_test_key_123", metrics_url=url)
    reporter.enqueue({"type": "t", "data": {}})
    await reporter.flush()

    assert route.called
    request = route.calls[0].request
    assert request.headers["authorization"] == "Bearer sk_test_key_123"


@respx.mock
@pytest.mark.asyncio
async def test_flush_sends_app_and_host():
    url = "http://fake/v1/metrics"
    route = respx.post(url).respond(json={"status": "accepted", "accepted": 1})

    reporter = MetricsReporter(api_key="test", metrics_url=url)
    reporter.enqueue({"type": "t", "data": {}})
    await reporter.flush()

    import json
    body = json.loads(route.calls[0].request.content)
    assert body["app"] == "scope-loadtest"
    assert "host" in body
