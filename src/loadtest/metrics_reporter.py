"""Report load test events to Daydream /v1/metrics endpoint.

Events are tagged with client_source="scope-loadtest" so they can be
filtered from production traffic in ClickHouse:

    WHERE data.client_source = 'scope-loadtest'
"""

import logging
import platform
import time
import uuid
from typing import Any

import httpx

from .results import RunResult

logger = logging.getLogger(__name__)

DEFAULT_METRICS_URL = "https://api.daydream.monster/v1/metrics"
MAX_BATCH_SIZE = 500
CLIENT_SOURCE = "scope-loadtest"
APP_NAME = "scope-loadtest"


class MetricsReporter:
    """Async HTTP client for the Daydream /v1/metrics endpoint."""

    def __init__(self, api_key: str, metrics_url: str | None = None):
        self._api_key = api_key
        self._url = metrics_url or DEFAULT_METRICS_URL
        self._host = platform.node()
        self._buffer: list[dict[str, Any]] = []
        self._backoff_s = 1.0

    def enqueue(self, event: dict[str, Any]) -> None:
        """Add an event to the buffer."""
        # Inject client_source into data if not present
        data = event.get("data", {})
        if isinstance(data, dict):
            data.setdefault("client_source", CLIENT_SOURCE)
        self._buffer.append(event)

    def enqueue_many(self, events: list[dict[str, Any]]) -> None:
        for event in events:
            self.enqueue(event)

    async def flush(self) -> int:
        """Flush buffered events to the endpoint. Returns count accepted."""
        if not self._buffer:
            return 0

        # Take up to MAX_BATCH_SIZE
        batch = self._buffer[:MAX_BATCH_SIZE]
        self._buffer = self._buffer[MAX_BATCH_SIZE:]

        payload = {
            "app": APP_NAME,
            "host": self._host,
            "events": batch,
        }

        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(30.0)) as client:
                resp = await client.post(
                    self._url,
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                )

            if resp.status_code == 200:
                result = resp.json()
                accepted = result.get("accepted", len(batch))
                self._backoff_s = 1.0
                logger.info("Metrics: %d events accepted by %s", accepted, self._url)
                return accepted

            if resp.status_code == 400:
                logger.error("Metrics: 400 bad request, dropping %d events: %s", len(batch), resp.text[:200])
                return 0

            if resp.status_code == 401:
                logger.error("Metrics: 401 unauthorized — check DAYDREAM_API_KEY")
                self._buffer = batch + self._buffer  # re-enqueue
                return 0

            if resp.status_code == 429:
                logger.warning("Metrics: rate limited, re-enqueuing %d events", len(batch))
                self._buffer = batch + self._buffer
                return 0

            # 5xx — retry with backoff
            logger.warning("Metrics: %d from endpoint, re-enqueuing %d events", resp.status_code, len(batch))
            self._buffer = batch + self._buffer
            return 0

        except (httpx.TimeoutException, httpx.ConnectError, OSError) as e:
            logger.warning("Metrics: network error (%s), re-enqueuing %d events", e, len(batch))
            self._buffer = batch + self._buffer
            return 0

    @property
    def pending(self) -> int:
        return len(self._buffer)


def _ts() -> str:
    """Unix millis timestamp string."""
    return str(int(time.time() * 1000))


def _common_data(result: RunResult, prompt_pool: str | None = None) -> dict[str, Any]:
    """Fields common to all events for a run."""
    return {
        "client_source": CLIENT_SOURCE,
        "timestamp": _ts(),
        "session_id": f"lt-{uuid.uuid4().hex[:12]}",
        "scenario": result.scenario,
        "pipeline": result.labels.get("pipeline", "unknown"),
        "mode": result.labels.get("mode", "unknown"),
        "duration_class": result.labels.get("duration_class", "unknown"),
        "orchestrator_id": result.orchestrator_id,
        "prompt_pool": prompt_pool,
    }


def build_run_events(
    result: RunResult, prompt_pool: str | None = None
) -> list[dict[str, Any]]:
    """Build network_events from a completed RunResult.

    Returns 2 events: started + completed (with all timing/quality data).
    """
    session_id = f"lt-{uuid.uuid4().hex[:12]}"
    ts = _ts()
    common = {
        "client_source": CLIENT_SOURCE,
        "session_id": session_id,
        "scenario": result.scenario,
        "pipeline": result.labels.get("pipeline", "unknown"),
        "mode": result.labels.get("mode", "unknown"),
        "duration_class": result.labels.get("duration_class", "unknown"),
        "orchestrator_id": result.orchestrator_id,
    }

    events = []

    # Event 1: run started
    events.append({
        "type": "loadtest_run_started",
        "timestamp": ts,
        "data": {
            **common,
            "prompt_pool": prompt_pool,
        },
    })

    # Event 2: run completed (with all metrics)
    completed_data: dict[str, Any] = {
        **common,
        "passed": result.passed,
        "connect_s": result.timings.connect_s,
        "first_frame_s": result.timings.first_frame_s,
        "stream_duration_s": result.timings.stream_duration_s,
        "total_s": result.timings.total_s,
        "cold_start": result.cold_start,
        "frames_validated": result.frames_validated,
        "frames_black": result.frames_black,
        "frames_corrupt": result.frames_corrupt,
        "prompt_sensitivity_checks": result.prompt_sensitivity_checks,
        "prompt_sensitivity_failures": result.prompt_sensitivity_failures,
        "prompt_pool": prompt_pool,
    }
    if not result.passed:
        completed_data["error_category"] = result.error_category.value if result.error_category else None
        completed_data["error_message"] = result.error_message

    events.append({
        "type": "loadtest_run_completed",
        "timestamp": _ts(),
        "data": completed_data,
    })

    return events
