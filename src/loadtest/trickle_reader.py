"""Read telemetry events from the trickle events channel.

The trickle events channel is an HTTP long-polling JSONL stream at
{events_url}/{seq}. Each segment contains one or more JSON events.

Event types observed on the channel:
- runner_ready: runner provisioned
- stream_started: per-stream channels ready, includes channel URLs
- logs: runner log lines (includes MediaPublishStats, pipeline timing)
- pong: keepalive response
- telemetry: runtime metrics (stream_heartbeat, etc.) — from PR 1040
- api_response: pipeline load/session start responses

This module reads the channel in a background task and collects
telemetry data into a TrickleMetrics object for the executor to use.
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Regex to extract MediaPublishStats values from log lines
_MEDIA_STATS_RE = re.compile(
    r"MediaPublishStats\("
    r"elapsed_s=(?P<elapsed>[0-9.]+).*?"
    r"segments_completed=(?P<segments>\d+).*?"
    r"bytes_streamed_to_trickle=(?P<bytes>\d+)"
)

# Regex to extract pipeline load timing from log lines
_PIPELINE_LOAD_RE = re.compile(
    r"All (\d+) pipeline\(s\) load"
)


@dataclass
class TrickleMetrics:
    """Metrics collected from the trickle events channel."""
    runner_ready: bool = False
    runner_ready_at: float | None = None  # monotonic time
    stream_started: bool = False
    pipeline_loaded: bool = False
    channel_urls: dict[str, str] = field(default_factory=dict)

    # From MediaPublishStats log lines
    media_stats: list[dict[str, Any]] = field(default_factory=list)

    # From telemetry events (PR 1040)
    telemetry_events: list[dict[str, Any]] = field(default_factory=list)

    # Raw events for debugging
    all_events: list[dict[str, Any]] = field(default_factory=list)

    @property
    def has_telemetry(self) -> bool:
        return len(self.telemetry_events) > 0

    @property
    def latest_media_stats(self) -> dict[str, Any] | None:
        return self.media_stats[-1] if self.media_stats else None


class TrickleEventsReader:
    """Reads the trickle events channel in a background task."""

    def __init__(self, events_url: str, timeout: float = 5.0):
        self._events_url = events_url
        self._timeout = timeout
        self._metrics = TrickleMetrics()
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()
        self._seq = 0

    @property
    def metrics(self) -> TrickleMetrics:
        return self._metrics

    def start(self) -> None:
        """Start reading events in background."""
        self._stop.clear()
        self._task = asyncio.create_task(self._read_loop())

    async def stop(self) -> None:
        """Stop the reader."""
        self._stop.set()
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _read_loop(self) -> None:
        """Read sequential segments from the events channel."""
        async with httpx.AsyncClient(timeout=httpx.Timeout(self._timeout)) as client:
            while not self._stop.is_set():
                url = f"{self._events_url}/{self._seq}"
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200 and resp.text.strip():
                        for line in resp.text.strip().split("\n"):
                            line = line.strip()
                            if line:
                                self._process_line(line)
                        self._seq += 1
                    elif resp.status_code == 404:
                        # No more segments yet — wait and retry
                        await asyncio.sleep(1)
                    else:
                        await asyncio.sleep(1)
                except httpx.ReadTimeout:
                    # Long poll timeout — normal, retry same segment
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.debug("Trickle read error at seg %d: %s", self._seq, e)
                    await asyncio.sleep(2)

    def _process_line(self, line: str) -> None:
        """Process a single JSON line from the events channel."""
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            return

        evt_type = evt.get("type", "")
        self._metrics.all_events.append(evt)

        if evt_type == "runner_ready":
            self._metrics.runner_ready = True
            logger.debug("Trickle: runner_ready")

        elif evt_type == "stream_started":
            self._metrics.stream_started = True
            channels = evt.get("channels", [])
            for ch in channels:
                role = ch.get("role", ch.get("direction", ""))
                self._metrics.channel_urls[role] = ch.get("url", "")
            logger.debug("Trickle: stream_started (%d channels)", len(channels))

        elif evt_type == "telemetry":
            # PR 1040: {"type": "telemetry", "event": {...}}
            inner = evt.get("event", {})
            self._metrics.telemetry_events.append(inner)
            inner_type = inner.get("data", {}).get("type", "") if isinstance(inner.get("data"), dict) else ""
            logger.info("Trickle telemetry: %s", inner_type or json.dumps(inner)[:100])

        elif evt_type == "logs":
            # Extract metrics from log lines
            for log_line in evt.get("lines", []):
                self._parse_log_line(log_line)

        elif evt_type == "api_response":
            status = evt.get("status")
            if status == 200:
                self._metrics.pipeline_loaded = True

    def _parse_log_line(self, line: str) -> None:
        """Extract metrics from runner log lines."""
        # MediaPublishStats
        m = _MEDIA_STATS_RE.search(line)
        if m:
            self._metrics.media_stats.append({
                "elapsed_s": float(m.group("elapsed")),
                "segments_completed": int(m.group("segments")),
                "bytes_streamed": int(m.group("bytes")),
            })

        # Pipeline load
        if _PIPELINE_LOAD_RE.search(line):
            self._metrics.pipeline_loaded = True
