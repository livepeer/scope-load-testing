"""Run results, error taxonomy, and log capture."""

import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx


class ErrorCategory(str, Enum):
    NETWORK = "network"
    ORCHESTRATOR = "orchestrator"
    RUNNER = "runner"
    PROTOCOL = "protocol"


@dataclass
class PhaseTimings:
    connect_s: float | None = None
    pipeline_load_s: float | None = None
    first_frame_s: float | None = None
    stream_duration_s: float | None = None
    total_s: float | None = None


@dataclass
class RunResult:
    scenario: str
    orchestrator_id: str
    passed: bool
    timings: PhaseTimings = field(default_factory=PhaseTimings)
    error_category: ErrorCategory | None = None
    error_message: str | None = None
    fps_samples: list[float] = field(default_factory=list)
    vram_samples: list[float] = field(default_factory=list)
    frames_validated: int = 0
    frames_black: int = 0
    frames_corrupt: int = 0
    prompt_sensitivity_checks: int = 0
    prompt_sensitivity_failures: int = 0
    cold_start: bool | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def avg_fps(self) -> float | None:
        if not self.fps_samples:
            return None
        return sum(self.fps_samples) / len(self.fps_samples)

    @property
    def vram_growth_mb(self) -> float | None:
        if len(self.vram_samples) < 4:
            return None
        q = len(self.vram_samples) // 4
        first_q = sum(self.vram_samples[:q]) / q
        last_q = sum(self.vram_samples[-q:]) / q
        return last_q - first_q


def classify_error(
    error: Exception, response_text: str | None = None
) -> ErrorCategory:
    """Classify an exception into the error taxonomy."""
    if isinstance(
        error, (httpx.TimeoutException, httpx.ConnectError, ConnectionError, OSError)
    ):
        return ErrorCategory.NETWORK

    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        text = (response_text or "").lower()

        if status in (502, 503, 504):
            return ErrorCategory.ORCHESTRATOR
        if status == 500:
            runner_keywords = ("cuda", "oom", "out of memory", "pipeline", "torch")
            if any(kw in text for kw in runner_keywords):
                return ErrorCategory.RUNNER
            return ErrorCategory.RUNNER
        if status in (400, 422):
            return ErrorCategory.PROTOCOL
        return ErrorCategory.ORCHESTRATOR

    return ErrorCategory.PROTOCOL


def save_failure_logs(
    logs: str, orchestrator_id: str, scenario: str, data_dir: Path
) -> Path:
    """Save failure logs to disk. Returns the path written."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%H%M%S")
    failures_dir = data_dir / "failures" / date_str
    failures_dir.mkdir(parents=True, exist_ok=True)
    path = failures_dir / f"{orchestrator_id}_{scenario}_{ts}.log"
    path.write_text(logs)
    return path


def cleanup_old_failures(data_dir: Path, max_age_days: int = 7) -> int:
    """Remove failure log directories older than max_age_days. Returns count removed."""
    failures_dir = data_dir / "failures"
    if not failures_dir.exists():
        return 0
    cutoff = datetime.now(timezone.utc).date()
    removed = 0
    for day_dir in failures_dir.iterdir():
        if not day_dir.is_dir():
            continue
        try:
            dir_date = datetime.strptime(day_dir.name, "%Y-%m-%d").date()
            if (cutoff - dir_date).days > max_age_days:
                shutil.rmtree(day_dir)
                removed += 1
        except ValueError:
            continue
    return removed
