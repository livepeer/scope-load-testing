"""Rolling baseline management and drift detection."""

import json
import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .results import RunResult

logger = logging.getLogger(__name__)


@dataclass
class DriftResult:
    first_frame_drifted: bool = False
    first_frame_drift_pct: float = 0.0
    fps_drifted: bool = False
    fps_drift_pct: float = 0.0
    pipeline_load_drifted: bool = False
    pipeline_load_drift_pct: float = 0.0


class BaselineStore:
    """Manages baselines.json and history.json files."""

    def __init__(
        self,
        baselines_path: Path,
        history_path: Path,
        max_history_days: int = 7,
    ):
        self._baselines_path = baselines_path
        self._history_path = history_path
        self._max_history_days = max_history_days

    def load_baselines(self) -> dict:
        if not self._baselines_path.exists():
            return {}
        with open(self._baselines_path) as f:
            return json.load(f)

    def save_baselines(self, data: dict) -> None:
        self._baselines_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._baselines_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def load_history(self) -> list[dict]:
        if not self._history_path.exists():
            return []
        with open(self._history_path) as f:
            return json.load(f)

    def save_history(self, data: list[dict]) -> None:
        self._history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._history_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def append_history(self, entry: dict) -> None:
        history = self.load_history()
        history.append(entry)

        # Prune entries older than max_history_days
        cutoff = (
            datetime.now(timezone.utc).timestamp()
            - self._max_history_days * 86400
        )
        history = [e for e in history if e.get("timestamp", 0) > cutoff]

        self.save_history(history)


def update_baseline(store: BaselineStore, result: RunResult) -> None:
    """Add a run result to history and recompute baselines."""
    entry = {
        "scenario": result.scenario,
        "timestamp": result.timestamp.timestamp(),
        "first_frame_s": result.timings.first_frame_s,
        "pipeline_load_s": result.timings.pipeline_load_s,
        "avg_fps": result.avg_fps,
    }
    store.append_history(entry)

    # Recompute baselines for this scenario
    history = store.load_history()
    scenario_entries = [e for e in history if e["scenario"] == result.scenario]

    first_frame_values = [
        e["first_frame_s"]
        for e in scenario_entries
        if e.get("first_frame_s") is not None
    ]
    fps_values = [
        e["avg_fps"] for e in scenario_entries if e.get("avg_fps") is not None
    ]
    load_values = [
        e["pipeline_load_s"]
        for e in scenario_entries
        if e.get("pipeline_load_s") is not None
    ]

    baselines = store.load_baselines()
    baselines[result.scenario] = {
        "first_frame_p50": statistics.median(first_frame_values)
        if first_frame_values
        else None,
        "first_frame_p95": _percentile(first_frame_values, 0.95)
        if first_frame_values
        else None,
        "steady_fps_p50": statistics.median(fps_values) if fps_values else None,
        "pipeline_load_p50": statistics.median(load_values)
        if load_values
        else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(scenario_entries),
    }
    store.save_baselines(baselines)


def check_drift(
    store: BaselineStore, result: RunResult, threshold: float = 0.20
) -> DriftResult:
    """Compare a run result against the rolling baseline."""
    baselines = store.load_baselines()
    baseline = baselines.get(result.scenario)
    drift = DriftResult()

    if not baseline or baseline.get("sample_count", 0) < 5:
        return drift

    # First frame drift (higher = worse)
    p50_ff = baseline.get("first_frame_p50")
    if p50_ff and p50_ff > 0 and result.timings.first_frame_s is not None:
        pct = (result.timings.first_frame_s - p50_ff) / p50_ff
        drift.first_frame_drift_pct = pct
        drift.first_frame_drifted = pct > threshold

    # FPS drift (lower = worse)
    p50_fps = baseline.get("steady_fps_p50")
    avg_fps = result.avg_fps
    if p50_fps and p50_fps > 0 and avg_fps is not None:
        pct = (p50_fps - avg_fps) / p50_fps
        drift.fps_drift_pct = pct
        drift.fps_drifted = pct > threshold

    # Pipeline load drift (higher = worse)
    p50_load = baseline.get("pipeline_load_p50")
    if p50_load and p50_load > 0 and result.timings.pipeline_load_s is not None:
        pct = (result.timings.pipeline_load_s - p50_load) / p50_load
        drift.pipeline_load_drift_pct = pct
        drift.pipeline_load_drifted = pct > threshold

    return drift


def _percentile(values: list[float], pct: float) -> float:
    """Compute a percentile from a list of values."""
    sorted_vals = sorted(values)
    idx = int(len(sorted_vals) * pct)
    idx = min(idx, len(sorted_vals) - 1)
    return sorted_vals[idx]
