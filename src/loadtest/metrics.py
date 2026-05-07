"""Prometheus metric definitions and push gateway integration."""

import logging

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    push_to_gateway,
)

from .results import RunResult

logger = logging.getLogger(__name__)

# Standard labels (3) on most metrics
STD_LABELS = ["orchestrator_id", "pipeline", "mode"]

# Extended labels (5) on counters that need scenario-level breakdown
EXT_LABELS = [*STD_LABELS, "scenario", "duration_class"]


class MetricsCollector:
    """Collects and pushes Prometheus metrics for load test runs."""

    def __init__(self, push_url: str | None, job_name: str = "scope_loadtest"):
        self._push_url = push_url
        self._job_name = job_name
        self._registry = CollectorRegistry()

        # Histograms (standard labels)
        self.connect_duration = Histogram(
            "connect_duration_seconds",
            "Cloud connect time",
            STD_LABELS,
            buckets=[5, 10, 20, 30, 60, 90, 120],
            registry=self._registry,
        )
        self.pipeline_load_duration = Histogram(
            "pipeline_load_seconds",
            "Pipeline load time",
            STD_LABELS,
            buckets=[10, 20, 30, 60, 120, 180, 300],
            registry=self._registry,
        )
        self.first_frame_duration = Histogram(
            "first_frame_seconds",
            "Prompt to first output frame",
            STD_LABELS,
            buckets=[5, 10, 15, 20, 30, 45, 60],
            registry=self._registry,
        )

        # Gauges (standard labels)
        self.stream_fps_out = Gauge(
            "stream_fps_out",
            "Output FPS during session",
            STD_LABELS,
            registry=self._registry,
        )
        self.vram_allocated_mb = Gauge(
            "vram_allocated_mb",
            "GPU memory usage on runner",
            STD_LABELS,
            registry=self._registry,
        )

        # Budget gauges (orchestrator only)
        self.budget_runs_planned = Gauge(
            "budget_runs_planned",
            "Planned runs per orchestrator per day",
            ["orchestrator_id"],
            registry=self._registry,
        )
        self.budget_runs_completed = Gauge(
            "budget_runs_completed",
            "Completed runs per orchestrator per day",
            ["orchestrator_id"],
            registry=self._registry,
        )
        self.orchestrator_coverage_percent = Gauge(
            "orchestrator_coverage_percent",
            "Percentage of orchestrators tested",
            registry=self._registry,
        )

        # Drift gauge (metric_name + scenario)
        self.baseline_drift_percent = Gauge(
            "baseline_drift_percent",
            "Current vs baseline deviation",
            ["metric_name", "scenario"],
            registry=self._registry,
        )

        # Counters (extended labels)
        self.runs_total = Counter(
            "runs_total",
            "Total runs attempted",
            [*EXT_LABELS, "result"],
            registry=self._registry,
        )
        self.failures_total = Counter(
            "failures_total",
            "Failures by category",
            [*EXT_LABELS, "category"],
            registry=self._registry,
        )
        self.frames_validated_total = Counter(
            "frames_validated_total",
            "Frame validation results",
            [*STD_LABELS, "result"],
            registry=self._registry,
        )
        self.prompt_sensitivity_checks_total = Counter(
            "prompt_sensitivity_checks_total",
            "Prompt sensitivity checks",
            [*STD_LABELS, "result"],
            registry=self._registry,
        )

    def _std_labels(self, result: RunResult) -> dict[str, str]:
        return {
            "orchestrator_id": result.orchestrator_id,
            "pipeline": result.labels.get("pipeline", "unknown"),
            "mode": result.labels.get("mode", "unknown"),
        }

    def _ext_labels(self, result: RunResult) -> dict[str, str]:
        return {
            **self._std_labels(result),
            "scenario": result.scenario,
            "duration_class": result.labels.get("duration_class", "unknown"),
        }

    def record_run(self, result: RunResult) -> None:
        """Record all metrics from a completed run."""
        std = self._std_labels(result)
        ext = self._ext_labels(result)
        result_label = "pass" if result.passed else "fail"

        # Counters
        self.runs_total.labels(**ext, result=result_label).inc()

        if not result.passed and result.error_category:
            self.failures_total.labels(**ext, category=result.error_category.value).inc()

        # Histograms
        if result.timings.connect_s is not None:
            self.connect_duration.labels(**std).observe(result.timings.connect_s)
        if result.timings.pipeline_load_s is not None:
            self.pipeline_load_duration.labels(**std).observe(
                result.timings.pipeline_load_s
            )
        if result.timings.first_frame_s is not None:
            self.first_frame_duration.labels(**std).observe(
                result.timings.first_frame_s
            )

        # Gauges
        if result.fps_samples:
            avg_fps = sum(result.fps_samples) / len(result.fps_samples)
            self.stream_fps_out.labels(**std).set(avg_fps)

        if result.vram_samples:
            self.vram_allocated_mb.labels(**std).set(result.vram_samples[-1])

    def push(self) -> None:
        """Push all collected metrics to the gateway."""
        if not self._push_url:
            logger.debug("No push URL configured, skipping metric push")
            return
        try:
            push_to_gateway(
                self._push_url, job=self._job_name, registry=self._registry
            )
            logger.info("Metrics pushed to %s", self._push_url)
        except Exception as e:
            logger.error("Failed to push metrics: %s", e)
