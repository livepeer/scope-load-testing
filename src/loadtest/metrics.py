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

STD_LABELS = ["orchestrator_id", "pipeline", "mode"]
EXT_LABELS = [*STD_LABELS, "scenario", "duration_class"]


class MetricsCollector:
    """Collects and pushes Prometheus metrics for load test runs.

    Uses push_to_gateway with method="POST" (add mode) instead of the
    default PUT (replace mode). This means counters and histograms
    accumulate across pushes rather than being reset each time.
    """

    def __init__(self, push_url: str | None, job_name: str = "scope_loadtest"):
        self._push_url = push_url
        self._job_name = job_name
        self._registry = CollectorRegistry()

        # Histograms
        self.connect_duration = Histogram(
            "connect_duration_seconds",
            "Cloud connect time",
            STD_LABELS,
            buckets=[5, 10, 20, 30, 60, 90, 120, 180, 300],
            registry=self._registry,
        )
        self.first_frame_duration = Histogram(
            "first_frame_seconds",
            "Time from stream start to first output frame",
            STD_LABELS,
            buckets=[0.5, 1, 2, 5, 10, 15, 20, 30, 45, 60],
            registry=self._registry,
        )
        self.run_duration = Histogram(
            "run_duration_seconds",
            "Total run duration",
            STD_LABELS,
            buckets=[30, 60, 120, 300, 600, 900, 1800],
            registry=self._registry,
        )

        # Gauges — last-observed values
        self.stream_fps_out = Gauge(
            "stream_fps_out",
            "Output FPS during last session",
            STD_LABELS,
            registry=self._registry,
        )
        self.frames_captured = Gauge(
            "frames_captured",
            "Frames captured in last session",
            STD_LABELS,
            registry=self._registry,
        )

        # Budget gauges
        self.budget_runs_planned = Gauge(
            "budget_runs_planned",
            "Planned runs per day",
            ["orchestrator_id"],
            registry=self._registry,
        )
        self.budget_runs_completed = Gauge(
            "budget_runs_completed",
            "Completed runs today",
            ["orchestrator_id"],
            registry=self._registry,
        )

        # Drift gauge
        self.baseline_drift_percent = Gauge(
            "baseline_drift_percent",
            "Current vs baseline deviation",
            ["metric_name", "scenario"],
            registry=self._registry,
        )

        # Counters
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

        # Counters (always increment)
        self.runs_total.labels(**ext, result=result_label).inc()

        if not result.passed and result.error_category:
            self.failures_total.labels(**ext, category=result.error_category.value).inc()

        # Histograms
        if result.timings.connect_s is not None:
            self.connect_duration.labels(**std).observe(result.timings.connect_s)
        if result.timings.first_frame_s is not None:
            self.first_frame_duration.labels(**std).observe(result.timings.first_frame_s)
        if result.timings.total_s is not None:
            self.run_duration.labels(**std).observe(result.timings.total_s)

        # Gauges (last-observed)
        if result.frames_validated > 0:
            self.frames_captured.labels(**std).set(result.frames_validated)

        # Estimate FPS from frames captured / stream duration
        if result.timings.stream_duration_s and result.timings.stream_duration_s > 0 and result.frames_validated > 0:
            estimated_fps = result.frames_validated / result.timings.stream_duration_s
            self.stream_fps_out.labels(**std).set(estimated_fps)

        # Frame validation counters
        if result.frames_validated > 0:
            valid = result.frames_validated - result.frames_black - result.frames_corrupt
            if valid > 0:
                self.frames_validated_total.labels(**std, result="valid").inc(valid)
            if result.frames_black > 0:
                self.frames_validated_total.labels(**std, result="black").inc(result.frames_black)
            if result.frames_corrupt > 0:
                self.frames_validated_total.labels(**std, result="corrupt").inc(result.frames_corrupt)

        # Prompt sensitivity
        if result.prompt_sensitivity_checks > 0:
            passed = result.prompt_sensitivity_checks - result.prompt_sensitivity_failures
            if passed > 0:
                self.prompt_sensitivity_checks_total.labels(**std, result="pass").inc(passed)
            if result.prompt_sensitivity_failures > 0:
                self.prompt_sensitivity_checks_total.labels(**std, result="fail").inc(result.prompt_sensitivity_failures)

    def update_budget(self, orchestrator_id: str, planned: int, completed: int) -> None:
        """Update budget gauges."""
        self.budget_runs_planned.labels(orchestrator_id=orchestrator_id).set(planned)
        self.budget_runs_completed.labels(orchestrator_id=orchestrator_id).set(completed)

    def push(self) -> None:
        """Push all collected metrics to the gateway (additive mode)."""
        if not self._push_url:
            logger.debug("No push URL configured, skipping metric push")
            return
        try:
            # Use POST handler (add/accumulate) instead of PUT (replace).
            # This preserves counter values across pushes.
            from prometheus_client import pushadd_to_gateway
            pushadd_to_gateway(
                self._push_url, job=self._job_name, registry=self._registry
            )
            logger.info("Metrics pushed to %s", self._push_url)
        except Exception as e:
            logger.error("Failed to push metrics: %s", e)
