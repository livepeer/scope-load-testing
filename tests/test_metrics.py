from loadtest.metrics import MetricsCollector
from loadtest.results import RunResult, PhaseTimings, ErrorCategory


def test_collector_creates_metrics():
    collector = MetricsCollector(push_url=None)
    assert collector.runs_total is not None
    assert collector.connect_duration is not None
    assert collector.first_frame_duration is not None
    assert collector.stream_fps_out is not None
    assert collector.failures_total is not None


def test_record_run_pass():
    collector = MetricsCollector(push_url=None)
    result = RunResult(
        scenario="longlive_t2v_1m",
        orchestrator_id="O-abc",
        passed=True,
        timings=PhaseTimings(connect_s=5.0, pipeline_load_s=15.0, first_frame_s=12.0),
        fps_samples=[9.0, 10.0],
        vram_samples=[8000.0],
        labels={"pipeline": "longlive", "mode": "t2v", "duration_class": "short"},
    )
    # Should not raise
    collector.record_run(result)


def test_record_run_fail():
    collector = MetricsCollector(push_url=None)
    result = RunResult(
        scenario="longlive_t2v_1m",
        orchestrator_id="O-abc",
        passed=False,
        error_category=ErrorCategory.RUNNER,
        error_message="OOM",
        timings=PhaseTimings(connect_s=5.0),
        labels={"pipeline": "longlive", "mode": "t2v", "duration_class": "short"},
    )
    # Should not raise
    collector.record_run(result)


def test_record_run_no_timings():
    collector = MetricsCollector(push_url=None)
    result = RunResult(
        scenario="longlive_t2v_1m",
        orchestrator_id="O-abc",
        passed=False,
        error_category=ErrorCategory.NETWORK,
        error_message="timeout",
        timings=PhaseTimings(),
        labels={"pipeline": "longlive", "mode": "t2v", "duration_class": "short"},
    )
    # Should handle None timings gracefully
    collector.record_run(result)


def test_push_no_url():
    collector = MetricsCollector(push_url=None)
    # Should no-op without error
    collector.push()


def test_record_run_missing_labels():
    """Labels default to 'unknown' when not provided."""
    collector = MetricsCollector(push_url=None)
    result = RunResult(
        scenario="test",
        orchestrator_id="O-abc",
        passed=True,
        timings=PhaseTimings(connect_s=1.0),
        # no labels dict
    )
    collector.record_run(result)


def test_update_budget_gauges():
    collector = MetricsCollector(push_url=None)
    collector.budget_runs_planned.labels(orchestrator_id="O-abc").set(10)
    collector.budget_runs_completed.labels(orchestrator_id="O-abc").set(7)
    collector.orchestrator_coverage_percent.set(85.0)
    # Should not raise


def test_update_drift_gauge():
    collector = MetricsCollector(push_url=None)
    collector.baseline_drift_percent.labels(
        metric_name="first_frame", scenario="longlive_t2v"
    ).set(0.15)
    # Should not raise
