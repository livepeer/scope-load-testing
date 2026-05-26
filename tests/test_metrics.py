from loadtest.metrics import MetricsCollector
from loadtest.results import RunResult, PhaseTimings, ErrorCategory


def test_collector_creates_metrics():
    collector = MetricsCollector(push_url=None)
    assert collector.runs_total is not None
    assert collector.connect_duration is not None
    assert collector.first_frame_duration is not None
    assert collector.run_duration is not None
    assert collector.stream_fps_out is not None
    assert collector.failures_total is not None
    assert collector.frames_captured is not None


def test_record_run_pass():
    collector = MetricsCollector(push_url=None)
    result = RunResult(
        scenario="longlive_t2v_1m",
        orchestrator_id="O-abc",
        passed=True,
        timings=PhaseTimings(connect_s=5.0, first_frame_s=1.2, stream_duration_s=60.0, total_s=66.2),
        frames_validated=6,
        labels={"pipeline": "longlive", "mode": "t2v", "duration_class": "short"},
    )
    collector.record_run(result)


def test_record_run_fail():
    collector = MetricsCollector(push_url=None)
    result = RunResult(
        scenario="longlive_t2v_1m",
        orchestrator_id="O-abc",
        passed=False,
        error_category=ErrorCategory.RUNNER,
        error_message="OOM",
        timings=PhaseTimings(connect_s=5.0, total_s=5.0),
        labels={"pipeline": "longlive", "mode": "t2v", "duration_class": "short"},
    )
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
    collector.record_run(result)


def test_push_no_url():
    collector = MetricsCollector(push_url=None)
    collector.push()


def test_record_run_missing_labels():
    collector = MetricsCollector(push_url=None)
    result = RunResult(
        scenario="test",
        orchestrator_id="O-abc",
        passed=True,
        timings=PhaseTimings(connect_s=1.0),
    )
    collector.record_run(result)


def test_update_budget():
    collector = MetricsCollector(push_url=None)
    collector.update_budget("sdk", planned=10, completed=7)


def test_update_drift_gauge():
    collector = MetricsCollector(push_url=None)
    collector.baseline_drift_percent.labels(
        metric_name="first_frame", scenario="longlive_t2v"
    ).set(0.15)


def test_record_run_with_frame_validation():
    collector = MetricsCollector(push_url=None)
    result = RunResult(
        scenario="longlive_v2v_5m",
        orchestrator_id="sdk",
        passed=True,
        timings=PhaseTimings(connect_s=13.0, first_frame_s=0.5, stream_duration_s=300.0, total_s=313.5),
        frames_validated=30,
        frames_black=1,
        frames_corrupt=0,
        prompt_sensitivity_checks=3,
        prompt_sensitivity_failures=0,
        labels={"pipeline": "longlive", "mode": "v2v", "duration_class": "mid"},
    )
    collector.record_run(result)


def test_record_multiple_runs_accumulate():
    """Counters should increment across multiple record_run calls."""
    collector = MetricsCollector(push_url=None)
    for i in range(3):
        result = RunResult(
            scenario="longlive_t2v_1m",
            orchestrator_id="sdk",
            passed=True,
            timings=PhaseTimings(connect_s=10.0 + i, first_frame_s=1.0, total_s=70.0),
            labels={"pipeline": "longlive", "mode": "t2v", "duration_class": "short"},
        )
        collector.record_run(result)
    # Counter should be 3 (not 1)
    val = collector.runs_total.labels(
        orchestrator_id="sdk", pipeline="longlive", mode="t2v",
        scenario="longlive_t2v_1m", duration_class="short", result="pass",
    )._value.get()
    assert val == 3.0
