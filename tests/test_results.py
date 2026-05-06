import pytest
import httpx
from pathlib import Path
from loadtest.results import (
    RunResult,
    ErrorCategory,
    PhaseTimings,
    classify_error,
    save_failure_logs,
    cleanup_old_failures,
)


def test_run_result_pass():
    result = RunResult(
        scenario="longlive_t2v_1m",
        orchestrator_id="O-abc",
        passed=True,
        timings=PhaseTimings(connect_s=5.0, pipeline_load_s=15.0, first_frame_s=12.0),
        fps_samples=[9.0, 10.0, 10.5],
        vram_samples=[8000.0, 8010.0, 8005.0],
    )
    assert result.passed is True
    assert result.error_category is None
    assert result.avg_fps == pytest.approx(9.833, abs=0.01)


def test_run_result_fail():
    result = RunResult(
        scenario="longlive_t2v_1m",
        orchestrator_id="O-abc",
        passed=False,
        error_category=ErrorCategory.RUNNER,
        error_message="CUDA out of memory",
        timings=PhaseTimings(connect_s=5.0),
    )
    assert result.passed is False
    assert result.error_category == ErrorCategory.RUNNER


def test_avg_fps_empty():
    result = RunResult(scenario="x", orchestrator_id="y", passed=True)
    assert result.avg_fps is None


def test_vram_growth_insufficient_samples():
    result = RunResult(scenario="x", orchestrator_id="y", passed=True, vram_samples=[100, 200])
    assert result.vram_growth_mb is None


def test_vram_growth_detected():
    # 12 samples: first quarter avg=100, last quarter avg=300 → growth=200
    result = RunResult(
        scenario="x", orchestrator_id="y", passed=True,
        vram_samples=[100, 100, 100, 150, 200, 200, 250, 250, 250, 300, 300, 300],
    )
    assert result.vram_growth_mb == pytest.approx(200.0)


def test_classify_timeout():
    assert classify_error(httpx.ReadTimeout("timeout")) == ErrorCategory.NETWORK


def test_classify_connect_error():
    assert classify_error(httpx.ConnectError("refused")) == ErrorCategory.NETWORK


def test_classify_connection_error():
    assert classify_error(ConnectionError("reset")) == ErrorCategory.NETWORK


def test_classify_http_502():
    err = httpx.HTTPStatusError(
        "Bad Gateway",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(502),
    )
    assert classify_error(err) == ErrorCategory.ORCHESTRATOR


def test_classify_http_503():
    err = httpx.HTTPStatusError(
        "Unavailable",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(503),
    )
    assert classify_error(err) == ErrorCategory.ORCHESTRATOR


def test_classify_http_500_cuda():
    err = httpx.HTTPStatusError(
        "error",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(500),
    )
    assert classify_error(err, response_text="CUDA out of memory") == ErrorCategory.RUNNER


def test_classify_http_500_generic():
    err = httpx.HTTPStatusError(
        "error",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(500),
    )
    assert classify_error(err) == ErrorCategory.RUNNER


def test_classify_http_400():
    err = httpx.HTTPStatusError(
        "bad request",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(400),
    )
    assert classify_error(err) == ErrorCategory.PROTOCOL


def test_classify_unknown():
    assert classify_error(ValueError("weird")) == ErrorCategory.PROTOCOL


def test_save_failure_logs(tmp_path: Path):
    path = save_failure_logs("error log content", "O-abc", "longlive_t2v_1m", tmp_path)
    assert path.exists()
    assert path.read_text() == "error log content"
    assert "O-abc" in path.name
    assert "longlive_t2v_1m" in path.name


def test_cleanup_old_failures(tmp_path: Path):
    failures_dir = tmp_path / "failures"
    (failures_dir / "2020-01-01").mkdir(parents=True)
    (failures_dir / "2020-01-01" / "test.log").write_text("old")
    (failures_dir / "2099-12-31").mkdir(parents=True)
    (failures_dir / "2099-12-31" / "test.log").write_text("future")

    removed = cleanup_old_failures(tmp_path, max_age_days=7)
    assert removed == 1
    assert not (failures_dir / "2020-01-01").exists()
    assert (failures_dir / "2099-12-31").exists()
