import json
import pytest
from pathlib import Path
from loadtest.regression import BaselineStore, check_drift, update_baseline
from loadtest.results import RunResult, PhaseTimings


def _make_result(
    scenario: str = "longlive_t2v",
    first_frame_s: float | None = 12.0,
    pipeline_load_s: float | None = 15.0,
    fps: float | None = 10.0,
) -> RunResult:
    return RunResult(
        scenario=scenario,
        orchestrator_id="O-abc",
        passed=True,
        timings=PhaseTimings(first_frame_s=first_frame_s, pipeline_load_s=pipeline_load_s),
        fps_samples=[fps] if fps is not None else [],
    )


def test_update_baseline_new(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")
    update_baseline(store, _make_result())

    baselines = store.load_baselines()
    assert "longlive_t2v" in baselines
    assert baselines["longlive_t2v"]["sample_count"] == 1
    assert baselines["longlive_t2v"]["first_frame_p50"] == 12.0
    assert baselines["longlive_t2v"]["steady_fps_p50"] == 10.0


def test_update_baseline_accumulates(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")
    update_baseline(store, _make_result(first_frame_s=10.0))
    update_baseline(store, _make_result(first_frame_s=14.0))
    update_baseline(store, _make_result(first_frame_s=12.0))

    baselines = store.load_baselines()
    assert baselines["longlive_t2v"]["sample_count"] == 3
    assert baselines["longlive_t2v"]["first_frame_p50"] == 12.0  # median of [10, 12, 14]


def test_update_baseline_multiple_scenarios(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")
    update_baseline(store, _make_result(scenario="longlive_t2v", first_frame_s=10.0))
    update_baseline(store, _make_result(scenario="ltx2_t2v", first_frame_s=20.0))

    baselines = store.load_baselines()
    assert "longlive_t2v" in baselines
    assert "ltx2_t2v" in baselines
    assert baselines["longlive_t2v"]["first_frame_p50"] == 10.0
    assert baselines["ltx2_t2v"]["first_frame_p50"] == 20.0


def test_check_drift_no_regression(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")

    for _ in range(10):
        update_baseline(store, _make_result(first_frame_s=12.0, fps=10.0))

    current = _make_result(first_frame_s=13.0, fps=9.5)
    drift = check_drift(store, current, threshold=0.20)

    assert not drift.first_frame_drifted
    assert not drift.fps_drifted
    assert not drift.pipeline_load_drifted


def test_check_drift_first_frame_regression(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")

    for _ in range(10):
        update_baseline(store, _make_result(first_frame_s=12.0))

    # 67% worse
    current = _make_result(first_frame_s=20.0)
    drift = check_drift(store, current, threshold=0.20)

    assert drift.first_frame_drifted
    assert drift.first_frame_drift_pct > 0.5


def test_check_drift_fps_regression(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")

    for _ in range(10):
        update_baseline(store, _make_result(fps=10.0))

    # 50% worse
    current = _make_result(fps=5.0)
    drift = check_drift(store, current, threshold=0.20)

    assert drift.fps_drifted
    assert drift.fps_drift_pct > 0.4


def test_check_drift_pipeline_load_regression(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")

    for _ in range(10):
        update_baseline(store, _make_result(pipeline_load_s=15.0))

    current = _make_result(pipeline_load_s=25.0)
    drift = check_drift(store, current, threshold=0.20)

    assert drift.pipeline_load_drifted


def test_check_drift_insufficient_data(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")

    # Only 3 samples — below the minimum 5
    for _ in range(3):
        update_baseline(store, _make_result())

    current = _make_result(first_frame_s=100.0, fps=1.0)
    drift = check_drift(store, current, threshold=0.20)

    # Not enough data → no drift flagged
    assert not drift.first_frame_drifted
    assert not drift.fps_drifted


def test_check_drift_no_baseline(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")

    current = _make_result(scenario="unknown_scenario")
    drift = check_drift(store, current)

    assert not drift.first_frame_drifted
    assert not drift.fps_drifted


def test_history_pruning(tmp_path: Path):
    store = BaselineStore(
        tmp_path / "baselines.json", tmp_path / "history.json", max_history_days=7
    )

    # Manually inject old entries
    old_entry = {
        "scenario": "longlive_t2v",
        "timestamp": 0,  # epoch = 1970, definitely > 7 days ago
        "first_frame_s": 99.0,
        "pipeline_load_s": 99.0,
        "avg_fps": 1.0,
    }
    store.save_history([old_entry])

    # Append a new one — old should be pruned
    update_baseline(store, _make_result(first_frame_s=12.0))

    history = store.load_history()
    # Only the new entry should remain
    assert len(history) == 1
    assert history[0]["first_frame_s"] == 12.0


def test_baseline_store_persistence(tmp_path: Path):
    store1 = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")
    update_baseline(store1, _make_result())

    store2 = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")
    baselines = store2.load_baselines()
    assert "longlive_t2v" in baselines
    assert baselines["longlive_t2v"]["sample_count"] == 1
