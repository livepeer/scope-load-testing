from pathlib import Path
from loadtest.coverage import CoverageTracker


def test_record_run(tmp_path: Path):
    tracker = CoverageTracker(tmp_path / "coverage.json")
    tracker.record_run("O-abc", "longlive_t2v_1m", passed=True)
    tracker.record_run("O-abc", "ltx2_t2v_5m", passed=False, failure_category="runner")

    day = tracker.get_today()
    assert day["O-abc"]["runs_completed"] == 2
    assert day["O-abc"]["failures"] == 1
    assert "longlive_t2v_1m" in day["O-abc"]["scenarios_covered"]
    assert "ltx2_t2v_5m" in day["O-abc"]["scenarios_covered"]
    assert day["O-abc"]["failure_categories"]["runner"] == 1


def test_set_planned(tmp_path: Path):
    tracker = CoverageTracker(tmp_path / "coverage.json")
    tracker.set_planned("O-abc", 10)
    tracker.set_planned("O-def", 10)

    day = tracker.get_today()
    assert day["O-abc"]["runs_planned"] == 10
    assert day["O-def"]["runs_planned"] == 10


def test_test_debt(tmp_path: Path):
    tracker = CoverageTracker(tmp_path / "coverage.json")
    tracker.set_planned("O-abc", 10)
    tracker.record_run("O-abc", "s1", passed=True)
    tracker.record_run("O-abc", "s2", passed=True)

    tracker.set_planned("O-def", 10)
    # O-def has 0 runs -> higher debt

    debt = tracker.get_test_debt()
    assert debt["O-def"] > debt["O-abc"]
    assert debt["O-def"] == 10
    assert debt["O-abc"] == 8


def test_test_debt_no_negative(tmp_path: Path):
    tracker = CoverageTracker(tmp_path / "coverage.json")
    tracker.set_planned("O-abc", 2)
    tracker.record_run("O-abc", "s1", passed=True)
    tracker.record_run("O-abc", "s2", passed=True)
    tracker.record_run("O-abc", "s3", passed=True)  # over budget

    debt = tracker.get_test_debt()
    assert debt["O-abc"] == 0


def test_persistence(tmp_path: Path):
    path = tmp_path / "coverage.json"
    tracker1 = CoverageTracker(path)
    tracker1.record_run("O-abc", "s1", passed=True)

    tracker2 = CoverageTracker(path)
    day = tracker2.get_today()
    assert day["O-abc"]["runs_completed"] == 1


def test_scenario_dedup(tmp_path: Path):
    tracker = CoverageTracker(tmp_path / "coverage.json")
    tracker.record_run("O-abc", "s1", passed=True)
    tracker.record_run("O-abc", "s1", passed=True)  # same scenario again

    day = tracker.get_today()
    assert day["O-abc"]["scenarios_covered"].count("s1") == 1
    assert day["O-abc"]["runs_completed"] == 2


def test_pruning(tmp_path: Path):
    path = tmp_path / "coverage.json"
    tracker = CoverageTracker(path, max_days=30)

    # Inject old data manually
    import json
    data = {
        "2020-01-01": {"O-old": {"runs_completed": 5, "runs_planned": 10, "scenarios_covered": [], "failures": 0, "failure_categories": {}}},
        "2099-12-31": {"O-future": {"runs_completed": 1, "runs_planned": 10, "scenarios_covered": [], "failures": 0, "failure_categories": {}}},
    }
    path.write_text(json.dumps(data))

    # Reload — old data should be pruned
    tracker2 = CoverageTracker(path, max_days=30)
    raw = tracker2._data
    assert "2020-01-01" not in raw
    assert "2099-12-31" in raw


def test_multiple_orchestrators(tmp_path: Path):
    tracker = CoverageTracker(tmp_path / "coverage.json")
    tracker.set_planned("O-1", 5)
    tracker.set_planned("O-2", 5)
    tracker.set_planned("O-3", 5)
    tracker.record_run("O-1", "s1", passed=True)
    tracker.record_run("O-2", "s1", passed=True)
    tracker.record_run("O-2", "s2", passed=True)

    debt = tracker.get_test_debt()
    assert debt["O-3"] == 5  # highest debt
    assert debt["O-1"] == 4
    assert debt["O-2"] == 3
