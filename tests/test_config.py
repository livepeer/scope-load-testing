import pytest
from pathlib import Path
from loadtest.config import load_config, LoadTestConfig, BudgetConfig


def test_load_config_from_yaml(tmp_path: Path):
    yaml_content = """
budget:
  daily_percent: 20
  max_run_duration_mins: 30
  min_run_gap_mins: 15
  schedule_start: "00:00"
  schedule_end: "23:59"

thresholds:
  connect_timeout_s: 120
  pipeline_load_timeout_s: 300
  first_frame_timeout_s: 60
  stall_timeout_s: 10
  min_fps: 6
  max_vram_percent: 90
  vram_leak_tolerance_mb: 200
  prompt_diff_min: 10.0
  regression_drift_threshold: 0.20
  frame_variance_min: 5.0
  frame_check_interval_s: 30
  prompt_switch_interval_s: 30
  cold_start_threshold_s: 60

discovery:
  refresh_interval_hours: 4
  health_check_timeout_s: 30
  max_consecutive_failures: 5

scenarios:
  - pipeline: longlive
    modes: [t2v, v2v]
    durations: [1, 5]
    prompts_pool: nature
    parameters:
      width: 512
      height: 512
"""
    (tmp_path / "config.yaml").write_text(yaml_content)
    config = load_config(tmp_path / "config.yaml")

    assert isinstance(config, LoadTestConfig)
    assert config.budget.daily_percent == 20
    assert config.budget.max_run_duration_mins == 30
    assert config.thresholds.connect_timeout_s == 120
    assert config.thresholds.min_fps == 6
    assert config.thresholds.cold_start_threshold_s == 60
    assert config.discovery.refresh_interval_hours == 4
    assert len(config.scenario_defs) == 1
    assert config.scenario_defs[0]["pipeline"] == "longlive"


def test_load_config_defaults():
    config = load_config(None)
    assert config.budget.daily_percent == 20
    assert config.thresholds.connect_timeout_s == 120
    assert config.thresholds.cold_start_threshold_s == 60


def test_budget_validates_percent():
    with pytest.raises(ValueError):
        BudgetConfig(daily_percent=0)
    with pytest.raises(ValueError):
        BudgetConfig(daily_percent=101)


def test_budget_runs_per_day():
    b = BudgetConfig(daily_percent=20, max_run_duration_mins=30)
    assert b.runs_per_orchestrator_per_day == 9  # 4.8hrs / 0.5hr = 9.6 -> 9
