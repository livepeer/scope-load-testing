"""Configuration loading and validation."""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BudgetConfig:
    daily_percent: int = 20
    max_run_duration_mins: int = 30
    min_run_gap_mins: int = 15
    schedule_start: str = "00:00"
    schedule_end: str = "23:59"

    def __post_init__(self):
        if not 1 <= self.daily_percent <= 100:
            raise ValueError(f"daily_percent must be 1-100, got {self.daily_percent}")

    @property
    def runs_per_orchestrator_per_day(self) -> int:
        daily_minutes = 24 * 60 * (self.daily_percent / 100)
        return max(1, int(daily_minutes / self.max_run_duration_mins))


@dataclass
class ThresholdsConfig:
    connect_timeout_s: int = 120
    pipeline_load_timeout_s: int = 300
    first_frame_timeout_s: int = 60
    stall_timeout_s: int = 10
    min_fps: float = 6.0
    max_vram_percent: float = 90.0
    vram_leak_tolerance_mb: float = 200.0
    prompt_diff_min: float = 10.0
    regression_drift_threshold: float = 0.20
    frame_variance_min: float = 5.0
    frame_check_interval_s: int = 30
    prompt_switch_interval_s: int = 30
    cold_start_threshold_s: int = 60


@dataclass
class DiscoveryConfig:
    refresh_interval_hours: int = 4
    health_check_timeout_s: int = 30
    max_consecutive_failures: int = 5


@dataclass
class LoadTestConfig:
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    discovery: DiscoveryConfig = field(default_factory=DiscoveryConfig)
    scenario_defs: list[dict[str, Any]] = field(default_factory=list)


def _dict_to_dataclass(cls, data: dict):
    """Create a dataclass from a dict, ignoring unknown keys."""
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def load_config(config_path: Path | None) -> LoadTestConfig:
    """Load config from YAML file, falling back to defaults."""
    if config_path is None or not config_path.exists():
        return LoadTestConfig()
    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}
    return LoadTestConfig(
        budget=_dict_to_dataclass(BudgetConfig, raw.get("budget", {})),
        thresholds=_dict_to_dataclass(ThresholdsConfig, raw.get("thresholds", {})),
        discovery=_dict_to_dataclass(DiscoveryConfig, raw.get("discovery", {})),
        scenario_defs=raw.get("scenarios", []),
    )
