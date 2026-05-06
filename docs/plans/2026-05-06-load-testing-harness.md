# Scope Load Testing Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully automated, configurable load testing harness that continuously validates Scope cloud inference across the Livepeer orchestrator network, reporting to Grafana.

**Architecture:** A lightweight Python harness (no ML deps) drives Scope instances via HTTP API. Packaged in docker-compose for portability. Scheduler manages daily traffic budgets per orchestrator. Prometheus push gateway feeds Grafana dashboards.

**Tech Stack:** Python 3.12, httpx, prometheus_client, pyyaml, Pillow, scikit-image, click, Docker, Prometheus, Grafana

**Design Spec:** `docs/design.md`

---

## Phase 1: Project Scaffolding & Config Layer

Produces: a working Python package with CLI skeleton, config loading, and test infrastructure. Everything builds, installs, and tests in Docker.

### Task 1: Project package setup

**Files:**
- Create: `pyproject.toml`
- Create: `src/loadtest/__init__.py`
- Create: `.gitignore`
- Create: `.env.example`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "scope-loadtest"
version = "0.1.0"
description = "Load testing harness for Scope cloud inference on the Livepeer network"
requires-python = ">=3.12"
dependencies = [
    "httpx>=0.28.0",
    "prometheus-client>=0.21.0",
    "pyyaml>=6.0",
    "Pillow>=11.0.0",
    "scikit-image>=0.24.0",
    "click>=8.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24.0",
    "respx>=0.22.0",
]

[project.scripts]
loadtest = "loadtest.cli:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/loadtest"]

[tool.pytest.ini_options]
asyncio_mode = "auto"
testpaths = ["tests"]
```

- [ ] **Step 2: Create src/loadtest/__init__.py**

```python
"""Scope load testing harness for Livepeer cloud inference."""
```

- [ ] **Step 3: Create .gitignore**

```
__pycache__/
*.pyc
.venv/
dist/
*.egg-info/
data/
.env
/videos/*.mp4
```

- [ ] **Step 4: Create .env.example**

```bash
# Livepeer orchestrator discovery
LIVEPEER_DISCOVERY_URL=https://discovery.livepeer.org
LIVEPEER_TOKEN=

# Scope cloud credentials
SCOPE_CLOUD_APP_ID=
SCOPE_CLOUD_API_KEY=

# Scope instances (comma-separated host:port)
SCOPE_INSTANCES=scope-1:8001,scope-2:8002

# Prometheus push gateway
GRAFANA_PUSH_URL=http://pushgateway:9091

# Optional: Scope image tag
SCOPE_IMAGE_TAG=latest
```

- [ ] **Step 5: Verify package installs**

Run: `pip install -e ".[dev]"` (from repo root)
Expected: installs without errors

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/loadtest/__init__.py .gitignore .env.example
git commit -s -m "feat: project scaffolding with pyproject.toml and package structure"
```

### Task 2: Config loading and validation

**Files:**
- Create: `src/loadtest/config.py`
- Create: `config/default.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test for config loading**

```python
# tests/test_config.py
from pathlib import Path
from loadtest.config import load_config, LoadTestConfig


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
  ssim_prompt_sensitivity_max: 0.85
  ssim_model_consistency_min: 0.7
  regression_drift_threshold: 0.20
  recording_duration_tolerance_s: 5
  frame_variance_min: 5.0

discovery:
  refresh_interval_hours: 4
  health_check_timeout_s: 30
  max_consecutive_failures: 5
"""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(yaml_content)

    config = load_config(config_file)

    assert isinstance(config, LoadTestConfig)
    assert config.budget.daily_percent == 20
    assert config.budget.max_run_duration_mins == 30
    assert config.thresholds.connect_timeout_s == 120
    assert config.thresholds.min_fps == 6
    assert config.discovery.refresh_interval_hours == 4


def test_load_config_defaults():
    """Config with missing optional fields uses defaults."""
    config = load_config(None)
    assert config.budget.daily_percent == 20
    assert config.thresholds.connect_timeout_s == 120


def test_load_config_validates_budget_percent():
    """daily_percent must be between 1 and 100."""
    import pytest
    from loadtest.config import BudgetConfig

    with pytest.raises(ValueError):
        BudgetConfig(daily_percent=0)

    with pytest.raises(ValueError):
        BudgetConfig(daily_percent=101)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'loadtest.config'`

- [ ] **Step 3: Implement config module**

```python
# src/loadtest/config.py
"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path

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
    ssim_prompt_sensitivity_max: float = 0.85
    ssim_model_consistency_min: float = 0.7
    regression_drift_threshold: float = 0.20
    recording_duration_tolerance_s: int = 5
    frame_variance_min: float = 5.0


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


def _dict_to_dataclass(cls, data: dict):
    """Create a dataclass from a dict, ignoring unknown keys."""
    import dataclasses

    field_names = {f.name for f in dataclasses.fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)


def load_config(config_path: Path | None) -> LoadTestConfig:
    """Load config from YAML file, falling back to defaults."""
    if config_path is None or not config_path.exists():
        return LoadTestConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f) or {}

    budget = _dict_to_dataclass(BudgetConfig, raw.get("budget", {}))
    thresholds = _dict_to_dataclass(ThresholdsConfig, raw.get("thresholds", {}))
    discovery = _dict_to_dataclass(DiscoveryConfig, raw.get("discovery", {}))

    return LoadTestConfig(budget=budget, thresholds=thresholds, discovery=discovery)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config.py -v`
Expected: 3 passed

- [ ] **Step 5: Create config/default.yaml**

```yaml
# config/default.yaml — Default load test configuration
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
  ssim_prompt_sensitivity_max: 0.85
  ssim_model_consistency_min: 0.7
  regression_drift_threshold: 0.20
  recording_duration_tolerance_s: 5
  frame_variance_min: 5.0

discovery:
  refresh_interval_hours: 4
  health_check_timeout_s: 30
  max_consecutive_failures: 5
```

- [ ] **Step 6: Commit**

```bash
git add src/loadtest/config.py config/default.yaml tests/test_config.py
git commit -s -m "feat: config loading with YAML support and validation"
```

### Task 3: CLI skeleton

**Files:**
- Create: `src/loadtest/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli.py
from click.testing import CliRunner
from loadtest.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "schedule" in result.output
    assert "run" in result.output
    assert "discover" in result.output
    assert "coverage" in result.output


def test_cli_run_requires_scenario():
    runner = CliRunner()
    result = runner.invoke(main, ["run"])
    assert result.exit_code != 0
    assert "scenario" in result.output.lower() or "missing" in result.output.lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_cli.py -v`
Expected: FAIL

- [ ] **Step 3: Implement CLI**

```python
# src/loadtest/cli.py
"""CLI entrypoint for the load testing harness."""

import click

from .config import load_config

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option(
    "--config",
    "config_path",
    type=click.Path(exists=True, dir_okay=False),
    default=None,
    help="Path to config YAML (default: config/default.yaml)",
)
@click.pass_context
def main(ctx: click.Context, config_path: str | None):
    """Scope cloud inference load testing harness."""
    from pathlib import Path

    path = Path(config_path) if config_path else Path("config/default.yaml")
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(path if path.exists() else None)


@main.command()
@click.option("--scenario", required=True, help="Scenario name or YAML path")
@click.option("--orchestrator", default=None, help="Target orchestrator ID")
@click.pass_context
def run(ctx: click.Context, scenario: str, orchestrator: str | None):
    """Execute a single test run."""
    click.echo(f"Running scenario: {scenario} (orchestrator: {orchestrator or 'auto'})")


@main.command()
@click.pass_context
def schedule(ctx: click.Context):
    """Start the scheduler daemon."""
    click.echo("Starting scheduler...")


@main.command()
@click.pass_context
def discover(ctx: click.Context):
    """List available orchestrators and their health status."""
    click.echo("Discovering orchestrators...")


@main.command()
@click.pass_context
def coverage(ctx: click.Context):
    """Show today's test coverage report."""
    click.echo("Coverage report...")


@main.command()
@click.pass_context
def baselines(ctx: click.Context):
    """Show current baseline metrics."""
    click.echo("Baselines...")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_cli.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add src/loadtest/cli.py tests/test_cli.py
git commit -s -m "feat: CLI skeleton with click (run, schedule, discover, coverage, baselines)"
```

### Task 4: Scenario loading

**Files:**
- Create: `src/loadtest/scenarios.py`
- Create: `config/scenarios/longlive_t2v_short.yaml`
- Create: `config/scenarios/longlive_v2v_mid.yaml`
- Create: `config/scenarios/chain_longlive_rife_mid.yaml`
- Create: `config/prompts/nature.yaml`
- Create: `config/prompts/stress.yaml`
- Create: `tests/test_scenarios.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_scenarios.py
from pathlib import Path
from loadtest.scenarios import load_scenario, load_prompt_pool, Scenario, build_session_body


def test_load_scenario_t2v(tmp_path: Path):
    yaml_content = """
name: longlive_t2v_short
pipeline: longlive
mode: t2v
duration_mins: 1
prompts:
  pool: nature
  switch_interval_s: 30
parameters:
  width: 512
  height: 512
validation:
  min_fps: 6
  max_first_frame_s: 60
  frame_check_interval_s: 30
  check_recording: false
"""
    scenario_file = tmp_path / "scenario.yaml"
    scenario_file.write_text(yaml_content)

    scenario = load_scenario(scenario_file)

    assert isinstance(scenario, Scenario)
    assert scenario.name == "longlive_t2v_short"
    assert scenario.pipeline == "longlive"
    assert scenario.mode == "t2v"
    assert scenario.duration_mins == 1
    assert scenario.pipeline_ids == ["longlive"]


def test_load_scenario_graph(tmp_path: Path):
    yaml_content = """
name: chain_longlive_rife_mid
pipeline: longlive+rife
mode: v2v
duration_mins: 5
graph:
  nodes:
    - id: input
      type: source
      source_mode: video_file
      source_name: /data/videos/gradient_512x512_30s.mp4
    - id: longlive
      type: pipeline
      pipeline_id: longlive
    - id: rife
      type: pipeline
      pipeline_id: rife
    - id: output
      type: sink
    - id: record
      type: record
  edges:
    - from: input
      from_port: video
      to_node: longlive
      to_port: video
      kind: stream
    - from: longlive
      from_port: video
      to_node: rife
      to_port: video
      kind: stream
    - from: rife
      from_port: video
      to_node: output
      to_port: video
      kind: stream
    - from: rife
      from_port: video
      to_node: record
      to_port: video
      kind: stream
prompts:
  pool: nature
  switch_interval_s: 30
parameters:
  noise_scale: 0.7
  width: 512
  height: 512
validation:
  min_fps: 6
  max_first_frame_s: 60
  frame_check_interval_s: 30
  check_recording: true
"""
    scenario_file = tmp_path / "scenario.yaml"
    scenario_file.write_text(yaml_content)

    scenario = load_scenario(scenario_file)

    assert scenario.graph is not None
    assert scenario.pipeline_ids == ["longlive", "rife"]
    assert len(scenario.graph["nodes"]) == 5
    assert len(scenario.graph["edges"]) == 4


def test_build_session_body_t2v():
    scenario = Scenario(
        name="longlive_t2v_short",
        pipeline="longlive",
        mode="t2v",
        duration_mins=1,
        graph=None,
        prompts={"pool": "nature", "switch_interval_s": 30},
        parameters={"width": 512, "height": 512},
        validation={"min_fps": 6, "max_first_frame_s": 60, "frame_check_interval_s": 30, "check_recording": False},
    )
    prompt = "a serene mountain lake"
    body = build_session_body(scenario, prompt)

    assert body["pipeline_id"] == "longlive"
    assert body["input_mode"] == "text"
    assert body["prompts"] == [{"text": "a serene mountain lake", "weight": 100}]


def test_build_session_body_v2v_graph():
    graph = {
        "nodes": [
            {"id": "input", "type": "source", "source_mode": "video_file", "source_name": "/data/videos/test.mp4"},
            {"id": "longlive", "type": "pipeline", "pipeline_id": "longlive"},
            {"id": "output", "type": "sink"},
        ],
        "edges": [
            {"from": "input", "from_port": "video", "to_node": "longlive", "to_port": "video", "kind": "stream"},
            {"from": "longlive", "from_port": "video", "to_node": "output", "to_port": "video", "kind": "stream"},
        ],
    }
    scenario = Scenario(
        name="longlive_v2v_mid",
        pipeline="longlive",
        mode="v2v",
        duration_mins=5,
        graph=graph,
        prompts={"pool": "nature", "switch_interval_s": 30},
        parameters={"noise_scale": 0.7, "width": 512, "height": 512},
        validation={"min_fps": 6, "max_first_frame_s": 60, "frame_check_interval_s": 30, "check_recording": True},
    )
    prompt = "ocean waves"
    body = build_session_body(scenario, prompt)

    assert "graph" in body
    assert body["input_mode"] == "video"
    assert "pipeline_id" not in body


def test_load_prompt_pool(tmp_path: Path):
    prompts_dir = tmp_path / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "nature.yaml").write_text(
        "prompts:\n  - 'mountain lake'\n  - 'ocean waves'\n  - 'forest path'\n"
    )

    pool = load_prompt_pool("nature", prompts_dir)
    assert len(pool) == 3
    assert "mountain lake" in pool
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scenarios.py -v`
Expected: FAIL

- [ ] **Step 3: Implement scenarios module**

```python
# src/loadtest/scenarios.py
"""Scenario loading and session body construction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Scenario:
    name: str
    pipeline: str
    mode: str  # t2v, v2v, i2v
    duration_mins: int
    graph: dict[str, Any] | None
    prompts: dict[str, Any]
    parameters: dict[str, Any]
    validation: dict[str, Any]

    @property
    def pipeline_ids(self) -> list[str]:
        """Extract unique pipeline IDs from graph or single pipeline."""
        if self.graph:
            return [
                n["pipeline_id"]
                for n in self.graph["nodes"]
                if n.get("type") == "pipeline"
            ]
        return [self.pipeline]

    @property
    def duration_class(self) -> str:
        if self.duration_mins <= 2:
            return "short"
        elif self.duration_mins <= 10:
            return "mid"
        return "long"

    @property
    def has_recording(self) -> bool:
        if not self.graph:
            return self.validation.get("check_recording", False)
        return any(n.get("type") == "record" for n in self.graph["nodes"])

    @property
    def record_node_id(self) -> str | None:
        if not self.graph:
            return None
        for n in self.graph["nodes"]:
            if n.get("type") == "record":
                return n["id"]
        return None

    @property
    def sink_node_id(self) -> str | None:
        if not self.graph:
            return None
        for n in self.graph["nodes"]:
            if n.get("type") == "sink":
                return n["id"]
        return None


def load_scenario(path: Path) -> Scenario:
    """Load a scenario from a YAML file."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    return Scenario(
        name=raw["name"],
        pipeline=raw["pipeline"],
        mode=raw["mode"],
        duration_mins=raw["duration_mins"],
        graph=raw.get("graph"),
        prompts=raw.get("prompts", {}),
        parameters=raw.get("parameters", {}),
        validation=raw.get("validation", {}),
    )


def load_all_scenarios(scenarios_dir: Path) -> list[Scenario]:
    """Load all scenario YAML files from a directory."""
    scenarios = []
    for path in sorted(scenarios_dir.glob("*.yaml")):
        scenarios.append(load_scenario(path))
    return scenarios


def load_prompt_pool(pool_name: str, prompts_dir: Path) -> list[str]:
    """Load a named prompt pool from the prompts directory."""
    path = prompts_dir / f"{pool_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt pool not found: {path}")

    with open(path) as f:
        raw = yaml.safe_load(f)

    return raw.get("prompts", [])


def build_session_body(scenario: Scenario, prompt: str) -> dict[str, Any]:
    """Build the POST /api/v1/session/start request body from a scenario."""
    if scenario.graph:
        body: dict[str, Any] = {
            "input_mode": "video" if scenario.mode in ("v2v", "i2v") else "text",
            "graph": scenario.graph,
        }
    else:
        body = {
            "pipeline_id": scenario.pipeline,
            "input_mode": "video" if scenario.mode in ("v2v", "i2v") else "text",
        }
        if scenario.mode in ("v2v", "i2v") and "source_name" in scenario.parameters:
            body["input_source"] = {
                "enabled": True,
                "source_type": "video_file",
                "source_name": scenario.parameters["source_name"],
            }

    body["prompts"] = [{"text": prompt, "weight": 100}]
    return body
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scenarios.py -v`
Expected: 5 passed

- [ ] **Step 5: Create scenario YAML files**

Create the full set of 15 scenario files in `config/scenarios/` and 4 prompt pool files in `config/prompts/`. Each scenario YAML follows the format from the design spec (section 5.1). See `docs/design.md` section 5.2 for the complete matrix.

Key scenario files to create:
- `config/scenarios/longlive_t2v_short.yaml` (t2v, 1 min, single pipeline)
- `config/scenarios/longlive_t2v_mid.yaml` (t2v, 5 min, single pipeline)
- `config/scenarios/longlive_t2v_long.yaml` (t2v, 15 min, single pipeline)
- `config/scenarios/longlive_v2v_short.yaml` (v2v, 1 min, single, video source)
- `config/scenarios/longlive_v2v_mid.yaml` (v2v, 5 min, single, video source)
- `config/scenarios/longlive_v2v_long.yaml` (v2v, 15 min, single, video source)
- `config/scenarios/longlive_i2v_short.yaml` (i2v, 1 min, single, image source)
- `config/scenarios/longlive_i2v_mid.yaml` (i2v, 5 min, single, image source)
- `config/scenarios/ltx2_t2v_short.yaml` (t2v, 1 min, single)
- `config/scenarios/ltx2_t2v_mid.yaml` (t2v, 5 min, single)
- `config/scenarios/ltx2_i2v_short.yaml` (i2v, 1 min, single)
- `config/scenarios/ltx2_i2v_mid.yaml` (i2v, 5 min, single)
- `config/scenarios/chain_longlive_rife_mid.yaml` (v2v, 5 min, chained graph)
- `config/scenarios/chain_depth_longlive_rife_mid.yaml` (v2v, 5 min, full chain graph)
- `config/scenarios/chain_longlive_rife_long.yaml` (v2v, 15 min, chained graph)

Key prompt files:
- `config/prompts/nature.yaml` (20-30 nature prompts)
- `config/prompts/urban.yaml` (20-30 urban prompts)
- `config/prompts/abstract.yaml` (20-30 abstract prompts)
- `config/prompts/stress.yaml` (edge cases: very long, special chars, minimal, empty)

- [ ] **Step 6: Commit**

```bash
git add src/loadtest/scenarios.py tests/test_scenarios.py config/scenarios/ config/prompts/
git commit -s -m "feat: scenario loading, prompt pools, and session body builder"
```

---

## Phase 2: Scope API Client & Results

Produces: a typed async HTTP client for all Scope API calls, plus result/error classification.

### Task 5: Scope HTTP client

**Files:**
- Create: `src/loadtest/scope_client.py`
- Create: `tests/test_scope_client.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_scope_client.py
import pytest
import httpx
import respx
from loadtest.scope_client import ScopeClient


@pytest.fixture
def scope_url():
    return "http://scope-1:8001"


@respx.mock
@pytest.mark.asyncio
async def test_health_check(scope_url: str):
    respx.get(f"{scope_url}/health").respond(json={"status": "healthy"})

    async with ScopeClient(scope_url) as client:
        result = await client.health()

    assert result["status"] == "healthy"


@respx.mock
@pytest.mark.asyncio
async def test_cloud_connect(scope_url: str):
    respx.post(f"{scope_url}/api/v1/cloud/connect").respond(
        json={"connected": False, "connecting": True, "webrtc_connected": False}
    )

    async with ScopeClient(scope_url) as client:
        result = await client.cloud_connect(app_id="test-app")

    assert result["connecting"] is True


@respx.mock
@pytest.mark.asyncio
async def test_cloud_status(scope_url: str):
    respx.get(f"{scope_url}/api/v1/cloud/status").respond(
        json={"connected": True, "connecting": False, "webrtc_connected": True}
    )

    async with ScopeClient(scope_url) as client:
        status = await client.cloud_status()

    assert status["connected"] is True


@respx.mock
@pytest.mark.asyncio
async def test_pipeline_load(scope_url: str):
    respx.post(f"{scope_url}/api/v1/pipeline/load").respond(
        json={"status": "loading"}
    )

    async with ScopeClient(scope_url) as client:
        result = await client.pipeline_load(["longlive"])

    assert result["status"] == "loading"


@respx.mock
@pytest.mark.asyncio
async def test_session_start(scope_url: str):
    respx.post(f"{scope_url}/api/v1/session/start").respond(
        json={"status": "ok", "graph": True, "sink_node_ids": ["output"]}
    )

    async with ScopeClient(scope_url) as client:
        result = await client.session_start({"pipeline_id": "longlive", "input_mode": "text"})

    assert result["status"] == "ok"


@respx.mock
@pytest.mark.asyncio
async def test_session_metrics(scope_url: str):
    respx.get(f"{scope_url}/api/v1/session/metrics").respond(
        json={
            "sessions": {"headless": {"fps_out": 10.0, "frames_out": 50}},
            "gpu": {"vram_allocated_mb": 8000, "vram_total_mb": 81920},
        }
    )

    async with ScopeClient(scope_url) as client:
        metrics = await client.session_metrics()

    assert metrics["sessions"]["headless"]["fps_out"] == 10.0


@respx.mock
@pytest.mark.asyncio
async def test_capture_frame(scope_url: str):
    jpeg_bytes = b"\xff\xd8\xff\xe0fake_jpeg_data"
    respx.get(f"{scope_url}/api/v1/session/frame").respond(
        content=jpeg_bytes, headers={"content-type": "image/jpeg"}
    )

    async with ScopeClient(scope_url) as client:
        data = await client.capture_frame(sink_node_id="output")

    assert data == jpeg_bytes


@respx.mock
@pytest.mark.asyncio
async def test_get_logs(scope_url: str):
    respx.get(f"{scope_url}/api/v1/logs/tail").respond(
        json={"logs": ["line1", "line2"]}
    )

    async with ScopeClient(scope_url) as client:
        logs = await client.get_logs(lines=50)

    assert len(logs["logs"]) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scope_client.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Scope client**

```python
# src/loadtest/scope_client.py
"""Typed async HTTP client for the Scope API."""

from typing import Any

import httpx


class ScopeClient:
    """Async HTTP client for a single Scope instance."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "ScopeClient":
        self._client = httpx.AsyncClient(
            base_url=self._base_url,
            timeout=httpx.Timeout(self._timeout),
        )
        return self

    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("ScopeClient not entered as async context manager")
        return self._client

    # --- Health ---

    async def health(self) -> dict[str, Any]:
        resp = await self.client.get("/health")
        resp.raise_for_status()
        return resp.json()

    # --- Cloud ---

    async def cloud_connect(
        self,
        app_id: str | None = None,
        api_key: str | None = None,
        user_id: str | None = None,
    ) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if app_id:
            body["app_id"] = app_id
        if api_key:
            body["api_key"] = api_key
        if user_id:
            body["user_id"] = user_id
        resp = await self.client.post("/api/v1/cloud/connect", json=body)
        resp.raise_for_status()
        return resp.json()

    async def cloud_status(self) -> dict[str, Any]:
        resp = await self.client.get("/api/v1/cloud/status")
        resp.raise_for_status()
        return resp.json()

    async def cloud_disconnect(self) -> dict[str, Any]:
        resp = await self.client.post("/api/v1/cloud/disconnect")
        resp.raise_for_status()
        return resp.json()

    # --- Pipeline ---

    async def pipeline_load(self, pipeline_ids: list[str]) -> dict[str, Any]:
        resp = await self.client.post(
            "/api/v1/pipeline/load", json={"pipeline_ids": pipeline_ids}
        )
        resp.raise_for_status()
        return resp.json()

    async def pipeline_status(self) -> dict[str, Any]:
        resp = await self.client.get("/api/v1/pipeline/status")
        resp.raise_for_status()
        return resp.json()

    # --- Session ---

    async def session_start(self, body: dict[str, Any]) -> dict[str, Any]:
        resp = await self.client.post("/api/v1/session/start", json=body)
        resp.raise_for_status()
        return resp.json()

    async def session_stop(self) -> dict[str, Any]:
        resp = await self.client.post("/api/v1/session/stop")
        resp.raise_for_status()
        return resp.json()

    async def session_metrics(self) -> dict[str, Any]:
        resp = await self.client.get("/api/v1/session/metrics")
        resp.raise_for_status()
        return resp.json()

    async def session_parameters(self, params: dict[str, Any]) -> dict[str, Any]:
        resp = await self.client.post("/api/v1/session/parameters", json=params)
        resp.raise_for_status()
        return resp.json()

    async def capture_frame(
        self, sink_node_id: str | None = None, quality: int = 85
    ) -> bytes:
        params: dict[str, Any] = {"quality": quality}
        if sink_node_id:
            params["sink_node_id"] = sink_node_id
        resp = await self.client.get("/api/v1/session/frame", params=params)
        resp.raise_for_status()
        return resp.content

    # --- Recording ---

    async def recording_start(self, node_id: str) -> dict[str, Any]:
        resp = await self.client.post(
            "/api/v1/recordings/headless/start", params={"node_id": node_id}
        )
        resp.raise_for_status()
        return resp.json()

    async def recording_stop(self, node_id: str) -> dict[str, Any]:
        resp = await self.client.post(
            "/api/v1/recordings/headless/stop", params={"node_id": node_id}
        )
        resp.raise_for_status()
        return resp.json()

    async def recording_download(self, node_id: str) -> bytes:
        resp = await self.client.get(
            "/api/v1/recordings/headless",
            params={"node_id": node_id},
            timeout=httpx.Timeout(120.0),
        )
        resp.raise_for_status()
        return resp.content

    # --- Logs ---

    async def get_logs(self, lines: int = 50) -> dict[str, Any]:
        resp = await self.client.get("/api/v1/logs/tail", params={"lines": lines})
        resp.raise_for_status()
        return resp.json()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scope_client.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/loadtest/scope_client.py tests/test_scope_client.py
git commit -s -m "feat: typed async Scope HTTP client with full API coverage"
```

### Task 6: Results and error taxonomy

**Files:**
- Create: `src/loadtest/results.py`
- Create: `tests/test_results.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_results.py
import httpx
from loadtest.results import (
    RunResult,
    ErrorCategory,
    classify_error,
    PhaseTimings,
)


def test_run_result_pass():
    result = RunResult(
        scenario="longlive_t2v_short",
        orchestrator_id="O-abc123",
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
        scenario="longlive_t2v_short",
        orchestrator_id="O-abc123",
        passed=False,
        error_category=ErrorCategory.RUNNER,
        error_message="CUDA out of memory",
        timings=PhaseTimings(connect_s=5.0, pipeline_load_s=15.0),
    )
    assert result.passed is False
    assert result.error_category == ErrorCategory.RUNNER


def test_classify_error_timeout():
    err = httpx.ReadTimeout("read timed out")
    category = classify_error(err)
    assert category == ErrorCategory.NETWORK


def test_classify_error_connect():
    err = httpx.ConnectError("connection refused")
    category = classify_error(err)
    assert category == ErrorCategory.NETWORK


def test_classify_error_http_502():
    err = httpx.HTTPStatusError(
        "Bad Gateway",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(502),
    )
    category = classify_error(err)
    assert category == ErrorCategory.ORCHESTRATOR


def test_classify_error_http_500_cuda():
    err = httpx.HTTPStatusError(
        "CUDA error",
        request=httpx.Request("GET", "http://x"),
        response=httpx.Response(500, text="CUDA out of memory"),
    )
    category = classify_error(err, response_text="CUDA out of memory")
    assert category == ErrorCategory.RUNNER


import pytest
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_results.py -v`
Expected: FAIL

- [ ] **Step 3: Implement results module**

```python
# src/loadtest/results.py
"""Run results, error taxonomy, and log capture."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import httpx


class ErrorCategory(str, Enum):
    NETWORK = "network"
    ORCHESTRATOR = "orchestrator"
    RUNNER = "runner"
    PROTOCOL = "protocol"


@dataclass
class PhaseTimings:
    connect_s: float | None = None
    pipeline_load_s: float | None = None
    first_frame_s: float | None = None
    stream_duration_s: float | None = None
    total_s: float | None = None


@dataclass
class RunResult:
    scenario: str
    orchestrator_id: str
    passed: bool
    timings: PhaseTimings = field(default_factory=PhaseTimings)
    error_category: ErrorCategory | None = None
    error_message: str | None = None
    fps_samples: list[float] = field(default_factory=list)
    vram_samples: list[float] = field(default_factory=list)
    frames_validated: int = 0
    frames_black: int = 0
    frames_corrupt: int = 0
    prompt_sensitivity_checks: int = 0
    prompt_sensitivity_failures: int = 0
    recording_valid: bool | None = None
    cold_start: bool | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def avg_fps(self) -> float | None:
        if not self.fps_samples:
            return None
        return sum(self.fps_samples) / len(self.fps_samples)

    @property
    def vram_growth_mb(self) -> float | None:
        if len(self.vram_samples) < 4:
            return None
        quarter = len(self.vram_samples) // 4
        first_q = sum(self.vram_samples[:quarter]) / quarter
        last_q = sum(self.vram_samples[-quarter:]) / quarter
        return last_q - first_q


def classify_error(
    error: Exception, response_text: str | None = None
) -> ErrorCategory:
    """Classify an exception into the error taxonomy."""
    if isinstance(error, (httpx.TimeoutException, httpx.ConnectError, ConnectionError, OSError)):
        return ErrorCategory.NETWORK

    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        text = (response_text or "").lower()

        if status in (502, 503, 504):
            return ErrorCategory.ORCHESTRATOR

        if status == 500:
            runner_keywords = ["cuda", "oom", "out of memory", "pipeline", "model", "torch"]
            if any(kw in text for kw in runner_keywords):
                return ErrorCategory.RUNNER
            return ErrorCategory.RUNNER  # default 500 to runner

        if status in (400, 422):
            return ErrorCategory.PROTOCOL

        return ErrorCategory.ORCHESTRATOR

    return ErrorCategory.PROTOCOL


def save_failure_logs(
    logs: str,
    orchestrator_id: str,
    scenario: str,
    data_dir: Path,
) -> Path:
    """Save failure logs to disk. Returns the path written."""
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%H%M%S")
    failures_dir = data_dir / "failures" / date_str
    failures_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{orchestrator_id}_{scenario}_{ts}.log"
    path = failures_dir / filename
    path.write_text(logs)
    return path


def cleanup_old_failures(data_dir: Path, max_age_days: int = 7) -> int:
    """Remove failure log directories older than max_age_days. Returns count removed."""
    failures_dir = data_dir / "failures"
    if not failures_dir.exists():
        return 0

    cutoff = datetime.now(timezone.utc).date()
    removed = 0
    for day_dir in failures_dir.iterdir():
        if not day_dir.is_dir():
            continue
        try:
            dir_date = datetime.strptime(day_dir.name, "%Y-%m-%d").date()
            if (cutoff - dir_date).days > max_age_days:
                import shutil
                shutil.rmtree(day_dir)
                removed += 1
        except ValueError:
            continue
    return removed
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_results.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/loadtest/results.py tests/test_results.py
git commit -s -m "feat: run results with error taxonomy, log capture, and cleanup"
```

---

## Phase 3: Validators & Metrics

Produces: frame quality validation, SSIM checks, recording validation, Prometheus metric definitions, and push logic.

### Task 7: Frame and recording validators

**Files:**
- Create: `src/loadtest/validators.py`
- Create: `tests/test_validators.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_validators.py
import io
import pytest
import numpy as np
from PIL import Image
from loadtest.validators import (
    FrameCheckResult,
    RecordingCheckResult,
    validate_frame,
    check_prompt_sensitivity,
)


def _make_jpeg(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_validate_frame_valid():
    jpeg = _make_jpeg(512, 512, (128, 64, 200))
    result = validate_frame(jpeg, expected_width=512, expected_height=512)
    assert result == FrameCheckResult.VALID


def test_validate_frame_black():
    jpeg = _make_jpeg(512, 512, (0, 0, 0))
    result = validate_frame(jpeg, expected_width=512, expected_height=512, variance_min=5.0)
    assert result == FrameCheckResult.BLACK


def test_validate_frame_wrong_size():
    jpeg = _make_jpeg(256, 256, (128, 64, 200))
    result = validate_frame(jpeg, expected_width=512, expected_height=512)
    assert result == FrameCheckResult.WRONG_SIZE


def test_validate_frame_corrupt():
    result = validate_frame(b"not a jpeg", expected_width=512, expected_height=512)
    assert result == FrameCheckResult.CORRUPT


def test_prompt_sensitivity_different_images():
    img_a = _make_jpeg(512, 512, (255, 0, 0))
    img_b = _make_jpeg(512, 512, (0, 0, 255))
    passed = check_prompt_sensitivity(img_a, img_b, max_ssim=0.85)
    assert passed is True  # very different images


def test_prompt_sensitivity_identical_images():
    img = _make_jpeg(512, 512, (128, 128, 128))
    passed = check_prompt_sensitivity(img, img, max_ssim=0.85)
    assert passed is False  # identical = model not responding
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_validators.py -v`
Expected: FAIL

- [ ] **Step 3: Implement validators**

```python
# src/loadtest/validators.py
"""Frame quality, SSIM prompt sensitivity, and recording validation."""

import io
import tempfile
from enum import Enum
from pathlib import Path

import numpy as np
from PIL import Image


class FrameCheckResult(str, Enum):
    VALID = "valid"
    BLACK = "black"
    CORRUPT = "corrupt"
    WRONG_SIZE = "wrong_size"


class RecordingCheckResult(str, Enum):
    VALID = "valid"
    ZERO_SIZE = "zero_size"
    CORRUPT = "corrupt"
    WRONG_DURATION = "wrong_duration"


def validate_frame(
    jpeg_bytes: bytes,
    expected_width: int,
    expected_height: int,
    variance_min: float = 5.0,
) -> FrameCheckResult:
    """Validate a JPEG frame for quality issues."""
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img.load()  # force decode
    except Exception:
        return FrameCheckResult.CORRUPT

    if img.size != (expected_width, expected_height):
        return FrameCheckResult.WRONG_SIZE

    arr = np.array(img, dtype=np.float32)
    if arr.std() < variance_min:
        return FrameCheckResult.BLACK

    return FrameCheckResult.VALID


def check_prompt_sensitivity(
    frame_before: bytes,
    frame_after: bytes,
    max_ssim: float = 0.85,
) -> bool:
    """Check that two frames are sufficiently different (model responds to prompts).

    Returns True if the frames differ enough (SSIM < max_ssim).
    """
    from skimage.metrics import structural_similarity as ssim

    try:
        img_a = np.array(Image.open(io.BytesIO(frame_before)).convert("L"))
        img_b = np.array(Image.open(io.BytesIO(frame_after)).convert("L"))
    except Exception:
        return False  # can't decode = fail

    if img_a.shape != img_b.shape:
        return True  # different sizes = definitely different

    score = ssim(img_a, img_b)
    return score < max_ssim


def validate_recording(
    mp4_bytes: bytes,
    expected_duration_s: float,
    tolerance_s: float = 5.0,
) -> RecordingCheckResult:
    """Validate a recording file."""
    if len(mp4_bytes) == 0:
        return RecordingCheckResult.ZERO_SIZE

    # Write to temp file and probe with Pillow/cv2-free approach
    try:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(mp4_bytes)
            tmp_path = Path(f.name)

        # Use ffprobe-style check via file size heuristic
        # A valid MP4 starts with ftyp box
        if mp4_bytes[:4] == b"\x00\x00\x00" and b"ftyp" in mp4_bytes[:12]:
            pass  # likely valid MP4
        elif mp4_bytes[:3] == b"\x00\x00\x01":
            pass  # MPEG-TS / raw stream
        else:
            return RecordingCheckResult.CORRUPT

        # Size-based duration estimate (rough: assume 1MB/s for 512x512 H264)
        estimated_duration_s = len(mp4_bytes) / (1024 * 1024)
        if abs(estimated_duration_s - expected_duration_s) > expected_duration_s * 2:
            # Way off — likely corrupt or wrong
            return RecordingCheckResult.WRONG_DURATION

        return RecordingCheckResult.VALID

    except Exception:
        return RecordingCheckResult.CORRUPT
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_validators.py -v`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/loadtest/validators.py tests/test_validators.py
git commit -s -m "feat: frame validation, SSIM prompt sensitivity, and recording checks"
```

### Task 8: Prometheus metrics and push gateway

**Files:**
- Create: `src/loadtest/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_metrics.py
from loadtest.metrics import MetricsCollector
from loadtest.results import RunResult, PhaseTimings, ErrorCategory


def test_metrics_collector_creates_metrics():
    collector = MetricsCollector(push_url=None)  # no push, just collect
    assert collector.runs_total is not None
    assert collector.connect_duration is not None
    assert collector.first_frame_duration is not None


def test_record_run_result_pass():
    collector = MetricsCollector(push_url=None)
    result = RunResult(
        scenario="longlive_t2v_short",
        orchestrator_id="O-abc",
        passed=True,
        timings=PhaseTimings(connect_s=5.0, pipeline_load_s=15.0, first_frame_s=12.0),
        fps_samples=[9.0, 10.0],
        vram_samples=[8000.0],
        labels={"pipeline": "longlive", "mode": "t2v", "duration_class": "short"},
    )
    # Should not raise
    collector.record_run(result)


def test_record_run_result_fail():
    collector = MetricsCollector(push_url=None)
    result = RunResult(
        scenario="longlive_t2v_short",
        orchestrator_id="O-abc",
        passed=False,
        error_category=ErrorCategory.RUNNER,
        error_message="OOM",
        timings=PhaseTimings(connect_s=5.0),
        labels={"pipeline": "longlive", "mode": "t2v", "duration_class": "short"},
    )
    collector.record_run(result)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_metrics.py -v`
Expected: FAIL

- [ ] **Step 3: Implement metrics module**

```python
# src/loadtest/metrics.py
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

COMMON_LABELS = ["orchestrator_id", "pipeline", "mode", "scenario", "duration_class"]


class MetricsCollector:
    """Collects and pushes Prometheus metrics for load test runs."""

    def __init__(self, push_url: str | None, job_name: str = "scope_loadtest"):
        self._push_url = push_url
        self._job_name = job_name
        self._registry = CollectorRegistry()

        # Histograms
        self.connect_duration = Histogram(
            "connect_duration_seconds",
            "Cloud connect time",
            COMMON_LABELS,
            buckets=[5, 10, 20, 30, 60, 90, 120],
            registry=self._registry,
        )
        self.pipeline_load_duration = Histogram(
            "pipeline_load_seconds",
            "Pipeline load time",
            COMMON_LABELS,
            buckets=[10, 20, 30, 60, 120, 180, 300],
            registry=self._registry,
        )
        self.first_frame_duration = Histogram(
            "first_frame_seconds",
            "Prompt to first output frame",
            COMMON_LABELS,
            buckets=[5, 10, 15, 20, 30, 45, 60],
            registry=self._registry,
        )
        self.prompt_switch_latency = Histogram(
            "prompt_switch_latency_seconds",
            "Time for output to change after prompt update",
            COMMON_LABELS,
            buckets=[2, 5, 10, 15, 20, 30],
            registry=self._registry,
        )

        # Gauges
        self.stream_fps_out = Gauge(
            "stream_fps_out", "Output FPS during session", COMMON_LABELS,
            registry=self._registry,
        )
        self.vram_allocated_mb = Gauge(
            "vram_allocated_mb", "GPU memory usage on runner", COMMON_LABELS,
            registry=self._registry,
        )
        self.budget_runs_planned = Gauge(
            "budget_runs_planned", "Planned runs per orchestrator per day",
            ["orchestrator_id"], registry=self._registry,
        )
        self.budget_runs_completed = Gauge(
            "budget_runs_completed", "Completed runs per orchestrator per day",
            ["orchestrator_id"], registry=self._registry,
        )
        self.orchestrator_coverage_percent = Gauge(
            "orchestrator_coverage_percent", "Percentage of orchestrators tested",
            registry=self._registry,
        )
        self.baseline_drift_percent = Gauge(
            "baseline_drift_percent", "Current vs baseline deviation",
            ["metric_name", "scenario"], registry=self._registry,
        )

        # Counters
        self.runs_total = Counter(
            "runs_total", "Total runs attempted",
            [*COMMON_LABELS, "result"], registry=self._registry,
        )
        self.failures_total = Counter(
            "failures_total", "Failures by category",
            [*COMMON_LABELS, "category"], registry=self._registry,
        )
        self.frames_validated_total = Counter(
            "frames_validated_total", "Frame validation results",
            [*COMMON_LABELS, "result"], registry=self._registry,
        )
        self.recordings_validated_total = Counter(
            "recordings_validated_total", "Recording validation results",
            [*COMMON_LABELS, "result"], registry=self._registry,
        )
        self.prompt_sensitivity_checks_total = Counter(
            "prompt_sensitivity_checks_total", "SSIM prompt checks",
            [*COMMON_LABELS, "result"], registry=self._registry,
        )

    def _labels(self, result: RunResult) -> dict[str, str]:
        return {
            "orchestrator_id": result.orchestrator_id,
            "pipeline": result.labels.get("pipeline", "unknown"),
            "mode": result.labels.get("mode", "unknown"),
            "scenario": result.scenario,
            "duration_class": result.labels.get("duration_class", "unknown"),
        }

    def record_run(self, result: RunResult) -> None:
        """Record all metrics from a completed run."""
        labels = self._labels(result)
        result_label = "pass" if result.passed else "fail"

        self.runs_total.labels(**labels, result=result_label).inc()

        if result.timings.connect_s is not None:
            self.connect_duration.labels(**labels).observe(result.timings.connect_s)
        if result.timings.pipeline_load_s is not None:
            self.pipeline_load_duration.labels(**labels).observe(result.timings.pipeline_load_s)
        if result.timings.first_frame_s is not None:
            self.first_frame_duration.labels(**labels).observe(result.timings.first_frame_s)

        if result.fps_samples:
            avg_fps = sum(result.fps_samples) / len(result.fps_samples)
            self.stream_fps_out.labels(**labels).set(avg_fps)

        if result.vram_samples:
            self.vram_allocated_mb.labels(**labels).set(result.vram_samples[-1])

        if not result.passed and result.error_category:
            self.failures_total.labels(**labels, category=result.error_category.value).inc()

    def push(self) -> None:
        """Push all collected metrics to the gateway."""
        if not self._push_url:
            logger.debug("No push URL configured, skipping metric push")
            return

        try:
            push_to_gateway(self._push_url, job=self._job_name, registry=self._registry)
            logger.info("Metrics pushed to %s", self._push_url)
        except Exception as e:
            logger.error("Failed to push metrics: %s", e)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_metrics.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/loadtest/metrics.py tests/test_metrics.py
git commit -s -m "feat: Prometheus metrics with histograms, gauges, counters, and push gateway"
```

---

## Phase 4: Executor & Regression

Produces: the core test executor that drives a complete scenario lifecycle, plus regression detection.

### Task 9: Test executor

**Files:**
- Create: `src/loadtest/executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write failing test for executor connect phase**

```python
# tests/test_executor.py
import pytest
import respx
import httpx
from loadtest.executor import Executor
from loadtest.config import LoadTestConfig
from loadtest.scenarios import Scenario


def _make_scenario(**overrides) -> Scenario:
    defaults = dict(
        name="longlive_t2v_short",
        pipeline="longlive",
        mode="t2v",
        duration_mins=1,
        graph=None,
        prompts={"pool": "nature", "switch_interval_s": 30},
        parameters={"width": 512, "height": 512},
        validation={"min_fps": 6, "max_first_frame_s": 60, "frame_check_interval_s": 30, "check_recording": False},
    )
    defaults.update(overrides)
    return Scenario(**defaults)


@respx.mock
@pytest.mark.asyncio
async def test_executor_connect_phase():
    base = "http://scope-1:8001"
    respx.post(f"{base}/api/v1/cloud/connect").respond(
        json={"connected": False, "connecting": True}
    )
    respx.get(f"{base}/api/v1/cloud/status").respond(
        json={"connected": True, "connecting": False, "webrtc_connected": True}
    )

    config = LoadTestConfig()
    executor = Executor(config)
    duration = await executor._connect_phase(base, app_id="test-app")

    assert duration > 0


@respx.mock
@pytest.mark.asyncio
async def test_executor_connect_phase_timeout():
    base = "http://scope-1:8001"
    respx.post(f"{base}/api/v1/cloud/connect").respond(
        json={"connected": False, "connecting": True}
    )
    # Always return connecting=True, never connected
    respx.get(f"{base}/api/v1/cloud/status").respond(
        json={"connected": False, "connecting": True, "error": None}
    )

    config = LoadTestConfig()
    config.thresholds.connect_timeout_s = 2  # short timeout for test
    executor = Executor(config)

    with pytest.raises(TimeoutError):
        await executor._connect_phase(base, app_id="test-app")


@respx.mock
@pytest.mark.asyncio
async def test_executor_load_phase():
    base = "http://scope-1:8001"
    respx.post(f"{base}/api/v1/pipeline/load").respond(json={"status": "loading"})
    respx.get(f"{base}/api/v1/pipeline/status").respond(
        json={"status": "loaded", "pipeline_ids": ["longlive"]}
    )

    config = LoadTestConfig()
    executor = Executor(config)
    duration = await executor._load_phase(base, ["longlive"])

    assert duration > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_executor.py -v`
Expected: FAIL

- [ ] **Step 3: Implement executor**

```python
# src/loadtest/executor.py
"""Drives a single test scenario against a Scope instance."""

import asyncio
import logging
import random
import time
from pathlib import Path
from typing import Any

from .config import LoadTestConfig
from .results import (
    ErrorCategory,
    PhaseTimings,
    RunResult,
    classify_error,
    save_failure_logs,
)
from .scenarios import Scenario, build_session_body
from .scope_client import ScopeClient
from .validators import (
    FrameCheckResult,
    check_prompt_sensitivity,
    validate_frame,
    validate_recording,
)

logger = logging.getLogger(__name__)


class Executor:
    """Executes a single test scenario against one Scope instance."""

    def __init__(self, config: LoadTestConfig, data_dir: Path | None = None):
        self._config = config
        self._data_dir = data_dir or Path("data")

    async def _connect_phase(
        self, scope_url: str, app_id: str, api_key: str | None = None
    ) -> float:
        """Connect to cloud. Returns duration in seconds. Raises TimeoutError on timeout."""
        start = time.monotonic()
        timeout = self._config.thresholds.connect_timeout_s

        async with ScopeClient(scope_url) as client:
            await client.cloud_connect(app_id=app_id, api_key=api_key)

            deadline = start + timeout
            while time.monotonic() < deadline:
                status = await client.cloud_status()
                if status.get("connected"):
                    return time.monotonic() - start
                if status.get("error"):
                    raise RuntimeError(f"Cloud connect error: {status['error']}")
                await asyncio.sleep(2)

        raise TimeoutError(f"Cloud connect timed out after {timeout}s")

    async def _load_phase(self, scope_url: str, pipeline_ids: list[str]) -> float:
        """Load pipelines. Returns duration in seconds."""
        start = time.monotonic()
        timeout = self._config.thresholds.pipeline_load_timeout_s

        async with ScopeClient(scope_url) as client:
            await client.pipeline_load(pipeline_ids)

            deadline = start + timeout
            while time.monotonic() < deadline:
                status = await client.pipeline_status()
                if status.get("status") == "loaded":
                    return time.monotonic() - start
                if status.get("status") == "error":
                    raise RuntimeError(f"Pipeline load error: {status.get('error_message')}")
                await asyncio.sleep(2)

        raise TimeoutError(f"Pipeline load timed out after {timeout}s")

    async def _stream_phase(
        self,
        scope_url: str,
        scenario: Scenario,
        prompts: list[str],
        result: RunResult,
    ) -> None:
        """Run the streaming session with monitoring loop."""
        thresholds = self._config.thresholds
        check_interval = scenario.validation.get("frame_check_interval_s", 30)
        switch_interval = scenario.prompts.get("switch_interval_s", 30)
        expected_w = scenario.parameters.get("width", 512)
        expected_h = scenario.parameters.get("height", 512)
        duration_s = scenario.duration_mins * 60
        prompt_idx = 0

        async with ScopeClient(scope_url, timeout=60.0) as client:
            # Start session
            body = build_session_body(scenario, prompts[prompt_idx])
            await client.session_start(body)

            # Wait for first frame
            ff_start = time.monotonic()
            while time.monotonic() - ff_start < thresholds.first_frame_timeout_s:
                metrics = await client.session_metrics()
                session = metrics.get("sessions", {}).get("headless", {})
                if session.get("frames_out", 0) > 0:
                    result.timings.first_frame_s = time.monotonic() - ff_start
                    break
                await asyncio.sleep(1)
            else:
                raise TimeoutError("No first frame within timeout")

            # Start recording if scenario has a record node
            if scenario.has_recording and scenario.record_node_id:
                await client.recording_start(scenario.record_node_id)

            # Monitoring loop
            stream_start = time.monotonic()
            last_check = 0.0
            last_prompt_switch = stream_start
            stall_start: float | None = None

            while time.monotonic() - stream_start < duration_s:
                now = time.monotonic()

                # Periodic metrics + frame check
                if now - last_check >= check_interval:
                    last_check = now

                    metrics = await client.session_metrics()
                    session = metrics.get("sessions", {}).get("headless", {})
                    gpu = metrics.get("gpu", {})

                    fps_out = session.get("fps_out", 0)
                    result.fps_samples.append(fps_out)

                    vram = gpu.get("vram_allocated_mb", 0)
                    if vram:
                        result.vram_samples.append(vram)

                    # Stall detection
                    if fps_out == 0:
                        if stall_start is None:
                            stall_start = now
                        elif now - stall_start > thresholds.stall_timeout_s:
                            raise RuntimeError(f"Stream stalled for {thresholds.stall_timeout_s}s")
                    else:
                        stall_start = None

                    # Frame validation
                    try:
                        frame = await client.capture_frame(
                            sink_node_id=scenario.sink_node_id
                        )
                        check = validate_frame(frame, expected_w, expected_h, thresholds.frame_variance_min)
                        result.frames_validated += 1
                        if check == FrameCheckResult.BLACK:
                            result.frames_black += 1
                        elif check == FrameCheckResult.CORRUPT:
                            result.frames_corrupt += 1
                    except Exception:
                        pass  # frame capture can fail transiently

                # Prompt switching
                if now - last_prompt_switch >= switch_interval and len(prompts) > 1:
                    # Capture frame before switch
                    try:
                        frame_before = await client.capture_frame(
                            sink_node_id=scenario.sink_node_id
                        )
                    except Exception:
                        frame_before = None

                    prompt_idx = (prompt_idx + 1) % len(prompts)
                    await client.session_parameters(
                        {"prompts": [{"text": prompts[prompt_idx], "weight": 100}]}
                    )
                    last_prompt_switch = now

                    # Wait and capture after
                    if frame_before:
                        await asyncio.sleep(10)
                        try:
                            frame_after = await client.capture_frame(
                                sink_node_id=scenario.sink_node_id
                            )
                            result.prompt_sensitivity_checks += 1
                            if not check_prompt_sensitivity(
                                frame_before,
                                frame_after,
                                thresholds.ssim_prompt_sensitivity_max,
                            ):
                                result.prompt_sensitivity_failures += 1
                        except Exception:
                            pass

                await asyncio.sleep(1)

            result.timings.stream_duration_s = time.monotonic() - stream_start

            # Stop recording and validate
            if scenario.has_recording and scenario.record_node_id:
                await client.recording_stop(scenario.record_node_id)
                try:
                    mp4_data = await client.recording_download(scenario.record_node_id)
                    rec_result = validate_recording(
                        mp4_data,
                        expected_duration_s=duration_s,
                        tolerance_s=thresholds.recording_duration_tolerance_s,
                    )
                    result.recording_valid = rec_result.value == "valid"
                except Exception:
                    result.recording_valid = False

            # Stop session
            await client.session_stop()

    async def run(
        self,
        scope_url: str,
        orchestrator_id: str,
        scenario: Scenario,
        prompts: list[str],
        app_id: str,
        api_key: str | None = None,
    ) -> RunResult:
        """Execute a complete test scenario. Always returns a RunResult (never raises)."""
        result = RunResult(
            scenario=scenario.name,
            orchestrator_id=orchestrator_id,
            passed=False,
            labels={
                "pipeline": scenario.pipeline,
                "mode": scenario.mode,
                "duration_class": scenario.duration_class,
            },
        )

        total_start = time.monotonic()

        # Hard timeout watchdog
        max_duration = self._config.budget.max_run_duration_mins * 60

        try:
            async with asyncio.timeout(max_duration):
                # 1. Connect
                result.timings.connect_s = await self._connect_phase(
                    scope_url, app_id, api_key
                )

                # 2. Load pipelines
                result.timings.pipeline_load_s = await self._load_phase(
                    scope_url, scenario.pipeline_ids
                )

                # 3. Stream with monitoring
                if not prompts:
                    prompts = ["a test scene"]
                await self._stream_phase(scope_url, scenario, prompts, result)

                # 4. Disconnect
                async with ScopeClient(scope_url) as client:
                    await client.cloud_disconnect()

                result.passed = True

        except Exception as e:
            result.error_category = classify_error(e)
            result.error_message = str(e)
            logger.error(
                "Run failed [%s/%s]: %s: %s",
                orchestrator_id, scenario.name, type(e).__name__, e,
            )

            # Capture logs on failure
            try:
                async with ScopeClient(scope_url) as client:
                    logs = await client.get_logs(lines=100)
                    log_text = "\n".join(logs.get("logs", []))
                    save_failure_logs(log_text, orchestrator_id, scenario.name, self._data_dir)
            except Exception:
                pass

            # Force cleanup
            try:
                async with ScopeClient(scope_url) as client:
                    await client.session_stop()
                    await client.cloud_disconnect()
            except Exception:
                pass

        result.timings.total_s = time.monotonic() - total_start
        return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_executor.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/loadtest/executor.py tests/test_executor.py
git commit -s -m "feat: test executor with connect/load/stream/cleanup phases and watchdog"
```

### Task 10: Regression detection

**Files:**
- Create: `src/loadtest/regression.py`
- Create: `tests/test_regression.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_regression.py
import json
import pytest
from pathlib import Path
from loadtest.regression import (
    BaselineStore,
    DriftResult,
    check_drift,
    update_baseline,
)
from loadtest.results import RunResult, PhaseTimings


def test_update_baseline_new(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")

    result = RunResult(
        scenario="longlive_t2v_short",
        orchestrator_id="O-abc",
        passed=True,
        timings=PhaseTimings(first_frame_s=12.0, pipeline_load_s=15.0),
        fps_samples=[9.0, 10.0],
    )
    update_baseline(store, result)

    baselines = store.load_baselines()
    assert "longlive_t2v_short" in baselines
    assert baselines["longlive_t2v_short"]["sample_count"] == 1


def test_check_drift_no_regression(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")

    # Seed baseline with some samples
    for i in range(10):
        result = RunResult(
            scenario="longlive_t2v_short",
            orchestrator_id="O-abc",
            passed=True,
            timings=PhaseTimings(first_frame_s=12.0),
            fps_samples=[10.0],
        )
        update_baseline(store, result)

    # Check a similar result — no drift
    current = RunResult(
        scenario="longlive_t2v_short",
        orchestrator_id="O-abc",
        passed=True,
        timings=PhaseTimings(first_frame_s=13.0),
        fps_samples=[9.5],
    )
    drift = check_drift(store, current, threshold=0.20)

    assert not drift.first_frame_drifted
    assert not drift.fps_drifted


def test_check_drift_regression(tmp_path: Path):
    store = BaselineStore(tmp_path / "baselines.json", tmp_path / "history.json")

    for i in range(10):
        result = RunResult(
            scenario="longlive_t2v_short",
            orchestrator_id="O-abc",
            passed=True,
            timings=PhaseTimings(first_frame_s=12.0),
            fps_samples=[10.0],
        )
        update_baseline(store, result)

    # Much worse result — should flag drift
    current = RunResult(
        scenario="longlive_t2v_short",
        orchestrator_id="O-abc",
        passed=True,
        timings=PhaseTimings(first_frame_s=20.0),  # 67% worse
        fps_samples=[5.0],  # 50% worse
    )
    drift = check_drift(store, current, threshold=0.20)

    assert drift.first_frame_drifted
    assert drift.fps_drifted
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_regression.py -v`
Expected: FAIL

- [ ] **Step 3: Implement regression module**

```python
# src/loadtest/regression.py
"""Rolling baseline management and drift detection."""

import json
import logging
import statistics
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .results import RunResult

logger = logging.getLogger(__name__)

MAX_HISTORY_DAYS = 7


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

    def __init__(self, baselines_path: Path, history_path: Path):
        self._baselines_path = baselines_path
        self._history_path = history_path

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

        # Prune entries older than MAX_HISTORY_DAYS
        cutoff = datetime.now(timezone.utc).timestamp() - (MAX_HISTORY_DAYS * 86400)
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

    first_frame_values = [e["first_frame_s"] for e in scenario_entries if e.get("first_frame_s") is not None]
    fps_values = [e["avg_fps"] for e in scenario_entries if e.get("avg_fps") is not None]
    load_values = [e["pipeline_load_s"] for e in scenario_entries if e.get("pipeline_load_s") is not None]

    baselines = store.load_baselines()
    baselines[result.scenario] = {
        "first_frame_p50": statistics.median(first_frame_values) if first_frame_values else None,
        "first_frame_p95": _percentile(first_frame_values, 0.95) if first_frame_values else None,
        "steady_fps_p50": statistics.median(fps_values) if fps_values else None,
        "pipeline_load_p50": statistics.median(load_values) if load_values else None,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "sample_count": len(scenario_entries),
    }
    store.save_baselines(baselines)


def check_drift(store: BaselineStore, result: RunResult, threshold: float = 0.20) -> DriftResult:
    """Compare a run result against the rolling baseline. Returns drift flags."""
    baselines = store.load_baselines()
    baseline = baselines.get(result.scenario)
    drift = DriftResult()

    if not baseline or baseline.get("sample_count", 0) < 5:
        return drift  # not enough data for comparison

    # First frame drift
    p50_ff = baseline.get("first_frame_p50")
    if p50_ff and p50_ff > 0 and result.timings.first_frame_s is not None:
        pct = (result.timings.first_frame_s - p50_ff) / p50_ff
        drift.first_frame_drift_pct = pct
        drift.first_frame_drifted = pct > threshold

    # FPS drift (lower is worse)
    p50_fps = baseline.get("steady_fps_p50")
    avg_fps = result.avg_fps
    if p50_fps and p50_fps > 0 and avg_fps is not None:
        pct = (p50_fps - avg_fps) / p50_fps
        drift.fps_drift_pct = pct
        drift.fps_drifted = pct > threshold

    # Pipeline load drift
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_regression.py -v`
Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add src/loadtest/regression.py tests/test_regression.py
git commit -s -m "feat: regression detection with rolling baselines and drift alerting"
```

---

## Phase 5: Discovery, Coverage & Scheduler

Produces: orchestrator discovery, coverage tracking, and the scheduler daemon that ties everything together.

### Task 11: Coverage tracker

**Files:**
- Create: `src/loadtest/coverage.py`
- Create: `tests/test_coverage.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_coverage.py
from pathlib import Path
from loadtest.coverage import CoverageTracker


def test_record_run(tmp_path: Path):
    tracker = CoverageTracker(tmp_path / "coverage.json")
    tracker.record_run("O-abc", "longlive_t2v_short", passed=True)
    tracker.record_run("O-abc", "ltx2_t2v_mid", passed=False, failure_category="runner")

    day = tracker.get_today()
    assert day["O-abc"]["runs_completed"] == 2
    assert day["O-abc"]["failures"] == 1
    assert "longlive_t2v_short" in day["O-abc"]["scenarios_covered"]
    assert day["O-abc"]["failure_categories"]["runner"] == 1


def test_set_planned_runs(tmp_path: Path):
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
    # O-def has 0 runs → higher debt

    debt = tracker.get_test_debt()
    assert debt["O-def"] > debt["O-abc"]


def test_coverage_persists(tmp_path: Path):
    path = tmp_path / "coverage.json"
    tracker1 = CoverageTracker(path)
    tracker1.record_run("O-abc", "s1", passed=True)

    tracker2 = CoverageTracker(path)
    day = tracker2.get_today()
    assert day["O-abc"]["runs_completed"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_coverage.py -v`
Expected: FAIL

- [ ] **Step 3: Implement coverage tracker**

```python
# src/loadtest/coverage.py
"""Track which orchestrators have been tested and their daily progress."""

import json
from datetime import datetime, timezone
from pathlib import Path


class CoverageTracker:
    """Tracks per-orchestrator, per-day test coverage. Persists to JSON."""

    def __init__(self, path: Path):
        self._path = path
        self._data: dict = self._load()

    def _load(self) -> dict:
        if self._path.exists():
            with open(self._path) as f:
                return json.load(f)
        return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump(self._data, f, indent=2)

    def _today_key(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _ensure_day(self, orchestrator_id: str) -> dict:
        day_key = self._today_key()
        if day_key not in self._data:
            self._data[day_key] = {}
        if orchestrator_id not in self._data[day_key]:
            self._data[day_key][orchestrator_id] = {
                "runs_completed": 0,
                "runs_planned": 0,
                "scenarios_covered": [],
                "failures": 0,
                "failure_categories": {},
            }
        return self._data[day_key][orchestrator_id]

    def get_today(self) -> dict:
        return self._data.get(self._today_key(), {})

    def set_planned(self, orchestrator_id: str, runs_planned: int) -> None:
        entry = self._ensure_day(orchestrator_id)
        entry["runs_planned"] = runs_planned
        self._save()

    def record_run(
        self,
        orchestrator_id: str,
        scenario: str,
        passed: bool,
        failure_category: str | None = None,
    ) -> None:
        entry = self._ensure_day(orchestrator_id)
        entry["runs_completed"] += 1
        if scenario not in entry["scenarios_covered"]:
            entry["scenarios_covered"].append(scenario)
        if not passed:
            entry["failures"] += 1
            if failure_category:
                cats = entry["failure_categories"]
                cats[failure_category] = cats.get(failure_category, 0) + 1
        self._save()

    def get_test_debt(self) -> dict[str, int]:
        """Return {orchestrator_id: remaining_runs} for today, sorted by most debt."""
        today = self.get_today()
        debt = {}
        for oid, entry in today.items():
            remaining = entry["runs_planned"] - entry["runs_completed"]
            debt[oid] = max(0, remaining)
        return dict(sorted(debt.items(), key=lambda x: -x[1]))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_coverage.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/loadtest/coverage.py tests/test_coverage.py
git commit -s -m "feat: coverage tracker with daily persistence and test debt calculation"
```

### Task 12: Orchestrator discovery

**Files:**
- Create: `src/loadtest/discovery.py`
- Create: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_discovery.py
import pytest
import respx
from loadtest.discovery import (
    Orchestrator,
    OrchestratorRegistry,
)


def test_orchestrator_is_healthy():
    o = Orchestrator(id="O-abc", address="http://gateway:8001")
    assert o.status == "healthy"
    assert o.consecutive_failures == 0


def test_orchestrator_blacklisted_after_failures():
    o = Orchestrator(id="O-abc", address="http://gateway:8001")
    for _ in range(5):
        o.record_failure()
    assert o.status == "blacklisted"


def test_orchestrator_recovers():
    o = Orchestrator(id="O-abc", address="http://gateway:8001")
    o.record_failure()
    o.record_failure()
    o.record_success()
    assert o.status == "healthy"
    assert o.consecutive_failures == 0


def test_registry_add_and_list():
    registry = OrchestratorRegistry(max_consecutive_failures=5)
    registry.upsert(Orchestrator(id="O-1", address="http://g1:8001"))
    registry.upsert(Orchestrator(id="O-2", address="http://g2:8001"))

    healthy = registry.get_healthy()
    assert len(healthy) == 2


def test_registry_filters_blacklisted():
    registry = OrchestratorRegistry(max_consecutive_failures=3)
    o = Orchestrator(id="O-bad", address="http://bad:8001")
    for _ in range(3):
        o.record_failure()
    registry.upsert(o)
    registry.upsert(Orchestrator(id="O-good", address="http://good:8001"))

    healthy = registry.get_healthy()
    assert len(healthy) == 1
    assert healthy[0].id == "O-good"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_discovery.py -v`
Expected: FAIL

- [ ] **Step 3: Implement discovery module**

```python
# src/loadtest/discovery.py
"""Livepeer orchestrator discovery and health tracking."""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

DEFAULT_MAX_FAILURES = 5


@dataclass
class Orchestrator:
    id: str
    address: str
    region: str | None = None
    first_seen: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_healthy: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_tested: datetime | None = None
    consecutive_failures: int = 0
    _max_failures: int = DEFAULT_MAX_FAILURES

    @property
    def status(self) -> str:
        if self.consecutive_failures >= self._max_failures:
            return "blacklisted"
        if self.consecutive_failures > 0:
            return "unhealthy"
        return "healthy"

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.last_healthy = datetime.now(timezone.utc)

    def record_tested(self) -> None:
        self.last_tested = datetime.now(timezone.utc)


class OrchestratorRegistry:
    """In-memory registry of known orchestrators."""

    def __init__(self, max_consecutive_failures: int = DEFAULT_MAX_FAILURES):
        self._orchestrators: dict[str, Orchestrator] = {}
        self._max_failures = max_consecutive_failures

    def upsert(self, orchestrator: Orchestrator) -> None:
        orchestrator._max_failures = self._max_failures
        self._orchestrators[orchestrator.id] = orchestrator

    def get(self, oid: str) -> Orchestrator | None:
        return self._orchestrators.get(oid)

    def get_all(self) -> list[Orchestrator]:
        return list(self._orchestrators.values())

    def get_healthy(self) -> list[Orchestrator]:
        return [o for o in self._orchestrators.values() if o.status == "healthy"]

    def reset_blacklists(self) -> int:
        """Reset all blacklisted orchestrators to healthy. Returns count reset."""
        count = 0
        for o in self._orchestrators.values():
            if o.status == "blacklisted":
                o.consecutive_failures = 0
                count += 1
        return count


async def discover_orchestrators(
    discovery_url: str,
    livepeer_token: str | None = None,
) -> list[Orchestrator]:
    """Query the Livepeer discovery endpoint for available orchestrators.

    This is a placeholder that should be implemented against the actual
    Livepeer discovery API. Returns a list of Orchestrator records.
    """
    import httpx

    orchestrators = []
    try:
        headers = {}
        if livepeer_token:
            headers["Authorization"] = f"Bearer {livepeer_token}"

        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(discovery_url, headers=headers)
            resp.raise_for_status()
            data = resp.json()

            for entry in data:
                orchestrators.append(
                    Orchestrator(
                        id=entry.get("id", entry.get("address", "unknown")),
                        address=entry.get("address", ""),
                        region=entry.get("region"),
                    )
                )
    except Exception as e:
        logger.error("Orchestrator discovery failed: %s", e)

    logger.info("Discovered %d orchestrators", len(orchestrators))
    return orchestrators
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_discovery.py -v`
Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add src/loadtest/discovery.py tests/test_discovery.py
git commit -s -m "feat: orchestrator discovery with health tracking and blacklist"
```

### Task 13: Scheduler daemon

**Files:**
- Create: `src/loadtest/scheduler.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_scheduler.py
from loadtest.scheduler import RunSlot, build_run_plan
from loadtest.discovery import Orchestrator


def test_build_run_plan_distributes_evenly():
    orchestrators = [
        Orchestrator(id="O-1", address="http://g1"),
        Orchestrator(id="O-2", address="http://g2"),
        Orchestrator(id="O-3", address="http://g3"),
    ]
    scenarios = ["s1", "s2", "s3", "s4"]
    runs_per_o = 4
    num_instances = 2

    plan = build_run_plan(orchestrators, scenarios, runs_per_o, num_instances)

    # Each orchestrator gets 4 runs
    for o in orchestrators:
        o_slots = [s for s in plan if s.orchestrator_id == o.id]
        assert len(o_slots) == runs_per_o

    # At most num_instances concurrent at any time slot
    by_time: dict[int, list[RunSlot]] = {}
    for slot in plan:
        by_time.setdefault(slot.slot_index, []).append(slot)
    for slots in by_time.values():
        assert len(slots) <= num_instances


def test_build_run_plan_rotates_scenarios():
    orchestrators = [Orchestrator(id="O-1", address="http://g1")]
    scenarios = ["s1", "s2", "s3"]
    runs_per_o = 6

    plan = build_run_plan(orchestrators, scenarios, runs_per_o, num_instances=1)

    assigned_scenarios = [s.scenario for s in plan]
    # Each scenario should appear exactly twice (6 runs / 3 scenarios)
    for sc in scenarios:
        assert assigned_scenarios.count(sc) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scheduler.py -v`
Expected: FAIL

- [ ] **Step 3: Implement scheduler**

```python
# src/loadtest/scheduler.py
"""Scheduler daemon: budget calculation, run timing, orchestrator rotation."""

import asyncio
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .config import LoadTestConfig
from .coverage import CoverageTracker
from .discovery import (
    Orchestrator,
    OrchestratorRegistry,
    discover_orchestrators,
)
from .executor import Executor
from .metrics import MetricsCollector
from .regression import BaselineStore, check_drift, update_baseline
from .scenarios import Scenario, load_all_scenarios, load_prompt_pool

logger = logging.getLogger(__name__)


@dataclass
class RunSlot:
    slot_index: int
    orchestrator_id: str
    scenario: str


def build_run_plan(
    orchestrators: list[Orchestrator],
    scenarios: list[str],
    runs_per_orchestrator: int,
    num_instances: int,
) -> list[RunSlot]:
    """Build a daily run plan interleaving orchestrators across time slots.

    Returns a list of RunSlot ordered by slot_index. At most num_instances
    slots share the same slot_index.
    """
    slots: list[RunSlot] = []

    # Build flat list: each orchestrator gets runs_per_orchestrator entries
    pending: list[tuple[str, str]] = []  # (orchestrator_id, scenario)
    for o in orchestrators:
        for i in range(runs_per_orchestrator):
            scenario = scenarios[i % len(scenarios)]
            pending.append((o.id, scenario))

    # Interleave: round-robin across orchestrators
    # Group by orchestrator, then zip across groups
    by_o: dict[str, list[str]] = {}
    for oid, sc in pending:
        by_o.setdefault(oid, []).append(sc)

    # Flatten interleaved
    interleaved: list[tuple[str, str]] = []
    max_runs = max(len(v) for v in by_o.values()) if by_o else 0
    oids = list(by_o.keys())
    for run_idx in range(max_runs):
        for oid in oids:
            if run_idx < len(by_o[oid]):
                interleaved.append((oid, by_o[oid][run_idx]))

    # Assign slot indices (at most num_instances per slot)
    slot_idx = 0
    count_in_slot = 0
    for oid, sc in interleaved:
        slots.append(RunSlot(slot_index=slot_idx, orchestrator_id=oid, scenario=sc))
        count_in_slot += 1
        if count_in_slot >= num_instances:
            slot_idx += 1
            count_in_slot = 0

    return slots


async def run_scheduler(
    config: LoadTestConfig,
    config_dir: Path,
    data_dir: Path,
) -> None:
    """Main scheduler loop. Runs until cancelled."""
    scenarios_dir = config_dir / "scenarios"
    prompts_dir = config_dir / "prompts"

    scenarios = load_all_scenarios(scenarios_dir)
    scenario_names = [s.name for s in scenarios]
    scenario_map = {s.name: s for s in scenarios}

    coverage = CoverageTracker(data_dir / "coverage.json")
    baseline_store = BaselineStore(
        data_dir / "baselines.json", data_dir / "history.json"
    )

    push_url = os.environ.get("GRAFANA_PUSH_URL")
    metrics = MetricsCollector(push_url=push_url)
    executor = Executor(config, data_dir=data_dir)

    scope_instances = os.environ.get("SCOPE_INSTANCES", "").split(",")
    scope_instances = [s.strip() for s in scope_instances if s.strip()]
    if not scope_instances:
        logger.error("No SCOPE_INSTANCES configured")
        return

    scope_urls = [f"http://{inst}" for inst in scope_instances]

    discovery_url = os.environ.get("LIVEPEER_DISCOVERY_URL", "")
    livepeer_token = os.environ.get("LIVEPEER_TOKEN")
    app_id = os.environ.get("SCOPE_CLOUD_APP_ID", "")
    api_key = os.environ.get("SCOPE_CLOUD_API_KEY")

    registry = OrchestratorRegistry(
        max_consecutive_failures=config.discovery.max_consecutive_failures,
    )

    logger.info(
        "Scheduler starting: %d instances, budget=%d%%, max_run=%dm",
        len(scope_urls),
        config.budget.daily_percent,
        config.budget.max_run_duration_mins,
    )

    # Discovery + planning loop
    last_discovery = 0.0
    discovery_interval = config.discovery.refresh_interval_hours * 3600

    while True:
        now = datetime.now(timezone.utc)

        # Periodic discovery
        if now.timestamp() - last_discovery > discovery_interval:
            logger.info("Running orchestrator discovery...")
            discovered = await discover_orchestrators(discovery_url, livepeer_token)
            for o in discovered:
                registry.upsert(o)
            last_discovery = now.timestamp()

            # Plan the day
            healthy = registry.get_healthy()
            runs_per_o = config.budget.runs_per_orchestrator_per_day
            for o in healthy:
                coverage.set_planned(o.id, runs_per_o)

            logger.info(
                "Plan: %d healthy orchestrators, %d runs each",
                len(healthy), runs_per_o,
            )

        # Find orchestrators with test debt
        debt = coverage.get_test_debt()
        if not debt or all(v == 0 for v in debt.values()):
            logger.info("All orchestrators at budget for today, sleeping 5m...")
            await asyncio.sleep(300)
            continue

        # Assign runs to available Scope instances
        tasks = []
        for i, scope_url in enumerate(scope_urls):
            # Pick the orchestrator with most debt
            candidates = [(oid, d) for oid, d in debt.items() if d > 0]
            if not candidates:
                break

            oid = candidates[0][0]
            debt[oid] -= 1

            # Pick scenario
            o_coverage = coverage.get_today().get(oid, {})
            covered = set(o_coverage.get("scenarios_covered", []))
            uncovered = [s for s in scenario_names if s not in covered]
            scenario_name = uncovered[0] if uncovered else scenario_names[i % len(scenario_names)]
            scenario = scenario_map[scenario_name]

            # Load prompts
            pool_name = scenario.prompts.get("pool", "nature")
            try:
                prompts = load_prompt_pool(pool_name, prompts_dir)
            except FileNotFoundError:
                prompts = ["a scenic landscape"]

            orchestrator = registry.get(oid)

            async def _run_one(url, orch_id, sc, pr):
                result = await executor.run(
                    scope_url=url,
                    orchestrator_id=orch_id,
                    scenario=sc,
                    prompts=pr,
                    app_id=app_id,
                    api_key=api_key,
                )
                # Record coverage
                coverage.record_run(
                    orch_id, sc.name, result.passed,
                    failure_category=result.error_category.value if result.error_category else None,
                )
                # Update baselines and check drift
                if result.passed:
                    update_baseline(baseline_store, result)
                    drift = check_drift(baseline_store, result, config.thresholds.regression_drift_threshold)
                    if drift.first_frame_drifted or drift.fps_drifted:
                        logger.warning("Drift detected for %s on %s", sc.name, orch_id)

                # Update orchestrator health
                if orchestrator:
                    if result.passed:
                        orchestrator.record_success()
                    elif result.error_category in ("network", "orchestrator"):
                        orchestrator.record_failure()
                    orchestrator.record_tested()

                # Push metrics
                metrics.record_run(result)
                metrics.push()

                return result

            tasks.append(_run_one(scope_url, oid, scenario, prompts))

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.error("Run task failed: %s", r)

        # Wait before next batch
        gap_s = config.budget.min_run_gap_mins * 60
        logger.info("Batch complete, waiting %ds before next batch", gap_s)
        await asyncio.sleep(gap_s)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_scheduler.py -v`
Expected: 2 passed

- [ ] **Step 5: Wire CLI commands to scheduler and executor**

Update `src/loadtest/cli.py` — replace the placeholder `schedule` and `run` commands with real implementations that call the scheduler and executor:

```python
# Replace the schedule command in cli.py:
@main.command()
@click.pass_context
def schedule(ctx: click.Context):
    """Start the scheduler daemon."""
    import asyncio
    from pathlib import Path
    from .scheduler import run_scheduler

    config = ctx.obj["config"]
    config_dir = Path("config")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    click.echo("Starting scheduler daemon...")
    asyncio.run(run_scheduler(config, config_dir, data_dir))


# Replace the run command in cli.py:
@main.command()
@click.option("--scenario", required=True, help="Scenario name or YAML path")
@click.option("--orchestrator", default=None, help="Target orchestrator ID")
@click.option("--scope-url", default="http://localhost:8001", help="Scope instance URL")
@click.pass_context
def run(ctx: click.Context, scenario: str, orchestrator: str | None, scope_url: str):
    """Execute a single test run."""
    import asyncio
    from pathlib import Path
    from .executor import Executor
    from .scenarios import load_scenario, load_prompt_pool

    config = ctx.obj["config"]
    scenario_path = Path(f"config/scenarios/{scenario}.yaml")
    if not scenario_path.exists():
        scenario_path = Path(scenario)
    sc = load_scenario(scenario_path)

    pool_name = sc.prompts.get("pool", "nature")
    try:
        prompts = load_prompt_pool(pool_name, Path("config/prompts"))
    except FileNotFoundError:
        prompts = ["a scenic landscape"]

    import os
    app_id = os.environ.get("SCOPE_CLOUD_APP_ID", "")
    oid = orchestrator or "manual"

    executor = Executor(config, data_dir=Path("data"))
    result = asyncio.run(executor.run(scope_url, oid, sc, prompts, app_id))

    if result.passed:
        click.echo(f"PASS: {sc.name} ({result.timings.total_s:.1f}s)")
    else:
        click.echo(f"FAIL: {sc.name} [{result.error_category}] {result.error_message}")
```

- [ ] **Step 6: Commit**

```bash
git add src/loadtest/scheduler.py src/loadtest/cli.py tests/test_scheduler.py
git commit -s -m "feat: scheduler daemon with budget planning, fair rotation, and CLI wiring"
```

---

## Phase 6: Docker Compose & Dashboards

Produces: fully working docker-compose stack and Grafana dashboard definition.

### Task 14: Docker compose and Dockerfile

**Files:**
- Create: `Dockerfile.harness`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create Dockerfile.harness**

```dockerfile
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir .
COPY src/ /app/src/
COPY config/ /app/config/
ENTRYPOINT ["python", "-m", "loadtest.cli"]
CMD ["schedule"]
```

- [ ] **Step 2: Create docker-compose.yml**

```yaml
services:
  harness:
    build:
      context: .
      dockerfile: Dockerfile.harness
    image: scope-loadtest-harness
    depends_on: [pushgateway]
    volumes:
      - ./config:/app/config
      - ./videos:/data/videos
      - ./data:/app/data
    environment:
      - SCOPE_INSTANCES=${SCOPE_INSTANCES:-scope-1:8001,scope-2:8002}
      - GRAFANA_PUSH_URL=http://pushgateway:9091
      - LIVEPEER_DISCOVERY_URL=${LIVEPEER_DISCOVERY_URL}
      - LIVEPEER_TOKEN=${LIVEPEER_TOKEN}
      - SCOPE_CLOUD_APP_ID=${SCOPE_CLOUD_APP_ID}
      - SCOPE_CLOUD_API_KEY=${SCOPE_CLOUD_API_KEY}
    restart: unless-stopped

  scope-1:
    image: daydreamlive/scope:${SCOPE_IMAGE_TAG:-latest}
    command: ["uv", "run", "daydream-scope", "--host", "0.0.0.0", "--port", "8001"]
    environment:
      - CUDA_VISIBLE_DEVICES=
    volumes:
      - ./videos:/data/videos
    ports: ["8001:8001"]
    restart: unless-stopped

  scope-2:
    image: daydreamlive/scope:${SCOPE_IMAGE_TAG:-latest}
    command: ["uv", "run", "daydream-scope", "--host", "0.0.0.0", "--port", "8002"]
    environment:
      - CUDA_VISIBLE_DEVICES=
    volumes:
      - ./videos:/data/videos
    ports: ["8002:8002"]
    restart: unless-stopped

  pushgateway:
    image: prom/pushgateway:latest
    ports: ["9091:9091"]
    restart: unless-stopped
```

- [ ] **Step 3: Test Docker build**

Run: `docker build -f Dockerfile.harness -t scope-loadtest-harness .`
Expected: builds successfully

Run: `docker run --rm scope-loadtest-harness --help`
Expected: shows CLI help with schedule, run, discover, coverage, baselines commands

- [ ] **Step 4: Commit**

```bash
git add Dockerfile.harness docker-compose.yml
git commit -s -m "feat: docker-compose stack with harness, scope instances, and pushgateway"
```

### Task 15: Test video generation

**Files:**
- Create: `scripts/generate_test_videos.py`

- [ ] **Step 1: Create video generation script**

```python
# scripts/generate_test_videos.py
"""Generate test input videos for load testing scenarios."""

import sys
from pathlib import Path


def create_test_video(
    path: str, color: tuple[int, int, int],
    width: int = 512, height: int = 512, fps: int = 30, duration_s: int = 30,
) -> None:
    import cv2
    import numpy as np

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = color
    for _ in range(fps * duration_s):
        writer.write(frame)
    writer.release()
    print(f"Created {path} ({width}x{height}, {fps}fps, {duration_s}s)")


def create_gradient_video(
    path: str, width: int = 512, height: int = 512, fps: int = 30, duration_s: int = 30,
) -> None:
    import cv2
    import numpy as np

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    total_frames = fps * duration_s
    for i in range(total_frames):
        t = i / total_frames
        r = int(255 * t)
        g = int(255 * (1 - t))
        b = 128
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (b, g, r)
        writer.write(frame)
    writer.release()
    print(f"Created {path} ({width}x{height}, {fps}fps, {duration_s}s, gradient)")


if __name__ == "__main__":
    videos_dir = "videos"

    create_test_video(f"{videos_dir}/solid_red_512x512_30s.mp4", (0, 0, 255))
    create_test_video(f"{videos_dir}/solid_green_512x512_30s.mp4", (0, 255, 0))
    create_gradient_video(f"{videos_dir}/gradient_512x512_30s.mp4")
    # scene_change: red first half, blue second half
    create_test_video(f"{videos_dir}/scene_change_512x512_60s.mp4", (255, 0, 0), duration_s=60)

    print("Done. Videos in:", videos_dir)
```

- [ ] **Step 2: Generate videos**

Run: `pip install opencv-python && python scripts/generate_test_videos.py`
Expected: 4 MP4 files in `videos/`

- [ ] **Step 3: Commit script (not videos — they're gitignored)**

```bash
git add scripts/generate_test_videos.py
git commit -s -m "feat: test video generation script for v2v/i2v scenarios"
```

### Task 16: Grafana dashboard definition

**Files:**
- Create: `dashboards/grafana/scope-loadtest.json`

- [ ] **Step 1: Create dashboard JSON**

Create a Grafana dashboard JSON in `dashboards/grafana/scope-loadtest.json` with 7 panels matching the design spec section 10.3:

1. **Overview** — stat panels: total runs, pass rate gauge, active sessions count, orchestrator count
2. **Per-Orchestrator** — table: orchestrator_id, success rate, avg connect time, avg FPS, runs completed/planned
3. **Per-Pipeline** — bar chart: load time, first-frame latency, steady FPS grouped by pipeline+mode
4. **Latency Trends** — time series: P50/P95 first_frame_seconds over 7 days
5. **Error Breakdown** — stacked bar: failures_total by category over time
6. **Budget & Coverage** — bar gauge: budget_percent_consumed per orchestrator
7. **Quality** — stat panels: frame validation pass rate, prompt sensitivity pass rate

Each panel queries from the Prometheus push gateway metrics defined in Task 8.

The dashboard should use variables for `$orchestrator_id`, `$pipeline`, `$mode` for filtering.

Set up alert rules:
- Pass rate < 70% → alert
- Any orchestrator 0 runs in 4 hours → alert
- First-frame P95 > 2x baseline → alert
- Runner errors > 10% → alert
- Black frame rate > 5% → alert

- [ ] **Step 2: Commit**

```bash
git add dashboards/grafana/scope-loadtest.json
git commit -s -m "feat: Grafana dashboard with 7 panels and alert rules"
```

---

## Final: Integration Test

### Task 17: End-to-end smoke test

- [ ] **Step 1: Verify full Docker stack starts**

```bash
docker compose build
docker compose up -d pushgateway
docker compose up -d harness
docker compose logs harness
```

Expected: Harness starts, prints "Starting scheduler daemon..."

- [ ] **Step 2: Test manual single run (against local mock or real Scope)**

```bash
docker compose exec harness python -m loadtest.cli run \
  --scenario longlive_t2v_short \
  --scope-url http://scope-1:8001
```

Expected: Either PASS or FAIL with a clear error (depending on whether cloud credentials are configured)

- [ ] **Step 3: Verify metrics reach push gateway**

```bash
curl -s http://localhost:9091/metrics | grep scope_loadtest
```

Expected: Prometheus metrics with `scope_loadtest` job label appear

- [ ] **Step 4: Run all unit tests**

```bash
pip install -e ".[dev]"
pytest -v
```

Expected: All tests pass (config, cli, scenarios, scope_client, results, validators, metrics, regression, coverage, discovery, scheduler)

- [ ] **Step 5: Final commit**

```bash
git add -A
git commit -s -m "chore: integration smoke test verification complete"
```
