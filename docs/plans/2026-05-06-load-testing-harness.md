# Scope Load Testing Harness — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a fully automated, configurable load testing harness that continuously validates Scope cloud inference across the Livepeer orchestrator network, reporting to Grafana.

**Architecture:** A lightweight Python harness (no ML deps) drives Scope instances via HTTP API. Packaged in docker-compose for portability. Scheduler manages daily traffic budgets per orchestrator. Prometheus push gateway feeds Grafana dashboards.

**Tech Stack:** Python 3.12, httpx, prometheus_client, pyyaml, Pillow, numpy, click, Docker, Prometheus, Grafana

**Design Spec:** `docs/design.md` (v2, revised 2026-05-06)

**Key simplifications from v1 plan:**
- Scenario matrix in config replaces 15 individual YAML files (2 graph templates only)
- Recording download/validation deferred to v2
- scikit-image removed — prompt sensitivity uses Pillow pixel diff
- Prometheus labels reduced from 5 to 3 standard
- Model consistency and routing fairness deferred to v2

---

## Phase 1: Project Scaffolding & Config Layer

Produces: installable Python package, CLI skeleton, config loading, scenario matrix generation.

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
    "numpy>=1.26.0",
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
videos/*.mp4
```

- [ ] **Step 4: Create .env.example**

```bash
LIVEPEER_DISCOVERY_URL=https://discovery.livepeer.org
LIVEPEER_TOKEN=
SCOPE_CLOUD_APP_ID=
SCOPE_CLOUD_API_KEY=
SCOPE_INSTANCES=scope-1:8001,scope-2:8002
GRAFANA_PUSH_URL=http://pushgateway:9091
SCOPE_IMAGE_TAG=latest
```

- [ ] **Step 5: Verify install**

Run: `pip install -e ".[dev]"`
Expected: installs without errors

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/loadtest/__init__.py .gitignore .env.example
git commit -s -m "feat: project scaffolding with pyproject.toml"
```

### Task 2: Config loading and validation

**Files:**
- Create: `src/loadtest/config.py`
- Create: `config/default.yaml`
- Create: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
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
    assert config.thresholds.connect_timeout_s == 120
    assert len(config.scenario_defs) == 1
    assert config.scenario_defs[0]["pipeline"] == "longlive"


def test_load_config_defaults():
    config = load_config(None)
    assert config.budget.daily_percent == 20
    assert config.thresholds.connect_timeout_s == 120


def test_budget_validates_percent():
    with pytest.raises(ValueError):
        BudgetConfig(daily_percent=0)
    with pytest.raises(ValueError):
        BudgetConfig(daily_percent=101)


def test_budget_runs_per_day():
    b = BudgetConfig(daily_percent=20, max_run_duration_mins=30)
    assert b.runs_per_orchestrator_per_day == 9  # 4.8hrs / 0.5hr = 9.6 → 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py -v`

- [ ] **Step 3: Implement config module**

```python
# src/loadtest/config.py
"""Configuration loading and validation."""

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
    import dataclasses
    names = {f.name for f in dataclasses.fields(cls)}
    return cls(**{k: v for k, v in data.items() if k in names})


def load_config(config_path: Path | None) -> LoadTestConfig:
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
```

- [ ] **Step 4: Create config/default.yaml**

Full default config with scenario matrix, thresholds, and budget. See design spec section 3.1 and 5.1 for the complete content.

- [ ] **Step 5: Run tests, verify pass, commit**

```bash
pytest tests/test_config.py -v
git add src/loadtest/config.py config/default.yaml tests/test_config.py
git commit -s -m "feat: config loading with YAML, validation, and scenario matrix"
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
    result = CliRunner().invoke(main, ["--help"])
    assert result.exit_code == 0
    for cmd in ["schedule", "run", "discover", "coverage", "baselines"]:
        assert cmd in result.output


def test_cli_run_requires_scenario():
    result = CliRunner().invoke(main, ["run"])
    assert result.exit_code != 0
```

- [ ] **Step 2: Implement CLI with placeholder commands**

```python
# src/loadtest/cli.py
"""CLI entrypoint for the load testing harness."""

import click
from .config import load_config

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None)
@click.pass_context
def main(ctx: click.Context, config_path: str | None):
    """Scope cloud inference load testing harness."""
    from pathlib import Path
    path = Path(config_path) if config_path else Path("config/default.yaml")
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(path if path.exists() else None)


@main.command()
@click.option("--scenario", required=True, help="Scenario key (e.g., longlive_t2v_5m)")
@click.option("--orchestrator", default=None, help="Target orchestrator ID")
@click.option("--scope-url", default="http://localhost:8001", help="Scope instance URL")
@click.pass_context
def run(ctx, scenario, orchestrator, scope_url):
    """Execute a single test run."""
    click.echo(f"Running scenario: {scenario}")


@main.command()
@click.pass_context
def schedule(ctx):
    """Start the scheduler daemon."""
    click.echo("Starting scheduler...")


@main.command()
@click.pass_context
def discover(ctx):
    """List available orchestrators and their health status."""
    click.echo("Discovering orchestrators...")


@main.command()
@click.pass_context
def coverage(ctx):
    """Show today's test coverage report."""
    click.echo("Coverage report...")


@main.command()
@click.pass_context
def baselines(ctx):
    """Show current baseline metrics."""
    click.echo("Baselines...")
```

- [ ] **Step 3: Run tests, verify pass, commit**

```bash
pytest tests/test_cli.py -v
git add src/loadtest/cli.py tests/test_cli.py
git commit -s -m "feat: CLI skeleton with click"
```

### Task 4: Scenario matrix and graph templates

**Files:**
- Create: `src/loadtest/scenarios.py`
- Create: `config/graphs/chain_longlive_rife.yaml`
- Create: `config/graphs/chain_depth_longlive_rife.yaml`
- Create: `config/prompts/nature.yaml`
- Create: `config/prompts/urban.yaml`
- Create: `config/prompts/abstract.yaml`
- Create: `config/prompts/stress.yaml`
- Create: `tests/test_scenarios.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_scenarios.py
from loadtest.scenarios import expand_scenario_matrix, Scenario, build_session_body, load_prompt_pool
from pathlib import Path


def test_expand_matrix_single_pipeline():
    defs = [{"pipeline": "longlive", "modes": ["t2v", "v2v"], "durations": [1, 5], "prompts_pool": "nature", "parameters": {"width": 512}}]
    scenarios = expand_scenario_matrix(defs, graphs_dir=Path("/nonexistent"))
    assert len(scenarios) == 4  # 2 modes * 2 durations
    names = {s.name for s in scenarios}
    assert "longlive_t2v_1m" in names
    assert "longlive_v2v_5m" in names


def test_expand_matrix_with_graph(tmp_path: Path):
    graphs_dir = tmp_path / "graphs"
    graphs_dir.mkdir()
    (graphs_dir / "chain_test.yaml").write_text("""
nodes:
  - {id: input, type: source, source_mode: video_file, source_name: /data/videos/test.mp4}
  - {id: longlive, type: pipeline, pipeline_id: longlive}
  - {id: rife, type: pipeline, pipeline_id: rife}
  - {id: output, type: sink}
edges:
  - {from: input, from_port: video, to_node: longlive, to_port: video, kind: stream}
  - {from: longlive, from_port: video, to_node: rife, to_port: video, kind: stream}
  - {from: rife, from_port: video, to_node: output, to_port: video, kind: stream}
""")
    defs = [{"pipeline": "longlive+rife", "modes": ["v2v"], "durations": [5], "graph_template": "chain_test", "prompts_pool": "nature", "parameters": {}}]
    scenarios = expand_scenario_matrix(defs, graphs_dir)
    assert len(scenarios) == 1
    assert scenarios[0].graph is not None
    assert scenarios[0].pipeline_ids == ["longlive", "rife"]


def test_expand_matrix_missing_graph_template(tmp_path: Path):
    """Missing graph template raises FileNotFoundError, not silent None."""
    defs = [{"pipeline": "longlive+rife", "modes": ["v2v"], "durations": [5], "graph_template": "nonexistent", "prompts_pool": "nature", "parameters": {}}]
    import pytest
    with pytest.raises(FileNotFoundError):
        expand_scenario_matrix(defs, tmp_path / "graphs")


def test_expand_matrix_source_files_for_v2v():
    """v2v/i2v scenarios get source_name from source_files config."""
    defs = [{"pipeline": "longlive", "modes": ["t2v", "v2v", "i2v"], "durations": [1], "prompts_pool": "nature",
             "parameters": {"width": 512}, "source_files": {"v2v": "/data/videos/gradient.mp4", "i2v": "/data/videos/red.mp4"}}]
    scenarios = expand_scenario_matrix(defs, graphs_dir=Path("/nonexistent"))
    by_mode = {s.mode: s for s in scenarios}
    assert "source_name" not in by_mode["t2v"].parameters
    assert by_mode["v2v"].parameters["source_name"] == "/data/videos/gradient.mp4"
    assert by_mode["i2v"].parameters["source_name"] == "/data/videos/red.mp4"


def test_scenario_duration_class():
    s = Scenario(name="test", pipeline="longlive", mode="t2v", duration_mins=1, graph=None, prompts_pool="nature", parameters={})
    assert s.duration_class == "short"
    s2 = Scenario(name="test", pipeline="longlive", mode="t2v", duration_mins=5, graph=None, prompts_pool="nature", parameters={})
    assert s2.duration_class == "mid"
    s3 = Scenario(name="test", pipeline="longlive", mode="t2v", duration_mins=15, graph=None, prompts_pool="nature", parameters={})
    assert s3.duration_class == "long"


def test_build_session_body_t2v():
    s = Scenario(name="longlive_t2v_1m", pipeline="longlive", mode="t2v", duration_mins=1, graph=None, prompts_pool="nature", parameters={"width": 512, "height": 512})
    body = build_session_body(s, "a mountain lake")
    assert body["pipeline_id"] == "longlive"
    assert body["input_mode"] == "text"
    assert body["prompts"] == [{"text": "a mountain lake", "weight": 100}]


def test_build_session_body_v2v_graph():
    graph = {"nodes": [{"id": "input", "type": "source"}, {"id": "output", "type": "sink"}], "edges": []}
    s = Scenario(name="chain_v2v_5m", pipeline="longlive+rife", mode="v2v", duration_mins=5, graph=graph, prompts_pool="nature", parameters={})
    body = build_session_body(s, "ocean waves")
    assert "graph" in body
    assert body["input_mode"] == "video"
    assert "pipeline_id" not in body


def test_load_prompt_pool(tmp_path: Path):
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "nature.yaml").write_text("prompts:\n  - 'lake'\n  - 'ocean'\n  - 'forest'\n")
    pool = load_prompt_pool("nature", d)
    assert len(pool) == 3
```

- [ ] **Step 2: Implement scenarios module**

```python
# src/loadtest/scenarios.py
"""Scenario matrix expansion and session body construction."""

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
    prompts_pool: str
    parameters: dict[str, Any]

    @property
    def pipeline_ids(self) -> list[str]:
        if self.graph:
            return [n["pipeline_id"] for n in self.graph["nodes"] if n.get("type") == "pipeline"]
        return [self.pipeline]

    @property
    def duration_class(self) -> str:
        if self.duration_mins <= 2:
            return "short"
        if self.duration_mins <= 10:
            return "mid"
        return "long"

    @property
    def sink_node_id(self) -> str | None:
        if not self.graph:
            return None
        for n in self.graph["nodes"]:
            if n.get("type") == "sink":
                return n["id"]
        return None


def expand_scenario_matrix(
    scenario_defs: list[dict[str, Any]], graphs_dir: Path
) -> list[Scenario]:
    """Expand compact matrix config into concrete Scenario objects."""
    scenarios = []
    for entry in scenario_defs:
        pipeline = entry["pipeline"]
        graph_template = entry.get("graph_template")
        source_files = entry.get("source_files", {})
        graph = None
        if graph_template:
            graph_path = graphs_dir / f"{graph_template}.yaml"
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph template not found: {graph_path}")
            with open(graph_path) as f:
                graph = yaml.safe_load(f)

        for mode in entry.get("modes", ["t2v"]):
            for dur in entry.get("durations", [5]):
                name = f"{pipeline.replace('+', '_')}_{mode}_{dur}m"
                params = dict(entry.get("parameters", {}))
                # Inject source_name for v2v/i2v modes from source_files config
                if mode in source_files:
                    params["source_name"] = source_files[mode]
                scenarios.append(Scenario(
                    name=name,
                    pipeline=pipeline,
                    mode=mode,
                    duration_mins=dur,
                    graph=graph,
                    prompts_pool=entry.get("prompts_pool", "nature"),
                    parameters=params,
                ))
    return scenarios


def load_prompt_pool(pool_name: str, prompts_dir: Path) -> list[str]:
    path = prompts_dir / f"{pool_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Prompt pool not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f).get("prompts", [])


def build_session_body(scenario: Scenario, prompt: str) -> dict[str, Any]:
    if scenario.graph:
        return {
            "input_mode": "video" if scenario.mode in ("v2v", "i2v") else "text",
            "graph": scenario.graph,
            "prompts": [{"text": prompt, "weight": 100}],
        }
    body: dict[str, Any] = {
        "pipeline_id": scenario.pipeline,
        "input_mode": "video" if scenario.mode in ("v2v", "i2v") else "text",
        "prompts": [{"text": prompt, "weight": 100}],
    }
    if scenario.mode in ("v2v", "i2v") and "source_name" in scenario.parameters:
        body["input_source"] = {
            "enabled": True,
            "source_type": "video_file",
            "source_name": scenario.parameters["source_name"],
        }
    return body
```

- [ ] **Step 3: Create graph templates and prompt pools**

Create 2 graph template files in `config/graphs/` and 4 prompt pool files in `config/prompts/`. See design spec sections 5.2 and 5.3.

- [ ] **Step 4: Run tests, verify pass, commit**

```bash
pytest tests/test_scenarios.py -v
git add src/loadtest/scenarios.py config/graphs/ config/prompts/ tests/test_scenarios.py
git commit -s -m "feat: scenario matrix expansion, graph templates, and prompt pools"
```

---

## Phase 2: Scope API Client & Results

Produces: typed async HTTP client, error classification, log capture.

### Task 5: Scope HTTP client

**Files:**
- Create: `src/loadtest/scope_client.py`
- Create: `tests/test_scope_client.py`

- [ ] **Step 1: Write failing test** — test health, cloud_connect, cloud_status, pipeline_load, session_start, session_metrics, capture_frame, session_stop, get_logs using `respx` mocks. See design spec section 11 for all endpoints and response shapes.

- [ ] **Step 2: Implement `ScopeClient`** — async context manager wrapping `httpx.AsyncClient`. One method per API endpoint. All Scope HTTP interactions go through this class.

Methods: `health()`, `cloud_connect(app_id, api_key, user_id)`, `cloud_status()`, `cloud_disconnect()`, `pipeline_load(pipeline_ids)`, `pipeline_status()`, `session_start(body)`, `session_stop()`, `session_metrics()`, `session_parameters(params)`, `capture_frame(sink_node_id, quality)`, `get_logs(lines)`.

- [ ] **Step 3: Run tests, verify pass, commit**

### Task 6: Results and error taxonomy

**Files:**
- Create: `src/loadtest/results.py`
- Create: `tests/test_results.py`

- [ ] **Step 1: Write failing test** — test `RunResult` construction (pass/fail), `classify_error` for timeout→network, connect error→network, HTTP 502→orchestrator, HTTP 500 with CUDA→runner, HTTP 400→protocol. Test `save_failure_logs` and `cleanup_old_failures`.

- [ ] **Step 2: Implement results module**

```python
# src/loadtest/results.py
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
    cold_start: bool | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    labels: dict[str, str] = field(default_factory=dict)

    @property
    def avg_fps(self) -> float | None:
        return sum(self.fps_samples) / len(self.fps_samples) if self.fps_samples else None

    @property
    def vram_growth_mb(self) -> float | None:
        if len(self.vram_samples) < 4:
            return None
        q = len(self.vram_samples) // 4
        return (sum(self.vram_samples[-q:]) / q) - (sum(self.vram_samples[:q]) / q)


def classify_error(error: Exception, response_text: str | None = None) -> ErrorCategory:
    if isinstance(error, (httpx.TimeoutException, httpx.ConnectError, ConnectionError, OSError)):
        return ErrorCategory.NETWORK
    if isinstance(error, httpx.HTTPStatusError):
        status = error.response.status_code
        text = (response_text or "").lower()
        if status in (502, 503, 504):
            return ErrorCategory.ORCHESTRATOR
        if status == 500:
            if any(kw in text for kw in ("cuda", "oom", "out of memory", "pipeline", "torch")):
                return ErrorCategory.RUNNER
            return ErrorCategory.RUNNER
        if status in (400, 422):
            return ErrorCategory.PROTOCOL
        return ErrorCategory.ORCHESTRATOR
    return ErrorCategory.PROTOCOL


def save_failure_logs(logs: str, orchestrator_id: str, scenario: str, data_dir: Path) -> Path:
    date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    ts = datetime.now(timezone.utc).strftime("%H%M%S")
    d = data_dir / "failures" / date_str
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{orchestrator_id}_{scenario}_{ts}.log"
    path.write_text(logs)
    return path


def cleanup_old_failures(data_dir: Path, max_age_days: int = 7) -> int:
    import shutil
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
                shutil.rmtree(day_dir)
                removed += 1
        except ValueError:
            continue
    return removed
```

- [ ] **Step 3: Run tests, verify pass, commit**

---

## Phase 3: Validators & Metrics

Produces: frame validation (Pillow-only), prompt sensitivity via pixel diff, Prometheus metrics with reduced cardinality.

### Task 7: Frame validators (no scikit-image)

**Files:**
- Create: `src/loadtest/validators.py`
- Create: `tests/test_validators.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_validators.py
import io
from PIL import Image
from loadtest.validators import FrameCheckResult, validate_frame, check_prompt_sensitivity


def _make_jpeg(w, h, color):
    img = Image.new("RGB", (w, h), color)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def test_validate_frame_valid():
    assert validate_frame(_make_jpeg(512, 512, (128, 64, 200)), 512, 512) == FrameCheckResult.VALID

def test_validate_frame_black():
    assert validate_frame(_make_jpeg(512, 512, (0, 0, 0)), 512, 512) == FrameCheckResult.BLACK

def test_validate_frame_wrong_size():
    assert validate_frame(_make_jpeg(256, 256, (128, 64, 200)), 512, 512) == FrameCheckResult.WRONG_SIZE

def test_validate_frame_corrupt():
    assert validate_frame(b"not jpeg", 512, 512) == FrameCheckResult.CORRUPT

def test_prompt_sensitivity_different():
    assert check_prompt_sensitivity(_make_jpeg(512, 512, (255, 0, 0)), _make_jpeg(512, 512, (0, 0, 255))) is True

def test_prompt_sensitivity_identical():
    img = _make_jpeg(512, 512, (128, 128, 128))
    assert check_prompt_sensitivity(img, img) is False
```

- [ ] **Step 2: Implement validators (Pillow + numpy only)**

```python
# src/loadtest/validators.py
"""Frame validation and prompt sensitivity using Pillow + numpy only."""

import io
from enum import Enum
import numpy as np
from PIL import Image


class FrameCheckResult(str, Enum):
    VALID = "valid"
    BLACK = "black"
    CORRUPT = "corrupt"
    WRONG_SIZE = "wrong_size"


def validate_frame(jpeg_bytes: bytes, expected_w: int, expected_h: int, variance_min: float = 5.0) -> FrameCheckResult:
    try:
        img = Image.open(io.BytesIO(jpeg_bytes))
        img.load()
    except Exception:
        return FrameCheckResult.CORRUPT
    if img.size != (expected_w, expected_h):
        return FrameCheckResult.WRONG_SIZE
    if np.array(img, dtype=np.float32).std() < variance_min:
        return FrameCheckResult.BLACK
    return FrameCheckResult.VALID


def check_prompt_sensitivity(frame_before: bytes, frame_after: bytes, min_diff: float = 10.0) -> bool:
    """Returns True if frames are sufficiently different (model responds to prompts)."""
    try:
        a = np.array(Image.open(io.BytesIO(frame_before)).convert("RGB"), dtype=np.float32)
        b = np.array(Image.open(io.BytesIO(frame_after)).convert("RGB"), dtype=np.float32)
    except Exception:
        return False
    if a.shape != b.shape:
        return True
    mean_diff = np.abs(a - b).mean()
    return mean_diff >= min_diff
```

- [ ] **Step 3: Run tests, verify pass, commit**

### Task 8: Prometheus metrics (3 standard labels)

**Files:**
- Create: `src/loadtest/metrics.py`
- Create: `tests/test_metrics.py`

- [ ] **Step 1: Write failing test** — verify MetricsCollector creates metrics, `record_run` accepts pass and fail results without error.

- [ ] **Step 2: Implement metrics**

Standard labels on all metrics: `orchestrator_id`, `pipeline`, `mode`.
Extended labels (`scenario`, `duration_class`) only on `runs_total` and `failures_total`.

Histograms: `connect_duration_seconds`, `pipeline_load_seconds`, `first_frame_seconds`.
Gauges: `stream_fps_out`, `vram_allocated_mb`, `budget_runs_planned/completed`, `orchestrator_coverage_percent`, `baseline_drift_percent`.
Counters: `runs_total`, `failures_total`, `frames_validated_total`, `prompt_sensitivity_checks_total`.

Push via `push_to_gateway` after each run.

- [ ] **Step 3: Run tests, verify pass, commit**

---

## Phase 4: Executor & Regression

Produces: the core executor driving complete scenario lifecycles, plus drift detection.

### Task 9: Test executor

**Files:**
- Create: `src/loadtest/executor.py`
- Create: `tests/test_executor.py`

- [ ] **Step 1: Write failing test** — test `_connect_phase` (success + timeout), `_load_phase` (success), and the full `run()` method with a short monitoring loop, using respx mocks.

The monitoring loop test should mock the full lifecycle:
```python
@respx.mock
@pytest.mark.asyncio
async def test_executor_full_run_t2v():
    """Full executor run with connect, load, stream monitoring, and cleanup."""
    base = "http://scope-1:8001"
    # Connect
    respx.post(f"{base}/api/v1/cloud/connect").respond(json={"connecting": True})
    respx.get(f"{base}/api/v1/cloud/status").respond(json={"connected": True, "connecting": False, "webrtc_connected": True})
    # Load
    respx.post(f"{base}/api/v1/pipeline/load").respond(json={"status": "loading"})
    respx.get(f"{base}/api/v1/pipeline/status").respond(json={"status": "loaded"})
    # Session
    respx.post(f"{base}/api/v1/session/start").respond(json={"status": "ok"})
    respx.get(f"{base}/api/v1/session/metrics").respond(json={
        "sessions": {"headless": {"fps_out": 10.0, "frames_out": 50, "fps_in": 30.0}},
        "gpu": {"vram_allocated_mb": 8000, "vram_total_mb": 81920},
    })
    # Frame capture (valid non-black JPEG)
    from tests.test_validators import _make_jpeg
    respx.get(f"{base}/api/v1/session/frame").respond(content=_make_jpeg(512, 512, (128, 64, 200)), headers={"content-type": "image/jpeg"})
    # Parameters (prompt switch)
    respx.post(f"{base}/api/v1/session/parameters").respond(json={"status": "ok"})
    # Stop + disconnect
    respx.post(f"{base}/api/v1/session/stop").respond(json={"status": "ok"})
    respx.post(f"{base}/api/v1/cloud/disconnect").respond(json={"connected": False})

    scenario = _make_scenario(duration_mins=1)  # 1 min = 60s, short enough for test
    config = LoadTestConfig()
    config.thresholds.frame_check_interval_s = 5  # fast checks for test
    config.thresholds.prompt_switch_interval_s = 10

    executor = Executor(config)
    result = await executor.run(
        scope_url=base, orchestrator_id="O-test", scenario=scenario,
        prompts=["prompt A", "prompt B"], app_id="test-app",
    )

    assert result.passed is True
    assert result.timings.connect_s is not None
    assert result.timings.first_frame_s is not None
    assert len(result.fps_samples) > 0
    assert result.frames_validated > 0
```

- [ ] **Step 2: Implement executor**

Core method: `async run(scope_url, orchestrator_id, scenario, prompts, app_id) -> RunResult`

Phases: connect → load → stream (with monitoring loop) → cleanup → report. Watchdog enforces `max_run_duration_mins`. On failure: classify error, capture logs, force cleanup. Always returns a `RunResult` (never raises).

Key difference from v1 plan: no recording download/validation. The monitoring loop captures frames and validates them, switches prompts with pixel diff comparison, and tracks VRAM samples. No recording start/stop/download.

- [ ] **Step 3: Run tests, verify pass, commit**

### Task 10: Regression detection

**Files:**
- Create: `src/loadtest/regression.py`
- Create: `tests/test_regression.py`

- [ ] **Step 1: Write failing test** — test `update_baseline` (new entry), `check_drift` (no regression), `check_drift` (regression detected). Test history pruning (entries > 7 days removed).

- [ ] **Step 2: Implement regression module**

`BaselineStore` manages `baselines.json` and `history.json`. `update_baseline` appends to history, recomputes P50/P95 per scenario key (`{pipeline}_{mode}`). `check_drift` compares current result against baseline with configurable threshold.

No model consistency checks (deferred to v2). No routing fairness (deferred to v2).

- [ ] **Step 3: Run tests, verify pass, commit**

---

## Phase 5: Discovery, Coverage & Scheduler

Produces: orchestrator discovery, coverage tracking (with pruning), scheduler split into plan generation + execution loop.

### Task 11: Coverage tracker (with pruning)

**Files:**
- Create: `src/loadtest/coverage.py`
- Create: `tests/test_coverage.py`

- [ ] **Step 1: Write failing test** — test `record_run`, `set_planned`, `get_test_debt`, persistence across instances, and `prune(max_days=30)` removes old entries.

- [ ] **Step 2: Implement coverage tracker**

Same as v1 plan but add `prune(max_days: int = 30)` called on load:
```python
def _load(self) -> dict:
    if self._path.exists():
        with open(self._path) as f:
            data = json.load(f)
        # Prune entries older than max_days
        cutoff = (datetime.now(timezone.utc) - timedelta(days=self._max_days)).strftime("%Y-%m-%d")
        return {k: v for k, v in data.items() if k >= cutoff}
    return {}
```

- [ ] **Step 3: Run tests, verify pass, commit**

### Task 12: Orchestrator discovery

**Files:**
- Create: `src/loadtest/discovery.py`
- Create: `tests/test_discovery.py`

- [ ] **Step 1: Write failing test** — test Orchestrator health/blacklist/recovery state machine, OrchestratorRegistry add/filter/reset.

- [ ] **Step 2: Implement** — `Orchestrator` dataclass with `record_failure/success/tested`. `OrchestratorRegistry` with `upsert/get_healthy/reset_blacklists`. `discover_orchestrators()` async function that queries Livepeer discovery endpoint. Supports Option C fallback (observational coverage).

- [ ] **Step 3: Run tests, verify pass, commit**

### Task 13a: Run plan generation (pure function)

**Files:**
- Create: `src/loadtest/scheduler.py`
- Create: `tests/test_scheduler.py`

- [ ] **Step 1: Write failing test** — test `build_run_plan` distributes evenly across orchestrators, respects `num_instances` concurrency limit, rotates scenarios.

- [ ] **Step 2: Implement `build_run_plan`**

Pure function: `(orchestrators, scenarios, runs_per_o, num_instances) -> list[RunSlot]`. `RunSlot` is `(slot_index, orchestrator_id, scenario_name)`. Interleaves orchestrators so at most `num_instances` are concurrent per slot.

- [ ] **Step 3: Run tests (3+ test cases), verify pass, commit**

### Task 13b: Scheduler execution loop

**Files:**
- Modify: `src/loadtest/scheduler.py`
- Modify: `src/loadtest/cli.py`

- [ ] **Step 1: Implement `run_scheduler` async function**

Main loop: discovery → plan → execute batches → track coverage → push metrics → sleep → repeat. Wires together all modules: discovery, coverage, executor, metrics, regression, scenarios.

- [ ] **Step 2: Wire CLI `schedule` and `run` commands to real implementations**

`schedule` calls `asyncio.run(run_scheduler(...))`. `run` loads scenario by name from the expanded matrix, runs the executor, prints result.

- [ ] **Step 3: Write integration test for scheduler loop**

```python
# Add to tests/test_scheduler.py
@pytest.mark.asyncio
async def test_scheduler_loop_runs_one_batch(tmp_path, monkeypatch):
    """Scheduler discovers orchestrators, picks one, executes one run, then stops."""
    from loadtest.scheduler import run_scheduler
    from loadtest.config import LoadTestConfig
    from loadtest.discovery import Orchestrator

    config = LoadTestConfig()
    config.budget.min_run_gap_mins = 0  # no gap for test

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    prompts_dir = config_dir / "prompts"
    prompts_dir.mkdir()
    (prompts_dir / "nature.yaml").write_text("prompts:\n  - 'test prompt'\n")

    # Minimal scenario matrix
    config.scenario_defs = [{"pipeline": "longlive", "modes": ["t2v"], "durations": [1], "prompts_pool": "nature", "parameters": {"width": 512, "height": 512}}]

    # Mock discovery to return one orchestrator
    mock_orchestrators = [Orchestrator(id="O-test", address="http://fake")]
    monkeypatch.setattr("loadtest.scheduler.discover_orchestrators", lambda *a, **kw: mock_orchestrators)

    # Mock executor to return a passing result
    from loadtest.results import RunResult, PhaseTimings
    mock_result = RunResult(scenario="longlive_t2v_1m", orchestrator_id="O-test", passed=True,
                            timings=PhaseTimings(connect_s=5, pipeline_load_s=10, first_frame_s=8),
                            fps_samples=[10.0], labels={"pipeline": "longlive", "mode": "t2v", "duration_class": "short"})
    monkeypatch.setattr("loadtest.scheduler.Executor.run", lambda *a, **kw: mock_result)

    # Set env
    monkeypatch.setenv("SCOPE_INSTANCES", "fake:8001")
    monkeypatch.setenv("GRAFANA_PUSH_URL", "")
    monkeypatch.setenv("LIVEPEER_DISCOVERY_URL", "http://fake")
    monkeypatch.setenv("SCOPE_CLOUD_APP_ID", "test")

    # Run scheduler with a short cancellation (one iteration)
    import asyncio
    task = asyncio.create_task(run_scheduler(config, config_dir, tmp_path / "data"))
    await asyncio.sleep(0.5)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Verify coverage was recorded
    coverage_file = tmp_path / "data" / "coverage.json"
    assert coverage_file.exists()
```

- [ ] **Step 4: Run tests, manual smoke test, commit**

```bash
pytest tests/test_scheduler.py -v
python -m loadtest.cli --help
git add src/loadtest/scheduler.py src/loadtest/cli.py tests/test_scheduler.py
git commit -s -m "feat: scheduler with plan generation and execution loop"
```

---

## Phase 6: Docker & Dashboards

Produces: docker-compose stack, test video generation, Grafana dashboard, end-to-end smoke test.

### Task 14: Docker compose and Dockerfile

**Files:**
- Create: `Dockerfile.harness`
- Create: `docker-compose.yml`

- [ ] **Step 1: Create files** — matching design spec section 11.1 and 11.2 exactly.

- [ ] **Step 2: Test build**

```bash
docker build -f Dockerfile.harness -t scope-loadtest-harness .
docker run --rm scope-loadtest-harness --help
```

- [ ] **Step 3: Commit**

### Task 15: Test video generation

**Files:**
- Create: `scripts/generate_test_videos.py`

- [ ] **Step 1: Create script** — generates solid red, solid green, gradient, and scene-change MP4 videos at 512x512 30fps using OpenCV.

- [ ] **Step 2: Generate videos, commit script**

### Task 16: Grafana dashboard

**Files:**
- Create: `dashboards/grafana/scope-loadtest.json`

- [ ] **Step 1: Create dashboard JSON** with 6 panels matching design spec section 10.3. Variables for `$orchestrator_id`, `$pipeline`, `$mode`. Alert rules for pass rate, zero-activity, drift, error rate.

- [ ] **Step 2: Commit**

### Task 17: End-to-end smoke test

- [ ] **Step 1: Docker stack starts**

```bash
docker compose build && docker compose up -d pushgateway harness
docker compose logs harness  # should show "Starting scheduler..."
```

- [ ] **Step 2: All unit tests pass**

```bash
pip install -e ".[dev]" && pytest -v
```

- [ ] **Step 3: Metrics reach push gateway**

```bash
curl -s http://localhost:9091/metrics | grep scope_loadtest
```

- [ ] **Step 4: Final commit**
