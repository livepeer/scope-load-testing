"""CLI entrypoint for the load testing harness."""

import asyncio
import json
import os
from pathlib import Path

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
    path = Path(config_path) if config_path else Path("config/default.yaml")
    ctx.ensure_object(dict)
    ctx.obj["config"] = load_config(path if path.exists() else None)
    ctx.obj["config_dir"] = path.parent if path.exists() else Path("config")
    ctx.obj["data_dir"] = Path("data")


@main.command()
@click.option("--scenario", required=True, help="Scenario key (e.g., longlive_t2v_5m)")
@click.option("--orchestrator", default=None, help="Target orchestrator ID")
@click.option("--scope-url", default="http://localhost:8001", help="Scope instance URL")
@click.pass_context
def run(ctx: click.Context, scenario: str, orchestrator: str | None, scope_url: str):
    """Execute a single test run."""
    from .executor import Executor
    from .scenarios import expand_scenario_matrix, load_prompt_pool

    config = ctx.obj["config"]
    config_dir = ctx.obj["config_dir"]
    data_dir = ctx.obj["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    # Find scenario in expanded matrix
    all_scenarios = expand_scenario_matrix(
        config.scenario_defs, config_dir / "graphs"
    )
    scenario_map = {s.name: s for s in all_scenarios}

    if scenario not in scenario_map:
        available = ", ".join(sorted(scenario_map.keys()))
        click.echo(f"Unknown scenario: {scenario}")
        click.echo(f"Available: {available}")
        raise SystemExit(1)

    sc = scenario_map[scenario]

    try:
        prompts = load_prompt_pool(sc.prompts_pool, config_dir / "prompts")
    except FileNotFoundError:
        prompts = ["a scenic landscape"]

    app_id = os.environ.get("SCOPE_CLOUD_APP_ID", "")
    api_key = os.environ.get("SCOPE_CLOUD_API_KEY")
    oid = orchestrator or "manual"

    executor = Executor(config, data_dir=data_dir)
    result = asyncio.run(
        executor.run(scope_url, oid, sc, prompts, app_id, api_key)
    )

    if result.passed:
        click.echo(f"PASS: {sc.name} ({result.timings.total_s:.1f}s)")
        if result.timings.first_frame_s:
            click.echo(f"  First frame: {result.timings.first_frame_s:.1f}s")
        if result.avg_fps:
            click.echo(f"  Avg FPS: {result.avg_fps:.1f}")
    else:
        click.echo(f"FAIL: {sc.name} [{result.error_category}] {result.error_message}")
    raise SystemExit(0 if result.passed else 1)


@main.command()
@click.pass_context
def schedule(ctx: click.Context):
    """Start the scheduler daemon."""
    from .scheduler import run_scheduler

    config = ctx.obj["config"]
    config_dir = ctx.obj["config_dir"]
    data_dir = ctx.obj["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    click.echo("Starting scheduler daemon...")
    asyncio.run(run_scheduler(config, config_dir, data_dir))


@main.command()
@click.pass_context
def discover(ctx: click.Context):
    """List available orchestrators and their health status."""
    from .discovery import discover_orchestrators

    discovery_url = os.environ.get("LIVEPEER_DISCOVERY_URL", "")
    livepeer_token = os.environ.get("LIVEPEER_TOKEN")

    if not discovery_url:
        click.echo("LIVEPEER_DISCOVERY_URL not set")
        raise SystemExit(1)

    orchestrators = asyncio.run(
        discover_orchestrators(discovery_url, livepeer_token)
    )

    if not orchestrators:
        click.echo("No orchestrators found")
        return

    click.echo(f"Found {len(orchestrators)} orchestrators:")
    for o in orchestrators:
        click.echo(f"  {o.id}  {o.address}  region={o.region or 'unknown'}  status={o.status}")


@main.command()
@click.pass_context
def coverage(ctx: click.Context):
    """Show today's test coverage report."""
    from .coverage import CoverageTracker

    data_dir = ctx.obj["data_dir"]
    tracker = CoverageTracker(data_dir / "coverage.json")
    today = tracker.get_today()

    if not today:
        click.echo("No coverage data for today")
        return

    click.echo("Today's coverage:")
    for oid, entry in sorted(today.items()):
        completed = entry["runs_completed"]
        planned = entry["runs_planned"]
        failures = entry["failures"]
        scenarios = len(entry["scenarios_covered"])
        pct = (completed / planned * 100) if planned > 0 else 0
        click.echo(
            f"  {oid}: {completed}/{planned} runs ({pct:.0f}%), "
            f"{scenarios} scenarios, {failures} failures"
        )

    debt = tracker.get_test_debt()
    total_debt = sum(debt.values())
    click.echo(f"\nTotal remaining runs: {total_debt}")


@main.command()
@click.pass_context
def baselines(ctx: click.Context):
    """Show current baseline metrics."""
    data_dir = ctx.obj["data_dir"]
    baselines_path = data_dir / "baselines.json"

    if not baselines_path.exists():
        click.echo("No baseline data yet")
        return

    with open(baselines_path) as f:
        data = json.load(f)

    if not data:
        click.echo("No baseline data yet")
        return

    click.echo("Current baselines (7-day rolling):")
    for scenario, vals in sorted(data.items()):
        ff = vals.get("first_frame_p50")
        fps = vals.get("steady_fps_p50")
        load = vals.get("pipeline_load_p50")
        n = vals.get("sample_count", 0)
        parts = [f"n={n}"]
        if ff is not None:
            parts.append(f"first_frame_p50={ff:.1f}s")
        if fps is not None:
            parts.append(f"fps_p50={fps:.1f}")
        if load is not None:
            parts.append(f"load_p50={load:.1f}s")
        click.echo(f"  {scenario}: {', '.join(parts)}")


if __name__ == "__main__":
    main()
