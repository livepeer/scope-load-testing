"""CLI entrypoint for the load testing harness."""

import asyncio
import json
import os
import sys
from pathlib import Path

import click

from .config import load_config

CONTEXT_SETTINGS = {"help_option_names": ["-h", "--help"]}


@click.group(context_settings=CONTEXT_SETTINGS)
@click.option("--config", "config_path", type=click.Path(exists=True, dir_okay=False), default=None)
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
@click.option("--sdk-url", envvar="SDK_URL", default="https://sdk.daydream.monster", help="SDK service URL")
@click.option("--api-key", envvar="DAYDREAM_API_KEY", required=True, help="Daydream API key")
@click.pass_context
def run(ctx: click.Context, scenario: str, sdk_url: str, api_key: str):
    """Execute a single test run via the Daydream SDK."""
    from .datasets import select_prompts
    from .scenarios import expand_scenario_matrix
    from .sdk_executor import SDKExecutor

    config = ctx.obj["config"]
    config_dir = ctx.obj["config_dir"]

    all_scenarios = expand_scenario_matrix(config.scenario_defs, config_dir / "graphs")
    scenario_map = {s.name: s for s in all_scenarios}

    if scenario not in scenario_map:
        click.echo(f"Unknown scenario: {scenario}")
        click.echo(f"Available: {', '.join(sorted(scenario_map.keys()))}")
        raise SystemExit(1)

    sc = scenario_map[scenario]

    pools = sc.prompts_pools or [sc.prompts_pool]
    pool_name, prompts = select_prompts(pools, config_dir / "prompts")
    click.echo(f"Prompt pool: {pool_name} ({len(prompts)} prompts, shuffled)")

    executor = SDKExecutor(config)
    result = asyncio.run(executor.run(sdk_url, api_key, sc, prompts))

    # Push Prometheus metrics if push gateway configured
    push_url = os.environ.get("PUSHGATEWAY_URL")
    if push_url:
        from .metrics import MetricsCollector
        metrics = MetricsCollector(push_url=push_url)
        metrics.record_run(result)
        metrics.push()
        click.echo(f"  Metrics pushed to {push_url}")

    # Report to Daydream /v1/metrics (network_events)
    metrics_url = os.environ.get("METRICS_URL", "https://api.daydream.monster/v1/metrics")
    metrics_api_key = os.environ.get("METRICS_API_KEY") or api_key
    from .metrics_reporter import MetricsReporter, build_run_events
    reporter = MetricsReporter(api_key=metrics_api_key, metrics_url=metrics_url)
    reporter.enqueue_many(build_run_events(result, prompt_pool=pool_name))
    accepted = asyncio.run(reporter.flush())
    if accepted:
        click.echo(f"  Events reported to {metrics_url} ({accepted} accepted)")

    if result.passed:
        click.echo(f"PASS: {sc.name} ({result.timings.total_s:.1f}s)")
        if result.timings.connect_s:
            click.echo(f"  Connect: {result.timings.connect_s:.1f}s {'(cold)' if result.cold_start else '(warm)'}")
        if result.timings.first_frame_s:
            click.echo(f"  First frame: {result.timings.first_frame_s:.1f}s")
        click.echo(f"  Frames validated: {result.frames_validated}")
    else:
        click.echo(f"FAIL: {sc.name} [{result.error_category}] {result.error_message}")
    raise SystemExit(0 if result.passed else 1)


@main.command()
@click.pass_context
def scenarios(ctx: click.Context):
    """List all available test scenarios."""
    from .scenarios import expand_scenario_matrix

    config = ctx.obj["config"]
    config_dir = ctx.obj["config_dir"]

    all_scenarios = expand_scenario_matrix(config.scenario_defs, config_dir / "graphs")
    click.echo(f"{len(all_scenarios)} scenarios:")
    for s in all_scenarios:
        click.echo(f"  {s.name:40s} {s.pipeline:20s} {s.mode:5s} {s.duration_mins:3d}m {s.duration_class}")


@main.command()
@click.option("--sdk-url", envvar="SDK_URL", default="https://sdk.daydream.monster")
@click.option("--api-key", envvar="DAYDREAM_API_KEY", required=True)
@click.pass_context
def schedule(ctx: click.Context, sdk_url: str, api_key: str):
    """Start the scheduler daemon."""
    from .scheduler import run_scheduler

    config = ctx.obj["config"]
    config_dir = ctx.obj["config_dir"]
    data_dir = ctx.obj["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    # Inject SDK vars into env for the scheduler to pick up
    os.environ.setdefault("SDK_URL", sdk_url)
    os.environ.setdefault("DAYDREAM_API_KEY", api_key)

    click.echo(f"Starting scheduler (sdk={sdk_url})...")
    asyncio.run(run_scheduler(config, config_dir, data_dir))


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
        n_scenarios = len(entry["scenarios_covered"])
        pct = (completed / planned * 100) if planned > 0 else 0
        click.echo(f"  {oid}: {completed}/{planned} runs ({pct:.0f}%), {n_scenarios} scenarios, {failures} failures")

    debt = tracker.get_test_debt()
    click.echo(f"\nTotal remaining runs: {sum(debt.values())}")


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
    for sc, vals in sorted(data.items()):
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
        click.echo(f"  {sc}: {', '.join(parts)}")


@main.command()
@click.pass_context
def datasets(ctx: click.Context):
    """Show dataset summary and health check."""
    from .datasets import get_dataset_summary

    summary = get_dataset_summary()

    click.echo("Prompt pools:")
    for pool in summary["prompt_pools"]:
        status = "OK" if pool["count"] >= 20 else f"LOW (need {20 - pool['count']} more)"
        click.echo(f"  {pool['name']:15s} {pool['count']:3d} prompts  {status}")

    click.echo(f"\nTotal: {summary['total_prompts']} prompts, "
               f"{summary['reference_images']} images, "
               f"{summary['video_clips']} clips")

    if summary["reference_images"] == 0:
        click.echo("\nTip: Use the enrich-datasets skill to generate reference images via storyboard MCP")
    if summary["video_clips"] == 0:
        click.echo("Tip: Use the enrich-datasets skill to generate video clips via storyboard MCP")


if __name__ == "__main__":
    main()
