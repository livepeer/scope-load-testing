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
@click.option("--scenario", required=True, help="Scenario key (e.g., longlive_t2v_5m)")
@click.option("--orchestrator", default=None, help="Target orchestrator ID")
@click.option("--scope-url", default="http://localhost:8001", help="Scope instance URL")
@click.pass_context
def run(ctx: click.Context, scenario: str, orchestrator: str | None, scope_url: str):
    """Execute a single test run."""
    click.echo(f"Running scenario: {scenario}")


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
