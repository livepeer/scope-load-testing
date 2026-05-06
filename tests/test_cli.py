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
