"""Tests for the CLI log handler and the global --verbose flag."""

import logging

import click
import pytest
from click.testing import CliRunner

from ftw_dataset_tools.api.logging_config import get_logger, success
from ftw_dataset_tools.cli import cli
from ftw_dataset_tools.commands.cli_logging import (
    PACKAGE_LOGGER,
    ClickHandler,
    configure_cli_logging,
    is_verbose,
)

logger = get_logger("ftw_dataset_tools.tests.cli_logging")


@click.command()
def emit() -> None:
    logger.debug("debug line")
    logger.info("info line")
    success(logger, "success line")
    logger.warning("  indented warning")
    logger.error("error line")


class TestClickHandler:
    def test_writes_to_stderr_with_level_prefixes(self) -> None:
        result = CliRunner().invoke(emit)

        assert result.exit_code == 0
        assert result.stdout == ""
        assert result.stderr.splitlines() == [
            "info line",
            "success line",
            "  Warning: indented warning",
            "Error: error line",
        ]

    def test_colours_by_level(self) -> None:
        result = CliRunner().invoke(emit, color=True)

        lines = result.stderr.splitlines()
        assert lines[0] == "info line"
        assert lines[1] == click.style("success line", fg="green")
        assert lines[2] == click.style("  Warning: indented warning", fg="yellow")
        assert lines[3] == click.style("Error: error line", fg="red")

    def test_debug_shown_when_verbose(self) -> None:
        configure_cli_logging(verbose=True)

        result = CliRunner().invoke(emit)

        assert result.stderr.splitlines()[0] == "debug line"


class TestConfigureCliLogging:
    def test_installs_a_single_handler(self) -> None:
        configure_cli_logging()
        configure_cli_logging()

        handlers = logging.getLogger(PACKAGE_LOGGER).handlers
        assert sum(isinstance(h, ClickHandler) for h in handlers) == 1

    @pytest.mark.parametrize(("verbose", "expected"), [(False, False), (True, True)])
    def test_sets_verbosity(self, verbose: bool, expected: bool) -> None:
        configure_cli_logging(verbose=verbose)

        assert is_verbose() is expected


class TestVerboseFlag:
    def test_global_flag_enables_debug(self) -> None:
        result = CliRunner().invoke(cli, ["--verbose", "get-grid", "--help"])

        assert result.exit_code == 0
        assert is_verbose()

    def test_default_is_not_verbose(self) -> None:
        result = CliRunner().invoke(cli, ["get-grid", "--help"])

        assert result.exit_code == 0
        assert not is_verbose()

    @pytest.mark.parametrize(
        "command", ["select-images", "convert-previews", "create-dataset-summary"]
    )
    @pytest.mark.parametrize("flag", ["--verbose", "-v"])
    def test_per_command_verbose_still_enables_debug(self, command: str, flag: str) -> None:
        """The old per-command flag is kept as an alias of ``ftwd -v``."""
        result = CliRunner().invoke(cli, [command, flag, "missing-path"])

        assert "No such option" not in result.output
        assert is_verbose()
