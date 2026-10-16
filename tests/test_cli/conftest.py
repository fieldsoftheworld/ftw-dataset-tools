"""Fixtures shared by the CLI tests."""

import pytest


@pytest.fixture(autouse=True)
def cli_logging() -> None:
    """Route package logs to the terminal as ``ftwd`` does, since tests invoke commands directly."""
    from ftw_dataset_tools.commands.cli_logging import configure_cli_logging

    configure_cli_logging()
