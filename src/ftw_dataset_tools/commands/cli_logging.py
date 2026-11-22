"""Terminal output for ftw_dataset_tools log records."""

from __future__ import annotations

import logging

import click
from tqdm import tqdm

from ftw_dataset_tools.api.logging_config import SUCCESS

PACKAGE_LOGGER = "ftw_dataset_tools"

_STYLES: dict[int, dict] = {
    logging.DEBUG: {"dim": True},
    SUCCESS: {"fg": "green"},
    logging.WARNING: {"fg": "yellow"},
    logging.ERROR: {"fg": "red"},
    logging.CRITICAL: {"fg": "red", "bold": True},
}
_PREFIXES = {logging.WARNING: "Warning: ", logging.ERROR: "Error: ", logging.CRITICAL: "Error: "}


class ClickHandler(logging.Handler):
    """Write records to stderr, coloured by level, without breaking tqdm bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            body = msg.lstrip(" ")
            msg = msg[: len(msg) - len(body)] + _PREFIXES.get(record.levelno, "") + body
            styles = _STYLES.get(record.levelno)
            with tqdm.external_write_mode():
                click.echo(click.style(msg, **styles) if styles else msg, err=True)
        except Exception:
            self.handleError(record)


def configure_cli_logging(verbose: bool = False) -> None:
    """Route package logs to the terminal; DEBUG only when ``verbose``."""
    logger = logging.getLogger(PACKAGE_LOGGER)
    for handler in [h for h in logger.handlers if isinstance(h, ClickHandler)]:
        logger.removeHandler(handler)
    logger.addHandler(ClickHandler())
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)


def is_verbose() -> bool:
    """Whether ``ftwd -v`` was given."""
    return logging.getLogger(PACKAGE_LOGGER).isEnabledFor(logging.DEBUG)
