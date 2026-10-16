"""Logging for ftw_dataset_tools: a SUCCESS level and per-thread log capture."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from contextvars import ContextVar
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

SUCCESS = 25
logging.addLevelName(SUCCESS, "SUCCESS")

_capture: ContextVar[list[logging.LogRecord] | None] = ContextVar("ftwd_capture", default=None)


def _divert_captured(record: logging.LogRecord) -> bool:
    buffer = _capture.get()
    if buffer is None:
        return True
    buffer.append(record)
    return False


def get_logger(name: str) -> logging.Logger:
    """Return the logger for an ftw_dataset_tools module, honouring capture_logs()."""
    logger = logging.getLogger(name)
    if _divert_captured not in logger.filters:
        logger.addFilter(_divert_captured)
    return logger


def success(logger: logging.Logger, msg: str, *args: Any, **kwargs: Any) -> None:
    logger.log(SUCCESS, msg, *args, **kwargs)


@contextmanager
def capture_logs(into: list[logging.LogRecord]) -> Iterator[None]:
    """Hold this thread's records in ``into`` instead of emitting them.

    Lets parallel workers keep each task's output together for replay().
    """
    token = _capture.set(into)
    try:
        yield
    finally:
        _capture.reset(token)


def replay(records: Iterable[logging.LogRecord]) -> None:
    """Emit records held by capture_logs() through the normal handlers."""
    for record in records:
        logging.getLogger(record.name).handle(record)
