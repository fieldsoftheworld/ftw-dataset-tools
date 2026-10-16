"""Tests for the logging_config module."""

import logging
import threading

import pytest

from ftw_dataset_tools.api.logging_config import (
    SUCCESS,
    capture_logs,
    get_logger,
    replay,
    success,
)

LOGGER_NAME = "ftw_dataset_tools.tests.logging_config"


class TestGetLogger:
    def test_returns_the_named_logger(self) -> None:
        assert get_logger(LOGGER_NAME) is logging.getLogger(LOGGER_NAME)

    def test_adds_the_capture_filter_once(self) -> None:
        logger = get_logger(LOGGER_NAME)
        get_logger(LOGGER_NAME)

        assert len(logger.filters) == 1


class TestSuccess:
    def test_logs_at_the_success_level(self, caplog: pytest.LogCaptureFixture) -> None:
        success(get_logger(LOGGER_NAME), "done")

        assert [(r.levelno, r.levelname, r.message) for r in caplog.records] == [
            (SUCCESS, "SUCCESS", "done")
        ]

    def test_sits_between_info_and_warning(self) -> None:
        assert logging.INFO < SUCCESS < logging.WARNING


class TestCaptureLogs:
    def test_holds_records_instead_of_emitting_them(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger(LOGGER_NAME)
        held: list[logging.LogRecord] = []

        with capture_logs(held):
            logger.info("first")
            logger.warning("second")

        assert caplog.records == []
        assert [r.getMessage() for r in held] == ["first", "second"]

    def test_emits_again_after_the_block(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger(LOGGER_NAME)
        with capture_logs([]):
            pass

        logger.info("after")

        assert caplog.messages == ["after"]

    def test_stops_capturing_when_the_block_raises(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger(LOGGER_NAME)
        held: list[logging.LogRecord] = []

        with pytest.raises(RuntimeError), capture_logs(held):
            logger.info("inside")
            raise RuntimeError

        logger.info("outside")
        assert [r.getMessage() for r in held] == ["inside"]
        assert caplog.messages == ["outside"]

    def test_only_affects_the_capturing_thread(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger(LOGGER_NAME)
        held: list[logging.LogRecord] = []
        inside = threading.Event()
        logged = threading.Event()

        def worker() -> None:
            with capture_logs(held):
                inside.set()
                logged.wait(5)
                logger.info("worker")

        thread = threading.Thread(target=worker)
        thread.start()
        inside.wait(5)
        logger.info("main")
        logged.set()
        thread.join(5)

        assert caplog.messages == ["main"]
        assert [r.getMessage() for r in held] == ["worker"]


class TestReplay:
    def test_emits_held_records_with_their_levels(self, caplog: pytest.LogCaptureFixture) -> None:
        logger = get_logger(LOGGER_NAME)
        held: list[logging.LogRecord] = []
        with capture_logs(held):
            logger.info("one")
            logger.warning("two")

        replay(held)

        assert [(r.levelname, r.message) for r in caplog.records] == [
            ("INFO", "one"),
            ("WARNING", "two"),
        ]
