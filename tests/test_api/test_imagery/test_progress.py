"""Tests for the imagery selection progress bar."""

from __future__ import annotations

import logging

from tqdm import tqdm

from ftw_dataset_tools.api.imagery.progress import (
    BAR_FORMAT,
    STATUS_ICON,
    ImageryProgressBar,
    SelectionStats,
    format_counters,
)

LOGGER_NAME = "ftw_dataset_tools.tests.progress"


def _render(stats: SelectionStats) -> str:
    """Render the real bar format the way tqdm would, without a terminal."""
    return tqdm.format_meter(
        n=19,
        total=775,
        elapsed=1.0,
        ncols=120,
        prefix="31UFR9620",
        unit="chip",
        postfix=format_counters(stats),
        bar_format=BAR_FORMAT,
    )


class TestFormatCounters:
    """The counters name themselves; all three always show."""

    def test_all_counters_are_labelled(self) -> None:
        assert format_counters(SelectionStats(successful=19)) == "ok=19 skip=0 fail=0"

    def test_reports_skips_and_failures(self) -> None:
        stats = SelectionStats(successful=1, skipped=2, failed=3)

        assert format_counters(stats) == "ok=1 skip=2 fail=3"


class TestBarFormat:
    """Regression: the bar rendered `ok=, 0 fail=19` -- tqdm prefixes `{postfix}`."""

    def test_counters_render_once_and_in_order(self) -> None:
        line = _render(SelectionStats(successful=19))

        assert "ok=19 skip=0 fail=0" in line
        assert "ok=," not in line

    def test_failures_are_visible_next_to_the_successes(self) -> None:
        line = _render(SelectionStats(successful=0, failed=19))

        assert "ok=0 skip=0 fail=19" in line


class TestReportFailures:
    """Failures are surfaced instead of vanishing into the swallowed exception."""

    def test_writes_the_first_three_errors(self, capsys) -> None:
        bar = ImageryProgressBar(total=5, leave=False)
        details = [{"chip": f"chip_{i}", "error": f"boom {i}"} for i in range(5)]

        with bar:
            bar.report_failures(details)

        out = capsys.readouterr().out
        assert "chip_0: boom 0" in out
        assert "chip_2: boom 2" in out
        assert "chip_3" not in out
        assert "2 more" in out

    def test_says_nothing_without_failures(self, capsys) -> None:
        bar = ImageryProgressBar(total=5, leave=False)

        with bar:
            bar.report_failures([])

        assert capsys.readouterr().out == ""


class TestShow:
    """Status-tagged records drive the status line; everything else is logged."""

    @staticmethod
    def _record(msg: str, level: int = logging.INFO, icon: str | None = None) -> logging.LogRecord:
        record = logging.LogRecord(LOGGER_NAME, level, __file__, 0, msg, None, None)
        if icon is not None:
            setattr(record, STATUS_ICON, icon)
        return record

    def test_tagged_record_goes_to_the_status_line(self, caplog) -> None:
        with ImageryProgressBar(total=1, leave=False) as bar:
            bar.show([self._record("Searching for planting scene", icon="○")])
            description = bar._pbar.desc

        assert description.startswith("○ Searching for planting scene")
        assert caplog.records == []

    def test_untagged_record_is_logged_with_its_level(self, caplog) -> None:
        with ImageryProgressBar(total=1, leave=False) as bar:
            bar.show([self._record("check failed", level=logging.WARNING)])
            description = bar._pbar.desc

        assert description == ""
        assert [(r.levelname, r.message) for r in caplog.records] == [("WARNING", "check failed")]

    def test_scene_selection_status_lines_carry_an_icon(self, caplog) -> None:
        from ftw_dataset_tools.api.imagery import scene_selection

        scene_selection._status("✗", "Skipping 9-28: 15.2% cloud")

        assert getattr(caplog.records[0], STATUS_ICON) == "✗"
