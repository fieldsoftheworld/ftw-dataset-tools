"""Tests for the imagery selection progress bar."""

from __future__ import annotations

from tqdm import tqdm

from ftw_dataset_tools.api.imagery.progress import (
    BAR_FORMAT,
    ImageryProgressBar,
    SelectionStats,
    format_counters,
)


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
