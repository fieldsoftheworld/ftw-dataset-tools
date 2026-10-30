"""One-line summaries of the imagery stages, shared by every command that runs them."""

from __future__ import annotations

from typing import Any

from ftw_dataset_tools.api.imagery.selection_workflow import MultiModeSelectionResult

__all__ = ["imagery_summary_line", "selection_summary_lines"]


def imagery_summary_line(label: str, result: Any) -> str:
    """One line counting an imagery stage's successes, skips and failures.

    Failures used to be invisible: selection swallows per-chip exceptions, so a
    run in which every chip failed still printed only "Imagery selected: 0".
    """
    line = f"  {label}: {result.successful} ok, {result.skipped} skipped, {result.failed} failed"
    if result.failed_details:
        first = result.failed_details[0].get("error", "unknown error")
        line += f" (first error: {first})"
    return line


def selection_summary_lines(result: Any) -> list[str]:
    """The selection summary: one line, or one per imagery mode when several ran.

    Chips that could not be read are counted once, not once per mode.
    """
    if not isinstance(result, MultiModeSelectionResult) or len(result.by_mode) == 1:
        return [imagery_summary_line("Imagery selection", result)]
    lines = [
        imagery_summary_line(f"Imagery selection ({mode})", mode_result)
        for mode, mode_result in result.by_mode.items()
    ]
    if result.unreadable:
        first = result.unreadable[0].get("error", "unknown error")
        lines.append(f"  Unreadable chips: {len(result.unreadable)} (first error: {first})")
    return lines
