"""One-line summaries of the imagery stages, shared by every command that runs them."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ftw_dataset_tools.api.imagery.selection_workflow import MultiModeSelectionResult


__all__ = ["imagery_summary_line", "preview_redraw_lines", "selection_summary_lines"]


def imagery_summary_line(label: str, result: Any) -> str:
    """One line counting an imagery stage's successes, skips and failures.

    Failures used to be invisible: selection swallows per-chip exceptions, so a
    run in which every chip failed still printed only "Imagery selected: 0".
    """
    line = f"  {label}: {result.successful} ok, {result.skipped} skipped, {result.failed} failed"
    if result.failed_details:
        first = result.failed_details[0].get("error", "unknown error")
        line += f" (first error: {first})"
    redraw = preview_redraw_lines(result)
    if redraw:
        line += "; " + "; ".join(text[0].lower() + text[1:] for text in redraw)
    return line


def preview_redraw_lines(result: Any) -> list[str]:
    """``Previews redrawn: N`` and ``Previews not redrawn: M (see log)``, when above zero.

    Shared by every command that downloads or previews imagery, so they word it alike.
    """
    redrawn = getattr(result, "previews_redrawn", 0)
    not_redrawn = getattr(result, "previews_redraw_failed", 0)
    lines = [f"Previews redrawn: {redrawn}"] if redrawn else []
    if not_redrawn:
        lines.append(f"Previews not redrawn: {not_redrawn} (see log)")
    return lines


def selection_summary_lines(result: MultiModeSelectionResult) -> list[str]:
    """The selection summary: one line, or one per imagery mode when several ran.

    Chips that could not be read are counted once, not once per mode.
    """
    if len(result.by_mode) == 1:
        return [imagery_summary_line("Imagery selection", result)]
    lines = [
        imagery_summary_line(f"Imagery selection ({mode})", mode_result)
        for mode, mode_result in result.by_mode.items()
    ]
    if result.unreadable:
        first = result.unreadable[0].get("error", "unknown error")
        lines.append(f"  Unreadable chips: {len(result.unreadable)} (first error: {first})")
    return lines
