"""Progress display for imagery selection operations."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import click
from tqdm import tqdm

if TYPE_CHECKING:
    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult


@dataclass
class SelectionStats:
    """Track selection statistics."""

    successful: int = 0
    skipped: int = 0
    failed: int = 0
    already_has: int = 0
    no_scenes: int = 0


@dataclass
class ImageryProgressBar:
    """Progress bar for imagery selection with status line updates.

    Provides a unified progress display for both select-images and create-dataset
    commands, with informative status updates below the progress bar.
    """

    total: int
    leave: bool = True
    verbose: bool = False
    stats: SelectionStats = field(default_factory=SelectionStats)
    _pbar: tqdm | None = field(default=None, repr=False)
    _current_chip: str | None = field(default=None, repr=False)
    _last_result: "SceneSelectionResult | None" = field(default=None, repr=False)

    def __enter__(self) -> "ImageryProgressBar":
        self._pbar = tqdm(
            total=self.total,
            desc="Selecting imagery",
            unit="chip",
            leave=self.leave,
        )
        self._update_postfix()
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb) -> None:
        if self._pbar:
            self._pbar.close()

    def _update_postfix(self) -> None:
        """Update the progress bar postfix with current stats."""
        if not self._pbar:
            return

        status = {
            "ok": self.stats.successful,
            "skip": self.stats.skipped,
            "fail": self.stats.failed,
        }

        if self._last_result and self._last_result.success:
            p_cc = (
                self._last_result.planting_scene.cloud_cover
                if self._last_result.planting_scene
                else 0
            )
            h_cc = (
                self._last_result.harvest_scene.cloud_cover
                if self._last_result.harvest_scene
                else 0
            )
            status["last"] = f"P:{p_cc:.0f}%/H:{h_cc:.0f}%"

        self._pbar.set_postfix(status)

    def _status(self, message: str, style: str | None = None) -> None:
        """Write a status message below the progress bar.

        Args:
            message: The message to display
            style: Optional color style ('green', 'yellow', 'red', 'cyan', 'dim')
        """
        if not self._pbar:
            return

        if style:
            styled = click.style(message, fg=style)
            self._pbar.write(styled)
        else:
            self._pbar.write(message)

    def start_chip(self, chip_id: str) -> None:
        """Called when starting to process a chip."""
        self._current_chip = chip_id

    def on_progress(self, message: str) -> None:
        """Progress callback for scene selection - shows detailed info.

        This is passed to select_scenes_for_chip() as the on_progress callback.
        It intelligently formats messages based on their content.
        """
        if not self._pbar:
            return

        # Format different message types with appropriate styling
        if message.startswith("Expansion"):
            # Buffer expansion - highlight in cyan
            self._status(f"  ↻ {message}", "cyan")
        elif message.startswith("Searching for"):
            # Search start - dim
            if self.verbose:
                self._status(f"  ○ {message}", "bright_black")
        elif message.startswith("STAC Query:") or message.startswith("  "):
            # Detailed query info - only show in verbose mode
            if self.verbose:
                self._status(f"    {message}", "bright_black")
        elif message.startswith("Found"):
            # Candidate count - show count info
            if self.verbose:
                self._status(f"  ○ {message}", "bright_black")
        elif message.startswith("Selected planting") or message.startswith("Selected harvest"):
            # Success - green
            self._status(f"  ✓ {message}", "green")
        elif "Skipping" in message or "exceeds" in message:
            # Skipped due to cloud cover - dim
            if self.verbose:
                self._status(f"  ○ {message}", "bright_black")
        elif message.startswith("Both seasons meet"):
            # Final success message
            self._status(f"  ✓ {message}", "green")
        elif message.startswith("Crop calendar:"):
            # Crop calendar info - dim
            if self.verbose:
                self._status(f"  ○ {message}", "bright_black")
        elif self.verbose:
            # Other messages only in verbose
            self._status(f"  ○ {message}", "bright_black")

    def mark_success(self, result: "SceneSelectionResult") -> None:
        """Mark current chip as successfully processed."""
        self.stats.successful += 1
        self._last_result = result
        self._update_postfix()
        if self._pbar:
            self._pbar.update(1)

    def mark_skipped(self, reason: str, was_existing: bool = False) -> None:
        """Mark current chip as skipped."""
        self.stats.skipped += 1
        if was_existing:
            self.stats.already_has += 1
        elif "No cloud-free" in reason or "no scenes" in reason.lower():
            self.stats.no_scenes += 1
            chip = self._current_chip or "chip"
            self._status(f"  ✗ {chip}: {reason}", "yellow")
        self._update_postfix()
        if self._pbar:
            self._pbar.update(1)

    def mark_failed(self, error: str) -> None:
        """Mark current chip as failed with error."""
        self.stats.failed += 1
        chip = self._current_chip or "chip"
        self._status(f"  ✗ {chip}: {error}", "red")
        self._update_postfix()
        if self._pbar:
            self._pbar.update(1)

    def get_stats_dict(self) -> dict:
        """Return stats as a dictionary for backward compatibility."""
        return {
            "successful": self.stats.successful,
            "skipped": self.stats.skipped,
            "failed": self.stats.failed,
        }
