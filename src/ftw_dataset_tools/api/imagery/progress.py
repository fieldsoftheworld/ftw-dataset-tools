"""Progress display for imagery selection operations."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

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
    """Progress bar for imagery selection with single-line status updates.

    Shows a compact progress bar with current chip status updating in-place.
    """

    total: int
    leave: bool = True
    verbose: bool = False
    stats: SelectionStats = field(default_factory=SelectionStats)
    _pbar: tqdm | None = field(default=None, repr=False)
    _current_chip: str | None = field(default=None, repr=False)
    _last_result: "SceneSelectionResult | None" = field(default=None, repr=False)

    def __enter__(self) -> "ImageryProgressBar":
        # Compact bar format: progress bar | count | stats | description (at end so width changes don't matter)
        bar_format = "|{bar:25}| {n}/{total} ok={postfix} {desc}"
        self._pbar = tqdm(
            total=self.total,
            desc="",
            unit="chip",
            leave=self.leave,
            bar_format=bar_format,
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

        # Build stats string: "5" or "5 skip=2" or "5 skip=2 fail=1"
        parts = [str(self.stats.successful)]
        if self.stats.skipped > 0:
            parts.append(f"skip={self.stats.skipped}")
        if self.stats.failed > 0:
            parts.append(f"fail={self.stats.failed}")

        self._pbar.set_postfix_str(" ".join(parts))

    def _status(self, message: str) -> None:
        """Update the status message in-place."""
        if not self._pbar:
            return
        # Truncate to keep display compact
        display_msg = message[:50] if len(message) > 50 else message
        self._pbar.set_description(display_msg)

    def start_chip(self, chip_id: str) -> None:
        """Called when starting to process a chip."""
        self._current_chip = chip_id
        # Show short chip ID (last part after 'ftw-')
        short_id = chip_id.replace("ftw-", "") if chip_id.startswith("ftw-") else chip_id
        self._status(short_id)

    def on_progress(self, message: str) -> None:
        """Progress callback for scene selection.

        All messages update in-place on the description line with original formatting.
        """
        if not self._pbar:
            return

        # Format messages with icons like the original
        if message.startswith("Expansion"):
            # "Expansion 1: planting buffer now 28 days"
            self._status(f"↻ {message}")
        elif message.startswith("Searching for"):
            # "Searching for planting scene around 2024-05-01..." -> "○ Searching planting..."
            season = "planting" if "planting" in message else "harvest"
            self._status(f"○ Searching {season}...")
        elif message.startswith("Selected"):
            # Keep the full message: "✓ Selected planting scene: S2B_T54TWN... (0.0% cloud)"
            self._status(f"✓ {message}")
        elif message.startswith("Found"):
            # "Found 5 planting scene candidates"
            self._status(f"○ {message}")
        elif message.startswith("Both seasons"):
            self._status(f"✓ {message}")
        elif "Skipping" in message:
            # "  Skipping 9-28: 15.2% cloud" -> "✗ Skipping 9-28: 15.2% cloud"
            self._status(f"✗ {message.strip()}")
        else:
            # Other messages
            self._status(message)

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
        self._update_postfix()
        if self._pbar:
            self._pbar.update(1)

    def mark_failed(self, error: str) -> None:  # noqa: ARG002
        """Mark current chip as failed with error."""
        self.stats.failed += 1
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
