"""Redraw chip overlays that are older than their masks, from images already on disk.

A resumed imagery run skips the chips it has already downloaded or previewed, so a
mask rebuilt since (at a new resolution, say) would otherwise keep showing the old
overlay. This redraws those overlays from the preview slot's image on disk; nothing is
downloaded. Shared by the download and preview workflows.

Staleness is read from file modification times only, so the check is cheap on large
catalogs and only stale chips have their item read. The rule:

- No reference mask: nothing to overlay, never stale.
- An overlay older than the mask is stale.
- A missing overlay is stale when a slot preview is on disk and the mask is newer
  than it. When the mask is older, the overlay was already attempted from that image
  (a download always draws it), so it is not retried.
- A failed draw bumps the slot preview's mtime, so the same mask is not retried (or
  counted) on every run; rebuilding the mask makes it stale again.
- An equal mtime is not stale. Coarse timestamps tie often, and in a normal run the
  overlay or image is always written after the mask (the masks stage runs first), so
  a tie means "drawn right after", not "mask rebuilt since". Counting ties as stale
  would redraw fresh chips on every run on such filesystems.
- When a chip's item cannot be written, its overlay (else its slot preview) is set
  just older than the mask, so the next run retries it rather than leave the item's
  thumbnail describing a different file.

Copies that do not preserve mtimes (``cp`` without ``-p``, some syncs) can therefore
cause extra redraws, or missed ones when a copied mask ends up older than its overlay.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pystac

from ftw_dataset_tools.api.imagery.chip_preview import (
    PREVIEW_SLOT_FIELD,
    best_slot_preview,
    existing_preview,
    preview_slot,
    redraw_chip_preview,
    refresh_chip_preview,
)
from ftw_dataset_tools.api.imagery.parallel import DEFAULT_WORKERS, run_in_parallel
from ftw_dataset_tools.api.imagery.thumbnails import PREVIEW_SUFFIX, REFERENCE_MASK_SUFFIX
from ftw_dataset_tools.api.logging_config import get_logger
from ftw_dataset_tools.api.stac_items import write_item

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.parallel import ParallelOutcome

__all__ = ["OverlayRedrawResult", "overlay_is_stale", "redraw_stale_overlays"]

logger = get_logger(__name__)


@dataclass
class OverlayRedrawResult:
    """Counts for one pass: overlays drawn, and stale ones that could not be drawn."""

    redrawn: int = 0
    failed: int = 0


@dataclass
class _RedrawTask:
    item: pystac.Item
    item_path: Path
    #: The slot to draw from; None when its image is gone and the best one left is used.
    slot: str | None
    checksums: bool


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def overlay_is_stale(chip_dir: Path, chip_id: str) -> bool:
    """Whether the chip's overlay needs drawing again (see the module docstring)."""
    mask_mtime = _mtime(chip_dir / f"{chip_id}{REFERENCE_MASK_SUFFIX}")
    if mask_mtime is None:
        return False
    overlay_mtime = _mtime(chip_dir / f"{chip_id}_overlay{PREVIEW_SUFFIX}")
    # Strictly older: a tie is not stale (see the module docstring).
    if overlay_mtime is not None:
        return overlay_mtime < mask_mtime
    best = best_slot_preview(chip_dir, chip_id)
    if best is None:
        return False
    base_mtime = _mtime(chip_dir / f"{chip_id}_{best}_image_s2{PREVIEW_SUFFIX}")
    return base_mtime is not None and base_mtime < mask_mtime


def _plan(item_path: Path) -> _RedrawTask:
    """Read a stale chip's item and pick the slot to draw from (calling thread)."""
    chip_dir = item_path.parent
    item = pystac.Item.from_file(str(item_path))
    slot = preview_slot(item, chip_dir)
    if slot is not None and existing_preview(chip_dir, f"{item.id}_{slot}_image_s2") is None:
        slot = None  # its image is gone: redraw from the best slot left
    old = item.assets.get("thumbnail")
    checksums = old is not None and "file:checksum" in old.extra_fields
    return _RedrawTask(item=item, item_path=item_path, slot=slot, checksums=checksums)


def _draw(task: _RedrawTask) -> bool:
    """Draw the overlay and set the thumbnail on the task's own item (worker thread).

    Returns whether the overlay is now on disk.
    """
    chip_dir = task.item_path.parent
    if task.slot is None:
        redraw_chip_preview(task.item, chip_dir, checksums=task.checksums)
    else:
        refresh_chip_preview(task.item, chip_dir, task.slot, redraw=True, checksums=task.checksums)
    return (chip_dir / f"{task.item.id}_overlay{PREVIEW_SUFFIX}").exists()


def _mark_attempted(task: _RedrawTask) -> None:
    """Make the image the overlay was tried from newer than the mask (see module doc)."""
    thumbnail = task.item.assets.get("thumbnail")
    slot = thumbnail.extra_fields.get(PREVIEW_SLOT_FIELD) if thumbnail else None
    base = (
        existing_preview(task.item_path.parent, f"{task.item.id}_{slot}_image_s2") if slot else None
    )
    if base is not None:
        os.utime(base)


def _fall_back_to_plain(task: _RedrawTask) -> None:
    """After a draw that raised: the plain slot preview as thumbnail, or none.

    The draw may have removed the old overlay already (or left a partial one), so the
    item must not keep pointing at it.
    """
    chip_dir = task.item_path.parent
    (chip_dir / f"{task.item.id}_overlay{PREVIEW_SUFFIX}").unlink(missing_ok=True)
    task.item.assets.pop("thumbnail", None)
    slot = task.slot or best_slot_preview(chip_dir, task.item.id)
    if slot is not None:
        refresh_chip_preview(task.item, chip_dir, slot, checksums=task.checksums)


def _retry_target(chip_dir: Path, chip_id: str) -> Path | None:
    """The file whose age makes the chip stale: its overlay, else its best slot preview."""
    overlay = chip_dir / f"{chip_id}_overlay{PREVIEW_SUFFIX}"
    if overlay.exists():
        return overlay
    best = best_slot_preview(chip_dir, chip_id)
    return existing_preview(chip_dir, f"{chip_id}_{best}_image_s2") if best else None


def _mark_for_retry(task: _RedrawTask) -> None:
    """Make the chip stale again after its item could not be written (best effort)."""
    chip_dir, chip_id = task.item_path.parent, task.item_path.stem
    mask_mtime = _mtime(chip_dir / f"{chip_id}{REFERENCE_MASK_SUFFIX}")
    target = _retry_target(chip_dir, chip_id)
    if mask_mtime is None or target is None:
        return
    try:
        os.utime(target, (mask_mtime - 1, mask_mtime - 1))
    except OSError as err:
        logger.warning(f"{chip_id}: could not mark the chip preview for a retry: {err}")


def _settle(outcome: ParallelOutcome[_RedrawTask, bool]) -> bool | None:
    """Write the chip's item after its draw (calling thread).

    Returns True when the overlay was drawn, False when it could not be, and None when
    no preview image was left to draw it from.
    """
    task = outcome.task
    if outcome.error is not None:
        logger.warning(f"{task.item.id}: could not draw the chip overlay: {outcome.error}")
        _fall_back_to_plain(task)
    drawn = outcome.error is None and bool(outcome.value)
    write_item(task.item, task.item_path)
    if drawn:
        return True
    _mark_attempted(task)
    return False if task.item.assets.get("thumbnail") is not None else None


def _apply(outcome: ParallelOutcome[_RedrawTask, bool], result: OverlayRedrawResult) -> None:
    """Settle one chip and count it once (calling thread)."""
    task = outcome.task
    try:
        drawn = _settle(outcome)
    except Exception as err:  # any one chip's failure is counted, never the run's
        result.failed += 1
        logger.warning(f"{task.item_path.stem}: could not redraw the chip preview: {err}")
        _mark_for_retry(task)
        return
    if drawn:
        result.redrawn += 1
    elif drawn is False:
        result.failed += 1
        if outcome.error is None:
            logger.warning(
                f"{task.item.id}: could not draw the chip overlay; kept the plain preview"
            )


def _stale_tasks(item_paths: Iterable[Path], result: OverlayRedrawResult) -> list[_RedrawTask]:
    tasks = []
    for item_path in item_paths:
        if not overlay_is_stale(item_path.parent, item_path.stem):
            continue
        try:
            tasks.append(_plan(item_path))
        except Exception as err:  # a malformed item is counted, not fatal to the run
            result.failed += 1
            logger.warning(f"{item_path.stem}: could not redraw the chip preview: {err}")
    return tasks


def redraw_stale_overlays(
    item_paths: Iterable[Path], *, workers: int = DEFAULT_WORKERS
) -> OverlayRedrawResult:
    """Redraw the stale overlays among these chip items, ``workers`` at a time.

    Drawing runs on a thread pool; each item is written on the calling thread. A chip
    whose overlay cannot be drawn keeps its plain preview as the thumbnail and is
    logged and counted in ``failed``, without failing the run.

    Args:
        item_paths: Chip item JSON files, ``<chip_dir>/<chip_id>.json``.
        workers: Number of overlays to draw concurrently.
    """
    result = OverlayRedrawResult()
    tasks = _stale_tasks(item_paths, result)
    run_in_parallel(tasks, _draw, lambda outcome: _apply(outcome, result), workers=workers)
    if result.redrawn or result.failed:
        logger.info(
            f"Redrew {result.redrawn} chip preview(s) older than their masks"
            + (f"; {result.failed} could not be drawn" if result.failed else "")
        )
    return result
