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


def _apply(outcome: ParallelOutcome[_RedrawTask, bool], result: OverlayRedrawResult) -> None:
    """Write the chip's item and count the outcome (calling thread)."""
    task = outcome.task
    if outcome.error is not None:
        result.failed += 1
        logger.warning(f"{task.item.id}: could not redraw the chip preview: {outcome.error}")
        return
    try:
        write_item(task.item, task.item_path)
        if not outcome.value:
            _mark_attempted(task)
    except (OSError, ValueError, pystac.STACError) as err:
        result.failed += 1
        logger.warning(f"{task.item.id}: could not redraw the chip preview: {err}")
        return
    if outcome.value:
        result.redrawn += 1
    elif task.item.assets.get("thumbnail") is not None:
        result.failed += 1
        logger.warning(f"{task.item.id}: could not draw the chip overlay; kept the plain preview")


def _stale_tasks(item_paths: Iterable[Path], result: OverlayRedrawResult) -> list[_RedrawTask]:
    tasks = []
    for item_path in item_paths:
        if not overlay_is_stale(item_path.parent, item_path.stem):
            continue
        try:
            tasks.append(_plan(item_path))
        except (OSError, ValueError, pystac.STACError) as err:
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
    logger.info(
        f"Redrew {result.redrawn} chip preview(s) older than their masks"
        + (f"; {result.failed} could not be drawn" if result.failed else "")
    )
    return result
