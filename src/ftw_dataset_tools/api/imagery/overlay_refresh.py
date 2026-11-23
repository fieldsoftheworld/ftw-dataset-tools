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
- A failed draw bumps the slot previews' mtimes, so the same mask is not retried (or
  counted) on every run; rebuilding the mask makes it stale again.
- An equal mtime is not stale. Coarse timestamps tie often, and in a normal run the
  overlay or image is always written after the mask (the masks stage runs first), so
  a tie means "drawn right after", not "mask rebuilt since". Counting ties as stale
  would redraw fresh chips on every run on such filesystems.
- A mask dated in the future (clock skew) would otherwise stay newer than anything
  drawn now, so what this module writes or marks is dated no earlier than the mask.
  The cost: a rebuild within that skew of the stamp does not read as newer, so it is
  missed for as long as the clocks disagree.

The on-disk state only changes once the item is written: the overlay is drawn to a
hidden temporary file, the item is written, and only then is the file moved into
place (or the old overlay removed). An error, or an interrupt between those steps,
leaves the chip stale, so the next run repairs it. The item write itself is not
atomic: a process killed during it can leave the item JSON truncated. A temporary
file left by an interrupt is removed the next time its chip goes through this pass
(every chip in the download workflow; the chips skipped as already rendered in the
preview workflow).

Copies that do not preserve mtimes (``cp`` without ``-p``, some syncs) can therefore
cause extra redraws, or missed ones when a copied mask ends up older than its overlay.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from itertools import islice
from typing import TYPE_CHECKING

import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.imagery.chip_preview import (
    best_slot_preview,
    existing_preview,
    preview_slot,
    set_thumbnail,
)
from ftw_dataset_tools.api.imagery.parallel import DEFAULT_WORKERS, run_in_parallel
from ftw_dataset_tools.api.imagery.thumbnails import (
    PREVIEW_SUFFIX,
    REFERENCE_MASK_SUFFIX,
    generate_overlay_thumbnail,
)
from ftw_dataset_tools.api.logging_config import get_logger
from ftw_dataset_tools.api.stac_items import write_item

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.parallel import ParallelOutcome

__all__ = ["OverlayRedrawResult", "overlay_is_stale", "redraw_stale_overlays"]

logger = get_logger(__name__)

_REDRAWN, _FAILED, _REMOVED = "redrawn", "failed", "removed"

#: Chips checked, planned and redrawn together; bounds the items held in memory.
PLAN_BATCH_SIZE = 2000


@dataclass
class OverlayRedrawResult:
    """Counts for one pass.

    ``redrawn``: overlays drawn. ``failed``: stale chips that could not be redrawn.
    ``removed``: stale overlays removed because no slot preview was left to draw from.
    """

    redrawn: int = 0
    failed: int = 0
    removed: int = 0


@dataclass
class _RedrawTask:
    item: pystac.Item
    item_path: Path
    #: The slot to draw from: the recorded one, else the best left; None when none is.
    slot: str | None
    checksums: bool

    @property
    def chip_dir(self) -> Path:
        return self.item_path.parent

    @property
    def chip_id(self) -> str:
        return self.item_path.stem


def _overlay(chip_dir: Path, chip_id: str) -> Path:
    return chip_dir / f"{chip_id}_overlay{PREVIEW_SUFFIX}"


def _scratch(chip_dir: Path, chip_id: str) -> Path:
    """Where the overlay is drawn before it is moved into place (hidden)."""
    return chip_dir / f".{chip_id}_overlay.tmp{PREVIEW_SUFFIX}"


def _mask(chip_dir: Path, chip_id: str) -> Path:
    return chip_dir / f"{chip_id}{REFERENCE_MASK_SUFFIX}"


def _slot_image(chip_dir: Path, chip_id: str, slot: str | None) -> Path | None:
    return existing_preview(chip_dir, f"{chip_id}_{slot}_image_s2") if slot else None


def _mtime(path: Path) -> float | None:
    try:
        return path.stat().st_mtime
    except FileNotFoundError:
        return None


def overlay_is_stale(chip_dir: Path, chip_id: str) -> bool:
    """Whether the chip's overlay needs drawing again (see the module docstring).

    Only a missing file reads as absent; any other error (a permission error) raises.
    """
    mask_mtime = _mtime(_mask(chip_dir, chip_id))
    if mask_mtime is None:
        return False
    overlay_mtime = _mtime(_overlay(chip_dir, chip_id))
    # Strictly older: a tie is not stale (see the module docstring).
    if overlay_mtime is not None:
        return overlay_mtime < mask_mtime
    base = _slot_image(chip_dir, chip_id, best_slot_preview(chip_dir, chip_id))
    base_mtime = _mtime(base) if base else None
    return base_mtime is not None and base_mtime < mask_mtime


def _stamp(paths: Iterable[Path | None], mask: Path) -> None:
    """Date ``paths`` now, or at the mask's time when that is later (clock skew)."""
    stamp = max(time.time(), _mtime(mask) or 0.0)
    for path in {p for p in paths if p is not None}:
        os.utime(path, (stamp, stamp))


def _plan(item_path: Path) -> _RedrawTask:
    """Read a stale chip's item and pick the slot to draw from (calling thread)."""
    chip_dir, chip_id = item_path.parent, item_path.stem
    item = pystac.Item.from_file(str(item_path))
    slot = preview_slot(item, chip_dir)
    if _slot_image(chip_dir, chip_id, slot) is None:
        slot = best_slot_preview(chip_dir, chip_id)  # its image is gone: the best left
    old = item.assets.get("thumbnail")
    checksums = old is not None and "file:checksum" in old.extra_fields
    return _RedrawTask(item=item, item_path=item_path, slot=slot, checksums=checksums)


def _draw(task: _RedrawTask) -> Path | None:
    """Draw the overlay to its temporary file (worker thread); None with no image left."""
    base = _slot_image(task.chip_dir, task.chip_id, task.slot)
    if base is None:
        return None
    scratch = _scratch(task.chip_dir, task.chip_id)
    try:
        generate_overlay_thumbnail(base, _mask(task.chip_dir, task.chip_id), scratch)
    except BaseException:
        scratch.unlink(missing_ok=True)
        raise
    return scratch


def _install(task: _RedrawTask, scratch: Path) -> None:
    """Point the item at the new overlay, write it, then move the overlay into place."""
    overlay = _overlay(task.chip_dir, task.chip_id)
    set_thumbnail(task.item, task.slot, overlay, file_info_from=scratch, checksums=task.checksums)
    write_item(task.item, task.item_path)
    scratch.replace(overlay)
    _stamp([overlay], _mask(task.chip_dir, task.chip_id))


def _fall_back_to_plain(task: _RedrawTask) -> None:
    """After a failed draw: the plain preview as thumbnail, the stale overlay removed.

    The slot previews are then marked as tried (see the module docstring): both the
    one drawn from and the one :func:`overlay_is_stale` reads, when they differ.
    """
    chip_dir, chip_id = task.chip_dir, task.chip_id
    base = _slot_image(chip_dir, chip_id, task.slot)
    set_thumbnail(task.item, task.slot, base, checksums=task.checksums)
    write_item(task.item, task.item_path)
    _overlay(chip_dir, chip_id).unlink(missing_ok=True)
    best = _slot_image(chip_dir, chip_id, best_slot_preview(chip_dir, chip_id))
    _stamp([base, best], _mask(chip_dir, chip_id))


def _remove(task: _RedrawTask) -> None:
    """No slot preview is left: drop the thumbnail and the stale overlay."""
    task.item.assets.pop("thumbnail", None)
    write_item(task.item, task.item_path)
    _overlay(task.chip_dir, task.chip_id).unlink(missing_ok=True)
    _scratch(task.chip_dir, task.chip_id).unlink(missing_ok=True)
    logger.warning(f"{task.chip_id}: removed the stale chip overlay; no preview image is left")


def _settle(outcome: ParallelOutcome[_RedrawTask, Path | None]) -> str:
    """Write the chip's item and update its files (calling thread); returns the outcome."""
    task = outcome.task
    if outcome.error is not None:
        logger.warning(f"{task.chip_id}: could not draw the chip overlay: {outcome.error}")
        _fall_back_to_plain(task)
        return _FAILED
    if outcome.value is None:
        _remove(task)
        return _REMOVED
    _install(task, outcome.value)
    return _REDRAWN


def _apply(outcome: ParallelOutcome[_RedrawTask, Path | None], result: OverlayRedrawResult) -> None:
    """Settle one chip and count it once (calling thread)."""
    task = outcome.task
    try:
        status = _settle(outcome)
    except Exception as err:  # any one chip's failure is counted, never the run's
        result.failed += 1
        logger.warning(f"{task.chip_id}: could not redraw the chip preview: {err}")
        _scratch(task.chip_dir, task.chip_id).unlink(missing_ok=True)
        return
    setattr(result, status, getattr(result, status) + 1)


def _stale_task(item_path: Path) -> _RedrawTask | None:
    """The chip's redraw task when stale; else None, its leftover temporary file gone."""
    chip_dir, chip_id = item_path.parent, item_path.stem
    if overlay_is_stale(chip_dir, chip_id):
        try:
            return _plan(item_path)
        except Exception:
            _scratch(chip_dir, chip_id).unlink(missing_ok=True)
            raise
    _scratch(chip_dir, chip_id).unlink(missing_ok=True)
    return None


def _stale_tasks(
    item_paths: Iterable[Path], result: OverlayRedrawResult, advance: Callable[[], None]
) -> list[_RedrawTask]:
    """Plan the stale chips among ``item_paths``; ``advance`` for each one not planned."""
    tasks = []
    for item_path in item_paths:
        try:
            task = _stale_task(item_path)
        except Exception as err:  # an unreadable chip is counted, not fatal to the run
            result.failed += 1
            logger.warning(f"{item_path.stem}: could not redraw the chip preview: {err}")
            task = None
        if task is None:
            advance()
        else:
            tasks.append(task)
    return tasks


def _redraw_batch(
    batch: list[Path],
    result: OverlayRedrawResult,
    advance: Callable[[], None],
    apply: Callable[[ParallelOutcome[_RedrawTask, Path | None]], None],
    workers: int,
) -> None:
    """Plan and redraw one batch; its tasks (and items) go when this returns."""
    run_in_parallel(_stale_tasks(batch, result, advance), _draw, apply, workers=workers)


def _batches(item_paths: Iterable[Path], size: int) -> Iterator[list[Path]]:
    iterator = iter(item_paths)
    while batch := list(islice(iterator, size)):
        yield batch


def _log(result: OverlayRedrawResult) -> None:
    parts = [f"redrew {result.redrawn} stale chip overlay(s)"] if result.redrawn else []
    if result.failed:
        parts.append(f"{result.failed} could not be redrawn")
    if result.removed:
        parts.append(f"{result.removed} removed (no preview image left)")
    if parts:
        logger.info("Chip previews: " + "; ".join(parts))


def redraw_stale_overlays(
    item_paths: Iterable[Path], *, workers: int = DEFAULT_WORKERS
) -> OverlayRedrawResult:
    """Redraw the stale overlays among these chip items, ``workers`` at a time.

    Drawing runs on a thread pool; each item is written on the calling thread. A chip
    whose overlay cannot be drawn gets its plain preview as the thumbnail and is
    logged and counted in ``failed``, without failing the run.

    Args:
        item_paths: Chip item JSON files, ``<chip_dir>/<chip_id>.json``.
        workers: Number of overlays to draw concurrently.
    """
    item_paths = list(item_paths)
    result = OverlayRedrawResult()
    # Over every chip, so it moves while the fresh ones are checked too.
    progress_bar = tqdm(total=len(item_paths), desc="Checking previews", unit="chip", leave=False)

    def advance() -> None:
        progress_bar.update(1)

    def apply(outcome: ParallelOutcome[_RedrawTask, Path | None]) -> None:
        _apply(outcome, result)
        advance()

    try:
        # In batches, so only one batch's items are held in memory at a time: each
        # batch's tasks are released before the next batch is planned.
        for batch in _batches(item_paths, PLAN_BATCH_SIZE):
            _redraw_batch(batch, result, advance, apply, workers)
    finally:
        progress_bar.close()
    _log(result)
    return result
