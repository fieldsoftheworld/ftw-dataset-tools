"""Convert a catalog's JPEG chip previews to WebP.

Previews are written as WebP, but a catalog built before that switch still has
``.jpg`` files on disk and ``.jpg`` hrefs in its items. This re-renders each preview
from the imagery it was made from, repoints the chip item and its season children at
the WebP, and only then removes the superseded ``.jpg``.

Re-rendering rather than transcoding keeps each preview compressed exactly once: the
``.jpg`` has already discarded pixel data, and re-encoding it to WebP would bake
those artifacts in permanently. The source is whatever the chip was built from - the
local clipped GeoTIFF when there is one, otherwise the remote scene COG - so this
covers both kinds of catalog, which re-running the preview stage alone does not.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.assets import add_file_info
from ftw_dataset_tools.api.imagery.parallel import DEFAULT_WORKERS, run_in_parallel
from ftw_dataset_tools.api.imagery.preview_workflow import build_preview_task, render_preview
from ftw_dataset_tools.api.imagery.selection_workflow import find_chip_items
from ftw_dataset_tools.api.imagery.stac_child_items import SEASONS, attach_thumbnail_to_parent
from ftw_dataset_tools.api.imagery.thumbnails import (
    LEGACY_PREVIEW_SUFFIXES,
    PREVIEW_MEDIA_TYPE,
    PREVIEW_SUFFIX,
    generate_overlay_thumbnail,
    generate_thumbnail,
)
from ftw_dataset_tools.api.stac_items import write_item

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.parallel import ParallelOutcome

__all__ = [
    "ConversionResult",
    "conversion_summary_line",
    "convert_previews_for_catalog",
]

#: Mask used both as the overlay's grid reference and as the overlay layer.
_REFERENCE_MASK_SUFFIX = "_semantic_3_class.tif"


@dataclass
class ConversionResult:
    """Counts and details for a conversion run."""

    chips_converted: int = 0
    previews_written: int = 0
    legacy_removed: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_details: list[dict] = field(default_factory=list)
    failed_details: list[dict] = field(default_factory=list)


def legacy_previews(chip_dir: Path, item_id: str) -> list[Path]:
    """Every JPEG preview this chip still has on disk.

    Args:
        chip_dir: Directory holding the chip's files
        item_id: Chip item id

    Returns:
        Paths to the chip's legacy previews, empty if it has none
    """
    stems = [f"{item_id}_overlay", *(f"{item_id}_{season}_image_s2" for season in SEASONS)]
    return [
        path
        for stem in stems
        for ext in LEGACY_PREVIEW_SUFFIXES
        if (path := chip_dir / f"{stem}{ext}").exists()
    ]


def _render_season_preview(chip_dir: Path, item_id: str, season: str) -> Path | None:
    """Re-render one season's preview from the chip's local clipped GeoTIFF."""
    tif_path = chip_dir / f"{item_id}_{season}_image_s2.tif"
    if not tif_path.exists():
        return None
    output_path = chip_dir / f"{item_id}_{season}_image_s2{PREVIEW_SUFFIX}"
    generate_thumbnail(tif_path, output_path)
    return output_path


def _render_overlay(item: pystac.Item, item_path: Path, base_path: Path | None) -> Path | None:
    """Re-render the chip's overlay preview, from local imagery or the remote scene."""
    chip_dir = item_path.parent
    mask_path = chip_dir / f"{item.id}{_REFERENCE_MASK_SUFFIX}"
    if not mask_path.exists():
        return None

    if base_path is not None:
        output_path = chip_dir / f"{item.id}_overlay{PREVIEW_SUFFIX}"
        generate_overlay_thumbnail(base_path, mask_path, output_path)
        return output_path

    # No local imagery: the chip references its scene remotely, so re-read the
    # window out of the scene COG exactly as the preview stage does.
    task = build_preview_task(item, item_path)
    if isinstance(task, str):
        return None
    render_preview(task)
    return task.output_path


def render_chip_previews(item: pystac.Item, item_path: Path) -> tuple[Path, ...]:
    """Re-render every preview this chip needs as WebP, writing them beside the JPEGs.

    Nothing is deleted and no item is touched here; this is the part that runs on a
    worker thread.

    Args:
        item: The chip item
        item_path: Path to the chip item JSON

    Returns:
        Paths to the previews written, empty if none could be rendered
    """
    chip_dir = item_path.parent
    rendered: list[Path] = []

    season_previews: dict[str, Path] = {}
    for season in SEASONS:
        written = _render_season_preview(chip_dir, item.id, season)
        if written is not None:
            season_previews[season] = written
            rendered.append(written)

    overlay = _render_overlay(item, item_path, season_previews.get("planting"))
    if overlay is not None:
        rendered.append(overlay)

    return tuple(rendered)


def _repoint_child_items(chip_dir: Path, item_id: str) -> None:
    """Point each season child item's thumbnail at its WebP preview.

    The children carry their own thumbnail asset, so converting only the chip item
    would leave them referencing a file this run is about to delete.
    """
    for season in SEASONS:
        child_path = chip_dir / f"{item_id}_{season}_s2.json"
        if not child_path.exists():
            continue

        preview_path = chip_dir / f"{item_id}_{season}_image_s2{PREVIEW_SUFFIX}"
        if not preview_path.exists():
            continue

        try:
            child = pystac.Item.from_file(str(child_path))
        except Exception:
            # A child that cannot be read is left alone rather than failing the
            # chip: its parent still converts, and the stale child is visible in
            # the catalog either way.
            continue

        thumbnail = child.assets.get("thumbnail")
        if thumbnail is None:
            continue

        thumbnail.href = f"./{preview_path.name}"
        thumbnail.media_type = PREVIEW_MEDIA_TYPE
        thumbnail.title = "WebP preview"
        add_file_info(thumbnail, preview_path)
        write_item(child, child_path)


def commit_chip_conversion(item: pystac.Item, item_path: Path) -> int:
    """Repoint a converted chip's items at the WebP and delete the superseded JPEGs.

    A JPEG is removed only once its WebP counterpart is on disk, so a chip whose
    render partly failed keeps the previews it still has.

    Args:
        item: The chip item, updated in place
        item_path: Path to the chip item JSON

    Returns:
        The number of legacy previews removed
    """
    chip_dir = item_path.parent

    attach_thumbnail_to_parent(item, chip_dir)
    write_item(item, item_path)
    _repoint_child_items(chip_dir, item.id)

    removed = 0
    for path in legacy_previews(chip_dir, item.id):
        if path.with_suffix(PREVIEW_SUFFIX).exists():
            path.unlink()
            removed += 1
    return removed


def convert_previews_for_catalog(
    catalog_dir: Path,
    *,
    dry_run: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    show_progress_bar: bool = True,
    workers: int = DEFAULT_WORKERS,
) -> ConversionResult:
    """Convert every JPEG chip preview in a catalog to WebP.

    Chips that already have only WebP previews are skipped, so a run is resumable
    and re-running over a converted catalog does nothing.

    Args:
        catalog_dir: The collection directory holding ``collection.json``.
        dry_run: Report what would be converted without writing or deleting.
        on_progress: Optional ``(done, total)`` callback.
        show_progress_bar: Show a tqdm bar.
        workers: Number of chips to render concurrently.

    Returns:
        ConversionResult with counts and per-chip details.
    """
    result = ConversionResult()

    unreadable: list[dict] = []
    chip_items = find_chip_items(catalog_dir, unreadable=unreadable)
    result.failed += len(unreadable)
    result.failed_details.extend(unreadable)
    if not chip_items:
        return result

    pending = _pending_conversions(chip_items, result, dry_run=dry_run)
    if dry_run or not pending:
        return result

    progress_bar = (
        tqdm(total=len(pending), desc="Converting previews", unit="chip", leave=False)
        if show_progress_bar
        else None
    )
    done = 0

    def advance() -> None:
        nonlocal done
        done += 1
        if progress_bar:
            progress_bar.update(1)
        if on_progress:
            on_progress(done, len(pending))

    def work(task: tuple[pystac.Item, Path]) -> tuple[Path, ...]:
        return render_chip_previews(*task)

    def apply(outcome: ParallelOutcome[tuple[pystac.Item, Path], tuple[Path, ...]]) -> None:
        item, item_path = outcome.task
        if outcome.error is not None:
            result.failed += 1
            result.failed_details.append({"chip": item.id, "error": str(outcome.error)})
            advance()
            return
        if not outcome.value:
            result.skipped += 1
            result.skipped_details.append(
                {"chip": item.id, "reason": "No imagery to re-render the preview from"}
            )
            advance()
            return
        try:
            removed = commit_chip_conversion(item, item_path)
        except (OSError, pystac.STACError) as err:
            # The WebP is on disk; only the item update failed. Report it rather
            # than counting a chip whose items still point at a deleted JPEG.
            result.failed += 1
            result.failed_details.append({"chip": item.id, "error": str(err)})
            advance()
            return
        result.chips_converted += 1
        result.previews_written += len(outcome.value)
        result.legacy_removed += removed
        advance()

    try:
        run_in_parallel(pending, work, apply, workers=workers)
    finally:
        if progress_bar:
            progress_bar.close()

    return result


def _pending_conversions(
    chip_items: list[tuple[pystac.Item, Path]],
    result: ConversionResult,
    *,
    dry_run: bool,
) -> list[tuple[pystac.Item, Path]]:
    """The chips that still have JPEG previews, counting the rest as skipped."""
    pending: list[tuple[pystac.Item, Path]] = []
    for item, item_path in chip_items:
        legacy = legacy_previews(item_path.parent, item.id)
        if not legacy:
            result.skipped += 1
            result.skipped_details.append({"chip": item.id, "reason": "No JPEG previews"})
            continue
        if dry_run:
            result.chips_converted += 1
            result.previews_written += len(legacy)
            result.legacy_removed += len(legacy)
            continue
        pending.append((item, item_path))
    return pending


def conversion_summary_line(result: ConversionResult) -> str:
    """One-line summary for the run report."""
    return (
        f"{result.chips_converted} chips converted, "
        f"{result.previews_written} previews written, "
        f"{result.legacy_removed} JPEGs removed, "
        f"{result.skipped} skipped, {result.failed} failed"
    )
