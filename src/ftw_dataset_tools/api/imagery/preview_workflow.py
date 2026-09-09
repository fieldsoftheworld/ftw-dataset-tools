"""Chip previews rendered from the remote scene COG, with no local clip.

The download stage writes a clipped 4-band GeoTIFF per chip and previews that. A
dataset that deliberately keeps the scene COG remote - referencing the full asset
and leaving the clipping to the client - has no such file, so there is nothing to
preview from. This module renders the same overlay preview by reading the chip's
window straight out of the scene's true-colour COG instead.

The output filename is the one ``attach_thumbnail_to_parent`` already looks for, so
a preview written here is picked up by the STAC stage exactly like a downloaded
one, with no separate asset wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.imagery.parallel import DEFAULT_WORKERS, run_in_parallel
from ftw_dataset_tools.api.imagery.selection_workflow import find_chip_items
from ftw_dataset_tools.api.imagery.stac_child_items import attach_thumbnail_to_parent
from ftw_dataset_tools.api.imagery.thumbnails import (
    generate_overlay_thumbnail,
    generate_scene_thumbnail,
)
from ftw_dataset_tools.api.stac_items import write_item

if TYPE_CHECKING:
    from collections.abc import Callable

    from ftw_dataset_tools.api.imagery.parallel import ParallelOutcome

#: Mask used both as the preview's grid reference and as the overlay layer.
_REFERENCE_MASK_SUFFIX = "_semantic_3_class.tif"

#: Seasons to draw the preview from, in preference order.
_PREVIEW_SEASONS = ("planting", "harvest")


@dataclass
class PreviewWorkflowResult:
    """Counts and details for a preview run."""

    successful: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_details: list[dict] = field(default_factory=list)
    failed_details: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class PreviewTask:
    """One chip's preview: which scene to read and which mask to overlay."""

    item_id: str
    item_path: Path
    visual_href: str
    mask_path: Path
    output_path: Path


def _visual_href(chip_dir: Path, item_id: str) -> str | None:
    """The true-colour scene href for this chip, preferring the planting season."""
    for season in _PREVIEW_SEASONS:
        child_path = chip_dir / f"{item_id}_{season}_s2.json"
        if not child_path.exists():
            continue
        try:
            child = pystac.Item.from_file(str(child_path))
        except Exception:
            # A half-written child is not worth failing the chip over; the next
            # season may still be readable, and a chip with neither is skipped.
            continue
        visual = child.assets.get("visual")
        href = visual.href if visual else None
        # The scene asset is a URL in every real catalog. An absolute filesystem path
        # is accepted too, so a local mirror - or a test - can stand in for it; what
        # is rejected is a relative href, which is a locally written file, not a scene.
        if href and (href.startswith(("http://", "https://")) or Path(href).is_absolute()):
            return href
    return None


def build_preview_task(item: pystac.Item, item_path: Path) -> PreviewTask | str:
    """Derive one chip's preview task, or a reason it has none."""
    chip_dir = item_path.parent
    mask_path = chip_dir / f"{item.id}{_REFERENCE_MASK_SUFFIX}"
    if not mask_path.exists():
        return "No semantic mask to use as the preview grid"
    href = _visual_href(chip_dir, item.id)
    if href is None:
        return "No remote true-colour scene selected"
    return PreviewTask(
        item_id=item.id,
        item_path=item_path,
        visual_href=href,
        mask_path=mask_path,
        output_path=chip_dir / f"{item.id}_overlay.jpg",
    )


def render_preview(task: PreviewTask) -> None:
    """Render one chip's overlay preview from its remote scene."""
    base_path = task.output_path.with_name(f".{task.output_path.name}.base.jpg")
    try:
        generate_scene_thumbnail(task.visual_href, task.mask_path, base_path)
        generate_overlay_thumbnail(base_path, task.mask_path, task.output_path)
    finally:
        base_path.unlink(missing_ok=True)


def preview_imagery_for_catalog(
    catalog_dir: Path,
    *,
    resume: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    show_progress_bar: bool = True,
    workers: int = DEFAULT_WORKERS,
) -> PreviewWorkflowResult:
    """Render an overlay preview for every chip whose scene stays remote.

    Reads are parallel because each is latency-bound on the scene COG; the STAC
    write that follows stays on the calling thread, as elsewhere in this package.

    Args:
        catalog_dir: The collection directory holding ``collection.json``.
        resume: Skip chips that already have a preview on disk.
        on_progress: Optional ``(done, total)`` callback.
        show_progress_bar: Show a tqdm bar.
        workers: Number of scenes to read concurrently.

    Returns:
        PreviewWorkflowResult with counts and per-chip details.
    """
    result = PreviewWorkflowResult()

    unreadable: list[dict] = []
    chip_items = find_chip_items(catalog_dir, unreadable=unreadable)
    result.failed += len(unreadable)
    result.failed_details.extend(unreadable)
    if not chip_items:
        return result

    progress_bar = (
        tqdm(total=len(chip_items), desc="Rendering previews", unit="chip", leave=False)
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
            on_progress(done, len(chip_items))

    try:
        tasks: list[PreviewTask] = []
        for item, item_path in chip_items:
            built = build_preview_task(item, item_path)
            if isinstance(built, str):
                result.skipped += 1
                result.skipped_details.append({"chip": item.id, "reason": built})
                advance()
                continue
            if resume and built.output_path.exists():
                result.skipped += 1
                result.skipped_details.append({"chip": item.id, "reason": "Already rendered"})
                advance()
                continue
            tasks.append(built)

        def apply(outcome: ParallelOutcome[PreviewTask, None]) -> None:
            task = outcome.task
            if outcome.error is not None:
                result.failed += 1
                result.failed_details.append({"chip": task.item_id, "error": str(outcome.error)})
                advance()
                return
            try:
                parent = pystac.Item.from_file(str(task.item_path))
                attach_thumbnail_to_parent(parent, task.item_path.parent)
                write_item(parent, task.item_path)
            except (OSError, pystac.STACError) as err:
                # The JPEG is on disk; only the item update failed. Report it rather
                # than counting a chip whose item never learned about its preview.
                result.failed += 1
                result.failed_details.append({"chip": task.item_id, "error": str(err)})
                advance()
                return
            result.successful += 1
            advance()

        run_in_parallel(tasks, render_preview, apply, workers=workers)
    finally:
        if progress_bar:
            progress_bar.close()

    return result


def preview_summary_line(result: PreviewWorkflowResult) -> str:
    """One-line summary for the run report."""
    return (
        f"Chip previews: {result.successful} ok, {result.skipped} skipped, {result.failed} failed"
    )
