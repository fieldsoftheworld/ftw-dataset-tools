"""Image download orchestration for STAC catalogs.

This module provides workflow functions for downloading imagery for all child
items in a catalog. It is used by both the standalone `download-images` command
and the `create-dataset` pipeline to ensure identical behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pystac
from tqdm import tqdm

from ftw_dataset_tools.api.imagery.catalog_ops import iter_chip_dirs
from ftw_dataset_tools.api.imagery.image_download import (
    download_and_clip_scene,
    process_downloaded_scene,
)
from ftw_dataset_tools.api.imagery.parallel import (
    DEFAULT_WORKERS,
    ParallelOutcome,
    run_in_parallel,
)
from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene
from ftw_dataset_tools.api.imagery.thumbnails import has_rgb_bands

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.image_download import DownloadResult

__all__ = [
    "DownloadTask",
    "DownloadWorkflowResult",
    "build_download_task",
    "download_imagery_for_catalog",
    "download_task_scene",
    "find_s2_child_items",
    "skip_download_reason",
]

DEFAULT_BANDS = ("red", "green", "blue", "nir")


@dataclass
class DownloadWorkflowResult:
    """Result of running image download across a catalog."""

    successful: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_details: list[dict] = field(default_factory=list)
    failed_details: list[dict] = field(default_factory=list)


def find_s2_child_items(
    catalog_dir: Path,
    unreadable: list[dict] | None = None,
) -> list[tuple[pystac.Item, Path]]:
    """Find all S2 child items (planting/harvest) in a catalog directory.

    Searches subdirectories for STAC item JSON files that end with
    _planting_s2 or _harvest_s2.

    Args:
        catalog_dir: Path to the collection directory (holding collection.json),
                     whose ``chips/<square>/<item_id>/`` subdirectories hold STAC
                     item files
        unreadable: Optional list that collects ``{"item": ..., "error": ...}``
                    entries for item files that could not be parsed. An item that
                    cannot be read is otherwise invisible: it is not downloaded and
                    is counted nowhere, so pass this in to report it.

    Returns:
        List of (pystac.Item, item_path) tuples for each S2 child item found.
        Returns empty list if no items found.
    """
    child_items = []

    for subdir in iter_chip_dirs(catalog_dir):
        for json_file in subdir.glob("*_s2.json"):
            try:
                item = pystac.Item.from_file(str(json_file))
            except Exception as e:
                if unreadable is not None:
                    unreadable.append({"item": json_file.stem, "error": f"Unreadable item: {e}"})
                continue
            # Only include child items (they have _planting_s2 or _harvest_s2 suffix)
            if item.id.endswith("_planting_s2") or item.id.endswith("_harvest_s2"):
                child_items.append((item, json_file))

    return child_items


def download_imagery_for_catalog(
    catalog_dir: Path,
    bands: list[str] | None = None,
    resolution: float = 10.0,
    generate_thumbnails: bool = True,
    resume: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    show_progress_bar: bool = True,
    workers: int = DEFAULT_WORKERS,
) -> DownloadWorkflowResult:
    """Download imagery for all S2 child items in a catalog.

    This is the core orchestration function used by both `download-images` command
    and `create-dataset` pipeline.

    Scenes are fetched on a thread pool; the STAC writes that follow each download
    stay on the calling thread, since the two child items of one chip update the
    same parent item file.

    Args:
        catalog_dir: Path to the chips collection directory
        bands: List of bands to download. Default: ["red", "green", "blue", "nir"]
        resolution: Target resolution in meters
        generate_thumbnails: Whether to generate JPEG preview thumbnails
        resume: If True, skip items that already have local imagery
        on_progress: Optional callback (current, total) for progress updates
        show_progress_bar: If True, show tqdm progress bar
        workers: Number of scenes to download concurrently

    Returns:
        DownloadWorkflowResult with success/skipped/failed counts and details
    """
    band_list = list(bands) if bands is not None else list(DEFAULT_BANDS)
    result = DownloadWorkflowResult()
    can_generate_thumbnail = generate_thumbnails and has_rgb_bands(band_list)

    # Find all child S2 items; items whose JSON cannot be read are reported as
    # failures rather than silently dropped from the run.
    unreadable: list[dict] = []
    child_items = find_s2_child_items(catalog_dir, unreadable=unreadable)
    result.failed += len(unreadable)
    result.failed_details.extend(unreadable)

    if not child_items:
        return result

    progress_bar = (
        tqdm(total=len(child_items), desc="Downloading imagery", unit="scene", leave=False)
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
            on_progress(done, len(child_items))

    try:
        tasks: list[DownloadTask] = []
        for item, item_path in child_items:
            skip_reason = skip_download_reason(item, item_path, resume=resume)
            if skip_reason is not None:
                result.skipped += 1
                result.skipped_details.append({"item": item.id, "reason": skip_reason})
                advance()
                continue
            tasks.append(build_download_task(item, item_path))

        def work(task: DownloadTask) -> DownloadResult:
            return download_task_scene(task, bands=band_list, resolution=resolution)

        def apply(outcome: ParallelOutcome[DownloadTask, DownloadResult]) -> None:
            _record_download(
                outcome,
                result=result,
                band_list=band_list,
                generate_thumbnails=can_generate_thumbnail,
            )
            advance()

        run_in_parallel(tasks, work=work, apply=apply, workers=workers)

    finally:
        if progress_bar:
            progress_bar.close()

    return result


@dataclass
class DownloadTask:
    """One scene to fetch, with the paths its outputs go to.

    ``logs`` collects the download's progress lines so the caller can replay them
    in one block rather than have concurrent downloads interleave their output.
    """

    item: pystac.Item
    item_path: Path
    bbox: tuple[float, ...]
    season: Literal["planting", "harvest"]
    base_id: str
    output_filename: str
    output_path: Path
    logs: list[str] = field(default_factory=list)


def skip_download_reason(item: pystac.Item, item_path: Path, *, resume: bool) -> str | None:
    """Return why this item needs no download, or None if it does."""
    if not item.bbox:
        return "No bbox in item"

    if not resume:
        return None

    local_asset = item.assets.get("image") or item.assets.get("clipped")
    if local_asset is None:
        return None

    local_path = item_path.parent / local_asset.href.lstrip("./")
    return "Already downloaded" if local_path.exists() else None


def build_download_task(item: pystac.Item, item_path: Path) -> DownloadTask:
    """Derive the season and output paths for one child item."""
    season: Literal["planting", "harvest"] = (
        "planting" if item.id.endswith("_planting_s2") else "harvest"
    )
    base_id = item.id.replace("_planting_s2", "").replace("_harvest_s2", "")
    output_filename = f"{base_id}_{season}_image_s2.tif"

    return DownloadTask(
        item=item,
        item_path=item_path,
        bbox=tuple(item.bbox),
        season=season,
        base_id=base_id,
        output_filename=output_filename,
        output_path=item_path.parent / output_filename,
    )


def download_task_scene(
    task: DownloadTask, *, bands: list[str], resolution: float
) -> DownloadResult:
    """Fetch and clip one scene.

    Safe to run on a worker thread: it writes only this task's own GeoTIFF and
    collects its log lines instead of printing them.
    """
    scene = SelectedScene(
        item=task.item,
        season=task.season,
        cloud_cover=task.item.properties.get("eo:cloud_cover", 0.0),
        datetime=task.item.datetime,
        stac_url=task.item.get_self_href() or "",
    )

    return download_and_clip_scene(
        scene=scene,
        bbox=task.bbox,
        output_path=task.output_path,
        bands=bands,
        resolution=resolution,
        on_progress=task.logs.append,
    )


def _record_download(
    outcome: ParallelOutcome[DownloadTask, DownloadResult],
    *,
    result: DownloadWorkflowResult,
    band_list: list[str],
    generate_thumbnails: bool,
) -> None:
    """Update the STAC items for one finished download (calling thread only).

    The parent chip item is shared between a chip's planting and harvest items,
    so this must never run concurrently with itself.
    """
    task = outcome.task

    if outcome.error is not None:
        result.failed += 1
        result.failed_details.append({"item": task.item.id, "error": str(outcome.error)})
        return

    download_result = outcome.value
    if download_result is None or not download_result.success:
        error = download_result.error if download_result else None
        result.failed += 1
        result.failed_details.append({"item": task.item.id, "error": error or "Unknown error"})
        return

    try:
        process_downloaded_scene(
            item=task.item,
            item_path=task.item_path,
            output_path=task.output_path,
            output_filename=task.output_filename,
            band_list=band_list,
            season=task.season,
            base_id=task.base_id,
            generate_thumbnails=generate_thumbnails,
        )
    except Exception as err:  # reported per item, like a failed download
        result.failed += 1
        result.failed_details.append({"item": task.item.id, "error": str(err)})
        return

    result.successful += 1
