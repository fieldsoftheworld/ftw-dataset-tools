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

from ftw_dataset_tools.api.imagery.image_download import (
    download_and_clip_scene,
    process_downloaded_scene,
)
from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene
from ftw_dataset_tools.api.imagery.thumbnails import has_rgb_bands

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

__all__ = [
    "DownloadWorkflowResult",
    "download_imagery_for_catalog",
    "find_s2_child_items",
]


@dataclass
class DownloadWorkflowResult:
    """Result of running image download across a catalog."""

    successful: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_details: list[dict] = field(default_factory=list)
    failed_details: list[dict] = field(default_factory=list)


def find_s2_child_items(catalog_dir: Path) -> list[tuple[pystac.Item, Path]]:
    """Find all S2 child items (planting/harvest) in a catalog directory.

    Searches subdirectories for STAC item JSON files that end with
    _planting_s2 or _harvest_s2.

    Args:
        catalog_dir: Path to the chips collection directory containing subdirectories
                     with STAC item files

    Returns:
        List of (pystac.Item, item_path) tuples for each S2 child item found.
        Returns empty list if no items found.
    """
    child_items = []

    for subdir in catalog_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            for json_file in subdir.glob("*_s2.json"):
                try:
                    item = pystac.Item.from_file(str(json_file))
                    # Only include child items (they have _planting_s2 or _harvest_s2 suffix)
                    if item.id.endswith("_planting_s2") or item.id.endswith("_harvest_s2"):
                        child_items.append((item, json_file))
                except Exception:
                    # Skip invalid JSON files
                    pass

    return child_items


def download_imagery_for_catalog(
    catalog_dir: Path,
    bands: list[str] | None = None,
    resolution: float = 10.0,
    generate_thumbnails: bool = True,
    resume: bool = False,
    on_progress: Callable[[int, int], None] | None = None,
    show_progress_bar: bool = True,
) -> DownloadWorkflowResult:
    """Download imagery for all S2 child items in a catalog.

    This is the core orchestration function used by both `download-images` command
    and `create-dataset` pipeline.

    Args:
        catalog_dir: Path to the chips collection directory
        bands: List of bands to download. Default: ["red", "green", "blue", "nir"]
        resolution: Target resolution in meters
        generate_thumbnails: Whether to generate JPEG preview thumbnails
        resume: If True, skip items that already have local imagery
        on_progress: Optional callback (current, total) for progress updates
        show_progress_bar: If True, show tqdm progress bar

    Returns:
        DownloadWorkflowResult with success/skipped/failed counts and details
    """
    if bands is None:
        bands = ["red", "green", "blue", "nir"]

    result = DownloadWorkflowResult()
    band_list = list(bands)
    can_generate_thumbnail = generate_thumbnails and has_rgb_bands(band_list)

    # Find all child S2 items
    child_items = find_s2_child_items(catalog_dir)

    if not child_items:
        return result

    # Create progress bar context
    progress_bar = (
        tqdm(total=len(child_items), desc="Downloading imagery", unit="scene", leave=False)
        if show_progress_bar
        else None
    )

    try:
        for idx, (item, item_path) in enumerate(child_items):
            bbox = tuple(item.bbox) if item.bbox else None

            if bbox is None:
                result.skipped += 1
                result.skipped_details.append({"item": item.id, "reason": "No bbox in item"})
                if progress_bar:
                    progress_bar.update(1)
                if on_progress:
                    on_progress(idx + 1, len(child_items))
                continue

            # Check if already downloaded (resume mode or existing local file)
            if resume and ("clipped" in item.assets or "image" in item.assets):
                # Check if the local file exists
                local_asset = item.assets.get("image") or item.assets.get("clipped")
                if local_asset:
                    local_path = item_path.parent / local_asset.href.lstrip("./")
                    if local_path.exists():
                        result.skipped += 1
                        result.skipped_details.append(
                            {"item": item.id, "reason": "Already downloaded"}
                        )
                        if progress_bar:
                            progress_bar.update(1)
                        if on_progress:
                            on_progress(idx + 1, len(child_items))
                        continue

            # Determine season from item ID
            if item.id.endswith("_planting_s2"):
                season: Literal["planting", "harvest"] = "planting"
            else:
                season = "harvest"

            # Construct output filename
            base_id = item.id.replace("_planting_s2", "").replace("_harvest_s2", "")
            output_filename = f"{base_id}_{season}_image_s2.tif"
            output_path = item_path.parent / output_filename

            try:
                scene = SelectedScene(
                    item=item,
                    season=season,
                    cloud_cover=item.properties.get("eo:cloud_cover", 0.0),
                    datetime=item.datetime,
                    stac_url=item.get_self_href() or "",
                )

                download_result = download_and_clip_scene(
                    scene=scene,
                    bbox=bbox,
                    output_path=output_path,
                    bands=band_list,
                    resolution=resolution,
                )

                if download_result.success:
                    # Use shared processing logic
                    process_downloaded_scene(
                        item=item,
                        item_path=item_path,
                        output_path=output_path,
                        output_filename=output_filename,
                        band_list=band_list,
                        season=season,
                        base_id=base_id,
                        generate_thumbnails=can_generate_thumbnail,
                    )
                    result.successful += 1
                else:
                    result.failed += 1
                    result.failed_details.append(
                        {"item": item.id, "error": download_result.error or "Unknown error"}
                    )

            except Exception as e:
                result.failed += 1
                result.failed_details.append({"item": item.id, "error": str(e)})

            if progress_bar:
                progress_bar.update(1)
            if on_progress:
                on_progress(idx + 1, len(child_items))

    finally:
        if progress_bar:
            progress_bar.close()

    return result
