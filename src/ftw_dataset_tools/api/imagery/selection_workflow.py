"""Image selection orchestration for STAC catalogs.

This module provides workflow functions for selecting imagery across all chips
in a catalog. It is used by both the standalone `select-images` command and
the `create-dataset` pipeline to ensure identical behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import pystac

from ftw_dataset_tools.api.imagery.progress import ImageryProgressBar
from ftw_dataset_tools.api.imagery.scene_selection import select_scenes_for_chip
from ftw_dataset_tools.api.imagery.stac_child_items import create_child_items_from_selection

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "SelectionWorkflowResult",
    "find_chip_items",
    "select_imagery_for_catalog",
]


@dataclass
class SelectionWorkflowResult:
    """Result of running image selection across a catalog."""

    successful: int = 0
    skipped: int = 0
    failed: int = 0
    skipped_details: list[dict] = field(default_factory=list)
    failed_details: list[dict] = field(default_factory=list)


def find_chip_items(catalog_dir: Path) -> list[tuple[pystac.Item, Path]]:
    """Find all parent chip items in a catalog directory.

    Searches subdirectories for STAC item JSON files, excluding child S2 items
    (those ending in _planting_s2 or _harvest_s2).

    Args:
        catalog_dir: Path to the chips collection directory containing subdirectories
                     with STAC item files

    Returns:
        List of (pystac.Item, item_path) tuples for each parent chip item found.
        Returns empty list if no items found.
    """
    chip_items = []

    for subdir in catalog_dir.iterdir():
        if subdir.is_dir() and not subdir.name.startswith("."):
            for json_file in subdir.glob("*.json"):
                # Skip child items (they have _planting_s2 or _harvest_s2 suffix)
                if "_planting_s2" in json_file.name or "_harvest_s2" in json_file.name:
                    continue
                try:
                    item = pystac.Item.from_file(str(json_file))
                    chip_items.append((item, json_file))
                except Exception:
                    # Skip invalid JSON files
                    pass

    return chip_items


def _has_existing_scenes(item: pystac.Item) -> bool:
    """Check if item already has planting and harvest scene links."""
    has_planting = any(link.rel == "ftw:planting" for link in item.links)
    has_harvest = any(link.rel == "ftw:harvest" for link in item.links)
    return has_planting and has_harvest


def select_imagery_for_catalog(
    catalog_dir: Path,
    year: int,
    cloud_cover_chip: float = 2.0,
    nodata_max: float = 0.0,
    buffer_days: int = 14,
    num_buffer_expansions: int = 3,
    buffer_expansion_size: int = 14,
    force: bool = False,
    on_missing: Literal["skip", "fail", "best-available"] = "skip",
    verbose: bool = False,
) -> SelectionWorkflowResult:
    """Select imagery for all chips in a catalog.

    This is the core orchestration function used by both `select-images` command
    and `create-dataset` pipeline.

    Args:
        catalog_dir: Path to the chips collection directory
        year: Calendar year for the crop cycle
        cloud_cover_chip: Maximum chip-level cloud cover percentage (0-100)
        nodata_max: Maximum nodata percentage (0-100). Default 0 rejects any nodata.
        buffer_days: Days to search around crop calendar dates
        num_buffer_expansions: Number of times to expand search window
        buffer_expansion_size: Days to add to buffer on each expansion
        force: If True, overwrite existing selections. If False, skip chips with scenes.
        on_missing: How to handle chips with no cloud-free scenes:
                    - "skip": Skip and record in skipped_details
                    - "fail": Raise exception
                    - "best-available": Use best available scene regardless of threshold
        verbose: If True, show detailed STAC query information

    Returns:
        SelectionWorkflowResult with success/skipped/failed counts and details

    Raises:
        Exception: If on_missing="fail" and no cloud-free scenes found
    """
    result = SelectionWorkflowResult()

    # Find all chip items
    chip_items = find_chip_items(catalog_dir)

    if not chip_items:
        return result

    # Pre-filter chips that already have imagery (unless force=True)
    chips_to_process: list[tuple[pystac.Item, Path]] = []

    for item, item_path in chip_items:
        bbox = tuple(item.bbox) if item.bbox else None

        if bbox is None:
            result.skipped += 1
            result.skipped_details.append({"chip": item.id, "reason": "No bbox in item"})
            continue

        # Skip chips that already have scene selections (unless --force)
        if not force and _has_existing_scenes(item):
            result.skipped += 1
            result.skipped_details.append(
                {"chip": item.id, "reason": "Already has imagery selections"}
            )
            continue

        chips_to_process.append((item, item_path))

    if not chips_to_process:
        return result

    # Process chips with unified progress display
    with ImageryProgressBar(total=len(chips_to_process), leave=False, verbose=verbose) as progress:
        for item, item_path in chips_to_process:
            progress.start_chip(item.id)
            bbox = tuple(item.bbox)  # Already validated above

            try:
                selection_result = select_scenes_for_chip(
                    chip_id=item.id,
                    bbox=bbox,
                    year=year,
                    cloud_cover_chip=cloud_cover_chip,
                    nodata_max=nodata_max,
                    buffer_days=buffer_days,
                    num_buffer_expansions=num_buffer_expansions,
                    buffer_expansion_size=buffer_expansion_size,
                    on_progress=progress.on_progress,
                )

                if selection_result.success:
                    # Create child STAC items using shared logic
                    create_child_items_from_selection(
                        chip_dir=item_path.parent,
                        parent_item=item,
                        result=selection_result,
                        year=year,
                        cloud_cover_chip=cloud_cover_chip,
                        buffer_days=buffer_days,
                        num_buffer_expansions=num_buffer_expansions,
                        buffer_expansion_size=buffer_expansion_size,
                    )
                    result.successful += 1
                    progress.mark_success(selection_result)
                else:
                    if on_missing == "fail":
                        raise ValueError(
                            f"No cloud-free scenes for {item.id}: {selection_result.skipped_reason}"
                        )
                    result.skipped += 1
                    result.skipped_details.append(
                        {
                            "chip": item.id,
                            "reason": selection_result.skipped_reason or "Unknown reason",
                            "candidates_checked": selection_result.candidates_checked,
                        }
                    )
                    progress.mark_skipped(selection_result.skipped_reason or "No cloud-free scenes")

            except ValueError:
                # Re-raise ValueError from on_missing="fail"
                raise
            except Exception as e:
                if on_missing == "fail":
                    raise
                result.failed += 1
                result.failed_details.append({"chip": item.id, "error": str(e)})
                progress.mark_failed(str(e))

    return result
