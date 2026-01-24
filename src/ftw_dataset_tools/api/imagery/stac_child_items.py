"""Create child STAC items for planting and harvest scenes.

This module provides shared logic for creating child STAC items that is used by both
the standalone `select-images` command and the `create-dataset` pipeline to ensure
identical behavior.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Literal

import pystac

if TYPE_CHECKING:
    from pathlib import Path

    from ftw_dataset_tools.api.imagery.scene_selection import SceneSelectionResult, SelectedScene

__all__ = [
    "create_child_items_from_selection",
]


def create_child_items_from_selection(
    chip_dir: Path,
    parent_item: pystac.Item,
    result: SceneSelectionResult,
    year: int,
    cloud_cover_chip: float,
    buffer_days: int,
    num_buffer_expansions: int = 3,
    buffer_expansion_size: int = 14,
) -> None:
    """Create child STAC items for planting and harvest scenes.

    Updates the parent item with FTW properties and creates child items
    with proper links and asset references.

    This is the canonical implementation used by both `select-images` command
    and `create-dataset` pipeline.

    Args:
        chip_dir: Directory containing the chip STAC items
        parent_item: Parent chip STAC item to update
        result: Scene selection result containing planting and harvest scenes
        year: Calendar year for the crop cycle
        cloud_cover_chip: Cloud cover threshold used for selection
        buffer_days: Initial buffer days used for selection
        num_buffer_expansions: Number of buffer expansions configured
        buffer_expansion_size: Days added per expansion

    Side Effects:
        - Updates and saves the parent item with FTW properties
        - Creates and saves child STAC items for planting and harvest
        - Adds ftw:planting and ftw:harvest links to parent item
    """
    # Ensure parent item has self_href set (required for saving with relative links)
    parent_path = chip_dir / f"{parent_item.id}.json"
    if parent_item.get_self_href() is None:
        parent_item.set_self_href(str(parent_path))

    # Update parent item with FTW properties
    parent_item.properties["ftw:calendar_year"] = year
    parent_item.properties["ftw:planting_day"] = result.crop_calendar.planting_day
    parent_item.properties["ftw:harvest_day"] = result.crop_calendar.harvest_day
    parent_item.properties["ftw:stac_host"] = "earthsearch"  # Always earthsearch
    parent_item.properties["ftw:cloud_cover_chip_threshold"] = cloud_cover_chip
    parent_item.properties["ftw:buffer_days"] = buffer_days
    parent_item.properties["ftw:num_buffer_expansions"] = num_buffer_expansions
    parent_item.properties["ftw:buffer_expansion_size"] = buffer_expansion_size

    # Track actual buffer used for each season
    parent_item.properties["ftw:planting_buffer_used"] = result.planting_buffer_used
    parent_item.properties["ftw:harvest_buffer_used"] = result.harvest_buffer_used
    parent_item.properties["ftw:expansions_performed"] = result.expansions_performed

    # Set temporal extent to the full calendar year
    # This represents the crop cycle year, not just the scene acquisition dates
    parent_item.properties["start_datetime"] = datetime(year, 1, 1, 0, 0, 0, tzinfo=UTC).isoformat()
    parent_item.properties["end_datetime"] = datetime(
        year, 12, 31, 23, 59, 59, tzinfo=UTC
    ).isoformat()

    # Add cloud cover from child scenes to parent
    if result.planting_scene:
        parent_item.properties["ftw:planting_cloud_cover"] = round(
            result.planting_scene.cloud_cover, 2
        )
    if result.harvest_scene:
        parent_item.properties["ftw:harvest_cloud_cover"] = round(
            result.harvest_scene.cloud_cover, 2
        )

    # Remove any existing planting/harvest links before adding new ones
    parent_item.links = [
        link
        for link in parent_item.links
        if link.rel not in ("ftw:planting", "ftw:harvest", "derived")
    ]

    # Add links from parent to child items
    if result.planting_scene:
        planting_child_id = f"{parent_item.id}_planting_s2"
        parent_item.add_link(
            pystac.Link(
                rel="ftw:planting",
                target=f"./{planting_child_id}.json",
                media_type="application/json",
                title="Planting season Sentinel-2 imagery",
            )
        )

    if result.harvest_scene:
        harvest_child_id = f"{parent_item.id}_harvest_s2"
        parent_item.add_link(
            pystac.Link(
                rel="ftw:harvest",
                target=f"./{harvest_child_id}.json",
                media_type="application/json",
                title="Harvest season Sentinel-2 imagery",
            )
        )

    # Save updated parent
    parent_item.save_object(dest_href=str(parent_path))

    # Create planting child item
    if result.planting_scene:
        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=result.planting_scene,
            season="planting",
            year=year,
        )

    # Create harvest child item
    if result.harvest_scene:
        _create_season_child_item(
            chip_dir=chip_dir,
            parent_item=parent_item,
            scene=result.harvest_scene,
            season="harvest",
            year=year,
        )


def _create_season_child_item(
    chip_dir: Path,
    parent_item: pystac.Item,
    scene: SelectedScene,
    season: Literal["planting", "harvest"],
    year: int,
) -> None:
    """Create a child STAC item for a season (planting or harvest).

    Args:
        chip_dir: Directory containing the chip STAC items
        parent_item: Parent chip STAC item
        scene: Selected scene for this season
        season: Season identifier
        year: Calendar year for the crop cycle
    """
    child_id = f"{parent_item.id}_{season}_s2"

    # Create child item
    child_path = chip_dir / f"{child_id}.json"
    child_item = pystac.Item(
        id=child_id,
        geometry=parent_item.geometry,
        bbox=parent_item.bbox,
        datetime=scene.datetime,
        properties={
            "ftw:season": season,
            "ftw:source": "sentinel-2",
            "ftw:calendar_year": year,
        },
    )

    # Set self_href before adding links (required for relative link resolution)
    child_item.set_self_href(str(child_path))

    # Copy relevant band assets from source scene
    bands_to_copy = ["red", "green", "blue", "nir", "scl", "visual"]
    for band in bands_to_copy:
        if band in scene.item.assets:
            child_item.assets[band] = scene.item.assets[band].clone()

    # Add cloud probability asset if available
    if "cloud" in scene.item.assets:
        child_item.assets["cloud_probability"] = scene.item.assets["cloud"].clone()

    # Add links
    child_item.add_link(
        pystac.Link(
            rel="ftw:parent_chip",
            target=f"./{parent_item.id}.json",
            media_type="application/json",
        )
    )

    if scene.stac_url:
        child_item.add_link(
            pystac.Link(
                rel="via",
                target=scene.stac_url,
                media_type="application/json",
            )
        )

    # Always include eo:cloud_cover, rounded to 2 decimal places
    child_item.properties["eo:cloud_cover"] = round(scene.cloud_cover, 2)

    # Save child item
    child_item.save_object(dest_href=str(child_path))
