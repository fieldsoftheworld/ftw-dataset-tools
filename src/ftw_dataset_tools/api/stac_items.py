"""STAC item manipulation utilities.

Provides safe saving and manipulation of STAC items with proper error handling.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime for path operations
from typing import Literal

import pystac


class STACSaveError(Exception):
    """Error saving STAC item."""


@dataclass
class STACSaveContext:
    """Context for saving STAC items after download."""

    item: pystac.Item
    item_dir: Path
    season: Literal["planting", "harvest"]
    band_list: list[str]
    output_filename: str


def save_child_item(ctx: STACSaveContext) -> None:
    """Save a child STAC item with cleanup on failure.

    Deletes the downloaded TIF if the JSON save fails.

    Args:
        ctx: Context containing item and path information

    Raises:
        STACSaveError: If the save operation fails
    """
    tif_path = ctx.item_dir / ctx.output_filename
    json_path = ctx.item_dir / f"{ctx.item.id}.json"

    try:
        ctx.item.save_object(str(json_path))
    except Exception as e:
        # Cleanup: delete the downloaded TIF
        if tif_path.exists():
            tif_path.unlink()
        raise STACSaveError(f"Failed to save STAC item {ctx.item.id} at {ctx.item_dir}: {e}") from e


def update_parent_item(
    parent_item: pystac.Item,
    parent_path: Path,
    season: Literal["planting", "harvest"],
    output_filename: str,
    band_list: list[str],
    thumbnail_filename: str | None = None,
    is_overlay: bool = False,
) -> None:
    """Update parent item with reference to downloaded image.

    Rolls back the in-memory asset if save fails.

    Args:
        parent_item: Parent STAC item to update
        parent_path: Path to parent item JSON
        season: Season identifier
        output_filename: Name of downloaded image file
        band_list: List of bands in the image
        thumbnail_filename: Optional thumbnail filename. If provided and season is
            "planting", adds as the chip's thumbnail asset.
        is_overlay: If True, thumbnail has mask overlay (used for title).

    Raises:
        STACSaveError: If the save operation fails
    """
    asset_key = f"{season}_image"
    added_thumbnail = False

    try:
        parent_item.assets[asset_key] = pystac.Asset(
            href=f"./{output_filename}",
            media_type="image/tiff; application=geotiff; profile=cloud-optimized",
            title=f"{season.capitalize()} season imagery ({','.join(band_list)})",
            roles=["data"],
        )

        # Add planting thumbnail as the chip's thumbnail
        if thumbnail_filename and season == "planting":
            thumb_title = (
                "Chip preview with field overlay"
                if is_overlay
                else "Chip preview (planting season)"
            )
            parent_item.assets["thumbnail"] = pystac.Asset(
                href=f"./{thumbnail_filename}",
                media_type=pystac.MediaType.JPEG,
                title=thumb_title,
                roles=["thumbnail"],
            )
            added_thumbnail = True

        parent_item.save_object(str(parent_path))
    except Exception as e:
        parent_item.assets.pop(asset_key, None)
        if added_thumbnail:
            parent_item.assets.pop("thumbnail", None)
        raise STACSaveError(f"Failed to update parent item at {parent_path}: {e}") from e


def copy_catalog(src: Path, dst: Path) -> None:
    """Copy catalog directory safely, not following symlinks.

    Copies symlinks as symlinks rather than following them.

    Args:
        src: Source catalog directory
        dst: Destination directory

    Raises:
        ValueError: If destination already exists
    """
    if dst.exists():
        raise ValueError(f"Destination already exists: {dst}")

    shutil.copytree(
        src,
        dst,
        symlinks=True,  # Copy symlinks as symlinks, don't follow
        ignore_dangling_symlinks=True,
    )
