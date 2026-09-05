"""STAC item manipulation utilities.

Provides safe saving and manipulation of STAC items with proper error handling.
"""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path  # noqa: TC003 - used at runtime for path operations
from typing import Literal
from urllib.parse import unquote, urlparse

import pystac

from ftw_dataset_tools.api.assets import add_file_info, add_raster_bands


class STACSaveError(Exception):
    """Error saving STAC item."""


def _relativize(href: str, base: Path) -> str:
    """Rewrite an absolute local `href` as a path relative to `base`.

    Remote hrefs (``http(s)://``, ``s3://``, ...) and already-relative hrefs are
    returned unchanged.
    """
    if href.startswith("file://"):
        local = unquote(urlparse(href).path)
    elif href.startswith("/"):
        local = href
    else:
        return href

    relative = os.path.relpath(local, base)
    return relative if relative.startswith("../") else f"./{relative}"


def write_item(item: pystac.Item, path: Path) -> Path:
    """Write `item` to `path` without resolving its root link.

    ``Item.save_object`` asks for ``get_root()`` (to pick up the root's ``StacIO``),
    so it raises ``pystac.STACError`` whenever the ``rel: root`` link points at a
    file that is not there -- which is exactly the case in a staging tree whose
    items already carry the *published* root href. Every chip write then fails.

    Checked against pystac 1.14.1: ``to_dict(transform_hrefs=True)`` does *not*
    always fail, because ``Link.get_href`` only consults ``owner.get_root()`` for
    hrefs that are already absolute -- but it does raise the same ``STACError`` as
    soon as one link carries an absolute href (a link built from an in-memory
    target object, for instance). So this serializes with ``transform_hrefs=False``
    and relativizes absolute local link and asset hrefs itself.

    Args:
        item: The item to serialize
        path: Destination JSON path; hrefs are made relative to its parent

    Returns:
        The path written.
    """
    document = item.to_dict(include_self_link=False, transform_hrefs=False)
    base = path.parent

    for link in document.get("links", []):
        href = link.get("href")
        if isinstance(href, str):
            link["href"] = _relativize(href, base)

    for asset in document.get("assets", {}).values():
        href = asset.get("href")
        if isinstance(href, str):
            asset["href"] = _relativize(href, base)

    base.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n")
    return path


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
        write_item(ctx.item, json_path)
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

    Rolls back the in-memory asset if save fails. file:checksum is not
    computed for imagery assets.

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
        parent_item.add_asset(
            asset_key,
            pystac.Asset(
                href=f"./{output_filename}",
                media_type="image/tiff; application=geotiff; profile=cloud-optimized",
                title=f"{season.capitalize()} season imagery ({','.join(band_list)})",
                roles=["data"],
            ),
        )
        image_path = parent_path.parent / output_filename
        if image_path.exists():
            add_file_info(parent_item.assets[asset_key], image_path)
            add_raster_bands(parent_item.assets[asset_key], image_path)

        # Add planting thumbnail as the chip's thumbnail
        if thumbnail_filename and season == "planting":
            thumb_title = (
                "Chip preview with field overlay"
                if is_overlay
                else "Chip preview (planting season)"
            )
            parent_item.add_asset(
                "thumbnail",
                pystac.Asset(
                    href=f"./{thumbnail_filename}",
                    media_type=pystac.MediaType.JPEG,
                    title=thumb_title,
                    roles=["thumbnail"],
                ),
            )
            added_thumbnail = True
            thumbnail_path = parent_path.parent / thumbnail_filename
            if thumbnail_path.exists():
                add_file_info(parent_item.assets["thumbnail"], thumbnail_path)

        write_item(parent_item, parent_path)
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
