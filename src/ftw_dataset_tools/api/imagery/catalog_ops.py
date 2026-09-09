"""Catalog-level operations for imagery management.

This module provides functions for managing imagery selections at the catalog level,
including checking selection status, gathering statistics, and clearing selections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pystac

from ftw_dataset_tools.api.stac_items import write_item

__all__ = [
    "IMAGERY_ASSET_KEYS",
    "IMAGERY_LINK_RELS",
    "IMAGERY_PROPERTIES",
    "IMAGERY_TEMPORAL_PROPERTIES",
    "ClearResult",
    "ImageryStats",
    "chip_dir_for_item",
    "clear_chip_selections",
    "find_collection_dir",
    "get_imagery_stats",
    "has_existing_scenes",
    "iter_chip_dirs",
    "preserve_imagery_selection",
]

# The bookkeeping that image selection and download write onto a *parent* chip
# item (see api/imagery/stac_child_items.py and api/stac_items.py). Shared so
# that clearing a selection and preserving one across catalog regeneration stay
# in sync: a key missing from these lists is silently dropped on a re-run.
IMAGERY_LINK_RELS = ("ftw:planting", "ftw:harvest")

IMAGERY_PROPERTIES = (
    "ftw:calendar_year",
    "ftw:planting_day",
    "ftw:harvest_day",
    "ftw:stac_host",
    "ftw:cloud_cover_scene_threshold",
    "ftw:cloud_cover_chip_threshold",
    "ftw:buffer_days",
    "ftw:pixel_check",
    "ftw:num_buffer_expansions",
    "ftw:buffer_expansion_size",
    "ftw:planting_buffer_used",
    "ftw:harvest_buffer_used",
    "ftw:expansions_performed",
    "ftw:planting_cloud_cover",
    "ftw:harvest_cloud_cover",
)

# Selection narrows these to the actual scene acquisition dates; a freshly
# generated item carries the dataset-wide extent instead.
IMAGERY_TEMPORAL_PROPERTIES = ("start_datetime", "end_datetime")

# Added to the parent once imagery has been downloaded.
IMAGERY_ASSET_KEYS = ("planting_image", "harvest_image", "thumbnail")


def iter_chip_dirs(collection_dir: Path) -> list[Path]:
    """Every chip item directory under ``<collection>/chips/<square>/``, sorted.

    Despite the ``iter_`` name, this returns a materialized (sorted) list, not a
    generator.
    """
    chips_root = Path(collection_dir) / "chips"
    if not chips_root.is_dir():
        return []
    dirs = [
        chip
        for square in chips_root.iterdir()
        if square.is_dir() and not square.name.startswith(".")
        for chip in square.iterdir()
        if chip.is_dir() and not chip.name.startswith(".")
    ]
    return sorted(dirs)


def find_collection_dir(path: Path) -> Path:
    """The collection directory for a user-supplied path (must hold collection.json)."""
    path = Path(path)
    if (path / "collection.json").is_file():
        return path
    raise FileNotFoundError(f"No collection.json in {path}; pass the dataset output directory")


def has_existing_scenes(item: pystac.Item) -> bool:
    """Check if item already has planting and harvest scene links.

    Args:
        item: STAC item to check

    Returns:
        True if item has both ftw:planting and ftw:harvest links
    """
    has_planting = any(link.rel == "ftw:planting" for link in item.links)
    has_harvest = any(link.rel == "ftw:harvest" for link in item.links)
    return has_planting and has_harvest


def preserve_imagery_selection(item: pystac.Item, existing_item_path: Path) -> bool:
    """Carry a previous run's imagery selection onto a freshly generated item.

    ``generate_stac_catalog`` rebuilds every parent chip item from the chips
    parquet and the mask files on disk, then overwrites the item JSON. Without
    this, a re-run wipes the ftw:planting/ftw:harvest links that
    :func:`has_existing_scenes` reads moments before selection runs, so every
    chip re-selects and ``--force-image-selection`` has nothing left to force.

    The child ``_planting_s2``/``_harvest_s2`` items are not regenerated, so only
    the parent bookkeeping needs preserving.

    Anything the rebuild already restored is left alone. ``_create_chip_item``
    runs :func:`~ftw_dataset_tools.api.imagery.stac_child_items.attach_existing_seasons`
    first, which re-derives the season links and imagery assets from the child
    items and GeoTIFFs actually on disk; re-adding them here would duplicate the
    ``ftw:<season>`` links and would replace freshly measured ``file:size`` and
    ``raster:bands`` with a stale clone. This stays as the fallback for what the
    rebuild could not reproduce -- most of all the ``ftw:`` bookkeeping
    properties, which live only on the parent.

    Args:
        item: Freshly generated parent chip item, modified in place
        existing_item_path: Path to this chip's item JSON from a previous run

    Returns:
        True if a previous selection was found and carried over
    """
    if not existing_item_path.exists():
        return False

    try:
        existing = pystac.Item.from_file(str(existing_item_path))
    except Exception:
        # A corrupt or half-written item just re-selects; not worth failing
        # catalog generation over.
        return False

    if not has_existing_scenes(existing):
        return False

    # The child season items are not regenerated, so a link is only worth
    # carrying over while the item JSON it points at is still on disk. A chip
    # whose children were deleted re-selects instead of keeping a dangling link
    # that would make it look selected forever.
    chip_dir = existing_item_path.parent
    rebuilt_rels = {link.rel for link in item.links}
    for link in existing.links:
        if link.rel not in IMAGERY_LINK_RELS or link.rel in rebuilt_rels:
            continue
        if (chip_dir / Path(link.href).name).exists():
            item.add_link(link.clone())

    for key in (*IMAGERY_PROPERTIES, *IMAGERY_TEMPORAL_PROPERTIES):
        if key in existing.properties:
            item.properties[key] = existing.properties[key]

    # Downloaded imagery outlives the catalog, but only advertise assets whose
    # files are still on disk.
    for key in IMAGERY_ASSET_KEYS:
        if key in item.assets:
            continue
        asset = existing.assets.get(key)
        if asset is not None and (chip_dir / Path(asset.href).name).exists():
            item.add_asset(key, asset.clone())

    return True


@dataclass
class ImageryStats:
    """Statistics about imagery selections in a catalog."""

    total: int = 0
    with_imagery: int = 0
    without_imagery: int = 0
    planting_cloud_covers: list[float] = field(default_factory=list)
    harvest_cloud_covers: list[float] = field(default_factory=list)

    @property
    def planting_cloud_cover_max(self) -> float | None:
        """Maximum planting cloud cover, or None if no data."""
        return max(self.planting_cloud_covers) if self.planting_cloud_covers else None

    @property
    def planting_cloud_cover_avg(self) -> float | None:
        """Average planting cloud cover, or None if no data."""
        if not self.planting_cloud_covers:
            return None
        return sum(self.planting_cloud_covers) / len(self.planting_cloud_covers)

    @property
    def harvest_cloud_cover_max(self) -> float | None:
        """Maximum harvest cloud cover, or None if no data."""
        return max(self.harvest_cloud_covers) if self.harvest_cloud_covers else None

    @property
    def harvest_cloud_cover_avg(self) -> float | None:
        """Average harvest cloud cover, or None if no data."""
        if not self.harvest_cloud_covers:
            return None
        return sum(self.harvest_cloud_covers) / len(self.harvest_cloud_covers)


def get_imagery_stats(chip_items: list[pystac.Item]) -> ImageryStats:
    """Gather imagery selection statistics from chip items.

    Args:
        chip_items: List of STAC items to analyze

    Returns:
        ImageryStats with counts and cloud cover data
    """
    stats = ImageryStats(total=len(chip_items))

    for item in chip_items:
        if has_existing_scenes(item):
            stats.with_imagery += 1
            # Get cloud cover values
            planting_cc = item.properties.get("ftw:planting_cloud_cover")
            harvest_cc = item.properties.get("ftw:harvest_cloud_cover")
            if planting_cc is not None:
                stats.planting_cloud_covers.append(planting_cc)
            if harvest_cc is not None:
                stats.harvest_cloud_covers.append(harvest_cc)
        else:
            stats.without_imagery += 1

    return stats


@dataclass
class ClearResult:
    """Result of clearing imagery selections for a chip."""

    stac_items_deleted: int = 0
    geotiffs_deleted: int = 0


def chip_dir_for_item(item: pystac.Item) -> Path:
    """Return the on-disk directory holding a chip item and its assets.

    Chip files are co-located with the item JSON, so the item's self href is the
    only reliable way to find them: the layout nests each chip under its MGRS
    square, and an item read from anywhere else may not follow that convention.

    Raises:
        ValueError: If the item has no self href, meaning its directory is
            unknowable and any file operation would silently do nothing.
    """
    self_href = item.get_self_href()
    if self_href is None:
        raise ValueError(
            f"Chip item '{item.id}' has no self href, so its directory on disk is "
            "unknown. Read the item from its catalog (or call set_self_href) "
            "before operating on its files."
        )
    return Path(self_href).parent


def clear_chip_selections(item: pystac.Item) -> ClearResult:
    """Clear imagery selections for a single chip.

    Removes child STAC items, GeoTIFF files, and imagery-related properties
    from the parent item. Restores the item's datetime to a valid state.

    Args:
        item: Parent chip STAC item (will be modified and saved). Must carry a
            self href, which locates the chip directory whose files are deleted.

    Returns:
        ClearResult with counts of deleted items

    Raises:
        ValueError: If the item has no self href.
    """
    chip_dir = chip_dir_for_item(item)
    result = ClearResult()

    # Delete planting and harvest child STAC items and their GeoTIFFs
    for season in ["planting", "harvest"]:
        child_json = chip_dir / f"{item.id}_{season}_s2.json"
        if child_json.exists():
            child_json.unlink()
            result.stac_items_deleted += 1

        # Delete associated GeoTIFFs (e.g., ftw-xxx_planting_image_s2.tif)
        for tif in chip_dir.glob(f"{item.id}_{season}_*.tif"):
            tif.unlink()
            result.geotiffs_deleted += 1

        # Delete thumbnails
        for jpg in chip_dir.glob(f"{item.id}_{season}_*.jpg"):
            jpg.unlink()

    # Delete overlay thumbnail if it exists
    overlay_jpg = chip_dir / f"{item.id}_overlay.jpg"
    if overlay_jpg.exists():
        overlay_jpg.unlink()

    # Remove ftw:planting and ftw:harvest links from parent item
    item.links = [link for link in item.links if link.rel not in IMAGERY_LINK_RELS]

    # Extract calendar year before removing properties (needed to restore datetime)
    calendar_year = item.properties.get("ftw:calendar_year")

    # Remove ftw: properties related to imagery selection
    for prop in IMAGERY_PROPERTIES:
        item.properties.pop(prop, None)

    # Remove planting_image, harvest_image, and thumbnail assets if they exist
    for asset_key in IMAGERY_ASSET_KEYS:
        item.assets.pop(asset_key, None)

    # Restore datetime - STAC requires either datetime or both start/end_datetime
    # Remove the selection-set temporal range and restore a single datetime
    for prop in IMAGERY_TEMPORAL_PROPERTIES:
        item.properties.pop(prop, None)

    # Set datetime to Jan 1 of the calendar year if known, otherwise current date
    if calendar_year:
        restored_dt = datetime(calendar_year, 1, 1, 0, 0, 0, tzinfo=UTC)
    else:
        restored_dt = datetime.now(UTC)
    # Set both the Item attribute and the properties dict
    item.datetime = restored_dt
    item.properties["datetime"] = restored_dt.isoformat()

    # Always save since we've modified the item (removed properties, restored datetime)
    parent_path = chip_dir / f"{item.id}.json"
    if item.get_self_href() is None:
        item.set_self_href(str(parent_path))
    write_item(item, parent_path)

    return result
