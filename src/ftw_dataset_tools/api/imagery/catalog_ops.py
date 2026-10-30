"""Catalog-level operations for imagery management.

This module provides functions for managing imagery selections at the catalog level,
including checking selection status, gathering statistics, and clearing selections.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pystac

from ftw_dataset_tools.api.imagery.slots import (
    IMAGERY_MODES,
    IMAGERY_SLOTS,
    SLOTS_BY_MODE,
    THUMBNAIL_SLOTS,
)
from ftw_dataset_tools.api.imagery.thumbnails import PREVIEW_EXTENSIONS
from ftw_dataset_tools.api.stac_items import write_item

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "IMAGERY_ASSET_KEYS",
    "IMAGERY_LINK_RELS",
    "IMAGERY_MODES_PROPERTY",
    "IMAGERY_PROPERTIES",
    "IMAGERY_TEMPORAL_PROPERTIES",
    "MOSAIC_PROPERTIES",
    "PROPERTIES_BY_MODE",
    "SCENE_PROPERTIES",
    "ClearResult",
    "ImageryStats",
    "SelectionConflictError",
    "chip_dir_for_item",
    "clear_chip_selections",
    "find_collection_dir",
    "get_imagery_stats",
    "has_existing_scenes",
    "has_mode_selection",
    "iter_chip_dirs",
    "mosaic_requested_year",
    "normalize_imagery_properties",
    "preserve_imagery_selection",
    "preview_slot",
    "refresh_temporal_extent",
    "selected_modes",
    "selection_conflict",
]


class SelectionConflictError(Exception):
    """Chips already hold a selection of the same imagery mode for a different year."""


# The bookkeeping that image selection and download write onto a *parent* chip
# item (see api/imagery/stac_child_items.py and api/stac_items.py). Shared so
# that clearing a selection and preserving one across catalog regeneration stay
# in sync: a key missing from these lists is silently dropped on a re-run.
IMAGERY_LINK_RELS = tuple(f"ftw:{slot}" for slot in IMAGERY_SLOTS)

#: What a scene selection records on the parent. The names already say which
#: mode they belong to (crop calendar, planting/harvest, scene search buffers).
SCENE_PROPERTIES = (
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

#: What a mosaic selection records on the parent.
MOSAIC_PROPERTIES = ("ftw:mosaic_requested_year", "ftw:mosaic_year")

#: The modes whose selection is complete on the chip, e.g. ``["scenes", "mosaics"]``.
#: Derived from the slot links, which stay the source of truth.
IMAGERY_MODES_PROPERTY = "ftw:imagery_modes"

# Catalogs written before a chip could hold both modes carry one scalar
# ``ftw:imagery_mode`` and un-namespaced mosaic years. They are still read, and are
# rewritten to the per-mode keys whenever the chip's item is written again.
_LEGACY_MODE_PROPERTY = "ftw:imagery_mode"
_LEGACY_MOSAIC_PROPERTIES = {
    "ftw:requested_year": "ftw:mosaic_requested_year",
    "ftw:imagery_year": "ftw:mosaic_year",
}

PROPERTIES_BY_MODE = {
    "scenes": SCENE_PROPERTIES,
    "mosaics": (*MOSAIC_PROPERTIES, *_LEGACY_MOSAIC_PROPERTIES),
}

IMAGERY_PROPERTIES = (
    IMAGERY_MODES_PROPERTY,
    _LEGACY_MODE_PROPERTY,
    *SCENE_PROPERTIES,
    *PROPERTIES_BY_MODE["mosaics"],
)

# Selection narrows these to the actual scene acquisition dates; a freshly
# generated item carries the dataset-wide extent instead.
IMAGERY_TEMPORAL_PROPERTIES = ("start_datetime", "end_datetime")

# Added to the parent once imagery has been downloaded.
IMAGERY_ASSET_KEYS = (*(f"{slot}_image" for slot in IMAGERY_SLOTS), "thumbnail")


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


def has_mode_selection(item: pystac.Item, mode: str) -> bool:
    """Whether the item links every slot of ``mode`` (both seasons, or all four quarters)."""
    rels = {link.rel for link in item.links}
    return all(f"ftw:{slot}" in rels for slot in SLOTS_BY_MODE[mode])


def selected_modes(item: pystac.Item) -> tuple[str, ...]:
    """The modes whose selection is complete on the item, in canonical order."""
    return tuple(mode for mode in IMAGERY_MODES if has_mode_selection(item, mode))


def has_existing_scenes(item: pystac.Item) -> bool:
    """Whether the item links a complete selection in any mode."""
    return bool(selected_modes(item))


def mosaic_requested_year(item: pystac.Item) -> int | None:
    """The year the chip's mosaic selection asked for, read from current or legacy keys."""
    props = item.properties
    year = props.get("ftw:mosaic_requested_year")
    return year if year is not None else props.get("ftw:requested_year")


def normalize_imagery_properties(item: pystac.Item) -> None:
    """Record ``ftw:imagery_modes`` from the links and retire the legacy single-mode keys.

    Call after any change to a chip's selection, right before the item is written.
    """
    props = item.properties
    for legacy, current in _LEGACY_MOSAIC_PROPERTIES.items():
        value = props.pop(legacy, None)
        if value is not None and current not in props:
            props[current] = value
    props.pop(_LEGACY_MODE_PROPERTY, None)
    modes = selected_modes(item)
    if modes:
        props[IMAGERY_MODES_PROPERTY] = list(modes)
    else:
        props.pop(IMAGERY_MODES_PROPERTY, None)


def selection_conflict(item: pystac.Item, imagery_mode: str, year: int | None) -> str | None:
    """Why the item's existing ``imagery_mode`` selection does not match this run, or None.

    Only a selection of the same mode can conflict: a chip holding scenes takes
    mosaics alongside them, and the other way round. ``year`` is only compared in
    mosaic mode; scene selections predate the property.
    """
    if imagery_mode != "mosaics" or not has_mode_selection(item, imagery_mode):
        return None
    existing_year = mosaic_requested_year(item)
    if existing_year != year:
        return f"has mosaics for {existing_year}"
    return None


def preview_slot(chip_dir: Path, chip_id: str) -> str | None:
    """The slot the chip preview is drawn from: the first thumbnail slot selected for it.

    On a chip holding both modes that is the planting scene, whichever image happens
    to download first, so the preview does not depend on download order.
    """
    for slot in THUMBNAIL_SLOTS:
        if (chip_dir / f"{chip_id}_{slot}_s2.json").exists():
            return slot
    return None


def _child_window(child: pystac.Item) -> tuple[datetime, datetime] | None:
    start = child.common_metadata.start_datetime or child.datetime
    end = child.common_metadata.end_datetime or child.datetime
    if start is None or end is None:
        return None
    # Naive times are UTC by STAC convention; comparing them with aware ones raises.
    return tuple(t if t.tzinfo else t.replace(tzinfo=UTC) for t in (start, end))  # type: ignore[return-value]


def _linked_slots(item: pystac.Item) -> list[tuple[str, pystac.Link]]:
    return [
        (link.rel[4:], link)
        for link in item.links
        if link.rel.startswith("ftw:") and link.rel[4:] in IMAGERY_SLOTS
    ]


def refresh_temporal_extent(
    item: pystac.Item,
    chip_dir: Path,
    children: dict[str, pystac.Item] | None = None,
) -> bool:
    """Set the parent's ``start_datetime``/``end_datetime`` to span every linked slot.

    A scene contributes its acquisition time, a mosaic its quarter, so a chip holding
    both modes spans both. ``children`` supplies slot items already in memory; any
    other linked slot is read from disk.

    Returns:
        False when no linked child has a date; the properties are then left alone.
    """
    children = children or {}
    windows = []
    for slot, link in _linked_slots(item):
        child = children.get(slot)
        if child is None:
            try:
                child = pystac.Item.from_file(str(chip_dir / Path(link.href).name))
            except Exception:
                # A half-written child only narrows the extent; not worth failing over.
                continue
        window = _child_window(child)
        if window is not None:
            windows.append(window)
    if not windows:
        return False
    item.properties["start_datetime"] = min(start for start, _end in windows).isoformat()
    item.properties["end_datetime"] = max(end for _start, end in windows).isoformat()
    return True


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

    normalize_imagery_properties(item)
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


def _delete_slot_files(chip_dir: Path, chip_id: str, slot: str, result: ClearResult) -> None:
    """Delete one slot's child item, GeoTIFFs and per-slot previews."""
    child_json = chip_dir / f"{chip_id}_{slot}_s2.json"
    if child_json.exists():
        child_json.unlink()
        result.stac_items_deleted += 1

    # Associated GeoTIFFs (e.g., ftw-xxx_planting_image_s2.tif)
    for tif in chip_dir.glob(f"{chip_id}_{slot}_*.tif"):
        tif.unlink()
        result.geotiffs_deleted += 1

    # Both extensions: previews are written as WebP, but a catalog built before
    # that switch still has .jpg here, and leaving either behind strands a preview
    # for imagery that no longer exists.
    for ext in PREVIEW_EXTENSIONS:
        for thumb in chip_dir.glob(f"{chip_id}_{slot}_*{ext}"):
            thumb.unlink()


def _restore_datetime(item: pystac.Item, calendar_year: int | None) -> None:
    """Give an item with no imagery left a single datetime (Jan 1 of its year, or now)."""
    for prop in IMAGERY_TEMPORAL_PROPERTIES:
        item.properties.pop(prop, None)
    if calendar_year:
        restored_dt = datetime(calendar_year, 1, 1, 0, 0, 0, tzinfo=UTC)
    else:
        restored_dt = datetime.now(UTC)
    # Set both the Item attribute and the properties dict
    item.datetime = restored_dt
    item.properties["datetime"] = restored_dt.isoformat()


def clear_chip_selections(item: pystac.Item, modes: Iterable[str] | None = None) -> ClearResult:
    """Clear imagery selections for a single chip.

    Removes child STAC items, GeoTIFF files, and imagery-related properties
    from the parent item, for every mode or only the given ones. The parent's
    temporal range then spans whatever imagery is left; with none left its
    datetime is restored to a valid single value.

    The chip preview (the field overlay and the ``thumbnail`` asset) goes too when
    it was drawn from a cleared slot; a preview drawn from a mode that stays is kept.

    Args:
        item: Parent chip STAC item (will be modified and saved). Must carry a
            self href, which locates the chip directory whose files are deleted.
        modes: Imagery modes to clear ("scenes", "mosaics"); None clears all.

    Returns:
        ClearResult with counts of deleted items

    Raises:
        ValueError: If the item has no self href.
    """
    chip_dir = chip_dir_for_item(item)
    result = ClearResult()
    modes = IMAGERY_MODES if modes is None else tuple(modes)
    slots = tuple(slot for mode in modes for slot in SLOTS_BY_MODE[mode])

    # Decided before the children go: afterwards the preview's source is unknowable.
    drop_preview = preview_slot(chip_dir, item.id) in (*slots, None)

    for slot in slots:
        _delete_slot_files(chip_dir, item.id, slot, result)
        item.assets.pop(f"{slot}_image", None)
        item.assets.pop(f"{slot}_visual", None)

    if drop_preview:
        for ext in PREVIEW_EXTENSIONS:
            (chip_dir / f"{item.id}_overlay{ext}").unlink(missing_ok=True)
        item.assets.pop("thumbnail", None)

    cleared_rels = {f"ftw:{slot}" for slot in slots}
    item.links = [link for link in item.links if link.rel not in cleared_rels]

    # Read before the properties go: the year a restored datetime falls in.
    calendar_year = item.properties.get("ftw:calendar_year") or mosaic_requested_year(item)
    for mode in modes:
        for prop in PROPERTIES_BY_MODE[mode]:
            item.properties.pop(prop, None)

    # STAC requires either datetime or both start/end_datetime.
    if not refresh_temporal_extent(item, chip_dir):
        _restore_datetime(item, calendar_year)

    normalize_imagery_properties(item)
    write_item(item, chip_dir / f"{item.id}.json")

    return result
