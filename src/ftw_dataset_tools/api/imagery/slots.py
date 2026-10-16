"""Imagery modes and the per-chip image slots each one fills."""

from __future__ import annotations

__all__ = [
    "DEFAULT_IMAGERY_MODE",
    "IMAGERY_MODES",
    "IMAGERY_SLOTS",
    "MOSAIC_SLOTS",
    "MOSAIC_SOURCE",
    "MOSAIC_THUMBNAIL_SLOT",
    "SCENE_SLOTS",
    "SLOTS_BY_MODE",
    "THUMBNAIL_SLOTS",
    "child_item_id",
    "image_filename",
    "parse_child_id",
    "parse_image_stem",
    "slot_title",
]

SCENE_SLOTS = ("planting", "harvest")
MOSAIC_SLOTS = ("q1", "q2", "q3", "q4")
IMAGERY_SLOTS = SCENE_SLOTS + MOSAIC_SLOTS

IMAGERY_MODES = ("scenes", "mosaics")
DEFAULT_IMAGERY_MODE = "scenes"
SLOTS_BY_MODE = {"scenes": SCENE_SLOTS, "mosaics": MOSAIC_SLOTS}

#: ``ftw:source`` of a quarterly mosaic child item.
MOSAIC_SOURCE = "sentinel-2-mosaic"

# Q3 is the main growing season across most of the northern-hemisphere cropland;
# Q2 would be the alternative.
MOSAIC_THUMBNAIL_SLOT = "q3"
THUMBNAIL_SLOTS = ("planting", MOSAIC_THUMBNAIL_SLOT)


def child_item_id(chip_id: str, slot: str) -> str:
    return f"{chip_id}_{slot}_s2"


def image_filename(chip_id: str, slot: str) -> str:
    return f"{chip_id}_{slot}_image_s2.tif"


def _split(value: str, tail: str) -> tuple[str, str] | None:
    for slot in IMAGERY_SLOTS:
        suffix = f"_{slot}{tail}"
        if value.endswith(suffix) and len(value) > len(suffix):
            return value[: -len(suffix)], slot
    return None


def parse_child_id(item_id: str) -> tuple[str, str] | None:
    """``(chip_id, slot)`` for a child item id like ``{chip}_q1_s2``, else None."""
    return _split(item_id, "_s2")


def parse_image_stem(stem: str) -> tuple[str, str] | None:
    """``(chip_id, slot)`` for an image stem like ``{chip}_q1_image_s2``, else None."""
    return _split(stem, "_image_s2")


def slot_title(slot: str) -> str:
    """``Planting season`` / ``Q1``, for asset and link titles."""
    return slot.upper() if slot in MOSAIC_SLOTS else f"{slot.capitalize()} season"
