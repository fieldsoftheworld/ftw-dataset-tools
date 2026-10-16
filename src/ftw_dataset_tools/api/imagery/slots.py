"""Imagery sources and modes, and the per-chip image slots each one fills."""

from __future__ import annotations

__all__ = [
    "DEFAULT_IMAGERY_MODE",
    "DEFAULT_SOURCE",
    "IMAGERY_MODES",
    "IMAGERY_SLOTS",
    "IMAGERY_SOURCES",
    "MOSAIC_SLOTS",
    "MOSAIC_SOURCE",
    "MOSAIC_THUMBNAIL_SLOT",
    "SCENE_SLOTS",
    "SLOTS_BY_MODE",
    "SOURCE_SUFFIXES",
    "THUMBNAIL_SLOTS",
    "child_item_id",
    "image_filename",
    "parse_child",
    "parse_child_id",
    "parse_image",
    "parse_image_stem",
    "slot_title",
]

SCENE_SLOTS = ("planting", "harvest")
MOSAIC_SLOTS = ("q1", "q2", "q3", "q4")
IMAGERY_SLOTS = SCENE_SLOTS + MOSAIC_SLOTS

IMAGERY_MODES = ("scenes", "mosaics")
DEFAULT_IMAGERY_MODE = "scenes"
SLOTS_BY_MODE = {"scenes": SCENE_SLOTS, "mosaics": MOSAIC_SLOTS}

#: Each imagery source and the suffix its child items and images carry, so one chip
#: can hold selections from several sources side by side.
SOURCE_SUFFIXES = {"sentinel-2": "s2", "planetscope": "planet"}
IMAGERY_SOURCES = tuple(SOURCE_SUFFIXES)
DEFAULT_SOURCE = "sentinel-2"

#: ``ftw:source`` of a quarterly mosaic child item.
MOSAIC_SOURCE = "sentinel-2-mosaic"

# Q3 is the main growing season across most of the northern-hemisphere cropland;
# Q2 would be the alternative.
MOSAIC_THUMBNAIL_SLOT = "q3"
THUMBNAIL_SLOTS = ("planting", MOSAIC_THUMBNAIL_SLOT)


def child_item_id(chip_id: str, slot: str, source: str = DEFAULT_SOURCE) -> str:
    return f"{chip_id}_{slot}_{SOURCE_SUFFIXES[source]}"


def image_filename(chip_id: str, slot: str, source: str = DEFAULT_SOURCE) -> str:
    return f"{chip_id}_{slot}_image_{SOURCE_SUFFIXES[source]}.tif"


def _split(value: str, infix: str) -> tuple[str, str, str] | None:
    for source, suffix in SOURCE_SUFFIXES.items():
        for slot in IMAGERY_SLOTS:
            tail = f"_{slot}{infix}_{suffix}"
            if value.endswith(tail) and len(value) > len(tail):
                return value[: -len(tail)], slot, source
    return None


def parse_child(item_id: str) -> tuple[str, str, str] | None:
    """``(chip_id, slot, source)`` for a child item id like ``{chip}_q1_planet``, else None."""
    return _split(item_id, "")


def parse_image(stem: str) -> tuple[str, str, str] | None:
    """``(chip_id, slot, source)`` for an image stem like ``{chip}_q1_image_s2``, else None."""
    return _split(stem, "_image")


def parse_child_id(item_id: str) -> tuple[str, str] | None:
    """``(chip_id, slot)`` for a child item id of any source, else None."""
    parsed = parse_child(item_id)
    return parsed[:2] if parsed else None


def parse_image_stem(stem: str) -> tuple[str, str] | None:
    """``(chip_id, slot)`` for an image stem of any source, else None."""
    parsed = parse_image(stem)
    return parsed[:2] if parsed else None


def slot_title(slot: str) -> str:
    """``Planting season`` / ``Q1``, for asset and link titles."""
    return slot.upper() if slot in MOSAIC_SLOTS else f"{slot.capitalize()} season"
