"""Imagery modes and the per-chip image slots each one fills."""

from __future__ import annotations

__all__ = [
    "DEFAULT_IMAGERY_MODE",
    "IMAGERY_MODES",
    "IMAGERY_SLOTS",
    "MOSAIC_SLOTS",
    "MOSAIC_SOURCE",
    "SCENE_SLOTS",
    "SLOTS_BY_MODE",
    "THUMBNAIL_SLOTS",
    "InvalidImageryModeError",
    "child_item_id",
    "image_filename",
    "normalize_imagery_modes",
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

#: Slots the chip preview can be drawn from, in preference order: a scene first, else
#: Q3, the main growing season across most of the northern-hemisphere cropland.
THUMBNAIL_SLOTS = ("planting", "harvest", "q3")


class InvalidImageryModeError(ValueError):
    """An imagery mode setting that is not a known mode, or a list of them."""


def normalize_imagery_modes(value: str | list[str] | tuple[str, ...]) -> tuple[str, ...]:
    """One mode or several, as a tuple in canonical order (scenes before mosaics).

    The order is the order the modes run in, not the order they were written: scenes
    go first so that, on a chip holding both, the planting scene is the one the chip
    preview is drawn from (see ``THUMBNAIL_SLOTS``). Repeats collapse.

    Raises:
        InvalidImageryModeError: For an empty list, a non-string entry or an unknown mode.
    """
    raw = [value] if isinstance(value, str) else value
    if not isinstance(raw, list | tuple) or not raw:
        raise InvalidImageryModeError(
            f"imagery_mode must be one of {list(IMAGERY_MODES)} or a list of them (got {value!r})"
        )
    unknown = [mode for mode in raw if mode not in IMAGERY_MODES]
    if unknown:
        raise InvalidImageryModeError(
            f"imagery_mode must be one of {list(IMAGERY_MODES)} or a list of them "
            f"(got {unknown[0]!r})"
        )
    return tuple(mode for mode in IMAGERY_MODES if mode in raw)


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
