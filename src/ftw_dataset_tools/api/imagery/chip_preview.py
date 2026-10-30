"""The chip preview: which imagery slot drew it, and the ``thumbnail`` asset.

When a slot draws the preview, the slot is recorded on the thumbnail asset
(``ftw:preview_slot``) and read back from there, so no path has to guess the
preview's source from the files on disk. A scene preview (planting, else harvest)
wins over the Q3 mosaic's.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pystac

from ftw_dataset_tools.api.assets import add_file_info
from ftw_dataset_tools.api.imagery.slots import THUMBNAIL_SLOTS, slot_title
from ftw_dataset_tools.api.imagery.thumbnails import (
    PREVIEW_EXTENSIONS,
    PREVIEW_SUFFIX,
    REFERENCE_MASK_SUFFIX,
    ThumbnailError,
    generate_overlay_thumbnail,
    preview_media_type,
)

if TYPE_CHECKING:
    from pathlib import Path

__all__ = [
    "PREVIEW_SLOT_FIELD",
    "preview_slot",
    "redraw_chip_preview",
    "refresh_chip_preview",
    "takes_preview",
]

#: Thumbnail asset field naming the slot the chip preview was drawn from.
PREVIEW_SLOT_FIELD = "ftw:preview_slot"

OVERLAY_TITLE = "Chip preview with field overlay"


def _existing(chip_dir: Path, stem: str) -> Path | None:
    """``{stem}.webp``, or a pre-WebP ``.jpg`` of it, if one is on disk."""
    for ext in PREVIEW_EXTENSIONS:
        path = chip_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def _best_slot_preview(chip_dir: Path, chip_id: str) -> str | None:
    """The first of ``THUMBNAIL_SLOTS`` with its own preview image on disk."""
    return next(
        (s for s in THUMBNAIL_SLOTS if _existing(chip_dir, f"{chip_id}_{s}_image_s2")), None
    )


def preview_slot(item: pystac.Item, chip_dir: Path) -> str | None:
    """The slot the chip preview was drawn from, or None when the chip has none.

    Read from the item's thumbnail asset, else from the item JSON on disk (a rebuilt
    item starts without assets). A preview with no recorded slot predates the record;
    older ftwd drew those from a scene, so it counts as the planting scene's. With no
    preview recorded at all, it is the best slot preview on disk.
    """
    thumbnail = item.assets.get("thumbnail")
    if thumbnail is not None:
        return thumbnail.extra_fields.get(PREVIEW_SLOT_FIELD, THUMBNAIL_SLOTS[0])
    on_disk = chip_dir / f"{item.id}.json"
    if on_disk.exists():
        recorded = (json.loads(on_disk.read_text()).get("assets") or {}).get("thumbnail")
        if recorded is not None:
            return recorded.get(PREVIEW_SLOT_FIELD, THUMBNAIL_SLOTS[0])
    if _existing(chip_dir, f"{item.id}_overlay") is not None:
        return THUMBNAIL_SLOTS[0]
    return _best_slot_preview(chip_dir, item.id)


def takes_preview(item: pystac.Item, chip_dir: Path, slot: str) -> bool:
    """Whether a new preview of ``slot`` replaces the chip's current one."""
    if slot not in THUMBNAIL_SLOTS:
        return False
    current = preview_slot(item, chip_dir)
    return current is None or THUMBNAIL_SLOTS.index(slot) <= THUMBNAIL_SLOTS.index(current)


def refresh_chip_preview(
    item: pystac.Item,
    chip_dir: Path,
    slot: str | None = None,
    *,
    redraw: bool = False,
    checksums: bool = False,
) -> str | None:
    """Set the chip's ``thumbnail`` asset, recording the slot it comes from.

    ``slot`` is the slot whose preview it now is; None keeps the current
    :func:`preview_slot`. With ``redraw`` the field overlay is first drawn afresh
    from that slot's preview and the chip's reference mask (or removed, when it
    cannot be drawn). The overlay is preferred; the slot's plain preview is the
    fallback.

    Returns:
        The preview slot, or None when the chip has no preview (and no thumbnail).
    """
    slot = slot or preview_slot(item, chip_dir)
    item.assets.pop("thumbnail", None)
    if slot is None:
        return None
    base = _existing(chip_dir, f"{item.id}_{slot}_image_s2")
    if redraw and base is not None:
        _redraw_overlay(chip_dir, item.id, base)
    # A WebP beats a pre-WebP JPEG left behind; then the overlay beats the plain preview.
    stems = (f"{item.id}_overlay", f"{item.id}_{slot}_image_s2")
    target = next(
        (
            chip_dir / f"{stem}{ext}"
            for ext in PREVIEW_EXTENSIONS
            for stem in stems
            if (chip_dir / f"{stem}{ext}").exists()
        ),
        None,
    )
    if target is None:
        return None
    is_overlay = target.stem == f"{item.id}_overlay"
    asset = pystac.Asset(
        href=f"./{target.name}",
        media_type=preview_media_type(target.name),
        title=OVERLAY_TITLE if is_overlay else f"Chip preview ({slot_title(slot).lower()})",
        roles=["thumbnail"],
        extra_fields={PREVIEW_SLOT_FIELD: slot},
    )
    item.add_asset("thumbnail", asset)
    add_file_info(asset, target, checksum=checksums)
    return slot


def redraw_chip_preview(item: pystac.Item, chip_dir: Path) -> str | None:
    """Draw the chip preview afresh from the best slot preview left on disk.

    For when the slot that drew it is gone: removes the old overlay and thumbnail,
    then draws from the first of ``THUMBNAIL_SLOTS`` that still has a preview image.
    """
    for ext in PREVIEW_EXTENSIONS:
        (chip_dir / f"{item.id}_overlay{ext}").unlink(missing_ok=True)
    item.assets.pop("thumbnail", None)
    best = _best_slot_preview(chip_dir, item.id)
    return None if best is None else refresh_chip_preview(item, chip_dir, best, redraw=True)


def _redraw_overlay(chip_dir: Path, chip_id: str, base: Path) -> None:
    for ext in PREVIEW_EXTENSIONS:
        (chip_dir / f"{chip_id}_overlay{ext}").unlink(missing_ok=True)
    mask = chip_dir / f"{chip_id}{REFERENCE_MASK_SUFFIX}"
    if not mask.exists():
        return
    overlay = chip_dir / f"{chip_id}_overlay{PREVIEW_SUFFIX}"
    try:
        generate_overlay_thumbnail(base, mask, overlay)
    except ThumbnailError:
        overlay.unlink(missing_ok=True)
