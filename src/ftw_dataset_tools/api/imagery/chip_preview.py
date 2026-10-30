"""The chip preview: which imagery slot it is drawn from, and the ``thumbnail`` asset.

Every path that touches the preview (download, remote preview, catalog rebuild,
WebP conversion, clearing a selection) goes through these two functions, so they
all agree on one rule.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pystac

from ftw_dataset_tools.api.assets import add_file_info
from ftw_dataset_tools.api.imagery.slots import SCENE_SLOTS, THUMBNAIL_SLOTS, slot_title
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

__all__ = ["preview_slot", "refresh_chip_preview"]

OVERLAY_TITLE = "Chip preview with field overlay"


def _existing(chip_dir: Path, stem: str) -> Path | None:
    """``{stem}.webp``, or a pre-WebP ``.jpg`` of it, if one is on disk."""
    for ext in PREVIEW_EXTENSIONS:
        path = chip_dir / f"{stem}{ext}"
        if path.exists():
            return path
    return None


def preview_slot(chip_dir: Path, chip_id: str) -> str | None:
    """The slot the chip preview is drawn from: the first of ``THUMBNAIL_SLOTS`` with a preview.

    A slot has a preview once its ``{chip}_{slot}_image_s2.webp`` is on disk (written by
    the download, or by the remote preview for a scene). So a scene's preview wins
    over the Q3 mosaic's whatever order they arrive in, and Q3 stands in while no
    scene has one.

    Older ftwd drew the field overlay from a scene without keeping the scene's own
    preview. Such an overlay counts as the preview of the selected scene it sits
    next to (or of planting, with no scene selected), so mosaics added later never
    replace or remove it.
    """
    legacy_overlay = _existing(chip_dir, f"{chip_id}_overlay") is not None
    for slot in THUMBNAIL_SLOTS:
        if _existing(chip_dir, f"{chip_id}_{slot}_image_s2") is not None:
            return slot
        scene_selected = (chip_dir / f"{chip_id}_{slot}_s2.json").exists()
        if legacy_overlay and slot in SCENE_SLOTS and scene_selected:
            return slot
    return THUMBNAIL_SLOTS[0] if legacy_overlay else None


def refresh_chip_preview(
    item: pystac.Item, chip_dir: Path, *, redraw: bool = False, checksums: bool = False
) -> str | None:
    """Set the chip's ``thumbnail`` asset from :func:`preview_slot`'s preview.

    With ``redraw`` the field overlay is first drawn afresh from that slot's preview
    and the chip's reference mask (or removed, when it cannot be drawn), so it shows
    the current preview slot. Without it, whatever overlay is on disk is attached.
    The overlay is preferred; the slot's plain preview is the fallback.

    Returns:
        The preview slot, or None when the chip has no preview (and no thumbnail).
    """
    item.assets.pop("thumbnail", None)
    slot = preview_slot(chip_dir, item.id)
    if slot is None:
        return None
    base = _existing(chip_dir, f"{item.id}_{slot}_image_s2")
    if redraw and base is not None:
        _redraw_overlay(chip_dir, item.id, base)
    # A WebP beats a pre-WebP JPEG left behind; then the overlay beats the plain preview.
    stems = (f"{item.id}_overlay", f"{item.id}_{slot}_image_s2")
    target = next(
        chip_dir / f"{stem}{ext}"
        for ext in PREVIEW_EXTENSIONS
        for stem in stems
        if (chip_dir / f"{stem}{ext}").exists()
    )
    is_overlay = target.stem == f"{item.id}_overlay"
    item.add_asset(
        "thumbnail",
        pystac.Asset(
            href=f"./{target.name}",
            media_type=preview_media_type(target.name),
            title=OVERLAY_TITLE if is_overlay else f"Chip preview ({slot_title(slot).lower()})",
            roles=["thumbnail"],
        ),
    )
    add_file_info(item.assets["thumbnail"], target, checksum=checksums)
    return slot


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
