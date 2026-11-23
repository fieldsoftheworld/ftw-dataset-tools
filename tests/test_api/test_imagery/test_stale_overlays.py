"""Tests for redrawing chip overlays older than their masks on a resumed imagery run."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - used at runtime in helpers
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pystac
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from ftw_dataset_tools.api.imagery import download_workflow
from ftw_dataset_tools.api.imagery.catalog_ops import refresh_stale_overlays
from ftw_dataset_tools.api.imagery.chip_preview import (
    PREVIEW_SLOT_FIELD,
    overlay_is_stale,
    refresh_chip_preview,
)
from ftw_dataset_tools.api.stac_items import write_item

if TYPE_CHECKING:
    import pytest

CHIP = "chip_a"
_BBOX = [10.0, 50.0, 10.01, 50.01]
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[[10.0, 50.0], [10.01, 50.0], [10.01, 50.01], [10.0, 50.01], [10.0, 50.0]]],
}


def _write_mask(chip_dir: Path) -> Path:
    size = 32
    arr = np.zeros((size, size), dtype="uint8")
    arr[5:20, 5:20] = 1
    arr[20:24, 5:20] = 2
    path = chip_dir / f"{CHIP}_semantic_3_class.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=1,
        dtype="uint8",
        crs="EPSG:32633",
        transform=from_bounds(501000, 5001000, 501320, 5001320, size, size),
    ) as dst:
        dst.write(arr, 1)
    return path


def _item(item_id: str) -> pystac.Item:
    return pystac.Item(
        id=item_id,
        geometry=_GEOMETRY,
        bbox=_BBOX,
        datetime=datetime(2024, 5, 1, tzinfo=UTC),
        properties={},
    )


def _chip(tmp_path: Path, *, mask: bool = True, base: bool = True) -> Path:
    """A downloaded chip: parent and planting child items, the image, its preview."""
    chip_dir = tmp_path / "chips" / "33TXM" / CHIP
    chip_dir.mkdir(parents=True)
    if mask:
        _write_mask(chip_dir)
    if base:
        Image.new("RGB", (32, 32), (90, 120, 60)).save(
            chip_dir / f"{CHIP}_planting_image_s2.webp", "WEBP"
        )
    (chip_dir / f"{CHIP}_planting_image_s2.tif").write_bytes(b"tif")

    child = _item(f"{CHIP}_planting_s2")
    child.add_asset("image", pystac.Asset(href=f"./{CHIP}_planting_image_s2.tif"))
    write_item(child, chip_dir / f"{child.id}.json")
    write_item(_item(CHIP), chip_dir / f"{CHIP}.json")
    return chip_dir


def _old_overlay(chip_dir: Path) -> Path:
    """An overlay drawn before the mask was rebuilt, recorded as the chip preview."""
    overlay = chip_dir / f"{CHIP}_overlay.webp"
    overlay.write_bytes(b"old overlay")
    item = pystac.Item.from_file(str(chip_dir / f"{CHIP}.json"))
    refresh_chip_preview(item, chip_dir, "planting")
    write_item(item, chip_dir / f"{CHIP}.json")
    mask_mtime = (chip_dir / f"{CHIP}_semantic_3_class.tif").stat().st_mtime
    os.utime(overlay, (mask_mtime - 100, mask_mtime - 100))
    return overlay


def _resume(tmp_path: Path) -> download_workflow.DownloadWorkflowResult:
    """A resumed download over the catalog, which must not download anything."""
    with (
        patch.object(download_workflow, "on_reference_grid", return_value=True),
        patch.object(
            download_workflow, "download_and_clip_scene", side_effect=AssertionError("download")
        ),
    ):
        result = download_workflow.download_imagery_for_catalog(tmp_path, workers=1)
    assert result.skipped == 1
    assert result.failed == 0
    return result


def _thumbnail(chip_dir: Path) -> pystac.Asset:
    return pystac.Item.from_file(str(chip_dir / f"{CHIP}.json")).assets["thumbnail"]


class TestResumedDownloadRedrawsStaleOverlays:
    def test_an_overlay_older_than_its_mask_is_redrawn(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir)

        result = _resume(tmp_path)

        assert result.previews_redrawn == 1
        assert overlay.read_bytes() != b"old overlay"
        assert not overlay_is_stale(chip_dir, CHIP)
        thumbnail = _thumbnail(chip_dir)
        assert thumbnail.href == f"./{CHIP}_overlay.webp"
        assert thumbnail.extra_fields["file:size"] == overlay.stat().st_size
        assert thumbnail.extra_fields[PREVIEW_SLOT_FIELD] == "planting"

    def test_an_overlay_newer_than_its_mask_is_left_alone(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir)
        mask_mtime = (chip_dir / f"{CHIP}_semantic_3_class.tif").stat().st_mtime
        os.utime(overlay, (mask_mtime + 100, mask_mtime + 100))
        item_json = (chip_dir / f"{CHIP}.json").read_text()

        result = _resume(tmp_path)

        assert result.previews_redrawn == 0
        assert overlay.read_bytes() == b"old overlay"
        assert (chip_dir / f"{CHIP}.json").read_text() == item_json

    def test_no_overlay_is_drawn_without_a_mask(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path, mask=False)

        result = _resume(tmp_path)

        assert result.previews_redrawn == 0
        assert not (chip_dir / f"{CHIP}_overlay.webp").exists()

    def test_a_missing_overlay_is_drawn_from_the_preview_image(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)

        result = _resume(tmp_path)

        assert result.previews_redrawn == 1
        overlay = chip_dir / f"{CHIP}_overlay.webp"
        assert overlay.stat().st_size > 0
        thumbnail = _thumbnail(chip_dir)
        assert thumbnail.href == f"./{CHIP}_overlay.webp"
        assert thumbnail.extra_fields[PREVIEW_SLOT_FIELD] == "planting"

    def test_no_thumbnails_means_no_redraw(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir)

        with patch.object(download_workflow, "on_reference_grid", return_value=True):
            result = download_workflow.download_imagery_for_catalog(
                tmp_path, workers=1, generate_thumbnails=False
            )

        assert result.previews_redrawn == 0
        assert overlay.read_bytes() == b"old overlay"


class TestRefreshStaleOverlays:
    def test_a_missing_overlay_without_a_preview_image_is_not_stale(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path, base=False)

        assert not overlay_is_stale(chip_dir, CHIP)
        assert refresh_stale_overlays(tmp_path) == 0

    def test_an_unreadable_item_is_logged_not_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        chip_dir = _chip(tmp_path)
        (chip_dir / f"{CHIP}.json").write_text("{ not json")

        assert refresh_stale_overlays(tmp_path) == 0
        assert "could not redraw the chip preview" in caplog.text
