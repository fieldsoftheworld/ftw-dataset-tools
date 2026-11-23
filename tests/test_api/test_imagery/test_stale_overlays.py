"""Tests for redrawing chip overlays older than their masks on a resumed imagery run."""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - used at runtime in helpers
from typing import TYPE_CHECKING
from unittest.mock import patch

import numpy as np
import pystac
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from ftw_dataset_tools.api.imagery import chip_preview, download_workflow, overlay_refresh
from ftw_dataset_tools.api.imagery.chip_preview import PREVIEW_SLOT_FIELD, refresh_chip_preview
from ftw_dataset_tools.api.imagery.overlay_refresh import overlay_is_stale, redraw_stale_overlays
from ftw_dataset_tools.api.imagery.thumbnails import ThumbnailError
from ftw_dataset_tools.api.stac_items import write_item

if TYPE_CHECKING:
    import pytest

CHIP = "chip_a"
_BBOX = [10.0, 50.0, 10.01, 50.01]
_GEOMETRY = {
    "type": "Polygon",
    "coordinates": [[[10.0, 50.0], [10.01, 50.0], [10.01, 50.01], [10.0, 50.01], [10.0, 50.0]]],
}


def _write_mask(chip_dir: Path, chip_id: str) -> Path:
    size = 32
    arr = np.zeros((size, size), dtype="uint8")
    arr[5:20, 5:20] = 1
    arr[20:24, 5:20] = 2
    path = chip_dir / f"{chip_id}_semantic_3_class.tif"
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


def _chip(tmp_path: Path, chip_id: str = CHIP, *, mask: bool = True, base: bool = True) -> Path:
    """A downloaded chip: parent and planting child items, the image, its preview."""
    chip_dir = tmp_path / "chips" / "33TXM" / chip_id
    chip_dir.mkdir(parents=True)
    if base:
        Image.new("RGB", (32, 32), (90, 120, 60)).save(
            chip_dir / f"{chip_id}_planting_image_s2.webp", "WEBP"
        )
    (chip_dir / f"{chip_id}_planting_image_s2.tif").write_bytes(b"tif")
    if mask:
        _write_mask(chip_dir, chip_id)

    child = _item(f"{chip_id}_planting_s2")
    child.add_asset("image", pystac.Asset(href=f"./{chip_id}_planting_image_s2.tif"))
    write_item(child, chip_dir / f"{child.id}.json")
    write_item(_item(chip_id), chip_dir / f"{chip_id}.json")
    return chip_dir


def _age(path: Path, chip_dir: Path, chip_id: str = CHIP, seconds: float = 100) -> None:
    """Make ``path`` older than the chip's mask."""
    mask_mtime = (chip_dir / f"{chip_id}_semantic_3_class.tif").stat().st_mtime
    os.utime(path, (mask_mtime - seconds, mask_mtime - seconds))


def _old_overlay(
    chip_dir: Path, chip_id: str = CHIP, slot: str = "planting", *, checksums: bool = False
) -> Path:
    """An overlay drawn before the mask was rebuilt, recorded as the chip preview."""
    overlay = chip_dir / f"{chip_id}_overlay.webp"
    overlay.write_bytes(b"old overlay")
    item = pystac.Item.from_file(str(chip_dir / f"{chip_id}.json"))
    refresh_chip_preview(item, chip_dir, slot, checksums=checksums)
    write_item(item, chip_dir / f"{chip_id}.json")
    _age(overlay, chip_dir, chip_id)
    return overlay


def _resume(tmp_path: Path, workers: int = 1, **kwargs) -> download_workflow.DownloadWorkflowResult:
    """A resumed download over the catalog, which must not download anything."""
    with (
        patch.object(download_workflow, "on_reference_grid", return_value=True),
        patch.object(
            download_workflow, "download_and_clip_scene", side_effect=AssertionError("download")
        ),
    ):
        result = download_workflow.download_imagery_for_catalog(tmp_path, workers=workers, **kwargs)
    assert result.failed == 0
    return result


def _thumbnail(chip_dir: Path, chip_id: str = CHIP) -> pystac.Asset:
    return pystac.Item.from_file(str(chip_dir / f"{chip_id}.json")).assets["thumbnail"]


def _failing_draw(*_args, **_kwargs):
    raise ThumbnailError("cannot draw")


def _concurrent_draws(parties: int):
    """``overlay_refresh._draw`` that only finishes once ``parties`` draws run at once.

    Drawn one at a time, the barrier times out and every draw fails.
    """
    barrier = threading.Barrier(parties, timeout=10)
    draw = overlay_refresh._draw

    def wait_then_draw(task):
        barrier.wait()
        return draw(task)

    return patch.object(overlay_refresh, "_draw", wait_then_draw)


class TestResumedDownloadRedrawsStaleOverlays:
    def test_an_overlay_older_than_its_mask_is_redrawn(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir)

        result = _resume(tmp_path)

        assert (result.skipped, result.previews_redrawn) == (1, 1)
        assert overlay.read_bytes() != b"old overlay"
        assert not overlay_is_stale(chip_dir, CHIP)
        thumbnail = _thumbnail(chip_dir)
        assert thumbnail.href == f"./{CHIP}_overlay.webp"
        assert thumbnail.extra_fields["file:size"] == overlay.stat().st_size
        assert thumbnail.extra_fields[PREVIEW_SLOT_FIELD] == "planting"

    def test_an_overlay_newer_than_its_mask_is_left_alone(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir)
        mask_mtime = (chip_dir / f"{CHIP}_semantic_3_class.tif").stat().st_mtime
        os.utime(overlay, (mask_mtime + 100, mask_mtime + 100))
        item_json = (chip_dir / f"{CHIP}.json").read_text()

        with caplog.at_level("INFO"):
            result = _resume(tmp_path)

        assert result.previews_redrawn == 0
        assert overlay.read_bytes() == b"old overlay"
        assert (chip_dir / f"{CHIP}.json").read_text() == item_json
        # Nothing redrawn, nothing logged about it.
        assert "Redrew" not in caplog.text

    def test_no_overlay_is_drawn_without_a_mask(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path, mask=False)

        result = _resume(tmp_path)

        assert result.previews_redrawn == 0
        assert not (chip_dir / f"{CHIP}_overlay.webp").exists()

    def test_a_missing_overlay_is_drawn_when_the_mask_is_newer(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        _age(chip_dir / f"{CHIP}_planting_image_s2.webp", chip_dir)

        result = _resume(tmp_path)

        assert result.previews_redrawn == 1
        assert (chip_dir / f"{CHIP}_overlay.webp").stat().st_size > 0
        thumbnail = _thumbnail(chip_dir)
        assert thumbnail.href == f"./{CHIP}_overlay.webp"
        assert thumbnail.extra_fields[PREVIEW_SLOT_FIELD] == "planting"

    def test_a_missing_overlay_drawn_from_after_the_mask_is_not_retried(
        self, tmp_path: Path
    ) -> None:
        """The image is newer than the mask, so its overlay was already attempted."""
        chip_dir = _chip(tmp_path)
        mask = chip_dir / f"{CHIP}_semantic_3_class.tif"
        os.utime(mask, (mask.stat().st_mtime - 100,) * 2)

        assert _resume(tmp_path).previews_redrawn == 0
        assert not (chip_dir / f"{CHIP}_overlay.webp").exists()

    def test_no_thumbnails_means_no_redraw(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir)

        result = _resume(tmp_path, generate_thumbnails=False)

        assert result.previews_redrawn == 0
        assert overlay.read_bytes() == b"old overlay"

    def test_checksums_are_kept_when_the_old_thumbnail_had_one(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        _old_overlay(chip_dir, checksums=True)
        old_checksum = _thumbnail(chip_dir).extra_fields["file:checksum"]

        _resume(tmp_path)

        new_checksum = _thumbnail(chip_dir).extra_fields.get("file:checksum")
        assert new_checksum is not None
        assert new_checksum != old_checksum

    def test_redraws_run_on_the_pool_and_items_are_written_on_the_caller(
        self, tmp_path: Path
    ) -> None:
        chips = [_chip(tmp_path, f"chip_{i}") for i in range(4)]
        for i, chip_dir in enumerate(chips):
            _old_overlay(chip_dir, f"chip_{i}")
        draw_threads: set[int] = set()
        write_threads: set[int] = set()
        draw, write = overlay_refresh._draw, overlay_refresh.write_item

        def record_draw(task):
            draw_threads.add(threading.get_ident())
            return draw(task)

        def record_write(*args, **kwargs):
            write_threads.add(threading.get_ident())
            return write(*args, **kwargs)

        barrier = threading.Barrier(4, timeout=10)

        def concurrent_draw(task):
            barrier.wait()
            return record_draw(task)

        with (
            patch.object(overlay_refresh, "_draw", concurrent_draw),
            patch.object(overlay_refresh, "write_item", record_write),
        ):
            result = redraw_stale_overlays([c / f"{c.name}.json" for c in chips], workers=4)

        assert (result.redrawn, result.failed) == (4, 0)
        assert threading.get_ident() not in draw_threads
        assert write_threads == {threading.get_ident()}

    def test_the_download_workers_reach_the_redraw(self, tmp_path: Path) -> None:
        for chip_id in ("chip_a", "chip_b"):
            _old_overlay(_chip(tmp_path, chip_id), chip_id)

        with _concurrent_draws(2):
            result = _resume(tmp_path, workers=2)

        assert (result.previews_redrawn, result.previews_redraw_failed) == (2, 0)


class TestMissingSlotImage:
    def test_falls_back_to_the_best_slot_left(self, tmp_path: Path) -> None:
        """The recorded harvest preview is gone; the planting one draws the overlay."""
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir, slot="harvest")
        assert _thumbnail(chip_dir).extra_fields[PREVIEW_SLOT_FIELD] == "harvest"

        result = _resume(tmp_path)

        assert result.previews_redrawn == 1
        assert overlay.read_bytes() != b"old overlay"
        assert _thumbnail(chip_dir).extra_fields[PREVIEW_SLOT_FIELD] == "planting"
        assert not overlay_is_stale(chip_dir, CHIP)

    def test_with_no_image_left_the_old_overlay_goes(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path, base=False)
        overlay = _old_overlay(chip_dir)

        result = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert (result.redrawn, result.failed) == (0, 0)
        assert not overlay.exists()
        item = pystac.Item.from_file(str(chip_dir / f"{CHIP}.json"))
        assert "thumbnail" not in item.assets
        assert not overlay_is_stale(chip_dir, CHIP)


class TestRedrawFailure:
    def test_is_counted_once_and_not_retried(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        _old_overlay(chip_dir)

        with patch.object(chip_preview, "generate_overlay_thumbnail", _failing_draw):
            first = _resume(tmp_path)
        assert (first.skipped, first.previews_redrawn, first.previews_redraw_failed) == (1, 0, 1)
        thumbnail = _thumbnail(chip_dir)
        assert thumbnail.href == f"./{CHIP}_planting_image_s2.webp"

        with patch.object(chip_preview, "generate_overlay_thumbnail") as draw:
            second = _resume(tmp_path)
        draw.assert_not_called()
        assert (second.previews_redrawn, second.previews_redraw_failed) == (0, 0)

    def test_a_rebuilt_mask_retries_it(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        _old_overlay(chip_dir)
        with patch.object(chip_preview, "generate_overlay_thumbnail", _failing_draw):
            _resume(tmp_path)

        # Rebuild the mask, as often as the clock's granularity needs for it to be
        # newer than the image the failed draw marked; no mtime is set by hand.
        base = chip_dir / f"{CHIP}_planting_image_s2.webp"
        mask = _write_mask(chip_dir, CHIP)
        deadline = time.monotonic() + 5
        while mask.stat().st_mtime <= base.stat().st_mtime and time.monotonic() < deadline:
            time.sleep(0.01)
            _write_mask(chip_dir, CHIP)

        assert _resume(tmp_path).previews_redrawn == 1

    def test_any_draw_error_falls_back_to_the_plain_preview_once(self, tmp_path: Path) -> None:
        """Not only ThumbnailError: the old overlay is already gone when the draw raises."""
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir)

        def boom(*_args, **_kwargs):
            raise ValueError("boom")

        with patch.object(chip_preview, "generate_overlay_thumbnail", boom):
            first = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)
            second = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert (first.redrawn, first.failed) == (0, 1)
        assert (second.redrawn, second.failed) == (0, 0)
        assert not overlay.exists()
        thumbnail = _thumbnail(chip_dir)
        assert thumbnail.href == f"./{CHIP}_planting_image_s2.webp"
        assert thumbnail.extra_fields["file:size"] == (
            (chip_dir / f"{CHIP}_planting_image_s2.webp").stat().st_size
        )

    def test_an_unreadable_item_is_logged_not_raised(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        chip_dir = _chip(tmp_path)
        _age(chip_dir / f"{CHIP}_planting_image_s2.webp", chip_dir)
        (chip_dir / f"{CHIP}.json").write_text("{ not json")

        result = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert (result.redrawn, result.failed) == (0, 1)
        assert "could not redraw the chip preview" in caplog.text


class TestOverlayIsStale:
    def test_a_missing_overlay_without_a_preview_image_is_not_stale(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path, base=False)

        assert not overlay_is_stale(chip_dir, CHIP)
