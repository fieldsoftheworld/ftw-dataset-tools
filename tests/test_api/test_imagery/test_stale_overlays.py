"""Tests for redrawing chip overlays older than their masks on a resumed imagery run."""

from __future__ import annotations

import os
import threading
import time
from datetime import UTC, datetime
from pathlib import Path  # noqa: TC003 - used at runtime in helpers
from unittest.mock import patch

import numpy as np
import pystac
import pytest
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from ftw_dataset_tools.api.imagery import download_workflow, overlay_refresh
from ftw_dataset_tools.api.imagery.chip_preview import PREVIEW_SLOT_FIELD, refresh_chip_preview
from ftw_dataset_tools.api.imagery.overlay_refresh import overlay_is_stale, redraw_stale_overlays
from ftw_dataset_tools.api.imagery.thumbnails import ThumbnailError
from ftw_dataset_tools.api.stac_items import write_item

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


def _failing_value_error(*_args, **_kwargs):
    raise ValueError("boom")


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
        """Five at once is more than the default workers, so only ``workers=5`` gets there."""
        for i in range(5):
            _old_overlay(_chip(tmp_path, f"chip_{i}"), f"chip_{i}")

        with _concurrent_draws(5):
            result = _resume(tmp_path, workers=5)

        assert (result.previews_redrawn, result.previews_redraw_failed) == (5, 0)


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

    def test_with_no_image_left_the_old_overlay_goes(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        chip_dir = _chip(tmp_path, base=False)
        overlay = _old_overlay(chip_dir)

        with caplog.at_level("INFO"):
            result = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert (result.redrawn, result.failed, result.removed) == (0, 0, 1)
        assert "1 removed (no preview image left)" in caplog.text
        assert not overlay.exists()
        item = pystac.Item.from_file(str(chip_dir / f"{CHIP}.json"))
        assert "thumbnail" not in item.assets
        assert not overlay_is_stale(chip_dir, CHIP)


class TestRedrawFailure:
    def test_is_counted_once_and_not_retried(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        _old_overlay(chip_dir)

        with patch.object(overlay_refresh, "generate_overlay_thumbnail", _failing_draw):
            first = _resume(tmp_path)
        assert (first.skipped, first.previews_redrawn, first.previews_redraw_failed) == (1, 0, 1)
        thumbnail = _thumbnail(chip_dir)
        assert thumbnail.href == f"./{CHIP}_planting_image_s2.webp"

        with patch.object(overlay_refresh, "generate_overlay_thumbnail") as draw:
            second = _resume(tmp_path)
        draw.assert_not_called()
        assert (second.previews_redrawn, second.previews_redraw_failed) == (0, 0)

    def test_a_rebuilt_mask_retries_it(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        _old_overlay(chip_dir)
        with patch.object(overlay_refresh, "generate_overlay_thumbnail", _failing_draw):
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

        with patch.object(overlay_refresh, "generate_overlay_thumbnail", boom):
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


class TestInterrupts:
    """An interrupt between drawing and writing the item leaves every chip repairable."""

    def test_an_interrupt_before_the_item_write_leaves_the_chips_stale(
        self, tmp_path: Path
    ) -> None:
        chips = [_chip(tmp_path, f"chip_{i}") for i in range(4)]
        for chip_dir in chips:
            _old_overlay(chip_dir, chip_dir.name)
        items = [c / f"{c.name}.json" for c in chips]

        with (
            patch.object(overlay_refresh, "write_item", side_effect=KeyboardInterrupt),
            pytest.raises(KeyboardInterrupt),
        ):
            redraw_stale_overlays(items, workers=4)

        assert all(overlay_is_stale(c, c.name) for c in chips)
        again = redraw_stale_overlays(items, workers=4)
        assert (again.redrawn, again.failed) == (4, 0)
        for chip_dir in chips:
            overlay = chip_dir / f"{chip_dir.name}_overlay.webp"
            thumbnail = _thumbnail(chip_dir, chip_dir.name)
            assert thumbnail.extra_fields["file:size"] == overlay.stat().st_size
            assert not list(chip_dir.glob(".*"))

    def test_a_leftover_is_removed_when_the_item_cannot_be_read(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        _old_overlay(chip_dir)
        leftover = chip_dir / f".{CHIP}_overlay.tmp.webp"
        leftover.write_bytes(b"half drawn")
        (chip_dir / f"{CHIP}.json").write_text("[]")

        result = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert result.failed == 1
        assert not leftover.exists()

    def test_a_leftover_is_removed_with_an_overlay_that_has_no_image(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path, base=False)
        _old_overlay(chip_dir)
        leftover = chip_dir / f".{CHIP}_overlay.tmp.webp"
        leftover.write_bytes(b"half drawn")

        result = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert result.removed == 1
        assert not leftover.exists()

    def test_a_leftover_temporary_file_is_removed(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir)
        mask_mtime = (chip_dir / f"{CHIP}_semantic_3_class.tif").stat().st_mtime
        os.utime(overlay, (mask_mtime + 100, mask_mtime + 100))
        leftover = chip_dir / f".{CHIP}_overlay.tmp.webp"
        leftover.write_bytes(b"half drawn")

        redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert not leftover.exists()


class TestClockSkew:
    def test_a_mask_dated_in_the_future_is_redrawn_once(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        _old_overlay(chip_dir)
        mask = chip_dir / f"{CHIP}_semantic_3_class.tif"
        future = time.time() + 3600
        os.utime(mask, (future, future))

        first = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)
        second = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert (first.redrawn, second.redrawn) == (1, 0)

    def test_a_failure_under_a_future_mask_is_not_retried(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        _old_overlay(chip_dir)
        mask = chip_dir / f"{CHIP}_semantic_3_class.tif"
        future = time.time() + 3600
        os.utime(mask, (future, future))

        with patch.object(overlay_refresh, "generate_overlay_thumbnail", _failing_draw):
            first = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)
        second = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert (first.failed, second.failed, second.redrawn) == (1, 0, 0)


class TestRecordedSlotIsNotTheBest:
    def test_a_failure_is_not_retried(self, tmp_path: Path) -> None:
        """The overlay came from harvest; staleness reads planting, the best slot."""
        chip_dir = _chip(tmp_path)
        Image.new("RGB", (32, 32), (60, 90, 120)).save(
            chip_dir / f"{CHIP}_harvest_image_s2.webp", "WEBP"
        )
        _age(chip_dir / f"{CHIP}_harvest_image_s2.webp", chip_dir)
        _age(chip_dir / f"{CHIP}_planting_image_s2.webp", chip_dir)
        _old_overlay(chip_dir, slot="harvest")

        with patch.object(overlay_refresh, "generate_overlay_thumbnail", _failing_draw):
            first = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)
        assert _thumbnail(chip_dir).href == f"./{CHIP}_harvest_image_s2.webp"
        with patch.object(overlay_refresh, "generate_overlay_thumbnail") as draw:
            second = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        draw.assert_not_called()
        assert (first.failed, second.failed) == (1, 0)


class TestPerChipErrors:
    def test_a_stat_error_on_one_chip_does_not_stop_the_others(self, tmp_path: Path) -> None:
        chips = [_chip(tmp_path, f"chip_{i}") for i in range(2)]
        for chip_dir in chips:
            _old_overlay(chip_dir, chip_dir.name)
        real = overlay_refresh.overlay_is_stale

        def denied(chip_dir, chip_id):
            if chip_id == "chip_0":
                raise PermissionError("denied")
            return real(chip_dir, chip_id)

        with patch.object(overlay_refresh, "overlay_is_stale", denied):
            result = redraw_stale_overlays([c / f"{c.name}.json" for c in chips], workers=1)

        assert (result.redrawn, result.failed) == (1, 1)

    def test_a_mark_error_after_a_failed_draw_is_counted(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        _old_overlay(chip_dir)

        with (
            patch.object(overlay_refresh, "generate_overlay_thumbnail", _failing_draw),
            patch.object(overlay_refresh.os, "utime", side_effect=OSError("read-only")),
        ):
            result = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert (result.redrawn, result.failed) == (0, 1)

    def test_the_progress_bar_starts_before_planning_and_counts_every_chip(
        self, tmp_path: Path
    ) -> None:
        chips = [_chip(tmp_path, f"chip_{i}") for i in range(3)]
        for chip_dir in chips[:2]:
            _old_overlay(chip_dir, chip_dir.name)
        # The third chip's preview postdates its mask: its overlay was already tried.
        _age(chips[2] / "chip_2_planting_image_s2.webp", chips[2], "chip_2", seconds=-100)
        events: list[str] = []
        plan = overlay_refresh._plan

        def record_plan(item_path):
            events.append("plan")
            return plan(item_path)

        with (
            patch.object(overlay_refresh, "tqdm") as bar,
            patch.object(overlay_refresh, "_plan", record_plan),
        ):
            bar.side_effect = lambda **_kwargs: events.append("bar") or bar.return_value
            redraw_stale_overlays([c / f"{c.name}.json" for c in chips], workers=2)

        assert events[0] == "bar"
        assert bar.call_args.kwargs["total"] == 3
        assert bar.return_value.update.call_count == 3
        bar.return_value.close.assert_called_once()

    def test_a_batch_is_released_before_the_next_is_planned(self, tmp_path: Path) -> None:
        """At most one batch of items is alive while a chip is planned."""
        import gc
        import weakref

        class Probe:
            pass

        alive: weakref.WeakSet = weakref.WeakSet()
        peak = [0]

        def plan(item_path):
            gc.collect()
            peak[0] = max(peak[0], len(alive))
            probe = Probe()
            alive.add(probe)
            return overlay_refresh._RedrawTask(
                item=probe, item_path=item_path, slot=None, checksums=False
            )

        with (
            patch.object(overlay_refresh, "PLAN_BATCH_SIZE", 3),
            patch.object(overlay_refresh, "overlay_is_stale", lambda *_: True),
            patch.object(overlay_refresh, "_plan", plan),
            patch.object(overlay_refresh, "_draw", lambda _task: None),
            patch.object(overlay_refresh, "_settle", lambda _outcome: "redrawn"),
        ):
            result = redraw_stale_overlays(
                [tmp_path / f"c{i}" / f"c{i}.json" for i in range(9)], workers=2
            )

        assert result.redrawn == 9
        # Planning the third chip of a batch sees its two batch-mates, nothing older.
        assert peak[0] == 2

    def test_items_are_planned_and_settled_in_batches(self, tmp_path: Path) -> None:
        """No more than one batch of items is read before they are settled."""
        chips = [_chip(tmp_path, f"chip_{i}") for i in range(5)]
        for chip_dir in chips:
            _old_overlay(chip_dir, chip_dir.name)
        live, peak = [0], [0]
        plan, settle = overlay_refresh._plan, overlay_refresh._settle

        def counted_plan(item_path):
            live[0] += 1
            peak[0] = max(peak[0], live[0])
            return plan(item_path)

        def counted_settle(outcome):
            live[0] -= 1
            return settle(outcome)

        with (
            patch.object(overlay_refresh, "PLAN_BATCH_SIZE", 2),
            patch.object(overlay_refresh, "_plan", counted_plan),
            patch.object(overlay_refresh, "_settle", counted_settle),
        ):
            result = redraw_stale_overlays([c / f"{c.name}.json" for c in chips], workers=2)

        assert result.redrawn == 5
        assert peak[0] == 2

    def test_a_missing_overlay_is_logged_as_stale_not_older(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        chip_dir = _chip(tmp_path)
        _age(chip_dir / f"{CHIP}_planting_image_s2.webp", chip_dir)

        with caplog.at_level("INFO"):
            redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert "Chip previews: redrew 1 stale chip overlay(s)" in caplog.text
        assert "older than" not in caplog.text


class TestOverlayIsStale:
    def test_a_missing_overlay_without_a_preview_image_is_not_stale(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path, base=False)

        assert not overlay_is_stale(chip_dir, CHIP)

    def test_an_overlay_as_old_as_its_mask_is_not_stale(self, tmp_path: Path) -> None:
        """Coarse timestamps tie a mask and the overlay drawn right after it."""
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir)
        mask_mtime = (chip_dir / f"{CHIP}_semantic_3_class.tif").stat().st_mtime
        os.utime(overlay, (mask_mtime, mask_mtime))

        assert not overlay_is_stale(chip_dir, CHIP)

    def test_a_preview_as_old_as_its_mask_is_not_stale(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        mask_mtime = (chip_dir / f"{CHIP}_semantic_3_class.tif").stat().st_mtime
        os.utime(chip_dir / f"{CHIP}_planting_image_s2.webp", (mask_mtime, mask_mtime))

        assert not overlay_is_stale(chip_dir, CHIP)


class TestUnexpectedErrors:
    """Any one chip's error is counted once, and leaves the chip to be retried."""

    def test_a_draw_that_writes_then_raises_leaves_no_overlay(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir)
        real = overlay_refresh.generate_overlay_thumbnail

        def draw_then_raise(base, mask, out, *_args, **_kwargs):
            real(base, mask, out)
            raise ValueError("after write")

        with patch.object(overlay_refresh, "generate_overlay_thumbnail", draw_then_raise):
            result = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert (result.redrawn, result.failed) == (0, 1)
        assert not overlay.exists()
        assert not (chip_dir / f".{CHIP}_overlay.tmp.webp").exists()
        assert _thumbnail(chip_dir).href == f"./{CHIP}_planting_image_s2.webp"

    def test_a_failed_item_write_is_retried(self, tmp_path: Path) -> None:
        chip_dir = _chip(tmp_path)
        overlay = _old_overlay(chip_dir)

        with patch.object(overlay_refresh, "write_item", side_effect=OSError("disk full")):
            first = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert (first.redrawn, first.failed) == (0, 1)
        assert overlay_is_stale(chip_dir, CHIP)
        # The drawn overlay was never moved in, and is not left behind either.
        assert not (chip_dir / f".{CHIP}_overlay.tmp.webp").exists()
        second = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)
        assert (second.redrawn, second.failed) == (1, 0)
        assert _thumbnail(chip_dir).extra_fields["file:size"] == overlay.stat().st_size

    def test_a_failed_fallback_is_retried(self, tmp_path: Path) -> None:
        """The plain-preview fallback itself raises; the chip is not left looking done."""
        chip_dir = _chip(tmp_path)
        _old_overlay(chip_dir)

        with (
            patch.object(overlay_refresh, "generate_overlay_thumbnail", _failing_value_error),
            patch.object(overlay_refresh, "set_thumbnail", side_effect=OSError("cannot stat")),
        ):
            first = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)

        assert (first.redrawn, first.failed) == (0, 1)
        assert overlay_is_stale(chip_dir, CHIP)
        second = redraw_stale_overlays([chip_dir / f"{CHIP}.json"], workers=1)
        assert (second.redrawn, second.failed) == (1, 0)
        assert _thumbnail(chip_dir).href == f"./{CHIP}_overlay.webp"

    @pytest.mark.parametrize(
        "content", ["[]", '{"type": "Feature", "assets": {"thumbnail": null}}']
    )
    def test_a_malformed_item_is_counted_not_raised(self, tmp_path: Path, content: str) -> None:
        chip_dir = _chip(tmp_path)
        _age(chip_dir / f"{CHIP}_planting_image_s2.webp", chip_dir)
        (chip_dir / f"{CHIP}.json").write_text(content)

        result = _resume(tmp_path)

        assert (result.previews_redrawn, result.previews_redraw_failed) == (0, 1)
