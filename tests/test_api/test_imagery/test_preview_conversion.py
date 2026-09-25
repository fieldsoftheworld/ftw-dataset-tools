"""Tests for converting a catalog's JPEG chip previews to WebP."""

from __future__ import annotations

import datetime
from pathlib import Path  # noqa: TC003 - used at runtime in helpers

import numpy as np
import pystac
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

from ftw_dataset_tools.api.assets import add_file_info, multihash_sha256
from ftw_dataset_tools.api.imagery.preview_conversion import (
    convert_previews_for_catalog,
    legacy_previews,
)

CRS = "EPSG:32633"
SIZE = 32
_BOUNDS = (501000, 5001000, 501320, 5001320)


def _write_raster(path: Path, data: np.ndarray, dtype: str) -> Path:
    count = data.shape[0]
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[2],
        height=data.shape[1],
        count=count,
        dtype=dtype,
        crs=CRS,
        transform=from_bounds(*_BOUNDS, data.shape[2], data.shape[1]),
    ) as dst:
        dst.write(data)
    return path


def _scene(path: Path) -> Path:
    """A 'scene' COG far larger than one chip."""
    width = height = 300
    data = np.zeros((3, height, width), dtype="uint8")
    ramp = np.linspace(0, 255, width, dtype="uint8")
    data[0, :, :] = ramp[None, :]
    data[1, :, :] = ramp[::-1][None, :]
    data[2, :, :] = 100
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=3,
        dtype="uint8",
        crs=CRS,
        transform=from_bounds(500000, 5000000, 503000, 5003000, width, height),
    ) as dst:
        dst.write(data)
    return path


def _collection(tmp_path: Path) -> Path:
    out = tmp_path / "ds"
    out.mkdir(parents=True, exist_ok=True)
    (out / "collection.json").write_text("{}")
    return out


def _chip_item(chip_dir: Path, chip_id: str) -> pystac.Item:
    item = pystac.Item(
        id=chip_id,
        geometry={"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]},
        bbox=[0.0, 0.0, 1.0, 1.0],
        datetime=None,
        start_datetime=datetime.datetime(2024, 1, 1, tzinfo=datetime.UTC),
        end_datetime=datetime.datetime(2024, 12, 31, tzinfo=datetime.UTC),
        properties={},
    )
    item.set_self_href(str(chip_dir / f"{chip_id}.json"))
    return item


def _clipped_chip(collection_dir: Path, chip_id: str, *, checksums: bool = False) -> Path:
    """A chip built the ``--download-images`` way: local GeoTIFFs and .jpg previews."""
    chip_dir = collection_dir / "chips" / "33TXM" / chip_id
    chip_dir.mkdir(parents=True, exist_ok=True)

    mask = np.zeros((1, SIZE, SIZE), dtype="uint8")
    mask[0, 5:20, 5:20] = 1
    mask[0, 20:24, 5:20] = 2
    _write_raster(chip_dir / f"{chip_id}_semantic_3_class.tif", mask, "uint8")

    rgb = np.random.default_rng(0).integers(0, 4000, (4, SIZE, SIZE)).astype("uint16")
    for season in ("planting", "harvest"):
        _write_raster(chip_dir / f"{chip_id}_{season}_image_s2.tif", rgb, "uint16")
        # The preview an earlier build left behind, as a real JPEG.
        Image.new("RGB", (SIZE, SIZE), (10, 20, 30)).save(
            chip_dir / f"{chip_id}_{season}_image_s2.jpg", "JPEG"
        )
    Image.new("RGB", (SIZE, SIZE), (40, 50, 60)).save(chip_dir / f"{chip_id}_overlay.jpg", "JPEG")

    item = _chip_item(chip_dir, chip_id)
    item.add_asset(
        "thumbnail",
        pystac.Asset(
            href=f"./{chip_id}_overlay.jpg",
            media_type="image/jpeg",
            roles=["thumbnail"],
        ),
    )
    if checksums:
        add_file_info(item.assets["thumbnail"], chip_dir / f"{chip_id}_overlay.jpg", checksum=True)
    item.save_object(include_self_link=False, dest_href=str(chip_dir / f"{chip_id}.json"))

    for season in ("planting", "harvest"):
        child = pystac.Item(
            id=f"{chip_id}_{season}_s2",
            geometry=item.geometry,
            bbox=item.bbox,
            datetime=None,
            start_datetime=item.common_metadata.start_datetime,
            end_datetime=item.common_metadata.end_datetime,
            properties={},
        )
        child.add_asset(
            "thumbnail",
            pystac.Asset(
                href=f"./{chip_id}_{season}_image_s2.jpg",
                media_type="image/jpeg",
                roles=["thumbnail"],
            ),
        )
        if checksums:
            add_file_info(
                child.assets["thumbnail"],
                chip_dir / f"{chip_id}_{season}_image_s2.jpg",
                checksum=True,
            )
        child_path = chip_dir / f"{chip_id}_{season}_s2.json"
        child.set_self_href(str(child_path))
        child.save_object(include_self_link=False, dest_href=str(child_path))

    return chip_dir


def _remote_chip(collection_dir: Path, chip_id: str, scene: Path) -> Path:
    """A chip whose scene stays remote: a mask, a visual href, and a .jpg overlay."""
    chip_dir = collection_dir / "chips" / "33TXM" / chip_id
    chip_dir.mkdir(parents=True, exist_ok=True)

    mask = np.zeros((1, SIZE, SIZE), dtype="uint8")
    mask[0, 5:20, 5:20] = 1
    _write_raster(chip_dir / f"{chip_id}_semantic_3_class.tif", mask, "uint8")
    Image.new("RGB", (SIZE, SIZE), (40, 50, 60)).save(chip_dir / f"{chip_id}_overlay.jpg", "JPEG")

    item = _chip_item(chip_dir, chip_id)
    item.save_object(include_self_link=False, dest_href=str(chip_dir / f"{chip_id}.json"))

    child = pystac.Item(
        id=f"{chip_id}_planting_s2",
        geometry=item.geometry,
        bbox=item.bbox,
        datetime=None,
        start_datetime=item.common_metadata.start_datetime,
        end_datetime=item.common_metadata.end_datetime,
        properties={},
    )
    child.add_asset("visual", pystac.Asset(href=str(scene), roles=["visual"]))
    child_path = chip_dir / f"{chip_id}_planting_s2.json"
    child.set_self_href(str(child_path))
    child.save_object(include_self_link=False, dest_href=str(child_path))

    return chip_dir


def _sourceless_chip(collection_dir: Path, chip_id: str) -> Path:
    """A chip with a .jpg preview but nothing to re-render it from: no mask, no scene."""
    chip_dir = collection_dir / "chips" / "33TXM" / chip_id
    chip_dir.mkdir(parents=True, exist_ok=True)
    item = _chip_item(chip_dir, chip_id)
    item.save_object(include_self_link=False, dest_href=str(chip_dir / f"{chip_id}.json"))
    Image.new("RGB", (SIZE, SIZE), (1, 2, 3)).save(chip_dir / f"{chip_id}_overlay.jpg", "JPEG")
    return chip_dir


class TestLegacyPreviews:
    def test_finds_every_jpeg_preview(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        chip_dir = _clipped_chip(out, "chip_a")
        found = {path.name for path in legacy_previews(chip_dir, "chip_a")}
        assert found == {
            "chip_a_overlay.jpg",
            "chip_a_planting_image_s2.jpg",
            "chip_a_harvest_image_s2.jpg",
        }

    def test_a_webp_only_chip_has_none(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        chip_dir = _clipped_chip(out, "chip_a")
        for path in legacy_previews(chip_dir, "chip_a"):
            path.unlink()
        assert legacy_previews(chip_dir, "chip_a") == []


class TestConvertPreviewsForCatalog:
    def test_converts_a_clipped_catalog_from_its_local_geotiffs(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        chip_dir = _clipped_chip(out, "chip_a")

        result = convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        assert (result.chips_converted, result.failed) == (1, 0)
        assert result.legacy_removed == 3
        for name in ("chip_a_overlay", "chip_a_planting_image_s2", "chip_a_harvest_image_s2"):
            webp = chip_dir / f"{name}.webp"
            assert webp.exists(), name
            with Image.open(webp) as img:
                assert img.format == "WEBP"

    def test_removes_the_superseded_jpegs(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        chip_dir = _clipped_chip(out, "chip_a")

        convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        assert legacy_previews(chip_dir, "chip_a") == []

    def test_repoints_the_chip_item(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        chip_dir = _clipped_chip(out, "chip_a")

        convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        item = pystac.Item.from_file(str(chip_dir / "chip_a.json"))
        assert item.assets["thumbnail"].href == "./chip_a_overlay.webp"
        assert item.assets["thumbnail"].media_type == "image/webp"

    def test_repoints_the_season_children(self, tmp_path: Path) -> None:
        """A child left pointing at a deleted .jpg is a broken asset in the catalog."""
        out = _collection(tmp_path)
        chip_dir = _clipped_chip(out, "chip_a")

        convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        for season in ("planting", "harvest"):
            child = pystac.Item.from_file(str(chip_dir / f"chip_a_{season}_s2.json"))
            thumbnail = child.assets["thumbnail"]
            assert thumbnail.href == f"./chip_a_{season}_image_s2.webp"
            assert thumbnail.media_type == "image/webp"

    def test_converts_a_remote_catalog_from_its_scene(self, tmp_path: Path) -> None:
        """Re-running the preview stage covers only this case; conversion covers both."""
        out = _collection(tmp_path)
        scene = _scene(tmp_path / "scene.tif")
        chip_dir = _remote_chip(out, "chip_b", scene)

        result = convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        assert (result.chips_converted, result.failed) == (1, 0)
        assert (chip_dir / "chip_b_overlay.webp").exists()
        assert not (chip_dir / "chip_b_overlay.jpg").exists()

    def test_skips_a_catalog_that_is_already_webp(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        _clipped_chip(out, "chip_a")
        convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        again = convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        assert (again.chips_converted, again.skipped, again.failed) == (0, 1, 0)

    def test_dry_run_changes_nothing(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        chip_dir = _clipped_chip(out, "chip_a")

        result = convert_previews_for_catalog(out, dry_run=True, show_progress_bar=False)

        assert result.chips_converted == 1
        assert result.previews_written == 3
        assert len(legacy_previews(chip_dir, "chip_a")) == 3
        assert not (chip_dir / "chip_a_overlay.webp").exists()

    def test_a_chip_with_nothing_to_render_from_keeps_its_jpeg(self, tmp_path: Path) -> None:
        """Without imagery or a scene there is no source, so the .jpg must survive."""
        out = _collection(tmp_path)
        chip_dir = _sourceless_chip(out, "chip_c")

        result = convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        assert (result.chips_converted, result.skipped) == (0, 1)
        assert (chip_dir / "chip_c_overlay.jpg").exists()


class TestConversionChecksums:
    """A catalog built with ``--checksums`` must come out of conversion with them.

    ``file:checksum`` is the only thing that tells a consumer the preview bytes are
    the ones the catalog was published with, so a run that drops it - or, worse,
    leaves the JPEG's checksum on the WebP asset - silently breaks verification.
    """

    def test_chip_thumbnail_checksum_matches_the_webp(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        chip_dir = _clipped_chip(out, "chip_a", checksums=True)

        convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        thumbnail = pystac.Item.from_file(str(chip_dir / "chip_a.json")).assets["thumbnail"]
        assert thumbnail.extra_fields["file:checksum"] == multihash_sha256(
            chip_dir / "chip_a_overlay.webp"
        )

    def test_child_thumbnail_checksums_match_the_webp(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        chip_dir = _clipped_chip(out, "chip_a", checksums=True)

        convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        for season in ("planting", "harvest"):
            child = pystac.Item.from_file(str(chip_dir / f"chip_a_{season}_s2.json"))
            preview = chip_dir / f"chip_a_{season}_image_s2.webp"
            assert child.assets["thumbnail"].extra_fields["file:checksum"] == multihash_sha256(
                preview
            )

    def test_a_catalog_without_checksums_gains_none(self, tmp_path: Path) -> None:
        """Checksums are opt-in, so conversion must not start adding them."""
        out = _collection(tmp_path)
        chip_dir = _clipped_chip(out, "chip_a")

        convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        item = pystac.Item.from_file(str(chip_dir / "chip_a.json"))
        assert "file:checksum" not in item.assets["thumbnail"].extra_fields
        child = pystac.Item.from_file(str(chip_dir / "chip_a_planting_s2.json"))
        assert "file:checksum" not in child.assets["thumbnail"].extra_fields


class TestDryRunMatchesTheRealRun:
    """--dry-run is the gate before a destructive migration, so it must not over-report."""

    def test_reports_a_sourceless_chip_as_skipped(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        _sourceless_chip(out, "chip_c")

        result = convert_previews_for_catalog(out, dry_run=True, show_progress_bar=False)

        assert (result.chips_converted, result.skipped) == (0, 1)
        assert (result.previews_written, result.legacy_removed) == (0, 0)
        assert result.skipped_details == [
            {"chip": "chip_c", "reason": "No imagery to re-render the preview from"}
        ]

    def test_counts_match_a_real_run_on_a_mixed_catalog(self, tmp_path: Path) -> None:
        out = _collection(tmp_path)
        _clipped_chip(out, "chip_a")
        _sourceless_chip(out, "chip_c")

        dry = convert_previews_for_catalog(out, dry_run=True, show_progress_bar=False)
        real = convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        assert (dry.chips_converted, dry.previews_written, dry.legacy_removed, dry.skipped) == (
            real.chips_converted,
            real.previews_written,
            real.legacy_removed,
            real.skipped,
        )

    def test_counts_only_the_previews_a_partial_source_can_render(self, tmp_path: Path) -> None:
        """One season's GeoTIFF is gone and no scene is referenced: one preview, one JPEG."""
        out = _collection(tmp_path)
        chip_dir = _clipped_chip(out, "chip_a")
        (chip_dir / "chip_a_planting_image_s2.tif").unlink()

        dry = convert_previews_for_catalog(out, dry_run=True, show_progress_bar=False)
        real = convert_previews_for_catalog(out, show_progress_bar=False, workers=1)

        assert (dry.chips_converted, dry.previews_written, dry.legacy_removed) == (1, 1, 1)
        assert (real.chips_converted, real.previews_written, real.legacy_removed) == (1, 1, 1)
