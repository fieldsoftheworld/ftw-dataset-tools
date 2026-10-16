"""Cutting selected PlanetScope scenes onto each chip's 3 m UTM grid (no network)."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pystac
import pytest
import rasterio
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import box

from ftw_dataset_tools.api import masks
from ftw_dataset_tools.api.geo import write_geoparquet
from ftw_dataset_tools.api.imagery.planet_download import download_planet_imagery_for_catalog

if TYPE_CHECKING:
    from pathlib import Path

GRID_ID = "ftw-32UMA0101"
CHIP = f"{GRID_ID}_2024"
CELL = box(10.0, 50.0, 10.01, 50.01)


def _catalog(tmp_path: Path, *, with_masks: bool = True) -> tuple[Path, Path]:
    """A one-chip catalog with a planting PlanetScope child; returns (catalog, chip dir)."""
    catalog = tmp_path / "dataset"
    chip_dir = catalog / "chips" / "32UMA" / CHIP
    chip_dir.mkdir(parents=True)
    parent = pystac.Item(
        CHIP, CELL.__geo_interface__, list(CELL.bounds), datetime(2024, 1, 1, tzinfo=UTC), {}
    )
    parent.save_object(dest_href=str(chip_dir / f"{CHIP}.json"), include_self_link=False)

    if with_masks:
        fields = write_geoparquet(
            tmp_path / "fields.parquet",
            gdf=gpd.GeoDataFrame(
                {"id": [1]}, geometry=[box(10.002, 50.002, 10.008, 50.008)], crs=4326
            ),
        )
        chips = write_geoparquet(
            tmp_path / "chips.parquet",
            gdf=gpd.GeoDataFrame(
                {"id": [GRID_ID], "field_coverage_pct": [40]}, geometry=[CELL], crs=4326
            ),
        )
        lines = write_geoparquet(
            tmp_path / "lines.parquet",
            gdf=gpd.GeoDataFrame(geometry=[box(10.002, 50.002, 10.008, 50.008).boundary], crs=4326),
        )
        masks.create_masks(
            chips_file=chips,
            boundaries_file=fields,
            boundary_lines_file=lines,
            output_dir=catalog / "out",
            mask_types=[masks.MaskType.SEMANTIC_2_CLASS],
            num_workers=1,
            year=2024,
            chip_dirs={CHIP: chip_dir},
            resolution=3.0,
            utm=True,
            filename_suffix="_planet",
        )

    # The delivered scene: Planet's own UTM zone (31, across the seam), band i = 1000 * i.
    delivery = catalog / ".planet" / "order-1"
    delivery.mkdir(parents=True)
    analytic = delivery / "s1_3B_AnalyticMS_SR_clip.tif"
    west, south, east, north = transform_bounds(
        "EPSG:4326", "EPSG:32631", 9.99, 49.99, 10.02, 50.02
    )
    width, height = int((east - west) / 3) + 1, int((north - south) / 3) + 1
    with rasterio.open(
        analytic,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=4,
        dtype="uint16",
        crs="EPSG:32631",
        transform=from_origin(west, north, 3, 3),
        nodata=0,
    ) as dst:
        for band in range(1, 5):
            dst.write(np.full((height, width), 1000 * band, dtype=np.uint16), band)

    child = pystac.Item(
        f"{CHIP}_planting_planet",
        CELL.__geo_interface__,
        list(CELL.bounds),
        datetime(2024, 4, 6, tzinfo=UTC),
        {"ftw:source": "planetscope"},
    )
    child.add_asset(
        "analytic", pystac.Asset("../../../.planet/order-1/s1_3B_AnalyticMS_SR_clip.tif")
    )
    child.save_object(dest_href=str(chip_dir / f"{child.id}.json"), include_self_link=False)
    return catalog, chip_dir


def test_scene_is_cut_onto_the_chip_grid(tmp_path: Path) -> None:
    catalog, chip_dir = _catalog(tmp_path)

    result = download_planet_imagery_for_catalog(catalog, workers=1)

    assert (result.successful, result.failed) == (1, 0), result.failed_details
    image = chip_dir / f"{CHIP}_planting_image_planet.tif"
    mask = chip_dir / f"{GRID_ID}_2024_semantic_2_class_planet.tif"
    with rasterio.open(image) as img, rasterio.open(mask) as ref:
        assert img.crs == ref.crs
        assert img.crs.to_string() == "EPSG:32632"
        assert (img.width, img.height, img.transform) == (ref.width, ref.height, ref.transform)
        assert img.descriptions == ("blue", "green", "red", "nir")
        assert img.dtypes[0] == "uint16"
        assert img.scales[0] == pytest.approx(0.0001)
        assert [int(np.median(img.read(b))) for b in range(1, 5)] == [1000, 2000, 3000, 4000]


def test_child_and_parent_point_at_the_image(tmp_path: Path) -> None:
    catalog, chip_dir = _catalog(tmp_path)

    download_planet_imagery_for_catalog(catalog, workers=1)

    child = pystac.Item.from_file(str(chip_dir / f"{CHIP}_planting_planet.json"))
    parent = pystac.Item.from_file(str(chip_dir / f"{CHIP}.json"))
    assert child.assets["image"].href.endswith(f"{CHIP}_planting_image_planet.tif")
    assert parent.assets["planting_image_planet"].href.endswith(f"{CHIP}_planting_image_planet.tif")
    assert "planting_image" not in parent.assets  # the Sentinel-2 asset key is untouched


def test_resume_skips_written_images(tmp_path: Path) -> None:
    catalog, _chip_dir = _catalog(tmp_path)
    download_planet_imagery_for_catalog(catalog, workers=1)

    again = download_planet_imagery_for_catalog(catalog, workers=1)

    assert (again.successful, again.skipped) == (0, 1)


def test_chip_without_planet_masks_fails_with_a_hint(tmp_path: Path) -> None:
    catalog, _chip_dir = _catalog(tmp_path, with_masks=False)

    result = download_planet_imagery_for_catalog(catalog, workers=1)

    assert result.failed == 1
    assert "create-masks --planet-grid" in result.failed_details[0]["error"]


class TestWiring:
    def test_download_images_cli(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from ftw_dataset_tools.cli import cli

        catalog, chip_dir = _catalog(tmp_path)
        (catalog / "collection.json").write_text("{}")

        result = CliRunner().invoke(
            cli, ["download-images", str(catalog), "--source", "planetscope"]
        )

        assert result.exit_code == 0, result.output
        assert (chip_dir / f"{CHIP}_planting_image_planet.tif").exists()

    def test_sentinel2_options_are_rejected(self, tmp_path: Path) -> None:
        from click.testing import CliRunner

        from ftw_dataset_tools.cli import cli

        catalog, _ = _catalog(tmp_path, with_masks=False)
        (catalog / "collection.json").write_text("{}")

        result = CliRunner().invoke(
            cli, ["download-images", str(catalog), "--source", "planetscope", "--bands", "red"]
        )

        assert result.exit_code == 2
        assert "only applies to --source sentinel-2" in result.output

    def test_planet_with_preview_download_is_a_config_error(self) -> None:
        from ftw_dataset_tools.api.config import ConfigError, DatasetConfig

        with pytest.raises(ConfigError, match="preview"):
            DatasetConfig.from_dict(
                {
                    "fields_file": "f.parquet",
                    "stages": {
                        "select_images": {"source": "planetscope"},
                        "download_images": {"mode": "preview"},
                    },
                }
            )
