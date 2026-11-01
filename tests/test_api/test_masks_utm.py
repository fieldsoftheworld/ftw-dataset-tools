"""Masks on a per-chip UTM grid, the default grid for every imagery source."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from ftw_dataset_tools.api import masks
from ftw_dataset_tools.api.chip_grid import utm_crs_for_cell
from ftw_dataset_tools.api.geo import write_geoparquet

if TYPE_CHECKING:
    from pathlib import Path

GRID_ID = "ftw-32UMA0101"
CELL = box(10.0, 50.0, 10.01, 50.01)
# Two fields filling the cell's southern half (with a gap between them).
FIELDS = [box(10.0, 50.0, 10.0045, 50.005), box(10.0055, 50.0, 10.01, 50.005)]


def _inputs(tmp_path: Path) -> dict:
    fields = write_geoparquet(
        tmp_path / "fields.parquet", gdf=gpd.GeoDataFrame({"id": [1, 2]}, geometry=FIELDS, crs=4326)
    )
    chips = write_geoparquet(
        tmp_path / "chips.parquet",
        gdf=gpd.GeoDataFrame(
            {"id": [GRID_ID], "field_coverage_pct": [50]}, geometry=[CELL], crs=4326
        ),
    )
    lines = write_geoparquet(
        tmp_path / "lines.parquet",
        gdf=gpd.GeoDataFrame(geometry=[f.boundary for f in FIELDS], crs=4326),
    )
    return {
        "chips_file": chips,
        "boundaries_file": fields,
        "boundary_lines_file": lines,
        "output_dir": tmp_path / "out",
        "mask_types": [masks.MaskType.SEMANTIC_2_CLASS],
        "num_workers": 1,
        "year": 2024,
    }


def _mask(tmp_path: Path, **kwargs) -> Path:
    result = masks.create_masks(**{**_inputs(tmp_path), **kwargs})[masks.MaskType.SEMANTIC_2_CLASS]
    assert not result.masks_skipped
    return result.masks_created[0].output_path


class TestUtmCrs:
    @pytest.mark.parametrize(
        ("grid_id", "bounds", "expected"),
        [
            ("ftw-32UMA0101_2024", (10.0, 50.0, 10.01, 50.01), "EPSG:32632"),
            # The chip id's own zone wins at a seam, even if the centre is across it.
            ("ftw-31UGQ0101", (6.05, 49.6, 6.07, 49.62), "EPSG:32631"),
            ("ftw-36MZE1020", (0.0, 0.0, 0.0, 0.0), "EPSG:32736"),
            ("custom-1", (6.05, 49.6, 6.07, 49.62), "EPSG:32632"),
            ("custom-2", (-47.0, -23.0, -46.9, -22.9), "EPSG:32723"),
        ],
    )
    def test_zone(self, grid_id: str, bounds: tuple, expected: str) -> None:
        assert utm_crs_for_cell(grid_id, bounds) == expected


class TestUtmMasks:
    def test_planet_masks_are_on_a_3m_utm_grid(self, tmp_path: Path) -> None:
        path = _mask(tmp_path, utm=True, resolution=3.0, filename_suffix="_planet")

        assert path.name == f"{GRID_ID}_2024_semantic_2_class_planet.tif"
        with rasterio.open(path) as src:
            assert src.crs.to_string() == "EPSG:32632"
            assert src.res == (3.0, 3.0)
            # ~716 m x ~1112 m of cell, rounded up to whole pixels
            assert 238 <= src.width <= 245
            assert 370 <= src.height <= 376

    def test_fields_land_where_they_are(self, tmp_path: Path) -> None:
        utm = _mask(tmp_path / "utm", utm=True, resolution=3.0, filename_suffix="_planet")
        lonlat = _mask(tmp_path / "lonlat", utm=False)

        with rasterio.open(utm) as a, rasterio.open(lonlat) as b:
            utm_share = float(np.mean(a.read(1) == 1))
            lonlat_share = float(np.mean(b.read(1) == 1))
            north, south = a.read(1)[: a.height // 4], a.read(1)[-a.height // 4 :]
        # Fields fill the southern half in both grids.
        assert utm_share == pytest.approx(lonlat_share, abs=0.05)
        assert np.mean(north == 1) < 0.01
        assert np.mean(south == 1) > 0.8

    def test_default_masks_are_on_a_10m_utm_grid(self, tmp_path: Path) -> None:
        path = _mask(tmp_path)

        assert path.name == f"{GRID_ID}_2024_semantic_2_class.tif"
        with rasterio.open(path) as src:
            assert src.crs.to_string() == "EPSG:32632"
            assert src.res == (10.0, 10.0)

    def test_lonlat_masks_on_request(self, tmp_path: Path) -> None:
        with rasterio.open(_mask(tmp_path, utm=False)) as src:
            assert src.crs.to_epsg() == 4326

    def test_ftw_chip_is_exactly_200_by_200(self, tmp_path: Path) -> None:
        # A 2 km chip as the FTW grid cuts it: a square in its UTM zone, stored in lon/lat.
        to_lonlat = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True).transform
        square = box(500_000, 5_540_000, 502_000, 5_542_000)
        chip = shapely_transform(to_lonlat, square)
        inputs = _inputs(tmp_path)
        write_geoparquet(
            inputs["chips_file"],
            gdf=gpd.GeoDataFrame(
                {"id": [GRID_ID], "field_coverage_pct": [50]}, geometry=[chip], crs=4326
            ),
        )

        result = masks.create_masks(**inputs)[masks.MaskType.SEMANTIC_2_CLASS]

        with rasterio.open(result.masks_created[0].output_path) as src:
            assert (src.width, src.height) == (200, 200)
            assert (src.transform.c, src.transform.f) == pytest.approx((500_000, 5_542_000))


class TestCreateMasksCli:
    @pytest.mark.parametrize(("flag", "epsg"), [([], 32632), (["--no-utm"], 4326)])
    def test_grid_flag(self, tmp_path: Path, flag: list[str], epsg: int) -> None:
        from click.testing import CliRunner

        from ftw_dataset_tools.cli import cli

        inputs = _inputs(tmp_path)
        out = tmp_path / "out"
        result = CliRunner().invoke(
            cli,
            [
                "create-masks",
                str(inputs["chips_file"]),
                str(inputs["boundaries_file"]),
                str(inputs["boundary_lines_file"]),
                "--output-dir",
                str(out),
                "--field-dataset",
                "test",
                "--mask-type",
                "semantic_2_class",
                "--year",
                "2024",
                *flag,
            ],
        )

        assert result.exit_code == 0, result.output
        (path,) = out.rglob("*_semantic_2_class.tif")
        with rasterio.open(path) as src:
            assert src.crs.to_epsg() == epsg
