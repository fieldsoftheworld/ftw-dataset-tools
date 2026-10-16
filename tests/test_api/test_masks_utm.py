"""Masks on a per-chip UTM grid, the native grid of PlanetScope imagery."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

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
        lonlat = _mask(tmp_path / "lonlat")

        with rasterio.open(utm) as a, rasterio.open(lonlat) as b:
            utm_share = float(np.mean(a.read(1) == 1))
            lonlat_share = float(np.mean(b.read(1) == 1))
            north, south = a.read(1)[: a.height // 4], a.read(1)[-a.height // 4 :]
        # Fields fill the southern half in both grids.
        assert utm_share == pytest.approx(lonlat_share, abs=0.05)
        assert np.mean(north == 1) < 0.01
        assert np.mean(south == 1) > 0.8

    def test_default_masks_are_unchanged(self, tmp_path: Path) -> None:
        path = _mask(tmp_path)

        assert path.name == f"{GRID_ID}_2024_semantic_2_class.tif"
        with rasterio.open(path) as src:
            assert src.crs.to_epsg() == 4326


class TestPlanetMaskWiring:
    def test_create_masks_cli_planet_grid(self, tmp_path: Path) -> None:
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
                "--planet-grid",
            ],
        )

        assert result.exit_code == 0, result.output
        (path,) = out.rglob("*_semantic_2_class_planet.tif")
        with rasterio.open(path) as src:
            assert src.crs.to_string() == "EPSG:32632"
            assert src.res == (3.0, 3.0)

    @pytest.mark.parametrize(("source", "passes"), [("sentinel-2", 1), ("planetscope", 2)])
    def test_pipeline_adds_a_planet_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source: str, passes: int
    ) -> None:
        from ftw_dataset_tools.api import pipeline
        from ftw_dataset_tools.api.config import DatasetConfig

        calls: list[dict] = []
        monkeypatch.setattr(masks, "create_masks", lambda **kw: calls.append(kw) or {})
        config = DatasetConfig.from_dict(
            {
                "fields_file": str(_inputs(tmp_path)["boundaries_file"]),
                "output_dir": str(tmp_path / "out"),
                "name": "ds",
                "year": 2024,
                "stages": {"select_images": {"source": source}},
            }
        )
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir()
        gpd.GeoDataFrame(
            {"id": [GRID_ID], "field_coverage_pct": [50.0]}, geometry=[CELL], crs=4326
        ).to_parquet(ctx.chips_path)
        ctx.output_fields_path.write_bytes(b"")
        ctx.boundary_lines_path.write_bytes(b"")

        pipeline.stage_masks(ctx)

        assert len(calls) == passes
        assert "utm" not in calls[0]
        if passes == 2:
            assert calls[1]["utm"] is True
            assert calls[1]["resolution"] == 3.0
            assert calls[1]["filename_suffix"] == "_planet"
            assert calls[1]["mask_types"] == calls[0]["mask_types"]


def test_stac_item_lists_planet_masks_beside_the_default_ones(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from ftw_dataset_tools.api.stac import ChipInfo, _create_chip_item

    chip_dir = tmp_path / "chip"
    chip_dir.mkdir()
    inputs = _inputs(tmp_path)
    common = {**inputs, "chip_dirs": {f"{GRID_ID}_2024": chip_dir}}
    masks.create_masks(**common)
    masks.create_masks(**common, utm=True, resolution=3.0, filename_suffix="_planet")
    info = ChipInfo(GRID_ID, CELL.__geo_interface__, CELL.bounds, year=2024)
    extent = (datetime(2024, 1, 1, tzinfo=UTC), datetime(2024, 12, 31, tzinfo=UTC))

    item = _create_chip_item(info, extent, chip_dir)

    assert {"semantic_2class_mask", "semantic_2class_mask_planet"} <= set(item.assets)
    planet = item.assets["semantic_2class_mask_planet"]
    assert planet.href == f"./{GRID_ID}_2024_semantic_2_class_planet.tif"
    assert "PlanetScope 3 m UTM grid" in planet.title
    band = planet.extra_fields["raster:bands"][0]
    assert band["classification:classes"]
    assert band["spatial_resolution"] == 3.0


def test_ftw_chip_grid_is_exactly_the_chip(tmp_path: Path) -> None:
    """An FTW chip is a 2 km square in its UTM zone; its grid starts at that square's corner."""
    from pyproj import Transformer

    to_lonlat = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True).transform
    from shapely.ops import transform

    square = box(286_000, 5_498_000, 288_000, 5_500_000)
    chip = transform(to_lonlat, square.segmentize(50))
    inputs = _inputs(tmp_path)
    inputs["chips_file"] = write_geoparquet(
        tmp_path / "ftw_chips.parquet",
        gdf=gpd.GeoDataFrame(
            {"id": ["ftw-32UKV8698"], "field_coverage_pct": [50]}, geometry=[chip], crs=4326
        ),
    )

    result = masks.create_masks(**inputs, utm=True, resolution=3.0, filename_suffix="_planet")

    (created,) = result[masks.MaskType.SEMANTIC_2_CLASS].masks_created
    with rasterio.open(created.output_path) as src:
        assert (src.width, src.height) == (667, 667)
        assert (src.transform.c, src.transform.f) == pytest.approx((286_000, 5_500_000), abs=0.05)
