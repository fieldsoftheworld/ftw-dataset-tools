"""Masks on a per-chip UTM grid, the default grid for every imagery source."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from shapely import segmentize
from shapely.geometry import box
from shapely.ops import transform as shapely_transform

from ftw_dataset_tools.api import masks
from ftw_dataset_tools.api.chip_grid import utm_crs_for_cell, utm_grid_for_cell
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
            ("ftw-04QFJ0101", (-160.0, 20.0, -159.99, 20.01), "EPSG:32604"),
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
    def test_3m_grid_snaps_to_the_zone_lattice(self, tmp_path: Path) -> None:
        path = _mask(tmp_path, resolution=3.0)

        with rasterio.open(path) as src:
            assert src.crs.to_string() == "EPSG:32632"
            assert src.res == (3.0, 3.0)
            assert src.transform.c % 3 == 0
            assert src.transform.f % 3 == 0
            # ~716 m x ~1112 m of cell, snapped outward to whole pixels
            assert 238 <= src.width <= 246
            assert 370 <= src.height <= 377

    def test_fields_land_where_they_are(self, tmp_path: Path) -> None:
        utm = _mask(tmp_path / "utm", resolution=3.0)
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

    @pytest.mark.parametrize("slack", [0.0, 5e-5, -5e-5])
    def test_ftw_chip_is_exactly_200_by_200(self, tmp_path: Path, slack: float) -> None:
        # A 2 km chip as the FTW grid stores it: a square in its UTM zone with a midpoint
        # on each edge, in lon/lat, whose round trip lands a fraction of a mm off.
        to_lonlat = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True).transform
        square = box(500_000 - slack, 5_540_000 - slack, 502_000 + slack, 5_542_000 + slack)
        chip = shapely_transform(to_lonlat, segmentize(square, 1000))
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


def _utm_chip(epsg: str, minx: float, miny: float, size: float = 2000) -> object:
    """A chip as the FTW grid stores it: a UTM square, edge midpoints included, in lon/lat."""
    to_lonlat = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True).transform
    return shapely_transform(to_lonlat, segmentize(box(minx, miny, minx + size, miny + size), 1000))


def _one_chip_inputs(tmp_path: Path, grid_id: str, chip, fields_utm, epsg: str) -> dict:
    to_lonlat = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True).transform
    fields = [shapely_transform(to_lonlat, f) for f in fields_utm]
    inputs = _inputs(tmp_path)
    write_geoparquet(
        inputs["boundaries_file"],
        gdf=gpd.GeoDataFrame({"id": list(range(1, len(fields) + 1))}, geometry=fields, crs=4326),
    )
    write_geoparquet(
        inputs["boundary_lines_file"],
        gdf=gpd.GeoDataFrame(geometry=[f.boundary for f in fields], crs=4326),
    )
    write_geoparquet(
        inputs["chips_file"],
        gdf=gpd.GeoDataFrame(
            {"id": [grid_id], "field_coverage_pct": [50]}, geometry=[chip], crs=4326
        ),
    )
    return inputs


class TestUtmGridEdgeCases:
    @pytest.mark.parametrize(
        ("grid_id", "epsg", "origin"),
        [
            ("ftw-32UMA0101", "EPSG:32632", (500_000, 5_540_000)),
            # Southern hemisphere: northings count up from the equator's false northing.
            ("ftw-36MZE1020", "EPSG:32736", (710_000, 9_820_000)),
        ],
    )
    def test_field_lands_on_its_exact_pixels(
        self, tmp_path: Path, grid_id: str, epsg: str, origin: tuple[int, int]
    ) -> None:
        x0, y0 = origin
        # A field covering pixels 50-99 in both axes, counted from the north-west corner.
        field = box(x0 + 500, y0 + 1000, x0 + 1000, y0 + 1500)
        inputs = _one_chip_inputs(tmp_path, grid_id, _utm_chip(epsg, x0, y0), [field], epsg)

        result = masks.create_masks(**inputs)[masks.MaskType.SEMANTIC_2_CLASS]

        with rasterio.open(result.masks_created[0].output_path) as src:
            assert src.crs.to_string() == epsg
            assert (src.width, src.height) == (200, 200)
            assert (src.transform.c, src.transform.f) == pytest.approx((x0, y0 + 2000))
            mask = src.read(1)
        # Interior pixels are field; the outline is burned as background.
        assert (mask[51:99, 51:99] == 1).all()
        assert 48 * 48 <= int(mask.sum()) <= 50 * 50
        assert (mask[:49] == 0).all()
        assert (mask[:, :49] == 0).all()

    def test_clipped_chip_keeps_the_pixel_lattice(self, tmp_path: Path) -> None:
        # A chip cut short at a square or zone edge: its west edge is off the 10 m lattice.
        chip = _utm_chip("EPSG:32632", 500_003.3, 5_540_000, size=1996.7)
        inputs = _one_chip_inputs(tmp_path, GRID_ID, chip, [], "EPSG:32632")
        inputs["boundaries_file"] = _inputs(tmp_path / "f")["boundaries_file"]
        inputs["boundary_lines_file"] = _inputs(tmp_path / "f")["boundary_lines_file"]

        result = masks.create_masks(**inputs)[masks.MaskType.SEMANTIC_2_CLASS]

        with rasterio.open(result.masks_created[0].output_path) as src:
            assert (src.transform.c, src.transform.f) == pytest.approx((500_000, 5_542_000))
            assert (src.width, src.height) == (200, 200)

    def test_chips_file_already_in_utm(self, tmp_path: Path) -> None:
        to_utm = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True).transform
        chip = _utm_chip("EPSG:32632", 500_000, 5_540_000)
        inputs = _one_chip_inputs(tmp_path, GRID_ID, chip, [], "EPSG:32632")
        inputs["boundaries_file"] = _inputs(tmp_path / "f")["boundaries_file"]
        inputs["boundary_lines_file"] = _inputs(tmp_path / "f")["boundary_lines_file"]
        write_geoparquet(
            inputs["chips_file"],
            gdf=gpd.GeoDataFrame(
                {"id": [GRID_ID], "field_coverage_pct": [50]},
                geometry=[shapely_transform(to_utm, chip)],
                crs=32632,
            ),
        )
        for name in ("boundaries_file", "boundary_lines_file"):
            gdf = gpd.read_parquet(inputs[name]).to_crs(32632)
            write_geoparquet(inputs[name], gdf=gdf)

        result = masks.create_masks(**inputs)[masks.MaskType.SEMANTIC_2_CLASS]

        with rasterio.open(result.masks_created[0].output_path) as src:
            assert src.crs.to_string() == "EPSG:32632"
            assert (src.width, src.height) == (200, 200)

    @pytest.mark.parametrize("grid_id", ["ftw-61UMA0101", "ftw-0UMA0101", "ftw-99UMA0101"])
    def test_out_of_range_zone_falls_back_to_the_centre(self, grid_id: str) -> None:
        assert utm_crs_for_cell(grid_id, (10.0, 50.0, 10.01, 50.01)) == "EPSG:32632"

    def test_existing_mask_on_a_shifted_grid_is_rebuilt(self, tmp_path: Path) -> None:
        _mask(tmp_path)
        path = next((tmp_path / "out").rglob("*_semantic_2_class.tif"))
        with rasterio.open(path) as src:
            profile, data = src.profile, src.read()
        # Same CRS, size and pixel size, origin one pixel east.
        profile["transform"] = profile["transform"] * profile["transform"].translation(1, 0)
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(data)

        result = masks.create_masks(**{**_inputs(tmp_path), "skip_existing": True})[
            masks.MaskType.SEMANTIC_2_CLASS
        ]

        assert result.masks_existing == 0
        assert len(result.masks_created) == 1


class TestNonDividingResolution:
    @pytest.mark.parametrize("x0", [500_000, 498_000, 166_000])
    def test_3m_grid_covers_the_chip_on_the_lattice(self, x0: int) -> None:
        chip = _utm_chip("EPSG:32632", x0, 5_540_000)

        _, transform, width, height = utm_grid_for_cell(
            "ftw-32UMA0101", chip.bounds, "EPSG:4326", 3.0, chip
        )

        left, top = transform.c, transform.f
        assert left % 3 == 0 and top % 3 == 0
        assert left <= x0 and left + 3 * width >= x0 + 2000
        assert top >= 5_542_000 and top - 3 * height <= 5_540_000
        # 2000 m is not a whole number of 3 m pixels: 667 or 668 depending on the offset.
        assert {width, height} <= {667, 668}

    def test_field_in_the_snapped_strip_is_burned(self, tmp_path: Path) -> None:
        # At 3 m the grid starts 2 m west of the chip; a field lying only in that strip
        # covers the first column's pixel centres and must not read as background.
        field = box(499_990, 5_540_500, 499_999.9, 5_541_500)
        chip = _utm_chip("EPSG:32632", 500_000, 5_540_000)
        inputs = _one_chip_inputs(tmp_path, GRID_ID, chip, [field], "EPSG:32632")
        # No outline, so the strip's only column is not burned back to background.
        write_geoparquet(
            inputs["boundary_lines_file"],
            gdf=gpd.GeoDataFrame(geometry=[], crs=4326),
        )

        result = masks.create_masks(**{**inputs, "resolution": 3.0})[
            masks.MaskType.SEMANTIC_2_CLASS
        ]

        with rasterio.open(result.masks_created[0].output_path) as src:
            assert src.transform.c == pytest.approx(499_998)
            mask = src.read(1)
        assert mask[:, 0].sum() > 300
        assert mask[:, 1:].sum() == 0
