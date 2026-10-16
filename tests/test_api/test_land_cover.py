"""Tests for per-chip land cover from the Impact Observatory maps."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import geopandas as gpd
import numpy as np
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box

from ftw_dataset_tools.api import land_cover
from ftw_dataset_tools.api.land_cover import (
    LandCoverResult,
    YearMatch,
    add_land_cover,
    class_entries,
    count_classes,
    drop_land_cover,
    gzd_for_point,
    land_cover_summary,
    normalize_gzd,
    resolve_year,
)

if TYPE_CHECKING:
    from pathlib import Path

# A 1 km x 1 km tile at 10 m in UTM 33N, with its top-left corner on the zone's
# central meridian at about 48.3 N (grid zone 33U).
UTM = "EPSG:32633"
X0, Y0 = 500_000.0, 5_350_000.0
TILE_PX = 100
TREES, CROPS, WATER = 2, 5, 1

_TO_LONLAT = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)


def write_tile(path: Path, data: np.ndarray | None = None) -> Path:
    """GeoTIFF over the tile area; by default columns 0-49 are Trees, 50-99 Crops."""
    if data is None:
        data = np.full((TILE_PX, TILE_PX), CROPS, dtype="uint8")
        data[:, :50] = TREES
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype="uint8",
        crs=UTM,
        transform=from_origin(X0, Y0, 10.0, 10.0),
        nodata=0,
    ) as dst:
        dst.write(data, 1)
    return path


def chip_polygon(col0: int, row0: int, cols: int, rows: int) -> Polygon:
    """A lon/lat polygon over tile pixels [col0, col0+cols) x [row0, row0+rows)."""
    left, right = X0 + 10 * col0, X0 + 10 * (col0 + cols)
    top, bottom = Y0 - 10 * row0, Y0 - 10 * (row0 + rows)
    corners = [(left, bottom), (right, bottom), (right, top), (left, top)]
    return Polygon([_TO_LONLAT.transform(x, y) for x, y in corners])


def write_chips(path: Path, chips: dict[str, Polygon], *, gzd: str | None = "33U") -> Path:
    data: dict[str, list] = {"id": list(chips)}
    if gzd is not None:
        data["gzd"] = [gzd] * len(chips)
    gpd.GeoDataFrame(data, geometry=list(chips.values()), crs="EPSG:4326").to_parquet(path)
    return path


class LocalSource:
    """A TileSource over local files, recording the bboxes it was asked for."""

    def __init__(self, index: dict[str, dict[int, str]]) -> None:
        self._index = index
        self.calls: list[tuple] = []

    def index(self, bbox: tuple[float, float, float, float]) -> dict[str, dict[int, str]]:
        self.calls.append(bbox)
        return self._index


def two_chip_setup(tmp_path: Path) -> tuple[Path, Path]:
    """Chip A covers cols 0-69 (50 Trees, 20 Crops); chip B covers cols 80-99 (all Crops)."""
    tile = write_tile(tmp_path / "tile.tif")
    chips = write_chips(
        tmp_path / "chips.parquet",
        {
            "ftw-33UXP0001": chip_polygon(0, 0, 70, 100),
            "ftw-33UXP0002": chip_polygon(80, 0, 20, 100),
        },
    )
    return chips, tile


def read_land_cover(chips: Path) -> dict[str, tuple]:
    con = duckdb.connect()
    try:
        cols = ", ".join(land_cover.OUTPUT_COLUMNS)
        rows = con.execute(f"SELECT id, {cols} FROM read_parquet('{chips}') ORDER BY id")
        return {row[0]: row[1:] for row in rows.fetchall()}
    finally:
        con.close()


class TestResolveYear:
    def test_exact(self) -> None:
        assert resolve_year(2021, range(2017, 2024)) == YearMatch(2021, exact=True)

    @pytest.mark.parametrize(
        ("dataset_year", "expected"),
        [(2010, 2017), (2016, 2017), (2024, 2023), (2026, 2023)],
    )
    def test_outside_the_range_uses_the_nearest_end(self, dataset_year: int, expected: int) -> None:
        assert resolve_year(dataset_year, range(2017, 2024)) == YearMatch(expected, exact=False)

    def test_gap_uses_the_nearest_year(self) -> None:
        assert resolve_year(2019, {2017, 2020}) == YearMatch(2020, exact=False)

    def test_tie_picks_the_later_year(self) -> None:
        assert resolve_year(2021, {2020, 2022}) == YearMatch(2022, exact=False)

    def test_nothing_available(self) -> None:
        assert resolve_year(2021, []) is None


class TestGridZones:
    @pytest.mark.parametrize(
        ("raw", "expected"), [("33U", "33U"), ("5v", "05V"), (" 05V ", "05V"), ("x", "X")]
    )
    def test_normalize(self, raw: str, expected: str) -> None:
        assert normalize_gzd(raw) == expected

    @pytest.mark.parametrize(
        ("lon", "lat", "expected"),
        [
            (14.3, 48.3, "33U"),
            (-153.5, 60.5, "05V"),
            (0.0, -80.0, "31C"),
            (179.9, 0.5, "60N"),
            (2.5, 56.5, "31V"),  # west of the Norway exception
            (3.5, 56.5, "32V"),  # Norway exception
            (5.0, 78.0, "31X"),  # Svalbard exceptions
            (10.0, 78.0, "33X"),
            (25.0, 78.0, "35X"),
            (40.0, 78.0, "37X"),
            (15.0, 84.0, "33X"),
        ],
    )
    def test_gzd_for_point(self, lon: float, lat: float, expected: str) -> None:
        assert gzd_for_point(lon, lat) == expected

    @pytest.mark.parametrize("lat", [-80.5, 84.5])
    def test_outside_utm_latitudes(self, lat: float) -> None:
        assert gzd_for_point(15.0, lat) is None


class TestClassEntries:
    def test_shares_of_valid_pixels_largest_first(self) -> None:
        counts = np.bincount([0, 0, 5, 2, 2, 2])

        entries = class_entries(counts)

        assert entries == [
            {"code": 2, "name": "Trees", "pct": 75.0},
            {"code": 5, "name": "Crops", "pct": 25.0},
        ]

    def test_ties_are_ordered_by_code(self) -> None:
        entries = class_entries(np.bincount([5, 2]))

        assert [e["code"] for e in entries] == [2, 5]

    def test_only_nodata_is_none(self) -> None:
        assert class_entries(np.bincount([0, 0, 0])) is None
        assert class_entries(np.zeros(0, dtype=np.int64)) is None

    def test_unknown_value_has_no_name(self) -> None:
        entries = class_entries(np.bincount([3]))

        assert entries == [{"code": 3, "name": None, "pct": 100.0}]


class TestCountClasses:
    def _square(self, col0: float, row0: float, cols: float, rows: float) -> dict:
        left, top = X0 + 10 * col0, Y0 - 10 * row0
        right, bottom = left + 10 * cols, top - 10 * rows
        ring = [[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]
        return {"type": "Polygon", "coordinates": [ring]}

    def test_pixel_aligned_square(self, tmp_path: Path) -> None:
        with rasterio.open(write_tile(tmp_path / "t.tif")) as src:
            counts = count_classes(src, self._square(40, 0, 20, 10))

        assert counts[TREES] == 100
        assert counts[CROPS] == 100

    def test_pixel_centre_rule(self, tmp_path: Path) -> None:
        """A pixel counts only when its centre is inside the geometry.

        The square overlaps pixel 0 (centre 0.5, outside) and pixel 1 (centre 1.5, inside).
        """
        with rasterio.open(write_tile(tmp_path / "t.tif")) as src:
            counts = count_classes(src, self._square(0.6, 0, 1.0, 1))

        assert counts.sum() == 1

    def test_partly_outside_the_raster(self, tmp_path: Path) -> None:
        with rasterio.open(write_tile(tmp_path / "t.tif")) as src:
            counts = count_classes(src, self._square(-10, -10, 20, 20))

        assert counts.sum() == 100

    def test_fully_outside_the_raster(self, tmp_path: Path) -> None:
        with rasterio.open(write_tile(tmp_path / "t.tif")) as src:
            counts = count_classes(src, self._square(500, 500, 10, 10))

        assert counts.sum() == 0


class TestAddLandCover:
    def test_exact_year(self, tmp_path: Path) -> None:
        chips, tile = two_chip_setup(tmp_path)

        result = add_land_cover(chips, year=2021, source=LocalSource({"33U": {2021: str(tile)}}))

        assert result == LandCoverResult(2, 2, 0, (2021,), 2021, skipped=False)
        rows = read_land_cover(chips)
        year, exact, code, name, pct, classes = rows["ftw-33UXP0001"]
        assert (year, exact, code, name) == (2021, True, TREES, "Trees")
        assert pct == pytest.approx(71.43, abs=0.01)
        assert [(c["code"], c["name"]) for c in classes] == [(TREES, "Trees"), (CROPS, "Crops")]
        assert classes[1]["pct"] == pytest.approx(28.57, abs=0.01)
        assert rows["ftw-33UXP0002"][2:5] == (CROPS, "Crops", 100.0)

    def test_nearest_year_is_flagged(self, tmp_path: Path) -> None:
        chips, tile = two_chip_setup(tmp_path)
        messages: list[str] = []

        result = add_land_cover(
            chips,
            year=2025,
            source=LocalSource({"33U": {2017: str(tile), 2023: str(tile)}}),
            on_progress=messages.append,
        )

        assert result.chips_nearest_year == 2
        assert result.years_used == (2023,)
        assert {row[:2] for row in read_land_cover(chips).values()} == {(2023, False)}
        assert any(m.startswith("Warning: 2 chips use the nearest IO year") for m in messages)

    def test_tile_missing_leaves_only_its_chips_blank(self, tmp_path: Path) -> None:
        chips, tile = two_chip_setup(tmp_path)
        gdf = gpd.read_parquet(chips)
        gdf.loc[1, "gzd"] = "34U"
        gdf.to_parquet(chips)

        result = add_land_cover(chips, year=2021, source=LocalSource({"33U": {2021: str(tile)}}))

        rows = read_land_cover(chips)
        assert rows["ftw-33UXP0001"][0] == 2021
        assert rows["ftw-33UXP0002"] == (None,) * len(land_cover.OUTPUT_COLUMNS)
        assert (result.chips_total, result.chips_with_land_cover) == (2, 1)

    def test_chip_without_valid_pixels_is_blank(self, tmp_path: Path) -> None:
        data = np.full((TILE_PX, TILE_PX), TREES, dtype="uint8")
        data[:, 50:] = 0
        tile = write_tile(tmp_path / "tile.tif", data)
        chips = write_chips(
            tmp_path / "chips.parquet",
            {
                "ftw-33UXP0001": chip_polygon(0, 0, 50, 100),
                "ftw-33UXP0002": chip_polygon(60, 0, 40, 100),
            },
        )

        add_land_cover(chips, year=2021, source=LocalSource({"33U": {2021: str(tile)}}))

        rows = read_land_cover(chips)
        assert rows["ftw-33UXP0001"][4] == 100.0
        assert rows["ftw-33UXP0002"][0] is None

    def test_nodata_is_left_out_of_the_denominator(self, tmp_path: Path) -> None:
        data = np.zeros((TILE_PX, TILE_PX), dtype="uint8")
        data[:, :30] = WATER
        data[:, 30:40] = TREES
        tile = write_tile(tmp_path / "tile.tif", data)
        chips = write_chips(
            tmp_path / "chips.parquet", {"ftw-33UXP0001": chip_polygon(0, 0, 100, 100)}
        )

        add_land_cover(chips, year=2021, source=LocalSource({"33U": {2021: str(tile)}}))

        classes = read_land_cover(chips)["ftw-33UXP0001"][5]
        assert [(c["code"], c["pct"]) for c in classes] == [(WATER, 75.0), (TREES, 25.0)]

    def test_tile_found_from_the_centroid_without_a_gzd_column(self, tmp_path: Path) -> None:
        tile = write_tile(tmp_path / "tile.tif")
        chips = write_chips(
            tmp_path / "chips.parquet", {"ftw-33UXP0001": chip_polygon(0, 0, 50, 100)}, gzd=None
        )

        add_land_cover(chips, year=2021, source=LocalSource({"33U": {2021: str(tile)}}))

        assert read_land_cover(chips)["ftw-33UXP0001"][2] == TREES

    def test_source_is_asked_for_the_chips_bbox(self, tmp_path: Path) -> None:
        chips, tile = two_chip_setup(tmp_path)
        source = LocalSource({"33U": {2021: str(tile)}})

        add_land_cover(chips, year=2021, source=source)

        (bbox,) = source.calls
        xmin, ymin, xmax, ymax = gpd.read_parquet(chips).total_bounds
        assert bbox == pytest.approx((xmin, ymin, xmax, ymax))

    @pytest.mark.parametrize("chips_per_read", [1, 2, 1000])
    def test_task_size_does_not_change_the_result(
        self, tmp_path: Path, chips_per_read: int
    ) -> None:
        chips, tile = two_chip_setup(tmp_path)
        source = LocalSource({"33U": {2021: str(tile)}})

        add_land_cover(chips, year=2021, source=source, chips_per_read=chips_per_read, workers=2)

        rows = read_land_cover(chips)
        assert rows["ftw-33UXP0001"][4] == pytest.approx(71.43, abs=0.01)
        assert rows["ftw-33UXP0002"][4] == 100.0

    def test_rerun_replaces_columns(self, tmp_path: Path) -> None:
        chips, tile = two_chip_setup(tmp_path)
        source = LocalSource({"33U": {2021: str(tile), 2023: str(tile)}})

        add_land_cover(chips, year=2021, source=source)
        add_land_cover(chips, year=2025, source=source)

        cols = list(gpd.read_parquet(chips).columns)
        assert cols.count("landcover_year") == 1
        assert read_land_cover(chips)["ftw-33UXP0001"][:2] == (2023, False)

    def test_column_types(self, tmp_path: Path) -> None:
        chips, tile = two_chip_setup(tmp_path)

        add_land_cover(chips, year=2021, source=LocalSource({"33U": {2021: str(tile)}}))

        con = duckdb.connect()
        types = {
            row[0]: row[1]
            for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{chips}')").fetchall()
        }
        con.close()
        assert types["landcover_year"] == "INTEGER"
        assert types["landcover_year_exact"] == "BOOLEAN"
        assert types["landcover_dominant_code"] == "BIGINT"
        assert types["landcover_dominant_pct"] == "DOUBLE"
        assert types["landcover_classes"].startswith('STRUCT(code BIGINT, "name" VARCHAR')

    def test_geoparquet_metadata_and_row_order_survive(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.geo import detect_crs, detect_geometry_column

        tile = write_tile(tmp_path / "tile.tif")
        chips = write_chips(
            tmp_path / "chips.parquet",
            {
                "ftw-33UXP0002": chip_polygon(80, 0, 20, 100),
                "ftw-33UXP0001": chip_polygon(0, 0, 70, 100),
            },
        )

        add_land_cover(chips, year=2021, source=LocalSource({"33U": {2021: str(tile)}}))

        assert detect_geometry_column(chips) == "geometry"
        assert detect_crs(chips, "geometry").authority_code in {"EPSG:4326", "OGC:CRS84"}
        ids = [row[0] for row in duckdb.sql(f"SELECT id FROM '{chips}'").fetchall()]
        assert ids == ["ftw-33UXP0001", "ftw-33UXP0002"]

    def test_skips_without_a_year(self, tmp_path: Path) -> None:
        chips, tile = two_chip_setup(tmp_path)
        source = LocalSource({"33U": {2021: str(tile)}})
        messages: list[str] = []

        result = add_land_cover(chips, year=None, source=source, on_progress=messages.append)

        assert result.skipped is True
        assert result.reason == land_cover.NO_YEAR_REASON
        assert result.chips_total == 2
        assert source.calls == []
        assert "landcover_year" not in gpd.read_parquet(chips).columns
        assert any("skipping land cover" in m for m in messages)

    def test_empty_chips_file(self, tmp_path: Path) -> None:
        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame({"id": [], "gzd": []}, geometry=[], crs="EPSG:4326").to_parquet(chips)
        source = LocalSource({})

        result = add_land_cover(chips, year=2021, source=source)

        assert (result.chips_total, result.chips_with_land_cover) == (0, 0)
        assert source.calls == []
        assert "landcover_classes" in [
            row[0] for row in duckdb.sql(f"DESCRIBE FROM '{chips}'").fetchall()
        ]

    def test_rejects_projected_chips(self, tmp_path: Path) -> None:
        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["a"]}, geometry=[box(X0, Y0 - 100, X0 + 100, Y0)], crs=UTM
        ).to_parquet(chips)

        with pytest.raises(ValueError, match="EPSG:4326"):
            add_land_cover(chips, year=2021, source=LocalSource({}))

    def test_rejects_a_missing_id_column(self, tmp_path: Path) -> None:
        chips, _tile = two_chip_setup(tmp_path)

        with pytest.raises(ValueError, match="no 'chip' column"):
            add_land_cover(chips, year=2021, source=LocalSource({}), chips_id_col="chip")

    def test_read_failure_propagates_and_leaves_the_file_intact(self, tmp_path: Path) -> None:
        chips, _tile = two_chip_setup(tmp_path)
        before = chips.read_bytes()

        with pytest.raises(rasterio.errors.RasterioIOError):
            add_land_cover(
                chips, year=2021, source=LocalSource({"33U": {2021: str(tmp_path / "gone.tif")}})
            )

        assert chips.read_bytes() == before


class TestDropLandCover:
    def test_removes_the_columns(self, tmp_path: Path) -> None:
        chips, tile = two_chip_setup(tmp_path)
        add_land_cover(chips, year=2021, source=LocalSource({"33U": {2021: str(tile)}}))

        assert drop_land_cover(chips) is True

        assert list(gpd.read_parquet(chips).columns) == ["id", "gzd", "geometry"]

    def test_is_a_no_op_without_them(self, tmp_path: Path) -> None:
        chips, _tile = two_chip_setup(tmp_path)
        before = chips.read_bytes()

        assert drop_land_cover(chips) is False
        assert chips.read_bytes() == before


class TestLandCoverSummary:
    def test_disabled(self) -> None:
        assert land_cover_summary(None) == "Land cover: disabled"

    def test_skipped(self) -> None:
        result = LandCoverResult(10, 0, 0, (), None, skipped=True, reason="no dataset year")

        assert land_cover_summary(result) == "Land cover: skipped (no dataset year)"

    def test_exact(self) -> None:
        result = LandCoverResult(10, 9, 0, (2021,), 2021, skipped=False)

        assert land_cover_summary(result) == "Land cover: 9/10 chips from IO 2021"

    def test_nearest(self) -> None:
        result = LandCoverResult(5800, 5770, 5770, (2023,), 2025, skipped=False)

        assert land_cover_summary(result) == (
            "Land cover: 5,770/5,800 chips from IO 2023 (5,770 nearest-year for dataset year 2025)"
        )

    def test_mixed_years(self) -> None:
        result = LandCoverResult(3, 3, 1, (2019, 2020), 2020, skipped=False)

        assert land_cover_summary(result).startswith("Land cover: 3/3 chips from IO 2019, 2020")

    def test_no_data(self) -> None:
        result = LandCoverResult(4, 0, 0, (), 2021, skipped=False)

        assert land_cover_summary(result) == "Land cover: 0/4 chips (no IO data for this area)"


@pytest.mark.network
class TestPlanetaryComputer:
    def test_one_real_chip(self, tmp_path: Path) -> None:
        """A 2 km chip near Linz, Austria reads the real IO 2021 map."""
        chips = write_chips(
            tmp_path / "chips.parquet", {"ftw-33UVP0001": box(14.28, 48.29, 14.31, 48.31)}
        )

        result = add_land_cover(chips, year=2021)

        assert (result.chips_with_land_cover, result.years_used) == (1, (2021,))
        year, exact, code, name, _pct, classes = read_land_cover(chips)["ftw-33UVP0001"]
        assert (year, exact) == (2021, True)
        assert name == land_cover.IO_CLASSES[code]
        assert sum(c["pct"] for c in classes) == pytest.approx(100.0, abs=0.1)
