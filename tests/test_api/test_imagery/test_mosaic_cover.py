"""Quarters built from several mosaic tiles, for chips no single tile contains.

Every test runs on a local index and synthetic rasters: two UTM 32N tiles, A south
of northing 5551 km and B north of 5550.94 km, overlapping by 60 m as the mirror's
neighbouring tiles do.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from unittest.mock import patch

import duckdb
import numpy as np
import pystac
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.warp import transform_bounds
from shapely.geometry import shape

from ftw_dataset_tools.api.chip_grid import utm_grid_for_cell
from ftw_dataset_tools.api.imagery import mosaic_search
from ftw_dataset_tools.api.imagery.download_workflow import download_imagery_for_catalog
from ftw_dataset_tools.api.imagery.image_download import (
    download_and_clip_scene,
    read_band_from_tiles,
)
from ftw_dataset_tools.api.imagery.mosaic_selection import select_mosaics_for_chip
from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene
from ftw_dataset_tools.api.imagery.selection_workflow import (
    SkipCode,
    select_imagery_modes_for_catalog,
)
from ftw_dataset_tools.api.imagery.slots import MOSAIC_SLOTS, MOSAIC_SOURCE

if TYPE_CHECKING:
    from pathlib import Path

_CRS = "EPSG:32632"
_BANDS = ("B02", "B03", "B04", "B08")
_A_VALUE, _B_VALUE = 1000, 2000
_TILE_A = (500000.0, 5500000.0, 700000.0, 5551000.0)
_TILE_B = (500000.0, 5550940.0, 700000.0, 5650000.0)

# Chips (south-west corner in UTM 32N, 2 km square).
_STRADDLING = (600000.0, 5549500.0)  # 1440 m in A only, 60 m in both, 500 m in B only
_INSIDE_A = (590000.0, 5545000.0)


def _chip_geometry(x0: float, y0: float) -> tuple[dict, tuple[float, float, float, float]]:
    to_lonlat = Transformer.from_crs(_CRS, "EPSG:4326", always_xy=True)
    corners = [(x0, y0), (x0 + 2000, y0), (x0 + 2000, y0 + 2000), (x0, y0 + 2000)]
    ring = [list(to_lonlat.transform(x, y)) for x, y in corners]
    lons, lats = zip(*ring, strict=True)
    return {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}, (
        min(lons),
        min(lats),
        max(lons),
        max(lats),
    )


def _write_raster(path: Path, bounds: tuple[float, float, float, float], data: np.ndarray) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=data.shape[1],
        height=data.shape[0],
        count=1,
        dtype="int16",
        crs=_CRS,
        transform=from_origin(bounds[0], bounds[3], 10.0, 10.0),
        nodata=-32768,
    ) as dst:
        dst.write(data, 1)


def _tile_rasters(
    directory: Path, name: str, bounds: tuple[float, float, float, float], value: int
) -> dict[str, str]:
    """One raster per band over ``bounds`` (a window of the tile around the chips)."""
    directory.mkdir(parents=True, exist_ok=True)
    width = round((bounds[2] - bounds[0]) / 10)
    height = round((bounds[3] - bounds[1]) / 10)
    hrefs = {}
    for band in _BANDS:
        path = directory / f"{name}_{band}.tif"
        _write_raster(path, bounds, np.full((height, width), value, dtype=np.int16))
        hrefs[band] = path.as_uri()
    return hrefs


def _local(uri: str) -> str:
    return uri.removeprefix("file://")


def _write_index(path: Path, tiles: list[dict]) -> None:
    con = duckdb.connect()
    con.execute(
        'CREATE TABLE t (_subtile VARCHAR, "proj:code" VARCHAR, "proj:bbox" DOUBLE[], '
        "bbox DOUBLE[], start_datetime TIMESTAMPTZ, end_datetime TIMESTAMPTZ, "
        "B02_href VARCHAR, B03_href VARCHAR, B04_href VARCHAR, B08_href VARCHAR)"
    )
    for tile in tiles:
        lonlat = transform_bounds(tile.get("crs", _CRS), "EPSG:4326", *tile["proj_bbox"])
        con.execute(
            "INSERT INTO t VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [
                tile["subtile"],
                tile.get("crs", _CRS),
                list(tile["proj_bbox"]),
                list(lonlat),
                tile.get("start", datetime(2024, 1, 1, tzinfo=UTC)),
                tile.get("end", datetime(2024, 3, 31, 23, 59, 59, tzinfo=UTC)),
                *(tile["hrefs"][band] for band in _BANDS),
            ],
        )
    con.execute(f"COPY t TO '{path.as_posix()}' (FORMAT PARQUET)")
    con.close()


def _use_index(monkeypatch: pytest.MonkeyPatch, path: Path) -> None:
    monkeypatch.setattr(mosaic_search, "index_url", lambda _year, _quarter: path.as_posix())
    monkeypatch.setattr(mosaic_search, "_indexes", {})


@pytest.fixture
def two_tiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, dict[str, str]]:
    """The A/B index, with rasters covering both chips; returns each tile's hrefs."""
    rasters = tmp_path / "rasters"
    hrefs = {
        "A": _tile_rasters(rasters, "A", (588000.0, 5543000.0, 604000.0, 5551000.0), _A_VALUE),
        "B": _tile_rasters(rasters, "B", (588000.0, 5550940.0, 604000.0, 5556000.0), _B_VALUE),
    }
    index = tmp_path / "index.parquet"
    _write_index(
        index,
        [
            {"subtile": "32UNA_0_0", "proj_bbox": _TILE_A, "hrefs": hrefs["A"]},
            {"subtile": "32UNB_0_0", "proj_bbox": _TILE_B, "hrefs": hrefs["B"]},
        ],
    )
    _use_index(monkeypatch, index)
    return hrefs


def _select(chip_id: str, corner: tuple[float, float], nodata_max: float = 0.0):
    footprint, bbox = _chip_geometry(*corner)
    with patch("ftw_dataset_tools.api.imagery.mosaic_selection.year_available", return_value=True):
        return select_mosaics_for_chip(chip_id, bbox, 2024, nodata_max, footprint), bbox, footprint


def _download(item: pystac.Item, bbox, footprint, output: Path):
    item.properties["ftw:source"] = MOSAIC_SOURCE
    scene = SelectedScene(item=item, season="q1", cloud_cover=0.0, datetime=None, stac_url="")
    return download_and_clip_scene(
        scene=scene, bbox=bbox, output_path=output, bands=["red"], cell=shape(footprint)
    )


class TestTwoTileCover:
    def test_a_chip_neither_tile_contains_gets_both(self, two_tiles) -> None:
        result, _bbox, _footprint = _select("chip_001", _STRADDLING)
        assert result.success, result.skipped_reason
        q1 = result.quarters["q1"].item
        assert q1.properties["ftw:mosaic_tile"] == "32UNA_0_0"
        assert q1.properties["ftw:mosaic_tiles"] == ["32UNA_0_0", "32UNB_0_0"]
        assert q1.assets["red"].href == two_tiles["A"]["B04"]
        assert q1.assets["red"].extra_fields[mosaic_search.MOSAIC_HREFS] == [
            two_tiles["A"]["B04"],
            two_tiles["B"]["B04"],
        ]
        assert result.quarters["q1"].nodata_pct == 0.0

    @pytest.mark.usefixtures("two_tiles")
    def test_each_pixel_comes_from_one_tile_primary_first(self, tmp_path) -> None:
        result, bbox, footprint = _select("chip_001", _STRADDLING)
        output = tmp_path / "chip_001" / "chip_001_q1_image_s2.tif"
        download = _download(result.quarters["q1"].item, bbox, footprint, output)
        assert download.success, download.error

        with rasterio.open(output) as ds:
            assert ds.crs.to_string() == _CRS
            assert (ds.width, ds.height) == (200, 200)
            data = ds.read(1)
        # Rows top-down from northing 5551500: 50 in B only, 6 in the overlap (A, the
        # primary, wins) and 144 in A only. Bilinear warping only ever blends within
        # one tile, so no pixel holds anything but the two tiles' values.
        assert set(np.unique(data)) == {_A_VALUE, _B_VALUE}
        assert (data[:50] == _B_VALUE).all()
        assert (data[50:] == _A_VALUE).all()

    def test_a_gap_in_the_primary_is_filled_from_the_other_tile(self, two_tiles, tmp_path) -> None:
        # Blank A's overlap rows (northing 5550940-5551000) under the chip.
        with rasterio.open(_local(two_tiles["A"]["B04"]), "r+") as ds:
            data = ds.read(1)
            data[:6] = -32768
            ds.write(data, 1)
        result, bbox, footprint = _select("chip_001", _STRADDLING)
        output = tmp_path / "chip_001" / "chip_001_q1_image_s2.tif"
        assert _download(result.quarters["q1"].item, bbox, footprint, output).success
        with rasterio.open(output) as ds:
            data = ds.read(1)
        assert (data[:56] == _B_VALUE).all()
        assert (data[56:] == _A_VALUE).all()

    def test_nodata_is_measured_on_the_combined_image(self, two_tiles) -> None:
        # A gap only A covers: 100 x 200 m (10 x 20 pixels of the 200 x 200 chip).
        for href in (two_tiles["A"]["B08"],):
            with rasterio.open(_local(href), "r+") as ds:
                data = ds.read(1)
                # Raster origin (588000, 5551000); chip rows 5550500..5550400, cols 600000..
                data[50:60, 1200:1220] = -32768
                ds.write(data, 1)
        result, _bbox, _footprint = _select("chip_001", _STRADDLING, nodata_max=100.0)
        assert result.success
        assert result.quarters["q1"].nodata_pct == pytest.approx(100 * 200 / 40000)
        assert result.quarters["q2"].nodata_pct == pytest.approx(100 * 200 / 40000)

    def test_the_child_records_every_tile_and_the_spanned_dates(
        self, tmp_path, monkeypatch
    ) -> None:
        rasters = tmp_path / "rasters"
        a = _tile_rasters(rasters, "A", (588000.0, 5543000.0, 604000.0, 5551000.0), _A_VALUE)
        b = _tile_rasters(rasters, "B", (588000.0, 5550940.0, 604000.0, 5556000.0), _B_VALUE)
        index = tmp_path / "index.parquet"
        _write_index(
            index,
            [
                {
                    "subtile": "32UNA_0_0",
                    "proj_bbox": _TILE_A,
                    "hrefs": a,
                    "start": datetime(2024, 1, 3, tzinfo=UTC),
                    "end": datetime(2024, 3, 31, tzinfo=UTC),
                },
                {
                    "subtile": "32UNB_0_0",
                    "proj_bbox": _TILE_B,
                    "hrefs": b,
                    "start": datetime(2024, 1, 1, tzinfo=UTC),
                    "end": datetime(2024, 3, 29, tzinfo=UTC),
                },
            ],
        )
        _use_index(monkeypatch, index)
        catalog = _catalog(tmp_path / "catalog", [("chip_001", _STRADDLING)])
        with patch(
            "ftw_dataset_tools.api.imagery.mosaic_selection.year_available", return_value=True
        ):
            select_imagery_modes_for_catalog(catalog, "mosaics", year=2024)

        chip_dir = catalog / "chips" / "32UNA" / "chip_001"
        child = json.loads((chip_dir / "chip_001_q1_s2.json").read_text())
        assert child["properties"]["ftw:mosaic_tile"] == "32UNA_0_0"
        assert child["properties"]["ftw:mosaic_tiles"] == ["32UNA_0_0", "32UNB_0_0"]
        assert child["properties"]["start_datetime"] == "2024-01-01T00:00:00Z"
        assert child["properties"]["end_datetime"] == "2024-03-31T00:00:00Z"
        assert child["assets"]["nir"]["ftw:mosaic_hrefs"] == [a["B08"], b["B08"]]


class TestCrossZoneCombine:
    def test_a_neighbour_zone_primary_is_warped_without_blending(self, tmp_path) -> None:
        """The western-edge case: the primary tile is in the next zone (33 here), and
        the chip's own zone fills the rest; only the two tiles' values come out."""
        footprint, bbox = _chip_geometry(*_STRADDLING)
        # Zone 33 raster over the chip's south part (to northing ~5550.5 km in zone 32).
        west, south, east, north = transform_bounds(
            _CRS, "EPSG:32633", 599000.0, 5549000.0, 603000.0, 5550500.0
        )
        primary = tmp_path / "primary.tif"
        width, height = round((east - west) / 10), round((north - south) / 10)
        with rasterio.open(
            primary,
            "w",
            driver="GTiff",
            width=width,
            height=height,
            count=1,
            dtype="int16",
            crs="EPSG:32633",
            transform=from_origin(west, north, 10.0, 10.0),
            nodata=-32768,
        ) as dst:
            dst.write(np.full((height, width), _A_VALUE, dtype=np.int16), 1)
        secondary = tmp_path / "secondary.tif"
        _write_raster(
            secondary,
            (599000.0, 5549000.0, 603000.0, 5552000.0),
            np.full((300, 400), _B_VALUE, dtype=np.int16),
        )
        grid = utm_grid_for_cell("chip_001", bbox, "EPSG:4326", 10.0, shape(footprint))
        _, data = read_band_from_tiles("red", [str(primary), str(secondary)], *grid)
        assert set(np.unique(data)) == {_A_VALUE, _B_VALUE}
        assert (data[-1] == _A_VALUE).all()
        assert (data[0] == _B_VALUE).all()


class TestStillUncovered:
    def test_a_gap_between_the_tiles_leaves_the_chip_skipped(self, tmp_path, monkeypatch) -> None:
        index = tmp_path / "index.parquet"
        hrefs = dict.fromkeys(_BANDS, "unread.tif")
        _write_index(
            index,
            [
                {"subtile": "32UNA_0_0", "proj_bbox": _TILE_A, "hrefs": hrefs},
                # Starts 100 m north of A's edge: a strip of the chip is in neither.
                {
                    "subtile": "32UNB_0_0",
                    "proj_bbox": (500000.0, 5551100.0, 700000.0, 5650000.0),
                    "hrefs": hrefs,
                },
            ],
        )
        _use_index(monkeypatch, index)
        result, _bbox, _footprint = _select("chip_001", _STRADDLING)
        assert not result.success
        assert result.skipped_reason == (
            "No complete 2024 mosaics: Q1 no mosaic tile, nor any set of tiles, covers the chip"
        )


class TestTieBreak:
    @staticmethod
    def _sets(monkeypatch, tmp_path, tiles: list[dict], crs: str = _CRS) -> list[list[str]]:
        index = tmp_path / "index.parquet"
        _write_index(index, tiles)
        _use_index(monkeypatch, index)
        footprint, bbox = _chip_geometry(*_STRADDLING)
        sets = mosaic_search.covering_tile_sets(bbox, 2024, 1, footprint, crs)
        return [item.properties["ftw:mosaic_tiles"] for item in sets]

    def test_larger_coverage_is_primary_and_ids_break_ties(self, tmp_path, monkeypatch) -> None:
        hrefs = dict.fromkeys(_BANDS, "unread.tif")
        tiles = [
            {"subtile": "b_north", "proj_bbox": _TILE_B, "hrefs": hrefs},
            {"subtile": "a_north", "proj_bbox": _TILE_B, "hrefs": hrefs},
            {"subtile": "south", "proj_bbox": _TILE_A, "hrefs": hrefs},
        ]
        expected = [["south", "a_north"], ["south", "b_north"]]
        assert self._sets(monkeypatch, tmp_path, tiles) == expected
        # Index order never matters.
        assert self._sets(monkeypatch, tmp_path, tiles[::-1]) == expected

    def test_a_tile_in_the_chips_own_crs_is_primary(self, tmp_path, monkeypatch) -> None:
        """The own-zone tile covers less of the chip but still supplies its pixels first."""
        hrefs = dict.fromkeys(_BANDS, "unread.tif")
        # A zone 33 tile covering the chip's south ~72%, and a tile of the chip's own
        # zone 32 its north 35%.
        to_33 = Transformer.from_crs(_CRS, "EPSG:32633", always_xy=True)
        corners = [to_33.transform(x, 5551000.0) for x in (600000.0, 602000.0)]
        northing = min(y for _x, y in corners)
        other = {
            "subtile": "33UUR_0_0",
            "crs": "EPSG:32633",
            "proj_bbox": (-500000.0, 5400000.0, 500000.0, northing),
            "hrefs": hrefs,
        }
        own_north = {
            "subtile": "32UNB_0_0",
            "proj_bbox": (500000.0, 5550800.0, 700000.0, 5650000.0),
            "hrefs": hrefs,
        }
        assert self._sets(monkeypatch, tmp_path, [other, own_north]) == [["32UNB_0_0", "33UUR_0_0"]]
        # Measured from zone 33 the zone 33 tile ranks first.
        assert self._sets(monkeypatch, tmp_path, [other, own_north], crs="EPSG:32633") == [
            ["33UUR_0_0", "32UNB_0_0"]
        ]


class TestSingleTileUnchanged:
    # SHA-256 of the image and the Q1 child item below as written before multi-tile
    # quarters existed (3f3c9e9). They depend on the locked GDAL, rasterio and pystac:
    # after a bump of those, regenerate them by running this class on the code before
    # any change to the download path, then here.
    _GOLDEN_IMAGE = "b35daae997b0c57795fa8b8fb74079c7228dcf4cbcf609386a6e94861162d133"
    _GOLDEN_CHILD = "c038b20da8b002310bcc23c5970e85cfdae4e92c6162b38de153df76ae4b9bdc"

    @pytest.mark.usefixtures("two_tiles")
    def test_a_single_tile_child_item_is_byte_identical(self, tmp_path) -> None:
        catalog = _catalog(tmp_path / "catalog", [("chip_002", _INSIDE_A)])
        with patch(
            "ftw_dataset_tools.api.imagery.mosaic_selection.year_available", return_value=True
        ):
            select_imagery_modes_for_catalog(catalog, "mosaics", year=2024)
        child = catalog / "chips" / "32UNA" / "chip_002" / "chip_002_q1_s2.json"
        assert hashlib.sha256(child.read_bytes()).hexdigest() == self._GOLDEN_CHILD

    def test_a_single_tile_chip_comes_out_byte_identical(self, two_tiles, tmp_path) -> None:
        result, bbox, footprint = _select("chip_002", _INSIDE_A)
        q1 = result.quarters["q1"].item
        # A recognisable pattern, so a mis-warp would change the bytes.
        with rasterio.open(_local(two_tiles["A"]["B04"]), "r+") as ds:
            pattern = (np.arange(ds.width * ds.height) % 9973).astype(np.int16)
            ds.write(pattern.reshape(ds.height, ds.width), 1)
        output = tmp_path / "chip_002" / "chip_002_q1_image_s2.tif"
        assert _download(q1, bbox, footprint, output).success
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        assert digest == self._GOLDEN_IMAGE
        assert "ftw:mosaic_tiles" not in q1.properties
        assert mosaic_search.MOSAIC_HREFS not in q1.assets["red"].extra_fields


def _catalog(root: Path, chips: list[tuple[str, tuple[float, float]]]) -> Path:
    for chip_id, corner in chips:
        footprint, bbox = _chip_geometry(*corner)
        chip_dir = root / "chips" / "32UNA" / chip_id
        chip_dir.mkdir(parents=True)
        item = pystac.Item(
            id=chip_id,
            geometry=footprint,
            bbox=list(bbox),
            datetime=datetime(2024, 1, 1, tzinfo=UTC),
            properties={},
        )
        path = chip_dir / f"{chip_id}.json"
        item.set_self_href(str(path))
        item.save_object(dest_href=str(path))
    return root


class TestRerunPicksUpSkippedChips:
    @pytest.mark.usefixtures("two_tiles")
    def test_a_rerun_selects_and_downloads_only_the_skipped_chip(
        self, tmp_path, monkeypatch
    ) -> None:
        """A dataset built before multi-tile quarters has the straddling chip skipped;
        rerunning selection and a resumed download fills it in and leaves the rest."""
        catalog = _catalog(
            tmp_path / "catalog", [("chip_inside", _INSIDE_A), ("chip_edge", _STRADDLING)]
        )
        # The child items store the local rasters' hrefs relative to themselves, as
        # pystac does with local paths (the mirror's are URLs). Every chip directory is
        # at the same depth, so reading them from one resolves them for all.
        monkeypatch.chdir(catalog / "chips" / "32UNA" / "chip_edge")
        available = patch(
            "ftw_dataset_tools.api.imagery.mosaic_selection.year_available", return_value=True
        )
        # The first build, as before this change: no set of tiles is ever tried.
        with (
            available,
            patch(
                "ftw_dataset_tools.api.imagery.mosaic_selection.covering_tile_sets",
                return_value=[],
            ),
        ):
            first = select_imagery_modes_for_catalog(catalog, "mosaics", year=2024)
        assert [d["chip"] for d in first.skipped_details] == ["chip_edge"]
        assert first.skipped_details[0]["code"] == SkipCode.NO_IMAGERY
        built = download_imagery_for_catalog(catalog, generate_thumbnails=False)
        assert built.successful == 4

        inside = catalog / "chips" / "32UNA" / "chip_inside"
        before = {path.name: path.read_bytes() for path in inside.iterdir()}

        with available:
            rerun = select_imagery_modes_for_catalog(catalog, "mosaics", year=2024)
        assert rerun.successful == 1
        assert [(d["chip"], d["code"]) for d in rerun.skipped_details] == [
            ("chip_inside", SkipCode.ALREADY_SELECTED)
        ]
        resumed = download_imagery_for_catalog(catalog, generate_thumbnails=False, resume=True)
        assert resumed.successful == 4
        assert resumed.skipped == 4
        assert {d["item"] for d in resumed.skipped_details} == {
            f"chip_inside_{slot}_s2" for slot in MOSAIC_SLOTS
        }

        edge = catalog / "chips" / "32UNA" / "chip_edge"
        for slot in MOSAIC_SLOTS:
            assert (edge / f"chip_edge_{slot}_image_s2.tif").exists()
        assert {path.name: path.read_bytes() for path in inside.iterdir()} == before


@pytest.mark.network
def test_a_chip_west_of_a_missing_square_is_covered_by_the_neighbour_zone() -> None:
    """ftw-35TKH5898_2021: the mirror has no 35TKH tile; 34TGN and 34TGP each miss the
    chip by 236 m and 1.8 km, and cover it together."""
    sets = mosaic_search.covering_tile_sets(
        (24.0158, 43.2959, 24.0413, 43.3145), 2021, 1, None, "EPSG:32635"
    )
    assert [item.properties["ftw:mosaic_tiles"] for item in sets] == [["34TGN_0_0", "34TGP_0_0"]]
