"""Tests for the quarterly mosaic imagery mode."""

from __future__ import annotations

import urllib.error
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import duckdb
import numpy as np
import pystac
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin

from ftw_dataset_tools.api.imagery import mosaic_search
from ftw_dataset_tools.api.imagery.catalog_ops import (
    SelectionConflictError,
    clear_chip_selections,
    has_existing_scenes,
    selection_conflict,
)
from ftw_dataset_tools.api.imagery.download_workflow import (
    build_download_task,
    find_s2_child_items,
)
from ftw_dataset_tools.api.imagery.image_download import (
    download_and_clip_scene,
    find_reference_mask_for_output,
)
from ftw_dataset_tools.api.imagery.mosaic_selection import (
    MosaicSelectionResult,
    MosaicYearError,
    SelectedMosaic,
    check_mosaic_year,
    select_mosaics_for_chip,
)
from ftw_dataset_tools.api.imagery.scene_selection import SelectedScene
from ftw_dataset_tools.api.imagery.selection_workflow import (
    ChipSelectionJob,
    _select_mode,
    find_chip_items,
    resolve_selection_conflicts,
    run_chip_selection,
)
from ftw_dataset_tools.api.imagery.slots import (
    MOSAIC_SOURCE,
    child_item_id,
    image_filename,
    parse_child_id,
    parse_image_stem,
)
from ftw_dataset_tools.api.imagery.stac_child_items import (
    attach_existing_seasons,
    create_mosaic_child_items,
)

CHIP_BBOX = (10.0, 50.0, 10.01, 50.01)

_SELECTION = "ftw_dataset_tools.api.imagery.mosaic_selection"


def _quarter_window(year: int, number: int) -> tuple[datetime, datetime]:
    start = datetime(year, 3 * number - 2, 1, tzinfo=UTC)
    end_month = 3 * number
    end = datetime(year, end_month, 30 if end_month in (6, 9) else 31, 23, 59, 59, tzinfo=UTC)
    return start, end


def _tile_item(year: int, number: int, subtile: str = "32UNA_0_0") -> pystac.Item:
    start, end = _quarter_window(year, number)
    item = pystac.Item(
        id=f"Sentinel-2_mosaic_{year}_Q{number}_{subtile}",
        geometry=None,
        bbox=None,
        datetime=None,
        start_datetime=start,
        end_datetime=end,
        properties={"ftw:mosaic_tile": subtile},
    )
    for band in ("red", "green", "blue", "nir"):
        item.add_asset(band, pystac.Asset(href=f"https://example.com/{year}/Q{number}/{band}.tif"))
    return item


def _selection(year: int = 2024, requested: int = 2024) -> MosaicSelectionResult:
    quarters = {
        f"q{n}": SelectedMosaic(item=_tile_item(year, n), quarter=f"q{n}", year=year)
        for n in (1, 2, 3, 4)
    }
    return MosaicSelectionResult(
        chip_id="chip_001",
        bbox=CHIP_BBOX,
        requested_year=requested,
        imagery_year=year,
        quarters=quarters,
    )


def _chip(tmp_path: Path, chip_id: str = "chip_001") -> tuple[pystac.Item, Path]:
    chip_dir = tmp_path / "chips" / "32UNA" / chip_id
    chip_dir.mkdir(parents=True)
    minx, miny, maxx, maxy = CHIP_BBOX
    item = pystac.Item(
        id=chip_id,
        geometry={
            "type": "Polygon",
            "coordinates": [[[minx, miny], [maxx, miny], [maxx, maxy], [minx, maxy], [minx, miny]]],
        },
        bbox=CHIP_BBOX,
        datetime=datetime(2024, 1, 1, tzinfo=UTC),
        properties={},
    )
    path = chip_dir / f"{chip_id}.json"
    item.set_self_href(str(path))
    item.save_object(dest_href=str(path))
    return item, path


class TestSlots:
    def test_child_ids_round_trip_for_every_slot(self) -> None:
        for slot in ("planting", "harvest", "q1", "q4"):
            assert parse_child_id(child_item_id("ftw-34UFF1628_2024", slot)) == (
                "ftw-34UFF1628_2024",
                slot,
            )

    def test_image_stems_round_trip(self) -> None:
        stem = image_filename("chip_001", "q3").removesuffix(".tif")
        assert parse_image_stem(stem) == ("chip_001", "q3")

    def test_chip_ids_are_not_children(self) -> None:
        assert parse_child_id("ftw-34UFF1628_2024") is None
        assert parse_child_id("_q1_s2") is None


class TestMosaicYear:
    def test_check_rejects_unavailable_year(self) -> None:
        with (
            patch(f"{_SELECTION}.year_available", return_value=False),
            pytest.raises(MosaicYearError, match="2023"),
        ):
            check_mosaic_year(2023)


@pytest.fixture
def local_index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the mosaic index at a local parquet with two UTM 32N tiles."""
    path = tmp_path / "index.parquet"
    con = duckdb.connect()
    con.execute(
        f"""
        COPY (
            SELECT * FROM (VALUES
                ('32UNA_0_0', 'EPSG:32632', [500000.0, 5500000.0, 600080.0, 5600000.0],
                 [9.0, 49.6, 10.4, 50.6]),
                ('32UNA_1_0', 'EPSG:32632', [600000.0, 5500000.0, 700080.0, 5600000.0],
                 [10.3, 49.6, 11.8, 50.6])
            ) AS t(_subtile, "proj:code", "proj:bbox", bbox)
            CROSS JOIN (SELECT
                TIMESTAMPTZ '2024-01-01 00:00:00+00' AS start_datetime,
                TIMESTAMPTZ '2024-03-31 23:59:59+00' AS end_datetime,
                'b02.tif' AS B02_href, 'b03.tif' AS B03_href,
                'b04.tif' AS B04_href, 'b08.tif' AS B08_href)
        ) TO '{path.as_posix()}' (FORMAT PARQUET)
        """
    )
    con.close()
    monkeypatch.setattr(mosaic_search, "index_url", lambda _year, _quarter: path.as_posix())
    monkeypatch.setattr(mosaic_search, "_indexes", {})


@pytest.mark.usefixtures("local_index")
class TestFindContainingTile:
    def test_returns_the_tile_containing_the_chip(self) -> None:
        item = mosaic_search.find_containing_tile((10.0, 50.0, 10.01, 50.01), 2024, 1)
        assert item is not None
        assert item.properties["ftw:mosaic_tile"] == "32UNA_0_0"
        assert set(item.assets) == {"red", "green", "blue", "nir"}
        assert item.assets["red"].href == "b04.tif"
        assert item.common_metadata.start_datetime == datetime(2024, 1, 1, tzinfo=UTC)

    def test_picks_the_subtile_that_holds_the_chip(self) -> None:
        # Around easting 640 km, inside only the second tile.
        item = mosaic_search.find_containing_tile((10.95, 50.0, 10.96, 50.01), 2024, 1)
        assert item is not None
        assert item.properties["ftw:mosaic_tile"] == "32UNA_1_0"

    def test_chip_across_a_tile_edge_has_no_tile(self) -> None:
        # Straddles easting 600.08 km, the end of the first tile.
        item = mosaic_search.find_containing_tile((10.39, 50.0, 10.41, 50.01), 2024, 1)
        assert item is None

    @staticmethod
    def _edge_chip() -> tuple[dict, tuple[float, float, float, float]]:
        """A 2 km UTM chip ending at easting 600.06 km, 20 m inside the first tile.

        Its lon/lat bbox taken back onto the grid ends near 600.10 km, past the tile.
        """
        to_lonlat = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
        corners = [(598060, 5550000), (600060, 5550000), (600060, 5552000), (598060, 5552000)]
        ring = [list(to_lonlat.transform(x, y)) for x, y in corners]
        lons, lats = zip(*ring, strict=True)
        footprint = {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}
        return footprint, (min(lons), min(lats), max(lons), max(lats))

    def test_chip_on_a_tile_edge_fits_by_its_footprint(self) -> None:
        footprint, bbox = self._edge_chip()
        item = mosaic_search.find_containing_tile(bbox, 2024, 1, footprint)
        assert item is not None
        assert item.properties["ftw:mosaic_tile"] == "32UNA_0_0"

    def test_chip_on_a_tile_edge_misses_by_its_bbox(self) -> None:
        """Why selection passes the footprint: the bbox envelope overhangs the tile."""
        _, bbox = self._edge_chip()
        assert mosaic_search.find_containing_tile(bbox, 2024, 1) is None

    @staticmethod
    def _utm_chip(x0: float, y0: float) -> tuple[dict, tuple[float, float, float, float]]:
        to_lonlat = Transformer.from_crs("EPSG:32632", "EPSG:4326", always_xy=True)
        corners = [(x0, y0), (x0 + 2000, y0), (x0 + 2000, y0 + 2000), (x0, y0 + 2000)]
        ring = [list(to_lonlat.transform(x, y)) for x, y in corners]
        lons, lats = zip(*ring, strict=True)
        footprint = {"type": "Polygon", "coordinates": [[*ring, ring[0]]]}
        return footprint, (min(lons), min(lats), max(lons), max(lats))

    def test_chip_exactly_on_a_tile_edge_fits(self) -> None:
        # Starts at easting 600 km, the second tile's west edge.
        footprint, bbox = self._utm_chip(600000, 5550000)
        item = mosaic_search.find_containing_tile(bbox, 2024, 1, footprint)
        assert item is not None
        assert item.properties["ftw:mosaic_tile"] == "32UNA_1_0"

    def test_chip_a_metre_past_every_tile_has_none(self) -> None:
        # Ends at easting 600.081 km, 1 m past the first tile; starts before the second.
        footprint, bbox = self._utm_chip(598081, 5550000)
        assert mosaic_search.find_containing_tile(bbox, 2024, 1, footprint) is None

    def test_the_tile_holding_the_chip_deepest_wins(self, monkeypatch) -> None:
        """Of two tiles that both contain the chip, the first in index order loses
        when the chip sits nearer its edge."""
        first, second = mosaic_search._load_index(2024, 1)
        lonlat = (9.0, 49.6, 11.8, 50.6)
        shallow = replace(
            first,
            subtile="shallow",
            bbox=lonlat,
            proj_bbox=(590000.0, 5500000.0, 606100.0, 5600000.0),
        )
        deep = replace(
            second,
            subtile="deep",
            bbox=lonlat,
            proj_bbox=(500000.0, 5500000.0, 700000.0, 5600000.0),
        )
        monkeypatch.setitem(mosaic_search._indexes, (2024, 1), [shallow, deep])
        footprint, bbox = self._utm_chip(604000, 5550000)
        item = mosaic_search.find_containing_tile(bbox, 2024, 1, footprint)
        assert item is not None
        assert item.properties["ftw:mosaic_tile"] == "deep"

    def test_an_own_crs_tile_the_chip_overhangs_never_wins(self, monkeypatch) -> None:
        first, second = mosaic_search._load_index(2024, 1)
        lonlat = (9.0, 49.6, 11.8, 50.6)
        short = replace(
            first,
            subtile="short",
            bbox=lonlat,
            proj_bbox=(590000.0, 5500000.0, 605000.0, 5600000.0),
        )
        other = replace(
            second,
            subtile="other",
            crs="EPSG:32633",
            bbox=lonlat,
            proj_bbox=(-1e6, 5000000.0, 1e6, 6000000.0),
        )
        footprint, bbox = self._utm_chip(604000, 5550000)
        monkeypatch.setitem(mosaic_search._indexes, (2024, 1), [short, other])
        item = mosaic_search.find_containing_tile(bbox, 2024, 1, footprint, "EPSG:32632")
        assert item is not None
        assert item.properties["ftw:mosaic_tile"] == "other"
        monkeypatch.setitem(mosaic_search._indexes, (2024, 1), [short])
        assert mosaic_search.find_containing_tile(bbox, 2024, 1, footprint, "EPSG:32632") is None

    def test_a_tile_in_the_chips_own_crs_wins(self, monkeypatch) -> None:
        """At a zone seam the chip's own zone beats a neighbour it sits deeper in."""
        first, second = mosaic_search._load_index(2024, 1)
        lonlat = (9.0, 49.6, 11.8, 50.6)
        own = replace(
            first, subtile="own", bbox=lonlat, proj_bbox=(590000.0, 5500000.0, 606100.0, 5600000.0)
        )
        other = replace(
            second,
            subtile="other",
            crs="EPSG:32633",
            bbox=lonlat,
            proj_bbox=(-1e6, 5000000.0, 1e6, 6000000.0),
        )
        monkeypatch.setitem(mosaic_search._indexes, (2024, 1), [other, own])
        footprint, bbox = self._utm_chip(604000, 5550000)
        tiles = mosaic_search.containing_tiles(bbox, 2024, 1, footprint, "EPSG:32632")
        assert [t.properties["ftw:mosaic_tile"] for t in tiles] == ["own", "other"]
        assert (
            mosaic_search.find_containing_tile(bbox, 2024, 1, footprint).properties[
                "ftw:mosaic_tile"
            ]
            == "other"
        )


class TestSelectMosaicsForChip:
    def test_selects_all_four_quarters_of_the_requested_year(self) -> None:
        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(
                f"{_SELECTION}.containing_tiles",
                side_effect=lambda _b, y, n, *_: [_tile_item(y, n)],
            ),
            patch(f"{_SELECTION}.calculate_nodata_percentage", return_value=0.0),
        ):
            result = select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024)
        assert result.success
        assert result.imagery_year == 2024
        assert sorted(result.quarters) == ["q1", "q2", "q3", "q4"]

    def test_any_nodata_leaves_the_chip_without_mosaics(self) -> None:
        """Never a nearby year: the imagery must show the labels' year."""

        def nodata(href: str, _bbox: object, _footprint: object) -> float:
            return 0.5 if "/2024/Q2/" in href else 0.0

        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(
                f"{_SELECTION}.containing_tiles",
                side_effect=lambda _b, y, n, *_: [_tile_item(y, n)],
            ),
            patch(f"{_SELECTION}.calculate_nodata_percentage", side_effect=nodata),
        ):
            result = select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024)
        assert not result.success
        assert result.quarters == {}
        assert result.skipped_reason == (
            "No complete 2024 mosaics: Q2 nodata in chip window (32UNA_0_0 0.50%)"
        )

    def test_footprint_reaches_the_tile_and_nodata_checks(self) -> None:
        footprint = {
            "type": "Polygon",
            "coordinates": [[[10.0, 50.0], [10.01, 50.0], [10.0, 50.0]]],
        }
        tiles, nodata = (
            MagicMock(side_effect=lambda _b, y, n, *_: [_tile_item(y, n)]),
            MagicMock(return_value=0.0),
        )
        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(f"{_SELECTION}.containing_tiles", tiles),
            patch(f"{_SELECTION}.calculate_nodata_percentage", nodata),
        ):
            select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024, footprint=footprint)
        assert all(c.args[3] is footprint for c in tiles.call_args_list)
        assert all(c.args[4] == "EPSG:32632" for c in tiles.call_args_list)
        assert all(c.args[2] is footprint for c in nodata.call_args_list)

    def test_an_ftw_chip_asks_for_its_own_zone(self) -> None:
        """The zone comes from the chip id, even where the chip's centre is in another."""
        tiles = MagicMock(side_effect=lambda _b, y, n, *_: [_tile_item(y, n)])
        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(f"{_SELECTION}.containing_tiles", tiles),
            patch(f"{_SELECTION}.calculate_nodata_percentage", return_value=0.0),
        ):
            # CHIP_BBOX's centre is in zone 32.
            select_mosaics_for_chip("ftw-33UUR0000_2024", CHIP_BBOX, 2024)
        assert all(c.args[4] == "EPSG:32633" for c in tiles.call_args_list)

    def test_a_neighbouring_tile_stands_in_for_one_with_nodata(self) -> None:
        """The chip's own-zone tile has a gap over it; the other zone's tile has none."""
        own, other = _tile_item(2024, 2), _tile_item(2024, 2, subtile="33UUR_0_0")
        other.assets["nir"].href = "https://example.com/other/nir.tif"

        def tiles(_b, y, n, *_):
            return [own, other] if (y, n) == (2024, 2) else [_tile_item(y, n)]

        def nodata(href: str, *_: object) -> float:
            return 30.0 if href == own.assets["nir"].href else 0.0

        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(f"{_SELECTION}.containing_tiles", side_effect=tiles),
            patch(f"{_SELECTION}.calculate_nodata_percentage", side_effect=nodata),
        ):
            result = select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024)
        assert result.imagery_year == 2024
        assert result.quarters["q2"].item is other
        assert result.candidates_checked == 5

    def test_the_year_fails_only_when_every_tile_has_nodata(self) -> None:
        own, other = _tile_item(2024, 1), _tile_item(2024, 1, subtile="33UUR_0_0")
        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(f"{_SELECTION}.containing_tiles", return_value=[own, other]),
            patch(f"{_SELECTION}.calculate_nodata_percentage", side_effect=[30.0, 5.0]),
        ):
            result = select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024)
        assert not result.success
        assert result.skipped_reason == (
            "No complete 2024 mosaics: Q1 nodata in chip window (32UNA_0_0 30.00%, 33UUR_0_0 5.00%)"
        )

    def test_with_gaps_allowed_the_tile_with_the_fewest_wins(self) -> None:
        """At nodata_max=100 the own-zone tile's gap is no reason to keep it."""
        gappy, clean = _tile_item(2024, 1), _tile_item(2024, 1, subtile="33UUR_0_0")
        clean.assets["nir"].href = "https://example.com/clean/nir.tif"

        def tiles(_b, y, n, *_):
            return [gappy, clean] if n == 1 else [_tile_item(y, n)]

        def nodata(href: str, *_: object) -> float:
            return 3.0 if href == gappy.assets["nir"].href else 0.0

        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(f"{_SELECTION}.containing_tiles", side_effect=tiles),
            patch(f"{_SELECTION}.calculate_nodata_percentage", side_effect=nodata),
        ):
            result = select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024, nodata_max=100.0)
        assert result.quarters["q1"].item is clean

    def test_with_gaps_allowed_a_gappy_tile_is_kept_when_it_is_the_best(self) -> None:
        own, other = _tile_item(2024, 1), _tile_item(2024, 1, subtile="33UUR_0_0")
        other.assets["nir"].href = "https://example.com/other/nir.tif"

        def tiles(_b, y, n, *_):
            return [own, other] if n == 1 else [_tile_item(y, n)]

        def nodata(href: str, *_: object) -> float:
            return 2.0 if href == own.assets["nir"].href else (7.0 if "other" in href else 0.0)

        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(f"{_SELECTION}.containing_tiles", side_effect=tiles),
            patch(f"{_SELECTION}.calculate_nodata_percentage", side_effect=nodata),
        ):
            result = select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024, nodata_max=100.0)
        assert result.success
        assert result.quarters["q1"].item is own

    def test_on_equal_nodata_the_better_ranked_tile_wins(self) -> None:
        own, other = _tile_item(2024, 1), _tile_item(2024, 1, subtile="33UUR_0_0")
        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(f"{_SELECTION}.containing_tiles", return_value=[own, other]),
            patch(f"{_SELECTION}.calculate_nodata_percentage", return_value=2.0),
        ):
            result = select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024, nodata_max=100.0)
        assert result.quarters["q1"].item is own

    def test_a_clean_tile_stops_the_search(self) -> None:
        clean, other = _tile_item(2024, 1), _tile_item(2024, 1, subtile="33UUR_0_0")
        nodata = MagicMock(return_value=0.0)
        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(f"{_SELECTION}.containing_tiles", return_value=[clean, other]),
            patch(f"{_SELECTION}.calculate_nodata_percentage", nodata),
        ):
            result = select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024)
        assert nodata.call_count == 4  # one read per quarter
        assert result.candidates_checked == 4

    def test_an_unreadable_tile_falls_through_to_the_next(self) -> None:
        own, other = _tile_item(2024, 1), _tile_item(2024, 1, subtile="33UUR_0_0")

        def tiles(_b, y, n, *_):
            return [own, other] if n == 1 else [_tile_item(y, n)]

        nodata = MagicMock(side_effect=[OSError("HTTP 503"), 0.0, 0.0, 0.0, 0.0])
        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(f"{_SELECTION}.containing_tiles", side_effect=tiles),
            patch(f"{_SELECTION}.calculate_nodata_percentage", nodata),
        ):
            result = select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024)
        assert result.success
        assert result.quarters["q1"].item is other

    def test_a_read_error_with_no_clean_tile_fails_the_chip(self) -> None:
        """Raised, not skipped: a rerun retries a failed chip but not a skipped one."""
        own, other = _tile_item(2024, 1), _tile_item(2024, 1, subtile="33UUR_0_0")
        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(f"{_SELECTION}.containing_tiles", return_value=[own, other]),
            patch(
                f"{_SELECTION}.calculate_nodata_percentage",
                side_effect=[OSError("HTTP 503"), 5.0],
            ),
            pytest.raises(OSError, match="503"),
        ):
            select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024)

    def test_skips_when_no_tile_covers_the_chip(self) -> None:
        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(f"{_SELECTION}.containing_tiles", return_value=[]),
        ):
            result = select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024)
        assert not result.success
        assert (
            result.skipped_reason
            == "No complete 2024 mosaics: no single mosaic tile covers the chip"
        )

    def test_an_unavailable_year_is_skipped_not_replaced(self) -> None:
        tiles = MagicMock()
        with (
            patch(f"{_SELECTION}.year_available", side_effect=lambda y: y != 2026),
            patch(f"{_SELECTION}.containing_tiles", tiles),
        ):
            result = select_mosaics_for_chip("chip_001", CHIP_BBOX, 2026)
        assert not result.success
        assert result.skipped_reason == "No mosaics for 2026"
        tiles.assert_not_called()


class TestMosaicChildItems:
    def test_writes_four_children_and_the_parent_record(self, tmp_path: Path) -> None:
        parent, path = _chip(tmp_path)
        create_mosaic_child_items(path.parent, parent, _selection(year=2025, requested=2024))

        saved = pystac.Item.from_file(str(path))
        assert saved.properties["ftw:imagery_modes"] == ["mosaics"]
        assert saved.properties["ftw:mosaic_requested_year"] == 2024
        assert saved.properties["ftw:mosaic_year"] == 2025
        assert "ftw:imagery_mode" not in saved.properties
        assert "ftw:requested_year" not in saved.properties
        assert saved.properties["start_datetime"].startswith("2025-01-01")
        assert saved.properties["end_datetime"].startswith("2025-12-31")
        assert {link.rel for link in saved.links} >= {"ftw:q1", "ftw:q2", "ftw:q3", "ftw:q4"}
        assert has_existing_scenes(saved)

        child = pystac.Item.from_file(str(path.parent / "chip_001_q3_s2.json"))
        assert child.properties["ftw:season"] == "q3"
        assert child.properties["ftw:source"] == MOSAIC_SOURCE
        assert child.properties["ftw:imagery_year"] == 2025
        assert child.datetime is None
        assert set(child.assets) == {"red", "green", "blue", "nir"}

    def test_children_are_not_mistaken_for_chips(self, tmp_path: Path) -> None:
        parent, path = _chip(tmp_path)
        create_mosaic_child_items(path.parent, parent, _selection())
        assert [item.id for item, _p in find_chip_items(tmp_path)] == ["chip_001"]

    def test_catalog_rebuild_reattaches_quarters(self, tmp_path: Path) -> None:
        parent, path = _chip(tmp_path)
        create_mosaic_child_items(path.parent, parent, _selection())
        fresh = pystac.Item.from_file(str(path))
        fresh.links = [link for link in fresh.links if not link.rel.startswith("ftw:q")]
        assert attach_existing_seasons(fresh, path.parent) == ["q1", "q2", "q3", "q4"]

    def test_clear_removes_quarters_and_mosaic_properties(self, tmp_path: Path) -> None:
        parent, path = _chip(tmp_path)
        create_mosaic_child_items(path.parent, parent, _selection())
        saved = pystac.Item.from_file(str(path))
        result = clear_chip_selections(saved)
        assert result.stac_items_deleted == 4
        assert not list(path.parent.glob("*_q*_s2.json"))
        assert "ftw:imagery_mode" not in saved.properties
        assert not has_existing_scenes(saved)


class TestSelectionConflicts:
    def _mosaic_chip(self, tmp_path: Path, requested: int = 2024) -> pystac.Item:
        parent, path = _chip(tmp_path)
        create_mosaic_child_items(path.parent, parent, _selection(requested=requested))
        return pystac.Item.from_file(str(path))

    def test_same_mode_and_year_is_no_conflict(self, tmp_path: Path) -> None:
        assert selection_conflict(self._mosaic_chip(tmp_path), "mosaics", 2024) is None

    def test_other_year_conflicts(self, tmp_path: Path) -> None:
        reason = selection_conflict(self._mosaic_chip(tmp_path), "mosaics", 2025)
        assert reason == "has mosaics for 2024"

    def test_scenes_over_mosaics_do_not_conflict(self, tmp_path: Path) -> None:
        assert selection_conflict(self._mosaic_chip(tmp_path), "scenes", 2024) is None

    def test_unselected_chip_never_conflicts(self, tmp_path: Path) -> None:
        parent, _path = _chip(tmp_path)
        assert selection_conflict(parent, "mosaics", 2025) is None

    def test_conflict_raises_without_force(self, tmp_path: Path) -> None:
        chip = self._mosaic_chip(tmp_path)
        with pytest.raises(SelectionConflictError, match="mosaics for 2024"):
            resolve_selection_conflicts([chip], imagery_modes="mosaics", year=2025, force=False)

    def test_force_clears_the_conflicting_chips(self, tmp_path: Path) -> None:
        chip = self._mosaic_chip(tmp_path)
        cleared = resolve_selection_conflicts(
            [chip], imagery_modes="mosaics", year=2025, force=True
        )
        assert cleared == 1
        assert not has_existing_scenes(chip)


class TestRunChipSelection:
    def test_mosaic_mode_selects_and_writes_children(self, tmp_path: Path) -> None:
        parent, path = _chip(tmp_path)
        job = ChipSelectionJob(item=parent, item_path=path, year=2024)
        with patch(
            "ftw_dataset_tools.api.imagery.selection_workflow.select_mosaics_for_chip",
            return_value=_selection(),
        ) as select:
            result = run_chip_selection(
                job,
                cloud_cover_chip=2.0,
                nodata_max=0.0,
                buffer_days=14,
                num_buffer_expansions=3,
                buffer_expansion_size=14,
                imagery_mode="mosaics",
            )
        assert result.success
        assert select.call_args.kwargs["nodata_max"] == 0.0
        assert select.call_args.kwargs["footprint"] == parent.geometry
        assert (path.parent / "chip_001_q1_s2.json").exists()


class TestMosaicDownload:
    def test_download_task_uses_the_quarter_slot(self, tmp_path: Path) -> None:
        item = pystac.Item(
            id="chip_001_q2_s2",
            geometry=None,
            bbox=list(CHIP_BBOX),
            datetime=datetime(2024, 4, 1, tzinfo=UTC),
            properties={},
        )
        task = build_download_task(item, tmp_path / "chip_001_q2_s2.json")
        assert (task.season, task.base_id, task.output_filename) == (
            "q2",
            "chip_001",
            "chip_001_q2_image_s2.tif",
        )

    def test_reference_mask_found_for_quarter_image(self, tmp_path: Path) -> None:
        mask = tmp_path / "chip_001_semantic_3_class.tif"
        mask.write_bytes(b"")
        assert find_reference_mask_for_output(tmp_path / "chip_001_q4_image_s2.tif") == mask

    def test_keeps_int16_values_nodata_and_scale(self, tmp_path: Path) -> None:
        source = tmp_path / "b04.tif"
        data = np.full((10, 10), 1200, dtype=np.int16)
        data[0, 0] = -1000  # mosaics carry negative reflectance codes
        data[9, 9] = -32768
        with rasterio.open(
            source,
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=1,
            dtype="int16",
            crs="EPSG:4326",
            transform=from_origin(10.0, 50.01, 0.001, 0.001),
            nodata=-32768,
        ) as dst:
            dst.write(data, 1)

        child = pystac.Item(
            id="chip_001_q1_s2",
            geometry=None,
            bbox=list(CHIP_BBOX),
            datetime=None,
            start_datetime=datetime(2024, 1, 1, tzinfo=UTC),
            end_datetime=datetime(2024, 3, 31, tzinfo=UTC),
            properties={"ftw:source": MOSAIC_SOURCE},
        )
        child.add_asset("red", pystac.Asset(href=str(source)))
        scene = SelectedScene(item=child, season="q1", cloud_cover=0.0, datetime=None, stac_url="")
        output = tmp_path / "chip_001_q1_image_s2.tif"
        with rasterio.open(
            tmp_path / "chip_001_semantic_3_class.tif",
            "w",
            driver="GTiff",
            width=10,
            height=10,
            count=1,
            dtype="uint8",
            crs="EPSG:4326",
            transform=from_origin(10.0, 50.01, 0.001, 0.001),
        ) as mask:
            mask.write(np.zeros((10, 10), dtype=np.uint8), 1)

        result = download_and_clip_scene(
            scene=scene, bbox=CHIP_BBOX, output_path=output, bands=["red"]
        )

        assert result.success, result.error
        with rasterio.open(output) as ds:
            assert ds.dtypes[0] == "int16"
            assert ds.nodata == -32768
            assert ds.scales == (0.0001,)
            assert ds.offsets == (0.0,)
            assert ds.read(1).min() < 0


def _owned_asset(path: Path) -> pystac.Asset:
    item = pystac.Item(
        id="owner", geometry=None, bbox=None, datetime=datetime.now(UTC), properties={}
    )
    item.add_asset("image", pystac.Asset(href=str(path)))
    return item.assets["image"]


class TestMosaicRasterBands:
    def test_scale_reaches_raster_bands(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_raster_bands
        from ftw_dataset_tools.api.imagery.image_download import write_cog

        path = tmp_path / "chip_001_q1_image_s2.tif"
        profile = {
            "driver": "COG",
            "dtype": "int16",
            "width": 4,
            "height": 4,
            "count": 1,
            "crs": "EPSG:4326",
            "transform": from_origin(10.0, 50.01, 0.001, 0.001),
            "nodata": -32768,
        }
        error = write_cog(
            path, np.ones((1, 4, 4), dtype=np.int16), ["red"], profile, nodata=-32768, scale=0.0001
        )
        assert error is None
        asset = _owned_asset(path)
        add_raster_bands(asset, path)
        band = asset.extra_fields["raster:bands"][0]
        assert band["scale"] == 0.0001
        assert band["offset"] == 0.0
        assert band["nodata"] == -32768

    def test_scene_files_carry_no_scale(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.assets import add_raster_bands

        path = tmp_path / "plain.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint16",
            crs="EPSG:4326",
            transform=from_origin(0, 1, 0.5, 0.5),
        ) as dst:
            dst.write(np.ones((2, 2), dtype=np.uint16), 1)
        asset = _owned_asset(path)
        add_raster_bands(asset, path)
        band = asset.extra_fields["raster:bands"][0]
        assert "scale" not in band
        assert "offset" not in band


def test_year_available_needs_all_four_quarters() -> None:
    def exists(url: str) -> bool:
        return "2023.Q2" not in url

    with (
        patch.object(mosaic_search, "_exists", side_effect=exists),
        patch.object(mosaic_search, "_years", {}),
    ):
        assert mosaic_search.year_available(2024)
        assert not mosaic_search.year_available(2023)


def test_exists_sends_a_user_agent() -> None:
    response = MagicMock(status=200)
    response.__enter__.return_value = response
    with patch.object(mosaic_search.urllib.request, "urlopen", return_value=response) as urlopen:
        assert mosaic_search._exists("https://example.com/x.parquet")
    request = urlopen.call_args.args[0]
    assert request.get_method() == "HEAD"
    assert request.get_header("User-agent").startswith("ftw-dataset-tools/")


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://example.com/x", code, "error", None, None)  # type: ignore[arg-type]


class TestMosaicSearchEdges:
    def test_missing_index_is_not_an_error(self) -> None:
        with patch.object(mosaic_search.urllib.request, "urlopen", side_effect=_http_error(404)):
            assert not mosaic_search._exists("https://example.com/x.parquet")

    def test_other_http_errors_propagate(self) -> None:
        # A 403/5xx must not silently mark every year unavailable.
        with (
            patch.object(mosaic_search.urllib.request, "urlopen", side_effect=_http_error(403)),
            pytest.raises(urllib.error.HTTPError),
        ):
            mosaic_search._exists("https://example.com/x.parquet")

    def test_year_availability_is_checked_once(self) -> None:
        with (
            patch.object(mosaic_search, "_exists", return_value=True) as exists,
            patch.object(mosaic_search, "_years", {}),
        ):
            assert mosaic_search.year_available(2024)
            assert mosaic_search.year_available(2024)
        assert exists.call_count == 4

    @pytest.mark.usefixtures("local_index")
    def test_index_is_read_once_per_quarter(self) -> None:
        with patch.object(mosaic_search, "_connection", wraps=mosaic_search._connection) as conn:
            mosaic_search.find_containing_tile(CHIP_BBOX, 2024, 1)
            mosaic_search.find_containing_tile(CHIP_BBOX, 2024, 1)
        assert conn.call_count == 1

    @pytest.mark.usefixtures("local_index")
    def test_tiles_outside_the_chip_latitude_are_skipped(self) -> None:
        assert mosaic_search.find_containing_tile((10.0, 60.0, 10.01, 60.01), 2024, 1) is None

    def test_antimeridian_tile_overlaps_both_sides(self) -> None:
        assert mosaic_search._lon_overlaps(178.0, -179.0, 179.5, 179.6)
        assert mosaic_search._lon_overlaps(178.0, -179.0, -179.6, -179.5)
        assert not mosaic_search._lon_overlaps(178.0, -179.0, 10.0, 10.1)


class TestMosaicSelectionEdges:
    def test_progress_lines_are_reported(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level("INFO")
        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(
                f"{_SELECTION}.containing_tiles",
                side_effect=lambda _b, y, n, *_: [_tile_item(y, n)],
            ),
            patch(f"{_SELECTION}.calculate_nodata_percentage", return_value=0.0),
        ):
            select_mosaics_for_chip("chip_001", CHIP_BBOX, 2024)
        assert caplog.messages[-1] == "Selected mosaics for 2024"
        assert any("2024 Q3" in line for line in caplog.messages)

    def test_failed_selection_writes_no_children(self, tmp_path: Path) -> None:
        parent, path = _chip(tmp_path)
        job = ChipSelectionJob(item=parent, item_path=path, year=2024)
        failed = MosaicSelectionResult(
            chip_id="chip_001", bbox=CHIP_BBOX, requested_year=2024, skipped_reason="none"
        )
        with patch(
            "ftw_dataset_tools.api.imagery.selection_workflow.select_mosaics_for_chip",
            return_value=failed,
        ):
            result = run_chip_selection(
                job,
                cloud_cover_chip=2.0,
                nodata_max=0.0,
                buffer_days=14,
                num_buffer_expansions=3,
                buffer_expansion_size=14,
                imagery_mode="mosaics",
            )
        assert not result.success
        assert not list(path.parent.glob("*_q*_s2.json"))

    def test_catalog_selection_skips_the_crop_calendar(self, tmp_path: Path) -> None:
        _chip(tmp_path)
        with (
            patch(f"{_SELECTION}.year_available", return_value=True),
            patch(
                "ftw_dataset_tools.api.imagery.selection_workflow.select_mosaics_for_chip",
                return_value=_selection(),
            ),
            patch(
                "ftw_dataset_tools.api.imagery.selection_workflow.ensure_crop_calendar_exists"
            ) as warm,
        ):
            result = _select_mode(tmp_path, year=2024, imagery_mode="mosaics")
        assert result.successful == 1
        warm.assert_not_called()

    def test_children_written_for_a_parent_without_self_href(self, tmp_path: Path) -> None:
        parent, path = _chip(tmp_path)
        parent.set_self_href(None)
        create_mosaic_child_items(path.parent, parent, _selection())
        assert Path(parent.get_self_href()) == path
        assert (path.parent / "chip_001_q4_s2.json").exists()


class TestDownloadDiscovery:
    def test_non_child_s2_files_are_ignored(self, tmp_path: Path) -> None:
        _parent, path = _chip(tmp_path)
        stray = pystac.Item(
            id="notes_s2", geometry=None, bbox=None, datetime=datetime.now(UTC), properties={}
        )
        stray.save_object(dest_href=str(path.parent / "notes_s2.json"), include_self_link=False)
        create_mosaic_child_items(path.parent, pystac.Item.from_file(str(path)), _selection())

        ids = sorted(item.id for item, _p in find_s2_child_items(tmp_path))
        assert ids == [f"chip_001_q{n}_s2" for n in (1, 2, 3, 4)]

    def test_download_task_rejects_a_non_child_item(self, tmp_path: Path) -> None:
        item = pystac.Item(
            id="chip_001",
            geometry=None,
            bbox=list(CHIP_BBOX),
            datetime=datetime.now(UTC),
            properties={},
        )
        with pytest.raises(ValueError, match="Not an imagery child item"):
            build_download_task(item, tmp_path / "chip_001.json")
