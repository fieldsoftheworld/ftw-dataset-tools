"""Find Sentinel-2 quarterly mosaic tiles in the Source Cooperative mirror's index.

The mirror publishes one GeoParquet item index per quarter
(``mosaics/quarter=YYYY.Qn/items.parquet``). A year counts as available only when
all four of its quarterly indexes exist.
"""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pystac
from pyproj import CRS, Transformer

from ftw_dataset_tools import __version__
from ftw_dataset_tools.api.imagery.parquet_search import _connection
from ftw_dataset_tools.api.imagery.settings import MOSAIC_BASE_URL

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "MOSAIC_BAND_ASSETS",
    "find_containing_tile",
    "index_url",
    "year_available",
]

#: Mirror asset key -> the band name ftwd uses.
MOSAIC_BAND_ASSETS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}

_COG = "image/tiff; application=geotiff; profile=cloud-optimized"


@dataclass(frozen=True)
class _Tile:
    subtile: str
    crs: str
    proj_bbox: tuple[float, float, float, float]
    bbox: tuple[float, float, float, float]
    start: datetime
    end: datetime
    hrefs: dict[str, str]


_lock = threading.Lock()
_indexes: dict[tuple[int, int], list[_Tile]] = {}
_years: dict[int, bool] = {}


def index_url(year: int, quarter: int, base_url: str = MOSAIC_BASE_URL) -> str:
    return f"{base_url}/mosaics/quarter={year}.Q{quarter}/items.parquet"


def _exists(url: str) -> bool:
    # The mirror answers 403 to urllib's default User-Agent.
    headers = {"User-Agent": f"ftw-dataset-tools/{__version__}"}
    request = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status == 200
    except urllib.error.HTTPError as err:
        if err.code == 404:
            return False
        raise


def year_available(year: int) -> bool:
    """Whether all four quarterly indexes of ``year`` are published."""
    with _lock:
        if year not in _years:
            _years[year] = all(_exists(index_url(year, q)) for q in (1, 2, 3, 4))
        return _years[year]


def _load_index(year: int, quarter: int) -> list[_Tile]:
    with _lock:
        cached = _indexes.get((year, quarter))
        if cached is not None:
            return cached
        bands = ", ".join(f'"{key}_href"' for key in MOSAIC_BAND_ASSETS)
        rows = (
            _connection()
            .execute(
                f'SELECT _subtile, "proj:code", "proj:bbox", bbox, start_datetime, '
                f"end_datetime, {bands} FROM read_parquet('{index_url(year, quarter)}')"
            )
            .fetchall()
        )
        tiles = [
            _Tile(
                subtile=row[0],
                crs=row[1],
                proj_bbox=tuple(row[2]),
                bbox=tuple(row[3]),
                start=row[4],
                end=row[5],
                hrefs=dict(zip(MOSAIC_BAND_ASSETS.values(), row[6:], strict=True)),
            )
            for row in rows
        ]
        _indexes[(year, quarter)] = tiles
        return tiles


def _lon_overlaps(tile_west: float, tile_east: float, west: float, east: float) -> bool:
    if tile_west <= tile_east:
        return tile_west <= east and west <= tile_east
    # Antimeridian tile: covers [tile_west, 180] and [-180, tile_east].
    return west <= tile_east or east >= tile_west


def _contains(tile: _Tile, bbox: tuple[float, float, float, float]) -> bool:
    """Whether the chip bbox lies wholly inside the tile, measured in the tile's own CRS."""
    transformer = Transformer.from_crs(
        CRS.from_epsg(4326), CRS.from_user_input(tile.crs), always_xy=True
    )
    minx, miny, maxx, maxy = transformer.transform_bounds(*bbox, densify_pts=21)
    tile_minx, tile_miny, tile_maxx, tile_maxy = tile.proj_bbox
    return tile_minx <= minx and tile_miny <= miny and maxx <= tile_maxx and maxy <= tile_maxy


def find_containing_tile(
    bbox: tuple[float, float, float, float], year: int, quarter: int
) -> pystac.Item | None:
    """The mosaic tile of ``year`` Q``quarter`` that fully contains ``bbox``, or None.

    The returned item carries ``red``/``green``/``blue``/``nir`` assets pointing at
    the mirror's COGs.
    """
    west, south, east, north = bbox
    for tile in _load_index(year, quarter):
        tile_west, tile_south, tile_east, tile_north = tile.bbox
        if tile_south > north or tile_north < south:
            continue
        if not _lon_overlaps(tile_west, tile_east, west, east):
            continue
        if _contains(tile, bbox):
            return _to_item(tile, year, quarter)
    return None


def _to_item(tile: _Tile, year: int, quarter: int) -> pystac.Item:
    item = pystac.Item(
        id=f"Sentinel-2_mosaic_{year}_Q{quarter}_{tile.subtile}",
        geometry=None,
        bbox=None,
        datetime=None,
        start_datetime=tile.start,
        end_datetime=tile.end,
        properties={"ftw:mosaic_tile": tile.subtile},
    )
    for band, href in tile.hrefs.items():
        item.add_asset(band, pystac.Asset(href=href, media_type=_COG, roles=["data"]))
    return item
