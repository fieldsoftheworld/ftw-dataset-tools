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
from typing import TYPE_CHECKING, Any

import pystac
from rasterio.crs import CRS

from ftw_dataset_tools import __version__
from ftw_dataset_tools.api.imagery.nodata_analysis import chip_in_crs
from ftw_dataset_tools.api.imagery.parquet_search import _connection
from ftw_dataset_tools.api.imagery.settings import MOSAIC_BASE_URL

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

__all__ = [
    "MOSAIC_BAND_ASSETS",
    "containing_tiles",
    "find_containing_tile",
    "index_url",
    "year_available",
]

#: Mirror asset key -> the band name ftwd uses.
MOSAIC_BAND_ASSETS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}

_COG = "image/tiff; application=geotiff; profile=cloud-optimized"

# Slack (metres) for a chip edge that lies exactly on a tile edge: its lon/lat
# footprint, reprojected, bows up to ~1 cm past it. Far below a 10 m pixel.
_EDGE_TOLERANCE = 0.1


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
                proj_bbox=tuple(float(v) for v in row[2]),
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


def _margin(
    tile: _Tile,
    bbox: tuple[float, float, float, float],
    footprint: Mapping[str, Any] | None = None,
) -> float:
    """How far (metres) the chip lies inside the tile, in the tile's own CRS.

    Negative when the chip overhangs the tile. Tiles are MGRS squares that overlap
    their neighbours by only ~60 m, while a lon/lat bbox taken onto the tile's grid
    swells by up to ~90 m per side, so a chip on a square's edge fits only when
    measured by its ``footprint``.
    """
    chip = chip_in_crs(CRS.from_user_input(tile.crs), bbox, footprint)
    minx, miny, maxx, maxy = chip.bounds
    tile_minx, tile_miny, tile_maxx, tile_maxy = tile.proj_bbox
    return min(minx - tile_minx, miny - tile_miny, tile_maxx - maxx, tile_maxy - maxy)


def containing_tiles(
    bbox: tuple[float, float, float, float],
    year: int,
    quarter: int,
    footprint: Mapping[str, Any] | None = None,
    crs: str | None = None,
) -> list[pystac.Item]:
    """The mosaic tiles of ``year`` Q``quarter`` that fully contain the chip, best first.

    Pass the chip's GeoJSON polygon as ``footprint`` whenever it is known; ``bbox``
    alone rejects chips along a tile's edge (see :func:`_margin`). Near a zone seam
    tiles of both zones can contain the chip: those in ``crs``, the CRS of the chip's
    own grid, come first so its pixels need no warping; then the tiles the chip lies
    deepest inside.

    Each item carries ``red``/``green``/``blue``/``nir`` assets pointing at the
    mirror's COGs.
    """
    west, south, east, north = bbox
    ranked = []
    for tile in _load_index(year, quarter):
        tile_west, tile_south, tile_east, tile_north = tile.bbox
        if tile_south > north or tile_north < south:
            continue
        if not _lon_overlaps(tile_west, tile_east, west, east):
            continue
        margin = _margin(tile, bbox, footprint)
        if margin >= -_EDGE_TOLERANCE:
            ranked.append(((tile.crs == crs, margin), tile))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [_to_item(tile, year, quarter) for _, tile in ranked]


def find_containing_tile(
    bbox: tuple[float, float, float, float],
    year: int,
    quarter: int,
    footprint: Mapping[str, Any] | None = None,
    crs: str | None = None,
) -> pystac.Item | None:
    """The best mosaic tile that fully contains the chip (see :func:`containing_tiles`)."""
    tiles = containing_tiles(bbox, year, quarter, footprint, crs)
    return tiles[0] if tiles else None


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
