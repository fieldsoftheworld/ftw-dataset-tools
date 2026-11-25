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
from itertools import combinations
from typing import TYPE_CHECKING, Any

import pystac
import shapely
from rasterio.crs import CRS
from shapely.geometry import box

from ftw_dataset_tools import __version__
from ftw_dataset_tools.api.chip_grid import reprojector
from ftw_dataset_tools.api.imagery.nodata_analysis import chip_in_crs
from ftw_dataset_tools.api.imagery.parquet_search import _connection
from ftw_dataset_tools.api.imagery.settings import MOSAIC_BASE_URL

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime

    from shapely.geometry.base import BaseGeometry

__all__ = [
    "MOSAIC_BAND_ASSETS",
    "MOSAIC_HREFS",
    "containing_tiles",
    "covering_tile_sets",
    "index_url",
    "year_available",
]

#: Mirror asset key -> the band name ftwd uses.
MOSAIC_BAND_ASSETS = {"B02": "blue", "B03": "green", "B04": "red", "B08": "nir"}

#: Band asset field listing every tile's href, primary first, on a quarter built from
#: several tiles (see :func:`covering_tile_sets`).
MOSAIC_HREFS = "ftw:mosaic_hrefs"

_COG = "image/tiff; application=geotiff; profile=cloud-optimized"

# Slack (metres) for a chip edge that lies exactly on a tile edge: its lon/lat
# footprint, reprojected, bows up to ~1 cm past it. Far below a 10 m pixel.
_EDGE_TOLERANCE = 0.1

# A chip on a corner of four MGRS squares needs at most four tiles.
_MAX_COVER_TILES = 4
# Share of the chip a set of tiles may leave uncovered: reprojection noise along an
# edge (0.01%, a 0.2 m strip along a 2 km edge), like nodata_analysis' tolerance.
_COVER_TOLERANCE = 1e-4
# Vertex spacing (metres) of a tile's square before it is taken into another zone.
_TILE_DENSIFY = 1000.0


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
    ranked = []
    for tile in _near_tiles(bbox, year, quarter):
        margin = _margin(tile, bbox, footprint)
        if margin >= -_EDGE_TOLERANCE:
            ranked.append(((tile.crs == crs, margin), tile))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [_to_item(tile, year, quarter) for _, tile in ranked]


def _near_tiles(bbox: tuple[float, float, float, float], year: int, quarter: int) -> list[_Tile]:
    """The tiles whose lon/lat bbox overlaps the chip's."""
    west, south, east, north = bbox
    return [
        tile
        for tile in _load_index(year, quarter)
        if tile.bbox[1] <= north
        and tile.bbox[3] >= south
        and _lon_overlaps(tile.bbox[0], tile.bbox[2], west, east)
    ]


def covering_tile_sets(
    bbox: tuple[float, float, float, float],
    year: int,
    quarter: int,
    footprint: Mapping[str, Any] | None = None,
    crs: str | None = None,
) -> list[pystac.Item]:
    """For a chip no single tile contains: sets of tiles whose union covers it, best first.

    Each set is returned as one item for the quarter, built by :func:`_combined_item`.
    Only the smallest sets that cover the chip are returned, so two tiles where two
    suffice; larger sets are not tried even when every smallest set has nodata. Tiles are ranked as :func:`containing_tiles` ranks them, by ``crs``
    first, then by how much of the chip they cover, then by tile id so a tie never
    depends on index order; sets come in that ranking's order, and within a set the
    best-ranked tile is the primary, which supplies every pixel it has data for.
    """
    chip_crs = CRS.from_user_input(crs or "EPSG:4326")
    chip = chip_in_crs(chip_crs, bbox, footprint)
    if chip.area <= 0:
        return []
    parts = _ranked_parts(chip, chip_crs, crs, _near_tiles(bbox, year, quarter))
    return [_combined_item(tiles, year, quarter) for tiles in _smallest_covers(chip, parts)]


def _ranked_parts(
    chip: BaseGeometry, chip_crs: CRS, crs: str | None, tiles: list[_Tile]
) -> list[tuple[_Tile, BaseGeometry]]:
    """Each tile with the part of the chip it covers, best first; tiles missing it dropped."""
    parts = [(tile, chip.intersection(_tile_square(tile, chip_crs))) for tile in tiles]
    parts = [part for part in parts if part[1].area > 0]
    parts.sort(key=lambda part: (part[0].crs != crs, -part[1].area, part[0].subtile))
    return parts


def _smallest_covers(
    chip: BaseGeometry, parts: list[tuple[_Tile, BaseGeometry]]
) -> list[list[_Tile]]:
    """The smallest sets of two or more ``parts`` that cover the chip, in ranking order."""
    for size in range(2, min(len(parts), _MAX_COVER_TILES) + 1):
        sets = [_tiles_of(combo) for combo in combinations(parts, size) if _covers(chip, combo)]
        if sets:
            return sets
    return []


def _tiles_of(parts: tuple[tuple[_Tile, BaseGeometry], ...]) -> list[_Tile]:
    return [tile for tile, _ in parts]


def _tile_square(tile: _Tile, crs: CRS) -> BaseGeometry:
    """The tile's extent in ``crs``."""
    square = box(*tile.proj_bbox)
    tile_crs = CRS.from_user_input(tile.crs)
    if tile_crs == crs:
        return square
    return shapely.transform(shapely.segmentize(square, _TILE_DENSIFY), reprojector(tile_crs, crs))


def _covers(chip: BaseGeometry, parts: tuple[tuple[_Tile, BaseGeometry], ...]) -> bool:
    uncovered = chip.difference(shapely.union_all([covered for _, covered in parts]))
    return uncovered.area <= _COVER_TOLERANCE * chip.area


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


def _combined_item(tiles: list[_Tile], year: int, quarter: int) -> pystac.Item:
    """One quarter's item built from several tiles, ``tiles[0]`` the primary.

    It reads as the primary tile's item, so ``ftw:mosaic_tile`` and the band hrefs
    stay those of one tile, plus ``ftw:mosaic_tiles`` and, on each band asset,
    :data:`MOSAIC_HREFS` naming every tile in order. Its time range spans every tile's.
    """
    item = _to_item(tiles[0], year, quarter)
    subtiles = [tile.subtile for tile in tiles]
    item.id = f"Sentinel-2_mosaic_{year}_Q{quarter}_{'+'.join(subtiles)}"
    item.common_metadata.start_datetime = min(tile.start for tile in tiles)
    item.common_metadata.end_datetime = max(tile.end for tile in tiles)
    item.properties["ftw:mosaic_tiles"] = subtiles
    for band, asset in item.assets.items():
        asset.extra_fields[MOSAIC_HREFS] = _band_hrefs(tiles, band)
    return item


def _band_hrefs(tiles: list[_Tile], band: str) -> list[str]:
    return [tile.hrefs[band] for tile in tiles]
