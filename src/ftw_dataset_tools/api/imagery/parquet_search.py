"""Scene search against the Sentinel-2 STAC-GeoParquet mirror.

Queries the partitioned STAC-GeoParquet catalog published at
source.coop/portolan-mirrors/sentinel-2-catalog instead of the Earth Search
API. The mirror indexes every Earth Search ``sentinel-2-l2a`` scene, and the
items reconstructed here carry the same sentinel-cogs COG hrefs the API would
return (rebuilt from the scene id), with no rate limit anywhere in the path.
"""

from __future__ import annotations

import contextlib
import math
import threading
from datetime import date
from typing import TYPE_CHECKING

import duckdb
import pystac
from pystac.extensions.eo import EOExtension

from ftw_dataset_tools.api.imagery.settings import STAC_URL

if TYPE_CHECKING:
    from datetime import datetime

__all__ = [
    "DEFAULT_PARQUET_URL",
    "PARQUET_COLLECTION",
    "part_urls_for_query",
    "query_scenes",
    "zones_for_bbox",
]

DEFAULT_PARQUET_URL = "https://data.source.coop/portolan-mirrors/sentinel-2-catalog/sentinel-2-l2a"

# The mirror holds Earth Search's sentinel-2-l2a collection (not c1).
PARQUET_COLLECTION = "sentinel-2-l2a"

# The archive starts in November 2016.
_FIRST_YEAR = 2016

# UTM zone ranges per part file, by year vintage (see the catalog README's
# layout table; both sets are fixed for the life of the catalog).
_RANGES_FOUR = [(1, 20), (21, 35), (36, 46), (47, 60)]
_RANGES_EIGHT = [(1, 15), (16, 20), (21, 31), (32, 35), (36, 40), (41, 46), (47, 52), (53, 60)]

_local = threading.local()


def _utm_zone(lon: float) -> int:
    """UTM zone number for a longitude."""
    return int((lon + 180.0) // 6.0) % 60 + 1


def zones_for_bbox(bbox: tuple[float, float, float, float]) -> set[int]:
    """UTM zones whose Sentinel-2 tiles can cover a bbox.

    A tile's footprint can overhang its zone's boundary meridian by up to
    ~110 km (the last 100 km grid column plus tile overlap), so the west and
    east edges are padded by that much before computing zones - a chip just
    east of 6E is routinely covered by a zone 31 tile. The Norway (32V) and
    Svalbard (31X-37X) grid exceptions are added on top.
    """
    west, south, east, north = bbox
    # 110 km in degrees of longitude at the bbox's widest latitude.
    widest_lat = min(max(abs(south), abs(north)), 80.0)
    margin = 110.0 / (111.32 * math.cos(math.radians(widest_lat)))
    zones = set(range(_utm_zone(west - margin), _utm_zone(east + margin) + 1))
    # Norway: zone 32 is widened over 3-12E in band V (56-64N)
    if north >= 56.0 and south <= 64.0 and east >= 3.0 and west <= 12.0:
        zones.add(32)
    # Svalbard: bands X (72N+) use zones 31/33/35/37 over 0-42E
    if north >= 72.0 and east >= 0.0 and west <= 42.0:
        zones.update({31, 33, 35, 37})
    return zones


def _part_name(year: int, zone: int) -> str:
    """Part file name holding a UTM zone's scenes for a year."""
    if year <= 2018:
        return "items.parquet"
    ranges = _RANGES_FOUR if year <= 2020 else _RANGES_EIGHT
    for lo, hi in ranges:
        if lo <= zone <= hi:
            return f"z{lo:02d}-{hi:02d}.parquet"
    raise ValueError(f"UTM zone out of range: {zone}")


def part_urls_for_query(
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    *,
    base_url: str = DEFAULT_PARQUET_URL,
    today: date | None = None,
) -> list[str]:
    """Catalog part URLs a bbox + date-window query must read.

    Years outside the archive (before 2016 or after the current year) are
    dropped; the current year also reads ``live.parquet``, the daily tail.
    """
    today = today or date.today()
    urls: list[str] = []
    for year in range(start.year, end.year + 1):
        if year < _FIRST_YEAR or year > today.year:
            continue
        names = {_part_name(year, zone) for zone in zones_for_bbox(bbox)}
        urls.extend(f"{base_url}/year={year}/{name}" for name in sorted(names))
        if year == today.year:
            urls.append(f"{base_url}/year={year}/live.parquet")
    return urls


def _months_in_window(start: datetime, end: datetime) -> list[int]:
    """Calendar months the window touches, for ``_month`` row-group pruning."""
    months: list[int] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        if month not in months:
            months.append(month)
        year, month = (year + 1, 1) if month == 12 else (year, month + 1)
        if len(months) == 12:
            break
    return months


def _connection() -> duckdb.DuckDBPyConnection:
    """A per-thread DuckDB connection with UTC session time and httpfs loaded.

    Reusing the connection lets DuckDB cache parquet footers across the many
    small queries a selection run makes against the same part files.
    """
    con = getattr(_local, "con", None)
    if con is not None:
        return con
    con = duckdb.connect()
    con.execute("SET TimeZone = 'UTC'")
    try:
        con.execute("LOAD httpfs")
    except duckdb.Error:
        # local-path queries still work without it; remote URLs error clearly
        with contextlib.suppress(duckdb.Error):
            con.execute("INSTALL httpfs; LOAD httpfs")
    for setting in (
        "SET http_timeout = 30000",
        "SET http_retries = 3",
        "SET enable_progress_bar = false",
    ):
        with contextlib.suppress(duckdb.Error):
            con.execute(setting)
    _local.con = con
    return con


def _bbox_polygon(bbox: list[float]) -> dict:
    """GeoJSON polygon for a [w, s, e, n] bbox."""
    w, s, e, n = bbox
    return {
        "type": "Polygon",
        "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]],
    }


# sentinel-cogs bucket files per Earth Search asset key. The upstream hrefs
# follow this layout exactly, so they can be rebuilt from a scene id without
# reading the mirror's wide `assets` column (a bulk read of which costs tens
# of seconds per query - see the catalog README's advice to fetch it only for
# chosen scenes).
_ASSET_FILES = {
    "coastal": "B01",
    "blue": "B02",
    "green": "B03",
    "red": "B04",
    "rededge1": "B05",
    "rededge2": "B06",
    "rededge3": "B07",
    "nir": "B08",
    "nir08": "B8A",
    "nir09": "B09",
    "swir16": "B11",
    "swir22": "B12",
    "scl": "SCL",
    "visual": "TCI",
}

_COG_BASE = "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs"
MEDIA_TYPE_COG = "image/tiff; application=geotiff; profile=cloud-optimized"


def _synthesize_assets(item_id: str) -> dict[str, pystac.Asset]:
    """Rebuild a scene's COG assets from its id.

    An id like ``S2B_33TVL_20250610_0_L2A`` names tile 33TVL acquired
    2025-06-10, whose files live under
    ``{bucket}/33/T/VL/2025/6/S2B_33TVL_20250610_0_L2A/``.
    """
    _, tile, date_s = item_id.split("_")[:3]
    zone, band, square = tile[:-3], tile[-3], tile[-2:]
    year, month = int(date_s[:4]), int(date_s[4:6])
    base = f"{_COG_BASE}/{zone}/{band}/{square}/{year}/{month}/{item_id}"
    return {
        key: pystac.Asset(
            href=f"{base}/{filename}.tif",
            media_type=MEDIA_TYPE_COG,
            roles=["visual"] if key == "visual" else ["data"],
        )
        for key, filename in _ASSET_FILES.items()
    }


def _row_to_item(
    item_id: str,
    bbox: list[float],
    dt: datetime,
    tile: str,
    cloud_cover: float | None,
    nodata_pct: float | None,
) -> pystac.Item:
    """Reconstruct a pystac Item from a mirror row."""
    properties: dict = {"s2:mgrs_tile": tile}
    if cloud_cover is not None:
        properties["eo:cloud_cover"] = cloud_cover
    if nodata_pct is not None:
        properties["s2:nodata_pixel_percentage"] = nodata_pct
    item = pystac.Item(
        id=item_id,
        geometry=_bbox_polygon(bbox),
        bbox=list(bbox),
        datetime=dt,
        properties=properties,
    )
    # Selection reads cloud cover through EOExtension, which requires the
    # extension to be declared on the item.
    EOExtension.ext(item, add_if_missing=True)
    for key, asset in _synthesize_assets(item_id).items():
        item.add_asset(key, asset)
    # A stable provenance URL for the scene; never used for search.
    item.set_self_href(f"{STAC_URL}/collections/{PARQUET_COLLECTION}/items/{item_id}")
    return item


def query_scenes(
    bbox: tuple[float, float, float, float],
    start: datetime,
    end: datetime,
    cloud_cover_max: int,
    *,
    base_url: str = DEFAULT_PARQUET_URL,
    today: date | None = None,
) -> list[pystac.Item]:
    """Scenes intersecting a bbox in a date window, sorted by cloud cover.

    Matches the Earth Search query the API backend runs: bbox intersection,
    datetime window, and ``eo:cloud_cover < cloud_cover_max``, ordered
    ascending by cloud cover.
    """
    urls = part_urls_for_query(bbox, start, end, base_url=base_url, today=today)
    if not urls:
        return []
    url_list = ", ".join("'" + u.replace("'", "''") + "'" for u in urls)
    months = ", ".join(str(m) for m in _months_in_window(start, end))
    west, south, east, north = bbox
    sql = f"""
        SELECT id, bbox, datetime, "s2:mgrs_tile", "eo:cloud_cover",
               "s2:nodata_pixel_percentage"
        FROM read_parquet([{url_list}], union_by_name = true)
        WHERE _month IN ({months})
          AND datetime BETWEEN ? AND ?
          AND "eo:cloud_cover" < ?
          AND bbox[1] <= ? AND bbox[3] >= ?
          AND bbox[2] <= ? AND bbox[4] >= ?
        ORDER BY "eo:cloud_cover" NULLS LAST, id
    """
    rows = (
        _connection()
        .execute(sql, [start, end, cloud_cover_max, east, west, north, south])
        .fetchall()
    )
    items: list[pystac.Item] = []
    seen: set[str] = set()
    for item_id, row_bbox, dt, tile, cloud, nodata in rows:
        if item_id in seen:
            continue
        seen.add(item_id)
        items.append(_row_to_item(item_id, row_bbox, dt, tile, cloud, nodata))
    return items
