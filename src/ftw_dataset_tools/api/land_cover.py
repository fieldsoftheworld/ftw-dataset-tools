"""Per-chip land cover from the Impact Observatory annual land-cover maps.

For every chip, the 10 m IO land-cover pixels inside the chip footprint are counted
per class, and the class shares are written back into the chips GeoParquet so items
can publish them without reopening the rasters.

IO publishes one map per year (2017-2023). A chip reads the map for the dataset's
year when there is one, and otherwise the nearest year, flagged with
``landcover_year_exact = false`` because the land may have changed in between. A
chip is left blank only where IO has no map at all.

IO tiles are MGRS grid zones, one COG per zone per year in the zone's UTM CRS. FTW
grid cells never cross a zone, so each chip reads exactly one tile: the one named by
its ``gzd`` column, or computed from its centroid when the column is absent.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import duckdb
import numpy as np
import rasterio
from rasterio.features import bounds as geometry_bounds
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from rasterio.windows import Window

from ftw_dataset_tools.api.chips_io import (
    drop_columns,
    parquet_columns,
    select_excluding,
    write_chips,
)
from ftw_dataset_tools.api.geo import (
    detect_crs,
    detect_geometry_column,
    ensure_spatial_loaded,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from rasterio.io import DatasetReader

    from ftw_dataset_tools.api.imagery.parallel import ParallelOutcome

PRODUCT = "io-lulc-annual-v02"
STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION_URL = f"{STAC_API_URL}/collections/{PRODUCT}"
ATTRIBUTION = (
    "10m Annual Land Use Land Cover (9-class) V2 by Impact Observatory and Esri, "
    "CC-BY-4.0 (source of the ftw:landcover_* properties)"
)
TILE_ID_PROPERTY = "io:tile_id"
DATA_ASSET = "data"
NODATA = 0

#: IO class values and names, from the collection's ``file:values`` (nodata excluded).
IO_CLASSES = {
    1: "Water",
    2: "Trees",
    4: "Flooded vegetation",
    5: "Crops",
    7: "Built area",
    8: "Bare ground",
    9: "Snow/ice",
    10: "Clouds",
    11: "Rangeland",
}

OUTPUT_COLUMNS = (
    "landcover_year",
    "landcover_year_exact",
    "landcover_dominant_code",
    "landcover_dominant_name",
    "landcover_dominant_pct",
    "landcover_classes",
)

GZD_COLUMN = "gzd"
CHIPS_CRS = "EPSG:4326"
_GEOGRAPHIC_CRS = {"EPSG:4326", "OGC:CRS84"}

# Concurrent tile reads. Each is a handful of small HTTP range requests, so a few
# threads hide the latency without hammering the host.
DEFAULT_WORKERS = 8

# Chips read per task. A tile's chips are split into tasks of this size so a
# country that sits in one or two grid zones still spreads over the worker pool.
CHIPS_PER_READ = 250

NO_YEAR_REASON = "no dataset year"

_BAND_LETTERS = "CDEFGHJKLMNPQRSTUVWX"


# ---- year and tile selection ----------------------------------------------


@dataclass(frozen=True)
class YearMatch:
    """The map year a chip reads, and whether it is the dataset year itself."""

    year: int
    exact: bool


def resolve_year(dataset_year: int, available: Iterable[int]) -> YearMatch | None:
    """The available year closest to ``dataset_year``; ties go to the later year.

    Returns None when nothing is available, which leaves the chip blank.
    """
    years = set(available)
    if not years:
        return None
    if dataset_year in years:
        return YearMatch(dataset_year, exact=True)
    nearest = min(years, key=lambda y: (abs(y - dataset_year), -y))
    return YearMatch(nearest, exact=False)


def normalize_gzd(value: object) -> str:
    """Canonical grid zone designator: upper case, zone zero-padded (``5v`` -> ``05V``)."""
    text = str(value).strip().upper()
    digits = len(text) - len(text.lstrip("0123456789"))
    if digits == 0:
        return text
    return f"{int(text[:digits]):02d}{text[digits:]}"


def gzd_for_point(lon: float, lat: float) -> str | None:
    """MGRS grid zone designator of a point, or None outside the UTM latitudes.

    Includes the Norway (32V) and Svalbard (31X/33X/35X/37X) zone exceptions, which
    IO's tiling follows.
    """
    if not -80 <= lat <= 84:
        return None
    band = _BAND_LETTERS[min(int((lat + 80) // 8), len(_BAND_LETTERS) - 1)]
    lon = ((lon + 180) % 360) - 180
    zone = _zone_exception(int((lon + 180) // 6) + 1, band, lon)
    return f"{zone:02d}{band}"


def _zone_exception(zone: int, band: str, lon: float) -> int:
    if band == "V" and 3 <= lon < 12:
        return 32
    if band == "X" and 0 <= lon < 42:
        return 31 if lon < 9 else 33 if lon < 21 else 35 if lon < 33 else 37
    return zone


# ---- tile sources ---------------------------------------------------------

#: tile id -> year -> COG href
TileIndex = dict[str, dict[int, str]]


class TileSource(Protocol):
    """Where the land-cover COGs for an area are found."""

    def index(self, bbox: tuple[float, float, float, float]) -> TileIndex:
        """Every tile intersecting ``bbox`` (lon/lat), with each year's COG href."""
        ...


class IOLulcSource:
    """IO annual LULC v02 on Microsoft Planetary Computer, with signed hrefs."""

    def __init__(self, stac_url: str = STAC_API_URL) -> None:
        self.stac_url = stac_url

    def index(self, bbox: tuple[float, float, float, float]) -> TileIndex:
        import planetary_computer
        from pystac_client import Client

        client = Client.open(self.stac_url, modifier=planetary_computer.sign_inplace)
        search = client.search(collections=[PRODUCT], bbox=list(bbox))
        index: TileIndex = {}
        for item in search.items():
            tile = normalize_gzd(item.properties[TILE_ID_PROPERTY])
            start = item.common_metadata.start_datetime or item.datetime
            index.setdefault(tile, {})[start.year] = item.assets[DATA_ASSET].href
        return index


# ---- pixel counting -------------------------------------------------------


def _pixel_window(src: DatasetReader, bounds: tuple[float, float, float, float]) -> Window | None:
    """Whole-pixel window covering ``bounds`` (in the dataset CRS), clipped to the raster."""
    left, bottom, right, top = bounds
    inverse = ~src.transform
    corners = [inverse * (x, y) for x in (left, right) for y in (bottom, top)]
    cols = [c for c, _ in corners]
    rows = [r for _, r in corners]
    col_off = max(0, math.floor(min(cols)))
    row_off = max(0, math.floor(min(rows)))
    col_end = min(src.width, math.ceil(max(cols)))
    row_end = min(src.height, math.ceil(max(rows)))
    if col_end <= col_off or row_end <= row_off:
        return None
    return Window(col_off, row_off, col_end - col_off, row_end - row_off)


def count_classes(src: DatasetReader, geometry: dict) -> np.ndarray:
    """Pixel count per value (the array index) inside ``geometry``, in ``src``'s CRS.

    A pixel counts when its centre is inside the geometry. Nodata is counted like any
    other value; ``class_entries`` leaves it out.
    """
    window = _pixel_window(src, geometry_bounds(geometry))
    if window is None:
        return np.zeros(0, dtype=np.int64)
    data = src.read(1, window=window)
    inside = geometry_mask(
        [geometry],
        out_shape=data.shape,
        transform=src.window_transform(window),
        invert=True,
    )
    return np.bincount(data[inside].astype(np.int64, copy=False))


def class_entries(counts: np.ndarray, classes: dict[int, str] = IO_CLASSES) -> list[dict] | None:
    """``{code, name, pct}`` per class present, largest first, or None without valid pixels.

    Shares are of valid (non-nodata) pixels, so they sum to ~100.
    """
    valid = {code: int(n) for code, n in enumerate(counts) if n and code != NODATA}
    total = sum(valid.values())
    if total == 0:
        return None
    ordered = sorted(valid.items(), key=lambda kv: (-kv[1], kv[0]))
    return [
        {"code": code, "name": classes.get(code), "pct": round(100.0 * n / total, 2)}
        for code, n in ordered
    ]


# ---- results --------------------------------------------------------------


@dataclass
class LandCoverResult:
    """What the land cover step did."""

    chips_total: int
    chips_with_land_cover: int
    chips_nearest_year: int
    years_used: tuple[int, ...]
    dataset_year: int | None
    skipped: bool
    reason: str | None = None


def land_cover_summary(result: LandCoverResult | None) -> str:
    """The one-line run summary for the land cover step (None means configured off)."""
    if result is None:
        return "Land cover: disabled"
    if result.skipped:
        return f"Land cover: skipped ({result.reason})"
    counted = f"{result.chips_with_land_cover:,}/{result.chips_total:,} chips"
    if not result.years_used:
        return f"Land cover: {counted} (no IO data for this area)"
    years = ", ".join(str(y) for y in result.years_used)
    line = f"Land cover: {counted} from IO {years}"
    if result.chips_nearest_year:
        line += (
            f" ({result.chips_nearest_year:,} nearest-year for dataset year {result.dataset_year})"
        )
    return line


def drop_land_cover(chips_file: Path | str) -> bool:
    """Remove the land cover columns from a chips GeoParquet, in place.

    Returns True when columns were dropped. Used when the step is configured off, so
    a rerun never republishes the previous run's land cover.
    """
    return drop_columns(chips_file, OUTPUT_COLUMNS)


# ---- the step -------------------------------------------------------------


@dataclass(frozen=True)
class _Chip:
    chip_id: str
    tile_id: str | None
    geometry: dict


@dataclass(frozen=True)
class _ReadTask:
    href: str
    match: YearMatch
    chips: tuple[_Chip, ...]


_ROWS_DDL = """
CREATE TABLE land_cover_rows (
    chip_id VARCHAR,
    landcover_year INTEGER,
    landcover_year_exact BOOLEAN,
    landcover_dominant_code BIGINT,
    landcover_dominant_name VARCHAR,
    landcover_dominant_pct DOUBLE,
    landcover_classes STRUCT(code BIGINT, name VARCHAR, pct DOUBLE)[]
)
"""


def _check_chips(chips_path: Path, chips_id_col: str) -> str:
    """Validate the chips file and return its geometry column."""
    if chips_id_col not in parquet_columns(chips_path):
        raise ValueError(f"Chips file has no '{chips_id_col}' column: {chips_path}")
    geom_col = detect_geometry_column(chips_path) or "geometry"
    crs = detect_crs(chips_path, geom_col).authority_code
    if crs is not None and crs.upper() not in _GEOGRAPHIC_CRS:
        raise ValueError(f"Chips must be in EPSG:4326 (got {crs}): {chips_path}")
    return geom_col


def _load_chips(
    con: duckdb.DuckDBPyConnection, geom_col: str, chips_id_col: str, has_gzd: bool
) -> list[_Chip]:
    """Every chip in ``chips_table`` with its tile id and GeoJSON footprint."""
    gzd = f'"{GZD_COLUMN}"' if has_gzd else "NULL"
    rows = con.execute(f"""
        SELECT CAST("{chips_id_col}" AS VARCHAR), {gzd},
               ST_X(ST_Centroid("{geom_col}")), ST_Y(ST_Centroid("{geom_col}")),
               ST_AsGeoJSON("{geom_col}")
        FROM chips_table
        WHERE "{geom_col}" IS NOT NULL
    """).fetchall()
    chips = []
    for chip_id, gzd_value, lon, lat, geojson in rows:
        tile = normalize_gzd(gzd_value) if gzd_value else gzd_for_point(lon, lat)
        chips.append(_Chip(chip_id, tile, json.loads(geojson)))
    return chips


def _chips_bbox(chips: Sequence[_Chip]) -> tuple[float, float, float, float]:
    boxes = [geometry_bounds(chip.geometry) for chip in chips]
    return (
        min(b[0] for b in boxes),
        min(b[1] for b in boxes),
        max(b[2] for b in boxes),
        max(b[3] for b in boxes),
    )


def _plan_reads(
    chips: Sequence[_Chip], index: TileIndex, dataset_year: int, chips_per_read: int
) -> list[_ReadTask]:
    """Group chips by the COG they read, in tasks of at most ``chips_per_read`` chips."""
    groups: dict[str, tuple[YearMatch, list[_Chip]]] = {}
    for chip in chips:
        years = index.get(chip.tile_id, {}) if chip.tile_id else {}
        match = resolve_year(dataset_year, years)
        if match is None:
            continue
        groups.setdefault(years[match.year], (match, []))[1].append(chip)
    tasks = []
    for href, (match, members) in groups.items():
        for start in range(0, len(members), chips_per_read):
            tasks.append(_ReadTask(href, match, tuple(members[start : start + chips_per_read])))
    return tasks


def _read_task(task: _ReadTask) -> list[tuple[str, np.ndarray]]:
    """Open one COG and count classes for each of the task's chips."""
    with (
        rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"),
        rasterio.open(task.href) as src,
    ):
        return [
            (chip.chip_id, count_classes(src, transform_geom(CHIPS_CRS, src.crs, chip.geometry)))
            for chip in task.chips
        ]


def _read_all(tasks: list[_ReadTask], workers: int) -> list[tuple]:
    """Run every read and return one land_cover_rows tuple per chip with valid pixels."""
    # Imported here: the imagery package imports api.stac, which imports this module.
    from ftw_dataset_tools.api.imagery.parallel import run_in_parallel

    rows: list[tuple] = []

    def apply(outcome: ParallelOutcome[_ReadTask, list[tuple[str, np.ndarray]]]) -> None:
        if outcome.error is not None:
            raise outcome.error
        match = outcome.task.match
        for chip_id, counts in outcome.value or []:
            entries = class_entries(counts)
            if entries is None:
                continue
            top = entries[0]
            rows.append(
                (chip_id, match.year, match.exact, top["code"], top["name"], top["pct"], entries)
            )

    run_in_parallel(tasks, _read_task, apply, workers=workers)
    return rows


def _write_results(
    con: duckdb.DuckDBPyConnection, chips_path: Path, chips_id_col: str, rows: list[tuple]
) -> None:
    con.execute(_ROWS_DDL)
    if rows:
        con.executemany("INSERT INTO land_cover_rows VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    selected = ", ".join(f"r.{col}" for col in OUTPUT_COLUMNS)
    write_chips(
        chips_path,
        con,
        f"SELECT g.*, {selected} FROM chips_table g "
        f'LEFT JOIN land_cover_rows r ON CAST(g."{chips_id_col}" AS VARCHAR) = r.chip_id',
    )


def _result(total: int, rows: list[tuple], year: int) -> LandCoverResult:
    return LandCoverResult(
        chips_total=total,
        chips_with_land_cover=len(rows),
        chips_nearest_year=sum(1 for row in rows if not row[2]),
        years_used=tuple(sorted({row[1] for row in rows})),
        dataset_year=year,
        skipped=False,
    )


def add_land_cover(
    chips_file: Path | str,
    *,
    year: int | None,
    source: TileSource | None = None,
    chips_id_col: str = "id",
    workers: int = DEFAULT_WORKERS,
    chips_per_read: int = CHIPS_PER_READ,
    on_progress: Callable[[str], None] | None = None,
) -> LandCoverResult:
    """Append the land cover columns to the chips GeoParquet, in place.

    Args:
        chips_file: Chips GeoParquet in EPSG:4326, rewritten in place.
        year: The dataset year. None skips the step.
        source: Where the land-cover COGs come from (default: IO on Planetary Computer).
        chips_id_col: Chip id column.
        workers: Concurrent tile reads.
        chips_per_read: Chips per read task; bounds work per task, not the result.
        on_progress: Optional callback for progress messages.
    """
    chips_path = Path(chips_file).resolve()

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    geom_col = _check_chips(chips_path, chips_id_col)
    has_gzd = GZD_COLUMN in parquet_columns(chips_path)

    con = duckdb.connect(":memory:")
    ensure_spatial_loaded(con)
    try:
        con.execute(f"CREATE TABLE chips_table AS {select_excluding(chips_path, OUTPUT_COLUMNS)}")
        total = con.execute("SELECT count(*) FROM chips_table").fetchone()[0]
        if year is None:
            log(f"Note: {NO_YEAR_REASON}; skipping land cover")
            return LandCoverResult(total, 0, 0, (), None, skipped=True, reason=NO_YEAR_REASON)

        chips = _load_chips(con, geom_col, chips_id_col, has_gzd)
        rows: list[tuple] = []
        if chips:
            index = (source or IOLulcSource()).index(_chips_bbox(chips))
            tasks = _plan_reads(chips, index, year, max(1, chips_per_read))
            log(f"Reading IO land cover for {len(chips):,} chips ({len(tasks)} read tasks)...")
            rows = _read_all(tasks, workers)
        _write_results(con, chips_path, chips_id_col, rows)
    finally:
        con.close()

    result = _result(total, rows, year)
    if result.chips_nearest_year:
        log(
            f"Warning: {result.chips_nearest_year:,} chips use the nearest IO year because "
            f"IO has no {year} map for them; landcover_year_exact is false on those chips"
        )
    log(land_cover_summary(result))
    return result
