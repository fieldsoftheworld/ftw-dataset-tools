"""Label extent: the area inside which a dataset's labels are complete.

Some label sets are drawn only inside their own sample tiles: every field inside a
tile is labelled, nothing outside is. FTW's chips sit on their own grid and so
straddle those tiles. A label extent is a polygon file giving where the labels are
complete. Inside it an unlabelled pixel is background (0); outside it the pixel is
unknown (3, the value presence-only labels use). Field pixels and boundary lines are
burned as usual wherever they fall.

This module validates the extent file, and in the chips stage drops chips wholly
outside it and records each kept chip's share inside it (``label_extent_pct``). The
masks stage burns the extent per chip (see ``masks._burn_label_extent``).
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import pyproj
import shapely

from ftw_dataset_tools.api.chips_io import select_excluding, write_chips
from ftw_dataset_tools.api.geo import (
    detect_crs,
    detect_geometry_column,
    ensure_spatial_loaded,
    sql_path,
)
from ftw_dataset_tools.api.logging_config import get_logger

if TYPE_CHECKING:
    import numpy as np

logger = get_logger(__name__)

#: Chips column (and, as ``ftw:label_extent_pct``, item property) holding the share
#: of a chip, 0-100, that lies inside the label extent.
LABEL_EXTENT_COLUMN = "label_extent_pct"

#: Semantic mask value for an unlabelled pixel outside the extent: the same
#: "unknown" class presence-only labels use.
OUTSIDE_EXTENT_VALUE = 3

_POLYGON_TYPES = frozenset({"POLYGON", "MULTIPOLYGON"})
_WGS84 = "EPSG:4326"

#: Extent pieces are clipped to a chip's bounds widened by this share of its size, so
#: the clip's own edges fall outside the raster.
_CLIP_MARGIN = 0.05

#: Pieces are densified to this many vertices per chip width before reprojection, so a
#: straight edge in the extent's CRS stays where it is on a UTM grid (well under a pixel
#: for a 200-pixel chip).
_DENSIFY_STEPS = 2048


class LabelExtentError(ValueError):
    """A label extent file is missing, unreadable, or not polygons in the expected CRS."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path = path


@dataclass(frozen=True)
class LabelExtent:
    """A validated label extent file. Plain strings, so mask tasks can pickle it."""

    path: str
    geom_col: str


@dataclass
class LabelExtentChipsResult:
    """Outcome of applying a label extent to the chips file."""

    total_chips: int
    kept_chips: int

    @property
    def dropped_chips(self) -> int:
        return self.total_chips - self.kept_chips


def label_extent_summary(result: LabelExtentChipsResult) -> str:
    """One line for a run summary."""
    return (
        f"Label extent: kept {result.kept_chips:,} of {result.total_chips:,} chips "
        f"({result.dropped_chips:,} wholly outside dropped)"
    )


def _crs_is_explicit_null(path: Path, geom_col: str) -> bool:
    """Whether the GeoParquet metadata sets ``crs: null``: an unknown CRS, not WGS84.

    The GeoParquet spec defaults a *missing* ``crs`` key to OGC:CRS84, but an explicit
    null means the CRS is undefined, so it cannot be trusted to line up with the chips.
    """
    conn = duckdb.connect(":memory:")
    try:
        row = conn.execute(
            "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'geo'", [str(path)]
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return False
    column = json.loads(row[0]).get("columns", {}).get(geom_col, {})
    return "crs" in column and column["crs"] is None


def _crs_of(path: Path, geom_col: str) -> pyproj.CRS:
    """The file's CRS; GeoParquet without one is WGS84 by definition."""
    info = detect_crs(path, geom_col)
    candidate = info.authority_code or info.projjson or info.wkt
    return pyproj.CRS.from_user_input(candidate or _WGS84)


def _check_crs(path: Path, geom_col: str, expected: pyproj.CRS) -> None:
    if _crs_is_explicit_null(path, geom_col):
        raise LabelExtentError(
            path, f"label_extent {path} declares no CRS (crs: null); write it in EPSG:4326"
        )
    try:
        actual = _crs_of(path, geom_col)
    except pyproj.exceptions.CRSError as err:
        raise LabelExtentError(path, f"label_extent {path}: unreadable CRS ({err})") from err
    if not actual.equals(expected, ignore_axis_order=True):
        raise LabelExtentError(
            path,
            f"label_extent {path} is in {actual.to_string()}, but the chips and fields are in "
            f"{expected.to_string()}. Reproject the extent first.",
        )


def _geometry_types(path: Path, geom_col: str) -> set[str]:
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    column = geom_col.replace('"', '""')
    try:
        rows = conn.execute(
            f'SELECT DISTINCT upper(ST_GeometryType("{column}")::VARCHAR) '
            f"FROM read_parquet('{sql_path(path)}')"
        ).fetchall()
    except duckdb.Error as err:
        raise LabelExtentError(path, f"label_extent {path} could not be read: {err}") from err
    finally:
        conn.close()
    return {row[0] for row in rows if row[0] is not None}


def load_label_extent(path: str | Path, expected_crs: str | None = None) -> LabelExtent:
    """Validate a label extent file and return it ready for the chips and masks stages.

    Args:
        path: GeoParquet of the polygons inside which the labels are complete.
        expected_crs: CRS the extent must be in (anything pyproj reads); defaults to
            EPSG:4326, the CRS of the pipeline's chips and fields.

    Raises:
        LabelExtentError: If the file is missing, has no geometry column, holds no
            geometries, holds anything but polygons, or is in another CRS.
    """
    extent_path = Path(path).resolve()
    if not extent_path.is_file():
        raise LabelExtentError(extent_path, f"label_extent file not found: {extent_path}")
    geom_col = detect_geometry_column(extent_path)
    if geom_col is None:
        raise LabelExtentError(
            extent_path, f"label_extent {extent_path} is not GeoParquet (no geometry column)"
        )
    _check_crs(extent_path, geom_col, pyproj.CRS.from_user_input(expected_crs or _WGS84))
    types = _geometry_types(extent_path, geom_col)
    if not types:
        raise LabelExtentError(extent_path, f"label_extent {extent_path} has no geometries")
    other = sorted(types - _POLYGON_TYPES)
    if other:
        raise LabelExtentError(
            extent_path,
            f"label_extent {extent_path} must hold polygons only, found {', '.join(other)}",
        )
    return LabelExtent(path=str(extent_path), geom_col=geom_col)


@functools.lru_cache(maxsize=4)
def _extent_index(
    path: str, geom_col: str, _stamp: tuple[int, int]
) -> tuple[np.ndarray, shapely.STRtree]:
    """The extent's geometries and a spatial index, read once per process.

    ``_stamp`` (mtime, size) makes a rewritten file a cache miss.
    """
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    column = geom_col.replace('"', '""')
    try:
        rows = conn.execute(
            f'SELECT ST_AsWKB(ST_MakeValid("{column}")) '
            f"FROM read_parquet('{sql_path(path)}') WHERE \"{column}\" IS NOT NULL"
        ).fetchall()
    finally:
        conn.close()
    geometries = shapely.from_wkb([bytes(row[0]) for row in rows])
    return geometries, shapely.STRtree(geometries)


def extent_pieces(extent: LabelExtent, bounds: tuple[float, float, float, float]) -> list:
    """The extent's polygons near ``bounds``, clipped to them and densified.

    In the extent's (= the chips') CRS, ready to reproject onto a chip's raster grid.
    Each worker loads the extent once and keeps it indexed, so a chip only touches the
    polygons it overlaps, and only their part under it, however detailed they are.
    """
    stat = Path(extent.path).stat()
    geometries, tree = _extent_index(extent.path, extent.geom_col, (stat.st_mtime_ns, stat.st_size))
    minx, miny, maxx, maxy = bounds
    span = max(maxx - minx, maxy - miny)
    pad = span * _CLIP_MARGIN
    window = (minx - pad, miny - pad, maxx + pad, maxy + pad)
    hits = tree.query(shapely.box(*window))
    clipped = shapely.clip_by_rect(geometries[hits], *window)
    clipped = clipped[shapely.area(clipped) > 0]
    return list(shapely.segmentize(clipped, span / _DENSIFY_STEPS))


def _extent_query(chips_sql: str, extent: LabelExtent, chips_geom_col: str) -> str:
    """Chips overlapping the extent, each with the share of it inside (0-100).

    The pieces of each chip inside the extent are unioned before measuring, so
    overlapping extent polygons are not counted twice. A chip that only touches the
    extent along an edge has nothing inside and is dropped with the rest.
    """
    chip_geom = f'ST_MakeValid(c."{chips_geom_col}")'
    extent_geom = f'ST_MakeValid(e."{extent.geom_col}")'
    return f"""
        WITH chips AS ({chips_sql}),
        numbered AS (SELECT row_number() OVER () AS __chip_row, * FROM chips),
        inside AS (
            SELECT c.__chip_row, ST_Union_Agg(ST_Intersection({chip_geom}, {extent_geom})) AS geom
            FROM numbered c
            JOIN read_parquet('{sql_path(extent.path)}') e
              ON ST_Intersects({chip_geom}, {extent_geom})
            GROUP BY c.__chip_row
        )
        SELECT c.* EXCLUDE (__chip_row),
            ROUND(100.0 * ST_Area(i.geom) / ST_Area({chip_geom}), 2) AS "{LABEL_EXTENT_COLUMN}"
        FROM numbered c
        JOIN inside i ON c.__chip_row = i.__chip_row
        WHERE ST_Area(i.geom) > 0
    """


def apply_label_extent(chips_file: str | Path, extent_file: str | Path) -> LabelExtentChipsResult:
    """Drop chips wholly outside the label extent and record each kept chip's share inside.

    Rewrites the chips GeoParquet in place, adding ``label_extent_pct`` (0-100, the
    same planar-area share ``field_coverage_pct`` uses). Chips that partly overlap
    the extent are kept: their masks mark the part outside it as unknown.

    Raises:
        LabelExtentError: If the extent is invalid or not in the chips' CRS.
    """
    chips_path = Path(chips_file).resolve()
    chips_geom_col = detect_geometry_column(chips_path) or "geometry"
    chips_crs = _crs_of(chips_path, chips_geom_col)
    extent = load_label_extent(extent_file, expected_crs=chips_crs.to_wkt())

    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    try:
        chips_sql = select_excluding(chips_path, [LABEL_EXTENT_COLUMN])
        (total,) = conn.execute(f"SELECT count(*) FROM ({chips_sql})").fetchone()
        conn.execute(f"CREATE TABLE kept AS {_extent_query(chips_sql, extent, chips_geom_col)}")
        (kept,) = conn.execute("SELECT count(*) FROM kept").fetchone()
        write_chips(chips_path, conn, "SELECT * FROM kept")
    finally:
        conn.close()

    result = LabelExtentChipsResult(total_chips=total, kept_chips=kept)
    logger.info(
        f"Label extent: kept {kept:,} of {total:,} chips "
        f"({result.dropped_chips:,} wholly outside the extent dropped)"
    )
    if kept == 0:
        logger.warning("No chip overlaps the label extent; check the extent file")
    return result
