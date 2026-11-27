"""Repair invalid field polygons before any stage turns them into lines or pixels.

Reprojecting fields to EPSG:4326 can make a valid polygon invalid: a hair-thin slit
or spike in the source collapses, and its edges cross. ``ST_Boundary`` of such a
polygon keeps the slit, so the boundary lines draw it across the field's interior,
and each consumer of the fields (lines, mask fill, per-chip outlines) would read the
broken shape its own way. The reproject stage therefore replaces every invalid
geometry with the polygonal part of its repair (``make_valid`` with the
``structure`` method: the area the rings enclose, overlapping parts unioned, and
collapsed slits, spikes and other line or point debris left out). Parts of the
repair under 1 m² (the folded triangles a crossing leaves) are left out too, unless
every part is that small. A field with no area left is dropped. Valid geometries, every other column, the row order and the
file metadata are kept; the bbox column is recomputed for the repaired rows.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from ftw_dataset_tools.api.chip_grid import reprojector
from ftw_dataset_tools.api.field_ids import AREA_CRS, SLIVER_M2, _native_geometry_columns
from ftw_dataset_tools.api.geo import (
    detect_geometry_column,
    ensure_spatial_loaded,
    geoparquet_crs,
    get_bbox_column_name,
    sql_path,
)
from ftw_dataset_tools.api.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Callable

logger = get_logger(__name__)

_ID_COLUMN = "id"
# Examples of repaired and dropped field ids kept for the log.
_EXAMPLES = 5
# The rewrite's zstd level, as the dedupe step writes the fields file.
_ZSTD_LEVEL = 6
_POLYGONAL = ("Polygon", "MultiPolygon")


@dataclass
class RepairResult:
    """What :func:`repair_geometries` did to a fields file."""

    rows_in: int = 0
    repaired: int = 0  # invalid geometries replaced by their repair
    dropped: int = 0  # invalid geometries with no area left; their rows were removed
    unreadable: int = 0  # geometries GEOS could not read at all; also removed
    repaired_ids: list[str] = field(default_factory=list)  # up to _EXAMPLES
    dropped_ids: list[str] = field(default_factory=list)  # up to _EXAMPLES

    @property
    def changed(self) -> bool:
        return bool(self.repaired or self.dropped or self.unreadable)

    def summary(self) -> str:
        """One line for the log."""
        if not self.changed:
            return f"Field geometries are valid ({self.rows_in:,} fields)"
        line = f"Repaired {self.repaired:,} invalid field geometries"
        if self.repaired_ids:
            line += f" (e.g. id {', '.join(self.repaired_ids)})"
        if self.dropped:
            line += f"; dropped {self.dropped:,} with no area left"
            if self.dropped_ids:
                line += f" (e.g. id {', '.join(self.dropped_ids)})"
        if self.unreadable:
            line += f"; dropped {self.unreadable:,} GEOS could not read"
        return line


def repaired_shape(
    shape: shapely.Geometry, to_area_crs: Callable | None = None
) -> shapely.Geometry | None:
    """The polygonal part of ``shape``'s repair, or None when it has no area.

    With ``to_area_crs`` (a ``shapely.transform`` function into an equal-area CRS in
    metres), parts under ``SLIVER_M2`` are left out unless all of them are.
    """
    fixed = shapely.make_valid(shape, method="structure", keep_collapsed=False)
    if fixed.geom_type not in _POLYGONAL:
        parts = [p for p in shapely.get_parts(fixed) if p.geom_type in _POLYGONAL]
        fixed = shapely.union_all(parts) if parts else None
    if fixed is None or fixed.is_empty or fixed.area <= 0:
        return None
    return _without_slivers(fixed, to_area_crs) if to_area_crs is not None else fixed


def _without_slivers(shape: shapely.Geometry, to_area_crs: Callable) -> shapely.Geometry:
    """``shape`` without its parts under ``SLIVER_M2``, unless that is all of it."""
    pieces = shapely.get_parts(shape)
    if len(pieces) == 1:
        return shape
    areas = shapely.area(shapely.transform(pieces, to_area_crs))
    kept = [p for p, a in zip(pieces, areas, strict=True) if a >= SLIVER_M2]
    if not kept or len(kept) == len(pieces):
        return shape
    return kept[0] if len(kept) == 1 else shapely.MultiPolygon(kept)


def _quoted(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def _invalid_rows(path: Path, geom_col: str) -> dict[int, tuple[bytes, str | None]]:
    """Each invalid geometry by row number: its WKB and the row's id as text.

    A parallel DuckDB scan, so a file of valid fields costs one pass and no decoding
    in Python; invalid geometries are few, so they fit in memory.
    """
    geom = _quoted(geom_col)
    con = duckdb.connect(":memory:")
    try:
        ensure_spatial_loaded(con)
        columns = {
            r[0] for r in con.execute(f"DESCRIBE SELECT * FROM '{sql_path(path)}'").fetchall()
        }
        fid = f"CAST({_quoted(_ID_COLUMN)} AS VARCHAR)" if _ID_COLUMN in columns else "NULL"
        rows = con.execute(
            f"""SELECT file_row_number, ST_AsWKB({geom}), {fid}
                FROM read_parquet('{sql_path(path)}', file_row_number = true)
                WHERE {geom} IS NOT NULL AND NOT ST_IsValid({geom})
                ORDER BY file_row_number"""
        ).fetchall()
    finally:
        con.close()
    return {int(n): (bytes(wkb), fid) for n, wkb, fid in rows}


def repair_geometries(path: Path | str) -> RepairResult:
    """Repair the invalid geometries of a GeoParquet fields file in place; see the module doc.

    A file whose geometries are all valid is not rewritten.
    """
    path = Path(path)
    geom_col = detect_geometry_column(path) or "geometry"
    result = RepairResult(rows_in=pq.ParquetFile(path).metadata.num_rows)
    invalid = _invalid_rows(path, geom_col)
    if invalid:
        to_area_crs = reprojector(geoparquet_crs(path, geom_col), AREA_CRS)
        repairs = _repairs(invalid, result, to_area_crs)
        tmp = path.with_name(f".{path.name}.repair.tmp")
        try:
            _rewrite(path, tmp, geom_col, repairs, result)
            tmp.replace(path)
        finally:
            tmp.unlink(missing_ok=True)
    logger.info(result.summary())
    return result


def _repairs(
    invalid: dict[int, tuple[bytes, str | None]], result: RepairResult, to_area_crs: Callable
) -> dict[int, shapely.Geometry | None]:
    """Each invalid row's repaired geometry, None for one to drop; counted on ``result``."""
    repairs: dict[int, shapely.Geometry | None] = {}
    for row, (wkb, fid) in invalid.items():
        try:
            fixed = repaired_shape(shapely.from_wkb(wkb, on_invalid="fix"), to_area_crs)
        except shapely.errors.GEOSException:
            repairs[row] = None
            result.unreadable += 1
            continue
        repairs[row] = fixed
        _count(result, fixed, fid)
    return repairs


def _count(result: RepairResult, fixed: shapely.Geometry | None, fid: str | None) -> None:
    if fixed is None:
        result.dropped += 1
        examples = result.dropped_ids
    else:
        result.repaired += 1
        examples = result.repaired_ids
    if fid is not None and len(examples) < _EXAMPLES:
        examples.append(fid)


def warn_if_invalid(path: Path | str) -> int:
    """Log a warning when ``path`` holds invalid geometries; returns how many.

    The pipeline repairs them in its reproject stage. A standalone command given a
    raw file burns them as they are, so their boundary lines may cross the fields.
    """
    path = Path(path)
    geom_col = detect_geometry_column(path)
    if geom_col is None:
        return 0  # not GeoParquet: the command reports that itself
    try:
        count = len(_invalid_rows(path, geom_col))
    except duckdb.Error:
        return 0
    if count:
        logger.warning(
            f"{path.name} has {count:,} invalid geometries; their boundary lines can cross "
            "the field. `ftwd run` repairs them in its reproject stage."
        )
    return count


@dataclass(frozen=True)
class _Columns:
    geom: int
    bbox: int | None


def _columns(schema: pa.Schema, path: Path, geom_col: str) -> _Columns:
    bbox_name = get_bbox_column_name(path)
    if bbox_name is None and "bbox" in schema.names:
        bbox_name = "bbox"
    bbox = schema.get_field_index(bbox_name) if bbox_name else -1
    if bbox >= 0 and not pa.types.is_struct(schema.field(bbox).type):
        bbox = -1
    return _Columns(geom=schema.get_field_index(geom_col), bbox=bbox if bbox >= 0 else None)


def _rewrite(
    src: Path,
    dst: Path,
    geom_col: str,
    repairs: dict[int, shapely.Geometry | None],
    result: RepairResult,
) -> None:
    """Write ``src`` to ``dst`` one row group at a time, with ``repairs`` applied."""
    if _native_geometry_columns(src):
        import geoarrow.pyarrow  # noqa: F401  (keeps a Parquet GEOMETRY column native)
    with pq.ParquetFile(src) as source:
        schema = source.schema_arrow
        columns = _columns(schema, src, geom_col)
        kept = [g.geom_type for g in repairs.values() if g is not None]
        metadata = _metadata(schema.metadata or {}, geom_col, set(kept), result)
        schema = schema.with_metadata(metadata)
        positions = np.fromiter(repairs, dtype=np.int64)  # in row order
        shapes = list(repairs.values())
        start = 0
        with pq.ParquetWriter(
            dst, schema, compression="zstd", compression_level=_ZSTD_LEVEL
        ) as writer:
            for i in range(source.num_row_groups):
                table = source.read_row_group(i)
                lo, hi = np.searchsorted(positions, [start, start + len(table)])
                rows = {int(positions[k]) - start: shapes[k] for k in range(lo, hi)}
                start += len(table)
                writer.write_table(_apply(table, columns, rows).cast(schema))


def _apply(
    table: pa.Table, columns: _Columns, rows: dict[int, shapely.Geometry | None]
) -> pa.Table:
    """``table`` with the geometries (and bboxes) of ``rows`` replaced, None rows removed."""
    if not rows:
        return table
    fixed = {r: g for r, g in rows.items() if g is not None}
    if fixed:
        table = _set_geometries(table, columns.geom, fixed)
        if columns.bbox is not None:
            table = _set_bboxes(table, columns.bbox, fixed)
    keep = np.ones(len(table), dtype=bool)
    keep[[r for r, g in rows.items() if g is None]] = False
    return table if keep.all() else table.filter(pa.array(keep))


def _set_geometries(table: pa.Table, index: int, fixed: dict[int, shapely.Geometry]) -> pa.Table:
    column = table.column(index).combine_chunks()
    kind = column.type
    storage = column.storage if isinstance(kind, pa.BaseExtensionType) else column
    wkb = storage.to_numpy(zero_copy_only=False).copy()
    for row, shape in fixed.items():
        wkb[row] = shapely.to_wkb(shape)
    values = pa.array(wkb, type=storage.type)
    if isinstance(kind, pa.BaseExtensionType):
        values = pa.ExtensionArray.from_storage(kind, values)
    return table.set_column(index, table.schema.field(index), values)


def _set_bboxes(table: pa.Table, index: int, fixed: dict[int, shapely.Geometry]) -> pa.Table:
    """The bbox column with the ``fixed`` rows' bounds, rounded outward for float32."""
    column = table.column(index).combine_chunks()
    sides = {
        f.name: column.field(f.name).to_numpy(zero_copy_only=False).copy() for f in column.type
    }
    rows = np.fromiter(fixed, dtype=np.int64)
    # Null sides stay null, except on the rows whose bounds are written here.
    nulls = {}
    for f in column.type:
        null = column.field(f.name).is_null().to_numpy(zero_copy_only=False).copy()
        null[rows] = False
        nulls[f.name] = null
    bounds = shapely.bounds(np.array(list(fixed.values()), dtype=object))
    for k, name in enumerate(("xmin", "ymin", "xmax", "ymax")):
        if name in sides:
            sides[name] = sides[name].astype(np.float64)
            sides[name][rows] = bounds[:, k]
    arrays = [
        _outward(sides[f.name], f.type, down=f.name in ("xmin", "ymin"), nulls=nulls[f.name])
        for f in column.type
    ]
    values = pa.StructArray.from_arrays(arrays, fields=list(column.type), mask=column.is_null())
    return table.set_column(index, table.schema.field(index), values)


def _outward(
    values: np.ndarray, kind: pa.DataType, *, down: bool, nulls: np.ndarray | None = None
) -> pa.Array:
    """``values`` as ``kind``, ``nulls`` masked; a float32 bound is rounded away from
    the geometry."""
    if pa.types.is_float32(kind):
        narrow = values.astype(np.float32)
        off = narrow > values if down else narrow < values
        narrow[off] = np.nextafter(narrow[off], np.float32(-np.inf if down else np.inf))
        values = narrow
    return pa.array(values, type=kind, mask=nulls)


def _metadata(
    metadata: dict[bytes, bytes], geom_col: str, added_types: set[str], result: RepairResult
) -> dict[bytes, bytes]:
    """The file's metadata with ``geo`` listing any new geometry type, and a pandas range
    index shortened by the rows dropped.

    The file-level ``geo`` bbox is kept: a repair only shrinks or removes shapes, so it
    still contains every geometry.
    """
    metadata = {k: v for k, v in metadata.items() if k != b"ARROW:schema"}
    if b"geo" in metadata:
        geo = json.loads(metadata[b"geo"])
        column = geo.get("columns", {}).get(geom_col, {})
        if column.get("geometry_types"):
            column["geometry_types"] = sorted(set(column["geometry_types"]) | added_types)
        metadata[b"geo"] = json.dumps(geo).encode()
    if b"pandas" in metadata and result.dropped:
        pandas = json.loads(metadata[b"pandas"])
        pandas["index_columns"] = [
            _shortened(index, result.dropped) for index in pandas.get("index_columns", [])
        ]
        metadata[b"pandas"] = json.dumps(pandas).encode()
    return metadata


def _shortened(index: object, dropped: int) -> object:
    if not isinstance(index, dict) or index.get("kind") != "range":
        return index
    return {**index, "stop": index.get("stop", 0) - dropped * index.get("step", 1)}
