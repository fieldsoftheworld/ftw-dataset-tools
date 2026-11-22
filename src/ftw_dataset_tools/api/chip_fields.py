"""Each chip's field outlines as a small GeoParquet beside its masks.

The file holds the fields that overlap the chip, cut to the masks' footprint and in
the masks' CRS, with every column of the fields file in its own types. Its ``id``
joins it to the instance mask (directly, or through the mask's ``_labels.json`` where
pieces were renumbered) and to the dataset's fields file.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import rasterio
import shapely
from rasterio.warp import transform_bounds

from ftw_dataset_tools.api.chip_grid import reprojector
from ftw_dataset_tools.api.geo import (
    detect_geometry_column,
    ensure_spatial_loaded,
    geoparquet_column_metadata,
    geoparquet_crs,
    sql_path,
)
from ftw_dataset_tools.api.logging_config import get_logger
from ftw_dataset_tools.api.masks import MaskType, _bbox_prefilter

logger = get_logger(__name__)

CHIP_FIELDS_SUFFIX = "_fields.parquet"
_WKB = "__ftw_wkb"


class ChipFieldsError(Exception):
    """The fields file cannot be read for per-chip outlines."""


@dataclass
class ChipFieldsResult:
    written: int = 0
    skipped: int = 0
    no_mask: list[str] = field(default_factory=list)
    # (chip, reason) for chips whose outlines could not be written
    failed: list[tuple[str, str]] = field(default_factory=list)


def chip_fields_path(chip_dir: Path) -> Path:
    return chip_dir / f"{chip_dir.name}{CHIP_FIELDS_SUFFIX}"


def _footprint_mask(chip_dir: Path) -> Path | None:
    """The chip's first mask on disk; every mask of a chip shares one grid."""
    for mask_type in MaskType:
        path = chip_dir / f"{chip_dir.name}_{mask_type.value}.tif"
        if path.is_file():
            return path
    return None


@dataclass(frozen=True)
class _Source:
    path: Path
    geom_col: str
    crs: object
    columns: tuple[str, ...]


def _covering_columns(fields_file: Path, geom_col: str) -> set[str]:
    """The bbox covering columns: they describe whole fields in the source CRS."""
    meta = geoparquet_column_metadata(fields_file, geom_col) or {}
    bbox = (meta.get("covering") or {}).get("bbox") or {}
    return {path[0] for path in bbox.values() if path} | {"bbox"}


def _source(fields_file: Path) -> _Source:
    try:
        geom_col = detect_geometry_column(fields_file) or "geometry"
        con = duckdb.connect()
        try:
            names = [
                r[0]
                for r in con.execute(f"DESCRIBE SELECT * FROM '{sql_path(fields_file)}'").fetchall()
            ]
        finally:
            con.close()
        crs = geoparquet_crs(fields_file, geom_col)
        dropped = _covering_columns(fields_file, geom_col)
    except (duckdb.Error, OSError, ValueError) as err:
        raise ChipFieldsError(f"Cannot read {fields_file}: {err}") from err
    columns = tuple(c for c in names if c != geom_col and c not in dropped)
    return _Source(Path(fields_file), geom_col, crs, columns)


_local = threading.local()


def _connection() -> duckdb.DuckDBPyConnection:
    con = getattr(_local, "con", None)
    if con is None:
        con = duckdb.connect(":memory:")
        con.execute("SET threads = 1")  # one per pool thread
        ensure_spatial_loaded(con)
        _local.con = con
    return con


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _query(source: _Source, bounds: tuple[float, float, float, float]) -> pa.Table:
    minx, miny, maxx, maxy = bounds
    cols = "".join(f"{_quote(c)}, " for c in source.columns)
    geom = _quote(source.geom_col)
    prefilter = _bbox_prefilter(source.path, source.geom_col, bounds)
    return (
        _connection()
        .execute(
            f"""SELECT {cols}ST_AsWKB({geom}) AS {_WKB}
                FROM '{sql_path(source.path)}'
                WHERE {prefilter}ST_Intersects(
                    {geom}, ST_MakeEnvelope({minx!r}, {miny!r}, {maxx!r}, {maxy!r}))"""
        )
        .fetch_arrow_table()
    )


def _polygonal(shape: shapely.Geometry) -> shapely.Geometry | None:
    """The areal part of a cut: a field touching the edge leaves a line or point."""
    if shape is None or shape.is_empty:
        return None
    if shape.geom_type in ("Polygon", "MultiPolygon"):
        return shape
    parts = [g for g in getattr(shape, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.unary_union(parts) if parts else None


def _clipped(source: _Source, mask_path: Path) -> tuple[pa.Table, object]:
    """The fields under ``mask_path``'s footprint, cut to it, and the mask's CRS."""
    with rasterio.open(mask_path) as mask:
        crs, bounds = mask.crs, mask.bounds
    table = _query(source, transform_bounds(crs, source.crs, *bounds, densify_pts=21))
    shapes = shapely.from_wkb(table.column(_WKB).to_numpy(zero_copy_only=False))
    shapes = shapely.make_valid(shapely.transform(shapes, reprojector(source.crs, crs)))
    cut = [_polygonal(g) for g in shapely.intersection(shapes, shapely.box(*bounds))]
    keep = np.array([g is not None for g in cut], dtype=bool)
    table = table.drop_columns([_WKB]).filter(pa.array(keep, type=pa.bool_()))
    kept = [g for g in cut if g is not None]
    table = table.append_column(source.geom_col, pa.array(shapely.to_wkb(kept), pa.binary()))
    return table, crs


def _with_geo_metadata(table: pa.Table, geom_col: str, crs) -> pa.Table:
    shapes = shapely.from_wkb(table.column(geom_col).to_numpy(zero_copy_only=False))
    column = {
        "encoding": "WKB",
        "geometry_types": sorted({g.geom_type for g in shapes}),
        "crs": json.loads(crs.to_json()),
    }
    geo = {"version": "1.1.0", "primary_column": geom_col, "columns": {geom_col: column}}
    return table.replace_schema_metadata({**(table.schema.metadata or {}), b"geo": json.dumps(geo)})


def _write(table: pa.Table, geom_col: str, crs, out: Path) -> None:
    import geoparquet_io as gpio
    from pyproj import CRS

    tmp = out.with_name(f".{out.name}.tmp")
    try:
        geo_table = _with_geo_metadata(table, geom_col, CRS.from_user_input(crs))
        gpio.Table(geo_table, geom_col).write(str(tmp), overwrite=True)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def _fresh(out: Path, *inputs: Path) -> bool:
    """Whether ``out`` exists and is newer than every input."""
    if not out.is_file():
        return False
    stamp = out.stat().st_mtime
    return all(stamp >= p.stat().st_mtime for p in inputs)


def _write_chip(source: _Source, chip_dir: Path, skip_existing: bool) -> tuple[str, str]:
    mask_path = _footprint_mask(chip_dir)
    if mask_path is None:
        return "no_mask", ""
    out = chip_fields_path(chip_dir)
    if skip_existing and _fresh(out, mask_path, source.path):
        return "skipped", ""
    try:
        table, crs = _clipped(source, mask_path)
        _write(table, source.geom_col, crs, out)
    except Exception as err:  # one bad chip must not cost every other chip's outlines
        return "failed", f"{type(err).__name__}: {err}"
    return "written", ""


def write_chip_fields(
    fields_file: Path | str,
    collection_dir: Path | str,
    *,
    skip_existing: bool = False,
    workers: int = 8,
) -> ChipFieldsResult:
    """Write ``<chip>_fields.parquet`` beside the masks of every chip under ``collection_dir``.

    ``fields_file`` is the GeoParquet the masks were burned from. A chip without a mask
    gets no file. With ``skip_existing`` a file newer than its chip's mask and than
    ``fields_file`` is kept. A chip whose outlines fail is reported in ``failed``.
    """
    # Imported here: api.imagery's package init imports stac, which imports this module.
    from ftw_dataset_tools.api.imagery.catalog_ops import iter_chip_dirs

    source = _source(Path(fields_file))
    chip_dirs = iter_chip_dirs(Path(collection_dir))
    result = ChipFieldsResult()
    with ThreadPoolExecutor(max(1, workers)) as pool:
        outcomes = pool.map(lambda d: _write_chip(source, d, skip_existing), chip_dirs)
        for chip_dir, (outcome, reason) in zip(chip_dirs, outcomes, strict=True):
            if outcome == "written":
                result.written += 1
            elif outcome == "skipped":
                result.skipped += 1
            elif outcome == "failed":
                result.failed.append((chip_dir.name, reason))
            else:
                result.no_mask.append(chip_dir.name)
    return result


def remove_chip_fields(collection_dir: Path | str) -> int:
    """Delete outlines an earlier run wrote, so a run without them doesn't publish stale ones."""
    from ftw_dataset_tools.api.imagery.catalog_ops import iter_chip_dirs

    removed = 0
    for chip_dir in iter_chip_dirs(Path(collection_dir)):
        path = chip_fields_path(chip_dir)
        if path.is_file():
            path.unlink()
            removed += 1
    return removed


def summary_lines(result: ChipFieldsResult) -> list[str]:
    """What a run reports about the outlines, shared by `ftwd run` and create-masks."""
    lines = [f"Field outlines: {result.written} written, {result.skipped} kept"]
    if result.no_mask:
        lines.append(f"Field outlines: {len(result.no_mask)} chip(s) have no mask to cut to")
    if result.failed:
        chip, reason = result.failed[0]
        lines.append(
            f"Field outlines failed for {len(result.failed)} chip(s) (first: {chip}: {reason})"
        )
    return lines
