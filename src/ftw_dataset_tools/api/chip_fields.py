"""Each chip's field outlines as a small GeoParquet beside its masks.

The file holds the fields that overlap the chip, cut to the masks' footprint and in
the masks' CRS, with every column of the fields file in its own types. Its ``id``
joins it to the instance mask (directly, or through the mask's ``_labels.json`` where
pieces were renumbered) and to the dataset's fields file.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
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
from ftw_dataset_tools.api.masks import MaskType

logger = get_logger(__name__)

CHIP_FIELDS_SUFFIX = "_fields.parquet"
_WKB = "__ftw_wkb"
# Rows per Arrow batch on load: a batch's binary and string columns share 32-bit
# offsets (2 GiB), which DuckDB's default of 1M rows passes at ~2 kB of WKB a field.
_ROWS_PER_BATCH = 100_000


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
    """The fields file held in memory, with an index over each field's bounds.

    One scan of the file serves every chip: a query per chip read whole row groups
    and took about 0.4 s, which made this step longer than the masks on big datasets.
    It costs memory instead: the file's columns plus ~500 B a field for the index,
    about 25 GB at peak for 30M fields (jp_2021).
    """

    path: Path
    geom_col: str
    crs: object
    # Every kept column, then the geometry as WKB in _WKB. Kept as batches: one
    # chunk could overflow binary's 32-bit offsets, and take() on a chunked table
    # concatenates every chunk first.
    batches: tuple[pa.RecordBatch, ...]
    starts: np.ndarray  # the first row of each batch
    schema: pa.Schema
    tree: shapely.STRtree  # over the fields' bounding boxes, in row order


def _covering_columns(fields_file: Path, geom_col: str) -> set[str]:
    """The bbox covering columns: they describe whole fields in the source CRS."""
    meta = geoparquet_column_metadata(fields_file, geom_col) or {}
    bbox = (meta.get("covering") or {}).get("bbox") or {}
    return {path[0] for path in bbox.values() if path} | {"bbox"}


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _load(fields_file: Path, geom_col: str, columns: list[str]) -> tuple[pa.Table, np.ndarray]:
    """The kept columns plus WKB, and each field's bounds; empty geometries never intersect."""
    cols = "".join(f"{_quote(c)}, " for c in columns)
    geom = _quote(geom_col)
    con = duckdb.connect()
    try:
        ensure_spatial_loaded(con)
        table = con.execute(
            f"""SELECT {cols}ST_AsWKB({geom}) AS {_WKB},
                    ST_XMin({geom}) AS __x0, ST_YMin({geom}) AS __y0,
                    ST_XMax({geom}) AS __x1, ST_YMax({geom}) AS __y1
                FROM '{sql_path(fields_file)}'
                WHERE {geom} IS NOT NULL AND NOT ST_IsEmpty({geom})"""
        ).fetch_arrow_table(rows_per_batch=_ROWS_PER_BATCH)
    finally:
        con.close()
    bounds = np.column_stack([table.column(c).to_numpy() for c in ("__x0", "__y0", "__x1", "__y1")])
    table = _source_zones(table.drop_columns(["__x0", "__y0", "__x1", "__y1"]), fields_file)
    return table, bounds


def _source_zones(table: pa.Table, fields_file: Path) -> pa.Table:
    """Timestamps in the zone the file gives them: DuckDB returns them in the machine's."""
    source = pq.read_schema(fields_file)
    for i, f in enumerate(table.schema):
        want = source.field(f.name).type if f.name in source.names else None
        if pa.types.is_timestamp(f.type) and want is not None and want != f.type:
            table = table.set_column(i, f.with_type(want), table.column(i).cast(want))
    return table


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
        columns = [c for c in names if c != geom_col and c not in dropped]
        table, bounds = _load(Path(fields_file), geom_col, columns)
        tree = shapely.STRtree(shapely.box(bounds[:, 0], bounds[:, 1], bounds[:, 2], bounds[:, 3]))
    except (duckdb.Error, OSError, ValueError) as err:
        raise ChipFieldsError(f"Cannot read {fields_file}: {err}") from err
    batches = tuple(table.to_batches())
    starts = np.cumsum([0] + [len(b) for b in batches[:-1]], dtype=np.int64)
    return _Source(Path(fields_file), geom_col, crs, batches, starts, table.schema, tree)


def _take(source: _Source, rows: np.ndarray) -> pa.Table:
    """Rows by position, ``rows`` sorted, taken batch by batch."""
    which = np.searchsorted(source.starts, rows, side="right") - 1
    parts = [
        source.batches[b].take(pa.array(rows[which == b] - source.starts[b], type=pa.int64()))
        for b in np.unique(which)
    ]
    return pa.Table.from_batches(parts, schema=source.schema)


def _envelopes(bounds: tuple[float, float, float, float]) -> shapely.Geometry:
    """The chip's footprint in the fields' CRS; split in two where it crosses 180 degrees."""
    minx, miny, maxx, maxy = bounds
    if minx <= maxx:
        return shapely.box(minx, miny, maxx, maxy)
    return shapely.MultiPolygon(
        [shapely.box(minx, miny, 180, maxy), shapely.box(-180, miny, maxx, maxy)]
    )


def _query(source: _Source, bounds: tuple[float, float, float, float]) -> pa.Table:
    """The fields that intersect ``bounds``, in file order."""
    envelope = _envelopes(bounds)
    rows = np.unique(np.concatenate([source.tree.query(e) for e in shapely.get_parts(envelope)]))
    table = _take(source, rows)
    shapes = shapely.from_wkb(table.column(_WKB).to_numpy(zero_copy_only=False))
    return table.filter(pa.array(shapely.intersects(shapes, envelope), type=pa.bool_()))


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
    if len(shapes):
        column["bbox"] = [float(v) for v in shapely.total_bounds(shapes)]
    geo = {"version": "1.1.0", "primary_column": geom_col, "columns": {geom_col: column}}
    return table.replace_schema_metadata({**(table.schema.metadata or {}), b"geo": json.dumps(geo)})


def _write(table: pa.Table, geom_col: str, crs, out: Path) -> None:
    from pyproj import CRS

    tmp = out.with_name(f".{out.name}.tmp")
    try:
        geo_table = _with_geo_metadata(table, geom_col, CRS.from_user_input(crs))
        # pyarrow, not gpio: gpio 1.x's default writer opens a DuckDB connection and loads
        # the spatial extension on every call, ~90 ms for a chip's few rows and ~9 files/s
        # across threads, which made this step take hours on big datasets. Its in-memory
        # writer needs DuckDB 1.5. The table already carries its full 'geo' metadata; level
        # 15 is what gpio wrote.
        pq.write_table(geo_table, tmp, compression="zstd", compression_level=15)
        tmp.replace(out)
    finally:
        tmp.unlink(missing_ok=True)


def _fresh(out: Path, *inputs: Path) -> bool:
    """Whether ``out`` exists and is newer than every input."""
    if not out.is_file():
        return False
    stamp = out.stat().st_mtime
    return all(stamp >= p.stat().st_mtime for p in inputs)


def _plan(chip_dir: Path, fields_file: Path, skip_existing: bool) -> tuple[str, Path | None]:
    """What a chip needs before the fields are loaded: no mask, kept, or its mask to cut to."""
    mask_path = _footprint_mask(chip_dir)
    if mask_path is None:
        return "no_mask", None
    if skip_existing and _fresh(chip_fields_path(chip_dir), mask_path, fields_file):
        return "skipped", None
    return "write", mask_path


def _write_chip(source: _Source, chip_dir: Path, mask_path: Path) -> tuple[str, str]:
    out = chip_fields_path(chip_dir)
    try:
        table, crs = _clipped(source, mask_path)
        _write(table, source.geom_col, crs, out)
    except Exception as err:  # one bad chip must not cost every other chip's outlines
        out.unlink(missing_ok=True)  # nor leave an earlier run's outlines to be published
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

    fields_file = Path(fields_file)
    if not fields_file.is_file():
        raise ChipFieldsError(f"Cannot read {fields_file}: no such file")
    chip_dirs = iter_chip_dirs(Path(collection_dir))
    plans = [_plan(d, fields_file, skip_existing) for d in chip_dirs]
    todo = [(d, mask) for d, (step, mask) in zip(chip_dirs, plans, strict=True) if step == "write"]
    written = _write_chips(fields_file, todo, workers)
    outcomes = [written.get(d, (step, "")) for d, (step, _) in zip(chip_dirs, plans, strict=True)]
    return _tally(chip_dirs, outcomes)


def _write_chips(
    fields_file: Path, todo: list[tuple[Path, Path]], workers: int
) -> dict[Path, tuple[str, str]]:
    """Each chip's outcome; the fields load only when a chip needs them, so a rerun
    that keeps every file skips the scan."""
    if not todo:
        return {}
    source = _source(fields_file)
    with ThreadPoolExecutor(max(1, workers)) as pool:
        outcomes = pool.map(lambda job: _write_chip(source, *job), todo)
        return {d: outcome for (d, _), outcome in zip(todo, outcomes, strict=True)}


def _tally(chip_dirs: list[Path], outcomes: list[tuple[str, str]]) -> ChipFieldsResult:
    result = ChipFieldsResult()
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
