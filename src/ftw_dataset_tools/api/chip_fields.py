"""Each chip's field outlines as a small GeoParquet beside its masks.

The file holds the fields that overlap the chip, cut to the masks' footprint and in
the masks' CRS, with every column of the fields file. Its ``id`` joins it to the
instance mask (directly, or through the mask's ``_labels.json`` where pieces were
renumbered) and to the dataset's fields file.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import rasterio
import shapely
from rasterio.warp import transform_bounds

from ftw_dataset_tools.api.chip_grid import reprojector
from ftw_dataset_tools.api.geo import (
    detect_geometry_column,
    ensure_spatial_loaded,
    geoparquet_crs,
    sql_path,
    write_geoparquet,
)
from ftw_dataset_tools.api.logging_config import get_logger
from ftw_dataset_tools.api.masks import MaskType, _bbox_prefilter

if TYPE_CHECKING:
    import geopandas as gpd

logger = get_logger(__name__)

CHIP_FIELDS_SUFFIX = "_fields.parquet"
# The fields file's bbox describes the whole field, not the clipped shape.
_DROPPED = {"bbox"}


class ChipFieldsError(Exception):
    """The fields file cannot be read for per-chip outlines."""


@dataclass
class ChipFieldsResult:
    written: int = 0
    skipped: int = 0
    no_mask: list[str] = field(default_factory=list)


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


def _source(fields_file: Path) -> _Source:
    try:
        geom_col = detect_geometry_column(fields_file) or "geometry"
        con = duckdb.connect()
        names = [
            r[0]
            for r in con.execute(f"DESCRIBE SELECT * FROM '{sql_path(fields_file)}'").fetchall()
        ]
        con.close()
        crs = geoparquet_crs(fields_file, geom_col)
    except (duckdb.Error, OSError, ValueError) as err:
        raise ChipFieldsError(f"Cannot read {fields_file}: {err}") from err
    columns = tuple(c for c in names if c != geom_col and c not in _DROPPED)
    return _Source(Path(fields_file), geom_col, crs, columns)


_local = threading.local()


def _connection() -> duckdb.DuckDBPyConnection:
    con = getattr(_local, "con", None)
    if con is None:
        con = duckdb.connect(":memory:")
        ensure_spatial_loaded(con)
        _local.con = con
    return con


def _quote(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _clipped(source: _Source, mask_path: Path) -> gpd.GeoDataFrame:
    """The fields under ``mask_path``'s footprint, cut to it, in its CRS."""
    import geopandas as gpd  # here: stac imports this module, and `ftwd --help` stays light

    with rasterio.open(mask_path) as mask:
        crs, bounds = mask.crs, mask.bounds
    minx, miny, maxx, maxy = transform_bounds(crs, source.crs, *bounds, densify_pts=21)
    cols = "".join(f"{_quote(c)}, " for c in source.columns)
    rows = (
        _connection()
        .execute(
            f"""SELECT {cols}ST_AsWKB({_quote(source.geom_col)})
            FROM '{sql_path(source.path)}'
            WHERE {_bbox_prefilter(source.path, source.geom_col, (minx, miny, maxx, maxy))}ST_Intersects({_quote(source.geom_col)},
                                ST_MakeEnvelope({minx!r}, {miny!r}, {maxx!r}, {maxy!r}))"""
        )
        .fetchall()
    )
    footprint = shapely.box(*bounds)
    attrs, shapes = [], []
    for row in rows:
        shape = shapely.transform(shapely.from_wkb(bytes(row[-1])), reprojector(source.crs, crs))
        cut = _polygonal(shapely.intersection(shape, footprint))
        if cut is not None:
            attrs.append(row[:-1])
            shapes.append(cut)
    frame = gpd.GeoDataFrame(attrs, columns=list(source.columns), geometry=shapes, crs=crs)
    return frame


def _polygonal(shape: shapely.Geometry) -> shapely.Geometry | None:
    """The areal part of a cut: a field touching the edge leaves a line or point."""
    if shape.is_empty:
        return None
    if shape.geom_type in ("Polygon", "MultiPolygon"):
        return shape
    parts = [g for g in getattr(shape, "geoms", []) if g.geom_type in ("Polygon", "MultiPolygon")]
    return shapely.unary_union(parts) if parts else None


def _write_chip(source: _Source, chip_dir: Path, skip_existing: bool) -> str:
    mask_path = _footprint_mask(chip_dir)
    if mask_path is None:
        return "no_mask"
    out = chip_fields_path(chip_dir)
    if skip_existing and out.is_file() and out.stat().st_mtime >= mask_path.stat().st_mtime:
        return "skipped"
    write_geoparquet(out, gdf=_clipped(source, mask_path))
    return "written"


def write_chip_fields(
    fields_file: Path | str,
    collection_dir: Path | str,
    *,
    skip_existing: bool = False,
    workers: int = 8,
) -> ChipFieldsResult:
    """Write ``<chip>_fields.parquet`` beside the masks of every chip under ``collection_dir``.

    ``fields_file`` is the GeoParquet the masks were burned from. A chip without a mask
    gets no file. With ``skip_existing`` a file newer than its chip's mask is kept.
    """
    # Imported here: api.imagery's package init imports stac, which imports this module.
    from ftw_dataset_tools.api.imagery.catalog_ops import iter_chip_dirs

    source = _source(Path(fields_file))
    chip_dirs = iter_chip_dirs(Path(collection_dir))
    result = ChipFieldsResult()
    with ThreadPoolExecutor(max(1, workers)) as pool:
        outcomes = pool.map(lambda d: _write_chip(source, d, skip_existing), chip_dirs)
        for chip_dir, outcome in zip(chip_dirs, outcomes, strict=True):
            if outcome == "written":
                result.written += 1
            elif outcome == "skipped":
                result.skipped += 1
            else:
                result.no_mask.append(chip_dir.name)
    return result
