"""PMTiles for a collection's vectors, built with tippecanoe.

tippecanoe is an optional external binary. Callers check ``tippecanoe_available``
and decide whether its absence is a warning or an error.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api.geo import detect_crs, detect_geometry_column, ensure_spatial_loaded

if TYPE_CHECKING:
    from collections.abc import Callable

    from ftw_dataset_tools.api.geo import CRSInfo

_NAN_SAFE_TYPES = {"DOUBLE", "FLOAT", "REAL"}


def _sql_path(path: Path | str) -> str:
    """Escape a path for interpolation into a single-quoted SQL string literal."""
    return str(path).replace("'", "''")


@dataclass(frozen=True)
class TileSpec:
    """Configuration for tiling one collection's vectors with tippecanoe."""

    layer: str
    attributes: tuple[str, ...]
    min_zoom: int | None
    max_zoom: int | None
    guess_zoom: bool
    drop_densest: bool


CHIPS_TILES = TileSpec(
    "chips",
    (
        "id",
        "split",
        "field_coverage_pct",
        "hcat_dominant_code",
        "hcat_dominant_name_en",
        "hcat_dominant_pct",
    ),
    4,
    12,
    False,
    False,
)
FIELDS_TILES = TileSpec(
    "fields",
    ("id", "hcat:code", "hcat:name_en", "hcat:name", "metrics:area"),
    None,
    None,
    True,
    True,
)


def tippecanoe_available() -> bool:
    """Return whether the ``tippecanoe`` binary is on PATH."""
    return shutil.which("tippecanoe") is not None


def _column_types(con: duckdb.DuckDBPyConnection, path: Path) -> dict[str, str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{_sql_path(path)}')").fetchall()
    return {row[0]: row[1] for row in rows}


def _attribute_expr(name: str, duckdb_type: str) -> str:
    """Column expression that turns NaN into SQL NULL for float-ish types."""
    quoted = f'"{name}"'
    if duckdb_type.upper() in _NAN_SAFE_TYPES:
        return f"CASE WHEN isnan({quoted}) THEN NULL ELSE {quoted} END AS {quoted}"
    return quoted


def _geometry_expr(geom_col: str, crs: CRSInfo) -> str:
    quoted = f'"{geom_col}"'
    if crs.authority_code and crs.authority_code.upper() != "EPSG:4326":
        code = crs.authority_code.replace("'", "''")
        return f"ST_Transform({quoted}, '{code}', 'EPSG:4326', always_xy := true) AS geometry"
    return f"{quoted} AS geometry"


def export_geojsonseq(parquet: Path, out: Path, attributes: tuple[str, ...]) -> list[str]:
    """Write newline-delimited GeoJSON (EPSG:4326) with only the attributes present.

    Reprojects to EPSG:4326 when the source CRS differs. Returns the attribute
    names that were actually present (and kept) in the source file.
    """
    parquet = Path(parquet)
    out = Path(out)
    con = duckdb.connect(":memory:")
    ensure_spatial_loaded(con)
    try:
        geom_col = detect_geometry_column(parquet, con) or "geometry"
        crs = detect_crs(parquet, geom_col, con)
        column_types = _column_types(con, parquet)
        present = [a for a in attributes if a in column_types]
        attr_exprs = [_attribute_expr(a, column_types[a]) for a in present]
        select_cols = ", ".join([*attr_exprs, _geometry_expr(geom_col, crs)])
        select = f"SELECT {select_cols} FROM read_parquet('{_sql_path(parquet)}')"
        out.parent.mkdir(parents=True, exist_ok=True)
        con.execute(
            f"COPY ({select}) TO '{_sql_path(out)}' WITH (FORMAT GDAL, DRIVER 'GeoJSONSeq')"
        )
    finally:
        con.close()
    return present


def _tippecanoe_command(out: Path, seq: Path, spec: TileSpec) -> list[str]:
    cmd = ["tippecanoe", "-o", str(out), "-l", spec.layer, "--force", "--quiet"]
    if spec.guess_zoom:
        cmd.append("-zg")
    else:
        cmd += [f"-Z{spec.min_zoom}", f"-z{spec.max_zoom}"]
    if spec.drop_densest:
        cmd += ["--drop-densest-as-needed", "--extend-zooms-if-still-dropping"]
    cmd.append(str(seq))
    return cmd


def build_pmtiles(
    parquet: Path,
    out: Path,
    spec: TileSpec,
    *,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """Export ``parquet`` to GeoJSONSeq and tile it into a PMTiles archive at ``out``.

    Raises RuntimeError if the ``tippecanoe`` binary is not available, or if it
    exits with a non-zero status. The intermediate GeoJSONSeq file is always
    removed afterwards.
    """
    if not tippecanoe_available():
        raise RuntimeError("tippecanoe is not installed; cannot build PMTiles")
    out = Path(out)
    seq = out.with_suffix(".geojsonseq")
    kept = export_geojsonseq(Path(parquet), seq, spec.attributes)
    cmd = _tippecanoe_command(out, seq, spec)
    if on_progress:
        on_progress(f"tippecanoe: {spec.layer} ({', '.join(kept) or 'geometry only'})")
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as err:
        raise RuntimeError(
            f"tippecanoe failed for {spec.layer}: {err.stderr.strip()[:500]}"
        ) from err
    finally:
        seq.unlink(missing_ok=True)
    return out
