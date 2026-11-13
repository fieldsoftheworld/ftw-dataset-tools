"""Per-chip administrative unit names (country, first and second level) from FAO GAUL 2024.

Each chip takes the GAUL level-2 unit that contains its centre. A chip whose centre
falls in no unit (offshore, or in a gap between units) takes the unit it overlaps most,
and keeps empty names only when it touches none. Where units overlap (disputed areas),
the lowest ``gaul2_code`` wins, so a rerun gives the same names.

The names are written back into the chips GeoParquet (``OUTPUT_COLUMNS``) and published
on every chip's STAC item as ``ftw:<column>``. GAUL marks units it has no name for as
"Administrative Unit Not Available"; those are left empty.

GAUL 2024 is published by FAO under CC BY 4.0. ftwd reads the GeoParquet copy on Source
Cooperative (nlebovits/gaul-l2-admin), one file per country with a bbox column, so only
the units near the chips are read.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api.chips_io import (
    drop_columns,
    parquet_columns,
    select_excluding,
    write_chips,
)
from ftw_dataset_tools.api.geo import (
    configure_source_coop_s3,
    detect_crs,
    detect_geometry_column,
    ensure_spatial_loaded,
    sql_path,
)
from ftw_dataset_tools.api.logging_config import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = get_logger(__name__)

GAUL_SOURCE = "s3://us-west-2.opendata.source.coop/nlebovits/gaul-l2-admin/by_country/*.parquet"
GAUL_URL = "https://source.coop/nlebovits/gaul-l2-admin"
ATTRIBUTION = (
    "FAO Global Administrative Unit Layers (GAUL) 2024, CC-BY-4.0 "
    "(source of the ftw:admin* properties)"
)

#: Chips columns this step writes, mapped from the GAUL columns they come from.
SOURCE_COLUMNS = {
    "admin0_iso3": "iso3_code",
    "admin0_name": "gaul0_name",
    "admin1_name": "gaul1_name",
    "admin2_name": "gaul2_name",
}
OUTPUT_COLUMNS: tuple[str, ...] = tuple(SOURCE_COLUMNS)

#: GAUL's placeholder for a unit without a name.
UNNAMED = "Administrative Unit Not Available"

#: Degrees added around the chips' bbox when selecting units, so a unit whose float32
#: bbox was rounded inwards is not missed.
BBOX_MARGIN = 0.01


class AdminNamesError(Exception):
    """The chips cannot be given admin names (wrong CRS, or no id column)."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path = path


@dataclass(frozen=True)
class AdminNamesResult:
    """Outcome of :func:`add_admin_names`."""

    total_chips: int
    by_centre: int
    by_overlap: int
    countries: tuple[str, ...]

    @property
    def unnamed(self) -> int:
        return self.total_chips - self.by_centre - self.by_overlap


def admin_names_summary(result: AdminNamesResult) -> str:
    """One line for a run summary."""
    line = (
        f"Admin names: {result.by_centre + result.by_overlap:,} of {result.total_chips:,} "
        f"chips named ({len(result.countries)} countries)"
    )
    if result.by_overlap:
        line += f", {result.by_overlap:,} by overlap because their centre is in no unit"
    if result.unnamed:
        line += f", {result.unnamed:,} touch no unit"
    return line


def drop_admin_names(chips_file: Path | str) -> bool:
    """Remove the admin name columns from a chips GeoParquet; True when it had any."""
    return drop_columns(chips_file, OUTPUT_COLUMNS)


def _check_chips(chips_path: Path, id_col: str) -> str:
    """The chips' geometry column, after checking they are lon/lat and have ids."""
    geom_col = detect_geometry_column(chips_path)
    crs = detect_crs(chips_path, geom_col)
    if crs.authority_code not in (None, "EPSG:4326", "OGC:CRS84"):
        raise AdminNamesError(
            chips_path, f"{chips_path.name} is in {crs.authority_code}; admin names need EPSG:4326"
        )
    if id_col not in parquet_columns(chips_path):
        raise AdminNamesError(chips_path, f"{chips_path.name} has no '{id_col}' column")
    return geom_col


def _load_units(con: duckdb.DuckDBPyConnection, source: str) -> None:
    """Table ``_units``: the GAUL units whose bbox meets the chips' bbox."""
    xmin, ymin, xmax, ymax = con.execute(
        "SELECT min(ST_XMin(geom)), min(ST_YMin(geom)), max(ST_XMax(geom)), max(ST_YMax(geom)) "
        "FROM _chips"
    ).fetchone()
    names = ", ".join(
        f"nullif({src}, '{UNNAMED}') AS {out}" for out, src in SOURCE_COLUMNS.items()
    )
    m = BBOX_MARGIN
    con.execute(f"""
        CREATE TEMP TABLE _units AS
        SELECT {names}, gaul2_code, geometry AS geom
        FROM read_parquet('{sql_path(source)}')
        WHERE geometry_bbox.xmax >= {xmin - m} AND geometry_bbox.xmin <= {xmax + m}
          AND geometry_bbox.ymax >= {ymin - m} AND geometry_bbox.ymin <= {ymax + m}""")


def _match(con: duckdb.DuckDBPyConnection) -> None:
    """Table ``_named``: per chip id, its unit's names and how it was matched."""
    cols = ", ".join(f"u.{c}" for c in OUTPUT_COLUMNS)
    con.execute(f"""
        CREATE TEMP TABLE _named AS
        SELECT c.id, {cols}, 'centre' AS how
        FROM _chips c JOIN _units u ON ST_Contains(u.geom, c.centre)
        QUALIFY row_number() OVER (PARTITION BY c.id ORDER BY u.gaul2_code) = 1""")
    con.execute(f"""
        INSERT INTO _named
        SELECT c.id, {cols}, 'overlap'
        FROM _chips c JOIN _units u ON ST_Intersects(u.geom, c.geom)
        WHERE c.id NOT IN (SELECT id FROM _named)
        QUALIFY row_number() OVER (
            PARTITION BY c.id ORDER BY ST_Area(ST_Intersection(u.geom, c.geom)) DESC,
            u.gaul2_code) = 1""")


def name_chips(
    con: duckdb.DuckDBPyConnection,
    chips_query: str,
    *,
    geom_col: str = "geometry",
    id_col: str = "id",
    source: str = GAUL_SOURCE,
) -> None:
    """Fill table ``_named`` (id, ``OUTPUT_COLUMNS``, how) for the chips ``chips_query`` selects.

    The connection needs the spatial extension, and S3 access to ``source`` when it is
    remote. Chips must be in lon/lat.
    """
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE _chips AS
        SELECT "{id_col}" AS id, "{geom_col}" AS geom, ST_Centroid("{geom_col}") AS centre
        FROM ({chips_query})""")
    for table in ("_units", "_named"):
        con.execute(f"DROP TABLE IF EXISTS {table}")
    if con.execute("SELECT count(*) FROM _chips").fetchone()[0] == 0:
        con.execute(
            "CREATE TEMP TABLE _named AS SELECT NULL::VARCHAR AS id, "
            + ", ".join(f"NULL::VARCHAR AS {c}" for c in OUTPUT_COLUMNS)
            + ", NULL::VARCHAR AS how WHERE false"
        )
        return
    _load_units(con, source)
    _match(con)


def connect(source: str = GAUL_SOURCE) -> duckdb.DuckDBPyConnection:
    """A connection ready for :func:`name_chips` against ``source``."""
    con = duckdb.connect(":memory:")
    con.execute("SET enable_progress_bar = false")
    ensure_spatial_loaded(con)
    if source.startswith("s3://"):
        configure_source_coop_s3(con)
    return con


def add_admin_names(
    chips_file: Path | str,
    *,
    source: str = GAUL_SOURCE,
    chips_id_col: str = "id",
) -> AdminNamesResult:
    """Append the admin name columns to the chips GeoParquet, in place.

    Args:
        chips_file: Chips GeoParquet in EPSG:4326, rewritten in place.
        source: GAUL level-2 GeoParquet (path, URL or glob) with ``geometry_bbox``.
        chips_id_col: Chip id column.
    """
    chips_path = Path(chips_file).resolve()
    geom_col = _check_chips(chips_path, chips_id_col)
    con = connect(source)
    try:
        base = select_excluding(chips_path, OUTPUT_COLUMNS)
        con.execute(f"CREATE TABLE chips_table AS {base}")
        name_chips(con, "SELECT * FROM chips_table", geom_col=geom_col,
                   id_col=chips_id_col, source=source)
        cols = ", ".join(f"n.{c}" for c in OUTPUT_COLUMNS)
        write_chips(chips_path, con, f"""
            SELECT t.*, {cols} FROM chips_table t
            LEFT JOIN _named n ON t."{chips_id_col}" = n.id""")
        result = _result(con)
    finally:
        con.close()
    logger.info(admin_names_summary(result))
    return result


def _result(con: duckdb.DuckDBPyConnection) -> AdminNamesResult:
    total = con.execute("SELECT count(*) FROM _chips").fetchone()[0]
    how = dict(con.execute("SELECT how, count(*) FROM _named GROUP BY how").fetchall())
    countries: Sequence[str] = [
        c for (c,) in con.execute(
            "SELECT DISTINCT admin0_name FROM _named WHERE admin0_name IS NOT NULL ORDER BY 1"
        ).fetchall()
    ]
    return AdminNamesResult(total, how.get("centre", 0), how.get("overlap", 0), tuple(countries))
