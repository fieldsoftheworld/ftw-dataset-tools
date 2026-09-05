"""Per-chip crop composition from the fiboa HCAT extension.

For every chip, the area of each HCAT crop code inside the chip is summed over
the field polygons (the same, possibly class-filtered, polygons the masks are
burned from). The dominant crop and the top-N composition are written back into
the chips GeoParquet so items and styles can use them without reopening the
fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api.geo import (
    detect_geometry_column,
    ensure_spatial_loaded,
    write_geoparquet,
)

if TYPE_CHECKING:
    from collections.abc import Callable

CODE_COLUMN = "hcat:code"
NAME_COLUMNS = ("hcat:name_en", "hcat:name")
OUTPUT_COLUMNS = ("hcat_dominant_code", "hcat_dominant_name_en", "hcat_dominant_pct", "hcat_top")


@dataclass
class CropStatsResult:
    """What the composition step did."""

    chips_total: int
    chips_with_crops: int
    distinct_codes: int
    skipped: bool
    reason: str | None = None


def _columns(path: Path) -> list[str]:
    con = duckdb.connect(":memory:")
    try:
        return [
            row[0]
            for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{path}')").fetchall()
        ]
    finally:
        con.close()


def detect_hcat_columns(fields_file: Path | str) -> tuple[str, str | None] | None:
    """The HCAT code column and the best available name column, or None."""
    columns = _columns(Path(fields_file))
    if CODE_COLUMN not in columns:
        return None
    name_col = next((c for c in NAME_COLUMNS if c in columns), None)
    return CODE_COLUMN, name_col


def build_crop_stats_query(
    chips_table: str,
    fields_table: str,
    *,
    chips_geom_col: str,
    fields_geom_col: str,
    chips_id_col: str,
    code_col: str,
    name_col: str | None,
    top_n: int,
) -> str:
    """SQL producing one row per chip with the four composition columns."""
    name_expr = f'ANY_VALUE(f."{name_col}")' if name_col else "CAST(NULL AS VARCHAR)"
    chip_geom = f'ST_MakeValid(g."{chips_geom_col}")'
    field_geom = f'ST_MakeValid(f."{fields_geom_col}")'
    return f"""
    WITH parts AS (
        SELECT g."{chips_id_col}" AS chip_id,
               CAST(f."{code_col}" AS BIGINT) AS code,
               {name_expr} AS name_en,
               SUM(ST_Area(ST_Intersection({chip_geom}, {field_geom}))) AS area
        FROM {chips_table} g
        JOIN {fields_table} f ON ST_Intersects({chip_geom}, {field_geom})
        WHERE f."{code_col}" IS NOT NULL
        GROUP BY g."{chips_id_col}", f."{code_col}"
    ),
    ranked AS (
        SELECT *,
               100.0 * area / SUM(area) OVER (PARTITION BY chip_id) AS pct,
               ROW_NUMBER() OVER (PARTITION BY chip_id ORDER BY area DESC, code) AS rn
        FROM parts
        WHERE area > 0
    ),
    composition AS (
        SELECT chip_id,
               MAX(CASE WHEN rn = 1 THEN code END) AS hcat_dominant_code,
               MAX(CASE WHEN rn = 1 THEN name_en END) AS hcat_dominant_name_en,
               MAX(CASE WHEN rn = 1 THEN ROUND(pct, 2) END) AS hcat_dominant_pct,
               LIST({{'code': code, 'name_en': name_en, 'pct': ROUND(pct, 2)}} ORDER BY area DESC, code)
                   FILTER (WHERE rn <= {top_n}) AS hcat_top
        FROM ranked
        GROUP BY chip_id
    )
    SELECT g.*, c.hcat_dominant_code, c.hcat_dominant_name_en, c.hcat_dominant_pct, c.hcat_top
    FROM {chips_table} g
    LEFT JOIN composition c ON g."{chips_id_col}" = c.chip_id
    """


def add_crop_stats(
    chips_file: Path | str,
    fields_file: Path | str,
    *,
    chips_id_col: str = "id",
    top_n: int = 5,
    on_progress: Callable[[str], None] | None = None,
) -> CropStatsResult:
    """Append the crop composition columns to the chips GeoParquet, in place."""
    chips_path = Path(chips_file).resolve()
    fields_path = Path(fields_file).resolve()

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    detected = detect_hcat_columns(fields_path)
    if detected is None:
        reason = "fields carry no hcat:code column"
        log(f"Note: {reason}; skipping crop composition")
        con = duckdb.connect(":memory:")
        try:
            total = con.execute(f"SELECT count(*) FROM read_parquet('{chips_path}')").fetchone()[0]
        finally:
            con.close()
        return CropStatsResult(total, 0, 0, skipped=True, reason=reason)
    code_col, name_col = detected

    chips_geom = detect_geometry_column(chips_path) or "geometry"
    fields_geom = detect_geometry_column(fields_path) or "geometry"
    existing = [c for c in _columns(chips_path) if c in OUTPUT_COLUMNS]

    con = duckdb.connect(":memory:")
    ensure_spatial_loaded(con)
    try:
        drop = f"EXCLUDE ({', '.join(existing)})" if existing else ""
        con.execute(
            f"CREATE TABLE chips_table AS SELECT * {drop} FROM read_parquet('{chips_path}')"
        )
        con.execute(f"CREATE TABLE fields_table AS SELECT * FROM read_parquet('{fields_path}')")
        query = build_crop_stats_query(
            "chips_table",
            "fields_table",
            chips_geom_col=chips_geom,
            fields_geom_col=fields_geom,
            chips_id_col=chips_id_col,
            code_col=code_col,
            name_col=name_col,
            top_n=top_n,
        )
        con.execute(f"CREATE TABLE composed AS {query}")
        total, with_crops = con.execute(
            "SELECT count(*), count(hcat_dominant_code) FROM composed"
        ).fetchone()
        (distinct,) = con.execute(
            "SELECT count(DISTINCT entry.code) "
            "FROM (SELECT unnest(hcat_top) AS entry FROM composed)"
        ).fetchone()
        write_geoparquet(chips_path, conn=con, query="SELECT * FROM composed")
    finally:
        con.close()

    log(
        f"Crop composition: {with_crops:,} of {total:,} chips have HCAT crops ({distinct} dominant codes)"
    )
    return CropStatsResult(total, with_crops, distinct, skipped=False)
