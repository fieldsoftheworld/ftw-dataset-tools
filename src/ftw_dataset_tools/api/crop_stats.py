"""Per-chip crop composition from the fiboa HCAT extension.

For every chip, the area of each HCAT crop code inside the chip is summed over
the field polygons (the same, possibly class-filtered, polygons the masks are
burned from). The dominant crop and the top-N composition are written back into
the chips GeoParquet so items and styles can use them without reopening the
fields.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api.field_stats import CHIP_ID_COLUMN, detect_bbox_column
from ftw_dataset_tools.api.geo import (
    detect_geometry_column,
    ensure_spatial_loaded,
    finalize_temp_file,
    sql_path,
    write_geoparquet,
)

if TYPE_CHECKING:
    from collections.abc import Callable

CODE_COLUMN = "hcat:code"
NAME_COLUMNS = ("hcat:name_en", "hcat:name")
OUTPUT_COLUMNS = ("hcat_dominant_code", "hcat_dominant_name_en", "hcat_dominant_pct", "hcat_top")

# Chips per intersection window. The per-field intersections for one window are
# held together, so this caps the geometry resident at any moment; the ranked rows
# they collapse into carry no geometry, so accumulating them costs almost nothing.
CHIP_BATCH_SIZE = 2000

NO_CODE_COLUMN_REASON = "fields carry no hcat:code column"
NO_NUMERIC_CODES_REASON = "hcat:code has no numeric values"


@dataclass
class CropStatsResult:
    """What the composition step did."""

    chips_total: int
    chips_with_crops: int
    distinct_codes: int
    skipped: bool
    reason: str | None = None


def _write_chips(chips_path: Path, con: duckdb.DuckDBPyConnection, query: str) -> None:
    """Write ``query`` over the chips GeoParquet through a temp file and a rename.

    The chips file is both the input and the output of this module, and it is the
    only copy: a partial write would leave a truncated file that a later resume from
    the splits stage cannot read. The temp file is a sibling of the target so the
    rename stays on one filesystem.

    The write is ordered by chip id. ``assign_splits`` maps a shuffled label array
    onto the rows by position, so any writer that leaves row order to the engine
    breaks split reproducibility at a fixed seed. ``add_field_stats`` orders its own
    write, but this module rewrites the same file afterwards through a LEFT JOIN,
    which does not preserve the probe side's order - so the ordering has to be
    re-applied here rather than inherited. Ordering inside this helper rather than at
    the call sites keeps a future writer from silently dropping it.
    """
    columns = [row[0] for row in con.execute(f"DESCRIBE {query}").fetchall()]
    if CHIP_ID_COLUMN in columns:
        query = f'SELECT * FROM ({query}) ORDER BY "{CHIP_ID_COLUMN}"'
    tmp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".parquet", delete=False, dir=chips_path.parent
        ) as tmp:
            tmp_path = Path(tmp.name)
        write_geoparquet(tmp_path, conn=con, query=query)
        finalize_temp_file(tmp_path, chips_path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _columns(path: Path) -> list[str]:
    con = duckdb.connect(":memory:")
    try:
        return [
            row[0]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{sql_path(path)}')"
            ).fetchall()
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


def _select_without_crop_stats(chips_path: Path) -> str:
    """SELECT over the chips file with any previous composition columns removed."""
    existing = [c for c in _columns(chips_path) if c in OUTPUT_COLUMNS]
    drop = f"EXCLUDE ({', '.join(existing)})" if existing else ""
    return f"SELECT * {drop} FROM read_parquet('{sql_path(chips_path)}')"


def drop_crop_stats(chips_file: Path | str) -> bool:
    """Remove the crop composition columns from a chips GeoParquet, in place.

    Returns True when columns were dropped. Used when the step is configured off, so
    a rerun never republishes the previous run's composition.
    """
    chips_path = Path(chips_file).resolve()
    if not any(col in OUTPUT_COLUMNS for col in _columns(chips_path)):
        return False

    con = duckdb.connect(":memory:")
    ensure_spatial_loaded(con)
    try:
        con.execute(f"CREATE TABLE chips_table AS {_select_without_crop_stats(chips_path)}")
        _write_chips(chips_path, con, "SELECT * FROM chips_table")
    finally:
        con.close()
    return True


def _join_condition(
    chip_geom: str,
    field_geom: str,
    chips_bbox_col: str | None,
    fields_bbox_col: str | None,
) -> str:
    """Join predicate, bbox-prefiltered when both files carry a bbox struct."""
    if not (chips_bbox_col and fields_bbox_col):
        return f"ST_Intersects({chip_geom}, {field_geom})"
    return f"""
        g."{chips_bbox_col}".xmin <= f."{fields_bbox_col}".xmax
        AND g."{chips_bbox_col}".xmax >= f."{fields_bbox_col}".xmin
        AND g."{chips_bbox_col}".ymin <= f."{fields_bbox_col}".ymax
        AND g."{chips_bbox_col}".ymax >= f."{fields_bbox_col}".ymin
        AND ST_Intersects({chip_geom}, {field_geom})
    """


def build_ranked_query(
    chips_table: str,
    fields_table: str,
    *,
    chips_geom_col: str,
    fields_geom_col: str,
    chips_id_col: str,
    code_col: str,
    name_col: str | None,
    chips_bbox_col: str | None = None,
    fields_bbox_col: str | None = None,
) -> str:
    """SQL ranking every (chip, HCAT code) pair by its share of the chip's field area.

    ``chips_table`` is any table expression, so callers can pass a rowid window over
    the chips rather than the whole table. Areas come from ``ST_Union_Agg`` of the
    per-field intersections, so overlapping or duplicated field rows are not double
    counted, matching ``field_coverage_pct``.
    ``pct`` is a share of *all* field area intersecting the chip, coded or not, so it
    sums below 100 when some fields carry no HCAT code.
    """
    name_expr = f'f."{name_col}"' if name_col else "CAST(NULL AS VARCHAR)"
    chip_geom = f'ST_MakeValid(g."{chips_geom_col}")'
    field_geom = f'ST_MakeValid(f."{fields_geom_col}")'
    join_condition = _join_condition(chip_geom, field_geom, chips_bbox_col, fields_bbox_col)
    return f"""
    WITH pairs AS MATERIALIZED (
        SELECT g."{chips_id_col}" AS chip_id,
               TRY_CAST(f."{code_col}" AS BIGINT) AS code,
               {name_expr} AS name_en,
               ST_Intersection({chip_geom}, {field_geom}) AS part
        FROM {chips_table} g
        JOIN {fields_table} f ON {join_condition}
    ),
    totals AS (
        SELECT chip_id, ST_Area(ST_Union_Agg(part)) AS total_area
        FROM pairs
        GROUP BY chip_id
    ),
    coded AS (
        SELECT chip_id,
               code,
               MIN(name_en) AS name_en,
               ST_Area(ST_Union_Agg(part)) AS area
        FROM pairs
        WHERE code IS NOT NULL
        GROUP BY chip_id, code
    )
    SELECT c.chip_id,
           c.code,
           c.name_en,
           c.area,
           100.0 * c.area / t.total_area AS pct,
           ROW_NUMBER() OVER (PARTITION BY c.chip_id ORDER BY c.area DESC, c.code) AS rn
    FROM coded c
    JOIN totals t ON c.chip_id = t.chip_id
    WHERE c.area > 0 AND t.total_area > 0
    """


def build_composition_query(
    chips_table: str,
    ranked_table: str,
    *,
    chips_id_col: str,
    top_n: int,
) -> str:
    """SQL joining the ranked codes back onto every chip as the four output columns."""
    return f"""
    WITH composition AS (
        SELECT chip_id,
               MAX(CASE WHEN rn = 1 THEN code END) AS hcat_dominant_code,
               MAX(CASE WHEN rn = 1 THEN name_en END) AS hcat_dominant_name_en,
               MAX(CASE WHEN rn = 1 THEN ROUND(pct, 2) END) AS hcat_dominant_pct,
               LIST({{'code': code, 'name_en': name_en, 'pct': ROUND(pct, 2)}}
                    ORDER BY area DESC, code) FILTER (WHERE rn <= {top_n}) AS hcat_top
        FROM {ranked_table}
        GROUP BY chip_id
    )
    SELECT g.*, c.hcat_dominant_code, c.hcat_dominant_name_en, c.hcat_dominant_pct, c.hcat_top
    FROM {chips_table} g
    LEFT JOIN composition c ON g."{chips_id_col}" = c.chip_id
    """


def crop_stats_summary(result: CropStatsResult | None) -> str:
    """The one-line run summary for the crop composition step.

    ``None`` means the step was configured off. Callers ask for a summary only when
    the chips stage actually ran, so a resume past chips reports nothing at all.
    """
    if result is None:
        return "Crop composition: disabled"
    if result.skipped:
        return f"Crop composition: skipped ({result.reason})"
    return (
        f"Crop composition: {result.chips_with_crops}/{result.chips_total} chips, "
        f"{result.distinct_codes} HCAT codes"
    )


def _code_value_counts(
    con: duckdb.DuckDBPyConnection, fields_table: str, code_col: str
) -> tuple[int, int]:
    """(non-null HCAT codes, codes that cast to BIGINT) in the fields table."""
    return con.execute(
        f'SELECT count(f."{code_col}"), count(TRY_CAST(f."{code_col}" AS BIGINT)) '
        f"FROM {fields_table} f"
    ).fetchone()


def _fill_ranked(
    con: duckdb.DuckDBPyConnection,
    chips_path: Path,
    fields_path: Path,
    *,
    chips_id_col: str,
    code_col: str,
    name_col: str | None,
    batch_size: int,
) -> None:
    """Build the ``ranked`` table one window of chip rowids at a time.

    Ranking every chip in one statement holds an intersection geometry for every
    (chip, field) pair in the country at once, which is what runs a large country out
    of memory. Windowing by ``chips_table.rowid`` caps that at ``batch_size`` chips'
    worth of pairs; the ranked rows are scalars, so the accumulated table is cheap.
    Chunking changes nothing about the result: every chip's codes are ranked against
    that chip's own field area, so no window depends on any other.
    """
    chips_geom = detect_geometry_column(chips_path) or "geometry"
    fields_geom = detect_geometry_column(fields_path) or "geometry"
    chips_bbox = detect_bbox_column(con, chips_path, chips_geom)
    fields_bbox = detect_bbox_column(con, fields_path, fields_geom)

    total = con.execute("SELECT count(*) FROM chips_table").fetchone()[0]
    # An empty chips file still needs one pass, to create ``ranked`` with its schema.
    for offset in range(0, max(total, 1), batch_size):
        window = (
            f"(SELECT * FROM chips_table WHERE rowid >= {offset} AND rowid < {offset + batch_size})"
        )
        query = build_ranked_query(
            window,
            "fields_table",
            chips_geom_col=chips_geom,
            fields_geom_col=fields_geom,
            chips_id_col=chips_id_col,
            code_col=code_col,
            name_col=name_col,
            chips_bbox_col=chips_bbox,
            fields_bbox_col=fields_bbox,
        )
        if offset == 0:
            con.execute(f"CREATE TABLE ranked AS {query}")
        else:
            con.execute(f"INSERT INTO ranked {query}")


def _compose(
    con: duckdb.DuckDBPyConnection,
    chips_path: Path,
    fields_path: Path,
    *,
    chips_id_col: str,
    code_col: str,
    name_col: str | None,
    top_n: int,
    batch_size: int,
) -> tuple[int, int, int]:
    """Build the ``composed`` table; returns (chips, chips with crops, distinct codes)."""
    _fill_ranked(
        con,
        chips_path,
        fields_path,
        chips_id_col=chips_id_col,
        code_col=code_col,
        name_col=name_col,
        batch_size=batch_size,
    )
    composition_query = build_composition_query(
        "chips_table", "ranked", chips_id_col=chips_id_col, top_n=top_n
    )
    con.execute(f"CREATE TABLE composed AS {composition_query}")

    total, with_crops = con.execute(
        "SELECT count(*), count(hcat_dominant_code) FROM composed"
    ).fetchone()
    (distinct,) = con.execute("SELECT count(DISTINCT code) FROM ranked").fetchone()
    return total, with_crops, distinct


def add_crop_stats(
    chips_file: Path | str,
    fields_file: Path | str,
    *,
    chips_id_col: str = "id",
    top_n: int = 5,
    chip_batch_size: int = CHIP_BATCH_SIZE,
    on_progress: Callable[[str], None] | None = None,
) -> CropStatsResult:
    """Append the crop composition columns to the chips GeoParquet, in place.

    ``chip_batch_size`` is how many chips are intersected against the fields at a
    time; it bounds peak memory and does not affect the statistics.
    """
    batch_size = max(1, chip_batch_size)
    chips_path = Path(chips_file).resolve()
    fields_path = Path(fields_file).resolve()

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    def skip(chips_total: int, reason: str) -> CropStatsResult:
        log(f"Note: {reason}; skipping crop composition")
        return CropStatsResult(chips_total, 0, 0, skipped=True, reason=reason)

    detected = detect_hcat_columns(fields_path)
    if detected is None:
        return skip(_chips_count(chips_path), NO_CODE_COLUMN_REASON)
    code_col, name_col = detected

    con = duckdb.connect(":memory:")
    ensure_spatial_loaded(con)
    try:
        con.execute(f"CREATE TABLE chips_table AS {_select_without_crop_stats(chips_path)}")
        con.execute(
            f"CREATE TABLE fields_table AS SELECT * FROM read_parquet('{sql_path(fields_path)}')"
        )
        coded, castable = _code_value_counts(con, "fields_table", code_col)
        if coded > 0 and castable == 0:
            chips_total = con.execute("SELECT count(*) FROM chips_table").fetchone()[0]
            return skip(chips_total, NO_NUMERIC_CODES_REASON)

        total, with_crops, distinct = _compose(
            con,
            chips_path,
            fields_path,
            chips_id_col=chips_id_col,
            code_col=code_col,
            name_col=name_col,
            top_n=top_n,
            batch_size=batch_size,
        )
        _write_chips(chips_path, con, "SELECT * FROM composed")
    finally:
        con.close()

    log(
        f"Crop composition: {with_crops:,} of {total:,} chips have HCAT crops "
        f"({distinct} HCAT codes)"
    )
    return CropStatsResult(total, with_crops, distinct, skipped=False)


def _chips_count(chips_path: Path) -> int:
    con = duckdb.connect(":memory:")
    try:
        return con.execute(
            f"SELECT count(*) FROM read_parquet('{sql_path(chips_path)}')"
        ).fetchone()[0]
    finally:
        con.close()
