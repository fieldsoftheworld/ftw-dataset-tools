"""Quick inspection/summarization of a fields GeoParquet file.

Produces a structured :class:`FieldSummary` (row/column counts, per-column
value counts and stats, and a geometry/CRS summary) for terminal or JSON
display. Handy for understanding a new dataset and for discovering the class
column and values to build a class filter (see ``ftwd run`` / ``--class-filter``).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import duckdb

from ftw_dataset_tools.api.geo import detect_geometry_column, ensure_spatial_loaded

# A column is treated as an identifier (value counts skipped) when it is almost
# entirely unique and has many distinct values.
_IDENTIFIER_MIN_DISTINCT = 1000
_IDENTIFIER_UNIQUENESS = 0.9
# Safety cap when dumping "all" values so a huge column can't flood the output.
_VALUE_DUMP_CAP = 10_000
# Categorical columns with at most this many distinct values are flagged as
# class-filter candidates.
_CANDIDATE_MAX_DISTINCT = 500


@dataclass
class ColumnSummary:
    """Per-column summary."""

    name: str
    dtype: str
    kind: str  # categorical | numeric | temporal | geometry | identifier | other
    non_null: int
    nulls: int
    distinct: int
    value_counts: list[tuple[str, int]] | None = None
    value_counts_truncated: bool = False
    stats: dict[str, Any] | None = None


@dataclass
class GeometrySummary:
    """Geometry/CRS summary."""

    column: str
    crs_name: str | None
    crs_kind: str | None
    epsg: int | None
    geometry_types: list[tuple[str, int]] = field(default_factory=list)
    bounds: tuple[float, float, float, float] | None = None


@dataclass
class FieldSummary:
    """Full summary of a fields file."""

    path: str
    file_size_bytes: int
    num_rows: int
    num_columns: int
    columns: list[ColumnSummary]
    geometry: GeometrySummary | None = None
    class_filter_candidates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict form suitable for JSON serialization."""
        return asdict(self)


def _classify(dtype: str, geom_col: str | None, name: str) -> str:
    upper = dtype.upper()
    if name == geom_col or upper.startswith("GEOMETRY"):
        return "geometry"
    if upper.startswith(("TIMESTAMP", "DATE", "TIME")):
        return "temporal"
    numeric_prefixes = (
        "TINYINT",
        "SMALLINT",
        "INTEGER",
        "BIGINT",
        "HUGEINT",
        "UTINYINT",
        "USMALLINT",
        "UINTEGER",
        "UBIGINT",
        "FLOAT",
        "DOUBLE",
        "DECIMAL",
        "REAL",
    )
    if upper.startswith(numeric_prefixes):
        return "numeric"
    if upper.startswith(("VARCHAR", "BOOLEAN", "CHAR", "TEXT", "ENUM")):
        return "categorical"
    return "other"


def _quote(name: str) -> str:
    if '"' in name:
        raise ValueError(f"Unsupported column name: {name!r}")
    return f'"{name}"'


def _distinct_and_nulls(
    conn: duckdb.DuckDBPyConnection, path: str, names: list[str]
) -> dict[str, tuple[int, int]]:
    """Return {column: (distinct, nulls)} for the given columns in one scan."""
    if not names:
        return {}
    selects = []
    for col in names:
        q = _quote(col)
        selects.append(f"COUNT(DISTINCT {q})")
        selects.append(f"COUNT(*) - COUNT({q})")
    row = conn.execute(f"SELECT {', '.join(selects)} FROM '{path}'").fetchone()
    return {col: (row[2 * i], row[2 * i + 1]) for i, col in enumerate(names)}


def _value_counts(
    conn: duckdb.DuckDBPyConnection, path: str, col: str, limit: int
) -> list[tuple[str, int]]:
    q = _quote(col)
    rows = conn.execute(
        f"SELECT CAST({q} AS VARCHAR), COUNT(*) AS n FROM '{path}' "
        f"WHERE {q} IS NOT NULL GROUP BY 1 ORDER BY n DESC, 1 LIMIT {limit}"
    ).fetchall()
    return [(r[0], r[1]) for r in rows]


def _numeric_stats(conn: duckdb.DuckDBPyConnection, path: str, col: str) -> dict[str, Any]:
    q = _quote(col)
    minv, maxv, mean, stddev, p25, median, p75 = conn.execute(
        f"SELECT MIN({q}), MAX({q}), AVG({q}), STDDEV_SAMP({q}), "
        f"QUANTILE_CONT({q}, 0.25), MEDIAN({q}), QUANTILE_CONT({q}, 0.75) "
        f"FROM '{path}'"
    ).fetchone()
    return {
        "min": minv,
        "max": maxv,
        "mean": mean,
        "stddev": stddev,
        "p25": p25,
        "median": median,
        "p75": p75,
    }


def _temporal_range(conn: duckdb.DuckDBPyConnection, path: str, col: str) -> dict[str, Any]:
    q = _quote(col)
    # Cast to VARCHAR so timezone-aware values don't require pytz to fetch.
    minv, maxv = conn.execute(
        f"SELECT CAST(MIN({q}) AS VARCHAR), CAST(MAX({q}) AS VARCHAR) FROM '{path}'"
    ).fetchone()
    return {"min": minv, "max": maxv}


def _crs_summary(path: str, geom_col: str) -> dict[str, Any]:
    """Read CRS name/kind/EPSG from GeoParquet 'geo' metadata (best effort)."""
    result: dict[str, Any] = {"crs_name": None, "crs_kind": None, "epsg": None}
    conn = duckdb.connect(":memory:")
    try:
        row = conn.execute(
            "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'geo'", [path]
        ).fetchone()
    finally:
        conn.close()
    if not row:
        return result
    try:
        geo = json.loads(row[0])
        column_meta = geo.get("columns", {}).get(geom_col, {})
        crs = column_meta.get("crs")
    except (ValueError, AttributeError):
        return result
    if crs is None:
        # GeoParquet default when crs is omitted.
        return {"crs_name": "OGC:CRS84 (lon/lat)", "crs_kind": "geographic", "epsg": 4326}
    if isinstance(crs, dict):
        result["crs_name"] = crs.get("name")
        crs_type = crs.get("type", "")
        result["crs_kind"] = (
            "geographic"
            if "Geographic" in crs_type
            else "projected"
            if "Projected" in crs_type
            else None
        )
        identifier = crs.get("id") or {}
        if str(identifier.get("authority", "")).upper() == "EPSG":
            result["epsg"] = identifier.get("code")
    return result


def _geometry_summary(path: str, geom_col: str) -> GeometrySummary:
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    q = _quote(geom_col)
    try:
        rows = conn.execute(
            f"SELECT ST_GeometryType({q}) AS gt, COUNT(*) AS n, "
            f"MIN(ST_XMin({q})), MIN(ST_YMin({q})), MAX(ST_XMax({q})), MAX(ST_YMax({q})) "
            f"FROM '{path}' GROUP BY gt ORDER BY n DESC"
        ).fetchall()
    finally:
        conn.close()

    geometry_types = [(r[0], r[1]) for r in rows]
    bounds: tuple[float, float, float, float] | None = None
    if rows:
        bounds = (
            min(r[2] for r in rows),
            min(r[3] for r in rows),
            max(r[4] for r in rows),
            max(r[5] for r in rows),
        )
    crs = _crs_summary(path, geom_col)
    return GeometrySummary(
        column=geom_col,
        crs_name=crs["crs_name"],
        crs_kind=crs["crs_kind"],
        epsg=crs["epsg"],
        geometry_types=geometry_types,
        bounds=bounds,
    )


def summarize_fields(
    path: str | Path,
    top: int = 20,
    focus_columns: list[str] | None = None,
    include_geometry: bool = True,
) -> FieldSummary:
    """Summarize a fields GeoParquet file.

    Args:
        path: Path to the GeoParquet file.
        top: Max number of values to show per categorical column (0 = all,
            capped for safety). Ignored (shows all) for focus_columns.
        focus_columns: Columns whose full value counts should always be shown.
        include_geometry: Whether to compute the geometry/CRS summary (a full
            scan of the geometry column; skip for speed).

    Returns:
        A populated FieldSummary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Fields file not found: {file_path}")
    path_str = str(file_path)
    focus = set(focus_columns or [])

    conn = duckdb.connect(":memory:")
    try:
        schema = conn.execute(f"DESCRIBE SELECT * FROM '{path_str}'").fetchall()
        num_rows = conn.execute(f"SELECT COUNT(*) FROM '{path_str}'").fetchone()[0]
        geom_col = detect_geometry_column(path_str)

        column_kinds = [(row[0], row[1], _classify(row[1], geom_col, row[0])) for row in schema]
        non_geom = [name for name, _, kind in column_kinds if kind != "geometry"]
        dn = _distinct_and_nulls(conn, path_str, non_geom)

        columns: list[ColumnSummary] = []
        candidates: list[str] = []
        for name, dtype, kind in column_kinds:
            if kind == "geometry":
                columns.append(
                    ColumnSummary(
                        name=name, dtype=dtype, kind=kind, non_null=num_rows, nulls=0, distinct=0
                    )
                )
                continue

            distinct, nulls = dn[name]
            non_null = num_rows - nulls
            effective_kind = kind
            if (
                kind == "categorical"
                and distinct >= _IDENTIFIER_MIN_DISTINCT
                and non_null > 0
                and distinct / non_null >= _IDENTIFIER_UNIQUENESS
            ):
                effective_kind = "identifier"

            summary = ColumnSummary(
                name=name,
                dtype=dtype,
                kind=effective_kind,
                non_null=non_null,
                nulls=nulls,
                distinct=distinct,
            )

            if effective_kind == "categorical":
                limit = _VALUE_DUMP_CAP if (name in focus or top == 0) else top
                summary.value_counts = _value_counts(conn, path_str, name, limit)
                summary.value_counts_truncated = distinct > len(summary.value_counts)
                if 1 <= distinct <= _CANDIDATE_MAX_DISTINCT:
                    candidates.append(name)
            elif effective_kind == "numeric" and non_null > 0:
                summary.stats = _numeric_stats(conn, path_str, name)
            elif effective_kind == "temporal" and non_null > 0:
                summary.stats = _temporal_range(conn, path_str, name)

            columns.append(summary)
    finally:
        conn.close()

    geometry = None
    if include_geometry and geom_col:
        geometry = _geometry_summary(path_str, geom_col)

    return FieldSummary(
        path=path_str,
        file_size_bytes=file_path.stat().st_size,
        num_rows=num_rows,
        num_columns=len(schema),
        columns=columns,
        geometry=geometry,
        class_filter_candidates=candidates,
    )
