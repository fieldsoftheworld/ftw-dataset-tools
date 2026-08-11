"""Data-plane helpers for applying a class filter to a fields file.

The :class:`~ftw_dataset_tools.api.config.ClassFilter` schema/validation lives in
``config.py``. This module holds the DuckDB/GeoParquet operations that need the
actual data: reading the distinct class values and writing the filtered fields
file (only the ``include`` classes) that downstream stages consume.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api.config import ClassFilter, ClassFilterError
from ftw_dataset_tools.api.geo import write_geoparquet

if TYPE_CHECKING:
    from pathlib import Path


def _safe_column(column: str) -> str:
    """Return a double-quoted identifier, rejecting embedded quotes."""
    if '"' in column:
        raise ClassFilterError(f"Invalid class filter column name: {column!r}")
    return f'"{column}"'


def resolve_column(fields_file: str | Path, class_filter: ClassFilter) -> str:
    """Return the filter's class column that actually exists in ``fields_file``.

    Tries the filter's primary column then its aliases, so one filter can be used
    across datasets that name the column differently (e.g. crop_code / crop:code).

    Raises:
        ClassFilterError: If none of the candidate columns are present.
    """
    conn = duckdb.connect(":memory:")
    try:
        columns = [
            row[0] for row in conn.execute(f"DESCRIBE SELECT * FROM '{fields_file}'").fetchall()
        ]
    finally:
        conn.close()
    return class_filter.resolve_column(columns)


def get_distinct_classes(fields_file: str | Path, column: str) -> set[str | None]:
    """Return the distinct values of ``column`` in ``fields_file`` (as strings).

    Values are cast to VARCHAR so numeric crop codes and string labels are
    compared uniformly. A NULL in the column is returned as ``None``.

    Raises:
        ClassFilterError: If the column does not exist in the fields file.
    """
    quoted = _safe_column(column)
    conn = duckdb.connect(":memory:")
    try:
        columns = [
            row[0] for row in conn.execute(f"DESCRIBE SELECT * FROM '{fields_file}'").fetchall()
        ]
        if column not in columns:
            raise ClassFilterError(
                f"Class filter column '{column}' not found in fields file. "
                f"Available columns: {sorted(columns)}"
            )
        rows = conn.execute(
            f"SELECT DISTINCT CAST({quoted} AS VARCHAR) FROM '{fields_file}'"
        ).fetchall()
    finally:
        conn.close()
    return {row[0] for row in rows}


def write_filtered_fields(
    fields_file: str | Path,
    output_path: str | Path,
    class_filter: ClassFilter,
    column: str | None = None,
) -> Path:
    """Write a fields file containing only the filter's ``include`` classes.

    The geometry column and GeoParquet metadata are preserved via
    :func:`~ftw_dataset_tools.api.geo.write_geoparquet`. If ``include`` is empty,
    the result is an empty field set (everything is background).

    Args:
        column: Class column to filter on. Defaults to the filter's primary
            column; pass a resolved name (see :func:`resolve_column`) when the
            dataset uses a fallback name.
    """
    quoted = _safe_column(column or class_filter.column)
    if class_filter.include:
        # Escape single quotes for safe string literals; matching is on VARCHAR.
        literals = ", ".join("'" + value.replace("'", "''") + "'" for value in class_filter.include)
        where = f"WHERE CAST({quoted} AS VARCHAR) IN ({literals})"
    else:
        where = "WHERE FALSE"

    query = f"SELECT * FROM '{fields_file}' {where}"
    conn = duckdb.connect(":memory:")
    try:
        return write_geoparquet(output_path, conn=conn, query=query)
    finally:
        conn.close()
