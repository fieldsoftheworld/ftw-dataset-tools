"""Rewrite the chips GeoParquet in place, for steps that add per-chip columns.

The chips-stage enrichment steps (crop composition, land cover) read the chips file,
join their columns on, and write the result back over the same file. The helpers
here keep that rewrite safe and reproducible for every step that does it.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import duckdb

from ftw_dataset_tools.api.field_stats import CHIP_ID_COLUMN
from ftw_dataset_tools.api.fs import create_temp_file, finalize_temp_file
from ftw_dataset_tools.api.geo import ensure_spatial_loaded, sql_path, write_geoparquet

if TYPE_CHECKING:
    from collections.abc import Iterable


def write_chips(chips_path: Path, con: duckdb.DuckDBPyConnection, query: str) -> None:
    """Write ``query`` over the chips GeoParquet through a temp file and a rename.

    The chips file is both the input and the output of the enrichment steps, and it
    is the only copy: a partial write would leave a truncated file that a later
    resume from the splits stage cannot read. The temp file is a sibling of the
    target so the rename stays on one filesystem.

    The write is ordered by chip id. ``assign_splits`` maps a shuffled label array
    onto the rows by position, so any writer that leaves row order to the engine
    breaks split reproducibility at a fixed seed. ``add_field_stats`` orders its own
    write, but the enrichment steps rewrite the same file afterwards through a LEFT
    JOIN, which does not preserve the probe side's order - so the ordering has to be
    re-applied here rather than inherited. Ordering inside this helper rather than at
    the call sites keeps a future writer from silently dropping it.
    """
    columns = [row[0] for row in con.execute(f"DESCRIBE {query}").fetchall()]
    if CHIP_ID_COLUMN in columns:
        query = f'SELECT * FROM ({query}) ORDER BY "{CHIP_ID_COLUMN}"'
    tmp_path: Path | None = None
    try:
        tmp_path = create_temp_file(chips_path, suffix=".parquet")
        write_geoparquet(tmp_path, conn=con, query=query)
        finalize_temp_file(tmp_path, chips_path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def parquet_columns(path: Path) -> list[str]:
    """Column names of a Parquet file, in file order."""
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


def select_excluding(chips_path: Path, columns: Iterable[str]) -> str:
    """SELECT over the chips file with any of ``columns`` that it carries removed."""
    drop_set = set(columns)
    existing = [c for c in parquet_columns(chips_path) if c in drop_set]
    drop = f"EXCLUDE ({', '.join(existing)})" if existing else ""
    return f"SELECT * {drop} FROM read_parquet('{sql_path(chips_path)}')"


def drop_columns(chips_file: Path | str, columns: Iterable[str]) -> bool:
    """Remove ``columns`` from a chips GeoParquet, in place.

    Returns True when columns were dropped, False (leaving the file untouched) when
    the file carries none of them.
    """
    chips_path = Path(chips_file).resolve()
    columns = tuple(columns)
    if not any(col in columns for col in parquet_columns(chips_path)):
        return False

    con = duckdb.connect(":memory:")
    ensure_spatial_loaded(con)
    try:
        con.execute(f"CREATE TABLE chips_table AS {select_excluding(chips_path, columns)}")
        write_chips(chips_path, con, "SELECT * FROM chips_table")
    finally:
        con.close()
    return True
