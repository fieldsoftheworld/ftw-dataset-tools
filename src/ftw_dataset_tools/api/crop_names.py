"""Give a built benchmark's field files the source's original crop names.

A benchmark's fields file holds every column of the harmonized edition it was built
from. A harmonized edition rebuilt after fiboa-cli learned to name crop codes from its
code tables (fiboa/cli#352) has ``crop:name`` (and sometimes ``crop:name_en``) that a
benchmark built earlier lacks. ``add_crop_names`` joins those columns in by field
``id``, and refuses when a field is missing from the edition or its ``crop:code``
differs there, so no field ever takes another field's name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

#: The columns taken from the harmonized edition, in the order they are written.
NAME_COLUMNS: tuple[str, ...] = ("crop:name", "crop:name_en")

#: The columns the join needs in both files.
KEY_COLUMNS: tuple[str, ...] = ("id", "crop:code")

#: How many mismatched field ids an error message lists.
_EXAMPLES = 5


class CropNameError(Exception):
    """The crop names cannot be joined onto a fields file without risking a wrong name."""

    def __init__(self, path: Path, message: str) -> None:
        super().__init__(message)
        self.path = path


@dataclass(frozen=True)
class CropNameResult:
    """What ``add_crop_names`` did to one fields file."""

    rows: int
    named: int
    columns: tuple[str, ...]
    changed: bool


def _columns(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return pq.read_schema(path).names


def _check_columns(fields_file: Path, source_file: Path) -> tuple[str, ...]:
    """The name columns the source offers; raises when a needed column is missing."""
    source_columns = _columns(source_file)
    names = tuple(c for c in NAME_COLUMNS if c in source_columns)
    if not names:
        raise CropNameError(source_file, f"{source_file} has no crop:name column")
    for path, columns in ((fields_file, _columns(fields_file)), (source_file, source_columns)):
        missing = [c for c in KEY_COLUMNS if c not in columns]
        if missing:
            raise CropNameError(path, f"{path} has no {', '.join(missing)} column")
    return names


def _joined_names(fields_file: Path, source_file: Path, names: tuple[str, ...]):
    """The name columns for each row of ``fields_file``, in its row order, checked."""
    import duckdb

    con = duckdb.connect()
    duplicates = con.execute(
        "SELECT count(*) - count(DISTINCT id) FROM read_parquet(?)", [str(source_file)]
    ).fetchone()[0]
    if duplicates:
        raise CropNameError(source_file, f"{source_file} repeats {duplicates} field id(s)")
    selected = ", ".join(f's."{c}"' for c in names)
    con.execute(
        f"""
        CREATE TEMP TABLE joined AS
        SELECT f.file_row_number AS row_number, f.id AS id,
               s.id IS NULL AS missing,
               s.id IS NOT NULL
                 AND CAST(f."crop:code" AS VARCHAR) IS DISTINCT FROM CAST(s."crop:code" AS VARCHAR)
                 AS differs,
               {selected}
        FROM read_parquet(?, file_row_number = true) f
        LEFT JOIN read_parquet(?) s ON f.id = s.id
        """,
        [str(fields_file), str(source_file)],
    )
    for flag, what in (("missing", "not in"), ("differs", "with another crop:code in")):
        count, examples = con.execute(
            f"SELECT count(*), list(id ORDER BY row_number)[1:{_EXAMPLES}] "
            f"FROM joined WHERE {flag}"
        ).fetchone()
        if count:
            raise CropNameError(
                fields_file,
                f"{count} field(s) of {fields_file} {what} {source_file}, e.g. {examples}",
            )
    quoted = ", ".join(f'"{c}"' for c in names)
    return con.execute(f"SELECT {quoted} FROM joined ORDER BY row_number").fetch_arrow_table()


def _with_names(table, names_table, names: tuple[str, ...]):
    """``table`` with ``names`` replaced by the joined columns, right after crop:code."""
    for name in names:
        if name in table.column_names:
            table = table.drop_columns([name])
    at = table.column_names.index("crop:code") + 1
    for offset, name in enumerate(names):
        table = table.add_column(at + offset, name, names_table.column(name))
    return table


def _unchanged(old, new, names: tuple[str, ...]) -> bool:
    return all(
        name in old.column_names and old.column(name).equals(new.column(name)) for name in names
    )


def add_crop_names(
    fields_file: Path | str, source_file: Path | str, output_file: Path | str | None = None
) -> CropNameResult:
    """Join the crop name columns of ``source_file`` onto ``fields_file`` by field id.

    Args:
        fields_file: A benchmark's fields (or fields_filtered) GeoParquet.
        source_file: The harmonized edition it was built from, with crop names.
        output_file: Where to write; ``fields_file`` itself when None. Nothing is
            written when the names are already there and equal.

    Raises:
        CropNameError: The source has no names, a key column is missing, the source
            repeats a field id, or a field is missing from the source or has another
            crop:code there.
    """
    import geoparquet_io as gpio
    import pyarrow.compute as pc

    fields_file, source_file = Path(fields_file), Path(source_file)
    output = Path(output_file) if output_file is not None else fields_file
    names = _check_columns(fields_file, source_file)
    names_table = _joined_names(fields_file, source_file, names)

    source = gpio.read(str(fields_file))
    table = source.to_arrow()
    new = _with_names(table, names_table, names)
    changed = not _unchanged(table, new, names)
    if changed or output != fields_file:
        tmp = output.with_name(f".{output.name}.tmp")
        gpio.Table(new, geometry_column=source.geometry_column).write(str(tmp), overwrite=True)
        os.replace(tmp, output)
    named = new.num_rows - pc.sum(pc.is_null(new.column("crop:name"))).as_py()
    return CropNameResult(rows=new.num_rows, named=named, columns=names, changed=changed)
