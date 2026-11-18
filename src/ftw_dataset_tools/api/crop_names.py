"""Give a built benchmark's field files the source's original crop names.

A benchmark's fields file holds every column of the harmonized edition it was built
from. A harmonized edition rebuilt after fiboa-cli learned to name crop codes from its
code tables (fiboa/cli#352) has ``crop:name`` (and sometimes ``crop:name_en``) that a
benchmark built earlier lacks. ``add_crop_names`` joins those columns in by field
``id``, and refuses when a field is missing from the edition or its ``crop:code``
differs there, so no field ever takes another field's name.

The file is rewritten one row group at a time with pyarrow, not read whole through
geoparquet-io: a fields file can hold ten million fields, and streaming keeps memory
flat and the file as it was (row groups, compression, the ``geo`` and fiboa metadata).
Only the name columns change, and the geometry is copied as is, so the ``geo``
metadata stays true.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import duckdb
    import pyarrow as pa

#: The columns taken from the harmonized edition, in the order they are written.
NAME_COLUMNS: tuple[str, ...] = ("crop:name", "crop:name_en")

#: The columns the join needs in both files.
KEY_COLUMNS: tuple[str, ...] = ("id", "crop:code")

#: How many mismatched field ids an error message lists.
_EXAMPLES = 5

#: DuckDB's memory for the join; it spills to disk beyond this.
_JOIN_MEMORY = "4GB"


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


def _check_columns(fields_file: Path, source_file: Path) -> tuple[str, ...]:
    """The name columns to take from the source; raises when the join could go wrong."""
    import pyarrow.parquet as pq

    fields_schema, source_schema = pq.read_schema(fields_file), pq.read_schema(source_file)
    if "crop:name" not in source_schema.names:
        raise CropNameError(source_file, f"{source_file} has no crop:name column")
    for path, schema in ((fields_file, fields_schema), (source_file, source_schema)):
        missing = [c for c in KEY_COLUMNS if c not in schema.names]
        if missing:
            raise CropNameError(path, f"{path} has no {', '.join(missing)} column")
    # Equal types, so the join never casts: "007" must not match 7.
    fields_id, source_id = fields_schema.field("id").type, source_schema.field("id").type
    if fields_id != source_id:
        raise CropNameError(
            fields_file, f"id is {fields_id} in {fields_file} but {source_id} in {source_file}"
        )
    return tuple(c for c in NAME_COLUMNS if c in source_schema.names)


def _check_ids(con: duckdb.DuckDBPyConnection, path: Path, *, unique: bool) -> None:
    """Refuse fields without an id, and (``unique``) ids that repeat."""
    nulls, repeats = con.execute(
        "SELECT count(*) - count(id), count(id) - count(DISTINCT id) FROM read_parquet(?)",
        [str(path)],
    ).fetchone()
    if nulls:
        raise CropNameError(path, f"{path} has {nulls} field(s) without an id")
    if unique and repeats:
        raise CropNameError(path, f"{path} repeats {repeats} field id(s)")


def _join(con: duckdb.DuckDBPyConnection, fields_file: Path, source_file: Path, names) -> None:
    """Join the source's names onto every field, in a temp table ``joined``.

    Both crop codes are compared as text, so codes stored as different types count as
    different and the join is refused rather than guessed.
    """
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


def _joined_names(
    fields_file: Path, source_file: Path, names: tuple[str, ...], rows: int
) -> pa.Table:
    """The name columns for each row of ``fields_file``, in its row order, checked."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{_JOIN_MEMORY}'")
    _check_ids(con, source_file, unique=True)
    _check_ids(con, fields_file, unique=False)
    _join(con, fields_file, source_file, names)
    for flag, what in (("missing", "not in"), ("differs", "with another crop:code in")):
        count, examples = con.execute(
            f"SELECT count(*), list(id ORDER BY row_number)[1:{_EXAMPLES}] FROM joined WHERE {flag}"
        ).fetchone()
        if count:
            raise CropNameError(
                fields_file,
                f"{count} field(s) of {fields_file} {what} {source_file}, e.g. {examples}",
            )
    quoted = ", ".join(f'"{c}"' for c in names)
    table = con.execute(f"SELECT {quoted} FROM joined ORDER BY row_number").fetch_arrow_table()
    if table.num_rows != rows:
        raise CropNameError(
            fields_file, f"the join gave {table.num_rows} rows for {rows} fields of {fields_file}"
        )
    return table


def _output_schema(schema: pa.Schema, names_table: pa.Table) -> pa.Schema:
    """``schema`` without any name column, then the joined ones right after crop:code."""
    for name in NAME_COLUMNS:
        if name in schema.names:
            schema = schema.remove(schema.get_field_index(name))
    at = schema.get_field_index("crop:code") + 1
    for offset, field in enumerate(names_table.schema):
        schema = schema.insert(at + offset, field)
    return schema


def _group_changed(group: pa.Table, part: pa.Table, schema: pa.Schema) -> bool:
    """Whether a row group's name columns differ from the joined ones."""
    stale = [c for c in NAME_COLUMNS if c in group.column_names and c not in schema.names]
    if stale:
        return True
    return any(
        name not in group.column_names
        or not group.column(name).cast(part.schema.field(name).type).equals(part.column(name))
        for name in part.column_names
    )


def _rewrite(fields_file: Path, tmp: Path, names_table: pa.Table) -> bool:
    """Write ``fields_file`` with the joined names to ``tmp``; whether anything changed."""
    import pyarrow.parquet as pq

    source = pq.ParquetFile(fields_file)
    schema = _output_schema(source.schema_arrow, names_table)
    meta = source.metadata
    compression = meta.row_group(0).column(0).compression if meta.num_row_groups else "ZSTD"
    changed = False
    offset = 0
    with pq.ParquetWriter(tmp, schema, compression=compression.lower()) as writer:
        for index in range(source.num_row_groups):
            group = source.read_row_group(index)
            part = names_table.slice(offset, group.num_rows)
            offset += group.num_rows
            changed = changed or _group_changed(group, part, schema)
            group = group.drop_columns([c for c in NAME_COLUMNS if c in group.column_names])
            for name in part.column_names:
                group = group.append_column(name, part.column(name))
            writer.write_table(
                group.select(schema.names).cast(schema), row_group_size=max(len(group), 1)
            )
    return changed


def add_crop_names(
    fields_file: Path | str, source_file: Path | str, output_file: Path | str | None = None
) -> CropNameResult:
    """Join the crop name columns of ``source_file`` onto ``fields_file`` by field id.

    A name column the fields file has but the source lacks is dropped, so the names
    always come from one edition.

    Args:
        fields_file: A benchmark's fields (or fields_filtered) GeoParquet.
        source_file: The harmonized edition it was built from, with crop names.
        output_file: Where to write; ``fields_file`` itself when None. Nothing is
            written when the names are already there and equal.

    Raises:
        CropNameError: The source has no crop:name, a key column is missing, the id
            types differ, an id is missing or (in the source) repeated, or a field is
            missing from the source or has another crop:code there. Nothing is written.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    fields_file, source_file = Path(fields_file), Path(source_file)
    output = Path(output_file) if output_file is not None else fields_file
    names = _check_columns(fields_file, source_file)
    rows = pq.ParquetFile(fields_file).metadata.num_rows
    names_table = _joined_names(fields_file, source_file, names, rows)
    tmp = output.with_name(f".{output.name}.tmp")
    try:
        changed = _rewrite(fields_file, tmp, names_table)
        if changed or output != fields_file:
            tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    named = rows - (pc.sum(pc.is_null(names_table.column("crop:name"))).as_py() or 0)
    return CropNameResult(rows=rows, named=named, columns=names, changed=changed)
