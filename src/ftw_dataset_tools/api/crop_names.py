"""Give a built benchmark's field files the source's original crop names.

A benchmark's fields file holds every column of the harmonized edition it was built
from. A harmonized edition rebuilt after fiboa-cli learned to name crop codes from its
code tables (fiboa/cli#352) has ``crop:name`` (and sometimes ``crop:name_en``) that a
benchmark built earlier lacks. ``add_crop_names`` joins those columns in by field
``id``, and refuses when a field is missing from the edition or its ``crop:code``
differs there, so no field ever takes another field's name.

When the edition's names come from a code table, so a name depends on nothing but
the code, ``by="code"`` takes each code's name from the edition instead. That serves a
benchmark built from an earlier release of the same edition whose ids no longer
line up (e.g. fr 2024: RPG 3.0 was replaced by RPG 4.0). It refuses when a code has
more than one name in the edition.

The file is rewritten one row group at a time with pyarrow, not read whole through
geoparquet-io: a fields file can hold ten million fields, and streaming keeps memory
flat and the file as it was (row groups, each column's codec, the ``geo`` and fiboa
metadata). A codec's level is not stored in the file, so it is pyarrow's default.
Only the name columns change, and the geometry is copied as is, so the ``geo``
metadata stays true.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from ftw_dataset_tools.api.logging_config import get_logger

if TYPE_CHECKING:
    import duckdb
    import pyarrow as pa
    import pyarrow.parquet as pq

#: The columns taken from the harmonized edition, in the order they are written.
NAME_COLUMNS: tuple[str, ...] = ("crop:name", "crop:name_en")

logger = get_logger(__name__)

#: The columns the join needs in both files.
KEY_COLUMNS: tuple[str, ...] = ("id", "crop:code")

#: What the names can be joined by: each field's id, or its crop code.
JoinKey = Literal["id", "code"]
JOIN_KEYS: tuple[str, ...] = ("id", "code")

#: A text crop code, with "NaN" as no code.
_TEXT_CODE = """CASE WHEN lower(CAST({column} AS VARCHAR)) = 'nan' THEN NULL
    ELSE CAST({column} AS VARCHAR) END"""

#: An integer crop code as text, exactly (never through a float).
_INTEGER_CODE = "CAST({column} AS VARCHAR)"

#: A float crop code as text: NaN is no code, and a whole float is written as the
#: integer it is, so 10.0 in one file matches 10 in the other. A fraction is written
#: as the shortest text that reads back to the float in its own width, so a float32
#: 0.6 matches a float64 0.6 (the decimal number both stand for), not the float64
#: 0.6000000238418579 that equals the float32 exactly.
_FLOAT_CODE = """CASE WHEN isnan(CAST({column} AS DOUBLE)) THEN NULL
    WHEN isfinite(CAST({column} AS DOUBLE))
         AND CAST({column} AS DOUBLE) = trunc(CAST({column} AS DOUBLE))
      THEN coalesce(CAST(TRY_CAST(CAST({column} AS DOUBLE) AS HUGEINT) AS VARCHAR),
                    CAST(CAST({column} AS DOUBLE) AS VARCHAR))
    ELSE CAST({column} AS VARCHAR) END"""

#: A decimal crop code as text, exactly: whole as the integer, otherwise its digits
#: without trailing zeros, so 10.50 matches the decimal 10.5 at any scale and the float
#: 10.5. It never goes through a float, so two different decimals never match. A float
#: that prints with an exponent (1e-05) does not match the decimal 0.00001: refused by
#: id, unnamed by code.
_DECIMAL_CODE = """CASE WHEN {column} = trunc({column})
      THEN CAST(CAST(trunc({column}) AS HUGEINT) AS VARCHAR)
    ELSE rtrim(CAST({column} AS VARCHAR), '0') END"""

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


def _code_kind(code_type: pa.DataType) -> str:
    """Whether a crop:code column holds text or numbers ("null" when it holds nothing)."""
    import pyarrow as pa

    if pa.types.is_dictionary(code_type):
        code_type = code_type.value_type
    if pa.types.is_null(code_type):
        return "null"
    if pa.types.is_string(code_type) or pa.types.is_large_string(code_type):
        return "text"
    if pa.types.is_integer(code_type) or pa.types.is_floating(code_type):
        return "number"
    if pa.types.is_decimal(code_type):
        return "number"
    return str(code_type)


def _code_text(code_type: pa.DataType, column: str = '"crop:code"') -> str:
    """SQL for a crop code of ``code_type`` as comparable text, NULL for no code."""
    import pyarrow as pa

    if pa.types.is_integer(code_type):
        template = _INTEGER_CODE
    elif pa.types.is_floating(code_type):
        template = _FLOAT_CODE
    elif pa.types.is_decimal(code_type):
        template = _DECIMAL_CODE
    else:
        template = _TEXT_CODE
    return template.format(column=column)


def _id_kind(id_type: pa.DataType) -> str:
    """What an id column holds, for comparing two files' ids without a cast."""
    import pyarrow as pa

    if pa.types.is_dictionary(id_type):
        id_type = id_type.value_type
    if pa.types.is_string(id_type) or pa.types.is_large_string(id_type):
        return "text"
    if pa.types.is_integer(id_type):
        return "integer"
    return str(id_type)


def _check_code_types(
    fields_file: Path, fields_schema: pa.Schema, source_file: Path, source_schema: pa.Schema
) -> None:
    """Refuse text codes against numeric ones: "01" and 1 would never match, silently."""
    kinds = {
        _code_kind(schema.field("crop:code").type) for schema in (fields_schema, source_schema)
    }
    kinds.discard("null")
    if len(kinds) > 1:
        raise CropNameError(
            fields_file,
            f"crop:code is {fields_schema.field('crop:code').type} in {fields_file} but "
            f"{source_schema.field('crop:code').type} in {source_file}",
        )


def _check_columns(fields_file: Path, source_file: Path, by: JoinKey) -> tuple[str, ...]:
    """The name columns to take from the source; raises when the join could go wrong."""
    import pyarrow.parquet as pq

    fields_schema, source_schema = pq.read_schema(fields_file), pq.read_schema(source_file)
    if "crop:name" not in source_schema.names:
        raise CropNameError(source_file, f"{source_file} has no crop:name column")
    keys = KEY_COLUMNS if by == "id" else ("crop:code",)
    for path, schema in ((fields_file, fields_schema), (source_file, source_schema)):
        missing = [c for c in keys if c not in schema.names]
        if missing:
            raise CropNameError(path, f"{path} has no {', '.join(missing)} column")
    if by == "id":
        _check_id_types(fields_file, fields_schema, source_file, source_schema)
    _check_code_types(fields_file, fields_schema, source_file, source_schema)
    return tuple(c for c in NAME_COLUMNS if c in source_schema.names)


def _check_id_types(
    fields_file: Path, fields_schema: pa.Schema, source_file: Path, source_schema: pa.Schema
) -> None:
    """Refuse ids of different kinds, so the join never casts: "007" must not match 7.

    Text ids match text ids (string or large_string), integer ids integer ids of any
    width; any other type only the same type.
    """
    fields_id, source_id = fields_schema.field("id").type, source_schema.field("id").type
    if _id_kind(fields_id) != _id_kind(source_id):
        raise CropNameError(
            fields_file, f"id is {fields_id} in {fields_file} but {source_id} in {source_file}"
        )


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


def _join(
    con: duckdb.DuckDBPyConnection, fields_file: Path, source_file: Path, names: tuple[str, ...]
) -> None:
    """Join the source's names onto every field by id, in a temp table ``joined``.

    Both crop codes are compared as text (NaN as no code, 10.0 as 10), flagging the
    fields whose code differs.
    """
    selected = ", ".join(f's."{c}"' for c in names)
    fields_code = _code_text(_code_type(fields_file), 'f."crop:code"')
    source_code = _code_text(_code_type(source_file), 's."crop:code"')
    con.execute(
        f"""
        CREATE TEMP TABLE joined AS
        SELECT f.file_row_number AS row_number, f.id AS id,
               s.id IS NULL AS missing,
               s.id IS NOT NULL
                 AND ({fields_code}) IS DISTINCT FROM ({source_code}) AS differs,
               {selected}
        FROM read_parquet(?, file_row_number = true) f
        LEFT JOIN read_parquet(?) s ON f.id = s.id
        """,
        [str(fields_file), str(source_file)],
    )


def _code_type(path: Path) -> pa.DataType:
    """The type of ``path``'s crop:code column."""
    import pyarrow.parquet as pq

    return pq.read_schema(path).field("crop:code").type


def _code_names(con: duckdb.DuckDBPyConnection, source_file: Path, names: tuple[str, ...]) -> None:
    """Each crop code's names in the source, in a temp table ``codes``; one row per code.

    A code whose rows do not all carry the same names (a name column differs, or is
    empty on some rows only) is refused: the code alone would not tell which is right.
    """
    quoted = ", ".join(f'"{c}"' for c in names)
    code = _code_text(_code_type(source_file))
    con.execute(
        f"""
        CREATE TEMP TABLE codes AS
        SELECT DISTINCT {code} AS code, {quoted}
        FROM read_parquet(?) WHERE ({code}) IS NOT NULL
        """,
        [str(source_file)],
    )
    count, examples = con.execute(
        f"""
        SELECT count(*), list(code ORDER BY code)[1:{_EXAMPLES}]
        FROM (SELECT code FROM codes GROUP BY code HAVING count(*) > 1)
        """
    ).fetchone()
    if count:
        raise CropNameError(
            source_file,
            f"{count} crop code(s) of {source_file} do not have the same {', '.join(names)} "
            f"on every row (or are unnamed on some), e.g. {examples}",
        )


def _warn_unnamed_codes(con: duckdb.DuckDBPyConnection, fields_file: Path) -> None:
    """Log the crop codes of ``fields_file`` that the source does not name."""
    count, fields, examples = con.execute(
        f"""
        SELECT count(*), sum(n), list(code ORDER BY n DESC)[1:{_EXAMPLES}]
        FROM (SELECT code, count(*) n FROM joined
              WHERE code IS NOT NULL AND "crop:name" IS NULL GROUP BY code)
        """
    ).fetchone()
    if count:
        logger.warning(
            f"{count} crop code(s) on {fields:,} field(s) of {fields_file} have no name in the "
            f"source, e.g. {examples}"
        )


def _join_by_code(
    con: duckdb.DuckDBPyConnection, fields_file: Path, source_file: Path, names: tuple[str, ...]
) -> None:
    """Give every field the names of its crop code in the source, in a temp table ``joined``.

    Codes are compared as text, NaN counting as no code and 10.0 as 10. A field whose
    code the source does not name stays unnamed, and the codes concerned are logged as a
    warning.
    """
    _code_names(con, source_file, names)
    selected = ", ".join(f'c."{c}"' for c in names)
    con.execute(
        f"""
        CREATE TEMP TABLE joined AS
        SELECT f.file_row_number AS row_number, f.code, {selected}
        FROM (SELECT file_row_number, {_code_text(_code_type(fields_file))} AS code
              FROM read_parquet(?, file_row_number = true)) f
        LEFT JOIN codes c ON f.code = c.code
        """,
        [str(fields_file)],
    )
    _warn_unnamed_codes(con, fields_file)


def _join_by_id(
    con: duckdb.DuckDBPyConnection, fields_file: Path, source_file: Path, names: tuple[str, ...]
) -> None:
    """Join the source's names onto every field by id, refusing any field that does not match."""
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


def _joined_names(
    fields_file: Path, source_file: Path, names: tuple[str, ...], rows: int, by: JoinKey
) -> pa.Table:
    """The name columns for each row of ``fields_file``, in its row order, checked."""
    import duckdb

    con = duckdb.connect()
    con.execute(f"SET memory_limit = '{_JOIN_MEMORY}'")
    join = _join_by_id if by == "id" else _join_by_code
    join(con, fields_file, source_file, names)
    # As text: a name column empty in the source would otherwise come back as INTEGER.
    text = ", ".join(f'CAST("{c}" AS VARCHAR) AS "{c}"' for c in names)
    table = con.execute(f"SELECT {text} FROM joined ORDER BY row_number").fetch_arrow_table()
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


def _columns(schema: pa.Schema) -> list[tuple[str, pa.DataType]]:
    return [(field.name, field.type) for field in schema]


def _codec(name: str) -> str:
    """A codec as parquet metadata names it, as pyarrow's writer takes it."""
    return "none" if name == "UNCOMPRESSED" else name.lower()


def _compression(meta: pq.FileMetaData, names: list[str]) -> dict[str, str]:
    """Each column's codec in ``meta``; the new name columns take crop:code's."""
    if not meta.num_row_groups:
        return dict.fromkeys(names, "zstd")
    group = meta.row_group(0)
    codecs = {
        group.column(i).path_in_schema: _codec(group.column(i).compression)
        for i in range(group.num_columns)
    }
    for name in names:
        codecs[name] = codecs.get("crop:code", "zstd")
    return codecs


def _rewrite(fields_file: Path, tmp: Path, names_table: pa.Table) -> bool:
    """Write ``fields_file`` with the joined names to ``tmp``; whether anything changed."""
    import pyarrow.parquet as pq

    source = pq.ParquetFile(fields_file)
    schema = _output_schema(source.schema_arrow, names_table)
    compression = _compression(source.metadata, names_table.column_names)
    # A name column added, dropped, moved or of another type is a change in itself.
    changed = _columns(source.schema_arrow) != _columns(schema)
    offset = 0
    with pq.ParquetWriter(tmp, schema, compression=compression) as writer:
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
    fields_file: Path | str,
    source_file: Path | str,
    output_file: Path | str | None = None,
    *,
    by: JoinKey = "id",
) -> CropNameResult:
    """Join the crop name columns of ``source_file`` onto ``fields_file``.

    The names always come from one edition: every name column is replaced, so a name
    column the source lacks is dropped, and by code a field whose code the source does
    not name loses any name it had.

    Args:
        fields_file: A benchmark's fields (or fields_filtered) GeoParquet.
        source_file: The harmonized edition it was built from, with crop names.
        output_file: Where to write; ``fields_file`` itself when None. Nothing is
            written when the names are already there and equal.
        by: ``"id"`` joins each field to its own row in the source. ``"code"`` gives
            each field the names of its crop code in the source, for sources whose
            names come from a code table; a field whose code the source does not
            name stays unnamed (logged as a warning).

    Raises:
        ValueError: ``by`` is neither "id" nor "code".
        CropNameError: The source has no crop:name, a key column is missing, or one
            file's crop codes are text and the other's numbers. By id:
            the id types differ, an id is missing or (in the source) repeated, or a
            field is missing from the source or has another crop:code there. By code:
            a code's rows in the source do not all have the same names. Nothing is
            written.
    """
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    if by not in JOIN_KEYS:
        raise ValueError(f"by must be one of {JOIN_KEYS}, not {by!r}")
    fields_file, source_file = Path(fields_file), Path(source_file)
    output = Path(output_file) if output_file is not None else fields_file
    names = _check_columns(fields_file, source_file, by)
    rows = pq.ParquetFile(fields_file).metadata.num_rows
    names_table = _joined_names(fields_file, source_file, names, rows, by)
    tmp = output.with_name(f".{output.name}.tmp")
    try:
        changed = _rewrite(fields_file, tmp, names_table)
        if changed or output != fields_file:
            tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    named = rows - (pc.sum(pc.is_null(names_table.column("crop:name"))).as_py() or 0)
    return CropNameResult(rows=rows, named=named, columns=names, changed=changed)
