"""HCAT3 crop labels for field polygons.

Fields that already carry ``hcat:code`` are left alone. Otherwise the source crop
column (``crop:code`` / ``crop:name``) is mapped to HCAT3 through a user mapping
CSV, or through the dataset's ``crop:code_list`` when it is one of the lists
bundled under ``data/code_lists``. Crops that cannot be mapped are written to a
template CSV that, once filled in, is itself a valid mapping.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb

from ftw_dataset_tools.api.fs import create_temp_file, finalize_temp_file
from ftw_dataset_tools.api.geo import ensure_spatial_loaded, sql_path, write_geoparquet

if TYPE_CHECKING:
    from collections.abc import Callable

CODE_COLUMN = "hcat:code"
NAME_COLUMN = "hcat:name"
NAME_EN_COLUMN = "hcat:name_en"
HCAT_COLUMNS = (CODE_COLUMN, NAME_COLUMN, NAME_EN_COLUMN)

SOURCE_CODE_COLUMNS = ("crop:code", "crop_code")
SOURCE_NAME_COLUMNS = ("crop:name", "crop_name")
SOURCE_NAME_EN_COLUMN = "crop:name_en"

PROVENANCE_KEY = "ftw:crop_labels"
HCAT_VERSION = "HCAT3"
HCAT_SCHEMA = "https://fiboa.org/hcat-extension/v0.3.0/schema.yaml"

# Parquet key-value entries that describe the old file and are regenerated on write.
_REGENERATED_KV = {"geo", "ARROW:schema", "pandas", PROVENANCE_KEY}

_CODE_LISTS = files("ftw_dataset_tools.api.data").joinpath("code_lists")
_URL_PATTERNS = (
    (re.compile(r"^https?://(?:www\.)?fiboa\.org/code/([\w./-]+\.csv)$"), "fiboa"),
    (re.compile(r"maja601/EuroCrops/.*?csvs/country_mappings/([\w.-]+\.csv)$"), "eurocrops"),
)

# Numeric-looking keys compare as numbers ("01" == "1" == "1.0"), everything else
# case-insensitively.
_KEY_MACRO = """
CREATE OR REPLACE MACRO crop_key(x) AS coalesce(
    CAST(TRY_CAST(nullif(trim(CAST(x AS VARCHAR)), '') AS DECIMAL(38, 6)) AS VARCHAR),
    nullif(lower(trim(CAST(x AS VARCHAR))), '')
)
"""


class CropLabelError(ValueError):
    """Crop labels could not be resolved, or a mapping file is invalid."""

    def __init__(self, message: str, template_path: Path | None = None) -> None:
        super().__init__(message)
        self.template_path = template_path


@dataclass
class CropLabelResult:
    """What the labeling step did."""

    source: str  # "native", "code_list", "mapping" or "skipped"
    fields_total: int = 0
    fields_labeled: int = 0
    distinct_codes: int = 0
    unmapped_values: int = 0
    reason: str | None = None
    template_path: Path | None = None


def crop_labels_summary(result: CropLabelResult | None) -> str:
    """The one-line run summary for the crop label step."""
    if result is None:
        return "Crop labels: disabled"
    if result.source == "skipped":
        line = f"Crop labels: skipped ({result.reason})"
    else:
        pct = 100.0 * result.fields_labeled / result.fields_total if result.fields_total else 0.0
        line = (
            f"Crop labels ({result.source}): {result.fields_labeled:,}/{result.fields_total:,} "
            f"fields ({pct:.1f}%), {result.distinct_codes} HCAT codes"
        )
    if result.template_path is not None:
        line += f"; mapping template: {result.template_path}"
    return line


def bundled_code_list(url: str | None) -> Path | None:
    """The bundled copy of a ``crop:code_list`` URL, or None when it is not bundled."""
    if not url:
        return None
    for pattern, folder in _URL_PATTERNS:
        match = pattern.search(url.strip())
        if match and ".." not in match.group(1):
            candidate = _CODE_LISTS.joinpath(folder, *match.group(1).split("/"))
            if candidate.is_file():
                return Path(str(candidate))
    return None


def hcat3_codes() -> dict[str, str]:
    """HCAT3 code -> name from the bundled core list."""
    rows = _read_csv(Path(str(_CODE_LISTS.joinpath("HCAT3.csv"))))
    return {row["HCAT3_code"]: row["HCAT3_name"] for row in rows}


def read_provenance(fields_file: Path | str) -> dict[str, Any] | None:
    """The ``ftw:crop_labels`` record ftwd wrote into a fields file, if any."""
    value = _kv_metadata(Path(fields_file)).get(PROVENANCE_KEY)
    return json.loads(value) if value else None


def load_mapping(path: Path | str) -> list[dict[str, str]]:
    """Read and validate a user mapping CSV.

    Raises:
        CropLabelError: If the file is missing required columns or uses codes that
            are not in HCAT3.
    """
    path = Path(path)
    if not path.is_file():
        raise CropLabelError(f"Crop mapping not found: {path}")
    rows = _read_csv(path)
    columns = set(rows[0]) if rows else set()
    if "HCAT3_code" not in columns or not columns & {"original_code", "original_name"}:
        raise CropLabelError(
            f"Crop mapping {path.name} needs an HCAT3_code column and an original_code "
            "or original_name column."
        )
    known = hcat3_codes()
    unknown = sorted({r["HCAT3_code"] for r in rows if r["HCAT3_code"]} - set(known))
    if unknown:
        raise CropLabelError(f"Crop mapping {path.name} has codes that are not HCAT3: {unknown}")
    for row in rows:
        if row["HCAT3_code"] and not row.get("HCAT3_name"):
            row["HCAT3_name"] = known[row["HCAT3_code"]]
    return rows


def add_crop_labels(
    fields_file: Path | str,
    output_file: Path | str | None = None,
    *,
    mapping: Path | str | None = None,
    column: str | None = None,
    strict: bool = False,
    template_path: Path | str | None = None,
    template_only: bool = False,
    on_progress: Callable[[str], None] | None = None,
) -> CropLabelResult:
    """Write ``hcat:code``, ``hcat:name`` and ``hcat:name_en`` onto a fields file.

    ``output_file`` defaults to rewriting ``fields_file`` in place. The template
    (all crop values, unmapped first, prefilled where a mapping is known) goes to
    ``template_path``, or beside the output, whenever a crop value is unmapped.

    Raises:
        CropLabelError: With ``strict``, when any crop value is unmapped or no crop
            column exists; always, for an invalid mapping or ``column``.
    """
    fields_path = Path(fields_file).resolve()
    output_path = Path(output_file).resolve() if output_file else fields_path
    if template_path is None:
        template_path = output_path.with_name(f"{output_path.stem}_crop_mapping_template.csv")
    template_path = Path(template_path)

    def log(msg: str) -> None:
        if on_progress:
            on_progress(msg)

    kv = _kv_metadata(fields_path)
    ftwd_added = PROVENANCE_KEY in kv

    con = duckdb.connect(":memory:")
    ensure_spatial_loaded(con)
    con.execute(_KEY_MACRO)
    try:
        src = f"read_parquet('{sql_path(fields_path)}', file_row_number=true)"
        columns = [r[0] for r in con.execute(f"DESCRIBE SELECT * FROM {src}").fetchall()]
        total = con.execute(f"SELECT count(*) FROM {src}").fetchone()[0]

        if mapping is None and not ftwd_added and CODE_COLUMN in columns and not template_only:
            labeled, distinct = con.execute(
                f'SELECT count("{CODE_COLUMN}"), count(DISTINCT "{CODE_COLUMN}") FROM {src}'
            ).fetchone()
            if labeled:
                _copy_if_needed(fields_path, output_path)
                log(f"Fields already carry {CODE_COLUMN}; keeping the source labels")
                return CropLabelResult("native", total, labeled, distinct)

        code_col, name_col = _key_columns(columns, column)
        if code_col is None and name_col is None:
            reason = f"no crop column ({', '.join(SOURCE_CODE_COLUMNS + SOURCE_NAME_COLUMNS)})"
            return _unlabeled(fields_path, output_path, total, reason, strict, log)

        if mapping is not None:
            source, rows = "mapping", load_mapping(mapping)
            record: dict[str, Any] = {
                "mapping": Path(mapping).name,
                "mapping_sha256": hashlib.sha256(Path(mapping).read_bytes()).hexdigest(),
            }
        else:
            url = _code_list_url(kv, con, src, columns)
            bundled = bundled_code_list(url)
            source, rows = "code_list", _read_csv(bundled) if bundled else []
            record = {"code_list": url}
            if bundled is None:
                log(
                    f"Note: crop code list {url} is not bundled"
                    if url
                    else "Note: fields name no crop:code_list"
                )

        list_cols = set(rows[0]) if rows else set()
        join_code = code_col if code_col and "original_code" in list_cols else None
        join_name = name_col if name_col and "original_name" in list_cols else None
        if join_code is None and join_name is None and rows:
            log("Note: the crop mapping's key columns do not match the fields' crop columns")
            rows = []
        _load_mapping_tables(con, rows, by_code=join_code is not None)

        key_col = join_code or join_name or code_col or name_col
        con.execute(
            f"""CREATE VIEW labeled AS
            SELECT f.*, coalesce(p.code, s.code) AS __code, coalesce(p.name, s.name) AS __name,
                   coalesce(p.name_en, s.name_en) AS __name_en
            FROM (SELECT *, crop_key("{key_col}") AS __k,
                         {f'crop_key("{name_col}")' if name_col else "NULL"} AS __n
                  FROM {src}) f
            LEFT JOIN pairs p ON f.__k = p.k AND f.__n = p.n
            LEFT JOIN singles s ON f.__k = s.k"""
        )
        labeled, distinct = con.execute(
            "SELECT count(__code), count(DISTINCT __code) FROM labeled"
        ).fetchone()
        unmapped = con.execute(
            "SELECT count(DISTINCT __k) FROM labeled WHERE __code IS NULL AND __k IS NOT NULL"
        ).fetchone()[0]

        written_template = None
        if unmapped or template_only:
            written_template = _write_template(
                con, template_path, key_col, key_col == code_col, name_col, columns
            )
            log(f"Wrote crop mapping template ({unmapped} unmapped value(s)): {template_path}")
        elif template_path.exists() and (
            mapping is None or Path(mapping).resolve() != template_path.resolve()
        ):
            template_path.unlink()

        if template_only:
            return CropLabelResult(
                source, total, labeled, distinct, unmapped, template_path=written_template
            )
        if unmapped and strict:
            raise CropLabelError(
                f"{unmapped} crop value(s) have no HCAT3 mapping. Fill in "
                f"{template_path} and pass it as the crop mapping.",
                template_path,
            )
        if labeled == 0:
            reason = f"no crop value maps to HCAT3 ({key_col})"
            result = _unlabeled(
                fields_path,
                output_path,
                total,
                reason,
                strict=False,
                log=log,
                con=con if ftwd_added else None,
                kv=kv,
            )
            result.template_path = written_template
            result.unmapped_values = unmapped
            return result

        record.update(
            source=source,
            hcat_version=HCAT_VERSION,
            key_column=key_col,
            fields_labeled=labeled,
            fields_total=total,
        )
        _write_labeled(con, output_path, kv, record)
    finally:
        con.close()

    if unmapped:
        log(f"Warning: {total - labeled:,} of {total:,} fields have no HCAT3 crop label")
    log(f"Crop labels: {labeled:,} of {total:,} fields mapped to {distinct} HCAT3 codes")
    return CropLabelResult(source, total, labeled, distinct, unmapped, None, written_template)


def _key_columns(columns: list[str], column: str | None) -> tuple[str | None, str | None]:
    """The (code, name) source columns to match on."""
    name_col = next((c for c in SOURCE_NAME_COLUMNS if c in columns), None)
    if column is not None:
        if column not in columns:
            raise CropLabelError(f"Crop column '{column}' is not in the fields file")
        if column in SOURCE_NAME_COLUMNS:
            return None, column
        return column, name_col
    code_col = next((c for c in SOURCE_CODE_COLUMNS if c in columns), None)
    return code_col, name_col


def _load_mapping_tables(
    con: duckdb.DuckDBPyConnection, rows: list[dict[str, str]], *, by_code: bool
) -> None:
    """Build ``singles`` (key -> label) and ``pairs`` ((key, name) -> label).

    A row matching on both key and crop name wins; a key that maps to more than
    one HCAT code in the list only matches together with its name.
    """
    key = "original_code" if by_code else "original_name"
    con.execute(
        "CREATE TABLE raw (i INTEGER, k VARCHAR, n VARCHAR, code UINTEGER, name VARCHAR, name_en VARCHAR)"
    )
    if rows:
        con.executemany(
            "INSERT INTO raw VALUES (?, crop_key(?), crop_key(?), TRY_CAST(nullif(?, '') AS UINTEGER), nullif(?, ''), nullif(?, ''))",
            [
                (
                    i,
                    row.get(key),
                    row.get("original_name"),
                    row.get("HCAT3_code"),
                    row.get("HCAT3_name"),
                    row.get("translated_name"),
                )
                for i, row in enumerate(rows)
            ],
        )
    con.execute(
        """CREATE TABLE singles AS
        SELECT k, min(code) AS code, arg_min(name, i) AS name, arg_min(name_en, i) AS name_en
        FROM raw WHERE k IS NOT NULL AND code IS NOT NULL
        GROUP BY k HAVING count(DISTINCT code) = 1"""
    )
    con.execute(
        """CREATE TABLE pairs AS
        SELECT k, n, arg_min(code, i) AS code, arg_min(name, i) AS name, arg_min(name_en, i) AS name_en
        FROM raw WHERE k IS NOT NULL AND code IS NOT NULL
        GROUP BY k, n"""
    )


def _write_template(
    con: duckdb.DuckDBPyConnection,
    template_path: Path,
    key_col: str,
    by_code: bool,
    name_col: str | None,
    columns: list[str],
) -> Path:
    key_header = "original_code" if by_code else "original_name"
    extra = []
    if by_code and name_col:
        extra.append(f'any_value("{name_col}") AS original_name')
    if SOURCE_NAME_EN_COLUMN in columns:
        extra.append(f'any_value("{SOURCE_NAME_EN_COLUMN}") AS translated_name')
    rows = con.execute(
        f"""SELECT CAST("{key_col}" AS VARCHAR) AS {key_header}, {"".join(e + ", " for e in extra)}
               count(*) AS fields, min(__code) AS HCAT3_code, min(__name) AS HCAT3_name
        FROM labeled WHERE __k IS NOT NULL
        GROUP BY "{key_col}"
        ORDER BY HCAT3_code IS NOT NULL, fields DESC, {key_header}"""
    )
    header = [d[0] for d in rows.description]
    template_path.parent.mkdir(parents=True, exist_ok=True)
    with template_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows.fetchall())
    return template_path


def _write_labeled(
    con: duckdb.DuckDBPyConnection, output_path: Path, kv: dict[str, str], record: dict
) -> None:
    query = f"""SELECT * EXCLUDE ({_helper_columns(con)}), __code AS "{CODE_COLUMN}",
                       __name AS "{NAME_COLUMN}", __name_en AS "{NAME_EN_COLUMN}"
                FROM labeled ORDER BY file_row_number"""
    metadata = _carried_kv(kv, add_hcat_schema=True)
    metadata[PROVENANCE_KEY] = json.dumps(record)
    _write_parquet(con, query, output_path, metadata)


def _helper_columns(con: duckdb.DuckDBPyConnection) -> str:
    """EXCLUDE list for ``labeled``: the join helpers plus any existing HCAT columns."""
    existing = [r[0] for r in con.execute("DESCRIBE labeled").fetchall()]
    drop = ["file_row_number", "__k", "__n", "__code", "__name", "__name_en"]
    drop += [c for c in HCAT_COLUMNS if c in existing]
    return ", ".join(f'"{c}"' for c in drop)


def _unlabeled(
    fields_path: Path,
    output_path: Path,
    total: int,
    reason: str,
    strict: bool,
    log: Callable[[str], None],
    *,
    con: duckdb.DuckDBPyConnection | None = None,
    kv: dict[str, str] | None = None,
) -> CropLabelResult:
    """Leave the fields without labels, dropping any an earlier run added."""
    if strict:
        raise CropLabelError(f"Cannot add crop labels: {reason}")
    if con is not None:
        query = f"SELECT * EXCLUDE ({_helper_columns(con)}) FROM labeled ORDER BY file_row_number"
        _write_parquet(con, query, output_path, _carried_kv(kv or {}, add_hcat_schema=False))
    else:
        _copy_if_needed(fields_path, output_path)
    log(f"Note: {reason}; skipping crop labels")
    return CropLabelResult("skipped", total, reason=reason)


def _write_parquet(
    con: duckdb.DuckDBPyConnection, query: str, output_path: Path, kv: dict[str, str]
) -> None:
    tmp_path: Path | None = None
    try:
        tmp_path = create_temp_file(output_path, suffix=".parquet")
        write_geoparquet(tmp_path, conn=con, query=query, kv_metadata=kv)
        finalize_temp_file(tmp_path, output_path)
        tmp_path = None
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def _carried_kv(kv: dict[str, str], *, add_hcat_schema: bool) -> dict[str, str]:
    """The source's key-value metadata, with the HCAT extension declared on its collection."""
    carried = {k: v for k, v in kv.items() if k not in _REGENERATED_KV}
    raw = carried.get("collection")
    if raw:
        try:
            collection = json.loads(raw)
        except json.JSONDecodeError:
            return carried
        for schemas in (collection.get("schemas") or {}).values():
            if isinstance(schemas, list):
                if add_hcat_schema and HCAT_SCHEMA not in schemas:
                    schemas.append(HCAT_SCHEMA)
                elif not add_hcat_schema and HCAT_SCHEMA in schemas:
                    schemas.remove(HCAT_SCHEMA)
        carried["collection"] = json.dumps(collection)
    return carried


def _code_list_url(
    kv: dict[str, str], con: duckdb.DuckDBPyConnection, src: str, columns: list[str]
) -> str | None:
    """``crop:code_list`` from the fiboa collection metadata, or the column of that name."""
    try:
        url = json.loads(kv.get("collection") or "{}").get("crop:code_list")
    except json.JSONDecodeError:
        url = None
    if not url and "crop:code_list" in columns:
        row = con.execute(
            f'SELECT "crop:code_list" FROM {src} WHERE "crop:code_list" IS NOT NULL LIMIT 1'
        ).fetchone()
        url = row[0] if row else None
    return url


def _kv_metadata(path: Path) -> dict[str, str]:
    con = duckdb.connect(":memory:")
    try:
        rows = con.execute(
            f"SELECT decode(key), decode(value) FROM parquet_kv_metadata('{sql_path(path)}')"
        ).fetchall()
    finally:
        con.close()
    return dict(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            {k.strip(): (v or "").strip() for k, v in row.items() if k}
            for row in csv.DictReader(handle)
        ]


def _copy_if_needed(source: Path, target: Path) -> None:
    if source != target:
        shutil.copy2(source, target)
