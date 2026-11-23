"""Make the field ids of a fields file unique.

Some source datasets repeat a field ``id``. Chips, masks, field outlines and field
embeddings all join on it, so the build resolves every group of rows sharing an id
right after the fields are written, by the first rule that fits the group:

``exact``
    The rows are the same: same geometry (WKB) and attributes, with ``metrics:*``
    values within 0.01% of each other. One row is kept.
``parts``
    One field exploded into several rows: the attributes other than ``metrics:*`` are
    equal, the parts' interiors don't overlap (their summed area exceeds the area of
    their union by less than 1% of the smallest part of at least 1 m², or by less
    than 1 m²) and every row carries the same ``metrics:area`` (rows without it are
    never parts) and the union is between half and twice it. The rows become one,
    whose geometry is the valid union of the parts without slivers under 1 m²;
    ``metrics:area`` keeps the source value and the bbox column is recomputed.
``same_parcel``
    The rows have the same polygon (their union is within 1% of the smallest row's
    area) but different attributes, like two declarations of one parcel. One row is
    kept: one whose crop passes the class filter when one is given, otherwise the
    first in content order. The crop names of the rows not kept, other than the
    kept row's, go in ``other_crop_names``.
``renumbered``
    Anything else. Every row is kept and, in content order, gets ``<id>_1`` ...
    ``<id>_k``; the id column becomes VARCHAR and every row of the file gets
    ``source_id``, the id it had in the source.

Rows identical apart from their position resolve the same way in any order: "content
order" sorts a group's rows by geometry WKB, then by their attribute values. Each
resolved row takes the position of its group's first row; other rows keep theirs.
Areas are measured in an equal-area CRS (EPSG:6933), never in degrees. Missing ids
are not duplicates and stay as they are.

Only the id column is read for a file without duplicates, and only the duplicate
groups are loaded into memory.
"""

from __future__ import annotations

import json
import math
import shutil
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import duckdb
import pyarrow as pa
import shapely

from ftw_dataset_tools.api.chip_grid import reprojector
from ftw_dataset_tools.api.geo import (
    detect_geometry_column,
    geoparquet_crs,
    get_bbox_column_name,
    sql_path,
)
from ftw_dataset_tools.api.logging_config import get_logger

if TYPE_CHECKING:
    from ftw_dataset_tools.api.config import ClassFilter

logger = get_logger(__name__)

ID_COLUMN = "id"
SOURCE_ID_COLUMN = "source_id"
OTHER_CROPS_COLUMN = "other_crop_names"
METRICS_PREFIX = "metrics:"
AREA_METRIC = "metrics:area"
# The crop name saved for a same_parcel row not kept: the first of these present
# (the class filter's column is tried between them).
CROP_NAME_COLUMNS = ("crop:name", "hcat:name")

RULES = ("exact", "parts", "same_parcel", "renumbered")
# Columns the rewrite adds or uses internally; a source that has one is refused.
RESERVED_COLUMNS = (
    SOURCE_ID_COLUMN,
    OTHER_CROPS_COLUMN,
    "_pos",
    "_sub",
    "_id_text",
    "_renumbered",
)

_THREADS = 8
AREA_CRS = "EPSG:6933"  # WGS 84 / NSIDC EASE-Grid 2.0 Global, equal-area
METRICS_TOLERANCE = 1e-4  # relative: 0.01%
OVERLAP_TOLERANCE = 0.01  # of the smallest part's area
# Parts and pieces smaller than this are slivers: they set no overlap tolerance (theirs
# would be below the rounding noise of the areas) and are left out of a merged field.
SLIVER_M2 = 1.0
SAME_POLYGON_TOLERANCE = 0.01  # of the smallest row's area
AREA_RATIO_RANGE = (0.5, 2.0)  # union area / metrics:area


class FieldIdsError(ValueError):
    """Raised when a fields file's ids cannot be checked or made unique."""


@dataclass(frozen=True)
class DuplicateIds:
    """The ids of a fields file that more than one row shares."""

    ids: tuple[Any, ...]
    rows: int

    @property
    def extra_rows(self) -> int:
        """Rows beyond one per duplicated id."""
        return self.rows - len(self.ids)

    def __bool__(self) -> bool:
        return bool(self.ids)


class DuplicateIdsError(FieldIdsError):
    """Raised when a fields file that must have unique ids repeats some."""

    def __init__(self, path: Path | str, duplicates: DuplicateIds, id_col: str = ID_COLUMN):
        self.path = Path(path)
        self.duplicates = duplicates
        examples = ", ".join(str(i) for i in sorted(duplicates.ids, key=str)[:5])
        super().__init__(
            f"{self.path.name}: {len(duplicates.ids):,} '{id_col}' value(s) are shared by "
            f"{duplicates.rows:,} rows (e.g. {examples})"
        )


@dataclass
class DedupResult:
    """What :func:`dedupe_fields` did: groups and rows removed per rule, and the ids."""

    rows_in: int = 0
    rows_out: int = 0
    groups: dict[str, int] = field(default_factory=lambda: dict.fromkeys(RULES, 0))
    rows_removed: dict[str, int] = field(default_factory=lambda: dict.fromkeys(RULES, 0))
    changed_ids: dict[str, list[str]] = field(default_factory=lambda: {r: [] for r in RULES})
    added_columns: tuple[str, ...] = ()

    @property
    def changed(self) -> bool:
        """Whether the fields had any duplicate id."""
        return any(self.groups.values())

    def summary(self) -> str:
        """One line for the log."""
        if not self.changed:
            return f"Field ids are unique ({self.rows_in:,} fields)"
        parts = ", ".join(
            f"{self.groups[r]:,} {r} (-{self.rows_removed[r]:,} rows)"
            for r in RULES
            if self.groups[r]
        )
        return (
            f"Resolved duplicate field ids: {parts}; {self.rows_in:,} -> {self.rows_out:,} fields"
        )


# --------------------------------------------------------------------------- #
# Finding duplicates
# --------------------------------------------------------------------------- #


def _quote(column: str) -> str:
    if '"' in column:
        raise FieldIdsError(f"Invalid column name: {column!r}")
    return f'"{column}"'


def _duplicates_sql(path: Path | str, id_col: str) -> str:
    quoted = _quote(id_col)
    return f"""
        SELECT {quoted}, count(*) AS n FROM read_parquet('{sql_path(path)}')
        WHERE {quoted} IS NOT NULL GROUP BY {quoted} HAVING count(*) > 1"""


def _columns(con: duckdb.DuckDBPyConnection, path: Path | str) -> list[str]:
    try:
        rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{sql_path(path)}')").fetchall()
    except duckdb.Error as err:
        raise FieldIdsError(f"Cannot read {Path(path).name} as Parquet: {err}") from err
    return [row[0] for row in rows]


def find_duplicate_ids(path: Path | str, id_col: str = ID_COLUMN) -> DuplicateIds:
    """The ids more than one row of ``path`` shares; scans only the id column.

    Raises:
        FieldIdsError: If the file has no ``id_col`` column.
    """
    con = duckdb.connect(":memory:")
    try:
        if id_col not in _columns(con, path):
            raise FieldIdsError(f"{Path(path).name} has no '{id_col}' column")
        found = con.execute(_duplicates_sql(path, id_col)).fetchall()
    finally:
        con.close()
    return DuplicateIds(ids=tuple(row[0] for row in found), rows=sum(row[1] for row in found))


def require_unique_ids(path: Path | str, id_col: str = ID_COLUMN) -> None:
    """Raise unless every non-missing ``id_col`` value of ``path`` is unique.

    A file without the column has no ids to clash and passes.

    Raises:
        DuplicateIdsError: If some id is shared by more than one row.
    """
    con = duckdb.connect(":memory:")
    try:
        present = id_col in _columns(con, path)
    finally:
        con.close()
    if not present:
        return
    duplicates = find_duplicate_ids(path, id_col)
    if duplicates:
        raise DuplicateIdsError(path, duplicates, id_col)


# --------------------------------------------------------------------------- #
# Resolving one group
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Layout:
    """Which columns play which part, and the preference for same_parcel rows."""

    id_col: str
    geom_col: str
    bbox_col: str | None
    attrs: tuple[str, ...]  # every column but geometry and bbox, in file order
    metrics: tuple[str, ...]
    core: tuple[str, ...]  # attrs that are neither metrics nor the id
    crop_col: str | None
    keep_col: str | None = None
    keep_values: frozenset[str] = frozenset()
    to_area_crs: Any = None  # shapely.transform function from the file's CRS to AREA_CRS


@dataclass
class _Row:
    pos: int  # row number in the source file
    record: dict[str, Any]
    id_text: str
    key: tuple
    shape: Any  # source CRS
    area: float  # m², of the valid geometry in AREA_CRS
    projected: Any


@dataclass
class _Outcome:
    rule: str
    records: list[dict[str, Any]]
    removed: int


def _text(record: dict[str, Any], columns: tuple[str, ...]) -> tuple:
    """``record``'s values in ``columns`` as sortable, hashable text."""
    return tuple((record[c] is None, str(record[c])) for c in columns)


def _content_key(record: dict[str, Any], layout: _Layout) -> tuple:
    """Geometry WKB, then the attributes, then the bbox (only a tie-breaker)."""
    bbox = str(record[layout.bbox_col]) if layout.bbox_col is not None else ""
    return (bytes(record[layout.geom_col] or b""), _text(record, layout.attrs), bbox)


def _close(a: Any, b: Any) -> bool:
    if isinstance(a, int | float) and isinstance(b, int | float):
        if math.isnan(a) or math.isnan(b):
            return math.isnan(a) and math.isnan(b)
        return abs(a - b) <= METRICS_TOLERANCE * max(abs(a), abs(b))
    return a == b


def _same_metrics(a: _Row, b: _Row, layout: _Layout) -> bool:
    return all(_close(a.record[c], b.record[c]) for c in layout.metrics)


def _collapse_exact(rows: list[_Row], layout: _Layout) -> list[_Row]:
    """One row per set of exact duplicates, in content order."""
    buckets: dict[tuple, list[_Row]] = defaultdict(list)
    distinct: list[_Row] = []
    for row in sorted(rows, key=lambda r: r.key):
        bucket = buckets[(row.key[0], _text(row.record, layout.core))]
        if not any(_same_metrics(kept, row, layout) for kept in bucket):
            bucket.append(row)
            distinct.append(row)
    return distinct


def _union_area(rows: list[_Row]) -> float:
    return float(shapely.area(shapely.union_all([r.projected for r in rows])))


def _area_metric_fits(rows: list[_Row], union_area: float, layout: _Layout) -> bool:
    """Whether every row carries the same ``metrics:area`` and the union is near it.

    Without the metric nothing bounds how far apart the parts lie, so a placeholder
    id reused across a region could become one field: such rows are never parts.
    """
    if AREA_METRIC not in layout.metrics:
        return False
    values = [r.record[AREA_METRIC] for r in rows]
    if any(v is None for v in values) or not all(_close(values[0], v) for v in values):
        return False
    low, high = AREA_RATIO_RANGE
    return values[0] > 0 and low <= union_area / values[0] <= high


def _interiors_apart(rows: list[_Row], union_area: float) -> bool:
    """Whether the rows' summed area exceeds their union's by no more than noise."""
    sizable = [r.area for r in rows if r.area >= SLIVER_M2]
    if not sizable:
        return False
    overlap = sum(r.area for r in rows) - union_area
    return overlap < max(OVERLAP_TOLERANCE * min(sizable), SLIVER_M2)


def _same_core(rows: list[_Row], layout: _Layout) -> bool:
    return len({_text(r.record, layout.core) for r in rows}) == 1


def _are_parts(rows: list[_Row], union_area: float, layout: _Layout) -> bool:
    return (
        _same_core(rows, layout)
        and _interiors_apart(rows, union_area)
        and _area_metric_fits(rows, union_area, layout)
    )


def _is_same_polygon(rows: list[_Row], union_area: float) -> bool:
    smallest = min(r.area for r in rows)
    return smallest > 0 and union_area <= (1 + SAME_POLYGON_TOLERANCE) * smallest


def _polygonal(shape: Any) -> Any:
    """``shape`` without the points and lines a union of polygons can pick up."""
    if shape.geom_type != "GeometryCollection":
        return shape
    parts = [g for g in shapely.get_parts(shape) if g.geom_type in ("Polygon", "MultiPolygon")]
    return _union_or(parts, shape)


def _union_or(parts: list, fallback: Any) -> Any:
    return shapely.union_all(parts) if parts else fallback


def _without_slivers(shape: Any, layout: _Layout) -> Any:
    """``shape`` without the pieces under ``SLIVER_M2``, unless that is all of it."""
    pieces = shapely.get_parts(shape)
    areas = shapely.area(shapely.make_valid(shapely.transform(pieces, layout.to_area_crs)))
    kept = [p for p, a in zip(pieces, areas, strict=True) if a >= SLIVER_M2]
    if len(kept) == len(pieces):
        return shape
    return _union_or(kept, shape)


def _bbox(shape: Any) -> dict[str, float]:
    xmin, ymin, xmax, ymax = shape.bounds
    return {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}


def _merged(rows: list[_Row], layout: _Layout) -> dict[str, Any]:
    """One row for the parts: their valid union, slivers left out."""
    parts = shapely.make_valid([r.shape for r in rows if r.area >= SLIVER_M2])
    shape = _without_slivers(_polygonal(shapely.make_valid(shapely.union_all(parts))), layout)
    record = dict(rows[0].record)
    record[layout.geom_col] = shapely.to_wkb(shape)
    if layout.bbox_col is not None:
        record[layout.bbox_col] = _bbox(shape)
    return record


def _passes_filter(row: _Row, layout: _Layout) -> bool:
    if layout.keep_col is None:
        return False
    value = row.record[layout.keep_col]
    return value is not None and str(value) in layout.keep_values


def _one_parcel(rows: list[_Row], layout: _Layout) -> dict[str, Any]:
    kept = min(rows, key=lambda r: (not _passes_filter(r, layout), r.key))
    record = dict(kept.record)
    if layout.crop_col is not None:
        others = {str(r.record[layout.crop_col]) for r in rows if r is not kept}
        others.discard(str(kept.record[layout.crop_col]))
        record[OTHER_CROPS_COLUMN] = sorted(others - {"None"})
    return record


def _renumbered(rows: list[_Row], layout: _Layout) -> list[dict[str, Any]]:
    records = []
    for n, row in enumerate(rows, start=1):
        record = dict(row.record)
        record[layout.id_col] = f"{row.id_text}_{n}"
        record["_renumbered"] = True
        records.append(record)
    return records


def _resolve(rows: list[_Row], layout: _Layout) -> _Outcome:
    """The rule for one group of rows sharing an id, and the rows it leaves."""
    distinct = _collapse_exact(rows, layout)
    if len(distinct) == 1:
        return _Outcome("exact", [distinct[0].record], len(rows) - 1)
    union_area = _union_area(distinct)
    if _are_parts(distinct, union_area, layout):
        return _Outcome("parts", [_merged(distinct, layout)], len(rows) - 1)
    if _is_same_polygon(distinct, union_area):
        return _Outcome("same_parcel", [_one_parcel(distinct, layout)], len(rows) - 1)
    return _Outcome("renumbered", _renumbered(distinct, layout), len(rows) - len(distinct))


# --------------------------------------------------------------------------- #
# Reading the groups and writing the result
# --------------------------------------------------------------------------- #


def _preference(
    columns: list[str], class_filter: ClassFilter | None
) -> tuple[str | None, frozenset[str]]:
    """The class filter's column in these fields and the values it includes."""
    if class_filter is None:
        return None, frozenset()
    return class_filter.resolve_column(columns), frozenset(class_filter.include)


def _crop_column(columns: list[str], keep_col: str | None) -> str | None:
    candidates = (CROP_NAME_COLUMNS[0], keep_col, *CROP_NAME_COLUMNS[1:])
    return next((c for c in candidates if c is not None and c in columns), None)


def _check_reserved(columns: list[str], path: Path) -> None:
    clash = sorted(set(columns) & set(RESERVED_COLUMNS))
    if clash:
        raise FieldIdsError(
            f"{path.name} already has column(s) {', '.join(clash)}, which resolving its "
            "duplicate ids writes; rename them first"
        )


def _bbox_column(columns: list[str], path: Path) -> str | None:
    return get_bbox_column_name(path) or ("bbox" if "bbox" in columns else None)


def _layout(
    columns: list[str], path: Path, id_col: str, class_filter: ClassFilter | None
) -> _Layout:
    _check_reserved(columns, path)
    geom_col = detect_geometry_column(path) or "geometry"
    bbox_col = _bbox_column(columns, path)
    attrs = tuple(c for c in columns if c not in (geom_col, bbox_col))
    metrics = tuple(c for c in attrs if c.startswith(METRICS_PREFIX))
    keep_col, keep_values = _preference(columns, class_filter)
    to_area_crs = reprojector(geoparquet_crs(path, geom_col), AREA_CRS)
    return _Layout(
        id_col=id_col,
        geom_col=geom_col,
        bbox_col=bbox_col,
        attrs=attrs,
        metrics=metrics,
        core=tuple(c for c in attrs if c not in (*metrics, id_col)),
        crop_col=_crop_column(columns, keep_col),
        keep_col=keep_col,
        keep_values=keep_values,
        to_area_crs=to_area_crs,
    )


def _read_groups(
    con: duckdb.DuckDBPyConnection, src: Path, layout: _Layout
) -> tuple[dict[str, list[_Row]], pa.Schema]:
    """Every row whose id is shared, grouped by that id (as text), and their schema."""
    quoted = _quote(layout.id_col)
    table = con.execute(f"""
        SELECT file_row_number AS _pos, CAST({quoted} AS VARCHAR) AS _id_text,
               * EXCLUDE (file_row_number)
        FROM read_parquet('{sql_path(src)}', file_row_number = true)
        WHERE {quoted} IN (SELECT {quoted} FROM dup_ids)""").fetch_arrow_table()
    positions = table.column("_pos").to_pylist()
    id_texts = table.column("_id_text").to_pylist()
    table = table.drop_columns(["_pos", "_id_text"])
    records = table.to_pylist()
    shapes = shapely.from_wkb([r[layout.geom_col] for r in records])
    projected = shapely.make_valid(shapely.transform(shapes, layout.to_area_crs))
    areas = shapely.area(projected)
    groups: dict[str, list[_Row]] = defaultdict(list)
    for i, record in enumerate(records):
        groups[id_texts[i]].append(
            _Row(
                pos=positions[i],
                record=record,
                id_text=id_texts[i],
                key=_content_key(record, layout),
                shape=shapes[i],
                area=float(areas[i]) if math.isfinite(areas[i]) else 0.0,
                projected=projected[i],
            )
        )
    return groups, table.schema


def _resolve_all(
    groups: dict[str, list[_Row]], layout: _Layout, result: DedupResult
) -> list[dict[str, Any]]:
    """Each group's resolved rows, tagged with the position and order they go at."""
    records = []
    for id_text, rows in groups.items():
        outcome = _resolve(rows, layout)
        result.groups[outcome.rule] += 1
        result.rows_removed[outcome.rule] += outcome.removed
        result.changed_ids[outcome.rule].append(id_text)
        first = min(r.pos for r in rows)
        for sub, record in enumerate(outcome.records):
            records.append({**record, "_id_text": id_text, "_pos": first, "_sub": sub})
    for ids in result.changed_ids.values():
        ids.sort()
    return records


def _added_columns(result: DedupResult, layout: _Layout) -> tuple[str, ...]:
    added = []
    if result.groups["renumbered"]:
        added.append(SOURCE_ID_COLUMN)
    if result.groups["same_parcel"] and layout.crop_col is not None:
        added.append(OTHER_CROPS_COLUMN)
    return tuple(added)


_ADDED_TYPES = {SOURCE_ID_COLUMN: pa.string(), OTHER_CROPS_COLUMN: pa.list_(pa.string())}
_ADDED_SQL = {SOURCE_ID_COLUMN: "CAST({id} AS VARCHAR)", OTHER_CROPS_COLUMN: "NULL::VARCHAR[]"}


def _resolved_table(
    records: list[dict[str, Any]], schema: pa.Schema, layout: _Layout, added: tuple[str, ...]
) -> pa.Table:
    """The resolved rows as Arrow, typed like the source plus the added columns."""
    renumbered = SOURCE_ID_COLUMN in added
    fields = [pa.field("_pos", pa.int64()), pa.field("_sub", pa.int64())]
    fields += [
        f.with_type(pa.string()) if renumbered and f.name == layout.id_col else f for f in schema
    ]
    fields += [pa.field(name, _ADDED_TYPES[name]) for name in added]
    if renumbered:
        for record in records:
            record[SOURCE_ID_COLUMN] = record["_id_text"]
            if not isinstance(record[layout.id_col], str):
                record[layout.id_col] = record["_id_text"]
    return pa.Table.from_pylist(records, schema=pa.schema(fields))


def _geo_metadata(src: Path, layout: _Layout, resolved: pa.Table) -> str:
    """The source's ``geo`` metadata, listing any geometry type the merges added."""
    con = duckdb.connect(":memory:")
    try:
        row = con.execute(
            "SELECT value FROM parquet_kv_metadata(?) WHERE key = 'geo'", [str(src)]
        ).fetchone()
    finally:
        con.close()
    if row is None:
        raise FieldIdsError(f"{src.name} is not GeoParquet: it has no 'geo' metadata")
    geo = json.loads(row[0])
    column = geo.get("columns", {}).get(layout.geom_col, {})
    if column.get("encoding", "WKB") != "WKB":
        raise FieldIdsError(f"{src.name}: only WKB geometries are supported")
    if column.get("geometry_types"):
        shapes = shapely.from_wkb(resolved.column(layout.geom_col).to_pylist())
        types = {g.geom_type for g in shapes if g is not None}
        column["geometry_types"] = sorted(set(column["geometry_types"]) | types)
    return json.dumps(geo)


def _write(
    con: duckdb.DuckDBPyConnection,
    src: Path,
    dst: Path,
    layout: _Layout,
    resolved: pa.Table,
    added: tuple[str, ...],
) -> None:
    """Write the rows without a shared id and the resolved ones, in source order.

    DuckDB streams the file through, spilling the sort to disk if it must, and the
    source's ``geo`` metadata is written as it was (CRS included) but for any
    geometry type a merge added.
    """
    quoted = _quote(layout.id_col)
    renumbered = SOURCE_ID_COLUMN in added
    replace = f" REPLACE (CAST({quoted} AS VARCHAR) AS {quoted})" if renumbered else ""
    extra = "".join(f", {_ADDED_SQL[c].format(id=quoted)} AS {_quote(c)}" for c in added)
    columns = ", ".join(_quote(f.name) for f in resolved.schema if f.name not in ("_pos", "_sub"))
    geo = _geo_metadata(src, layout, resolved).replace("'", "''")
    con.register("resolved", resolved)
    con.execute(f"""
        COPY (
            SELECT {columns} FROM (
                SELECT file_row_number AS _pos, 0::BIGINT AS _sub,
                       * EXCLUDE (file_row_number){replace}{extra}
                FROM read_parquet('{sql_path(src)}', file_row_number = true)
                WHERE {quoted} IS NULL OR {quoted} NOT IN (SELECT {quoted} FROM dup_ids)
                UNION ALL BY NAME
                SELECT * FROM resolved
            ) ORDER BY _pos, _sub
        ) TO '{sql_path(dst)}' (FORMAT PARQUET, COMPRESSION zstd, KV_METADATA {{geo: '{geo}'}})""")


def _copy_unchanged(src: Path, dst: Path) -> None:
    if src.resolve() != dst.resolve():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def dedupe_fields(
    src: Path | str,
    dst: Path | str,
    *,
    id_col: str = ID_COLUMN,
    class_filter: ClassFilter | None = None,
) -> DedupResult:
    """Write ``src`` to ``dst`` with every ``id_col`` value unique; see the module doc.

    A file without duplicate ids is copied unchanged, or left alone when ``dst`` is
    ``src``. Otherwise ``dst`` (which may be ``src``) is replaced once fully written.

    Args:
        class_filter: The recipe's class filter. Of rows describing one parcel
            (``same_parcel``), one whose class the filter includes is kept.

    Raises:
        FieldIdsError: If the file has no ``id_col``, or isn't WKB GeoParquet.
        ClassFilterError: If ``class_filter``'s column is not in the file.
    """
    src, dst = Path(src), Path(dst)
    result = DedupResult(rows_in=_row_count(src))
    con = duckdb.connect(":memory:")
    try:
        columns = _columns(con, src)
        if id_col not in columns:
            raise FieldIdsError(f"{src.name} has no '{id_col}' column")
        con.execute(f"CREATE TEMP TABLE dup_ids AS {_duplicates_sql(src, id_col)}")
        if con.execute("SELECT count(*) FROM dup_ids").fetchone()[0] == 0:
            _copy_unchanged(src, dst)
        else:
            _dedupe(con, src, dst, _layout(columns, src, id_col, class_filter), result)
        result.rows_out = result.rows_in - sum(result.rows_removed.values())
    finally:
        con.close()
    logger.info(result.summary())
    return result


def _row_count(path: Path) -> int:
    """From the Parquet footer, so it reads no data."""
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows


def _check_new_ids(
    con: duckdb.DuckDBPyConnection, src: Path, layout: _Layout, records: list[dict[str, Any]]
) -> None:
    """Refuse renumbered ids that another row of the file already has."""
    new = [r[layout.id_col] for r in records if r.get("_renumbered")]
    if not new:
        return
    con.register("new_ids", pa.table({"new_id": pa.array(new, pa.string())}))
    quoted = _quote(layout.id_col)
    taken = con.execute(f"""
        SELECT DISTINCT CAST({quoted} AS VARCHAR) AS taken
        FROM read_parquet('{sql_path(src)}')
        WHERE CAST({quoted} AS VARCHAR) IN (SELECT new_id FROM new_ids)
        ORDER BY taken LIMIT 5""").fetchall()
    if taken:
        examples = ", ".join(row[0] for row in taken)
        raise FieldIdsError(
            f"{src.name}: renumbering rows that share an id would give ids other rows "
            f"already have (e.g. {examples})"
        )


def _dedupe(
    con: duckdb.DuckDBPyConnection, src: Path, dst: Path, layout: _Layout, result: DedupResult
) -> None:
    # A scan buffers row groups per thread: bound them, as on a 192-core host the
    # unbounded default ran out of memory on other full-file scans.
    con.execute(f"SET threads = {_THREADS}")
    groups, schema = _read_groups(con, src, layout)
    records = _resolve_all(groups, layout, result)
    _check_new_ids(con, src, layout, records)
    result.added_columns = _added_columns(result, layout)
    resolved = _resolved_table(records, schema, layout, result.added_columns)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dst.parent, prefix=".ftwd-dedupe-") as work:
        con.execute(f"SET temp_directory = '{sql_path(Path(work) / 'duckdb')}'")
        tmp = Path(work) / "fields.parquet"
        _write(con, src, tmp, layout, resolved, result.added_columns)
        tmp.replace(dst)
