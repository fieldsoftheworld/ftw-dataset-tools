"""Drop the fields the instance mask cannot show.

Some sources hold fields that lie (almost) entirely under other fields: the same parcel
declared twice under two ids, or a sliver nested inside a field. The instance mask
burns fields in one order (:func:`~ftw_dataset_tools.api.masks.instance_paint_order`:
file order, or by id, geometry and crop when the masks have a crop column), and a
later field paints over an earlier one, so such a field gets few or no pixels while its
outline is still burned as boundary lines across the field that shows.

:func:`drop_hidden_fields` drops each field of which at least ``share`` of the area
lies under fields burned after it that are themselves kept, so the field that shows
today stays. Partial overlaps stay. A dropped field leaves the chips' field counts,
coverage and crop shares too, since those stages read the same polygons, and the at
most ``1 - share`` of it that showed goes to whatever lies beneath it.

It runs in three passes, so a source with millions of fields stays cheap: an STRtree
over the fields' bounding boxes finds the fields whose box overlap with later fields
could reach ``share`` of their area; for those, the exact overlap with each later field
is summed (an upper bound of the area covered); only fields that pass both get the
union of the later fields that cover them.
"""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from ftw_dataset_tools.api.config import DEFAULT_HIDDEN_FIELD_SHARE
from ftw_dataset_tools.api.field_ids import _native_geometry_columns
from ftw_dataset_tools.api.geo import detect_geometry_column, ensure_spatial_loaded, sql_path
from ftw_dataset_tools.api.logging_config import get_logger
from ftw_dataset_tools.api.masks import (
    HIDDEN_FIELDS_METADATA_KEY,
    _instance_id_column,
    _total_ram_bytes,
    instance_paint_order,
)

logger = get_logger(__name__)

DEFAULT_SHARE = DEFAULT_HIDDEN_FIELD_SHARE
# Relative slack on area comparisons: a duplicate's computed overlap can come out a
# hair below its area, which would keep it at share 1.
_AREA_TOLERANCE = 1e-9
#: Key of the Parquet key-value metadata recording what was dropped, read by the docs.
METADATA_KEY = HIDDEN_FIELDS_METADATA_KEY
_ID_COLUMN = "id"
_EXAMPLES = 5
# Fields per STRtree query: bounds the candidate pairs held at once.
_CHUNK = 200_000
# Candidate pairs per exact-overlap task; tasks run on a thread pool (GEOS releases
# the GIL), and only one task's geometries are decoded at a time per thread.
_PAIR_CHUNK = 50_000
_ZSTD_LEVEL = 6
# DuckDB's share of RAM for the scan (at least _MIN_MEMORY_MB), on at most _THREADS
# threads: each thread buffers row groups of geometry, so 192 threads ran out of a
# 4 GB limit on a 6.6M-field file before the first row came back.
_MEMORY_FRACTION = 0.25
_MIN_MEMORY_MB = 4096
_THREADS = 8


class HiddenFieldsError(ValueError):
    """Raised when the fields cannot be ranked the way the instance mask burns them."""


@dataclass
class HiddenFieldsResult:
    """What :func:`drop_hidden_fields` dropped."""

    rows_in: int = 0
    dropped: int = 0
    share: float = DEFAULT_SHARE
    crop_column: str | None = None
    dropped_ids: list[str] = field(default_factory=list)  # up to _EXAMPLES

    def summary(self) -> str:
        """One line for the log."""
        line = (
            f"Dropped {self.dropped:,} of {self.rows_in:,} fields with at least "
            f"{self.share:.0%} of their area under fields drawn over them"
        )
        if self.dropped_ids:
            line += f" (e.g. id {', '.join(self.dropped_ids)})"
        return line

    def record(self) -> dict:
        """What the fields file records under :data:`METADATA_KEY`."""
        return {
            "dropped": self.dropped,
            "share": self.share,
            "crop_column": self.crop_column,
            "rows_in": self.rows_in,
        }


def recorded(path: Path | str) -> dict | None:
    """What :func:`drop_hidden_fields` recorded in ``path``, or None if it didn't write it."""
    metadata = pq.read_schema(path).metadata or {}
    raw = metadata.get(METADATA_KEY.encode())
    return json.loads(raw) if raw else None


# --------------------------------------------------------------------------- #
# Ranking the fields
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _Index:
    """Every field's paint rank, bounding box and area (0 when invalid), by row."""

    paint: np.ndarray
    bounds: np.ndarray  # (n, 4)
    area: np.ndarray


def _index(path: Path, geom_col: str, crop_column: str | None) -> _Index:
    """The fields in row order, ranked the way the instance mask burns them.

    DuckDB is held to a share of RAM and a few threads, and spills beside the file.
    """
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".ftwd-hidden-") as spill:
        con = _connect(Path(spill))
        try:
            cols = _rank(con, path, geom_col, crop_column)
        except duckdb.BinderException as err:
            raise HiddenFieldsError(f"Cannot rank the fields of {path.name}: {err}") from err
        finally:
            con.close()
    bounds = np.column_stack([np.asarray(cols[k], dtype=float) for k in ("x0", "y0", "x1", "y1")])
    area = np.nan_to_num(np.asarray(cols["area"], dtype=float))
    return _Index(np.asarray(cols["paint"]), bounds, area)


def _memory_limit_mb() -> int:
    total = _total_ram_bytes()
    if total is None:
        return _MIN_MEMORY_MB
    return max(_MIN_MEMORY_MB, int(_MEMORY_FRACTION * total / 2**20))


def _connect(spill: Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(":memory:")
    con.execute(f"SET memory_limit = '{_memory_limit_mb()}MB'")
    con.execute(f"SET threads = {min(_THREADS, os.cpu_count() or 1)}")
    con.execute(f"SET temp_directory = '{sql_path(spill)}'")
    con.execute("SET enable_progress_bar = false")
    ensure_spatial_loaded(con)
    return con


def _rank(
    con: duckdb.DuckDBPyConnection, path: Path, geom_col: str, crop_column: str | None
) -> dict:
    """Each field's paint rank, bounds and area; file order needs no sort."""
    id_col = _instance_id_column(con, path)
    order = instance_paint_order(id_col, geom_col, crop_column)
    paint = f"row_number() OVER (ORDER BY {order})" if order else "file_row_number"
    g = f'"{geom_col}"'
    return con.execute(
        f"""SELECT {paint} AS paint,
                   coalesce(ST_XMin({g}), 'NaN') AS x0, coalesce(ST_YMin({g}), 'NaN') AS y0,
                   coalesce(ST_XMax({g}), 'NaN') AS x1, coalesce(ST_YMax({g}), 'NaN') AS y1,
                   coalesce(CASE WHEN ST_IsValid({g}) THEN ST_Area({g}) END, 0) AS area
            FROM read_parquet('{sql_path(path)}', file_row_number = true)
            ORDER BY file_row_number"""
    ).fetchnumpy()


# --------------------------------------------------------------------------- #
# Finding the hidden fields
# --------------------------------------------------------------------------- #


def _box_overlap(bounds: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    a, b = bounds[pairs[:, 0]], bounds[pairs[:, 1]]
    width = np.minimum(a[:, 2], b[:, 2]) - np.maximum(a[:, 0], b[:, 0])
    height = np.minimum(a[:, 3], b[:, 3]) - np.maximum(a[:, 1], b[:, 1])
    return np.clip(width, 0, None) * np.clip(height, 0, None)


def _reaching(pairs: np.ndarray, overlap: np.ndarray, area: np.ndarray, share: float) -> np.ndarray:
    """The pairs whose field's summed overlap reaches ``share`` of its area (or it's invalid)."""
    total = np.bincount(pairs[:, 0], weights=overlap, minlength=len(area))
    reach = (total >= share * area * (1 - _AREA_TOLERANCE)) & (total > 0)
    return pairs[reach[pairs[:, 0]] & (overlap > 0)]


def _candidate_pairs(index: _Index, share: float) -> np.ndarray:
    """Pairs of fields whose bounding boxes could hide the first under the later ones."""
    bounds = index.bounds
    valid = np.isfinite(bounds).all(axis=1) & (bounds[:, 0] <= bounds[:, 2])
    rows = np.flatnonzero(valid)
    tree = shapely.STRtree(shapely.box(*bounds[rows].T))
    found = []
    for start in range(0, len(rows), _CHUNK):
        chunk = rows[start : start + _CHUNK]
        b = bounds[chunk]
        under, over = tree.query(shapely.box(b[:, 0], b[:, 1], b[:, 2], b[:, 3]))
        pairs = np.column_stack([chunk[under], rows[over]])
        pairs = pairs[index.paint[pairs[:, 1]] > index.paint[pairs[:, 0]]]
        found.append(_reaching(pairs, _box_overlap(bounds, pairs), index.area, share))
    return np.concatenate(found) if found else np.empty((0, 2), dtype=np.int64)


def _wkbs(path: Path, geom_col: str, rows: np.ndarray) -> dict[int, bytes]:
    """The WKB of ``rows``, by row number: compact until a task decodes it."""
    with tempfile.TemporaryDirectory(dir=path.parent, prefix=".ftwd-hidden-") as spill:
        con = _connect(Path(spill))
        try:
            con.register("wanted", pa.table({"row": pa.array(rows, pa.int64())}))
            table = con.execute(
                f"""SELECT file_row_number AS row, ST_AsWKB("{geom_col}") AS wkb
                    FROM read_parquet('{sql_path(path)}', file_row_number = true)
                    WHERE file_row_number IN (SELECT row FROM wanted)"""
            ).fetch_arrow_table()
        finally:
            con.close()
    return dict(zip(table.column("row").to_pylist(), table.column("wkb").to_pylist(), strict=True))


def _decode(wkbs: dict[int, bytes], rows: np.ndarray) -> dict[int, shapely.Geometry]:
    unique = np.unique(rows)
    shapes = shapely.make_valid(shapely.from_wkb([wkbs[int(r)] for r in unique], on_invalid="fix"))
    return dict(zip(unique.tolist(), shapes, strict=True))


def _overlaps(pairs: np.ndarray, wkbs: dict[int, bytes]) -> tuple[np.ndarray, dict[int, float]]:
    """Each pair's exact overlap, and the area of each pair's first field."""
    shapes = _decode(wkbs, pairs)
    unders = [shapes[int(r)] for r in pairs[:, 0]]
    overs = [shapes[int(r)] for r in pairs[:, 1]]
    overlap = shapely.area(shapely.intersection(unders, overs))
    return overlap, {r: float(shapely.area(shapes[r])) for r in np.unique(pairs[:, 0]).tolist()}


def _exact_pairs(pairs: np.ndarray, wkbs: dict[int, bytes], share: float) -> np.ndarray:
    """The pairs whose field's summed exact overlap still reaches ``share``.

    Pairs are sorted by their first field, so each task holds every pair of its fields.
    """
    pairs = pairs[np.argsort(pairs[:, 0], kind="stable")]
    cuts = _task_cuts(pairs[:, 0])
    tasks = [pairs[a:b] for a, b in pairwise(cuts)]
    with ThreadPoolExecutor(min(_THREADS, os.cpu_count() or 1)) as pool:
        results = list(pool.map(lambda task: _overlaps(task, wkbs), tasks))
    overlap = np.concatenate([o for o, _ in results])
    areas = np.zeros(int(pairs[:, 0].max()) + 1)
    for _, area in results:
        areas[list(area)] = list(area.values())
    return _reaching(pairs, overlap, areas, share)


def _task_cuts(unders: np.ndarray) -> list[int]:
    """Indexes splitting sorted ``unders`` into tasks of about _PAIR_CHUNK pairs."""
    cuts = [0]
    while cuts[-1] < len(unders):
        end = min(cuts[-1] + _PAIR_CHUNK, len(unders))
        if end < len(unders):  # never split one field's pairs across tasks
            end = int(np.searchsorted(unders, unders[end - 1], side="right"))
        cuts.append(end)
    return cuts


def _covered_share(shape: shapely.Geometry, covers: list[shapely.Geometry]) -> float:
    if not covers or shapely.area(shape) <= 0:
        return 0.0
    covered = shapely.intersection(shape, shapely.union_all(covers))
    covered_share = float(shapely.area(covered)) / float(shapely.area(shape))
    return covered_share * (1 + _AREA_TOLERANCE)


def _hidden(
    pairs: np.ndarray, wkbs: dict[int, bytes], paint: np.ndarray, share: float
) -> list[int]:
    """The fields hidden under later fields that are kept, decided last burned first."""
    shapes = _decode(wkbs, pairs)
    later: dict[int, list[int]] = {}
    for under, over in pairs:
        later.setdefault(int(under), []).append(int(over))
    dropped: set[int] = set()
    for row in sorted(later, key=lambda r: paint[r], reverse=True):
        covers = [shapes[o] for o in later[row] if o not in dropped]
        if _covered_share(shapes[row], covers) >= share:
            dropped.add(row)
    return sorted(dropped)


def find_hidden_fields(
    path: Path | str, *, share: float = DEFAULT_SHARE, crop_column: str | None = None
) -> list[int]:
    """The row numbers of the fields in ``path`` the instance mask cannot show.

    ``crop_column`` is the masks' crop column: with one, the masks burn the fields in
    another order (see :func:`~ftw_dataset_tools.api.masks.instance_paint_order`).

    Raises:
        HiddenFieldsError: If the fields can't be ranked (e.g. no ``crop_column``).
    """
    path = Path(path)
    geom_col = detect_geometry_column(path) or "geometry"
    index = _index(path, geom_col, crop_column)
    pairs = _candidate_pairs(index, share)
    if len(pairs) == 0:
        return []
    wkbs = _wkbs(path, geom_col, np.unique(pairs))
    pairs = _exact_pairs(pairs, wkbs, share)
    if len(pairs) == 0:
        return []
    return _hidden(pairs, wkbs, index.paint, share)


# --------------------------------------------------------------------------- #
# Writing the fields that show
# --------------------------------------------------------------------------- #


def drop_hidden_fields(
    src: Path | str,
    dst: Path | str,
    *,
    share: float = DEFAULT_SHARE,
    crop_column: str | None = None,
) -> HiddenFieldsResult:
    """Write ``src`` to ``dst`` without the fields the instance mask cannot show.

    Every other row, its columns and the file metadata are kept in order; ``dst`` may
    be ``src``. ``dst`` records the count under :data:`METADATA_KEY` for the docs.

    Raises:
        HiddenFieldsError: If the fields can't be ranked (e.g. no ``crop_column``).
    """
    src, dst = Path(src), Path(dst)
    hidden = find_hidden_fields(src, share=share, crop_column=crop_column)
    result = HiddenFieldsResult(
        rows_in=pq.ParquetFile(src).metadata.num_rows, share=share, crop_column=crop_column
    )
    result.dropped = len(hidden)
    result.dropped_ids = _ids(src, hidden[:_EXAMPLES])
    tmp = dst.with_name(f".{dst.name}.hidden.tmp")
    try:
        _write_without(src, tmp, np.asarray(hidden, dtype=np.int64), result)
        tmp.replace(dst)
    finally:
        tmp.unlink(missing_ok=True)
    logger.info(result.summary())
    return result


def _ids(path: Path, rows: list[int]) -> list[str]:
    if not rows or _ID_COLUMN not in pq.read_schema(path).names:
        return []
    table = pq.read_table(path, columns=[_ID_COLUMN])
    return [str(table.column(0)[r].as_py()) for r in rows]


def _write_without(src: Path, dst: Path, hidden: np.ndarray, result: HiddenFieldsResult) -> None:
    """Copy ``src`` to ``dst`` one row group at a time, leaving out the ``hidden`` rows."""
    if _native_geometry_columns(src):
        import geoarrow.pyarrow  # noqa: F401  (keeps a Parquet GEOMETRY column native)
    with pq.ParquetFile(src) as source:
        schema = source.schema_arrow
        schema = schema.with_metadata(_metadata(schema.metadata or {}, result))
        start = 0
        with pq.ParquetWriter(
            dst, schema, compression="zstd", compression_level=_ZSTD_LEVEL
        ) as writer:
            for i in range(source.num_row_groups):
                table = source.read_row_group(i)
                keep = np.ones(len(table), dtype=bool)
                lo, hi = np.searchsorted(hidden, [start, start + len(table)])
                keep[hidden[lo:hi] - start] = False
                start += len(table)
                writer.write_table(table.filter(pa.array(keep)).cast(schema))


def _metadata(metadata: dict[bytes, bytes], result: HiddenFieldsResult) -> dict[bytes, bytes]:
    """The source's metadata with the record added and a pandas range index shortened."""
    metadata = {k: v for k, v in metadata.items() if k != b"ARROW:schema"}
    metadata[METADATA_KEY.encode()] = json.dumps(result.record()).encode()
    if b"pandas" in metadata and result.dropped:
        pandas = json.loads(metadata[b"pandas"])
        pandas["index_columns"] = [
            _shortened(index, result.dropped) for index in pandas.get("index_columns", [])
        ]
        metadata[b"pandas"] = json.dumps(pandas).encode()
    return metadata


def _shortened(index: object, dropped: int) -> object:
    if not isinstance(index, dict) or index.get("kind") != "range":
        return index
    return {**index, "stop": index.get("stop", 0) - dropped * index.get("step", 1)}
