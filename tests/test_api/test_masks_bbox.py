"""The masks stage's per-chip query uses a file's declared bbox covering to skip row groups."""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import shapely

from ftw_dataset_tools.api import masks
from ftw_dataset_tools.api.geo import ensure_spatial_loaded

if TYPE_CHECKING:
    from pathlib import Path

BOUNDS = (0.0, 0.0, 10.0, 10.0)
POLYGONS = {
    "inside": shapely.box(2, 2, 4, 4),
    "straddles": shapely.box(8, 8, 12, 12),
    "touches": shapely.box(10, 3, 11, 4),
    "outside": shapely.box(20, 20, 22, 22),
    # Its bbox overlaps the query but the shape itself does not.
    "bbox_only": shapely.Polygon([(11, 9), (12, 9), (12, 12), (9, 12), (11, 11)]),
}
COVERING = {
    "bbox": {side: ["bbox", side] for side in ("xmin", "ymin", "xmax", "ymax")}
}


def _write(path: Path, covering: dict | None = COVERING, bbox: dict | None = None) -> Path:
    """A GeoParquet with one polygon per row group, so row-group skipping matters."""
    names = list(POLYGONS)
    geoms = [POLYGONS[n] for n in names]
    bounds = [g.bounds for g in geoms] if bbox is None else [bbox[n] for n in names]
    table = pa.table(
        {
            "id": names,
            "geometry": [shapely.to_wkb(g) for g in geoms],
            "bbox": [dict(zip(("xmin", "ymin", "xmax", "ymax"), b, strict=True)) for b in bounds],
        }
    )
    column = {"encoding": "WKB", "geometry_types": ["Polygon"]}
    if covering is not None:
        column["covering"] = covering
    geo = {"version": "1.1.0", "primary_column": "geometry", "columns": {"geometry": column}}
    table = table.replace_schema_metadata({b"geo": json.dumps(geo).encode()})
    pq.write_table(table, path, row_group_size=1)
    return path


def _ids(path: Path) -> list[str]:
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    rows = masks._get_geometries_in_bounds(conn, path, "geometry", BOUNDS, id_col="id")
    return sorted(r[0] for r in rows)


class TestBboxPrefilter:
    def test_same_rows_with_and_without_covering(self, tmp_path) -> None:
        with_cover = _ids(_write(tmp_path / "a.parquet"))
        without = _ids(_write(tmp_path / "b.parquet", covering=None))
        assert with_cover == without == ["inside", "straddles", "touches"]

    def test_uses_the_declared_columns(self, tmp_path) -> None:
        clause = masks._bbox_prefilter(_write(tmp_path / "a.parquet"), "geometry", BOUNDS)
        assert '"bbox"."xmax" >= 0.0' in clause
        assert '"bbox"."ymin" <= 10.0' in clause
        assert clause.endswith(" AND ")

    def test_no_covering_means_no_prefilter(self, tmp_path) -> None:
        path = _write(tmp_path / "a.parquet", covering=None)
        assert masks._bbox_prefilter(path, "geometry", BOUNDS) == ""

    def test_other_geometry_column_has_no_prefilter(self, tmp_path) -> None:
        path = _write(tmp_path / "a.parquet")
        assert masks._bbox_prefilter(path, "geom", BOUNDS) == ""

    def test_malformed_covering_is_ignored(self, tmp_path) -> None:
        path = _write(tmp_path / "a.parquet", covering={"bbox": {"xmin": ["bbox"]}})
        assert masks._bbox_prefilter(path, "geometry", BOUNDS) == ""
        assert _ids(path) == ["inside", "straddles", "touches"]

    def test_missing_file_has_no_prefilter(self, tmp_path) -> None:
        assert masks._bbox_prefilter(tmp_path / "nope.parquet", "geometry", BOUNDS) == ""

    def test_rewritten_file_is_read_again(self, tmp_path) -> None:
        path = _write(tmp_path / "a.parquet")
        assert masks._bbox_prefilter(path, "geometry", BOUNDS)
        _write(path, covering=None)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        assert masks._bbox_prefilter(path, "geometry", BOUNDS) == ""

    def test_prefilter_is_what_selects_rows(self, tmp_path) -> None:
        """A declared covering is trusted: rows whose bbox misses the query are skipped."""
        far = {n: (50.0, 50.0, 51.0, 51.0) for n in POLYGONS}
        assert _ids(_write(tmp_path / "a.parquet", bbox=far)) == []
        assert _ids(_write(tmp_path / "b.parquet", covering=None, bbox=far)) == [
            "inside",
            "straddles",
            "touches",
        ]
