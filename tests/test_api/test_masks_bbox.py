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
COVERING = {"bbox": {side: ["bbox", side] for side in ("xmin", "ymin", "xmax", "ymax")}}


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


def _stale_copy(tmp_path: Path) -> Path:
    """Like _write, plus a declared ``geometry_bbox`` covering still in another CRS."""
    path = _write(tmp_path / "fresh.parquet")
    table = pq.read_table(path)
    stale = [
        {"xmin": 5e5, "ymin": 5e6, "xmax": 5e5 + 10, "ymax": 5e6 + 10}
        for _ in range(table.num_rows)
    ]
    table = table.append_column("geometry_bbox", pa.array(stale))
    geo = json.loads(table.schema.metadata[b"geo"])
    geo["columns"]["geometry"]["covering"] = {
        "bbox": {side: ["geometry_bbox", side] for side in ("xmin", "ymin", "xmax", "ymax")}
    }
    out = tmp_path / "stale.parquet"
    pq.write_table(
        table.replace_schema_metadata({b"geo": json.dumps(geo).encode()}), out, row_group_size=1
    )
    return out


EXPECTED = ["inside", "straddles", "touches"]


class TestBboxPrefilter:
    def test_same_rows_with_and_without_prefilter(self, tmp_path) -> None:
        path = _write(tmp_path / "a.parquet")
        assert masks._bbox_prefilter(path, "geometry", BOUNDS)
        assert _ids(path) == EXPECTED
        # The same data with no bbox column at all takes the plain ST_Intersects path.
        table = pq.read_table(path).drop_columns(["bbox"])
        geo = json.loads(table.schema.metadata[b"geo"])
        del geo["columns"]["geometry"]["covering"]
        plain = tmp_path / "plain.parquet"
        pq.write_table(table.replace_schema_metadata({b"geo": json.dumps(geo).encode()}), plain)
        assert masks._bbox_prefilter(plain, "geometry", BOUNDS) == ""
        assert _ids(plain) == EXPECTED

    def test_uses_the_declared_columns(self, tmp_path) -> None:
        clause = masks._bbox_prefilter(_write(tmp_path / "a.parquet"), "geometry", BOUNDS)
        assert '"bbox"."xmax" >= -1e-06' in clause
        assert '"bbox"."ymin" <= 10.000011' in clause
        assert clause.endswith(" AND ")

    def test_undeclared_bbox_column_is_used_when_it_matches(self, tmp_path) -> None:
        path = _write(tmp_path / "a.parquet", covering=None)
        assert '"bbox"."xmin"' in masks._bbox_prefilter(path, "geometry", BOUNDS)
        assert _ids(path) == EXPECTED

    def test_stale_declared_covering_falls_back_to_bbox(self, tmp_path) -> None:
        """A covering left in the source CRS by a reprojection must not empty the masks."""
        path = _stale_copy(tmp_path)
        clause = masks._bbox_prefilter(path, "geometry", BOUNDS)
        assert "geometry_bbox" not in clause
        assert '"bbox"."xmin"' in clause
        assert _ids(path) == EXPECTED

    def test_bbox_that_does_not_hold_the_geometry_is_not_trusted(self, tmp_path) -> None:
        far = dict.fromkeys(POLYGONS, (50.0, 50.0, 51.0, 51.0))
        path = _write(tmp_path / "a.parquet", bbox=far)
        assert masks._bbox_prefilter(path, "geometry", BOUNDS) == ""
        assert _ids(path) == EXPECTED

    def test_covering_naming_a_missing_column_is_ignored(self, tmp_path) -> None:
        missing = {"bbox": {side: ["nope", side] for side in ("xmin", "ymin", "xmax", "ymax")}}
        path = _write(tmp_path / "a.parquet", covering=missing)
        assert "nope" not in masks._bbox_prefilter(path, "geometry", BOUNDS)
        assert _ids(path) == EXPECTED

    def test_malformed_covering_is_ignored(self, tmp_path) -> None:
        path = _write(tmp_path / "a.parquet", covering={"bbox": {"xmin": ["bbox"]}})
        assert _ids(path) == EXPECTED

    def test_other_geometry_column_has_no_prefilter(self, tmp_path) -> None:
        path = _write(tmp_path / "a.parquet")
        assert masks._bbox_prefilter(path, "geom", BOUNDS) == ""

    def test_missing_file_has_no_prefilter(self, tmp_path) -> None:
        assert masks._bbox_prefilter(tmp_path / "nope.parquet", "geometry", BOUNDS) == ""

    def test_rewritten_file_is_read_again(self, tmp_path) -> None:
        path = _write(tmp_path / "a.parquet")
        assert masks._bbox_prefilter(path, "geometry", BOUNDS)
        far = dict.fromkeys(POLYGONS, (50.0, 50.0, 51.0, 51.0))
        _write(path, bbox=far)
        stat = path.stat()
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
        assert masks._bbox_prefilter(path, "geometry", BOUNDS) == ""


class TestReprojectedSource:
    """ftwd's own reproject -> default CRS -> boundaries path, from a GDAL-style source."""

    def test_masks_find_fields_after_reprojection(self, tmp_path) -> None:
        import pyproj

        from ftw_dataset_tools.api import boundaries
        from ftw_dataset_tools.api.geo import reproject
        from ftw_dataset_tools.api.pipeline import _drop_declared_crs

        box = shapely.box(500000, 5000000, 500100, 5000100)  # UTM 33N, near 15E 45.1N
        sides = ("xmin", "ymin", "xmax", "ymax")
        table = pa.table(
            {
                "id": ["f1"],
                "geometry": [shapely.to_wkb(box)],
                "geometry_bbox": [dict(zip(sides, box.bounds, strict=True))],
            }
        )
        geo = {
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {
                "geometry": {
                    "encoding": "WKB",
                    "geometry_types": ["Polygon"],
                    "crs": pyproj.CRS("EPSG:32633").to_json_dict(),
                    "covering": {"bbox": {s: ["geometry_bbox", s] for s in sides}},
                }
            },
        }
        source = tmp_path / "source.parquet"
        pq.write_table(table.replace_schema_metadata({b"geo": json.dumps(geo).encode()}), source)
        fields = tmp_path / "fields.parquet"
        reproject(str(source), str(fields), target_crs="EPSG:4326")
        _drop_declared_crs(fields)
        result = boundaries.create_boundaries(
            input_path=str(fields), output_dir=str(tmp_path), output_prefix="bl_"
        )
        lines = result.files_processed[0].output_path

        lon, lat = pyproj.Transformer.from_crs(32633, 4326, always_xy=True).transform(
            500050, 5000050
        )
        bounds = (lon - 0.01, lat - 0.01, lon + 0.01, lat + 0.01)
        conn = duckdb.connect(":memory:")
        ensure_spatial_loaded(conn)
        for path in (fields, lines):
            assert "geometry_bbox" not in masks._bbox_prefilter(path, "geometry", bounds)
            assert len(masks._get_geometries_in_bounds(conn, path, "geometry", bounds)) == 1


def _lonlat_file(path: Path, geoms: list, bboxes: list, row_group_size: int = 1000) -> Path:
    sides = ("xmin", "ymin", "xmax", "ymax")
    table = pa.table(
        {
            "id": [f"f{i}" for i in range(len(geoms))],
            "geometry": [shapely.to_wkb(g) for g in geoms],
            "bbox": pa.array(
                [None if b is None else dict(zip(sides, b, strict=True)) for b in bboxes],
                type=pa.struct([(s, pa.float64()) for s in sides]),
            ),
        }
    )
    geo = {
        "version": "1.1.0",
        "primary_column": "geometry",
        "columns": {
            "geometry": {
                "encoding": "WKB",
                "geometry_types": ["Polygon"],
                "covering": {"bbox": {s: ["bbox", s] for s in sides}},
            }
        },
    }
    pq.write_table(
        table.replace_schema_metadata({b"geo": json.dumps(geo).encode()}),
        path,
        row_group_size=row_group_size,
    )
    return path


def _count(path: Path, bounds) -> int:
    conn = duckdb.connect(":memory:")
    ensure_spatial_loaded(conn)
    return len(masks._get_geometries_in_bounds(conn, path, "geometry", bounds))


class TestBboxEdgeCases:
    def test_bbox_within_tolerance_still_finds_edge_geometry(self, tmp_path) -> None:
        """At 170E the check's slack is ~1.7e-4 deg; the prefilter must allow the same."""
        geom = shapely.box(169.99, 10.0, 170.00005, 10.01)
        short = (169.99, 10.0, 169.9999, 10.01)  # xmax short by 1.5e-4, within the slack
        path = _lonlat_file(tmp_path / "a.parquet", [geom], [short])
        bounds = (170.0, 10.0, 170.02, 10.02)
        assert masks._bbox_prefilter(path, "geometry", bounds)
        assert _count(path, bounds) == 1

    def test_null_bbox_disables_prefilter(self, tmp_path) -> None:
        geoms = [shapely.box(1, 1, 2, 2), shapely.box(3, 3, 4, 4)]
        path = _lonlat_file(tmp_path / "a.parquet", geoms, [geoms[0].bounds, None])
        bounds = (2.5, 2.5, 5.0, 5.0)
        assert masks._bbox_prefilter(path, "geometry", bounds) == ""
        assert _count(path, bounds) == 1

    def test_stale_rows_past_the_sample_disable_prefilter(self, tmp_path) -> None:
        """Projected values anywhere in a lon/lat file are caught from the footer."""
        n = masks._BBOX_CHECK_ROWS + 100
        geoms = [shapely.box(10 + i * 1e-4, 45, 10 + i * 1e-4 + 5e-5, 45.0001) for i in range(n)]
        bboxes = [g.bounds for g in geoms[: masks._BBOX_CHECK_ROWS]]
        bboxes += [(5e5, 5e6, 5e5 + 10, 5e6 + 10)] * 100
        path = _lonlat_file(tmp_path / "a.parquet", geoms, bboxes)
        last = geoms[-1].bounds
        assert masks._bbox_prefilter(path, "geometry", last) == ""
        assert _count(path, last) >= 1

    def test_projected_file_skips_the_range_check(self, tmp_path) -> None:
        """Metres are fine when the geometry's declared CRS is projected."""
        import pyproj

        path = _write(tmp_path / "a.parquet")
        table = pq.read_table(path)
        geo = json.loads(table.schema.metadata[b"geo"])
        geo["columns"]["geometry"]["crs"] = pyproj.CRS("EPSG:32633").to_json_dict()
        big = tmp_path / "utm.parquet"
        pq.write_table(table.replace_schema_metadata({b"geo": json.dumps(geo).encode()}), big)
        assert masks._bbox_prefilter(big, "geometry", BOUNDS)
