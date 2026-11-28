"""Tests for repairing invalid field geometries (api/field_geometry.py)."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from shapely.geometry import box

from ftw_dataset_tools.api import field_geometry

if TYPE_CHECKING:
    from pathlib import Path

# Self-intersecting: the ring crosses itself at (5, 5).
BOW_TIE = shapely.from_wkt("POLYGON ((0 0, 10 10, 10 0, 0 10, 0 0))")
# The ring touches itself along a spike that runs into the field and back: its
# boundary draws the spike across the interior.
SPIKE = shapely.from_wkt("POLYGON ((20 0, 30 0, 30 10, 20 10, 20 5, 26 5, 20 5, 20 0))")
# Two overlapping parts: the boundary of each runs inside the other.
OVERLAPPING_PARTS = shapely.from_wkt(
    "MULTIPOLYGON (((40 0, 46 0, 46 6, 40 6, 40 0)), ((44 4, 50 4, 50 10, 44 10, 44 4)))"
)
# A ring with no area at all.
FLAT = shapely.from_wkt("POLYGON ((60 0, 70 0, 65 0, 60 0))")


def _write(tmp_path: Path, geometries: list, *, bbox: bool = True) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "id": [str(i) for i in range(1, len(geometries) + 1)],
            "crop": ["wheat"] * len(geometries),
        },
        geometry=geometries,
        crs="EPSG:4326",
    )
    path = tmp_path / "fields.parquet"
    gdf.to_parquet(path, write_covering_bbox=bbox)
    return path


class TestRepairedShape:
    def test_bow_tie_becomes_its_two_triangles(self) -> None:
        fixed = field_geometry.repaired_shape(BOW_TIE)
        assert fixed.is_valid
        assert fixed.geom_type == "MultiPolygon"
        assert fixed.area == pytest.approx(50.0)

    def test_spike_is_left_out(self) -> None:
        fixed = field_geometry.repaired_shape(SPIKE)
        assert fixed.is_valid
        assert fixed.equals(box(20, 0, 30, 10))
        assert fixed.boundary.length == pytest.approx(40.0)
        assert SPIKE.boundary.length == pytest.approx(52.0)

    def test_overlapping_parts_are_unioned(self) -> None:
        fixed = field_geometry.repaired_shape(OVERLAPPING_PARTS)
        assert fixed.is_valid
        assert fixed.equals(shapely.union_all(list(OVERLAPPING_PARTS.geoms)))

    def test_no_area_left_is_none(self) -> None:
        assert field_geometry.repaired_shape(FLAT) is None


class TestRepairGeometries:
    def test_valid_file_is_not_rewritten(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [box(0, 0, 1, 1), box(2, 0, 3, 1)])
        before = path.read_bytes()
        result = field_geometry.repair_geometries(path)
        assert not result.changed
        assert result.rows_in == 2
        assert path.read_bytes() == before

    def test_repairs_invalid_rows_and_keeps_the_rest(self, tmp_path: Path) -> None:
        valid = box(80, 0, 90, 10)
        path = _write(tmp_path, [valid, BOW_TIE, SPIKE, OVERLAPPING_PARTS])
        source = pq.read_table(path)

        result = field_geometry.repair_geometries(path)

        assert (result.repaired, result.dropped) == (3, 0)
        assert result.repaired_ids == ["2", "3", "4"]
        table = pq.read_table(path)
        assert table.schema == source.schema
        assert table.column("id").to_pylist() == ["1", "2", "3", "4"]
        assert table.column("crop").to_pylist() == source.column("crop").to_pylist()
        shapes = shapely.from_wkb(table.column("geometry").to_pylist())
        assert shapely.is_valid(shapes).all()
        # The valid row is byte-identical; the others are their repairs.
        assert table.column("geometry")[0] == source.column("geometry")[0]
        assert shapes[2].equals(box(20, 0, 30, 10))
        bboxes = table.column("bbox").to_pylist()
        for shape, bbox in zip(shapes, bboxes, strict=True):
            assert (bbox["xmin"], bbox["ymin"], bbox["xmax"], bbox["ymax"]) == shape.bounds

    def test_geo_metadata_lists_new_geometry_types(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [box(80, 0, 90, 10), BOW_TIE])
        geo = json.loads(pq.read_schema(path).metadata[b"geo"])
        assert geo["columns"]["geometry"]["geometry_types"] == ["Polygon"]

        field_geometry.repair_geometries(path)

        geo = json.loads(pq.read_schema(path).metadata[b"geo"])
        assert geo["columns"]["geometry"]["geometry_types"] == ["MultiPolygon", "Polygon"]

    def test_drops_rows_with_no_area_left(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [box(80, 0, 90, 10), FLAT, SPIKE])

        result = field_geometry.repair_geometries(path)

        assert (result.repaired, result.dropped) == (1, 1)
        assert result.dropped_ids == ["2"]
        assert "dropped 1 with no area left" in result.summary()
        # pandas' range index follows the rows left, so geopandas still reads the file.
        gdf = gpd.read_parquet(path)
        assert gdf["id"].tolist() == ["1", "3"]
        assert gdf.is_valid.all()

    def test_bbox_rows_repaired_across_row_groups(self, tmp_path: Path) -> None:
        """Repairs land on the right rows when the file has several row groups."""
        shapes = [box(i * 2, 0, i * 2 + 1, 1) for i in range(10)]
        shapes[7] = shapely.affinity.translate(SPIKE, xoff=100)
        path = tmp_path / "fields.parquet"
        gpd.GeoDataFrame(
            {"id": [str(i) for i in range(10)]}, geometry=shapes, crs="EPSG:4326"
        ).to_parquet(path, write_covering_bbox=True, row_group_size=3)
        assert pq.ParquetFile(path).metadata.num_row_groups == 4

        field_geometry.repair_geometries(path)

        table = pq.read_table(path)
        fixed = shapely.from_wkb(table.column("geometry").to_pylist())
        assert fixed[7].equals(box(120, 0, 130, 10))
        assert shapely.equals(fixed[:7], shapes[:7]).all()
        assert pq.ParquetFile(path).metadata.num_row_groups == 4

    def test_logs_the_summary(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = _write(tmp_path, [BOW_TIE])
        with caplog.at_level("INFO"):
            field_geometry.repair_geometries(path)
        assert "Repaired 1 invalid field geometries (e.g. id 1)" in caplog.text


class TestOutwardBounds:
    def test_float32_bounds_contain_the_geometry(self) -> None:
        values = np.array([0.1, 0.3])
        low = field_geometry._outward(values, pa.float32(), down=True).to_numpy()
        high = field_geometry._outward(values, pa.float32(), down=False).to_numpy()
        assert (low.astype(np.float64) <= values).all()
        assert (high.astype(np.float64) >= values).all()


def _polygon_wkb(coords: list[tuple[float, float]]) -> bytes:
    """A WKB polygon with exactly ``coords`` as its ring, closed or not."""
    import struct

    wkb = struct.pack("<BII", 1, 3, 1) + struct.pack("<I", len(coords))
    return wkb + b"".join(struct.pack("<dd", x, y) for x, y in coords)


class TestUnusualInput:
    def test_unclosed_ring_is_closed_not_a_crash(self, tmp_path: Path) -> None:
        path = tmp_path / "fields.parquet"
        table = pa.table(
            {
                "id": ["1", "2"],
                "geometry": [
                    _polygon_wkb([(0, 0), (1, 0), (1, 1), (0, 1)]),  # unclosed
                    shapely.to_wkb(box(2, 0, 3, 1)),
                ],
            }
        )
        geo = {
            "version": "1.1.0",
            "primary_column": "geometry",
            "columns": {"geometry": {"encoding": "WKB", "geometry_types": ["Polygon"]}},
        }
        pq.write_table(table.replace_schema_metadata({"geo": json.dumps(geo)}), path)

        result = field_geometry.repair_geometries(path)

        assert result.repaired == 1
        shapes = shapely.from_wkb(pq.read_table(path).column("geometry").to_pylist())
        assert shapes[0].equals(box(0, 0, 1, 1))

    def test_null_geometries_pass_through(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [None, BOW_TIE, box(80, 0, 90, 10)])

        result = field_geometry.repair_geometries(path)

        assert (result.repaired, result.dropped) == (1, 0)
        table = pq.read_table(path)
        assert table.column("geometry")[0].as_py() is None
        assert table.column("id").to_pylist() == ["1", "2", "3"]

    def test_float32_bbox_contains_the_repaired_shape(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [shapely.affinity.translate(SPIKE, 0.1, 0.3)])
        table = pq.read_table(path)
        bbox_type = pa.struct([(n, pa.float32()) for n in ("xmin", "ymin", "xmax", "ymax")])
        index = table.schema.get_field_index("bbox")
        table = table.set_column(index, "bbox", table.column("bbox").cast(bbox_type))
        pq.write_table(table, path)

        field_geometry.repair_geometries(path)

        table = pq.read_table(path)
        assert table.schema.field("bbox").type == bbox_type
        bbox = table.column("bbox")[0].as_py()
        xmin, ymin, xmax, ymax = shapely.from_wkb(table.column("geometry")[0].as_py()).bounds
        assert bbox["xmin"] <= xmin and bbox["ymin"] <= ymin
        assert bbox["xmax"] >= xmax and bbox["ymax"] >= ymax

    def test_native_geometry_column_stays_native(self, tmp_path: Path) -> None:
        import duckdb

        path = tmp_path / "native.parquet"
        con = duckdb.connect()
        con.execute("INSTALL spatial; LOAD spatial")
        con.execute(
            f"""COPY (SELECT * FROM (VALUES
                    ('1', ST_GeomFromText('{BOW_TIE.wkt}')),
                    ('2', ST_GeomFromText('POLYGON ((0 0, 1 0, 1 1, 0 1, 0 0))'))
                ) t(id, geometry))
                TO '{path}' (FORMAT parquet, GEOPARQUET_VERSION 'V2')"""
        )
        con.close()

        result = field_geometry.repair_geometries(path)

        assert result.repaired == 1
        column = pq.ParquetFile(path).schema.column(1)
        assert str(column.logical_type).startswith("Geometry")
        assert field_geometry._invalid_rows(path, "geometry") == {}


# The last edge folds back across the first, leaving a ~0.6 m² triangle beside a
# ~1.2 ha field (in degrees near the equator).
FOLDED = shapely.from_wkt("POLYGON ((0 0, 0.001 0, 0.001 0.001, 0 0.001, 0.00001 -0.00001, 0 0))")


class TestSliverParts:
    def test_folded_triangle_is_left_out(self, tmp_path: Path) -> None:
        assert field_geometry.repaired_shape(FOLDED).geom_type == "MultiPolygon"
        path = _write(tmp_path, [FOLDED])

        field_geometry.repair_geometries(path)

        shape = shapely.from_wkb(pq.read_table(path).column("geometry")[0].as_py())
        assert shape.geom_type == "Polygon"
        assert shape.is_valid
        geo = json.loads(pq.read_schema(path).metadata[b"geo"])
        assert geo["columns"]["geometry"]["geometry_types"] == ["Polygon"]

    def test_a_field_of_only_tiny_parts_stays_whole(self) -> None:
        from ftw_dataset_tools.api.chip_grid import reprojector

        tiny = shapely.affinity.scale(BOW_TIE, 1e-6, 1e-6, origin=(0, 0))
        fixed = field_geometry.repaired_shape(tiny, reprojector("EPSG:4326", "EPSG:6933"))
        assert fixed.geom_type == "MultiPolygon"
        assert len(fixed.geoms) == 2


class TestBboxNulls:
    def test_null_bbox_sides_stay_null(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [box(80, 0, 90, 10), BOW_TIE])
        table = pq.read_table(path)
        index = table.schema.get_field_index("bbox")
        bboxes = table.column("bbox").to_pylist()
        bboxes[0] = {**bboxes[0], "xmin": None}
        table = table.set_column(
            index, "bbox", pa.array(bboxes, type=table.schema.field("bbox").type)
        )
        pq.write_table(table, path)

        field_geometry.repair_geometries(path)

        written = pq.read_table(path).column("bbox").to_pylist()
        assert written[0]["xmin"] is None
        assert written[0]["xmax"] == 90
        assert written[1] == {"xmin": 0, "ymin": 0, "xmax": 10, "ymax": 10}


class TestUnreadable:
    def test_counted_apart_from_no_area(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = _write(tmp_path, [BOW_TIE, FLAT])

        def broken(*_args, **_kwargs):
            raise shapely.errors.GEOSException("unreadable")

        monkeypatch.setattr(field_geometry, "repaired_shape", broken)
        result = field_geometry.repair_geometries(path)

        assert (result.unreadable, result.dropped, result.repaired) == (2, 0, 0)
        assert "dropped 2 GEOS could not read" in result.summary()
        assert pq.read_table(path).num_rows == 0


class TestWarnIfInvalid:
    def test_warns_with_the_count(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = _write(tmp_path, [BOW_TIE, SPIKE, box(80, 0, 90, 10)])
        with caplog.at_level("WARNING"):
            assert field_geometry.warn_if_invalid(path) == 2
        assert "has 2 invalid geometries" in caplog.text

    def test_quiet_for_valid_and_non_geo_files(self, tmp_path: Path) -> None:
        assert field_geometry.warn_if_invalid(_write(tmp_path, [box(0, 0, 1, 1)])) == 0
        plain = tmp_path / "plain.parquet"
        pq.write_table(pa.table({"id": [1]}), plain)
        assert field_geometry.warn_if_invalid(plain) == 0
