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
