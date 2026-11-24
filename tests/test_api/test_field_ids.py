"""Tests for making field ids unique."""

from __future__ import annotations

import json
import random
import subprocess
import sys
from typing import TYPE_CHECKING, ClassVar

import geopandas as gpd
import numpy as np
import pyarrow.parquet as pq
import pytest
import shapely
from shapely.geometry import box

from ftw_dataset_tools.api import field_ids
from ftw_dataset_tools.api.chip_grid import reprojector
from ftw_dataset_tools.api.config import ClassFilter, ClassFilterError
from ftw_dataset_tools.api.field_ids import (
    DuplicateIdsError,
    FieldIdsError,
    dedupe_fields,
    find_duplicate_ids,
    require_unique_ids,
)
from ftw_dataset_tools.api.geo import geoparquet_crs, write_geoparquet

if TYPE_CHECKING:
    from pathlib import Path

LEFT = box(10.000, 50.000, 10.001, 50.001)
RIGHT = box(10.001, 50.000, 10.002, 50.001)  # shares LEFT's east edge
FAR = box(10.010, 50.000, 10.011, 50.001)
OVERLAPPING = box(10.0005, 50.000, 10.0015, 50.001)


def m2(*shapes) -> float:
    """Area of the shapes' union in an equal-area CRS."""
    return float(gpd.GeoSeries([shapely.union_all(shapes)], crs=4326).to_crs(6933).area[0])


def write(path: Path, rows: list[dict], *, crs: int = 4326, bbox: bool = True) -> Path:
    """Write ``rows`` (dicts with a ``geometry``) as GeoParquet."""
    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
    if crs != 4326:
        gdf = gdf.to_crs(crs)
    if bbox:
        write_geoparquet(path, gdf=gdf)
    else:
        gdf.to_parquet(path)
    return path


def row(id_, geometry, crop="wheat", area=None, **extra) -> dict:
    return {"id": id_, "crop:name": crop, "metrics:area": area, **extra, "geometry": geometry}


def dedupe(tmp_path: Path, rows: list[dict], **kwargs) -> tuple[gpd.GeoDataFrame, object]:
    src = write(tmp_path / "in.parquet", rows)
    result = dedupe_fields(src, tmp_path / "out.parquet", **kwargs)
    return gpd.read_parquet(tmp_path / "out.parquet"), result


class TestFindDuplicateIds:
    def test_lists_the_shared_ids(self, tmp_path: Path) -> None:
        path = write(tmp_path / "f.parquet", [row(1, LEFT), row(2, RIGHT), row(2, FAR)])
        found = find_duplicate_ids(path)
        assert found.ids == (2,)
        assert found.rows == 2
        assert found.extra_rows == 1
        assert found

    def test_unique_ids_are_falsy(self, tmp_path: Path) -> None:
        path = write(tmp_path / "f.parquet", [row(1, LEFT), row(2, RIGHT)])
        assert not find_duplicate_ids(path)

    def test_missing_ids_are_not_duplicates(self, tmp_path: Path) -> None:
        path = write(tmp_path / "f.parquet", [row(None, LEFT), row(None, RIGHT)])
        assert not find_duplicate_ids(path)

    def test_missing_column_raises(self, tmp_path: Path) -> None:
        path = write(tmp_path / "f.parquet", [row(1, LEFT)])
        with pytest.raises(FieldIdsError, match="no 'field_id' column"):
            find_duplicate_ids(path, "field_id")

    def test_unreadable_file_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "f.parquet"
        path.write_text("not parquet")
        with pytest.raises(FieldIdsError, match="Cannot read"):
            find_duplicate_ids(path)


class TestRequireUniqueIds:
    def test_duplicates_raise(self, tmp_path: Path) -> None:
        path = write(tmp_path / "f.parquet", [row(7, LEFT), row(7, RIGHT)])
        with pytest.raises(DuplicateIdsError, match=r"1 'id' value\(s\) are shared by 2 rows"):
            require_unique_ids(path)

    def test_unique_ids_pass(self, tmp_path: Path) -> None:
        require_unique_ids(write(tmp_path / "f.parquet", [row(1, LEFT), row(2, RIGHT)]))

    def test_a_file_without_the_column_passes(self, tmp_path: Path) -> None:
        path = write(tmp_path / "f.parquet", [row(1, LEFT), row(1, RIGHT)])
        require_unique_ids(path, "field_id")


class TestExact:
    def test_identical_rows_become_one(self, tmp_path: Path) -> None:
        out, result = dedupe(tmp_path, [row(1, LEFT), row(1, LEFT), row(2, RIGHT)])
        assert out["id"].tolist() == [1, 2]
        assert result.groups["exact"] == 1
        assert result.rows_removed["exact"] == 1
        assert result.changed_ids["exact"] == ["1"]
        assert result.added_columns == ()
        assert (result.rows_in, result.rows_out) == (3, 2)

    def test_metrics_within_a_hundredth_of_a_percent_are_equal(self, tmp_path: Path) -> None:
        out, result = dedupe(tmp_path, [row(1, LEFT, area=1000.0), row(1, LEFT, area=1000.05)])
        assert len(out) == 1
        assert result.groups["exact"] == 1
        assert out["metrics:area"].tolist() == [1000.0]  # the first in content order

    def test_nan_metrics_are_equal(self) -> None:
        assert field_ids._close(float("nan"), float("nan"))
        assert not field_ids._close(float("nan"), 1.0)

    def test_larger_metric_differences_are_not_exact(self, tmp_path: Path) -> None:
        _, result = dedupe(tmp_path, [row(1, LEFT, area=1000.0), row(1, LEFT, area=1010.0)])
        assert result.groups["exact"] == 0
        assert result.groups["same_parcel"] == 1


class TestParts:
    def test_parts_merge_into_one_row(self, tmp_path: Path) -> None:
        area = m2(LEFT, RIGHT)
        out, result = dedupe(
            tmp_path, [row(1, LEFT, area=area), row(2, FAR), row(1, RIGHT, area=area)]
        )
        assert out["id"].tolist() == [1, 2]
        assert result.groups["parts"] == 1
        merged = out.geometry.iloc[0]
        assert merged.is_valid
        assert merged.equals(shapely.union_all([LEFT, RIGHT]))
        assert out["metrics:area"].iloc[0] == area
        bbox = pq.read_table(tmp_path / "out.parquet").column("bbox").to_pylist()[0]
        assert bbox == pytest.approx({"xmin": 10.0, "ymin": 50.0, "xmax": 10.002, "ymax": 50.001})

    def test_disjoint_parts_become_a_multipolygon(self, tmp_path: Path) -> None:
        area = m2(LEFT, FAR)
        out, result = dedupe(tmp_path, [row(1, LEFT, area=area), row(1, FAR, area=area)])
        assert result.groups["parts"] == 1
        assert out.geometry.iloc[0].geom_type == "MultiPolygon"
        geo = pq.read_schema(tmp_path / "out.parquet").metadata[b"geo"].decode()
        assert "MultiPolygon" in geo

    def test_without_an_area_metric_rows_are_renumbered(self, tmp_path: Path) -> None:
        rows = [{"id": 1, "geometry": LEFT}, {"id": 1, "geometry": RIGHT}]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["parts"] == 0
        assert result.groups["renumbered"] == 1
        assert sorted(out["id"]) == ["1_1", "1_2"]

    def test_a_null_area_metric_never_merges_far_apart_rows(self, tmp_path: Path) -> None:
        """A placeholder id reused across a region must not become one MultiPolygon."""
        rows = [row(0, LEFT), row(0, FAR), row(0, box(10.5, 50.5, 10.501, 50.501))]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["renumbered"] == 1
        assert sorted(out["id"]) == ["0_1", "0_2", "0_3"]
        assert set(out.geometry.geom_type) == {"Polygon"}

    def test_overlapping_rows_are_not_parts(self, tmp_path: Path) -> None:
        area = m2(LEFT, OVERLAPPING)
        _, result = dedupe(tmp_path, [row(1, LEFT, area=area), row(1, OVERLAPPING, area=area)])
        assert result.groups["parts"] == 0
        assert result.groups["renumbered"] == 1

    @pytest.mark.parametrize("factor", [0.4, 2.5])
    def test_a_union_far_from_the_area_metric_is_not_parts(
        self, tmp_path: Path, factor: float
    ) -> None:
        area = m2(LEFT, RIGHT) * factor
        _, result = dedupe(tmp_path, [row(1, LEFT, area=area), row(1, RIGHT, area=area)])
        assert result.groups["renumbered"] == 1

    @pytest.mark.parametrize("factor", [0.55, 1.9])
    def test_a_union_within_half_to_twice_the_metric_is_parts(
        self, tmp_path: Path, factor: float
    ) -> None:
        area = m2(LEFT, RIGHT) * factor
        _, result = dedupe(tmp_path, [row(1, LEFT, area=area), row(1, RIGHT, area=area)])
        assert result.groups["parts"] == 1

    def test_rows_with_different_area_metrics_are_not_parts(self, tmp_path: Path) -> None:
        area = m2(LEFT, RIGHT)
        _, result = dedupe(tmp_path, [row(1, LEFT, area=area), row(1, RIGHT, area=area / 2)])
        assert result.groups["renumbered"] == 1

    def test_rows_with_different_attributes_are_not_parts(self, tmp_path: Path) -> None:
        area = m2(LEFT, RIGHT)
        rows = [row(1, LEFT, "wheat", area), row(1, RIGHT, "maize", area)]
        _, result = dedupe(tmp_path, rows)
        assert result.groups["renumbered"] == 1

    def test_areas_are_measured_in_metres_in_a_projected_source(self, tmp_path: Path) -> None:
        area = m2(LEFT, RIGHT)
        src = write(
            tmp_path / "in.parquet", [row(1, LEFT, area=area), row(1, RIGHT, area=area)], crs=3035
        )
        result = dedupe_fields(src, tmp_path / "out.parquet")
        assert result.groups["parts"] == 1
        assert geoparquet_crs(tmp_path / "out.parquet", "geometry").to_epsg() == 3035

    def test_a_group_spread_over_kilometres_is_renumbered(self, tmp_path: Path) -> None:
        """Rows far apart with one small area metric are different fields."""
        spread = [box(10 + k * 0.05, 50.0, 10.001 + k * 0.05, 50.001) for k in range(5)]
        rows = [row(1, shape, area=m2(spread[0])) for shape in spread]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["renumbered"] == 1
        assert sorted(out["id"]) == ["1_1", "1_2", "1_3", "1_4", "1_5"]


# A field of about 10 ha in Austria Lambert (EPSG:31287) with a sliver of about 4e-10 m²
# at one vertex, the shape of at-2023's id 108361087.
_FIELD_31287 = shapely.Point(379200.0, 365500.0).buffer(182.0, quad_segs=64)
_CORNER = _FIELD_31287.exterior.coords[5]
_SLIVER_31287 = shapely.Polygon(
    [
        _CORNER,
        (_CORNER[0] + 0.00144526427, _CORNER[1] - 0.0017034812),
        (_CORNER[0] + 0.0014447445, _CORNER[1] - 0.0017034038),
    ]
)


class TestSlivers:
    @pytest.mark.parametrize("crs", [31287, 4326])
    def test_a_field_with_a_sliver_is_one_polygon_in_either_crs(
        self, tmp_path: Path, crs: int
    ) -> None:
        gdf = gpd.GeoDataFrame(
            {"id": [7, 7], "crop:name": ["w", "w"], "metrics:area": [_FIELD_31287.area] * 2},
            geometry=[_FIELD_31287, _SLIVER_31287],
            crs=31287,
        ).to_crs(crs)
        src = tmp_path / "in.parquet"
        write_geoparquet(src, gdf=gdf)
        result = dedupe_fields(src, tmp_path / "out.parquet")
        assert result.groups["parts"] == 1
        out = gpd.read_parquet(tmp_path / "out.parquet")
        assert out.geometry.iloc[0].geom_type == "Polygon"
        assert out.geometry.iloc[0].equals(gdf.geometry.iloc[0])

    def test_a_sliver_touching_a_part_is_left_out_before_the_union(self, tmp_path: Path) -> None:
        """A sliver sharing an edge would join the field's outline if unioned."""
        sliver = shapely.Polygon([(10.0, 50.0), (10.0005, 50.0), (10.00025, 49.99999999)])
        area = m2(LEFT, RIGHT)
        rows = [row(1, LEFT, area=area), row(1, sliver, area=area), row(1, RIGHT, area=area)]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["parts"] == 1
        assert out.geometry.iloc[0].equals(shapely.union_all([LEFT, RIGHT]))

    def test_a_sliver_piece_of_a_part_is_left_out(self, tmp_path: Path) -> None:
        tiny = box(10.0030, 50.0, 10.0030001, 50.0000001)  # about 0.1 m², apart
        area = m2(LEFT, RIGHT)
        rows = [row(1, shapely.MultiPolygon([LEFT, tiny]), area=area), row(1, RIGHT, area=area)]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["parts"] == 1
        assert out.geometry.iloc[0].geom_type == "Polygon"
        assert out.geometry.iloc[0].equals(shapely.union_all([LEFT, RIGHT]))

    def test_invalid_parts_merge_into_a_valid_field(self, tmp_path: Path) -> None:
        bowtie = shapely.Polygon([(10.0, 50.0), (10.001, 50.001), (10.001, 50.0), (10.0, 50.001)])
        area = m2(shapely.make_valid(bowtie), RIGHT)
        out, result = dedupe(tmp_path, [row(1, bowtie, area=area), row(1, RIGHT, area=area)])
        assert result.groups["parts"] == 1
        merged = out.geometry.iloc[0]
        assert merged.is_valid
        assert merged.geom_type in ("Polygon", "MultiPolygon")

    def test_only_the_polygons_of_a_collection_are_kept(self) -> None:
        mixed = shapely.GeometryCollection(
            [LEFT, shapely.LineString([(0, 0), (1, 1)]), shapely.Point(5, 5)]
        )
        assert field_ids._polygonal(mixed).equals(LEFT)
        assert field_ids._polygonal(LEFT) is LEFT
        lines = shapely.GeometryCollection([shapely.LineString([(0, 0), (1, 1)])])
        assert field_ids._polygonal(lines) is lines

    def test_the_overlap_floor_is_one_square_metre(self) -> None:
        def r(area: float) -> field_ids._Row:
            return field_ids._Row(0, {}, "1", (), None, area, None)

        # 0.5 m² of overlap against a 1e-9 m² part: noise, not an overlap.
        assert field_ids._interiors_apart([r(1000.0), r(1e-9)], 1000.0 - 0.5)
        assert not field_ids._interiors_apart([r(1000.0), r(500.0)], 1000.0 - 0.5 + 500 - 20)
        assert not field_ids._interiors_apart([r(0.5), r(0.2)], 0.0)
        # 0.8 m² against a 50 m² part: above its 1% (0.5 m²) but under the 1 m² floor.
        assert field_ids._interiors_apart([r(1000.0), r(50.0)], 1050.0 - 0.8)
        assert not field_ids._interiors_apart([r(1000.0), r(50.0)], 1050.0 - 1.2)

    def test_a_field_of_slivers_only_is_kept_whole(self) -> None:
        tiny = shapely.MultiPolygon(
            [box(10.0, 50.0, 10.0000001, 50.0000001), box(10.1, 50.0, 10.1000001, 50.0000001)]
        )
        layout = field_ids._Layout(
            id_col="id",
            geom_col="geometry",
            bbox_col=None,
            attrs=(),
            metrics=(),
            core=(),
            crop_col=None,
            to_area_crs=reprojector("EPSG:4326", field_ids.AREA_CRS),
        )
        assert field_ids._without_slivers(tiny, layout) is tiny


class TestSameParcel:
    def test_the_kept_crop_is_not_listed_again(self, tmp_path: Path) -> None:
        rows = [row(1, LEFT, "A", year=2020), row(1, LEFT, "A", year=2021)]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["same_parcel"] == 1
        assert out["other_crop_names"].iloc[0] is None

    def test_one_declaration_is_kept_and_the_other_crops_saved(self, tmp_path: Path) -> None:
        rows = [row(1, LEFT, "B"), row(1, LEFT, "A"), row(1, LEFT, "C"), row(2, RIGHT, "A")]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["same_parcel"] == 1
        assert result.rows_removed["same_parcel"] == 2
        assert out["crop:name"].tolist() == ["A", "A"]
        assert out["id"].tolist() == [1, 2]
        assert [list(v) if v is not None else None for v in out["other_crop_names"]] == [
            ["B", "C"],
            None,
        ]
        assert result.added_columns == (field_ids.OTHER_CROPS_COLUMN,)

    def test_the_class_filter_picks_the_row_kept(self, tmp_path: Path) -> None:
        rows = [row(1, LEFT, "A"), row(1, LEFT, "B")]
        out, _ = dedupe(tmp_path, rows, class_filter=ClassFilter("crop:name", ["B"], ["A"]))
        assert out["crop:name"].tolist() == ["B"]
        assert list(out["other_crop_names"].iloc[0]) == ["A"]

    @pytest.mark.parametrize(
        ("values", "include", "kept"),
        [
            # DuckDB casts float32 0.1 to '0.1'; Python's str() gives 0.10000000149011612.
            (np.array([0.2, 0.1], dtype="float32"), "0.1", 1),
            # DuckDB casts booleans to 'true'/'false', Python's str() to 'True'/'False'.
            ([False, True], "true", 1),
        ],
        ids=["float32", "boolean"],
    )
    def test_the_filter_matches_values_as_its_stage_does(
        self, tmp_path: Path, values, include: str, kept: int
    ) -> None:
        # Content order alone would keep the first row: the filter must pick the second.
        rows = [row(1, LEFT, "A"), row(1, LEFT, "B")]
        gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=4326)
        gdf["cls"] = values
        src = tmp_path / "in.parquet"
        write_geoparquet(src, gdf=gdf)
        cf = ClassFilter("cls", [include], [])
        dedupe_fields(src, tmp_path / "out.parquet", class_filter=cf)
        out = gpd.read_parquet(tmp_path / "out.parquet")
        assert out["crop:name"].tolist() == [rows[kept]["crop:name"]]

    def test_a_missing_crop_name_is_not_listed(self, tmp_path: Path) -> None:
        rows = [row(1, LEFT, "A"), row(1, LEFT, None, year=1), row(1, LEFT, "B", year=2)]
        out, _ = dedupe(tmp_path, rows)
        assert out["crop:name"].tolist() == ["A"]
        assert list(out["other_crop_names"].iloc[0]) == ["B"]

    @pytest.mark.parametrize(
        ("columns", "listed"),
        [
            (("crop:name", "cls", "hcat:name"), ["crop-b"]),
            (("cls", "hcat:name"), ["cls-b"]),
            (("hcat:name",), ["hcat-b"]),
        ],
    )
    def test_the_crop_name_column_is_chosen_in_order(
        self, tmp_path: Path, columns: tuple[str, ...], listed: list[str]
    ) -> None:
        rows = [
            {"id": 1, **{c: f"{c.split(':')[0]}-{x}" for c in columns}, "geometry": LEFT}
            for x in ("a", "b")
        ]
        cf = ClassFilter("cls", [], []) if "cls" in columns else None
        out, _ = dedupe(tmp_path, rows, class_filter=cf)
        assert list(out["other_crop_names"].iloc[0]) == listed

    def test_a_polygon_within_one_percent_is_the_same_parcel(self, tmp_path: Path) -> None:
        wider = box(10.000, 50.000, 10.001005, 50.001)  # 0.5% larger than LEFT
        out, result = dedupe(tmp_path, [row(1, LEFT, "A"), row(1, wider, "B")])
        assert result.groups["same_parcel"] == 1
        assert len(out) == 1

    def test_a_polygon_beyond_one_percent_is_not(self, tmp_path: Path) -> None:
        wider = box(10.000, 50.000, 10.00102, 50.001)  # 2% larger than LEFT
        _, result = dedupe(tmp_path, [row(1, LEFT, "A"), row(1, wider, "B")])
        assert result.groups["renumbered"] == 1

    def test_the_filter_compares_the_text_not_the_value(self, tmp_path: Path) -> None:
        """'1.0' doesn't name the class 1 in the filter stage, so it can't here either."""
        rows = [row(1, LEFT, "A", cls=2), row(1, LEFT, "B", cls=1)]
        out, _ = dedupe(tmp_path, rows, class_filter=ClassFilter("cls", ["1.0"], ["2"]))
        assert out["crop:name"].tolist() == ["A"]  # content order: nothing passes the filter
        out, _ = dedupe(tmp_path, rows, class_filter=ClassFilter("cls", ["1"], ["2"]))
        assert out["crop:name"].tolist() == ["B"]

    def test_rows_without_area_are_not_one_polygon(self, tmp_path: Path) -> None:
        """Zero-area rows at one place are not a parcel: nothing to keep one of."""
        line = shapely.Polygon([(10.0, 50.0), (10.001, 50.0), (10.002, 50.0)])
        _, result = dedupe(tmp_path, [row(1, line, "A"), row(1, line, "B")])
        assert result.groups["renumbered"] == 1

    def test_the_filter_column_alias_is_used(self, tmp_path: Path) -> None:
        rows = [row(1, LEFT, "A", code=1), row(1, LEFT, "B", code=2)]
        cf = ClassFilter("crop_code", ["2"], ["1"], column_aliases=["code"])
        out, _ = dedupe(tmp_path, rows, class_filter=cf)
        assert out["code"].tolist() == [2]

    def test_a_filter_column_missing_from_the_fields_raises(self, tmp_path: Path) -> None:
        rows = [row(1, LEFT, "A"), row(1, LEFT, "B")]
        with pytest.raises(ClassFilterError):
            dedupe(tmp_path, rows, class_filter=ClassFilter("nope", ["A"], []))

    def test_without_a_crop_column_no_list_is_added(self, tmp_path: Path) -> None:
        rows = [
            {"id": 1, "note": "x", "geometry": LEFT},
            {"id": 1, "note": "y", "geometry": LEFT},
        ]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["same_parcel"] == 1
        assert field_ids.OTHER_CROPS_COLUMN not in out.columns


class TestRenumbered:
    def test_renumbered_rows_keep_their_order(self, tmp_path: Path) -> None:
        shapes = [box(10 + k * 0.01, 50.0, 10.001 + k * 0.01, 50.001) for k in range(12)]
        rows = [row(0, LEFT, "first")] + [row(5, s, f"c{k:02d}") for k, s in enumerate(shapes)]
        out, _ = dedupe(tmp_path, rows)
        expected = sorted(shapes, key=lambda g: (g.wkb,))
        assert out["id"].tolist() == ["0", *[f"5_{n}" for n in range(1, 13)]]
        assert [g.wkb for g in out.geometry.iloc[1:]] == [g.wkb for g in expected]

    def test_resolved_rows_are_numbered_within_their_group(self) -> None:
        layout = field_ids._Layout(
            id_col="id",
            geom_col="geometry",
            bbox_col=None,
            attrs=("id", "c"),
            metrics=(),
            core=("c",),
            crop_col=None,
        )
        rows = []
        for pos, (shape, c) in enumerate([(LEFT, "a"), (FAR, "b"), (RIGHT, "c")]):
            record = {"id": 5, "c": c, "geometry": shape.wkb}
            projected = shapely.transform(shape, reprojector("EPSG:4326", field_ids.AREA_CRS))
            rows.append(
                field_ids._Row(
                    pos=pos + 3,
                    record=record,
                    id_text="5",
                    key=field_ids._content_key(record, layout),
                    shape=shape,
                    area=float(projected.area),
                    projected=projected,
                )
            )
        records = field_ids._resolve_all({"5": rows}, layout, field_ids.DedupResult())
        assert [(r["_pos"], r["_sub"], r["id"]) for r in records] == [
            (3, 0, "5_1"),
            (3, 1, "5_2"),
            (3, 2, "5_3"),
        ]

    def test_every_row_is_kept_under_a_new_id(self, tmp_path: Path) -> None:
        rows = [row(5, FAR, "maize"), row(1, RIGHT), row(5, LEFT, "wheat"), row(9, RIGHT)]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["renumbered"] == 1
        assert result.rows_removed["renumbered"] == 0
        assert result.changed_ids["renumbered"] == ["5"]
        # Numbered in WKB order, at the place of the group's first row.
        assert out["id"].tolist() == ["5_1", "5_2", "1", "9"]
        first = min((LEFT, FAR), key=lambda g: g.wkb)
        assert out.geometry.iloc[0].equals(first)
        assert out["source_id"].tolist() == ["5", "5", "1", "9"]
        schema = pq.read_schema(tmp_path / "out.parquet")
        assert str(schema.field("id").type) == "string"
        assert result.added_columns == (field_ids.SOURCE_ID_COLUMN,)

    def test_exact_duplicates_inside_a_group_are_dropped_first(self, tmp_path: Path) -> None:
        rows = [row(5, LEFT, "a"), row(5, FAR, "b"), row(5, LEFT, "a")]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["renumbered"] == 1
        assert result.rows_removed["renumbered"] == 1
        assert out["id"].tolist() == ["5_1", "5_2"]

    def test_missing_ids_are_kept(self, tmp_path: Path) -> None:
        rows = [row(None, LEFT), row("5", LEFT, "a"), row("5", FAR, "b"), row(None, RIGHT)]
        out, _ = dedupe(tmp_path, rows)
        assert out["id"].tolist() == [None, "5_1", "5_2", None]

    def test_a_missing_geometry_is_kept(self, tmp_path: Path) -> None:
        rows = [row(5, None, "a"), row(5, LEFT, "b")]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["renumbered"] == 1
        assert sorted(out["id"]) == ["5_1", "5_2"]
        assert out.geometry.isna().sum() == 1


class TestOutput:
    ROWS: ClassVar[list[dict]] = [
        row(3, FAR, "x"),
        row(1, LEFT, "A"),
        row(7, LEFT, "a"),
        row(1, LEFT, "B"),
        row(2, LEFT, area=m2(LEFT, RIGHT)),
        row(7, FAR, "b"),
        row(2, RIGHT, area=m2(LEFT, RIGHT)),
        row(4, RIGHT, "y"),
        row(4, RIGHT, "y"),
    ]

    def test_resolved_rows_take_their_groups_first_place(self, tmp_path: Path) -> None:
        out, result = dedupe(tmp_path, self.ROWS)
        assert out["id"].tolist() == ["3", "1", "7_1", "7_2", "2", "4"]
        assert result.rows_out == len(out) == 6
        assert result.added_columns == (
            field_ids.SOURCE_ID_COLUMN,
            field_ids.OTHER_CROPS_COLUMN,
        )

    def test_rows_reordered_within_groups_give_the_same_file(self, tmp_path: Path) -> None:
        swapped = list(self.ROWS)
        swapped[1], swapped[3] = swapped[3], swapped[1]
        swapped[2], swapped[5] = swapped[5], swapped[2]
        swapped[4], swapped[6] = swapped[6], swapped[4]
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        first, _ = dedupe(a, self.ROWS)
        second, _ = dedupe(b, swapped)
        assert first.to_wkb().equals(second.to_wkb())

    @pytest.mark.parametrize("seed", [1, 2, 3])
    def test_shuffled_input_gives_the_same_rows(self, tmp_path: Path, seed: int) -> None:
        shuffled = list(self.ROWS)
        random.Random(seed).shuffle(shuffled)
        a = tmp_path / "a"
        b = tmp_path / "b"
        a.mkdir()
        b.mkdir()
        first, _ = dedupe(a, self.ROWS)
        second, _ = dedupe(b, shuffled)

        def by_id(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
            return df.sort_values("id").reset_index(drop=True).to_wkb()

        assert by_id(first).equals(by_id(second))

    def test_in_place(self, tmp_path: Path) -> None:
        src = write(tmp_path / "f.parquet", self.ROWS)
        dedupe_fields(src, src)
        assert not find_duplicate_ids(src)
        assert [p.name for p in tmp_path.iterdir()] == ["f.parquet"]

    def test_the_source_metadata_is_kept(self, tmp_path: Path) -> None:
        """fiboa's collection and pandas' records survive; Arrow's schema is written anew."""
        import pyarrow as pa

        src = write(tmp_path / "geo.parquet", self.ROWS)
        table = pq.read_table(src)
        metadata = {**table.schema.metadata, b"collection": b'{"fiboa_version": "0.2.0"}'}
        pq.write_table(table.replace_schema_metadata(metadata), tmp_path / "in.parquet")
        assert b"ARROW:schema" in pq.ParquetFile(tmp_path / "in.parquet").metadata.metadata

        dedupe_fields(tmp_path / "in.parquet", tmp_path / "out.parquet")
        written = pq.ParquetFile(tmp_path / "out.parquet").metadata.metadata
        collection = json.loads(written[b"collection"])
        assert collection["fiboa_version"] == "0.2.0"
        assert set(collection["schemas:custom"]["properties"]) == {
            "source_id",
            "other_crop_names",
        }
        source_pandas = json.loads(metadata[b"pandas"])
        written_pandas = json.loads(written[b"pandas"])
        assert written_pandas["pandas_version"] == source_pandas["pandas_version"]
        assert b"geo" in written
        source_arrow = pq.ParquetFile(tmp_path / "in.parquet").metadata.metadata[b"ARROW:schema"]
        assert written[b"ARROW:schema"] != source_arrow
        assert "source_id" in pq.read_schema(tmp_path / "out.parquet").names
        assert isinstance(pq.read_table(tmp_path / "out.parquet"), pa.Table)
        assert len(gpd.read_parquet(tmp_path / "out.parquet")) == 6

    def test_the_collection_declares_the_added_columns(self) -> None:
        source = {
            "collection": "cz",
            "schemas:custom": {
                "properties": {"block_id": {"type": "string"}},
                "$schema": "https://vecorel.org/sdl/v0.2.0/schema.json",
                "required": [],
                "collection": {},
            },
        }
        text = field_ids._collection_metadata(json.dumps(source), ("source_id",))
        custom = json.loads(text)["schemas:custom"]
        assert custom["properties"] == {
            "block_id": {"type": "string"},
            "source_id": {"type": "string"},
        }
        assert custom["collection"] == {}
        assert json.loads(field_ids._collection_metadata("{}", ("other_crop_names",)))[
            "schemas:custom"
        ]["properties"] == {"other_crop_names": {"type": "array", "items": {"type": "string"}}}
        assert field_ids._collection_metadata("not json", ("source_id",)) == "not json"
        assert field_ids._collection_metadata("[1]", ("source_id",)) == "[1]"

    @pytest.mark.parametrize("dtype", ["Int64", "UInt32"])
    def test_a_renumbered_nullable_integer_id_still_reads(self, tmp_path: Path, dtype: str) -> None:
        """pandas' record must not keep calling a renumbered (now text) id an integer."""
        import pandas as pd

        gdf = gpd.GeoDataFrame(
            {"id": pd.array([1, 1, 2], dtype=dtype), "crop:name": ["a", "b", "c"]},
            geometry=[LEFT, FAR, RIGHT],
            crs="EPSG:4326",
        )
        gdf.to_parquet(tmp_path / "in.parquet")
        result = dedupe_fields(tmp_path / "in.parquet", tmp_path / "out.parquet")
        assert result.groups["renumbered"] == 1
        out = gpd.read_parquet(tmp_path / "out.parquet")
        assert sorted(out["id"]) == ["1_1", "1_2", "2"]
        assert out["source_id"].tolist() == ["1", "1", "2"]

    @pytest.mark.parametrize("dtype", ["int64", "Int64"])
    def test_an_index_named_like_the_id_keeps_its_entry(self, tmp_path: Path, dtype: str) -> None:
        """set_index("id", drop=False) stores the index as __index_level_0__, named "id"."""
        import pandas as pd

        gdf = gpd.GeoDataFrame(
            {"id": pd.array([1, 1, 2], dtype=dtype), "crop:name": ["a", "b", "c"]},
            geometry=[LEFT, FAR, RIGHT],
            crs="EPSG:4326",
        ).set_index("id", drop=False)
        gdf.to_parquet(tmp_path / "in.parquet")
        dedupe_fields(tmp_path / "in.parquet", tmp_path / "out.parquet")

        def entries(name: str) -> dict:
            meta = pq.ParquetFile(tmp_path / name).metadata.metadata[b"pandas"]
            return {c["field_name"]: c for c in json.loads(meta)["columns"]}

        source, written = entries("in.parquet"), entries("out.parquet")
        index = written["__index_level_0__"]
        assert index["name"] == "id"
        assert (index["pandas_type"], index["numpy_type"]) == (
            source["__index_level_0__"]["pandas_type"],
            source["__index_level_0__"]["numpy_type"],
        )
        assert written["id"]["pandas_type"] == "unicode"
        for read in (pd.read_parquet, gpd.read_parquet):
            out = read(tmp_path / "out.parquet")
            assert out.index.tolist() == [1, 1, 2]
            assert sorted(out["id"]) == ["1_1", "1_2", "2"]

    def test_an_unnamed_index_used_as_the_id_is_retyped(self, tmp_path: Path) -> None:
        import pandas as pd

        gdf = gpd.GeoDataFrame(
            {"crop:name": ["a", "b", "c"]},
            geometry=[LEFT, FAR, RIGHT],
            crs="EPSG:4326",
            index=pd.Index(pd.array([7, 7, 8], dtype="Int64")),
        )
        gdf.to_parquet(tmp_path / "in.parquet")
        dedupe_fields(tmp_path / "in.parquet", tmp_path / "out.parquet", id_col="__index_level_0__")
        out = gpd.read_parquet(tmp_path / "out.parquet")
        assert sorted(out.index) == ["7_1", "7_2", "8"]

    def test_added_columns_are_recorded_for_pandas(self) -> None:
        layout = field_ids._Layout(
            id_col="id",
            geom_col="geometry",
            bbox_col=None,
            attrs=(),
            metrics=(),
            core=(),
            crop_col=None,
        )
        text = field_ids._pandas_metadata(
            {"columns": [], "index_columns": []}, layout, ("source_id", "other_crop_names"), 3
        )
        recorded = {c["name"]: c for c in json.loads(text)["columns"]}
        assert recorded["source_id"]["pandas_type"] == "unicode"
        assert recorded["other_crop_names"]["pandas_type"] == "list[unicode]"
        assert recorded["other_crop_names"]["numpy_type"] == "object"

    def test_a_range_index_spans_the_rows_written(self) -> None:
        index = {"kind": "range", "name": None, "start": 10, "stop": 40, "step": 3}
        assert field_ids._range_index(index, 4) == {**index, "stop": 22}
        assert field_ids._range_index("fid", 4) == "fid"

    def test_nullability_follows_the_source(self, tmp_path: Path) -> None:
        """fiboa's validator rejects a column the source required but the output doesn't."""
        import pyarrow as pa

        src = write(tmp_path / "geo.parquet", self.ROWS)
        table = pq.read_table(src)
        required = {"crop:name", "geometry"}
        schema = pa.schema(
            [f.with_nullable(f.name not in required) for f in table.schema],
            metadata=table.schema.metadata,
        )
        pq.write_table(table.cast(schema), tmp_path / "in.parquet")
        dedupe_fields(tmp_path / "in.parquet", tmp_path / "out.parquet")
        source = pq.read_schema(tmp_path / "in.parquet")
        written = pq.read_schema(tmp_path / "out.parquet")
        for name in source.names:
            assert written.field(name).nullable == source.field(name).nullable, name
        assert written.field("source_id").nullable
        assert written.field("other_crop_names").nullable

    def _with_column(self, tmp_path: Path, name: str, values) -> Path:
        src = write(tmp_path / "geo.parquet", self.ROWS)
        table = pq.read_table(src)
        table = table.append_column(name, values)
        pq.write_table(table, tmp_path / "in.parquet")
        return tmp_path / "in.parquet"

    def test_an_all_null_column_stays_null(self, tmp_path: Path) -> None:
        import pyarrow as pa

        src = self._with_column(tmp_path, "note", pa.nulls(len(self.ROWS)))
        assert pq.read_schema(src).field("note").type == pa.null()
        dedupe_fields(src, tmp_path / "out.parquet")
        written = pq.read_table(tmp_path / "out.parquet")
        assert written.schema.field("note").type == pa.null()
        assert written.column("note").null_count == written.num_rows

    def test_a_type_that_cannot_return_is_a_field_ids_error(self) -> None:
        import pyarrow as pa

        schema = pa.schema([pa.field("x", pa.decimal128(5, 2))])
        with pytest.raises(FieldIdsError, match="source's types"):
            field_ids._conform(pa.table({"x": pa.array([1e9])}), schema)

    def test_an_extension_type_is_written_as_its_storage(self) -> None:
        """A geoarrow.wkb type, which an Arrow schema in the source can declare on a plain
        binary column, would make pyarrow write a native Parquet GEOMETRY column."""
        ga = pytest.importorskip("geoarrow.pyarrow")
        import pyarrow as pa

        source = pa.schema([pa.field("geometry", ga.wkb(), nullable=False)])
        written = pa.field("geometry", pa.binary())
        field = field_ids._source_field(written, source, None, frozenset())
        assert field.type == pa.binary()
        assert not field.nullable
        native = field_ids._source_field(written, source, None, frozenset({"geometry"}))
        assert native.type == ga.wkb()

    def test_a_geoparquet_2_source_keeps_its_native_geometry(self, tmp_path: Path) -> None:
        """GeoParquet 2.0 stores the geometry as a Parquet GEOMETRY column; so must we."""
        gpio = pytest.importorskip("geoparquet_io")
        plain, src = tmp_path / "plain.parquet", tmp_path / "in.parquet"
        gpd.GeoDataFrame(
            {"id": ["1", "1", "2"], "crop:name": ["a", "b", "c"]},
            geometry=[LEFT, FAR, RIGHT],
            crs="EPSG:4326",
        ).to_parquet(plain)
        try:
            gpio.read(str(plain)).write(str(src), geoparquet_version="2.0")
        except Exception as err:  # an older geoparquet-io without 2.0
            pytest.skip(f"geoparquet-io can't write GeoParquet 2.0: {err}")

        def geometry(path: Path) -> tuple[str, str]:
            f = pq.ParquetFile(path)
            column = f.schema.column(f.schema_arrow.get_field_index("geometry"))
            return str(column.logical_type), json.loads(f.metadata.metadata[b"geo"])["version"]

        source_type, source_version = geometry(src)
        assert source_type.startswith("Geometry")
        # A fresh interpreter: writing GeoParquet 2.0 above imported geoarrow.pyarrow,
        # which would otherwise hide a dedupe that relies on someone else importing it.
        code = (
            "import sys; from ftw_dataset_tools.api.field_ids import dedupe_fields; "
            "assert not any(m.startswith('geoarrow.pyarrow') for m in sys.modules); "
            "dedupe_fields(sys.argv[1], sys.argv[2])"
        )
        subprocess.run(
            [sys.executable, "-c", code, str(src), str(tmp_path / "out.parquet")], check=True
        )
        assert geometry(tmp_path / "out.parquet") == (source_type, source_version)
        out = gpd.read_parquet(tmp_path / "out.parquet")
        assert out.crs.to_epsg() == 4326
        assert sorted(out["id"]) == ["1_1", "1_2", "2"]

    def test_a_plain_geometry_column_stays_plain(self, tmp_path: Path) -> None:
        dedupe(tmp_path, self.ROWS)
        source = pq.ParquetFile(tmp_path / "in.parquet")
        written = pq.ParquetFile(tmp_path / "out.parquet")
        i, j = (f.schema_arrow.get_field_index("geometry") for f in (source, written))
        assert str(written.schema.column(j).logical_type) == "None"
        assert str(source.schema.column(i).logical_type) == "None"

    def test_native_geoarrow_geometries_are_refused(self, tmp_path: Path) -> None:
        src = tmp_path / "in.parquet"
        gpd.GeoDataFrame({"id": ["1", "1"]}, geometry=[LEFT, FAR], crs="EPSG:4326").to_parquet(
            src, geometry_encoding="geoarrow"
        )
        with pytest.raises(FieldIdsError, match="only WKB geometries"):
            dedupe_fields(src, tmp_path / "out.parquet")

    def test_the_output_keeps_duckdbs_row_groups(self, tmp_path: Path) -> None:
        """40,000 rows are one DuckDB row group; the final pass must not split them."""
        rows = [row(k, box(k * 1e-4, 0, k * 1e-4 + 5e-5, 5e-5)) for k in range(40_000)]
        rows += [row(0, FAR, "other")]
        out, _ = dedupe(tmp_path, rows)
        written = pq.ParquetFile(tmp_path / "out.parquet")
        assert written.num_row_groups == 1
        assert written.metadata.row_group(0).num_rows == len(out) == 40_001
        assert written.metadata.row_group(0).column(0).compression == "ZSTD"

    def test_large_outputs_keep_row_groups_of_122880(self, tmp_path: Path) -> None:
        """More rows than one DuckDB row group, so smaller batches would show."""
        n = 130_000
        x = np.arange(n) * 1e-4
        gdf = gpd.GeoDataFrame(
            {"id": np.r_[np.arange(n - 1), 0], "crop:name": ["w"] * n},
            geometry=shapely.box(x, 0, x + 5e-5, 5e-5),
            crs="EPSG:4326",
        )
        gdf.to_parquet(tmp_path / "in.parquet")
        dedupe_fields(tmp_path / "in.parquet", tmp_path / "out.parquet")
        written = pq.ParquetFile(tmp_path / "out.parquet").metadata
        sizes = [written.row_group(i).num_rows for i in range(written.num_row_groups)]
        assert sizes == [122_880, n - 122_880]

    def test_the_output_is_zstd_at_level_6(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Parquet doesn't record the level, so the writer's arguments are checked."""
        seen = {}
        writer = pq.ParquetWriter

        def recording(*args, **kwargs):
            seen.update(kwargs)
            return writer(*args, **kwargs)

        monkeypatch.setattr(pq, "ParquetWriter", recording)
        dedupe(tmp_path, self.ROWS)
        assert (seen["compression"], seen["compression_level"]) == ("zstd", 6)

    def test_duckdbs_file_is_deleted_once_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        final_pass = field_ids._with_source_schema
        left: list[bool] = []

        def spy(written, dst, src):
            assert written.path.exists()
            final_pass(written, dst, src)
            left.append(written.path.exists())

        monkeypatch.setattr(field_ids, "_with_source_schema", spy)
        dedupe(tmp_path, self.ROWS)
        assert left == [False]

    def test_column_types_follow_the_source(self, tmp_path: Path) -> None:
        """DuckDB widens millisecond timestamps to microseconds; fiboa wants them in ms."""
        import pyarrow as pa

        src = write(tmp_path / "geo.parquet", self.ROWS)
        table = pq.read_table(src)
        stamps = pa.array([1_700_000_000_000] * table.num_rows, pa.timestamp("ms", tz="UTC"))
        table = table.append_column(pa.field("determination:datetime", stamps.type), stamps)
        pq.write_table(table, tmp_path / "in.parquet")
        dedupe_fields(tmp_path / "in.parquet", tmp_path / "out.parquet")
        written = pq.read_schema(tmp_path / "out.parquet")
        assert written.field("determination:datetime").type == pa.timestamp("ms", tz="UTC")
        assert written.field("id").type == pa.string()

    def test_pandas_reads_the_added_columns_and_index(self, tmp_path: Path) -> None:
        import pandas as pd

        gdf = gpd.GeoDataFrame(
            {"id": [1, 1, 2, 2, 3], "crop:name": ["a", "b", "x", "y", "z"]},
            geometry=[LEFT, LEFT, LEFT, FAR, RIGHT],
            crs="EPSG:4326",
            index=pd.RangeIndex(10, 15, name=None),
        )
        gdf.to_parquet(tmp_path / "in.parquet")
        dedupe_fields(tmp_path / "in.parquet", tmp_path / "out.parquet")
        out = gpd.read_parquet(tmp_path / "out.parquet")
        assert out.index.tolist() == [10, 11, 12, 13]
        assert list(out["other_crop_names"].iloc[0]) == ["b"]
        assert out["source_id"].tolist() == ["1", "2", "2", "3"]
        recorded = json.loads(pq.ParquetFile(tmp_path / "out.parquet").metadata.metadata[b"pandas"])
        names = [c["name"] for c in recorded["columns"]]
        assert names[-2:] == ["source_id", "other_crop_names"]

    def test_a_large_renumbered_group_is_written_in_order(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The write sorts on (_pos, _sub), not on the order the resolved rows arrive in."""
        resolved_table = field_ids._resolved_table

        def reversed_rows(*args, **kwargs):
            table = resolved_table(*args, **kwargs)
            return table.take(list(range(table.num_rows - 1, -1, -1)))

        monkeypatch.setattr(field_ids, "_resolved_table", reversed_rows)
        shapes = [box(10 + k * 0.002, 50.0, 10.001 + k * 0.002, 50.001) for k in range(3000)]
        rows = [row(0, LEFT, "first")]
        rows += [row(5, s, f"c{k:04d}") for k, s in enumerate(shapes)]
        rows += [row(9, RIGHT, "last")]
        out, _ = dedupe(tmp_path, rows)
        assert out["id"].tolist() == ["0", *[f"5_{n}" for n in range(1, 3001)], "9"]
        expected = sorted(shapes, key=lambda g: g.wkb)
        assert [g.wkb for g in out.geometry.iloc[1:-1]] == [g.wkb for g in expected]

    def test_logs_a_summary(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        dedupe(tmp_path, self.ROWS)
        assert (
            "Resolved duplicate field ids: 1 exact (-1 rows), 1 parts (-1 rows), "
            "1 same_parcel (-1 rows), 1 renumbered (-0 rows); 9 -> 6 fields"
        ) in caplog.messages


class TestNoDuplicates:
    ROWS: ClassVar[list[dict]] = [row(1, LEFT), row(2, RIGHT), row(3, FAR)]

    def test_the_file_is_copied_unchanged(self, tmp_path: Path) -> None:
        src = write(tmp_path / "in.parquet", self.ROWS)
        result = dedupe_fields(src, tmp_path / "out.parquet")
        assert (tmp_path / "out.parquet").read_bytes() == src.read_bytes()
        assert not result.changed
        assert (result.rows_in, result.rows_out) == (3, 3)

    def test_a_symlinked_output_is_replaced_not_written_through(self, tmp_path: Path) -> None:
        src = write(tmp_path / "in.parquet", self.ROWS)
        other = tmp_path / "other.parquet"
        other.write_bytes(b"keep me")
        (tmp_path / "link.parquet").symlink_to(other)
        dedupe_fields(src, tmp_path / "link.parquet")
        assert other.read_bytes() == b"keep me"
        assert not (tmp_path / "link.parquet").is_symlink()
        assert (tmp_path / "link.parquet").read_bytes() == src.read_bytes()

    def test_a_hardlinked_output_is_the_input(self, tmp_path: Path) -> None:
        src = write(tmp_path / "in.parquet", self.ROWS)
        (tmp_path / "link.parquet").hardlink_to(src)
        before = src.read_bytes()
        dedupe_fields(src, tmp_path / "link.parquet")
        assert src.read_bytes() == before
        assert (tmp_path / "link.parquet").samefile(src)

    def test_in_place_leaves_the_file_alone(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = write(tmp_path / "in.parquet", self.ROWS)
        before = src.stat().st_mtime_ns, src.read_bytes()

        def no_group_read(*_args, **_kwargs):
            raise AssertionError("only the id column should be read")

        monkeypatch.setattr(field_ids, "_read_groups", no_group_read)
        dedupe_fields(src, src)
        assert (src.stat().st_mtime_ns, src.read_bytes()) == before


def test_duckdb_is_bounded(tmp_path: Path) -> None:
    con = field_ids._connect(tmp_path)
    try:
        settings = dict(
            con.execute(
                "SELECT name, value FROM duckdb_settings() "
                "WHERE name IN ('memory_limit', 'threads', 'temp_directory', "
                "'preserve_insertion_order')"
            ).fetchall()
        )
    finally:
        con.close()
    assert settings["memory_limit"] == "3.7 GiB"
    assert settings["threads"] == "4"
    assert settings["preserve_insertion_order"] == "false"
    assert settings["temp_directory"] == str(tmp_path / "duckdb")


class TestErrors:
    def test_missing_id_column(self, tmp_path: Path) -> None:
        src = write(tmp_path / "in.parquet", [row(1, LEFT)])
        with pytest.raises(FieldIdsError, match="no 'fid' column"):
            dedupe_fields(src, tmp_path / "out.parquet", id_col="fid")

    def test_a_renumbered_id_that_is_taken_raises(self, tmp_path: Path) -> None:
        rows = [row("A", LEFT, "x"), row("A", FAR, "y"), row("A_1", RIGHT)]
        src = write(tmp_path / "in.parquet", rows)
        with pytest.raises(FieldIdsError, match=r"already have \(e.g. A_1\)"):
            dedupe_fields(src, tmp_path / "out.parquet")
        assert not (tmp_path / "out.parquet").exists()

    @pytest.mark.parametrize("column", ["source_id", "other_crop_names", "_pos", "file_row_number"])
    def test_a_column_the_rewrite_writes_is_refused(self, tmp_path: Path, column: str) -> None:
        rows = [row(1, LEFT, **{column: "x"}), row(1, LEFT, **{column: "x"})]
        src = write(tmp_path / "in.parquet", rows)
        with pytest.raises(FieldIdsError, match=f"already has column\\(s\\) {column}"):
            dedupe_fields(src, tmp_path / "out.parquet")

    def test_reserved_columns_are_fine_without_duplicates(self, tmp_path: Path) -> None:
        src = write(tmp_path / "in.parquet", [row(1, LEFT, source_id="x"), row(2, RIGHT)])
        dedupe_fields(src, tmp_path / "out.parquet")
        assert (tmp_path / "out.parquet").read_bytes() == src.read_bytes()

    def test_plain_parquet_with_duplicates(self, tmp_path: Path) -> None:
        import pyarrow as pa

        src = tmp_path / "in.parquet"
        pq.write_table(
            pa.table({"id": [1, 1], "geometry": [LEFT.wkb, RIGHT.wkb]}),
            src,
        )
        with pytest.raises(FieldIdsError, match="no 'geo' metadata"):
            dedupe_fields(src, tmp_path / "out.parquet")


class TestStripFloatSuffix:
    def _ids(self, path: Path) -> list:
        return pq.read_table(path, columns=["id"]).column("id").to_pylist()

    def test_whole_number_floats_lose_the_fraction(self, tmp_path: Path) -> None:
        src = write(
            tmp_path / "in.parquet",
            [row("432033.0", LEFT), row("105448990.00", RIGHT), row(None, FAR)],
        )
        before = pq.ParquetFile(src).schema_arrow.metadata[b"geo"]

        assert field_ids.strip_float_suffix(src) == 2

        assert self._ids(src) == ["432033", "105448990", None]
        assert pq.ParquetFile(src).schema_arrow.metadata[b"geo"] == before
        frame = gpd.read_parquet(src)
        assert frame.geometry.iloc[0].equals(LEFT)
        assert frame["crop:name"].tolist() == ["wheat", "wheat", "wheat"]
        assert not list(tmp_path.glob(".*.tmp"))

    @pytest.mark.parametrize(
        "ids",
        [
            ["432033.0", "432034"],  # mixed: not every id is a float
            ["432033.5", "432034.0"],  # a real fraction
            ["5_1.0", "6.0"],  # not a plain number
            ["7", "8"],  # already plain
        ],
    )
    def test_other_text_ids_are_left_alone(self, tmp_path: Path, ids: list[str]) -> None:
        src = write(tmp_path / "in.parquet", [row(ids[0], LEFT), row(ids[1], RIGHT)])
        before = src.read_bytes()

        assert field_ids.strip_float_suffix(src) == 0

        assert src.read_bytes() == before

    def test_integer_ids_and_files_without_ids_are_left_alone(self, tmp_path: Path) -> None:
        ints = write(tmp_path / "ints.parquet", [row(1, LEFT), row(2, RIGHT)])
        no_ids = tmp_path / "no_ids.parquet"
        write_geoparquet(no_ids, gdf=gpd.GeoDataFrame(geometry=[LEFT], crs=4326))
        empty = write(tmp_path / "empty.parquet", [row(None, LEFT)])
        sizes = {p: p.read_bytes() for p in (ints, no_ids, empty)}

        for path in sizes:
            assert field_ids.strip_float_suffix(path) == 0
            assert path.read_bytes() == sizes[path]
