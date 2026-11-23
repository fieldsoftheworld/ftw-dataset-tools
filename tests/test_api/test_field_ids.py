"""Tests for making field ids unique."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, ClassVar

import geopandas as gpd
import pyarrow.parquet as pq
import pytest
import shapely
from shapely.geometry import box

from ftw_dataset_tools.api import field_ids
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


class TestSameParcel:
    def test_the_kept_crop_is_not_listed_again(self, tmp_path: Path) -> None:
        rows = [row(1, LEFT, "A", year=2020), row(1, LEFT, "A", year=2021)]
        out, result = dedupe(tmp_path, rows)
        assert result.groups["same_parcel"] == 1
        assert list(out["other_crop_names"].iloc[0]) == []

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

    @pytest.mark.parametrize("column", ["source_id", "other_crop_names", "_pos"])
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
