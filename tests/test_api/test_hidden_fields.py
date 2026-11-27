"""Tests for dropping the fields the instance mask cannot show (api/hidden_fields.py)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box

from ftw_dataset_tools.api import hidden_fields

if TYPE_CHECKING:
    from pathlib import Path

HOST = box(0, 0, 10, 10)
SLIVER = box(4, 4, 4.5, 4.2)  # inside HOST


def _write(tmp_path: Path, rows: list[tuple[str, object]], **columns: list) -> Path:
    path = tmp_path / "fields.parquet"
    gpd.GeoDataFrame(
        {"id": [fid for fid, _ in rows], **columns},
        geometry=[shape for _, shape in rows],
        crs="EPSG:4326",
    ).to_parquet(path, write_covering_bbox=True)
    return path


def _ids(path: Path) -> list[str]:
    return pq.read_table(path, columns=["id"]).column("id").to_pylist()


class TestFindHiddenFields:
    def test_duplicate_pair_keeps_the_one_burned_last(self, tmp_path: Path) -> None:
        # File order: the second copy paints over the first, so the first can't show.
        path = _write(tmp_path, [("a", HOST), ("b", HOST), ("c", box(20, 0, 30, 10))])
        assert hidden_fields.find_hidden_fields(path) == [0]

    def test_exact_duplicate_group_keeps_only_the_last(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [("a", HOST), ("b", HOST), ("c", HOST)])
        assert hidden_fields.find_hidden_fields(path) == [0, 1]

    def test_with_a_crop_column_the_masks_order_decides(self, tmp_path: Path) -> None:
        """With a crop column the masks burn by id, so the higher id shows."""
        path = _write(tmp_path, [("b", HOST), ("a", HOST)], crop=["wheat", "maize"])
        assert hidden_fields.find_hidden_fields(path) == [0]  # file order: "a" shows
        assert hidden_fields.find_hidden_fields(path, crop_column="crop") == [1]
        path = _write(tmp_path, [("a", HOST), ("b", HOST)], crop=["wheat", "maize"])
        assert hidden_fields.find_hidden_fields(path, crop_column="crop") == [0]

    def test_nested_sliver_under_its_host_is_dropped(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [("sliver", SLIVER), ("host", HOST)])
        assert hidden_fields.find_hidden_fields(path) == [0]

    def test_sliver_burned_over_its_host_shows_and_stays(self, tmp_path: Path) -> None:
        """The host is only partly covered; the sliver paints over it and shows."""
        path = _write(tmp_path, [("host", HOST), ("sliver", SLIVER)])
        assert hidden_fields.find_hidden_fields(path) == []

    def test_partial_overlap_stays(self, tmp_path: Path) -> None:
        # 40% of the first field lies under the second.
        path = _write(tmp_path, [("a", HOST), ("b", box(6, 0, 16, 10))])
        assert hidden_fields.find_hidden_fields(path) == []

    def test_covered_by_several_later_fields_together(self, tmp_path: Path) -> None:
        path = _write(
            tmp_path, [("big", HOST), ("west", box(0, 0, 5, 10)), ("east", box(5, 0, 10, 10))]
        )
        assert hidden_fields.find_hidden_fields(path) == [0]

    def test_share_is_the_threshold(self, tmp_path: Path) -> None:
        # 90% of the first field lies under the second.
        path = _write(tmp_path, [("a", HOST), ("b", box(1, 0, 11, 10))])
        assert hidden_fields.find_hidden_fields(path) == []
        assert hidden_fields.find_hidden_fields(path, share=0.9) == [0]

    def test_a_dropped_field_hides_nothing(self, tmp_path: Path) -> None:
        """Only fields that are kept and burned later count as cover."""
        # "b" is hidden under "c" (96%), so it is dropped; "a" lay only under the part
        # of "b" that "c" leaves uncovered, so "a" shows once "b" is gone.
        path = _write(
            tmp_path,
            [("a", box(1, 9.7, 2, 9.9)), ("b", HOST), ("c", box(0, 0, 10, 9.6))],
        )
        assert hidden_fields.find_hidden_fields(path) == [1]

    def test_touching_fields_stay(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [("a", HOST), ("b", box(10, 0, 20, 10)), ("c", box(0, 10, 10, 20))])
        assert hidden_fields.find_hidden_fields(path) == []

    def test_null_geometries_are_ignored(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [("a", None), ("b", HOST), ("c", HOST)])
        assert hidden_fields.find_hidden_fields(path) == [1]

    def test_unknown_crop_column_is_an_error(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [("a", HOST), ("b", HOST)])
        with pytest.raises(hidden_fields.HiddenFieldsError, match="Cannot rank"):
            hidden_fields.find_hidden_fields(path, crop_column="crop")


class TestDropHiddenFields:
    def test_writes_the_rest_in_order_and_records_the_count(self, tmp_path: Path) -> None:
        src = _write(
            tmp_path,
            [("sliver", SLIVER), ("a", HOST), ("b", HOST), ("far", box(20, 0, 30, 10))],
            crop=["x", "y", "z", "w"],
        )
        dst = tmp_path / "out.parquet"

        result = hidden_fields.drop_hidden_fields(src, dst)

        assert (result.rows_in, result.dropped) == (4, 2)
        assert result.dropped_ids == ["sliver", "a"]
        assert _ids(dst) == ["b", "far"]
        assert pq.read_schema(dst).metadata[b"geo"] == pq.read_schema(src).metadata[b"geo"]
        assert hidden_fields.recorded(dst) == {"dropped": 2, "share": 0.95, "rows_in": 4}
        assert hidden_fields.recorded(src) is None
        # pandas' range index follows the rows left.
        assert gpd.read_parquet(dst)["crop"].tolist() == ["z", "w"]

    def test_in_place_and_nothing_hidden(self, tmp_path: Path) -> None:
        path = _write(tmp_path, [("a", HOST), ("b", box(20, 0, 30, 10))])
        result = hidden_fields.drop_hidden_fields(path, path)
        assert result.dropped == 0
        assert _ids(path) == ["a", "b"]
        assert hidden_fields.recorded(path) == {"dropped": 0, "share": 0.95, "rows_in": 2}

    def test_logs_the_summary(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = _write(tmp_path, [("a", HOST), ("b", HOST)])
        with caplog.at_level("INFO"):
            hidden_fields.drop_hidden_fields(path, tmp_path / "out.parquet")
        assert "Dropped 1 of 2 fields with at least 95% of their area" in caplog.text
