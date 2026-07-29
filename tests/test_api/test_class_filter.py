"""Tests for the class filter schema, validation, and data-plane helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
import geopandas as gpd
import pytest
import yaml
from shapely.geometry import box

from ftw_dataset_tools.api import class_filter as cf_module
from ftw_dataset_tools.api.config import ClassFilter, ClassFilterError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def fields_with_classes(tmp_path: Path) -> Path:
    """Fields file with a 'crop' column across four classes."""
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3, 4], "crop": ["wheat", "water", "maize", "urban"]},
        geometry=[
            box(10.0, 50.0, 10.01, 50.01),
            box(10.02, 50.0, 10.03, 50.01),
            box(10.0, 50.02, 10.01, 50.03),
            box(10.02, 50.02, 10.03, 50.03),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "fields_crop.parquet"
    gdf.to_parquet(path)
    return path


class TestClassFilterFromFile:
    def _write(self, tmp_path: Path, data: dict) -> Path:
        path = tmp_path / "filter.yaml"
        path.write_text(yaml.safe_dump(data))
        return path

    def test_valid(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"column": "crop", "include": ["wheat"], "exclude": ["water"]})
        cf = ClassFilter.from_file(path)
        assert cf.column == "crop"
        assert cf.include == ["wheat"]
        assert cf.exclude == ["water"]
        assert cf.source == str(path)

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(ClassFilterError, match="not found"):
            ClassFilter.from_file(tmp_path / "nope.yaml")

    def test_missing_column(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"include": ["wheat"]})
        with pytest.raises(ClassFilterError, match="'column'"):
            ClassFilter.from_file(path)

    def test_overlap_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"column": "crop", "include": ["wheat"], "exclude": ["wheat"]})
        with pytest.raises(ClassFilterError, match="both include and exclude"):
            ClassFilter.from_file(path)

    def test_empty_lists_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"column": "crop", "include": [], "exclude": []})
        with pytest.raises(ClassFilterError, match="at least one class"):
            ClassFilter.from_file(path)

    def test_unknown_key_rejected(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"column": "crop", "include": ["wheat"], "bogus": 1})
        with pytest.raises(ClassFilterError, match="Unknown key"):
            ClassFilter.from_file(path)

    def test_numeric_codes_coerced_to_strings(self, tmp_path: Path) -> None:
        path = self._write(tmp_path, {"column": "code", "include": [1, 2], "exclude": [9]})
        cf = ClassFilter.from_file(path)
        assert cf.include == ["1", "2"]
        assert cf.exclude == ["9"]


class TestValidateAgainst:
    def test_full_coverage_passes(self) -> None:
        cf = ClassFilter("crop", ["wheat", "maize"], ["water", "urban"])
        cf.validate_against({"wheat", "maize", "water", "urban"})  # no raise

    def test_unlisted_value_errors(self) -> None:
        cf = ClassFilter("crop", ["wheat"], ["water"])
        with pytest.raises(ClassFilterError, match="not covered"):
            cf.validate_against({"wheat", "water", "rye"})

    def test_null_value_errors_as_placeholder(self) -> None:
        cf = ClassFilter("crop", ["wheat"], ["water"])
        with pytest.raises(ClassFilterError, match="<null>"):
            cf.validate_against({"wheat", "water", None})

    def test_absent_listed_class_warns(self) -> None:
        cf = ClassFilter("crop", ["wheat", "rye"], ["water"])
        msgs: list[str] = []
        cf.validate_against({"wheat", "water"}, on_progress=msgs.append)
        assert any("not present" in m and "rye" in m for m in msgs)


class TestDataPlane:
    def test_get_distinct_classes(self, fields_with_classes: Path) -> None:
        assert cf_module.get_distinct_classes(fields_with_classes, "crop") == {
            "wheat",
            "water",
            "maize",
            "urban",
        }

    def test_missing_column_errors(self, fields_with_classes: Path) -> None:
        with pytest.raises(ClassFilterError, match="not found"):
            cf_module.get_distinct_classes(fields_with_classes, "nope")

    def test_write_filtered_fields_keeps_only_include(
        self, fields_with_classes: Path, tmp_path: Path
    ) -> None:
        cf = ClassFilter("crop", ["wheat", "maize"], ["water", "urban"])
        out = tmp_path / "filtered.parquet"
        cf_module.write_filtered_fields(fields_with_classes, out, cf)
        assert out.exists()
        rows = duckdb.connect().execute(f"SELECT DISTINCT crop FROM '{out}'").fetchall()
        assert {r[0] for r in rows} == {"wheat", "maize"}

    def test_bad_column_name_rejected(self, fields_with_classes: Path) -> None:
        with pytest.raises(ClassFilterError, match="Invalid class filter column"):
            cf_module.get_distinct_classes(fields_with_classes, 'crop"; DROP')
