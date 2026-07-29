"""Tests for the fields Parquet summarizer."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pytest
from shapely.geometry import box

from ftw_dataset_tools.api.field_summary import summarize_fields

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture
def sample_fields(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {
            "id": ["a", "b", "c", "d"],
            "crop": ["wheat", "wheat", "maize", "water"],
            "area": [1.0, 2.0, 3.0, 4.0],
        },
        geometry=[
            box(0, 0, 1, 1),
            box(1, 1, 2, 2),
            box(2, 2, 3, 3),
            box(3, 3, 4, 4),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "fields.parquet"
    gdf.to_parquet(path)
    return path


class TestSummarizeFields:
    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            summarize_fields(tmp_path / "nope.parquet")

    def test_basic_shape(self, sample_fields: Path) -> None:
        s = summarize_fields(sample_fields)
        assert s.num_rows == 4
        assert s.num_columns == 4
        assert s.file_size_bytes > 0
        names = {c.name for c in s.columns}
        assert {"id", "crop", "area", "geometry"} <= names

    def test_categorical_value_counts(self, sample_fields: Path) -> None:
        s = summarize_fields(sample_fields)
        crop = next(c for c in s.columns if c.name == "crop")
        assert crop.kind == "categorical"
        assert crop.distinct == 3
        assert crop.value_counts[0] == ("wheat", 2)  # most common first
        assert not crop.value_counts_truncated

    def test_top_truncation_and_focus(self, sample_fields: Path) -> None:
        s = summarize_fields(sample_fields, top=1)
        crop = next(c for c in s.columns if c.name == "crop")
        assert len(crop.value_counts) == 1
        assert crop.value_counts_truncated

        # Focus overrides the top cap for the chosen column.
        s2 = summarize_fields(sample_fields, top=1, focus_columns=["crop"])
        crop2 = next(c for c in s2.columns if c.name == "crop")
        assert len(crop2.value_counts) == 3
        assert not crop2.value_counts_truncated

    def test_numeric_stats(self, sample_fields: Path) -> None:
        s = summarize_fields(sample_fields)
        area = next(c for c in s.columns if c.name == "area")
        assert area.kind == "numeric"
        assert area.stats["min"] == 1.0
        assert area.stats["max"] == 4.0

    def test_identifier_detection(self, tmp_path: Path) -> None:
        # A high-cardinality unique string column is flagged as an identifier.
        n = 2000
        gdf = gpd.GeoDataFrame(
            {"id": [f"id_{i}" for i in range(n)], "crop": ["wheat"] * n},
            geometry=[box(i, i, i + 1, i + 1) for i in range(n)],
            crs="EPSG:4326",
        )
        path = tmp_path / "many.parquet"
        gdf.to_parquet(path)
        s = summarize_fields(path)
        idcol = next(c for c in s.columns if c.name == "id")
        assert idcol.kind == "identifier"
        assert idcol.value_counts is None

    def test_geometry_and_crs(self, sample_fields: Path) -> None:
        s = summarize_fields(sample_fields)
        assert s.geometry is not None
        assert s.geometry.column == "geometry"
        assert ("POLYGON", 4) in s.geometry.geometry_types
        assert s.geometry.bounds == (0.0, 0.0, 4.0, 4.0)
        assert s.geometry.epsg == 4326

    def test_no_geometry_flag(self, sample_fields: Path) -> None:
        assert summarize_fields(sample_fields, include_geometry=False).geometry is None

    def test_class_filter_candidates(self, sample_fields: Path) -> None:
        s = summarize_fields(sample_fields)
        assert "crop" in s.class_filter_candidates

    def test_identifier_excluded_from_candidates_at_scale(self, tmp_path: Path) -> None:
        n = 2000
        gdf = gpd.GeoDataFrame(
            {"id": [f"id_{i}" for i in range(n)], "crop": ["wheat"] * n},
            geometry=[box(i, i, i + 1, i + 1) for i in range(n)],
            crs="EPSG:4326",
        )
        path = tmp_path / "many.parquet"
        gdf.to_parquet(path)
        s = summarize_fields(path)
        assert "crop" in s.class_filter_candidates
        assert "id" not in s.class_filter_candidates  # identifier, not a candidate
