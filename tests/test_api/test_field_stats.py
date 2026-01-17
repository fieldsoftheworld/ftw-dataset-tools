"""Tests for the field_stats API."""

from pathlib import Path

import duckdb
import geopandas as gpd
import geoparquet_io as gpio
import pytest
from shapely.geometry import box

from ftw_dataset_tools.api.field_stats import FieldStatsResult


class TestDetectBboxColumn:
    """Tests for detect_bbox_column function."""

    def test_returns_none_without_bbox(self, sample_geoparquet_4326: Path) -> None:
        """Test returns None when no bbox column."""
        from ftw_dataset_tools.api.field_stats import detect_bbox_column

        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL spatial; LOAD spatial;")

        result = detect_bbox_column(conn, sample_geoparquet_4326, "geometry")
        assert result is None
        conn.close()

    def test_returns_bbox_column_name(self, tmp_path: Path) -> None:
        """Test returns bbox column name when present."""
        from ftw_dataset_tools.api.field_stats import detect_bbox_column

        # Create file with bbox
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        path = tmp_path / "with_bbox.parquet"
        gdf.to_parquet(path)
        gpio.read(str(path)).add_bbox().write(str(path))

        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL spatial; LOAD spatial;")

        result = detect_bbox_column(conn, path, "geometry")
        assert result == "bbox"
        conn.close()


class TestBuildCoverageQuery:
    """Tests for _build_coverage_query function."""

    def test_query_with_bbox_optimization(self) -> None:
        """Test query includes bbox conditions when columns provided."""
        from ftw_dataset_tools.api.field_stats import _build_coverage_query

        query = _build_coverage_query(
            grid_geom_col="geometry",
            fields_geom_col="geometry",
            grid_bbox_col="bbox",
            fields_bbox_col="bbox",
            coverage_col="coverage",
        )

        assert "bbox" in query.lower()
        assert "st_intersects" in query.lower()

    def test_query_without_bbox_optimization(self) -> None:
        """Test query works without bbox columns."""
        from ftw_dataset_tools.api.field_stats import _build_coverage_query

        query = _build_coverage_query(
            grid_geom_col="geometry",
            fields_geom_col="geometry",
            grid_bbox_col=None,
            fields_bbox_col=None,
            coverage_col="coverage",
        )

        assert "st_intersects" in query.lower()
        assert "coverage" in query.lower()


class TestFieldStatsResult:
    """Tests for FieldStatsResult dataclass."""

    def test_coverage_percentage_calculation(self) -> None:
        """Test that coverage_percentage is calculated correctly."""
        from pathlib import Path

        result = FieldStatsResult(
            output_path=Path("/tmp/test.parquet"),
            total_cells=100,
            cells_with_coverage=25,
            average_coverage=15.5,
            max_coverage=95.0,
        )
        assert result.coverage_percentage == 25.0

    def test_coverage_percentage_zero_cells(self) -> None:
        """Test that coverage_percentage handles zero total cells."""
        from pathlib import Path

        result = FieldStatsResult(
            output_path=Path("/tmp/test.parquet"),
            total_cells=0,
            cells_with_coverage=0,
            average_coverage=0.0,
            max_coverage=0.0,
        )
        assert result.coverage_percentage == 0.0


class TestAddFieldStats:
    """Tests for add_field_stats function."""

    def test_file_not_found_grid(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test that FileNotFoundError is raised for missing grid file."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        # Create a dummy fields file (must exist to test grid file error)
        fields_file = tmp_path / "fields.parquet"
        fields_file.touch()

        with pytest.raises(FileNotFoundError, match="Grid file not found"):
            add_field_stats(
                grid_file="/nonexistent/grid.parquet",
                fields_file=str(fields_file),
            )

    def test_file_not_found_fields(self, tmp_path: pytest.TempPathFactory) -> None:
        """Test that FileNotFoundError is raised for missing fields file."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        # Create a dummy grid file
        grid_file = tmp_path / "grid.parquet"
        grid_file.touch()

        with pytest.raises(FileNotFoundError, match="Fields file not found"):
            add_field_stats(
                grid_file=str(grid_file),
                fields_file="/nonexistent/fields.parquet",
            )
