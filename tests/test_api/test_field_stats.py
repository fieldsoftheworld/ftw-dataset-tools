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


class TestAddFieldStatsWithLocalGrid:
    """Tests for add_field_stats with local grid file."""

    def test_add_field_stats_basic(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test add_field_stats with local grid and fields files."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        output_file = tmp_path / "chips_output.parquet"
        result = add_field_stats(
            grid_file=sample_grid_geoparquet,
            fields_file=sample_fields_geoparquet,
            output_file=output_file,
        )

        assert result.output_path == output_file
        assert result.total_cells == 2  # Two grid cells
        assert output_file.exists()

    def test_add_field_stats_with_progress(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test add_field_stats with progress callback."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        progress_messages: list[str] = []

        def on_progress(msg: str) -> None:
            progress_messages.append(msg)

        output_file = tmp_path / "chips_output.parquet"
        add_field_stats(
            grid_file=sample_grid_geoparquet,
            fields_file=sample_fields_geoparquet,
            output_file=output_file,
            on_progress=on_progress,
        )

        assert len(progress_messages) > 0
        assert any("Loading" in msg for msg in progress_messages)

    def test_add_field_stats_default_output_name(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path
    ) -> None:
        """Test default output filename is chips_<fields_basename>.parquet."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        result = add_field_stats(
            grid_file=sample_grid_geoparquet,
            fields_file=sample_fields_geoparquet,
        )

        expected_name = "chips_fields.parquet"
        assert result.output_path.name == expected_name
        # Cleanup
        if result.output_path.exists():
            result.output_path.unlink()

    def test_add_field_stats_min_coverage_filter(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test min_coverage parameter filters low-coverage cells."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        output_file = tmp_path / "chips_filtered.parquet"
        result = add_field_stats(
            grid_file=sample_grid_geoparquet,
            fields_file=sample_fields_geoparquet,
            output_file=output_file,
            min_coverage=1.0,  # Filter cells with coverage < 1%
        )

        # Some cells may be filtered out
        assert result.output_path.exists()


class TestDetectBboxColumnFallback:
    """Tests for detect_bbox_column schema fallback behavior."""

    def test_detect_bbox_column_from_schema(self, tmp_path: Path) -> None:
        """Test bbox detection from schema when metadata not available."""
        from ftw_dataset_tools.api.field_stats import detect_bbox_column

        # Create file with bbox column structure
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        path = tmp_path / "with_bbox_struct.parquet"
        gdf.to_parquet(path)

        # Add bbox using geoparquet-io
        gpio.read(str(path)).add_bbox().write(str(path))

        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL spatial; LOAD spatial;")

        result = detect_bbox_column(conn, path, "geometry")
        conn.close()

        assert result == "bbox"


class TestCRSMismatchHandling:
    """Tests for CRS mismatch detection and handling."""

    def test_crs_mismatch_error(
        self, sample_geoparquet_3035: Path, sample_grid_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test CRSMismatchError when CRS don't match."""
        from ftw_dataset_tools.api.field_stats import add_field_stats
        from ftw_dataset_tools.api.geo import CRSMismatchError

        output_file = tmp_path / "chips_output.parquet"

        with pytest.raises(CRSMismatchError):
            add_field_stats(
                grid_file=sample_grid_geoparquet,  # EPSG:4326
                fields_file=sample_geoparquet_3035,  # EPSG:3035
                output_file=output_file,
                reproject_to_4326=False,
            )


class TestFieldStatsResultProperties:
    """Additional tests for FieldStatsResult."""

    def test_average_and_max_coverage(self) -> None:
        """Test average_coverage and max_coverage fields."""
        result = FieldStatsResult(
            output_path=Path("/tmp/test.parquet"),
            total_cells=10,
            cells_with_coverage=5,
            average_coverage=45.5,
            max_coverage=95.0,
        )
        assert result.average_coverage == 45.5
        assert result.max_coverage == 95.0
        assert result.coverage_percentage == 50.0
