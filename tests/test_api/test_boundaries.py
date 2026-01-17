"""Tests for the boundaries API module."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box


class TestBoundaryResult:
    """Tests for BoundaryResult dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test BoundaryResult has expected fields."""
        from ftw_dataset_tools.api.boundaries import BoundaryResult

        result = BoundaryResult(
            input_path=Path("/tmp/input.parquet"),
            output_path=Path("/tmp/output.parquet"),
            feature_count=100,
        )
        assert result.input_path == Path("/tmp/input.parquet")
        assert result.output_path == Path("/tmp/output.parquet")
        assert result.feature_count == 100


class TestCreateBoundariesResult:
    """Tests for CreateBoundariesResult dataclass."""

    def test_total_processed(self) -> None:
        """Test total_processed property."""
        from ftw_dataset_tools.api.boundaries import (
            BoundaryResult,
            CreateBoundariesResult,
        )

        result = CreateBoundariesResult(
            files_processed=[
                BoundaryResult(Path("a"), Path("b"), 10),
                BoundaryResult(Path("c"), Path("d"), 20),
            ],
            files_skipped=[],
        )
        assert result.total_processed == 2

    def test_total_skipped(self) -> None:
        """Test total_skipped property."""
        from ftw_dataset_tools.api.boundaries import CreateBoundariesResult

        result = CreateBoundariesResult(
            files_processed=[],
            files_skipped=[(Path("a"), "reason1"), (Path("b"), "reason2")],
        )
        assert result.total_skipped == 2

    def test_total_features(self) -> None:
        """Test total_features property."""
        from ftw_dataset_tools.api.boundaries import (
            BoundaryResult,
            CreateBoundariesResult,
        )

        result = CreateBoundariesResult(
            files_processed=[
                BoundaryResult(Path("a"), Path("b"), 10),
                BoundaryResult(Path("c"), Path("d"), 20),
            ],
            files_skipped=[],
        )
        assert result.total_features == 30


class TestCreateBoundaries:
    """Tests for create_boundaries function."""

    def test_file_not_found(self) -> None:
        """Test FileNotFoundError for missing input."""
        from ftw_dataset_tools.api.boundaries import create_boundaries

        with pytest.raises(FileNotFoundError, match="Input path not found"):
            create_boundaries("/nonexistent/path.parquet")

    def test_no_parquet_files_error(self, tmp_path: Path) -> None:
        """Test ValueError when no parquet files found."""
        from ftw_dataset_tools.api.boundaries import create_boundaries

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(ValueError, match="No parquet files found"):
            create_boundaries(empty_dir)

    def test_creates_boundary_lines_from_polygons(self, tmp_path: Path) -> None:
        """Test boundary lines are created from polygon geometries."""
        from ftw_dataset_tools.api.boundaries import create_boundaries

        # Create input with polygon
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        input_file = tmp_path / "polygons.parquet"
        gdf.to_parquet(input_file)

        result = create_boundaries(input_file, output_dir=tmp_path)

        assert result.total_processed == 1
        assert result.total_features == 1
        assert result.files_processed[0].output_path.exists()

        # Verify output has LineString geometry
        output_gdf = gpd.read_parquet(result.files_processed[0].output_path)
        assert output_gdf.iloc[0].geometry.geom_type in ["LineString", "LinearRing"]

    def test_skips_non_polygon_files(self, tmp_path: Path) -> None:
        """Test files without polygon geometries are skipped."""
        from ftw_dataset_tools.api.boundaries import create_boundaries

        # Create input with LineString (not polygon)
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[LineString([(0, 0), (1, 1)])],
            crs="EPSG:4326",
        )
        input_file = tmp_path / "lines.parquet"
        gdf.to_parquet(input_file)

        result = create_boundaries(input_file, output_dir=tmp_path)

        assert result.total_processed == 0
        assert result.total_skipped == 1

    def test_progress_callback(self, tmp_path: Path) -> None:
        """Test progress callback is invoked."""
        from ftw_dataset_tools.api.boundaries import create_boundaries

        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        input_file = tmp_path / "polygons.parquet"
        gdf.to_parquet(input_file)

        messages: list[str] = []
        create_boundaries(input_file, on_progress=messages.append)

        assert len(messages) > 0
        assert any("Found" in msg for msg in messages)
