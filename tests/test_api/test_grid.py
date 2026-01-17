"""Tests for the grid API module."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box


class TestGetGridResult:
    """Tests for GetGridResult dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test GetGridResult has expected fields."""
        from ftw_dataset_tools.api.grid import GetGridResult

        result = GetGridResult(
            output_path=Path("/tmp/output.parquet"),
            grid_count=100,
            bounds=(0.0, 0.0, 10.0, 10.0),
        )
        assert result.output_path == Path("/tmp/output.parquet")
        assert result.grid_count == 100
        assert result.bounds == (0.0, 0.0, 10.0, 10.0)


class TestCRSError:
    """Tests for CRSError exception."""

    def test_error_message_format(self) -> None:
        """Test CRSError message contains helpful info."""
        from ftw_dataset_tools.api.grid import CRSError

        error = CRSError("EPSG:3035", "/path/to/input.parquet")
        assert "EPSG:3035" in str(error)
        assert "EPSG:4326" in str(error)
        assert "ftwd reproject" in str(error)


class TestGetGrid:
    """Tests for get_grid function."""

    def test_file_not_found(self) -> None:
        """Test FileNotFoundError for missing input."""
        from ftw_dataset_tools.api.grid import get_grid

        with pytest.raises(FileNotFoundError, match="Input file not found"):
            get_grid("/nonexistent/input.parquet")

    def test_crs_error_non_4326(self, tmp_path: Path) -> None:
        """Test CRSError when input is not EPSG:4326."""
        from ftw_dataset_tools.api.grid import CRSError, get_grid

        # Create file in EPSG:3035
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(5150000, 3540000, 5160000, 3550000)],
            crs="EPSG:3035",
        )
        input_file = tmp_path / "input_3035.parquet"
        gdf.to_parquet(input_file)

        with pytest.raises(CRSError):
            get_grid(input_file)

    def test_progress_callback(self, sample_geoparquet_4326: Path) -> None:
        """Test progress callback is invoked."""
        import contextlib

        from ftw_dataset_tools.api.grid import get_grid

        messages: list[str] = []

        # This will fail at the S3 fetch step, but progress should be called first
        with contextlib.suppress(Exception):
            get_grid(sample_geoparquet_4326, on_progress=messages.append)

        assert len(messages) > 0
        assert any("CRS" in msg for msg in messages)

    @pytest.mark.network
    def test_fetches_grid_from_source_coop(self, sample_geoparquet_4326: Path) -> None:
        """Test fetching grid from Source Coop (requires network)."""
        from ftw_dataset_tools.api.grid import get_grid

        result = get_grid(sample_geoparquet_4326)

        assert result.output_path.exists()
        assert result.grid_count > 0
