"""Tests for the ftw_grid API module."""

from pathlib import Path

import duckdb
import geopandas as gpd
import pytest
from shapely.geometry import box


class TestInvalidKmSizeError:
    """Tests for InvalidKmSizeError exception."""

    def test_error_message_contains_valid_sizes(self) -> None:
        """Test error message lists valid km sizes."""
        from ftw_dataset_tools.api.ftw_grid import InvalidKmSizeError

        error = InvalidKmSizeError(3)
        message = str(error)
        assert "3" in message
        assert "1" in message
        assert "2" in message
        assert "5" in message


class TestMultipleGZDError:
    """Tests for MultipleGZDError exception."""

    def test_error_message_lists_gzds(self) -> None:
        """Test error message lists found GZDs."""
        from ftw_dataset_tools.api.ftw_grid import MultipleGZDError

        error = MultipleGZDError(3, ["32U", "33U", "34U"])
        message = str(error)
        assert "32U" in message
        assert "33U" in message
        assert "exactly one GZD" in message


class TestCreateFTWGridResult:
    """Tests for CreateFTWGridResult dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test CreateFTWGridResult has expected fields."""
        from ftw_dataset_tools.api.ftw_grid import CreateFTWGridResult

        result = CreateFTWGridResult(
            output_path=Path("/tmp/output"),
            gzd="32U",
            km_size=2,
            total_cells=100,
            gzd_count=1,
        )
        assert result.output_path == Path("/tmp/output")
        assert result.gzd == "32U"
        assert result.km_size == 2


class TestCreateFTWGrid:
    """Tests for create_ftw_grid function."""

    def test_file_not_found(self) -> None:
        """Test FileNotFoundError for missing input."""
        from ftw_dataset_tools.api.ftw_grid import create_ftw_grid

        with pytest.raises(FileNotFoundError, match="Input path not found"):
            create_ftw_grid("/nonexistent/input.parquet")

    def test_invalid_km_size(self, tmp_path: Path) -> None:
        """Test InvalidKmSizeError for invalid km_size."""
        from ftw_dataset_tools.api.ftw_grid import InvalidKmSizeError, create_ftw_grid

        # Create dummy file
        input_file = tmp_path / "input.parquet"
        input_file.touch()

        with pytest.raises(InvalidKmSizeError):
            create_ftw_grid(input_file, km_size=3)

    def test_folder_input_requires_output_path(self, tmp_path: Path) -> None:
        """Test ValueError when folder input without output_path."""
        from ftw_dataset_tools.api.ftw_grid import create_ftw_grid

        input_dir = tmp_path / "input"
        input_dir.mkdir()

        with pytest.raises(ValueError, match="Output path is required"):
            create_ftw_grid(input_dir, output_path=None)

    def test_valid_km_sizes(self) -> None:
        """Test VALID_KM_SIZES contains expected values."""
        from ftw_dataset_tools.api.ftw_grid import VALID_KM_SIZES

        assert 1 in VALID_KM_SIZES
        assert 2 in VALID_KM_SIZES
        assert 5 in VALID_KM_SIZES
        assert 10 in VALID_KM_SIZES
        assert 3 not in VALID_KM_SIZES

    def test_multiple_gzd_error(self, sample_mgrs_multiple_gzd_geoparquet: Path) -> None:
        """Test MultipleGZDError when input has multiple GZDs."""
        from ftw_dataset_tools.api.ftw_grid import MultipleGZDError, create_ftw_grid

        with pytest.raises(MultipleGZDError) as exc_info:
            create_ftw_grid(sample_mgrs_multiple_gzd_geoparquet)
        assert exc_info.value.gzd_count == 2
        assert "32U" in exc_info.value.gzds
        assert "33U" in exc_info.value.gzds

    def test_create_ftw_grid_single_file(
        self, sample_mgrs_1km_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test create_ftw_grid with valid single GZD input."""
        from ftw_dataset_tools.api.ftw_grid import create_ftw_grid

        output_file = tmp_path / "ftw_output.parquet"
        result = create_ftw_grid(sample_mgrs_1km_geoparquet, output_file, km_size=2)

        assert result.output_path == output_file.resolve()
        assert result.gzd == "32U"
        assert result.km_size == 2
        assert result.total_cells >= 1
        assert result.gzd_count == 1
        assert output_file.exists()

    def test_create_ftw_grid_default_output_name(self, sample_mgrs_1km_geoparquet: Path) -> None:
        """Test default output filename is ftw_<gzd>.parquet."""
        from ftw_dataset_tools.api.ftw_grid import create_ftw_grid

        result = create_ftw_grid(sample_mgrs_1km_geoparquet, km_size=2)

        expected_name = "ftw_32U.parquet"
        assert result.output_path.name == expected_name
        # Cleanup
        if result.output_path.exists():
            result.output_path.unlink()

    def test_create_ftw_grid_with_progress_callback(
        self, sample_mgrs_1km_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test progress callback is called."""
        from ftw_dataset_tools.api.ftw_grid import create_ftw_grid

        progress_messages: list[str] = []

        def on_progress(msg: str) -> None:
            progress_messages.append(msg)

        output_file = tmp_path / "ftw_output.parquet"
        create_ftw_grid(sample_mgrs_1km_geoparquet, output_file, km_size=2, on_progress=on_progress)

        assert len(progress_messages) > 0
        assert any("32U" in msg for msg in progress_messages)

    def test_create_ftw_grid_minimal_columns(
        self, sample_mgrs_1km_minimal_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test create_ftw_grid with only required columns (MGRS, GZD, geometry)."""
        from ftw_dataset_tools.api.ftw_grid import create_ftw_grid

        output_file = tmp_path / "ftw_output.parquet"
        result = create_ftw_grid(sample_mgrs_1km_minimal_geoparquet, output_file, km_size=2)

        assert result.output_path.exists()
        assert result.gzd == "32U"
        assert result.total_cells >= 1


class TestHelperFunctions:
    """Tests for ftw_grid helper functions."""

    def test_normalize_columns(self, sample_mgrs_1km_geoparquet: Path) -> None:
        """Test _normalize_columns returns lowercase to actual name mapping."""
        from ftw_dataset_tools.api.ftw_grid import _normalize_columns

        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute(f"CREATE TABLE mgrs_1km AS SELECT * FROM '{sample_mgrs_1km_geoparquet}'")

        col_map = _normalize_columns(conn)
        conn.close()

        assert "mgrs" in col_map
        assert "gzd" in col_map
        assert col_map["mgrs"] == "MGRS"
        assert col_map["gzd"] == "GZD"

    def test_check_required_columns_all_present(self, sample_mgrs_1km_geoparquet: Path) -> None:
        """Test _check_required_columns returns empty list when all present."""
        from ftw_dataset_tools.api.ftw_grid import _check_required_columns, _normalize_columns

        conn = duckdb.connect(":memory:")
        conn.execute("INSTALL spatial; LOAD spatial;")
        conn.execute(f"CREATE TABLE mgrs_1km AS SELECT * FROM '{sample_mgrs_1km_geoparquet}'")

        col_map = _normalize_columns(conn)
        missing = _check_required_columns(col_map)
        conn.close()

        assert missing == []

    def test_check_required_columns_missing(self) -> None:
        """Test _check_required_columns returns missing column names."""
        from ftw_dataset_tools.api.ftw_grid import _check_required_columns

        col_map = {"mgrs": "MGRS", "other": "OTHER"}  # Missing "gzd"
        missing = _check_required_columns(col_map)

        assert "GZD" in missing

    def test_get_column_case_insensitive(self) -> None:
        """Test _get_column returns actual column name from lowercase key."""
        from ftw_dataset_tools.api.ftw_grid import _get_column

        col_map = {"mgrs": "MGRS", "gzd": "GZD", "kmsq_id": "kmSQ_ID"}

        assert _get_column(col_map, "MGRS") == "MGRS"
        assert _get_column(col_map, "gzd") == "GZD"
        assert _get_column(col_map, "KmSq_Id") == "kmSQ_ID"

    def test_get_column_fallback(self) -> None:
        """Test _get_column returns original name if not in map."""
        from ftw_dataset_tools.api.ftw_grid import _get_column

        col_map = {"mgrs": "MGRS"}

        assert _get_column(col_map, "unknown") == "unknown"


class TestMultipleGZDErrorMessages:
    """Additional tests for MultipleGZDError exception messages."""

    def test_error_message_lists_gzds(self) -> None:
        """Test error message lists found GZDs."""
        from ftw_dataset_tools.api.ftw_grid import MultipleGZDError

        error = MultipleGZDError(3, ["32U", "33U", "34U"])
        message = str(error)
        assert "32U" in message
        assert "33U" in message
        assert "exactly one GZD" in message

    def test_truncation_for_many_gzds(self) -> None:
        """Test error message truncates when more than 10 GZDs."""
        from ftw_dataset_tools.api.ftw_grid import MultipleGZDError

        gzds = [f"{i}U" for i in range(20)]
        error = MultipleGZDError(20, gzds)
        message = str(error)
        assert "20 total" in message


class TestCreateFTWGridPartitioned:
    """Tests for create_ftw_grid with partitioned input."""

    def test_empty_folder_raises_error(self, tmp_path: Path) -> None:
        """Test ValueError when input folder has no parquet files."""
        from ftw_dataset_tools.api.ftw_grid import create_ftw_grid

        input_dir = tmp_path / "empty_input"
        input_dir.mkdir()
        output_dir = tmp_path / "output"

        with pytest.raises(ValueError, match="No parquet files found"):
            create_ftw_grid(input_dir, output_dir)

    def test_partitioned_folder_with_single_file(
        self, sample_mgrs_1km_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test partitioned input with a single GZD file."""
        from ftw_dataset_tools.api.ftw_grid import create_ftw_grid

        # Create input folder with the sample file
        input_dir = tmp_path / "partitioned_input"
        input_dir.mkdir()
        import shutil

        shutil.copy(sample_mgrs_1km_geoparquet, input_dir / "32U.parquet")

        output_dir = tmp_path / "output"
        result = create_ftw_grid(input_dir, output_dir, km_size=2)

        assert result.output_path == output_dir
        assert result.gzd is None  # Multiple GZD mode
        assert result.gzd_count == 1
        assert result.total_cells >= 1
        # Check output structure
        assert (output_dir / "gzd=32U").exists()


class TestMissingRequiredColumns:
    """Tests for handling files with missing required columns."""

    def test_missing_mgrs_column(self, tmp_path: Path) -> None:
        """Test ValueError when MGRS column is missing."""
        from ftw_dataset_tools.api.ftw_grid import create_ftw_grid

        # Create file without MGRS column
        gdf = gpd.GeoDataFrame(
            {"GZD": ["32U"], "other": ["value"]},
            geometry=[box(9.0, 48.0, 9.01, 48.01)],
            crs="EPSG:4326",
        )
        input_file = tmp_path / "no_mgrs.parquet"
        gdf.to_parquet(input_file)

        with pytest.raises(ValueError, match=r"Missing required columns.*MGRS"):
            create_ftw_grid(input_file)

    def test_missing_gzd_column(self, tmp_path: Path) -> None:
        """Test ValueError when GZD column is missing."""
        from ftw_dataset_tools.api.ftw_grid import create_ftw_grid

        # Create file without GZD column
        gdf = gpd.GeoDataFrame(
            {"MGRS": ["32UPE0000"], "other": ["value"]},
            geometry=[box(9.0, 48.0, 9.01, 48.01)],
            crs="EPSG:4326",
        )
        input_file = tmp_path / "no_gzd.parquet"
        gdf.to_parquet(input_file)

        with pytest.raises(ValueError, match=r"Missing required columns.*GZD"):
            create_ftw_grid(input_file)
