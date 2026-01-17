"""Tests for the ftw_grid API module."""

from pathlib import Path

import pytest


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
