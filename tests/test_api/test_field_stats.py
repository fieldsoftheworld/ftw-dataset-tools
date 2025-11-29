"""Tests for the field_stats API."""

import pytest

from ftw_dataset_tools.api.field_stats import FieldStatsResult


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

    def test_file_not_found_grid(self) -> None:
        """Test that FileNotFoundError is raised for missing grid file."""
        from ftw_dataset_tools.api.field_stats import add_field_stats

        with pytest.raises(FileNotFoundError, match="Grid file not found"):
            add_field_stats(
                grid_file="/nonexistent/grid.parquet",
                fields_file="/nonexistent/fields.parquet",
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
