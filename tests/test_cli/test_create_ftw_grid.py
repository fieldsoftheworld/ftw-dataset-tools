"""Tests for the create-ftw-grid CLI command."""

from pathlib import Path

from click.testing import CliRunner

from ftw_dataset_tools.cli import cli


class TestCreateFTWGridCommand:
    """Tests for create-ftw-grid command."""

    def test_help(self) -> None:
        """Test --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-ftw-grid", "--help"])
        assert result.exit_code == 0

    def test_missing_input(self) -> None:
        """Test error for missing input argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-ftw-grid"])
        assert result.exit_code != 0

    def test_valid_input(self, sample_mgrs_1km_geoparquet: Path, tmp_path: Path) -> None:
        """Test create-ftw-grid with valid MGRS input."""
        output_file = tmp_path / "ftw_grid.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-ftw-grid",
                str(sample_mgrs_1km_geoparquet),
                "--output",
                str(output_file),
                "--km-size",
                "2",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_km_size_option(self, sample_mgrs_1km_geoparquet: Path, tmp_path: Path) -> None:
        """Test --km-size option with valid value."""
        output_file = tmp_path / "ftw_grid_5km.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-ftw-grid",
                str(sample_mgrs_1km_geoparquet),
                "--output",
                str(output_file),
                "--km-size",
                "5",
            ],
        )
        assert result.exit_code == 0

    def test_invalid_km_size(self, sample_mgrs_1km_geoparquet: Path, tmp_path: Path) -> None:
        """Test --km-size option with invalid value."""
        output_file = tmp_path / "ftw_grid_3km.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-ftw-grid",
                str(sample_mgrs_1km_geoparquet),
                "--output",
                str(output_file),
                "--km-size",
                "3",
            ],
        )
        assert result.exit_code != 0
        assert "km_size must divide 100" in result.output
