"""Tests for the add-field-stats CLI command."""

from pathlib import Path

from click.testing import CliRunner

from ftw_dataset_tools.cli import cli


class TestAddFieldStatsCommand:
    """Tests for the add-field-stats CLI command."""

    def test_help(self) -> None:
        """Test that --help works for add-field-stats command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["add-field-stats", "--help"])
        assert result.exit_code == 0
        assert "Add field coverage statistics" in result.output
        assert "GRID_FILE" in result.output
        assert "FIELDS_FILE" in result.output

    def test_missing_arguments(self) -> None:
        """Test that missing required arguments produce an error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["add-field-stats"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_nonexistent_file(self) -> None:
        """Test that nonexistent files produce an error."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["add-field-stats", "/nonexistent/grid.parquet", "/nonexistent/fields.parquet"]
        )
        assert result.exit_code != 0

    def test_valid_input(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test add-field-stats with valid input files."""
        output_file = tmp_path / "chips.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add-field-stats",
                str(sample_grid_geoparquet),
                str(sample_fields_geoparquet),
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_output_option(
        self, sample_grid_geoparquet: Path, sample_fields_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test -o / --output option."""
        output_file = tmp_path / "custom_output.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "add-field-stats",
                str(sample_grid_geoparquet),
                str(sample_fields_geoparquet),
                "--output",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()


class TestCli:
    """Tests for the main CLI."""

    def test_version(self) -> None:
        """Test that --version works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert "ftwd" in result.output
        assert "0.1.0" in result.output

    def test_help(self) -> None:
        """Test that --help works for main CLI."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "FTW Dataset Tools" in result.output
        assert "add-field-stats" in result.output
