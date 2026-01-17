"""Tests for the create-boundaries CLI command."""

from pathlib import Path

from click.testing import CliRunner

from ftw_dataset_tools.cli import cli


class TestCreateBoundariesCommand:
    """Tests for create-boundaries command."""

    def test_help(self) -> None:
        """Test --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-boundaries", "--help"])
        assert result.exit_code == 0
        assert "Convert polygon geometries" in result.output

    def test_missing_input(self) -> None:
        """Test error for missing input argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-boundaries"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_nonexistent_file(self) -> None:
        """Test error for nonexistent input file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-boundaries", "/nonexistent/file.parquet"])
        assert result.exit_code != 0

    def test_valid_input(self, sample_fields_geoparquet: Path, tmp_path: Path) -> None:
        """Test create-boundaries with valid input."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-boundaries",
                str(sample_fields_geoparquet),
                "--output-dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert "Created" in result.output or "Processing" in result.output

    def test_output_dir_option(self, sample_fields_geoparquet: Path, tmp_path: Path) -> None:
        """Test --output-dir option."""
        output_dir = tmp_path / "boundaries_output"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-boundaries",
                str(sample_fields_geoparquet),
                "--output-dir",
                str(output_dir),
            ],
        )
        assert result.exit_code == 0
