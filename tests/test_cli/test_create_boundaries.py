"""Tests for the create-boundaries CLI command."""

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
