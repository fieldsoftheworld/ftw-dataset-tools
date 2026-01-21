"""Tests for the add-field-stats CLI command."""

from click.testing import CliRunner

from ftw_dataset_tools.cli import cli


class TestAddFieldStatsCommand:
    """Tests for the add-field-stats CLI command."""

    def test_help(self) -> None:
        """Test that --help works for create-chips command."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-chips", "--help"])
        assert result.exit_code == 0
        assert "Create chip definitions" in result.output
        assert "FIELDS_FILE" in result.output

    def test_missing_arguments(self) -> None:
        """Test that missing required arguments produce an error."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-chips"])
        assert result.exit_code != 0
        assert "Missing argument" in result.output

    def test_nonexistent_file(self) -> None:
        """Test that nonexistent files produce an error."""
        runner = CliRunner()
        result = runner.invoke(
            cli, ["create-chips", "/nonexistent/grid.parquet", "/nonexistent/fields.parquet"]
        )
        assert result.exit_code != 0


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
        assert "create-chips" in result.output
