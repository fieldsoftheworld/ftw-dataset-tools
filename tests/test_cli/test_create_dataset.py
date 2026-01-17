"""Tests for the create-dataset CLI command."""

from click.testing import CliRunner

from ftw_dataset_tools.cli import cli


class TestCreateDatasetCommand:
    """Tests for create-dataset command."""

    def test_help(self) -> None:
        """Test --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-dataset", "--help"])
        assert result.exit_code == 0
        assert "Create a complete training dataset" in result.output

    def test_missing_input(self) -> None:
        """Test error for missing input argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-dataset"])
        assert result.exit_code != 0

    def test_nonexistent_file(self) -> None:
        """Test error for nonexistent input file."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-dataset", "/nonexistent/fields.parquet"])
        assert result.exit_code != 0
