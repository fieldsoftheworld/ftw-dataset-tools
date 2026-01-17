"""Tests for the create-ftw-grid CLI command."""

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
