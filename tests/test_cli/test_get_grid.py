"""Tests for the get-grid CLI command."""

from click.testing import CliRunner

from ftw_dataset_tools.cli import cli


class TestGetGridCommand:
    """Tests for get-grid command."""

    def test_help(self) -> None:
        """Test --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get-grid", "--help"])
        assert result.exit_code == 0

    def test_missing_input(self) -> None:
        """Test error for missing input argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["get-grid"])
        assert result.exit_code != 0
