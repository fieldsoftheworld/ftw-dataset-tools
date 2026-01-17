"""Tests for the create-chips CLI command."""

from click.testing import CliRunner

from ftw_dataset_tools.cli import cli


class TestCreateChipsCommand:
    """Tests for create-chips command."""

    def test_help(self) -> None:
        """Test --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-chips", "--help"])
        assert result.exit_code == 0

    def test_missing_input(self) -> None:
        """Test error for missing input argument."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-chips"])
        assert result.exit_code != 0
