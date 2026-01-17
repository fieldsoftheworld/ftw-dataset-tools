"""Tests for the create-masks CLI command."""

from click.testing import CliRunner

from ftw_dataset_tools.cli import cli


class TestCreateMasksCommand:
    """Tests for create-masks command."""

    def test_help(self) -> None:
        """Test --help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-masks", "--help"])
        assert result.exit_code == 0
        assert "CHIPS_FILE" in result.output

    def test_missing_arguments(self) -> None:
        """Test error for missing required arguments."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-masks"])
        assert result.exit_code != 0
