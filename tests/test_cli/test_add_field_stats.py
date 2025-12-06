"""Tests for the main CLI."""

from click.testing import CliRunner

from ftw_dataset_tools.cli import cli


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
        # Check for commands that actually exist
        assert "create-chips" in result.output
        assert "reproject" in result.output
