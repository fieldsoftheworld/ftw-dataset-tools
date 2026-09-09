"""Tests for the create-chips CLI command."""

from pathlib import Path

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

    def test_valid_input_with_local_grid(
        self, sample_fields_geoparquet: Path, sample_grid_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test create-chips with valid local grid file."""
        output_file = tmp_path / "chips.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-chips",
                str(sample_fields_geoparquet),
                "--grid-file",
                str(sample_grid_geoparquet),
                "-o",
                str(output_file),
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()

    def test_min_coverage_option(
        self, sample_fields_geoparquet: Path, sample_grid_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Test --min-coverage option."""
        output_file = tmp_path / "chips_filtered.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-chips",
                str(sample_fields_geoparquet),
                "--grid-file",
                str(sample_grid_geoparquet),
                "-o",
                str(output_file),
                "--min-coverage",
                "1.0",
            ],
        )
        assert result.exit_code == 0

    def test_batch_size_option(
        self, sample_fields_geoparquet: Path, sample_grid_geoparquet: Path, tmp_path: Path
    ) -> None:
        """--batch-size reaches the coverage step (the OOM escape hatch users need)."""
        output_file = tmp_path / "chips_batched.parquet"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-chips",
                str(sample_fields_geoparquet),
                "--grid-file",
                str(sample_grid_geoparquet),
                "-o",
                str(output_file),
                "--batch-size",
                "1",
            ],
        )
        assert result.exit_code == 0
        assert output_file.exists()
        assert "Coverage: 1/2 grid cells" in result.output

    def test_batch_size_must_be_positive(
        self, sample_fields_geoparquet: Path, sample_grid_geoparquet: Path, tmp_path: Path
    ) -> None:
        """An invalid --batch-size is rejected at parse time, before any work."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-chips",
                str(sample_fields_geoparquet),
                "--grid-file",
                str(sample_grid_geoparquet),
                "-o",
                str(tmp_path / "chips.parquet"),
                "--batch-size",
                "0",
            ],
        )
        assert result.exit_code != 0
        assert "Invalid value for '--batch-size'" in result.output
        assert "range x>=1" in result.output
