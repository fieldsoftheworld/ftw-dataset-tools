"""Tests for the create-chips CLI command."""

from pathlib import Path

import pytest
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
                # The sample grid's cells are smaller than a real 2 km chip, so the
                # default size filter would drop them before coverage ever runs.
                "--min-chip-area",
                "0",
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

    def test_size_filter_is_on_by_default(
        self, sample_fields_geoparquet: Path, sample_grid_geoparquet: Path, tmp_path: Path
    ) -> None:
        """Chips truncated below a full cell are dropped without asking.

        The sample grid's cells are under 2 km, so the default threshold removes them
        and says so rather than shipping short chips.
        """
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
            ],
        )
        assert result.exit_code == 0
        assert "Removed 2 undersized chips" in result.output
        assert "Undersized chips dropped: 2" in result.output

    def test_size_filter_can_be_disabled(
        self, sample_fields_geoparquet: Path, sample_grid_geoparquet: Path, tmp_path: Path
    ) -> None:
        """--min-chip-area 0 keeps every cell, whatever its size."""
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
                "--min-chip-area",
                "0",
            ],
        )
        assert result.exit_code == 0
        assert "Total grid cells: 2" in result.output
        assert "undersized" not in result.output.lower()

    @pytest.mark.parametrize("bad", ["150", "-1"])
    def test_min_chip_area_outside_percentage_range_rejected(
        self, sample_fields_geoparquet: Path, tmp_path: Path, bad: str
    ) -> None:
        """A percentage outside 0-100 is a mistake, not a licence to drop every chip."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-chips",
                str(sample_fields_geoparquet),
                "-o",
                str(tmp_path / "chips.parquet"),
                "--min-chip-area",
                bad,
            ],
        )
        assert result.exit_code != 0
        assert "Invalid value for '--min-chip-area'" in result.output

    @pytest.mark.parametrize("bad", ["0", "-2"])
    def test_non_positive_km_size_rejected(
        self, sample_fields_geoparquet: Path, tmp_path: Path, bad: str
    ) -> None:
        """--km-size 0 would silently disable the filter, so it is refused."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-chips",
                str(sample_fields_geoparquet),
                "-o",
                str(tmp_path / "chips.parquet"),
                "--km-size",
                bad,
            ],
        )
        assert result.exit_code != 0
        assert "Invalid value for '--km-size'" in result.output
