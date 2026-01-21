"""Tests for the create-masks CLI command."""

from pathlib import Path

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

    def test_valid_inputs(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        tmp_path: Path,
    ) -> None:
        """Test create-masks with valid input files."""
        output_dir = tmp_path / "masks"
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-masks",
                str(sample_chips_with_coverage),
                str(sample_boundaries_geoparquet),
                str(sample_boundary_lines_geoparquet),
                "--output-dir",
                str(output_dir),
                "--field-dataset",
                "test",
                "--mask-type",
                "semantic_2_class",
                "--min-coverage",
                "0.0",
            ],
        )
        assert result.exit_code == 0
        assert output_dir.exists()

    def test_mask_type_option(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        tmp_path: Path,
    ) -> None:
        """Test --mask-type option with different values."""
        for mask_type in ["instance", "semantic_2_class", "semantic_3_class"]:
            output_dir = tmp_path / f"masks_{mask_type}"
            runner = CliRunner()
            result = runner.invoke(
                cli,
                [
                    "create-masks",
                    str(sample_chips_with_coverage),
                    str(sample_boundaries_geoparquet),
                    str(sample_boundary_lines_geoparquet),
                    "--output-dir",
                    str(output_dir),
                    "--field-dataset",
                    "test",
                    "--mask-type",
                    mask_type,
                    "--min-coverage",
                    "0.0",
                ],
            )
            assert result.exit_code == 0
