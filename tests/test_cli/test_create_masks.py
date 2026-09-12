"""Tests for the create-masks CLI command."""

from pathlib import Path

from click.testing import CliRunner

from ftw_dataset_tools.api.config import VALID_MASK_TYPES
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
                "--year",
                "2024",
            ],
        )
        assert result.exit_code == 0
        assert output_dir.exists()
        # The command writes the pipeline's catalog, not just its directory shape.
        assert (output_dir / "collection.json").exists()

    def test_mask_type_option(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        tmp_path: Path,
    ) -> None:
        """Test --mask-type option with different values."""
        for mask_type in VALID_MASK_TYPES:
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
                    "--year",
                    "2024",
                ],
            )
            assert result.exit_code == 0
            # grid_001 is not an FTW grid id, so it lands under the 'other' square.
            assert (
                output_dir / "chips" / "other" / "grid_001_2024" / f"grid_001_2024_{mask_type}.tif"
            ).exists()

    def test_workers_help_documents_the_cap(self) -> None:
        """The default is the CPU count capped at 8, not half of the CPUs."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-masks", "--help"])
        assert result.exit_code == 0
        assert "capped at 8" in result.output
        assert "half of CPUs" not in result.output

    def test_skip_existing_reuses_masks_on_a_rerun(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        tmp_path: Path,
    ) -> None:
        """The standalone command supports the gap-filling rerun the pipeline has."""
        output_dir = tmp_path / "masks"
        args = [
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
            "--year",
            "2024",
        ]
        runner = CliRunner()
        first = runner.invoke(cli, args)
        assert first.exit_code == 0
        assert "Masks reused" not in first.output

        second = runner.invoke(cli, [*args, "--skip-existing"])
        assert second.exit_code == 0
        assert "Masks reused: 3" in second.output
        assert "Masks created: 0" in second.output

    def test_reports_worker_pool_restarts(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        tmp_path: Path,
        monkeypatch,
    ) -> None:
        """A run degraded by repeated worker deaths must not look clean."""
        from ftw_dataset_tools.api import masks

        def fake_create_masks(**kwargs):
            kwargs["on_start"](1, 1, 1)
            return {
                masks.MaskType.SEMANTIC_2_CLASS: masks.CreateMasksResult(
                    masks_created=[],
                    masks_skipped=[],
                    field_dataset="test",
                    pool_restarts=3,
                )
            }

        monkeypatch.setattr(masks, "create_masks", fake_create_masks)

        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-masks",
                str(sample_chips_with_coverage),
                str(sample_boundaries_geoparquet),
                str(sample_boundary_lines_geoparquet),
                "--output-dir",
                str(tmp_path / "masks"),
                "--field-dataset",
                "test",
                "--mask-type",
                "semantic_2_class",
                "--year",
                "2024",
            ],
        )
        assert result.exit_code == 0
        assert "Worker pool restarts: 3" in result.output

    def test_year_reaches_item_ids_and_filenames(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        tmp_path: Path,
    ) -> None:
        """--year has to fold into the item id, as create-dataset does."""
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
                "--year",
                "2024",
            ],
        )
        assert result.exit_code == 0
        chip_dir = output_dir / "chips" / "other" / "grid_001_2024"
        assert (chip_dir / "grid_001_2024_semantic_2_class.tif").exists()

    def test_writes_a_readable_stac_catalog(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        tmp_path: Path,
    ) -> None:
        """The output must be a catalog, not just the shape of one."""
        import json

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
                "--year",
                "2024",
            ],
        )
        assert result.exit_code == 0
        assert (output_dir / "collection.json").exists()
        assert (output_dir / "chips" / "other" / "catalog.json").exists()

        item_path = output_dir / "chips" / "other" / "grid_001_2024" / "grid_001_2024.json"
        assert item_path.exists()
        # The mask has to be registered as an asset, or nothing downstream finds it.
        item = json.loads(item_path.read_text())
        hrefs = [a["href"] for a in item["assets"].values()]
        assert any(h.endswith("grid_001_2024_semantic_2_class.tif") for h in hrefs)

    def test_missing_year_without_datetime_column_fails_fast(
        self,
        sample_chips_with_coverage: Path,
        sample_boundaries_geoparquet: Path,
        sample_boundary_lines_geoparquet: Path,
        tmp_path: Path,
    ) -> None:
        """The temporal-extent error must arrive before a full mask run, not after."""
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
                "--min-coverage",
                "0.0",
            ],
        )
        assert result.exit_code != 0
        assert "--year" in result.output
        # Nothing was rasterized before the check fired.
        assert not list(output_dir.rglob("*.tif"))
