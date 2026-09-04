"""Tests for the create-dataset CLI command."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import geopandas as gpd
import pytest
from click.testing import CliRunner, Result
from shapely.geometry import box

from ftw_dataset_tools.api.dataset import CreateDatasetResult
from ftw_dataset_tools.api.imagery.selection_workflow import SelectionWorkflowResult
from ftw_dataset_tools.api.stac import STACGenerationResult
from ftw_dataset_tools.cli import cli
from ftw_dataset_tools.commands import create_dataset as create_dataset_module

if TYPE_CHECKING:
    from pathlib import Path


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


class TestCreateDatasetMaskTypes:
    """Tests for --mask-types parsing and validation."""

    def test_decode_types_listed_in_help(self) -> None:
        """The DECODE layers are discoverable from --help."""
        runner = CliRunner()
        result = runner.invoke(cli, ["create-dataset", "--help"])

        assert result.exit_code == 0
        assert "decode_boundary" in result.output
        assert "decode_distance" in result.output

    def test_decode_types_not_on_by_default(self) -> None:
        """The shown default stays the three standard mask types."""
        from ftw_dataset_tools.api.config import DEFAULT_MASK_TYPES

        assert "decode_boundary" not in DEFAULT_MASK_TYPES
        assert "decode_distance" not in DEFAULT_MASK_TYPES

    def test_invalid_mask_type_rejected(self, sample_fields_geoparquet: Path) -> None:
        """An unknown mask type fails before any work starts."""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            [
                "create-dataset",
                str(sample_fields_geoparquet),
                "--split-type",
                "random-uniform",
                "--mask-types",
                "decode_bounary",  # typo
            ],
        )

        assert result.exit_code != 0
        assert "Invalid mask type" in result.output
        # The error lists the valid options, including the DECODE ones.
        assert "decode_boundary" in result.output

    def test_decode_mask_types_accepted(
        self, sample_fields_geoparquet: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Requesting only the DECODE layers passes validation and reaches the pipeline."""
        captured: dict[str, Any] = {}

        def fake_create_dataset(**kwargs: Any) -> CreateDatasetResult:
            captured.update(kwargs)
            raise SystemExit(0)

        monkeypatch.setattr(create_dataset_module.dataset, "create_dataset", fake_create_dataset)

        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "create-dataset",
                str(sample_fields_geoparquet),
                "--split-type",
                "random-uniform",
                "--mask-types",
                "decode_boundary,decode_distance",
            ],
        )

        assert captured["mask_types"] == ["decode_boundary", "decode_distance"]

    def test_mask_types_are_whitespace_tolerant(
        self, sample_fields_geoparquet: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Spaces around the commas do not break validation."""
        captured: dict[str, Any] = {}

        def fake_create_dataset(**kwargs: Any) -> CreateDatasetResult:
            captured.update(kwargs)
            raise SystemExit(0)

        monkeypatch.setattr(create_dataset_module.dataset, "create_dataset", fake_create_dataset)

        runner = CliRunner()
        runner.invoke(
            cli,
            [
                "create-dataset",
                str(sample_fields_geoparquet),
                "--split-type",
                "random-uniform",
                "--mask-types",
                "semantic_2_class, decode_distance",
            ],
        )

        assert captured["mask_types"] == ["semantic_2_class", "decode_distance"]


@dataclass
class SelectionStub:
    """Records calls to select_imagery_for_catalog and controls what it does."""

    calls: list[dict[str, Any]] = field(default_factory=list)
    result: SelectionWorkflowResult = field(
        default_factory=lambda: SelectionWorkflowResult(successful=2, skipped=1, failed=0)
    )
    error: BaseException | None = None


@pytest.fixture
def stub_pipeline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> SelectionStub:
    """Stub out dataset creation and imagery selection.

    Dataset creation is replaced with a fixed result pointing at an empty chips
    catalog, so tests exercise only the imagery handling in the command.
    """
    chips_base_dir = tmp_path / "out" / "fields-chips"
    chips_base_dir.mkdir(parents=True)

    def fake_create_dataset(**_kwargs: Any) -> CreateDatasetResult:
        return CreateDatasetResult(
            output_dir=tmp_path / "out",
            field_dataset="fields",
            fields_file=tmp_path / "out" / "fields_fields.parquet",
            chips_file=tmp_path / "out" / "fields_chips.parquet",
            boundary_lines_file=tmp_path / "out" / "fields_boundary_lines.parquet",
            chips_base_dir=chips_base_dir,
            stac_result=STACGenerationResult(
                catalog_path=tmp_path / "out" / "catalog.json",
                source_collection_path=tmp_path / "out" / "fields-source" / "collection.json",
                chips_collection_path=chips_base_dir / "collection.json",
                items_parquet_path=chips_base_dir / "items.parquet",
                total_items=3,
                temporal_extent=(
                    datetime(2023, 1, 1, tzinfo=UTC),
                    datetime(2023, 12, 31, tzinfo=UTC),
                ),
            ),
        )

    stub = SelectionStub()

    def fake_select(**kwargs: Any) -> SelectionWorkflowResult:
        stub.calls.append(kwargs)
        if stub.error is not None:
            raise stub.error
        return stub.result

    monkeypatch.setattr(create_dataset_module.dataset, "create_dataset", fake_create_dataset)
    monkeypatch.setattr(create_dataset_module, "select_imagery_for_catalog", fake_select)
    return stub


def _invoke(fields_file: Path, *extra: str, year: str | None = "2023") -> Result:
    """Run create-dataset with the required options plus any extras."""
    args = ["create-dataset", str(fields_file), "--split-type", "random-uniform"]
    if year is not None:
        args += ["--year", year]
    return CliRunner().invoke(cli, [*args, *extra])


class TestCreateDatasetImageSelection:
    """Tests for how create-dataset delegates image selection."""

    def test_uses_shared_selection_workflow(
        self, stub_pipeline: SelectionStub, sample_fields_geoparquet: Path
    ) -> None:
        """Selection is delegated to select_imagery_for_catalog with CLI options."""
        result = _invoke(sample_fields_geoparquet)

        assert result.exit_code == 0, result.output
        assert len(stub_pipeline.calls) == 1
        kwargs = stub_pipeline.calls[0]
        assert kwargs["year"] == 2023
        assert kwargs["cloud_cover_chip"] == 2.0
        assert kwargs["buffer_days"] == 14
        assert kwargs["catalog_dir"].name == "fields-chips"

    def test_passes_through_selection_options(
        self, stub_pipeline: SelectionStub, sample_fields_geoparquet: Path
    ) -> None:
        """Non-default imagery options reach the workflow."""
        result = _invoke(
            sample_fields_geoparquet,
            "--cloud-cover-chip",
            "10",
            "--nodata-max",
            "5",
            "--buffer-days",
            "21",
            "--num-buffer-expansions",
            "1",
            "--buffer-expansion-size",
            "7",
        )

        assert result.exit_code == 0, result.output
        kwargs = stub_pipeline.calls[0]
        assert kwargs["cloud_cover_chip"] == 10.0
        assert kwargs["nodata_max"] == 5.0
        assert kwargs["buffer_days"] == 21
        assert kwargs["num_buffer_expansions"] == 1
        assert kwargs["buffer_expansion_size"] == 7

    def test_skips_existing_selections_by_default(
        self, stub_pipeline: SelectionStub, sample_fields_geoparquet: Path
    ) -> None:
        """Chips with existing selections are skipped unless forced (issue #32)."""
        result = _invoke(sample_fields_geoparquet)

        assert result.exit_code == 0, result.output
        assert stub_pipeline.calls[0]["force"] is False

    def test_force_image_selection_overwrites(
        self, stub_pipeline: SelectionStub, sample_fields_geoparquet: Path
    ) -> None:
        """--force-image-selection re-selects chips that already have imagery."""
        result = _invoke(sample_fields_geoparquet, "--force-image-selection")

        assert result.exit_code == 0, result.output
        assert stub_pipeline.calls[0]["force"] is True

    @pytest.mark.usefixtures("stub_pipeline")
    def test_reports_selection_counts(self, sample_fields_geoparquet: Path) -> None:
        """Summary reports counts from the workflow result."""
        result = _invoke(sample_fields_geoparquet)

        assert result.exit_code == 0, result.output
        assert "Selected: 2" in result.output
        assert "Skipped: 1" in result.output
        assert "Failed:" not in result.output

    def test_skip_images_does_not_select(
        self, stub_pipeline: SelectionStub, sample_fields_geoparquet: Path
    ) -> None:
        """--skip-images bypasses selection entirely."""
        result = _invoke(sample_fields_geoparquet, "--skip-images")

        assert result.exit_code == 0, result.output
        assert stub_pipeline.calls == []

    def test_download_images_forces_selection(
        self, stub_pipeline: SelectionStub, sample_fields_geoparquet: Path
    ) -> None:
        """--download-images still selects, even alongside --skip-images."""
        result = _invoke(sample_fields_geoparquet, "--skip-images", "--download-images")

        assert result.exit_code == 0, result.output
        assert len(stub_pipeline.calls) == 1
        assert "Downloaded: 0" in result.output


class TestCreateDatasetImageSelectionErrors:
    """Error and edge cases for image selection."""

    def test_reports_failed_chips(
        self, stub_pipeline: SelectionStub, sample_fields_geoparquet: Path
    ) -> None:
        """Failed chips are reported without failing the run."""
        stub_pipeline.result = SelectionWorkflowResult(successful=1, skipped=0, failed=2)

        result = _invoke(sample_fields_geoparquet)

        assert result.exit_code == 0, result.output
        assert "Failed: 2" in result.output

    def test_reports_empty_catalog(
        self, stub_pipeline: SelectionStub, sample_fields_geoparquet: Path
    ) -> None:
        """A catalog with nothing to select reports zeros rather than erroring."""
        stub_pipeline.result = SelectionWorkflowResult()

        result = _invoke(sample_fields_geoparquet)

        assert result.exit_code == 0, result.output
        assert "Selected: 0" in result.output

    def test_selection_value_error_exits_nonzero(
        self, stub_pipeline: SelectionStub, sample_fields_geoparquet: Path
    ) -> None:
        """A ValueError from the workflow aborts with exit 1 and a readable message."""
        stub_pipeline.error = ValueError("No cloud-free scenes for chip")

        result = _invoke(sample_fields_geoparquet)

        assert result.exit_code == 1
        assert "No cloud-free scenes for chip" in result.output

    def test_keyboard_interrupt_exits_130(
        self, stub_pipeline: SelectionStub, sample_fields_geoparquet: Path
    ) -> None:
        """Interrupting selection exits 130 instead of dumping a traceback."""
        stub_pipeline.error = KeyboardInterrupt()

        result = _invoke(sample_fields_geoparquet)

        assert result.exit_code == 130
        assert "Interrupted by user" in result.output

    @pytest.mark.usefixtures("stub_pipeline")
    def test_year_required_when_not_derivable(self, tmp_path: Path) -> None:
        """Selection without --year fails when fields have no datetime column."""
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(10, 50, 10.01, 50.01)], crs="EPSG:4326")
        fields_file = tmp_path / "no_datetime.parquet"
        gdf.to_parquet(fields_file)

        result = _invoke(fields_file, year=None)

        assert result.exit_code != 0
        assert "--year is required for image selection" in result.output

    def test_invalid_cloud_cover_rejected(self, sample_fields_geoparquet: Path) -> None:
        """Out-of-range cloud cover is rejected before any work happens."""
        result = _invoke(sample_fields_geoparquet, "--cloud-cover-chip", "150")

        assert result.exit_code != 0
