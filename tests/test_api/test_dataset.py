"""Tests for the dataset API module."""

from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box


class TestCreateDatasetResult:
    """Tests for CreateDatasetResult dataclass."""

    def test_dataclass_fields(self) -> None:
        """Test CreateDatasetResult has expected fields."""
        from ftw_dataset_tools.api.dataset import CreateDatasetResult

        result = CreateDatasetResult(
            output_dir=Path("/tmp/output"),
            field_dataset="test_dataset",
            fields_file=Path("/tmp/fields.parquet"),
            chips_file=Path("/tmp/chips.parquet"),
            boundary_lines_file=Path("/tmp/boundary_lines.parquet"),
        )
        assert result.output_dir == Path("/tmp/output")
        assert result.field_dataset == "test_dataset"

    def test_total_masks_created(self) -> None:
        """Test total_masks_created property sums all mask types."""
        from ftw_dataset_tools.api.dataset import CreateDatasetResult
        from ftw_dataset_tools.api.masks import CreateMasksResult, MaskResult

        result = CreateDatasetResult(
            output_dir=Path("/tmp"),
            field_dataset="test",
            fields_file=Path("/tmp/fields.parquet"),
            chips_file=Path("/tmp/chips.parquet"),
            boundary_lines_file=Path("/tmp/boundary_lines.parquet"),
            masks_results={
                "instance": CreateMasksResult(
                    masks_created=[
                        MaskResult("a", Path("a.tif"), 512, 512),
                    ],
                    masks_skipped=[],
                    field_dataset="test",
                ),
                "semantic_2class": CreateMasksResult(
                    masks_created=[
                        MaskResult("a", Path("a.tif"), 512, 512),
                        MaskResult("b", Path("b.tif"), 512, 512),
                    ],
                    masks_skipped=[],
                    field_dataset="test",
                ),
            },
        )
        assert result.total_masks_created == 3


class TestCreateDatasetInputValidation:
    """Tests for create_dataset input validation."""

    def test_fields_file_not_found(self, tmp_path: Path) -> None:
        """Test FileNotFoundError for missing fields file."""
        from ftw_dataset_tools.api.dataset import create_dataset

        with pytest.raises(FileNotFoundError, match="Fields file not found"):
            create_dataset("/nonexistent/fields.parquet", output_dir=tmp_path / "out")

    def test_year_required_without_datetime_column(self, tmp_path: Path) -> None:
        """Test ValueError when year not provided and no datetime column."""
        from ftw_dataset_tools.api.dataset import create_dataset

        # Create fields file without determination_datetime
        gdf = gpd.GeoDataFrame({"id": [1]}, geometry=[box(10, 50, 10.01, 50.01)], crs="EPSG:4326")
        fields_file = tmp_path / "fields.parquet"
        gdf.to_parquet(fields_file)

        with pytest.raises(ValueError, match="Cannot determine temporal extent"):
            create_dataset(
                fields_file, output_dir=tmp_path / "out", year=None, split_type="random-uniform"
            )

    def test_skip_reproject_error_non_4326(self, tmp_path: Path) -> None:
        """Test ValueError when skip_reproject=True with non-4326 input."""
        from ftw_dataset_tools.api.dataset import create_dataset

        # Create fields file in EPSG:3035
        gdf = gpd.GeoDataFrame(
            {"id": [1]},
            geometry=[box(5150000, 3540000, 5160000, 3550000)],
            crs="EPSG:3035",
        )
        fields_file = tmp_path / "fields_3035.parquet"
        gdf.to_parquet(fields_file)

        with pytest.raises(ValueError, match="EPSG:4326 is required"):
            create_dataset(
                fields_file,
                output_dir=tmp_path / "out",
                year=2023,
                skip_reproject=True,
                split_type="random-uniform",
            )


class TestCreateDatasetResultProperties:
    """Tests for CreateDatasetResult dataclass properties."""

    def test_total_masks_created_empty(self) -> None:
        """Test total_masks_created with empty masks_results."""
        from ftw_dataset_tools.api.dataset import CreateDatasetResult

        result = CreateDatasetResult(
            output_dir=Path("/tmp"),
            field_dataset="test",
            fields_file=Path("/tmp/fields.parquet"),
            chips_file=Path("/tmp/chips.parquet"),
            boundary_lines_file=Path("/tmp/boundary_lines.parquet"),
            masks_results={},
        )
        assert result.total_masks_created == 0

    def test_result_optional_fields(self) -> None:
        """Test CreateDatasetResult optional fields have correct defaults."""
        from ftw_dataset_tools.api.dataset import CreateDatasetResult

        result = CreateDatasetResult(
            output_dir=Path("/tmp"),
            field_dataset="test",
            fields_file=Path("/tmp/fields.parquet"),
            chips_file=Path("/tmp/chips.parquet"),
            boundary_lines_file=Path("/tmp/boundary_lines.parquet"),
        )
        assert result.chips_base_dir is None
        assert result.was_reprojected is False
        assert result.source_crs is None
        assert result.chips_result is None
        assert result.boundaries_result is None
        assert result.stac_result is None


class TestCreateDatasetSignature:
    """Tests for create_dataset function signature."""

    def test_function_accepts_all_parameters(self) -> None:
        """Test that create_dataset accepts all expected parameters."""
        import inspect

        from ftw_dataset_tools.api.dataset import create_dataset

        sig = inspect.signature(create_dataset)
        param_names = list(sig.parameters.keys())

        assert "fields_file" in param_names
        assert "output_dir" in param_names
        assert "field_dataset" in param_names
        assert "min_coverage" in param_names
        assert "resolution" in param_names
        assert "num_workers" in param_names
        assert "skip_reproject" in param_names
        assert "year" in param_names
        assert "on_progress" in param_names
        assert "on_mask_progress" in param_names
        assert "on_mask_start" in param_names

    def test_default_values(self) -> None:
        """Test that default parameter values are correct."""
        import inspect

        from ftw_dataset_tools.api.dataset import create_dataset

        sig = inspect.signature(create_dataset)

        assert sig.parameters["output_dir"].default == "./dataset"
        assert sig.parameters["min_coverage"].default == 0.01
        assert sig.parameters["resolution"].default == 10.0
        assert sig.parameters["skip_reproject"].default is False


class TestCreateDatasetChecksumsFlag:
    def test_checksums_kwarg_reaches_config(self, monkeypatch, tmp_path: Path) -> None:
        from ftw_dataset_tools.api import dataset as dataset_module

        captured: dict = {}

        def fake_run_pipeline(ctx, _stages, **_kwargs):
            captured["checksums"] = ctx.config.stages.stac.checksums
            return ctx

        monkeypatch.setattr(dataset_module.pipeline, "run_pipeline", fake_run_pipeline)

        import geopandas as gpd
        from shapely.geometry import box

        fields = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        fields_path = tmp_path / "f.parquet"
        fields.to_parquet(fields_path)

        dataset_module.create_dataset(
            fields_file=fields_path,
            output_dir=tmp_path / "out",
            split_type="random-uniform",
            year=2024,
            checksums=True,
        )

        assert captured["checksums"] is True


class TestCreateDatasetStageOrder:
    """create-dataset must document the collection after its imagery, like ``ftwd run``.

    It runs image selection and download through the ``on_imagery`` hook rather than
    through the imagery stages, so the only thing keeping the two entry points in
    step is STAGE_ORDER. This fails if either ordering is hand-written again.
    """

    def test_docs_run_after_the_imagery_hook(self, monkeypatch, tmp_path: Path) -> None:
        from ftw_dataset_tools.api import dataset as dataset_module
        from ftw_dataset_tools.api import pipeline

        fields = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        fields_path = tmp_path / "f.parquet"
        fields.to_parquet(fields_path)

        seen: list[str] = []
        for name in pipeline.STAGE_ORDER:
            monkeypatch.setitem(
                pipeline._STAGE_FUNCS, name, lambda _ctx, name=name: seen.append(name)
            )

        dataset_module.create_dataset(
            fields_file=fields_path,
            output_dir=tmp_path / "out",
            split_type="random-uniform",
            year=2024,
            on_imagery=lambda _collection_dir: seen.append("imagery"),
        )

        # The stages themselves keep STAGE_ORDER; from_kwargs disables filter and the
        # imagery stages, which the hook stands in for.
        assert [s for s in seen if s != "imagery"] == [
            s for s in pipeline.STAGE_ORDER if s != "filter" and s not in pipeline.IMAGERY_STAGES
        ]
        assert seen.index("stac") < seen.index("imagery") < seen.index("docs")

    def test_no_hook_means_no_imagery_step(self, monkeypatch, tmp_path: Path) -> None:
        from ftw_dataset_tools.api import dataset as dataset_module
        from ftw_dataset_tools.api import pipeline

        fields = gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326")
        fields_path = tmp_path / "f.parquet"
        fields.to_parquet(fields_path)

        seen: list[str] = []
        for name in pipeline.STAGE_ORDER:
            monkeypatch.setitem(
                pipeline._STAGE_FUNCS, name, lambda _ctx, name=name: seen.append(name)
            )

        dataset_module.create_dataset(
            fields_file=fields_path,
            output_dir=tmp_path / "out",
            split_type="random-uniform",
            year=2024,
        )

        assert "docs" in seen and "imagery" not in seen
