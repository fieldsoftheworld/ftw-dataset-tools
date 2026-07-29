"""Tests for the stage-based pipeline orchestration."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import pytest
from shapely.geometry import box

from ftw_dataset_tools.api import pipeline
from ftw_dataset_tools.api.config import ClassFilter, ClassFilterError, DatasetConfig
from ftw_dataset_tools.api.pipeline import StageInputError

if TYPE_CHECKING:
    from pathlib import Path


def _config(fields_file: Path, output_dir: Path, **kwargs: object) -> DatasetConfig:
    data: dict = {"fields_file": str(fields_file), "output_dir": str(output_dir)}
    data.update(kwargs)  # type: ignore[arg-type]
    return DatasetConfig.from_dict(data)


class TestResolveStages:
    """Tests for stage selection logic."""

    def test_full_run_gates_disabled_imagery(self) -> None:
        config = DatasetConfig.from_dict({"fields_file": "f.parquet"})
        stages = pipeline.resolve_stages(config=config)
        # select_images defaults enabled; download_images defaults disabled.
        assert stages == [
            "reproject",
            "chips",
            "splits",
            "boundaries",
            "masks",
            "stac",
            "select_images",
        ]

    def test_download_enabled_included(self) -> None:
        config = DatasetConfig.from_dict(
            {"fields_file": "f.parquet", "stages": {"download_images": {"enabled": True}}}
        )
        assert "download_images" in pipeline.resolve_stages(config=config)

    def test_only_forces_single_stage(self) -> None:
        config = DatasetConfig.from_dict(
            {"fields_file": "f.parquet", "stages": {"download_images": {"enabled": False}}}
        )
        # --only forces a stage even if its config is disabled.
        assert pipeline.resolve_stages(only="download_images", config=config) == ["download_images"]

    def test_from_and_through(self) -> None:
        config = DatasetConfig.from_dict({"fields_file": "f.parquet"})
        assert pipeline.resolve_stages(from_stage="masks", config=config) == [
            "masks",
            "stac",
            "select_images",
        ]
        assert pipeline.resolve_stages(through_stage="chips", config=config) == [
            "reproject",
            "chips",
        ]

    def test_unknown_stage_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown stage"):
            pipeline.resolve_stages(only="bogus")


class TestBuildContext:
    """Tests for context construction and temporal detection."""

    def test_missing_input_raises(self, tmp_path: Path) -> None:
        config = _config(tmp_path / "nope.parquet", tmp_path / "out")
        with pytest.raises(FileNotFoundError, match="Fields file not found"):
            pipeline.build_context(config)

    def test_derived_paths_and_name(self, sample_geoparquet_4326: Path, tmp_path: Path) -> None:
        out = tmp_path / "out"
        config = _config(sample_geoparquet_4326, out, name="myds", year=2023)
        ctx = pipeline.build_context(config)
        assert ctx.field_dataset == "myds"
        assert ctx.output_fields_path == out.resolve() / "myds_fields.parquet"
        assert ctx.chips_path == out.resolve() / "myds_chips.parquet"
        assert ctx.boundary_lines_path == out.resolve() / "myds_boundary_lines.parquet"
        assert ctx.chips_base_dir == out.resolve() / "myds-chips"
        assert ctx.effective_year == 2023
        assert ctx.has_temporal is True

    def test_name_defaults_to_stem(self, sample_geoparquet_4326: Path, tmp_path: Path) -> None:
        config = _config(sample_geoparquet_4326, tmp_path / "out", year=2023)
        ctx = pipeline.build_context(config)
        assert ctx.field_dataset == sample_geoparquet_4326.stem

    def test_no_temporal_without_year_or_column(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        config = _config(sample_geoparquet_4326, tmp_path / "out")
        ctx = pipeline.build_context(config)
        assert ctx.has_temporal is False
        assert ctx.effective_year is None

    def test_build_context_does_not_create_output_dir(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        config = _config(sample_geoparquet_4326, out, year=2023)
        pipeline.build_context(config)
        # Directory creation is deferred to run_pipeline (after validation).
        assert not out.exists()


class TestStageValidation:
    """Tests for front-loaded validation in run_pipeline."""

    def test_splits_requires_split_type(self, sample_geoparquet_4326: Path, tmp_path: Path) -> None:
        config = _config(sample_geoparquet_4326, tmp_path / "out", year=2023)
        ctx = pipeline.build_context(config)
        with pytest.raises(ValueError, match="split_type is required"):
            pipeline.run_pipeline(ctx, ["splits"])

    def test_year_stage_requires_temporal(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        config = _config(sample_geoparquet_4326, tmp_path / "out")
        ctx = pipeline.build_context(config)
        with pytest.raises(ValueError, match="Cannot determine temporal extent"):
            pipeline.run_pipeline(ctx, ["stac"])

    def test_validation_runs_before_output_dir_created(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        config = _config(sample_geoparquet_4326, out)
        ctx = pipeline.build_context(config)
        with pytest.raises(ValueError, match="Cannot determine temporal extent"):
            pipeline.run_pipeline(ctx, ["masks"])
        assert not out.exists()


class TestStageInputErrors:
    """Standalone stages should error clearly when inputs are missing."""

    def test_chips_requires_reprojected_fields(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        config = _config(sample_geoparquet_4326, tmp_path / "out", year=2023)
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir(parents=True)
        with pytest.raises(StageInputError, match="reproject"):
            pipeline.stage_chips(ctx)

    def test_masks_requires_chips(self, sample_geoparquet_4326: Path, tmp_path: Path) -> None:
        config = _config(sample_geoparquet_4326, tmp_path / "out", year=2023)
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir(parents=True)
        with pytest.raises(StageInputError, match="chips"):
            pipeline.stage_masks(ctx)


class TestReprojectStage:
    """Tests for the reproject stage (no network required)."""

    def test_copies_4326_input(self, sample_geoparquet_4326: Path, tmp_path: Path) -> None:
        config = _config(sample_geoparquet_4326, tmp_path / "out", year=2023)
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir(parents=True)
        pipeline.stage_reproject(ctx)
        assert ctx.output_fields_path.exists()
        assert ctx.was_reprojected is False

    def test_skip_reproject_errors_on_non_4326(
        self, sample_geoparquet_3035: Path, tmp_path: Path
    ) -> None:
        config = _config(sample_geoparquet_3035, tmp_path / "out", year=2023, skip_reproject=True)
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir(parents=True)
        with pytest.raises(ValueError, match="EPSG:4326 is required"):
            pipeline.stage_reproject(ctx)


class TestRunPipelineProvenance:
    """End-to-end (no network): run only the reproject stage via run_pipeline."""

    def test_writes_provenance_and_runs_reproject(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        out = tmp_path / "out"
        config = _config(sample_geoparquet_4326, out, name="ds", year=2023)
        provenance = config.provenance_dict()
        ctx = pipeline.build_context(config, provenance=provenance)
        pipeline.run_pipeline(ctx, ["reproject"])

        assert ctx.output_fields_path.exists()
        prov_file = ctx.output_dir / "ftwd-config.resolved.yaml"
        assert prov_file.exists()


def _fields_with_classes(tmp_path: Path) -> Path:
    gdf = gpd.GeoDataFrame(
        {"id": [1, 2, 3], "crop": ["wheat", "water", "maize"]},
        geometry=[
            box(10.0, 50.0, 10.01, 50.01),
            box(10.02, 50.0, 10.03, 50.01),
            box(10.0, 50.02, 10.01, 50.03),
        ],
        crs="EPSG:4326",
    )
    path = tmp_path / "fields_crop.parquet"
    gdf.to_parquet(path)
    return path


class TestFilterStage:
    """Tests for the optional class-filter stage (no network required)."""

    def test_field_polygons_path_switches_with_filter(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        config = _config(sample_geoparquet_4326, tmp_path / "out", name="ds", year=2023)
        # Without a filter, downstream reads the full reprojected fields file.
        ctx = pipeline.build_context(config)
        assert ctx.field_polygons_path == ctx.output_fields_path
        assert ctx.field_polygons_producer == "reproject"

        # With a filter, downstream reads the filtered file.
        config.class_filter = ClassFilter("crop", ["wheat"], ["water"])
        ctx2 = pipeline.build_context(config)
        assert ctx2.field_polygons_path.name == "ds_fields_filtered.parquet"
        assert ctx2.field_polygons_producer == "filter"

    def test_resolve_stages_gates_filter_on_config(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        config = _config(sample_geoparquet_4326, tmp_path / "out", year=2023)
        assert "filter" not in pipeline.resolve_stages(config=config)
        config.class_filter = ClassFilter("crop", ["wheat"], ["water"])
        assert "filter" in pipeline.resolve_stages(config=config)

    def test_run_reproject_then_filter_writes_filtered(self, tmp_path: Path) -> None:
        fields = _fields_with_classes(tmp_path)
        config = _config(fields, tmp_path / "out", name="ds", year=2023)
        config.class_filter = ClassFilter("crop", ["wheat", "maize"], ["water"])
        ctx = pipeline.build_context(config)
        pipeline.run_pipeline(ctx, ["reproject", "filter"])

        assert ctx.output_fields_path.exists()  # full source preserved
        assert ctx.field_polygons_path.exists()  # filtered field polygons
        import duckdb

        rows = (
            duckdb.connect()
            .execute(f"SELECT DISTINCT crop FROM '{ctx.field_polygons_path}'")
            .fetchall()
        )
        assert {r[0] for r in rows} == {"wheat", "maize"}

    def test_unhandled_class_aborts(self, tmp_path: Path) -> None:
        fields = _fields_with_classes(tmp_path)  # has wheat/water/maize
        config = _config(fields, tmp_path / "out", name="ds", year=2023)
        config.class_filter = ClassFilter("crop", ["wheat"], ["water"])  # maize unhandled
        ctx = pipeline.build_context(config)
        with pytest.raises(ClassFilterError, match="not covered"):
            pipeline.run_pipeline(ctx, ["reproject", "filter"])

    def test_masks_requires_filtered_fields_when_configured(self, tmp_path: Path) -> None:
        fields = _fields_with_classes(tmp_path)
        config = _config(fields, tmp_path / "out", name="ds", year=2023)
        config.class_filter = ClassFilter("crop", ["wheat", "maize"], ["water"])
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir(parents=True)
        ctx.chips_path.touch()  # chips present so we reach the field-polygons check
        # filtered fields do not exist yet -> error points at the filter stage.
        with pytest.raises(StageInputError, match="filter"):
            pipeline.stage_masks(ctx)
