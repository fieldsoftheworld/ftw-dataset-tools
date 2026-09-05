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


class TestMaskTypeRegistries:
    """A mask type has to be registered in several places to actually work.

    Adding one to VALID_MASK_TYPES but forgetting the pipeline mapping makes
    ``create-dataset`` silently skip it; forgetting the STAC map makes the file
    get written but never referenced by any item. These guard that drift.
    """

    def test_every_valid_mask_type_is_in_the_pipeline_mapping(self) -> None:
        from ftw_dataset_tools.api.config import VALID_MASK_TYPES
        from ftw_dataset_tools.api.pipeline import _MASK_TYPE_MAPPING

        mapped = {type_name for _, _, type_name in _MASK_TYPE_MAPPING}
        assert mapped == set(VALID_MASK_TYPES)

    def test_every_valid_mask_type_is_a_mask_type_enum_member(self) -> None:
        from ftw_dataset_tools.api.config import VALID_MASK_TYPES
        from ftw_dataset_tools.api.masks import MaskType

        assert {m.value for m in MaskType} == set(VALID_MASK_TYPES)

    def test_every_mask_type_can_become_a_stac_asset(self) -> None:
        from ftw_dataset_tools.api.pipeline import _MASK_TYPE_MAPPING
        from ftw_dataset_tools.api.stac import _get_mask_title

        for mask_type, subdir_name, _ in _MASK_TYPE_MAPPING:
            title = _get_mask_title(subdir_name)
            # The fallback title means the type was never given a real one.
            assert title != f"{subdir_name} mask", f"{mask_type.value} has no STAC title"

    def test_defaults_are_a_subset_of_valid_types(self) -> None:
        from ftw_dataset_tools.api.config import DEFAULT_MASK_TYPES, VALID_MASK_TYPES

        assert set(DEFAULT_MASK_TYPES) <= set(VALID_MASK_TYPES)

    def test_derived_mask_types_agree_between_config_and_masks(self) -> None:
        # config.py holds the string copy so validation need not import api.masks;
        # a mismatch means config would enforce a dependency masks does not honour.
        from ftw_dataset_tools.api.config import DERIVED_MASK_SOURCE, DERIVED_MASK_TYPES
        from ftw_dataset_tools.api.masks import _DERIVED_MASK_TYPES, MaskType

        assert {m.value for m in _DERIVED_MASK_TYPES} == set(DERIVED_MASK_TYPES)
        assert MaskType.SEMANTIC_2_CLASS.value == DERIVED_MASK_SOURCE

    def test_derived_mask_types_are_valid_types(self) -> None:
        from ftw_dataset_tools.api.config import (
            DERIVED_MASK_SOURCE,
            DERIVED_MASK_TYPES,
            VALID_MASK_TYPES,
        )

        assert set(DERIVED_MASK_TYPES) <= set(VALID_MASK_TYPES)
        assert DERIVED_MASK_SOURCE in VALID_MASK_TYPES

    def test_every_mask_type_is_in_the_stac_asset_registry(self) -> None:
        from ftw_dataset_tools.api.pipeline import _MASK_TYPE_MAPPING
        from ftw_dataset_tools.api.stac import _MASK_TYPE_BY_ASSET_NAME

        # A mask type missing here gets written to disk but silently dropped
        # from the STAC items.
        expected = {subdir_name: mask_type for mask_type, subdir_name, _ in _MASK_TYPE_MAPPING}
        assert expected == _MASK_TYPE_BY_ASSET_NAME


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


class TestLocalGridSubset:
    """Tests for bbox-subsetting a local grid before chips loads it."""

    def _ctx(self, fields_path: Path, out_dir: Path) -> pipeline.PipelineContext:
        config = DatasetConfig.from_dict({"fields_file": str(fields_path)})
        ctx = pipeline.PipelineContext(
            config=config,
            fields_input=fields_path,
            output_dir=out_dir,
            field_dataset="t",
            effective_year=None,
            has_temporal=False,
        )
        ctx.field_polygons_path = fields_path  # fields file carries a bbox column
        out_dir.mkdir(parents=True, exist_ok=True)
        return ctx

    def _write_grid(self, path: Path) -> None:
        import duckdb

        conn = duckdb.connect()
        # Three cells; only the first overlaps the fields' [0.2,0.2,0.8,0.8] extent.
        conn.execute(
            f"""
            COPY (
                SELECT * FROM (VALUES
                    ('a', {{'xmin': 0.0, 'ymin': 0.0, 'xmax': 1.0, 'ymax': 1.0}}),
                    ('b', {{'xmin': 1.0, 'ymin': 0.0, 'xmax': 2.0, 'ymax': 1.0}}),
                    ('c', {{'xmin': 5.0, 'ymin': 5.0, 'xmax': 6.0, 'ymax': 6.0}})
                ) AS t(id, bbox)
            ) TO '{path}' (FORMAT PARQUET)
            """
        )
        conn.close()

    def _write_fields(self, path: Path) -> None:
        import duckdb

        conn = duckdb.connect()
        conn.execute(
            f"""
            COPY (SELECT 1 AS id, {{'xmin': 0.2, 'ymin': 0.2, 'xmax': 0.8, 'ymax': 0.8}} AS bbox)
            TO '{path}' (FORMAT PARQUET)
            """
        )
        conn.close()

    def test_subset_keeps_only_overlapping_cells(self, tmp_path: Path) -> None:
        import duckdb

        fields = tmp_path / "fields.parquet"
        grid = tmp_path / "grid.parquet"
        self._write_fields(fields)
        self._write_grid(grid)
        ctx = self._ctx(fields, tmp_path / "out")

        subset = pipeline._subset_local_grid(ctx, str(grid))
        ids = [r[0] for r in duckdb.connect().execute(f"SELECT id FROM '{subset}'").fetchall()]
        assert ids == ["a"]  # only the overlapping cell survives

    def test_grid_without_bbox_is_passed_through(self, tmp_path: Path) -> None:
        import duckdb

        fields = tmp_path / "fields.parquet"
        self._write_fields(fields)
        grid = tmp_path / "nobbox.parquet"
        duckdb.connect().execute(f"COPY (SELECT 'a' AS id) TO '{grid}' (FORMAT PARQUET)")
        ctx = self._ctx(fields, tmp_path / "out")
        # No bbox column -> return the original path unchanged (no subset written).
        assert pipeline._subset_local_grid(ctx, str(grid)) == str(grid)


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


class TestMasksStage:
    """stage_masks wiring: one pass over the chips, one result per requested type."""

    @staticmethod
    def _write_stage_inputs(ctx: pipeline.PipelineContext) -> None:
        """Write the chips, field polygons and boundary lines stage_masks reads."""
        from shapely.geometry import LineString

        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        ctx.chips_base_dir.mkdir(parents=True, exist_ok=True)

        cell = box(18.0, 40.0, 18.01, 40.01)
        gpd.GeoDataFrame(
            {"id": ["cell_001"], "field_coverage_pct": [20.0]},
            geometry=[cell],
            crs="EPSG:4326",
        ).to_parquet(ctx.chips_path)

        fields = [box(18.002, 40.002, 18.004, 40.004), box(18.006, 40.002, 18.008, 40.004)]
        gpd.GeoDataFrame({"id": [1, 2]}, geometry=fields, crs="EPSG:4326").to_parquet(
            ctx.field_polygons_path
        )
        gpd.GeoDataFrame(
            {"id": [1, 2]},
            geometry=[LineString(f.exterior.coords) for f in fields],
            crs="EPSG:4326",
        ).to_parquet(ctx.boundary_lines_path)

    def test_produces_a_result_and_file_per_requested_type(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        mask_types = ["semantic_2_class", "decode_boundary", "decode_distance"]
        config = _config(
            sample_geoparquet_4326,
            tmp_path / "out",
            year=2024,
            stages={"masks": {"mask_types": mask_types, "workers": 1}},
        )
        ctx = pipeline.build_context(config)
        self._write_stage_inputs(ctx)

        pipeline.stage_masks(ctx)

        # Keyed by the STAC asset name, which is what stage_stac later looks up.
        assert set(ctx.masks_results) == {"semantic_2class", "decode_boundary", "decode_distance"}
        for subdir_name, result in ctx.masks_results.items():
            assert result.total_created == 1, (subdir_name, result.masks_skipped)
            assert result.masks_created[0].output_path.exists()

    def test_derived_layers_land_beside_the_mask_they_derive_from(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        """All three outputs belong to the same chip directory."""
        config = _config(
            sample_geoparquet_4326,
            tmp_path / "out",
            year=2024,
            stages={
                "masks": {
                    "mask_types": ["semantic_2_class", "decode_distance"],
                    "workers": 1,
                }
            },
        )
        ctx = pipeline.build_context(config)
        self._write_stage_inputs(ctx)

        pipeline.stage_masks(ctx)

        paths = {
            name: result.masks_created[0].output_path for name, result in ctx.masks_results.items()
        }
        assert paths["semantic_2class"].parent == paths["decode_distance"].parent
        assert paths["semantic_2class"].name == "cell_001_2024_semantic_2_class.tif"
        assert paths["decode_distance"].name == "cell_001_2024_decode_distance.tif"
