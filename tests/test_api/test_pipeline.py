"""Tests for the stage-based pipeline orchestration."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import duckdb
import geopandas as gpd
import pytest
from shapely.geometry import box

from ftw_dataset_tools.api import field_stats, pipeline
from ftw_dataset_tools.api.config import ClassFilter, ClassFilterError, DatasetConfig
from ftw_dataset_tools.api.pipeline import StageInputError


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
        assert ctx.chips_base_dir == out.resolve() / "chips"
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


class TestStacStageFlags:
    def test_stac_stage_passes_checksums_and_background(
        self, sample_geoparquet_4326: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from ftw_dataset_tools.api import pipeline, stac
        from ftw_dataset_tools.api.config import DatasetConfig

        captured: dict = {}

        def fake_generate(**kwargs):
            captured.update(kwargs)
            return stac.STACGenerationResult(
                collection_path=tmp_path / "collection.json",
                items_parquet_path=tmp_path / "items.parquet",
                subcatalog_paths={},
                total_items=0,
                temporal_extent=(
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 12, 31, tzinfo=UTC),
                ),
            )

        monkeypatch.setattr(stac, "generate_stac_catalog", fake_generate)

        config = DatasetConfig.from_dict(
            {
                "fields_file": str(sample_geoparquet_4326),
                "output_dir": str(tmp_path / "out"),
                "year": 2024,
                "stages": {"stac": {"checksums": True}, "masks": {"presence_only": True}},
            }
        )
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir()
        for name in ("chips", "fields", "boundary_lines"):
            (ctx.output_dir / f"{ctx.field_dataset}_{name}.parquet").write_bytes(b"")

        pipeline.stage_stac(ctx)

        assert captured["checksums"] is True
        assert captured["background_class_value"] == 3

    def test_stac_stage_passes_config(
        self, sample_geoparquet_4326: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from ftw_dataset_tools.api import pipeline, stac
        from ftw_dataset_tools.api.config import DatasetConfig

        captured: dict = {}

        def fake_generate(**kwargs):
            captured.update(kwargs)
            return stac.STACGenerationResult(
                collection_path=tmp_path / "collection.json",
                items_parquet_path=tmp_path / "items.parquet",
                subcatalog_paths={},
                total_items=0,
                temporal_extent=(
                    datetime(2024, 1, 1, tzinfo=UTC),
                    datetime(2024, 12, 31, tzinfo=UTC),
                ),
            )

        monkeypatch.setattr(stac, "generate_stac_catalog", fake_generate)
        config = DatasetConfig.from_dict(
            {
                "fields_file": str(sample_geoparquet_4326),
                "output_dir": str(tmp_path / "out"),
                "year": 2024,
            }
        )
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir()
        for name in ("chips", "fields", "boundary_lines"):
            (ctx.output_dir / f"{ctx.field_dataset}_{name}.parquet").write_bytes(b"")

        pipeline.stage_stac(ctx)

        assert captured["config"] is config


class TestChipDirLayout:
    def test_chips_base_dir_is_chips_subdir(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        from ftw_dataset_tools.api import pipeline
        from ftw_dataset_tools.api.config import DatasetConfig

        config = DatasetConfig.from_dict(
            {
                "fields_file": str(sample_geoparquet_4326),
                "output_dir": str(tmp_path / "out"),
                "year": 2024,
            }
        )
        ctx = pipeline.build_context(config)

        assert ctx.chips_base_dir == (tmp_path / "out").resolve() / "chips"

    def test_build_chip_dirs_nests_by_square(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        import geopandas as gpd
        from shapely.geometry import box

        from ftw_dataset_tools.api import pipeline
        from ftw_dataset_tools.api.config import DatasetConfig

        config = DatasetConfig.from_dict(
            {
                "fields_file": str(sample_geoparquet_4326),
                "output_dir": str(tmp_path / "out"),
                "year": 2024,
            }
        )
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir()
        chips = gpd.GeoDataFrame(
            {
                "id": ["ftw-33UXP0410", "ftw-33UXQ0001", "grid_001"],
                "field_coverage_pct": [5.0, 5.0, 5.0],
            },
            geometry=[box(0, 0, 1, 1)] * 3,
            crs="EPSG:4326",
        )
        chips.to_parquet(ctx.chips_path)

        dirs = pipeline._build_chip_dirs(ctx)

        assert dirs["ftw-33UXP0410_2024"] == ctx.chips_base_dir / "33UXP" / "ftw-33UXP0410_2024"
        assert dirs["ftw-33UXQ0001_2024"] == ctx.chips_base_dir / "33UXQ" / "ftw-33UXQ0001_2024"
        assert dirs["grid_001_2024"] == ctx.chips_base_dir / "other" / "grid_001_2024"
        assert all(p.is_dir() for p in dirs.values())


class TestSourceResolution:
    def test_url_input_is_fetched_and_recorded(self, tmp_path: Path, monkeypatch) -> None:
        import geopandas as gpd
        from shapely.geometry import box

        from ftw_dataset_tools.api import pipeline
        from ftw_dataset_tools.api.config import DatasetConfig

        local = tmp_path / "cached.parquet"
        gpd.GeoDataFrame({"id": [1]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_parquet(local)

        def fake_fetch(url, cache_dir, *, refresh=False, **_kwargs):  # noqa: ARG001
            from ftw_dataset_tools.api.source import SourceRecord

            assert url == "https://x/lu.parquet"
            assert str(cache_dir).endswith("cache")
            return SourceRecord(url, local, "ab" * 32, 3, "2026-09-04T00:00:00Z")

        monkeypatch.setattr(pipeline, "fetch_source", fake_fetch)
        monkeypatch.setattr(pipeline, "installed_git_commit", lambda: "c" * 40)

        config = DatasetConfig.from_dict(
            {
                "fields_file": "https://x/lu.parquet",
                "source_via": "https://x/collection.json",
                "output_dir": str(tmp_path / "out"),
                "year": 2024,
                "stages": {"fetch": {"cache_dir": str(tmp_path / "cache")}},
            }
        )
        provenance = config.provenance_dict()
        ctx = pipeline.build_context(config, provenance=provenance)

        assert ctx.fields_input == local
        assert provenance["source"]["href"] == "https://x/lu.parquet"
        assert provenance["source"]["via"] == "https://x/collection.json"
        assert provenance["source"]["sha256"] == "ab" * 32
        assert provenance["ftwd_git_commit"] == "c" * 40

    def test_local_input_is_described(self, sample_geoparquet_4326: Path, tmp_path: Path) -> None:
        from ftw_dataset_tools.api import pipeline
        from ftw_dataset_tools.api.config import DatasetConfig

        config = DatasetConfig.from_dict(
            {
                "fields_file": str(sample_geoparquet_4326),
                "output_dir": str(tmp_path / "out"),
                "year": 2024,
            }
        )
        provenance = config.provenance_dict()
        pipeline.build_context(config, provenance=provenance)

        assert provenance["source"]["href"] == str(sample_geoparquet_4326.resolve())
        assert provenance["source"]["fetched_at"] is None
        assert len(provenance["source"]["sha256"]) == 64


class TestChipsStageCropStats:
    def _ctx(self, tmp_path: Path, monkeypatch, *, crop_stats: bool) -> pipeline.PipelineContext:
        """A context whose chips stage produces one chip over two HCAT-coded fields."""
        fields = tmp_path / "fields.parquet"
        gpd.GeoDataFrame(
            {"id": [1, 2], "hcat:code": [1, 2], "hcat:name_en": ["Wheat", "Pasture"]},
            geometry=[box(0, 0, 0.5, 1), box(0.5, 0, 1, 1)],
            crs="EPSG:4326",
        ).to_parquet(fields)
        config = _config(
            fields,
            tmp_path / "out",
            year=2024,
            stages={
                "chips": {"crop_stats": crop_stats},
                "splits": {"split_type": "random-uniform"},
            },
        )
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir()
        gpd.read_parquet(fields).to_parquet(ctx.field_polygons_path)

        def fake_field_stats(**kwargs):
            gpd.GeoDataFrame(
                {"id": ["ftw-33UXP0001"], "field_coverage_pct": [50.0]},
                geometry=[box(0, 0, 1, 1)],
                crs="EPSG:4326",
            ).to_parquet(kwargs["output_file"])
            return field_stats.FieldStatsResult(
                output_path=Path(kwargs["output_file"]),
                total_cells=1,
                cells_with_coverage=1,
                average_coverage=50.0,
                max_coverage=50.0,
            )

        monkeypatch.setattr(field_stats, "add_field_stats", fake_field_stats)
        return ctx

    def test_crop_stats_called_after_coverage(self, tmp_path: Path, monkeypatch) -> None:
        from ftw_dataset_tools.api import crop_stats

        ctx = self._ctx(tmp_path, monkeypatch, crop_stats=True)
        calls: list[tuple] = []

        def fake_add_crop_stats(chips, fields, **_kwargs):
            calls.append((Path(chips), Path(fields)))
            return crop_stats.CropStatsResult(1, 0, 0, True, "x")

        monkeypatch.setattr(crop_stats, "add_crop_stats", fake_add_crop_stats)

        pipeline.stage_chips(ctx)

        assert calls == [(ctx.chips_path, ctx.field_polygons_path)]
        assert ctx.crop_stats_result is not None and ctx.crop_stats_result.skipped is True

    def test_crop_stats_disabled(self, tmp_path: Path, monkeypatch) -> None:
        from ftw_dataset_tools.api import crop_stats

        ctx = self._ctx(tmp_path, monkeypatch, crop_stats=False)

        def fail_if_called(*_args, **_kwargs):
            raise AssertionError("called")

        monkeypatch.setattr(crop_stats, "add_crop_stats", fail_if_called)

        pipeline.stage_chips(ctx)

        assert ctx.crop_stats_result is None

    def test_splits_keep_the_dominant_code_an_integer(self, tmp_path: Path, monkeypatch) -> None:
        """The chips GeoParquet is published as-is, so the split rewrite must not
        widen the nullable BIGINT to DOUBLE."""
        ctx = self._ctx(tmp_path, monkeypatch, crop_stats=True)

        pipeline.stage_chips(ctx)
        pipeline.stage_splits(ctx)

        con = duckdb.connect()
        types = {
            row[0]: row[1]
            for row in con.execute(
                f"DESCRIBE SELECT * FROM read_parquet('{ctx.chips_path}')"
            ).fetchall()
        }
        con.close()
        assert types["hcat_dominant_code"] == "BIGINT"
        assert types["split"] == "VARCHAR"
