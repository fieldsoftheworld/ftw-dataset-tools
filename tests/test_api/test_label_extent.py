"""Tests for the optional label extent (stages.masks.label_extent)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
import rasterio
import yaml
from shapely.geometry import LineString, Point, box

from ftw_dataset_tools.api import config as config_module
from ftw_dataset_tools.api.config import ConfigError, DatasetConfig
from ftw_dataset_tools.api.label_extent import (
    LABEL_EXTENT_COLUMN,
    LabelExtentError,
    apply_label_extent,
    load_label_extent,
)

if TYPE_CHECKING:
    from pathlib import Path


def _write_extent(path: Path, geometries: list, crs: str = "EPSG:4326") -> Path:
    gpd.GeoDataFrame(
        {"tile": list(range(len(geometries)))}, geometry=geometries, crs=crs
    ).to_parquet(path)
    return path


class TestLoadLabelExtent:
    """Validation of the extent file itself."""

    def test_valid_polygons(self, tmp_path: Path) -> None:
        path = _write_extent(tmp_path / "extent.parquet", [box(0, 0, 1, 1)])
        extent = load_label_extent(path)
        assert extent.path == str(path.resolve())
        assert extent.geom_col == "geometry"

    def test_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(LabelExtentError, match="not found"):
            load_label_extent(tmp_path / "nope.parquet")

    def test_rejects_points(self, tmp_path: Path) -> None:
        path = _write_extent(tmp_path / "extent.parquet", [box(0, 0, 1, 1), Point(0, 0)])
        with pytest.raises(LabelExtentError, match="polygons only, found POINT"):
            load_label_extent(path)

    def test_rejects_other_crs(self, tmp_path: Path) -> None:
        path = _write_extent(tmp_path / "extent.parquet", [box(0, 0, 1, 1)], crs="EPSG:3035")
        with pytest.raises(LabelExtentError, match="Reproject"):
            load_label_extent(path)

    def test_accepts_expected_crs(self, tmp_path: Path) -> None:
        path = _write_extent(tmp_path / "extent.parquet", [box(0, 0, 1, 1)], crs="EPSG:3035")
        assert load_label_extent(path, expected_crs="EPSG:3035").geom_col == "geometry"

    def test_rejects_plain_parquet(self, tmp_path: Path) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        path = tmp_path / "plain.parquet"
        pq.write_table(pa.table({"a": [1]}), path)
        with pytest.raises(LabelExtentError, match="not GeoParquet"):
            load_label_extent(path)


class TestLabelExtentConfig:
    """stages.masks.label_extent in the config schema."""

    def _write_config(self, tmp_path: Path, masks: dict) -> Path:
        config_path = tmp_path / "config.yaml"
        config_path.write_text(
            yaml.safe_dump({"fields_file": "f.parquet", "year": 2023, "stages": {"masks": masks}})
        )
        return config_path

    def test_default_is_none_and_not_written(self) -> None:
        config = DatasetConfig.from_dict({"fields_file": "f.parquet"})
        assert config.stages.masks.label_extent is None
        # A config without an extent resolves exactly as before the option existed.
        assert "label_extent" not in config.config_dict()["stages"]["masks"]

    def test_resolved_relative_to_config(self, tmp_path: Path) -> None:
        _write_extent(tmp_path / "extent.parquet", [box(0, 0, 1, 1)])
        config_path = self._write_config(tmp_path, {"label_extent": "extent.parquet"})

        config = config_module.load_config(config_path)

        expected = str((tmp_path / "extent.parquet").resolve())
        assert config.stages.masks.label_extent == expected
        resolved = config.provenance_dict()["config"]["stages"]["masks"]
        assert resolved["label_extent"] == expected

    def test_missing_file_is_config_error(self, tmp_path: Path) -> None:
        config_path = self._write_config(tmp_path, {"label_extent": "nope.parquet"})
        with pytest.raises(ConfigError, match=r"stages\.masks\.label_extent file not found"):
            config_module.load_config(config_path)

    def test_non_polygon_is_config_error(self, tmp_path: Path) -> None:
        _write_extent(tmp_path / "extent.parquet", [Point(0, 0)])
        config_path = self._write_config(tmp_path, {"label_extent": "extent.parquet"})
        with pytest.raises(ConfigError, match="polygons only"):
            config_module.load_config(config_path)

    def test_conflicts_with_presence_only(self) -> None:
        with pytest.raises(ConfigError, match="alternatives"):
            DatasetConfig.from_dict(
                {
                    "fields_file": "f.parquet",
                    "stages": {"masks": {"label_extent": "x.parquet", "presence_only": True}},
                }
            )

    def test_rejects_empty_path(self) -> None:
        with pytest.raises(ConfigError, match="must be a path"):
            DatasetConfig.from_dict(
                {"fields_file": "f.parquet", "stages": {"masks": {"label_extent": " "}}}
            )


class TestLabelExtentMasks:
    """Rasterization with an extent: one 1 km chip, extent over its western half."""

    CRS = "EPSG:3035"

    def _build_inputs(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        cell = box(4000000, 3000000, 4001000, 3001000)
        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["grid_001"], "field_coverage_pct": [20.0]}, geometry=[cell], crs=self.CRS
        ).to_parquet(chips)

        # One field inside the extent (west), one outside it (east).
        fields = [
            box(4000100, 3000100, 4000300, 3000300),
            box(4000700, 3000100, 4000900, 3000300),
        ]
        boundaries = tmp_path / "fields.parquet"
        gpd.GeoDataFrame({"id": [1, 2]}, geometry=fields, crs=self.CRS).to_parquet(boundaries)
        lines = tmp_path / "lines.parquet"
        gpd.GeoDataFrame(
            {"id": [1, 2]}, geometry=[LineString(f.exterior.coords) for f in fields], crs=self.CRS
        ).to_parquet(lines)
        return chips, boundaries, lines

    def _create(self, tmp_path: Path, mask_type, extent: Path | None, name: str):
        from ftw_dataset_tools.api.masks import create_masks

        chips, boundaries, lines = self._build_inputs(tmp_path)
        result = create_masks(
            chips_file=chips,
            boundaries_file=boundaries,
            boundary_lines_file=lines,
            output_dir=tmp_path / name,
            field_dataset="test",
            mask_types=[mask_type],
            num_workers=1,
            label_extent_file=extent,
        )[mask_type]
        assert result.total_created == 1, result.masks_skipped
        with rasterio.open(result.masks_created[0].output_path) as src:
            return src.read(1), (src.transform, src.crs)

    def _west_extent(self, tmp_path: Path) -> Path:
        return _write_extent(
            tmp_path / "extent.parquet", [box(4000000, 3000000, 4000500, 3001000)], crs=self.CRS
        )

    def _pixel(self, grid, x: float, y: float) -> tuple[int, int]:
        """Row/column of a point given in the inputs' CRS (masks are on a UTM grid)."""
        from rasterio.warp import transform as warp

        transform, crs = grid
        xs, ys = warp(self.CRS, crs, [x], [y])
        col, row = ~transform * (xs[0], ys[0])
        return int(row), int(col)

    def test_3_class_inside_outside_field_and_line(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.masks import MaskType

        data, grid = self._create(
            tmp_path, MaskType.SEMANTIC_3_CLASS, self._west_extent(tmp_path), "m3"
        )
        assert set(np.unique(data)) == {0, 1, 2, 3}
        # Unlabelled inside the extent: background.
        assert data[self._pixel(grid, 4000050, 3000800)] == 0
        # Unlabelled outside the extent: unknown.
        assert data[self._pixel(grid, 4000950, 3000800)] == 3
        # A field outside the extent is still a field.
        assert data[self._pixel(grid, 4000800, 3000200)] == 1
        assert data[self._pixel(grid, 4000200, 3000200)] == 1
        # Its boundary line is still a boundary.
        assert data[self._pixel(grid, 4000700, 3000200)] == 2

    def test_2_class_lines_stay_zero_outside_extent(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.masks import MaskType

        data, grid = self._create(
            tmp_path, MaskType.SEMANTIC_2_CLASS, self._west_extent(tmp_path), "m2"
        )
        assert set(np.unique(data)) == {0, 1, 3}
        assert data[self._pixel(grid, 4000950, 3000800)] == 3
        assert data[self._pixel(grid, 4000800, 3000200)] == 1
        # The labelled field's edge is known, so it stays 0 even outside the extent.
        assert data[self._pixel(grid, 4000700, 3000200)] == 0

    def test_instance_ignores_extent(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.masks import MaskType

        with_extent, _ = self._create(
            tmp_path, MaskType.INSTANCE, self._west_extent(tmp_path), "with"
        )
        without, _ = self._create(tmp_path, MaskType.INSTANCE, None, "without")
        np.testing.assert_array_equal(with_extent, without)

    def test_decode_folds_unknown_into_background(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.masks import MaskType

        with_extent, _ = self._create(
            tmp_path, MaskType.DECODE_BOUNDARY, self._west_extent(tmp_path), "with"
        )
        without, _ = self._create(tmp_path, MaskType.DECODE_BOUNDARY, None, "without")
        np.testing.assert_array_equal(with_extent, without)

    def test_extent_covering_the_chip_matches_no_extent(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.masks import MaskType

        extent = _write_extent(
            tmp_path / "all.parquet", [box(3999000, 2999000, 4002000, 3002000)], crs=self.CRS
        )
        with_extent, _ = self._create(tmp_path, MaskType.SEMANTIC_3_CLASS, extent, "with")
        without, _ = self._create(tmp_path, MaskType.SEMANTIC_3_CLASS, None, "without")
        np.testing.assert_array_equal(with_extent, without)

    def test_extent_in_another_crs_is_rejected(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        chips, boundaries, lines = self._build_inputs(tmp_path)
        extent = _write_extent(tmp_path / "extent.parquet", [box(0, 0, 1, 1)])
        with pytest.raises(LabelExtentError, match="Reproject"):
            create_masks(
                chips_file=chips,
                boundaries_file=boundaries,
                boundary_lines_file=lines,
                output_dir=tmp_path / "out",
                mask_types=[MaskType.SEMANTIC_2_CLASS],
                num_workers=1,
                label_extent_file=extent,
            )


class TestApplyLabelExtentToChips:
    """Chip selection and the label_extent_pct column."""

    def _chips(self, tmp_path: Path) -> Path:
        path = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["c", "a", "b"], "field_coverage_pct": [10.0, 20.0, 30.0]},
            geometry=[box(2, 0, 3, 1), box(0, 0, 1, 1), box(1, 0, 2, 1)],
            crs="EPSG:4326",
        ).to_parquet(path)
        return path

    @staticmethod
    def _read(path: Path) -> dict[str, float]:
        import duckdb

        rows = duckdb.sql(
            f"SELECT id, {LABEL_EXTENT_COLUMN} FROM read_parquet('{path}') ORDER BY id"
        ).fetchall()
        return dict(rows)

    def test_drops_outside_and_records_share(self, tmp_path: Path) -> None:
        chips = self._chips(tmp_path)
        # Covers chip a fully and the western quarter of chip b; chip c lies wholly
        # outside, so it goes. The overlapping pieces must not be counted twice.
        extent = _write_extent(
            tmp_path / "extent.parquet", [box(-1, -1, 1.25, 2), box(0.5, 0, 1.25, 1)]
        )

        result = apply_label_extent(chips, extent)

        assert (result.total_chips, result.kept_chips, result.dropped_chips) == (3, 2, 1)
        assert self._read(chips) == {"a": 100.0, "b": 25.0}

    def test_edge_touching_chip_is_dropped(self, tmp_path: Path) -> None:
        chips = self._chips(tmp_path)
        extent = _write_extent(tmp_path / "extent.parquet", [box(-1, -1, 1, 2)])

        apply_label_extent(chips, extent)

        assert self._read(chips) == {"a": 100.0}

    def test_rerun_replaces_the_column(self, tmp_path: Path) -> None:
        chips = self._chips(tmp_path)
        apply_label_extent(chips, _write_extent(tmp_path / "e1.parquet", [box(-1, -1, 9, 2)]))
        apply_label_extent(chips, _write_extent(tmp_path / "e2.parquet", [box(-1, -1, 1.5, 2)]))

        assert self._read(chips) == {"a": 100.0, "b": 50.0}

    def test_output_ordered_by_id_and_geoparquet(self, tmp_path: Path) -> None:
        import duckdb

        from ftw_dataset_tools.api.geo import detect_geometry_column

        chips = self._chips(tmp_path)
        apply_label_extent(chips, _write_extent(tmp_path / "e.parquet", [box(-1, -1, 9, 2)]))

        ids = [r[0] for r in duckdb.sql(f"SELECT id FROM read_parquet('{chips}')").fetchall()]
        assert ids == ["a", "b", "c"]
        assert detect_geometry_column(chips) == "geometry"


class TestLabelExtentStac:
    """Collection and item properties."""

    def test_collection_property_only_with_extent(self) -> None:
        from ftw_dataset_tools.api.stac import _collection_ftw_properties

        plain = _collection_ftw_properties(DatasetConfig.from_dict({"fields_file": "f.parquet"}))
        assert "ftw:label_extent" not in plain

        extent = _collection_ftw_properties(
            DatasetConfig.from_dict(
                {"fields_file": "f.parquet", "stages": {"masks": {"label_extent": "e.parquet"}}}
            )
        )
        assert extent["ftw:label_extent"] is True
        assert extent["ftw:presence_only"] is False

    def test_item_property_from_chips_column(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.stac import _extract_chips_info

        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["a"], "field_coverage_pct": [5.0], LABEL_EXTENT_COLUMN: [42.5]},
            geometry=[box(0, 0, 1, 1)],
            crs="EPSG:4326",
        ).to_parquet(chips)

        (chip,) = _extract_chips_info(chips)

        assert chip.properties["ftw:label_extent_pct"] == 42.5
        assert chip.properties["ftw:field_coverage_pct"] == 5.0


class TestLabelExtentPipeline:
    """The chips, masks and stac stages pick the extent up from the config."""

    @staticmethod
    def _ctx(tmp_path: Path, fields: Path, extent: Path, **masks):
        from ftw_dataset_tools.api import pipeline

        config = DatasetConfig.from_dict(
            {
                "fields_file": str(fields),
                "output_dir": str(tmp_path / "out"),
                "year": 2024,
                "stages": {
                    "chips": {"crop_stats": False},
                    "masks": {"label_extent": str(extent), "workers": 1, **masks},
                },
            }
        )
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir(parents=True, exist_ok=True)
        ctx.chips_base_dir.mkdir(parents=True, exist_ok=True)
        return ctx

    def test_chips_stage_drops_chips_outside(
        self, sample_geoparquet_4326: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from pathlib import Path as _Path

        from ftw_dataset_tools.api import field_stats, pipeline

        extent = _write_extent(tmp_path / "extent.parquet", [box(-1, -1, 1.5, 2)])
        ctx = self._ctx(tmp_path, sample_geoparquet_4326, extent)
        gpd.read_parquet(sample_geoparquet_4326).to_parquet(ctx.field_polygons_path)

        def fake_field_stats(**kwargs):
            gpd.GeoDataFrame(
                {"id": ["a", "b", "c"], "field_coverage_pct": [50.0, 50.0, 50.0]},
                geometry=[box(0, 0, 1, 1), box(1, 0, 2, 1), box(5, 5, 6, 6)],
                crs="EPSG:4326",
            ).to_parquet(kwargs["output_file"])
            return field_stats.FieldStatsResult(_Path(kwargs["output_file"]), 3, 3, 50.0, 50.0)

        monkeypatch.setattr(field_stats, "add_field_stats", fake_field_stats)
        pipeline.stage_chips(ctx)

        assert TestApplyLabelExtentToChips._read(ctx.chips_path) == {"a": 100.0, "b": 50.0}
        assert ctx.label_extent_result is not None
        assert ctx.label_extent_result.dropped_chips == 1

    def test_masks_stage_marks_outside_unknown(
        self, sample_geoparquet_4326: Path, tmp_path: Path
    ) -> None:
        from ftw_dataset_tools.api import pipeline

        extent = _write_extent(tmp_path / "extent.parquet", [box(17, 39, 18.005, 41)])
        ctx = self._ctx(tmp_path, sample_geoparquet_4326, extent, mask_types=["semantic_3_class"])
        cell = box(18.0, 40.0, 18.01, 40.01)
        gpd.GeoDataFrame(
            {"id": ["cell_001"], "field_coverage_pct": [20.0]}, geometry=[cell], crs="EPSG:4326"
        ).to_parquet(ctx.chips_path)
        fields = [box(18.002, 40.002, 18.004, 40.004)]
        gpd.GeoDataFrame({"id": [1]}, geometry=fields, crs="EPSG:4326").to_parquet(
            ctx.field_polygons_path
        )
        gpd.GeoDataFrame(
            {"id": [1]}, geometry=[LineString(fields[0].exterior.coords)], crs="EPSG:4326"
        ).to_parquet(ctx.boundary_lines_path)

        pipeline.stage_masks(ctx)

        (created,) = ctx.masks_results["semantic_3class"].masks_created
        with rasterio.open(created.output_path) as src:
            data = src.read(1)
        assert set(np.unique(data)) == {0, 1, 2, 3}
        # The extent's east edge runs down the middle: west known, east unknown.
        assert data[data.shape[0] // 2, -1] == 3
        assert data[data.shape[0] // 4, 0] == 0

    def test_stac_stage_passes_label_extent(
        self, sample_geoparquet_4326: Path, tmp_path: Path, monkeypatch
    ) -> None:
        from datetime import UTC, datetime

        from ftw_dataset_tools.api import pipeline, stac

        captured: dict = {}

        def fake_generate(**kwargs):
            captured.update(kwargs)
            return stac.STACGenerationResult(
                collection_path=tmp_path / "collection.json",
                items_parquet_path=None,
                subcatalog_paths={},
                total_items=0,
                temporal_extent=(datetime(2024, 1, 1, tzinfo=UTC),) * 2,
            )

        monkeypatch.setattr(stac, "generate_stac_catalog", fake_generate)
        extent = _write_extent(tmp_path / "extent.parquet", [box(0, 0, 1, 1)])
        ctx = self._ctx(tmp_path, sample_geoparquet_4326, extent)
        gpd.GeoDataFrame({"id": ["a"]}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_parquet(
            ctx.chips_path
        )
        ctx.output_fields_path.write_bytes(b"")
        ctx.boundary_lines_path.write_bytes(b"")

        pipeline.stage_stac(ctx)

        assert captured["label_extent"] is True
        assert captured["background_class_value"] == 0
