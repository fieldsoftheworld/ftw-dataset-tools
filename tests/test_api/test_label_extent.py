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
    LABEL_EXTENT_SHA_COLUMN,
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


def _applied(pct: float, extent: Path | None = None) -> dict:
    """Chips columns the chips stage adds when it applies ``extent``."""
    from ftw_dataset_tools.api.label_extent import extent_fingerprint

    sha = extent_fingerprint(extent) if extent is not None else "0" * 64
    return {LABEL_EXTENT_COLUMN: [pct], LABEL_EXTENT_SHA_COLUMN: [sha]}


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

    def test_rejects_explicit_null_crs(self, tmp_path: Path) -> None:
        """``crs: null`` means unknown in GeoParquet, unlike a missing key."""
        path = tmp_path / "nocrs.parquet"
        gpd.GeoDataFrame({"tile": [0]}, geometry=[box(0, 0, 1, 1)]).to_parquet(path)
        with pytest.raises(LabelExtentError, match="no CRS"):
            load_label_extent(path)

    def test_missing_crs_key_means_wgs84(self, tmp_path: Path) -> None:
        import json

        import pyarrow.parquet as pq

        path = _write_extent(tmp_path / "e.parquet", [box(0, 0, 1, 1)])
        table = pq.read_table(path)
        geo = json.loads(table.schema.metadata[b"geo"])
        del geo["columns"]["geometry"]["crs"]
        meta = {**table.schema.metadata, b"geo": json.dumps(geo).encode()}
        pq.write_table(table.replace_schema_metadata(meta), path)

        assert load_label_extent(path).geom_col == "geometry"

    def test_accepts_crs84(self, tmp_path: Path) -> None:
        path = _write_extent(tmp_path / "e.parquet", [box(0, 0, 1, 1)], crs="OGC:CRS84")
        assert load_label_extent(path).geom_col == "geometry"

    def test_summary_line(self) -> None:
        from ftw_dataset_tools.api.label_extent import LabelExtentChipsResult, label_extent_summary

        line = label_extent_summary(LabelExtentChipsResult(total_chips=1200, kept_chips=57))
        assert line == "Label extent: kept 57 of 1,200 chips (1,143 wholly outside dropped)"

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
        # The provenance keeps the config's own reference, not this machine's path; the
        # chips record the extent's sha256, which identifies the file wherever it is.
        recorded = config.provenance_dict()["config"]["stages"]["masks"]
        assert recorded["label_extent"] == "extent.parquet"

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


class TestLabelExtentOnUtmGrid:
    """Lon/lat extents much larger than a chip, burned on the chip's UTM grid."""

    CELL = box(12.0, 45.995, 12.01, 46.005)

    def _unknown_share(self, tmp_path: Path, extent_box) -> tuple[float, float]:
        """(label_extent_pct of the chip, share of its 3-class pixels that are 3)."""
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["c1"], "field_coverage_pct": [50.0]}, geometry=[self.CELL], crs="EPSG:4326"
        ).to_parquet(chips)
        field = box(12.001, 45.996, 12.002, 45.997)
        fields = tmp_path / "fields.parquet"
        gpd.GeoDataFrame({"id": [1]}, geometry=[field], crs="EPSG:4326").to_parquet(fields)
        lines = tmp_path / "lines.parquet"
        gpd.GeoDataFrame(
            {"id": [1]}, geometry=[LineString(field.exterior.coords)], crs="EPSG:4326"
        ).to_parquet(lines)
        extent = _write_extent(tmp_path / "extent.parquet", [extent_box])

        apply_label_extent(chips, extent)
        pct = TestApplyLabelExtentToChips._read(chips)["c1"]
        result = create_masks(
            chips_file=chips,
            boundaries_file=fields,
            boundary_lines_file=lines,
            output_dir=tmp_path / "out",
            mask_types=[MaskType.SEMANTIC_3_CLASS],
            num_workers=1,
            min_coverage=0,
            label_extent_file=extent,
        )[MaskType.SEMANTIC_3_CLASS]
        with rasterio.open(result.masks_created[0].output_path) as src:
            assert src.crs.to_epsg() == 32633
            data = src.read(1)
        return pct, float((data == 3).mean())

    def test_long_edge_stays_put(self, tmp_path: Path) -> None:
        """A 1-degree edge at lat 46 splits the chip in half, not a 120 m chord off it."""
        pct, unknown = self._unknown_share(tmp_path, box(11.5, 45, 12.5, 46))
        assert pct == 50.0
        assert unknown == pytest.approx(0.5, abs=0.03)

    def test_very_long_edge_stays_put(self, tmp_path: Path) -> None:
        """An edge 12 degrees long, whose raw chord would miss the chip entirely."""
        pct, unknown = self._unknown_share(tmp_path, box(6, 40, 18, 46))
        assert pct == 50.0
        assert unknown == pytest.approx(0.5, abs=0.03)

    def test_unknown_share_matches_label_extent_pct(self, tmp_path: Path) -> None:
        """Outside share of the mask tracks 100 - label_extent_pct for a corner overlap."""
        pct, unknown = self._unknown_share(tmp_path, box(11, 45, 12.0025, 46.0025))
        assert 0 < pct < 100
        assert unknown == pytest.approx(1 - pct / 100, abs=0.03)

    def test_detailed_extent_is_clipped_per_chip(self, tmp_path: Path) -> None:
        """Only the piece of a large polygon under the chip comes back, densified."""
        import shapely

        from ftw_dataset_tools.api.label_extent import extent_pieces

        big = shapely.segmentize(box(0, 0, 50, 50), 0.001)  # ~200k vertices
        extent = load_label_extent(_write_extent(tmp_path / "big.parquet", [big]))

        (piece,) = extent_pieces(extent, (49.99, 10.0, 50.01, 10.02))

        _minx, miny, maxx, maxy = piece.bounds
        assert maxx == pytest.approx(50.0)
        assert (miny, maxy) == pytest.approx((9.999, 10.021))
        assert shapely.get_num_coordinates(piece) < 20_000

    def test_no_pieces_far_from_the_extent(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.label_extent import extent_pieces

        extent = load_label_extent(_write_extent(tmp_path / "e.parquet", [box(0, 0, 1, 1)]))
        assert extent_pieces(extent, (5, 5, 5.01, 5.01)) == []

    def test_antimeridian_bounds_are_split(self, tmp_path: Path) -> None:
        """A UTM grid snapped past 180 gives minx > maxx; both sides are clipped."""
        from ftw_dataset_tools.api.label_extent import extent_pieces

        extent = load_label_extent(
            _write_extent(
                tmp_path / "e.parquet",
                [box(179.5, -17.5, 180, -16.5), box(-180, -17.5, -179.5, -16.5)],
            )
        )
        pieces = extent_pieces(extent, (179.9848, -17.0, -179.99976, -16.99))
        sides = sorted(round(piece.bounds[0]) for piece in pieces)
        assert sides == [-180, 180]

    def test_antimeridian_chip_gets_a_mask(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        chips = tmp_path / "chips.parquet"
        gpd.GeoDataFrame(
            {"id": ["c1"], "field_coverage_pct": [50.0]},
            geometry=[box(179.99, -17.0, 180.0, -16.99)],
            crs="EPSG:4326",
        ).to_parquet(chips)
        field = box(179.992, -16.998, 179.995, -16.995)
        fields = tmp_path / "fields.parquet"
        gpd.GeoDataFrame({"id": [1]}, geometry=[field], crs="EPSG:4326").to_parquet(fields)
        lines = tmp_path / "lines.parquet"
        gpd.GeoDataFrame(
            {"id": [1]}, geometry=[LineString(field.exterior.coords)], crs="EPSG:4326"
        ).to_parquet(lines)
        extent = _write_extent(tmp_path / "e.parquet", [box(179.5, -17.5, 180, -16.5)])

        result = create_masks(
            chips_file=chips,
            boundaries_file=fields,
            boundary_lines_file=lines,
            output_dir=tmp_path / "out",
            mask_types=[MaskType.SEMANTIC_3_CLASS],
            num_workers=1,
            min_coverage=0,
            label_extent_file=extent,
        )[MaskType.SEMANTIC_3_CLASS]

        assert result.total_created == 1, result.masks_skipped
        with rasterio.open(result.masks_created[0].output_path) as src:
            data = src.read(1)
        # The chip is wholly inside; only grid pixels snapped past 180 may be unknown.
        assert 0 < (data == 3).mean() < 0.05

    def test_repair_leftover_lines_are_not_inside(self, tmp_path: Path) -> None:
        """A spike that ST_MakeValid turns into a line must not burn as inside."""
        from rasterio.transform import from_bounds
        from shapely import wkt

        from ftw_dataset_tools.api.label_extent import extent_pieces
        from ftw_dataset_tools.api.masks import _burn_label_extent

        spike = wkt.loads("POLYGON((0 0, 0.5 0, 0.5 0.5, 0.9 0.9, 0.5 0.5, 0.5 1, 0 1, 0 0))")
        extent = load_label_extent(_write_extent(tmp_path / "spike.parquet", [spike]))
        bounds = (0, 0, 1, 1)

        pieces = extent_pieces(extent, bounds)
        assert {piece.geom_type for piece in pieces} == {"Polygon"}

        mask = np.zeros((10, 10), np.uint8)
        _burn_label_extent(mask, extent, bounds, from_bounds(*bounds, 10, 10), None)
        # The east half (where the spike runs to 0.9, 0.9) is outside: no 0 pixels there.
        assert (mask[:, 5:] == 3).all()
        assert (mask[:, :5] == 0).all()


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
            {"id": ["cell_001"], "field_coverage_pct": [20.0], **_applied(50.0, extent)},
            geometry=[cell],
            crs="EPSG:4326",
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
        gpd.GeoDataFrame(
            {"id": ["a"], **_applied(100.0, extent)}, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326"
        ).to_parquet(ctx.chips_path)
        ctx.output_fields_path.write_bytes(b"")
        ctx.boundary_lines_path.write_bytes(b"")

        pipeline.stage_stac(ctx)

        assert captured["label_extent"] is True
        assert captured["background_class_value"] == 0


class TestLabelExtentStageGuard:
    """masks and stac refuse a chips file built with a different label_extent setting."""

    @staticmethod
    def _ctx(tmp_path: Path, fields: Path, *, configured: bool, applied: bool):
        from ftw_dataset_tools.api import pipeline

        masks: dict = {"workers": 1}
        extent = _write_extent(tmp_path / "e.parquet", [box(0, 0, 1, 1)])
        if configured:
            masks["label_extent"] = str(extent)
        config = DatasetConfig.from_dict(
            {
                "fields_file": str(fields),
                "output_dir": str(tmp_path / "out"),
                "year": 2024,
                "stages": {"masks": masks},
            }
        )
        ctx = pipeline.build_context(config)
        ctx.output_dir.mkdir(parents=True)
        ctx.chips_base_dir.mkdir()
        columns: dict = {"id": ["a"], "field_coverage_pct": [50.0]}
        if applied:
            columns.update(_applied(100.0, extent))
        gpd.GeoDataFrame(columns, geometry=[box(0, 0, 1, 1)], crs="EPSG:4326").to_parquet(
            ctx.chips_path
        )
        for path in (ctx.field_polygons_path, ctx.output_fields_path, ctx.boundary_lines_path):
            path.write_bytes(b"")
        return ctx

    @pytest.mark.parametrize("stage", ["masks", "stac"])
    @pytest.mark.parametrize(
        ("configured", "applied", "message"),
        [(True, False, "is set, but .* without one"), (False, True, "is not set, but .* with one")],
    )
    def test_mismatch_raises(
        self, sample_geoparquet_4326: Path, tmp_path: Path, stage, configured, applied, message
    ) -> None:
        from ftw_dataset_tools.api import pipeline
        from ftw_dataset_tools.api.pipeline import StageInputError

        ctx = self._ctx(tmp_path, sample_geoparquet_4326, configured=configured, applied=applied)
        stage_func = {"masks": pipeline.stage_masks, "stac": pipeline.stage_stac}[stage]

        with pytest.raises(StageInputError, match=f"Stage '{stage}': .*{message}.*'chips'"):
            stage_func(ctx)

    @pytest.mark.parametrize("configured", [True, False])
    def test_matching_setting_passes(
        self, sample_geoparquet_4326: Path, tmp_path: Path, configured
    ) -> None:
        from ftw_dataset_tools.api.pipeline import _require_label_extent_match

        ctx = self._ctx(tmp_path, sample_geoparquet_4326, configured=configured, applied=configured)
        _require_label_extent_match(ctx, stage="masks")


class TestRunSummary:
    def test_summary_reports_the_extent_drop(self, tmp_path: Path) -> None:
        from unittest.mock import Mock

        import click
        from click.testing import CliRunner

        from ftw_dataset_tools.api.label_extent import LabelExtentChipsResult
        from ftw_dataset_tools.commands import run

        ctx = Mock()
        ctx.output_dir = tmp_path
        for name in (
            "chips_result",
            "splits_result",
            "masks_results",
            "stac_result",
            "selection_result",
            "download_result",
            "docs_result",
        ):
            setattr(ctx, name, None)
        ctx.label_extent_result = LabelExtentChipsResult(total_chips=10, kept_chips=4)

        @click.command()
        def show() -> None:
            run._print_summary(ctx)

        output = CliRunner().invoke(show).output
        assert "Label extent: kept 4 of 10 chips (6 wholly outside dropped)" in output


class TestLabelExtentFingerprint:
    """Chips and masks record which extent file they were built from."""

    def test_chips_record_the_extent_sha(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.label_extent import chips_fingerprint, extent_fingerprint

        chips = TestApplyLabelExtentToChips()._chips(tmp_path)
        assert chips_fingerprint(chips) is None
        extent = _write_extent(tmp_path / "e.parquet", [box(-1, -1, 9, 2)])

        apply_label_extent(chips, extent)

        assert chips_fingerprint(chips) == extent_fingerprint(extent)

    @pytest.mark.parametrize("stage", ["masks", "stac"])
    def test_rewritten_extent_is_refused(
        self, sample_geoparquet_4326: Path, tmp_path: Path, stage
    ) -> None:
        """Chips from the old extent file don't pass for the rewritten one."""
        from ftw_dataset_tools.api import pipeline
        from ftw_dataset_tools.api.pipeline import StageInputError

        ctx = TestLabelExtentStageGuard._ctx(
            tmp_path, sample_geoparquet_4326, configured=True, applied=True
        )
        _write_extent(tmp_path / "e.parquet", [box(0, 0, 0.5, 1)])
        stage_func = {"masks": pipeline.stage_masks, "stac": pipeline.stage_stac}[stage]

        with pytest.raises(StageInputError, match=r"different extent file.*'chips'"):
            stage_func(ctx)


class TestLabelExtentSkipExisting:
    """skip_existing reuses a semantic mask only when it was burned with this run's extent."""

    @staticmethod
    def _run(tmp_path: Path, extent: Path | None):
        from ftw_dataset_tools.api.masks import MaskType, create_masks

        chips, boundaries, lines = TestLabelExtentMasks()._build_inputs(tmp_path)
        types = [MaskType.SEMANTIC_2_CLASS, MaskType.SEMANTIC_3_CLASS, MaskType.INSTANCE]
        results = create_masks(
            chips_file=chips,
            boundaries_file=boundaries,
            boundary_lines_file=lines,
            output_dir=tmp_path / "out",
            field_dataset="test",
            mask_types=types,
            num_workers=1,
            skip_existing=True,
            label_extent_file=extent,
        )
        reused = {t: results[t].masks_existing for t in types}
        path = (tmp_path / "out").rglob("*_semantic_3_class.tif").__next__()
        with rasterio.open(path) as src:
            return reused, src.tags().get("FTW_LABEL_EXTENT"), src.read(1)

    def _extent(self, tmp_path: Path, name: str, east: float) -> Path:
        return _write_extent(
            tmp_path / name,
            [box(4000000, 3000000, east, 3001000)],
            crs=TestLabelExtentMasks.CRS,
        )

    def test_default_build_unchanged(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.masks import MaskType

        _, tag, data = self._run(tmp_path, None)
        assert tag is None
        assert 3 not in data
        reused, tag, _ = self._run(tmp_path, None)
        assert tag is None
        assert reused == dict.fromkeys(reused, 1)
        assert MaskType.INSTANCE in reused

    def test_adding_an_extent_rebuilds_semantic_masks(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.label_extent import extent_fingerprint
        from ftw_dataset_tools.api.masks import MaskType

        self._run(tmp_path, None)
        extent = self._extent(tmp_path, "e.parquet", 4000500)

        reused, tag, data = self._run(tmp_path, extent)

        assert reused[MaskType.SEMANTIC_2_CLASS] == reused[MaskType.SEMANTIC_3_CLASS] == 0
        # The instance mask does not depend on the extent, so it is kept.
        assert reused[MaskType.INSTANCE] == 1
        assert tag == extent_fingerprint(extent)
        assert 3 in data

        reused, _, _ = self._run(tmp_path, extent)
        assert reused[MaskType.SEMANTIC_3_CLASS] == 1

    def test_changed_or_removed_extent_rebuilds(self, tmp_path: Path) -> None:
        from ftw_dataset_tools.api.masks import MaskType

        self._run(tmp_path, self._extent(tmp_path, "e.parquet", 4000500))
        other = self._extent(tmp_path, "other.parquet", 4000800)

        reused, _, _ = self._run(tmp_path, other)
        assert reused[MaskType.SEMANTIC_3_CLASS] == 0

        reused, tag, data = self._run(tmp_path, None)
        assert reused[MaskType.SEMANTIC_3_CLASS] == 0
        assert tag is None
        assert 3 not in data
