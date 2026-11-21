"""Crop labels must describe the actual instances left in the raster."""

import json
import shutil
from datetime import UTC, datetime

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from click.testing import CliRunner
from rasterio.warp import transform
from shapely.geometry import MultiPolygon, box, mapping

from ftw_dataset_tools.api import masks, pipeline, stac
from ftw_dataset_tools.api.config import ConfigError, DatasetConfig
from ftw_dataset_tools.api.geo import write_geoparquet
from ftw_dataset_tools.cli import cli


def inputs(tmp_path, ids=("field-b", "field-a"), crops=("Wheat", None)):
    cell = box(10, 50, 10.01, 50.01)
    polygons = [box(10.001, 50.001, 10.004, 50.004), box(10.006, 50.001, 10.009, 50.004)]
    data = {"crop:name": list(crops)}
    if ids is not None:
        data["id"] = list(ids)
    fields = write_geoparquet(
        tmp_path / "fields.parquet",
        gdf=gpd.GeoDataFrame(
            data,
            geometry=polygons,
            crs=4326,
        ),
    )
    chips = write_geoparquet(
        tmp_path / "chips.parquet",
        gdf=gpd.GeoDataFrame(
            {"id": ["grid"], "field_coverage_pct": [20]},
            geometry=[cell],
            crs=4326,
        ),
    )
    lines = write_geoparquet(
        tmp_path / "lines.parquet",
        gdf=gpd.GeoDataFrame(
            geometry=[p.boundary for p in polygons],
            crs=4326,
        ),
    )
    return {
        "chips_file": chips,
        "boundaries_file": fields,
        "boundary_lines_file": lines,
        "output_dir": tmp_path / "out",
        "mask_types": [masks.MaskType.INSTANCE],
        "crop_column": "crop:name",
        "num_workers": 1,
        "year": 2024,
    }


def create(kwargs):
    result = masks.create_masks(**kwargs)[masks.MaskType.INSTANCE]
    assert not result.masks_skipped
    path = result.masks_created[0].output_path
    labels = json.loads(masks.instance_labels_path(path).read_text(encoding="utf-8"))
    return path, labels


@pytest.mark.parametrize(
    "ids,background",
    [
        (("field-b", "field-a"), 0),
        ((7, 7), 0),
        ((3, 4), 3),
        (None, 0),
    ],
)
def test_labels_match_pixels_with_arbitrary_ids(tmp_path, ids, background):
    kwargs = inputs(tmp_path, ids)
    kwargs["background_class_value"] = background
    path, labels = create(kwargs)
    assert labels["crop_column"] == "crop:name"
    # Instance masks use 0 as background even for presence-only datasets (semantic 3),
    # so a genuine field id of 3 keeps its own label.
    assert labels["background_value"] == 0
    with rasterio.open(path) as src:
        raster = src.read(1)
        values = set(np.unique(raster)) - {0}
        by_value = {row["instance_value"]: row for row in labels["instances"]}
        assert set(by_value) == values
        assert len(by_value) == 2
        for i, x in enumerate((10.002, 10.007)):
            (east,), (north,) = transform("EPSG:4326", src.crs, [x], [50.002])
            value = int(raster[src.index(east, north)])
            assert by_value[value]["crop_value"] == ("Wheat", None)[i]
            assert by_value[value]["field_id"] == (str(ids[i]) if ids else None)


def test_resume_requires_matching_lookup_and_stac_exposes_it(tmp_path):
    kwargs = inputs(tmp_path, crops=("001", "Wheat"))
    path, labels = create(kwargs)
    assert {r["crop_value"] for r in labels["instances"]} == {"001", "Wheat"}
    kwargs["skip_existing"] = True
    assert masks.create_masks(**kwargs)[masks.MaskType.INSTANCE].masks_existing == 1
    masks.instance_labels_path(path).unlink()
    assert create(kwargs)[1] == labels
    cell = box(10, 50, 10.01, 50.01)
    item = stac._create_chip_item(
        stac.ChipInfo("grid", mapping(cell), cell.bounds, year=2024),
        (datetime(2024, 1, 1, tzinfo=UTC), datetime(2025, 1, 1, tzinfo=UTC)),
        path.parent,
    )
    assert item.assets["instance_labels"].href == "./grid_2024_instance_labels.json"
    # Without a crop column the string ids still need a lookup, now without crops.
    kwargs["crop_column"] = None
    masks.create_masks(**kwargs)
    labels = json.loads(masks.instance_labels_path(path).read_text(encoding="utf-8"))
    assert labels["crop_column"] is None
    assert [set(r) for r in labels["instances"]] == [{"instance_value", "field_id"}] * 2


def test_disabling_labels_removes_the_lookup_when_ids_are_the_values(tmp_path):
    kwargs = inputs(tmp_path, ids=(3, 4))
    path, _ = create(kwargs)
    kwargs["crop_column"] = None
    kwargs["skip_existing"] = True
    masks.create_masks(**kwargs)
    assert not masks.instance_labels_path(path).exists()


def test_hidden_polygon_has_no_label(tmp_path):
    kwargs = inputs(tmp_path, ids=(1, 2))
    polygon = box(10.001, 50.001, 10.004, 50.004)
    write_geoparquet(
        kwargs["boundaries_file"],
        gdf=gpd.GeoDataFrame(
            {"id": [1, 2], "crop:name": ["Wheat", "Barley"]},
            geometry=[polygon, polygon],
            crs=4326,
        ),
    )
    _, labels = create(kwargs)
    assert labels["instances"] == [{"instance_value": 2, "field_id": "2", "crop_value": "Barley"}]


def test_missing_column_fails_before_workers(tmp_path):
    kwargs = inputs(tmp_path)
    kwargs["crop_column"] = "missing"
    with pytest.raises(ValueError, match="Crop column 'missing'"):
        masks.create_masks(**kwargs)


def test_failed_lookup_write_is_repaired_on_resume(tmp_path, monkeypatch):
    kwargs = inputs(tmp_path)
    path, labels = create(kwargs)

    def fail(*_args):
        raise OSError("simulated lookup write failure")

    def run_inline(tasks, *_args):
        # Windows workers spawn fresh interpreters and do not inherit patches.
        (task,) = tasks
        written, error = masks._create_masks_with_retry(
            task, rasterio.crs.CRS.from_wkt(task.crs_wkt)
        )
        return written, [(task, error, set())] if error else [], 0

    with monkeypatch.context() as patch:
        patch.setattr(masks, "_write_instance_labels", fail)
        patch.setattr(masks, "_run_work_items", run_inline)
        result = masks.create_masks(**kwargs)[masks.MaskType.INSTANCE]
        assert result.total_skipped == 1
        assert not masks.instance_labels_path(path).exists()
    kwargs["skip_existing"] = True
    assert create(kwargs)[1] == labels


def test_standalone_matches_api(tmp_path):
    kwargs = inputs(tmp_path)
    path, labels = create(kwargs)
    result = CliRunner().invoke(
        cli,
        [
            "create-masks",
            str(kwargs["chips_file"]),
            str(kwargs["boundaries_file"]),
            str(kwargs["boundary_lines_file"]),
            "--output-dir",
            str(tmp_path / "cli"),
            "--field-dataset",
            "test",
            "--year",
            "2024",
            "--mask-type",
            "instance",
            "--crop-column",
            "crop:name",
            "--workers",
            "1",
        ],
    )
    assert result.exit_code == 0, result.output
    cli_path = tmp_path / "cli" / "chips" / "other" / "grid_2024" / path.name
    assert json.loads(masks.instance_labels_path(cli_path).read_text()) == labels
    with rasterio.open(path) as a, rasterio.open(cli_path) as b:
        np.testing.assert_array_equal(a.read(1), b.read(1))


def test_crop_column_requires_instances():
    with pytest.raises(ConfigError, match="crop_column"):
        DatasetConfig.from_dict(
            {
                "fields_file": "fields.parquet",
                "stages": {
                    "masks": {"crop_column": "crop_name", "mask_types": ["semantic_2_class"]},
                },
            }
        )


def test_pipeline_matches_api(tmp_path):
    kwargs = inputs(tmp_path)
    path, labels = create(kwargs)
    config = DatasetConfig.from_dict(
        {
            "fields_file": str(kwargs["boundaries_file"]),
            "year": 2024,
            "stages": {
                "masks": {"crop_column": "crop:name", "mask_types": ["instance"], "workers": 1}
            },
        }
    )
    ctx = pipeline.PipelineContext(
        config, kwargs["boundaries_file"], tmp_path / "pipeline", "test", effective_year=2024
    )
    ctx.output_dir.mkdir()
    for source, target in (
        (kwargs["boundaries_file"], ctx.field_polygons_path),
        (kwargs["chips_file"], ctx.chips_path),
        (kwargs["boundary_lines_file"], ctx.boundary_lines_path),
    ):
        shutil.copyfile(source, target)
    pipeline.stage_masks(ctx)
    assert not ctx.masks_skipped
    actual = ctx.masks_results["instance"].masks_created[0].output_path
    assert json.loads(masks.instance_labels_path(actual).read_text()) == labels
    with rasterio.open(path) as a, rasterio.open(actual) as b:
        np.testing.assert_array_equal(a.read(1), b.read(1))


def _instances(tmp_path, ids, geometries, crop_column=None, counts=False):
    """Burn an instance mask: (pixel values, or {value: pixel count}, and lookup or None)."""
    kwargs = inputs(tmp_path, crops=(None, None))
    write_geoparquet(
        kwargs["boundaries_file"],
        gdf=gpd.GeoDataFrame(
            {"id": list(ids), "crop:name": [f"crop-{i}" for i in ids]},
            geometry=list(geometries),
            crs=4326,
        ),
    )
    kwargs["crop_column"] = crop_column
    result = masks.create_masks(**kwargs)[masks.MaskType.INSTANCE]
    path = result.masks_created[0].output_path
    with rasterio.open(path) as src:
        pixels = src.read(1)
    found, n = np.unique(pixels[pixels > 0], return_counts=True)
    values = dict(zip(found.tolist(), n.tolist(), strict=True)) if counts else set(found.tolist())
    lookup_path = masks.instance_labels_path(path)
    lookup = json.loads(lookup_path.read_text(encoding="utf-8")) if lookup_path.exists() else None
    return values, lookup


_LEFT = box(10.001, 50.001, 10.004, 50.004)
_RIGHT = box(10.006, 50.001, 10.009, 50.004)


class TestInstancePerPiece:
    def test_unique_ids_stay_the_pixel_values(self, tmp_path) -> None:
        values, lookup = _instances(tmp_path, (3, 4), (_LEFT, _RIGHT))
        assert values == {3, 4}
        assert lookup is None

    def test_repeated_id_gets_one_instance_per_polygon(self, tmp_path) -> None:
        values, lookup = _instances(tmp_path, (7, 7), (_LEFT, _RIGHT))
        assert values == {1, 2}
        assert lookup["crop_column"] is None
        assert sorted((r["instance_value"], r["field_id"]) for r in lookup["instances"]) == [
            (1, "7"),
            (2, "7"),
        ]

    def test_parts_that_do_not_touch_are_separate_instances(self, tmp_path) -> None:
        values, lookup = _instances(tmp_path, (5,), (MultiPolygon([_LEFT, _RIGHT]),))
        assert values == {1, 2}
        assert {r["field_id"] for r in lookup["instances"]} == {"5"}

    def test_touching_parts_stay_one_field(self, tmp_path) -> None:
        touching = box(10.004, 50.001, 10.006, 50.004)
        values, lookup = _instances(tmp_path, (5,), (MultiPolygon([_LEFT, touching]),))
        assert values == {5}
        assert lookup is None

    def test_overlapping_fields_burn_in_file_order(self, tmp_path) -> None:
        # Where fields overlap the later row wins, as before; the values stay field ids.
        wide = box(10.001, 50.001, 10.006, 50.004)
        values, lookup = _instances(tmp_path, (4, 3), (wide, _LEFT), counts=True)
        assert lookup is None
        assert values[3] > values[4]

    def test_corner_touching_parts_stay_one_field(self, tmp_path) -> None:
        corner = box(10.004, 50.004, 10.007, 50.007)
        values, lookup = _instances(tmp_path, (5,), (MultiPolygon([_LEFT, corner]),))
        assert values == {5}
        assert lookup is None

    def test_parts_outside_the_chip_do_not_renumber_it(self, tmp_path) -> None:
        outside = box(10.02, 50.001, 10.023, 50.004)
        values, lookup = _instances(tmp_path, (5, 6), (MultiPolygon([_LEFT, outside]), _RIGHT))
        assert values == {5, 6}
        assert lookup is None

    def test_values_are_contiguous_when_a_piece_is_hidden(self, tmp_path) -> None:
        # Field 7's second polygon lies under field 8, so it shows no pixels.
        values, lookup = _instances(tmp_path, (7, 7, 8), (_LEFT, _RIGHT, _RIGHT))
        assert values == {1, 2}
        assert sorted((r["instance_value"], r["field_id"]) for r in lookup["instances"]) == [
            (1, "7"),
            (2, "8"),
        ]

    def test_crop_labels_follow_the_pieces(self, tmp_path) -> None:
        values, lookup = _instances(
            tmp_path, (5,), (MultiPolygon([_LEFT, _RIGHT]),), crop_column="crop:name"
        )
        assert values == {1, 2}
        assert {(r["field_id"], r["crop_value"]) for r in lookup["instances"]} == {("5", "crop-5")}

    @pytest.mark.parametrize("raw", ["12.7", "inf"])
    def test_ids_that_are_not_whole_numbers_are_renumbered(self, tmp_path, raw) -> None:
        values, lookup = _instances(tmp_path, (raw, "4"), (_LEFT, _RIGHT))
        assert values == {1, 2}
        assert {r["field_id"] for r in lookup["instances"]} == {raw, "4"}

    def test_a_missing_lookup_is_rewritten_on_resume(self, tmp_path) -> None:
        kwargs = inputs(tmp_path, ids=(7, 7))
        kwargs["crop_column"] = None
        path = masks.create_masks(**kwargs)[masks.MaskType.INSTANCE].masks_created[0].output_path
        masks.instance_labels_path(path).unlink()
        kwargs["skip_existing"] = True
        result = masks.create_masks(**kwargs)[masks.MaskType.INSTANCE]
        assert result.masks_existing == 0
        assert masks.instance_labels_path(path).exists()

    def test_masks_from_before_are_rewritten_on_resume(self, tmp_path) -> None:
        kwargs = inputs(tmp_path, ids=(3, 4))
        kwargs["crop_column"] = None
        path = masks.create_masks(**kwargs)[masks.MaskType.INSTANCE].masks_created[0].output_path
        # Strip the tag, as on a mask written before instances were numbered per piece.
        with rasterio.open(path, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as dst:
            dst.update_tags(**{masks.INSTANCE_VALUES_TAG: ""})
        kwargs["skip_existing"] = True
        result = masks.create_masks(**kwargs)[masks.MaskType.INSTANCE]
        assert result.masks_existing == 0
        assert len(result.masks_created) == 1
