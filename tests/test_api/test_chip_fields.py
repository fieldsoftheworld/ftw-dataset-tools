"""Tests for each chip's clipped field outlines."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from ftw_dataset_tools.api.chip_fields import (
    ChipFieldsError,
    chip_fields_path,
    write_chip_fields,
)
from ftw_dataset_tools.api.geo import write_geoparquet

if TYPE_CHECKING:
    from pathlib import Path

# A 100 m chip at 2.5 m in zone 32N.
X0, Y0 = 500_000, 5_540_000
UTM = "EPSG:32632"


def _mask(chip_dir: Path, kind: str = "semantic_2_class") -> Path:
    chip_dir.mkdir(parents=True, exist_ok=True)
    path = chip_dir / f"{chip_dir.name}_{kind}.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=40,
        width=40,
        count=1,
        dtype="uint8",
        crs=UTM,
        transform=from_origin(X0, Y0 + 100, 2.5, 2.5),
    ) as dst:
        dst.write(np.zeros((1, 40, 40), dtype="uint8"))
    return path


def _fields(tmp_path: Path) -> Path:
    shapes = [
        box(X0 + 10, Y0 + 10, X0 + 40, Y0 + 40),  # inside
        box(X0 + 80, Y0 + 80, X0 + 150, Y0 + 150),  # crosses the edge
        box(X0 + 100, Y0, X0 + 160, Y0 + 50),  # only touches the edge
        box(X0 + 500, Y0 + 500, X0 + 550, Y0 + 550),  # elsewhere
    ]
    gdf = gpd.GeoDataFrame(
        {"id": ["a", "b", "c", "d"], "crop:name": ["wheat", "maize", "rye", "oat"]},
        geometry=shapes,
        crs=UTM,
    ).to_crs(4326)
    return write_geoparquet(tmp_path / "fields.parquet", gdf=gdf)


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "ds"
    _mask(root / "chips" / "32UMA" / "ftw-a_2024")
    return root


def test_fields_are_cut_to_the_mask_footprint(tmp_path: Path, dataset: Path) -> None:
    result = write_chip_fields(_fields(tmp_path), dataset)

    assert (result.written, result.skipped, result.no_mask) == (1, 0, [])
    out = gpd.read_parquet(chip_fields_path(dataset / "chips" / "32UMA" / "ftw-a_2024"))
    assert out.crs == UTM
    assert list(out["id"]) == ["a", "b"], "the edge-touching and distant fields are left out"
    assert list(out["crop:name"]) == ["wheat", "maize"]
    assert out.geometry.iloc[0].area == pytest.approx(900, rel=1e-6)
    assert out.geometry.iloc[1].bounds == pytest.approx((X0 + 80, Y0 + 80, X0 + 100, Y0 + 100))


def test_chip_without_fields_gets_an_empty_file(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    chip = root / "chips" / "32UMA" / "ftw-b_2024"
    _mask(chip)
    gdf = gpd.GeoDataFrame({"id": ["z"]}, geometry=[box(X0 + 900, Y0, X0 + 950, Y0 + 50)], crs=UTM)
    fields = write_geoparquet(tmp_path / "f.parquet", gdf=gdf.to_crs(4326))

    write_chip_fields(fields, root)

    assert len(gpd.read_parquet(chip_fields_path(chip))) == 0


def test_chip_without_a_mask_gets_no_file(tmp_path: Path, dataset: Path) -> None:
    bare = dataset / "chips" / "32UMA" / "ftw-c_2024"
    bare.mkdir()

    result = write_chip_fields(_fields(tmp_path), dataset)

    assert result.no_mask == ["ftw-c_2024"]
    assert not chip_fields_path(bare).exists()


def test_any_mask_type_gives_the_footprint(tmp_path: Path) -> None:
    root = tmp_path / "ds"
    chip = root / "chips" / "32UMA" / "ftw-d_2024"
    _mask(chip, "instance")

    assert write_chip_fields(_fields(tmp_path), root).written == 1


def test_skip_existing_keeps_a_file_newer_than_the_mask(tmp_path: Path, dataset: Path) -> None:
    fields = _fields(tmp_path)
    write_chip_fields(fields, dataset)
    chip = dataset / "chips" / "32UMA" / "ftw-a_2024"

    assert write_chip_fields(fields, dataset, skip_existing=True).skipped == 1
    # a rebuilt mask makes the outlines stale
    mask = chip / "ftw-a_2024_semantic_2_class.tif"
    later = chip_fields_path(chip).stat().st_mtime + 10
    os.utime(mask, (later, later))
    assert write_chip_fields(fields, dataset, skip_existing=True).written == 1


def test_unreadable_fields_file(tmp_path: Path, dataset: Path) -> None:
    bad = tmp_path / "bad.parquet"
    bad.write_text("not parquet")
    with pytest.raises(ChipFieldsError, match="Cannot read"):
        write_chip_fields(bad, dataset)


def _fields_from(tmp_path: Path, gdf: gpd.GeoDataFrame, name: str = "f.parquet") -> Path:
    return write_geoparquet(tmp_path / name, gdf=gdf.to_crs(4326))


def test_an_invalid_field_is_made_valid(tmp_path: Path, dataset: Path) -> None:
    from shapely.geometry import Polygon

    bowtie = Polygon(
        [(X0 + 10, Y0 + 10), (X0 + 50, Y0 + 50), (X0 + 50, Y0 + 10), (X0 + 10, Y0 + 50)]
    )
    fields = _fields_from(tmp_path, gpd.GeoDataFrame({"id": ["x"]}, geometry=[bowtie], crs=UTM))

    result = write_chip_fields(fields, dataset)

    assert (result.written, result.failed) == (1, [])
    out = gpd.read_parquet(chip_fields_path(dataset / "chips" / "32UMA" / "ftw-a_2024"))
    assert out.geometry.is_valid.all()
    assert out.geometry.iloc[0].area == pytest.approx(800, rel=1e-3)


def test_column_types_survive_including_empty_chips(tmp_path: Path) -> None:
    import pandas as pd
    import pyarrow.parquet as pq

    root = tmp_path / "ds"
    full, empty = (root / "chips" / "32UMA" / n for n in ("ftw-a_2024", "ftw-b_2024"))
    _mask(full)
    _mask(empty)
    # move the empty chip's footprint far away
    with rasterio.open(empty / "ftw-b_2024_semantic_2_class.tif", "r+") as dst:
        dst.transform = from_origin(X0 + 5000, Y0 + 5100, 2.5, 2.5)
    gdf = gpd.GeoDataFrame(
        {
            "id": ["a", "b"],
            "n": pd.array([7, None], dtype="Int32"),
            "when": pd.to_datetime(["2024-05-01", "2024-06-01"], utc=True),
        },
        geometry=[box(X0 + 10, Y0 + 10, X0 + 40, Y0 + 40), box(X0 + 50, Y0 + 50, X0 + 60, Y0 + 60)],
        crs=UTM,
    )
    source = _fields_from(tmp_path, gdf)

    write_chip_fields(source, root)

    want = pq.read_schema(source)
    for chip in (full, empty):
        got = pq.read_schema(chip_fields_path(chip))
        for name in ("id", "n", "when"):
            assert got.field(name).type == want.field(name).type, (chip.name, name)
    assert gpd.read_parquet(chip_fields_path(full))["n"].tolist()[0] == 7


def test_skip_existing_rewrites_after_the_fields_change(tmp_path: Path, dataset: Path) -> None:
    fields = _fields(tmp_path)
    write_chip_fields(fields, dataset)
    out = chip_fields_path(dataset / "chips" / "32UMA" / "ftw-a_2024")
    later = out.stat().st_mtime + 10
    os.utime(fields, (later, later))

    assert write_chip_fields(fields, dataset, skip_existing=True).written == 1


def test_a_differently_named_covering_column_is_dropped(tmp_path: Path, dataset: Path) -> None:
    import geoparquet_io as gpio

    plain = tmp_path / "plain.parquet"
    gpd.GeoDataFrame(
        {"id": ["a"]}, geometry=[box(X0 + 10, Y0 + 10, X0 + 40, Y0 + 40)], crs=UTM
    ).to_crs(4326).to_parquet(plain)
    fields = tmp_path / "covered.parquet"
    gpio.read(str(plain)).add_bbox(column_name="geometry_bbox").write(str(fields))

    write_chip_fields(fields, dataset)

    out = gpd.read_parquet(chip_fields_path(dataset / "chips" / "32UMA" / "ftw-a_2024"))
    assert "geometry_bbox" not in out.columns


def test_a_failing_chip_is_reported_not_raised(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ftw_dataset_tools.api import chip_fields

    def boom(*_a, **_k):
        raise RuntimeError("bad shape")

    fields = _fields(tmp_path)
    write_chip_fields(fields, dataset)
    monkeypatch.setattr(chip_fields, "_clipped", boom)

    result = write_chip_fields(fields, dataset)

    assert result.failed == [("ftw-a_2024", "RuntimeError: bad shape")]
    assert "failed for 1 chip(s)" in chip_fields.summary_lines(result)[-1]
    assert not chip_fields_path(dataset / "chips" / "32UMA" / "ftw-a_2024").exists()


def test_remove_chip_fields(tmp_path: Path, dataset: Path) -> None:
    from ftw_dataset_tools.api.chip_fields import remove_chip_fields

    write_chip_fields(_fields(tmp_path), dataset)

    assert remove_chip_fields(dataset) == 1
    assert not chip_fields_path(dataset / "chips" / "32UMA" / "ftw-a_2024").exists()


def test_empty_and_missing_geometries_are_skipped(tmp_path: Path, dataset: Path) -> None:
    from shapely.geometry import Polygon

    gdf = gpd.GeoDataFrame(
        {"id": ["a", "empty", "none"]},
        geometry=[box(X0 + 10, Y0 + 10, X0 + 40, Y0 + 40), Polygon(), None],
        crs=UTM,
    )
    fields = _fields_from(tmp_path, gdf)

    result = write_chip_fields(fields, dataset)

    assert (result.written, result.failed) == (1, [])
    out = gpd.read_parquet(chip_fields_path(dataset / "chips" / "32UMA" / "ftw-a_2024"))
    assert out["id"].tolist() == ["a"]


def test_fields_keep_file_order(tmp_path: Path, dataset: Path) -> None:
    shapes = [box(X0 + 10 * i, Y0 + 10 * i, X0 + 10 * i + 5, Y0 + 10 * i + 5) for i in (5, 1, 3)]
    gdf = gpd.GeoDataFrame({"id": ["p", "q", "r"]}, geometry=shapes, crs=UTM)

    write_chip_fields(_fields_from(tmp_path, gdf), dataset)

    out = gpd.read_parquet(chip_fields_path(dataset / "chips" / "32UMA" / "ftw-a_2024"))
    assert out["id"].tolist() == ["p", "q", "r"]


def test_rows_are_taken_across_many_batches(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow as pa

    from ftw_dataset_tools.api import chip_fields

    load = chip_fields._load

    def one_row_batches(*args):
        table, bounds = load(*args)
        return pa.Table.from_batches(table.to_batches(max_chunksize=1), table.schema), bounds

    monkeypatch.setattr(chip_fields, "_load", one_row_batches)
    shapes = [
        box(X0 + 10 * i, Y0 + 10 * i, X0 + 10 * i + 5, Y0 + 10 * i + 5) for i in (5, 1, 30, 3)
    ]
    gdf = gpd.GeoDataFrame({"id": ["p", "q", "far", "r"]}, geometry=shapes, crs=UTM)

    write_chip_fields(_fields_from(tmp_path, gdf), dataset)

    out = gpd.read_parquet(chip_fields_path(dataset / "chips" / "32UMA" / "ftw-a_2024"))
    assert out["id"].tolist() == ["p", "q", "r"]


def test_a_footprint_across_the_antimeridian_finds_fields_on_both_sides() -> None:
    import pyarrow as pa
    import shapely

    from ftw_dataset_tools.api.chip_fields import _query, _Source

    shapes = [box(179.9, 0, 179.95, 1), box(-179.95, 0, -179.9, 1), box(0, 0, 1, 1)]
    table = pa.table(
        {"id": ["east", "west", "middle"], "__ftw_wkb": shapely.to_wkb(shapes).tolist()}
    )
    source = _Source(
        None,
        "geometry",
        "EPSG:4326",
        tuple(table.to_batches()),
        np.array([0]),
        table.schema,
        shapely.STRtree(shapes),
    )

    assert _query(source, (179.8, 0, -179.8, 1))["id"].to_pylist() == ["east", "west"]


def test_a_rerun_that_keeps_every_file_does_not_load_the_fields(
    tmp_path: Path, dataset: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ftw_dataset_tools.api import chip_fields

    fields = _fields(tmp_path)
    write_chip_fields(fields, dataset)

    def boom(*_a, **_k):
        raise AssertionError("loaded")

    monkeypatch.setattr(chip_fields, "_source", boom)

    assert write_chip_fields(fields, dataset, skip_existing=True).skipped == 1


def test_fields_are_loaded_in_small_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ftw_dataset_tools.api import chip_fields

    monkeypatch.setattr(chip_fields, "_ROWS_PER_BATCH", 1)
    source = chip_fields._source(_fields(tmp_path))

    assert [len(b) for b in source.batches] == [1, 1, 1, 1]


def test_a_missing_fields_file(tmp_path: Path, dataset: Path) -> None:
    with pytest.raises(ChipFieldsError, match="no such file"):
        write_chip_fields(tmp_path / "missing.parquet", dataset)
