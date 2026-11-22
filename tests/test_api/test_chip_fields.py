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
